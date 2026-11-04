# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""kv_connector_extra_config "cpu.draft_offload": which blocks of a speculative
draft model's sliding-window KV cache group the CPU tier stores.

Geometry (a scaled-down DFlash2 hybrid): full attention and Mamba ``align``
states on 32-token blocks, the drafter's sliding window (45 tokens) on 16-token
blocks, so a prefix hit at ``H`` needs draft blocks ``H // 16 - 3 .. H // 16``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockOffloadingSpec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    CPU_DRAFT_SKIPPED_KEY,
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    is_draft_sliding_window_group,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash, KVCacheBlock
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KpoolTailSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_offload.cpu.manager import CPUOffloadingManager

pytestmark = pytest.mark.cpu_test

FULL, MAMBA, DRAFT = 0, 1, 2
TARGET_BLOCK = 32
DRAFT_BLOCK = 16
DRAFT_WINDOW_TOKENS = 45  # 3 offloaded draft blocks
DRAFT_WINDOW_BLOCKS = 3
NUM_TARGET_LAYERS = 4
PROMPT = 200  # 6 target blocks, 12 draft blocks
MODES = ("all", "window", "none")


def _kv_config(draft_layer: str = "model.layers.4.self_attn.attn", **draft_kw):
    draft_spec_cls = draft_kw.pop("spec_cls", SlidingWindowSpec)
    return KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["language_model.model.layers.0.self_attn.attn"],
                FullAttentionSpec(
                    block_size=TARGET_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["language_model.model.layers.1.linear_attn"],
                MambaSpec(
                    block_size=TARGET_BLOCK,
                    shapes=((1, 1),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
            KVCacheGroupSpec(
                [draft_layer],
                draft_spec_cls(
                    block_size=DRAFT_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                    sliding_window=DRAFT_WINDOW_TOKENS,
                ),
                **draft_kw,
            ),
        ],
    )


def _vllm_config(mode: str | None, speculative: bool = True):
    config = MagicMock()
    config.parallel_config.decode_context_parallel_size = 1
    config.parallel_config.prefill_context_parallel_size = 1
    config.parallel_config.world_size = 1
    config.cache_config.enable_prefix_caching = True
    config.cache_config.block_size = TARGET_BLOCK
    config.cache_config.hash_block_size = None
    config.cache_config.prefix_cache_retention_interval = 0
    config.model_config.get_total_num_hidden_layers.return_value = NUM_TARGET_LAYERS
    extra: dict = {"offload_prompt_only": True}
    if mode is not None:
        extra["cpu.draft_offload"] = mode
    config.kv_transfer_config.kv_connector_extra_config = extra
    config.speculative_config = (
        SimpleNamespace(use_eagle_kv_cache=lambda: False) if speculative else None
    )
    return config


def _hashes(n: int, salt: str = "p"):
    return [BlockHash(f"{salt}{i}".encode()) for i in range(n)]


def _request(req_id: str, num_tokens: int, hashes=None, junction: int = 0):
    req = MagicMock()
    req.request_id = req_id
    req.skip_reading_prefix_cache = False
    req.kv_transfer_params = None
    req.block_hashes = (
        hashes if hashes is not None else _hashes(num_tokens // DRAFT_BLOCK)
    )
    req.num_computed_tokens = 0
    req.num_prompt_tokens = req.num_tokens = num_tokens
    req.shared_prefix_boundary = junction
    req.is_finished.return_value = False
    return req


def _make(mode: str | None, manager=None):
    spec = MockOffloadingSpec(_vllm_config(mode), _kv_config())
    if manager is not None:
        spec.manager = manager
    scheduler = OffloadingConnectorScheduler(spec)
    if manager is None:
        from tests.v1.kv_connector.unit.offloading_connector.utils import (
            generate_store_output,
        )

        scheduler.manager.prepare_store.side_effect = lambda keys, ctx: (
            generate_store_output(keys)
        )
    return scheduler


def _output(req, num_scheduled, *, new=None, cached=False, boundaries=(), **kw):
    """One scheduler step for ``req``; ``new`` is its block-id tuple."""
    output = SimpleNamespace(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], new_block_ids=[], resumed_req_ids=set()
        ),
        num_scheduled_tokens={req.request_id: num_scheduled},
        finished_req_ids=set(),
        preempted_req_ids=set(),
        boundary_state_offloads=(
            {req.request_id: [(MAMBA, 900 + b // TARGET_BLOCK, b) for b in boundaries]}
            if boundaries
            else None
        ),
    )
    if new is not None:
        output.scheduled_new_reqs = [
            SimpleNamespace(req_id=req.request_id, block_ids=new)
        ]
    elif cached:
        output.scheduled_cached_reqs = SimpleNamespace(
            req_ids=[req.request_id], new_block_ids=[None], resumed_req_ids=set()
        )
    for name, value in kw.items():
        setattr(output, name, value)
    return output


def _block_ids(num_tokens: int, base: int = 0):
    """GPU block IDs of a fresh prefill (the connector mirror never learns
    that sliding-window blocks are freed)."""
    full = [base + 100 + i for i in range(num_tokens // TARGET_BLOCK + 1)]
    mamba = [0] * (num_tokens // TARGET_BLOCK + 1)
    draft = [base + 300 + i for i in range(num_tokens // DRAFT_BLOCK + 1)]
    return (full, mamba, draft)


class Run:
    """Drive one request through chunked prefill; collect its store jobs."""

    def __init__(self, scheduler, req, complete=False):
        self.scheduler = scheduler
        self.req = req
        self.complete = complete
        self.jobs: list = []
        scheduler.on_new_request(req)
        scheduler.get_num_new_matched_tokens(req, 0)
        self.first = True

    def step(self, num_scheduled: int, boundaries=(), **kw):
        req = self.req
        new = _block_ids(req.num_tokens) if self.first else None
        meta = self.scheduler.build_connector_meta(
            _output(
                req,
                num_scheduled,
                new=new,
                cached=not self.first,
                boundaries=boundaries,
                **kw,
            )
        )
        self.first = False
        req.num_computed_tokens += num_scheduled
        jobs = sorted(meta.store_jobs.items())
        self.jobs.extend(job for _, job in jobs)
        if self.complete and meta.store_jobs:
            self.scheduler.update_connector_output(
                SimpleNamespace(
                    kv_connector_worker_meta=OffloadingWorkerMetadata(
                        completed_jobs={jid: 1 for jid in meta.store_jobs}
                    )
                )
            )
        return meta

    def stored_blocks(self, group_idx: int) -> list[int]:
        """Offloaded block indices of ``group_idx`` stored by this request."""
        keys = (
            self.scheduler._req_status[self.req.request_id]
            .group_states[group_idx]
            .offload_keys
        )
        index = {key: i for i, key in enumerate(keys)}
        out: list[int] = []
        for job in self.jobs:
            _, dst = job.transfer_spec
            out.extend(index[k] for k in dst.offload_keys if k in index)
        return sorted(out)

    def finish(self):
        self.req.is_finished.return_value = True
        self.scheduler.request_finished(self.req)


def _standard_run(scheduler, junction: int = 160, complete=False):
    """Prefill 200 tokens; the core stops chunks at the junction (160) and the
    replay boundary (192) and hands off the recurrent states there."""
    req = _request("r1", PROMPT, junction=junction)
    run = Run(scheduler, req, complete=complete)
    run.step(128)
    run.step(32, boundaries=(160,))
    run.step(32, boundaries=(192,))
    run.step(8)
    return run


def _spec_signature(jobs):
    out = []
    for job in jobs:
        src, dst = job.transfer_spec
        out.append(
            (
                src.block_ids.tolist(),
                list(src.group_sizes),
                list(src.block_indices),
                list(dst.offload_keys),
            )
        )
    return out


# ----------------------------------------------------------------- config


def test_value_validated_at_startup():
    with pytest.raises(ValueError, match="cpu.draft_offload"):
        MockOffloadingSpec(_vllm_config("partial"), _kv_config())
    for mode in MODES:
        assert (
            MockOffloadingSpec(_vllm_config(mode), _kv_config()).cpu_draft_offload
            == mode
        )
    assert (
        MockOffloadingSpec(_vllm_config(None), _kv_config()).cpu_draft_offload == "all"
    )


def test_draft_group_identification():
    def draft_groups(vllm_config, kv_config):
        spec = MockOffloadingSpec(vllm_config, kv_config)
        return [
            i
            for i in range(len(kv_config.kv_cache_groups))
            if is_draft_sliding_window_group(spec, i)
        ]

    # Layers numbered after the target's: the drafter (DFlash, EAGLE).
    assert draft_groups(_vllm_config("window"), _kv_config()) == [DRAFT]
    # A target model's own sliding window (layer 2 < 4 target layers).
    assert (
        draft_groups(
            _vllm_config("window"), _kv_config("model.layers.2.self_attn.attn")
        )
        == []
    )
    # Explicitly flagged draft group, whatever its layer names.
    assert draft_groups(
        _vllm_config("window"),
        _kv_config("mtp.self_attn.attn", is_eagle_group=True),
    ) == [DRAFT]
    # No speculative decoding: no draft model.
    assert draft_groups(_vllm_config("window", speculative=False), _kv_config()) == []
    # Sliding-window variants freed by other managers are left alone.
    assert (
        draft_groups(_vllm_config("window"), _kv_config(spec_cls=KpoolTailSpec)) == []
    )

    scheduler = _make("window")
    configs = {g.group_idx: g for g in scheduler.config.kv_group_configs}
    assert configs[DRAFT].draft_window_tokens == DRAFT_WINDOW_TOKENS
    assert configs[DRAFT].sliding_window_size_in_blocks == DRAFT_WINDOW_BLOCKS
    # Recurrent states (window of one block) and full attention: untouched.
    assert configs[FULL].draft_window_tokens is None
    assert configs[MAMBA].draft_window_tokens is None
    # The SWA alignment skip does not apply in this geometry either.
    assert configs[DRAFT].alignment_block_count is None


# ------------------------------------------------------------- store jobs


def test_all_matches_absent_key_exactly():
    absent = _standard_run(_make(None))
    explicit = _standard_run(_make("all"))
    assert _spec_signature(absent.jobs) == _spec_signature(explicit.jobs)
    assert absent.stored_blocks(DRAFT) == list(range(12))
    assert explicit.scheduler.num_skipped_draft_blocks == 0


@pytest.mark.parametrize("mode", MODES)
def test_store_jobs_per_mode(mode):
    run = _standard_run(_make(mode))
    scheduler = run.scheduler
    expected_draft = {
        "all": list(range(12)),
        # Windows before 160 (junction + state) and 192 (state).
        "window": [7, 8, 9, 10, 11],
        "none": [],
    }[mode]
    assert run.stored_blocks(DRAFT) == expected_draft
    # Full attention and recurrent states are stored the same in every mode.
    assert run.stored_blocks(FULL) == list(range(6))
    assert run.stored_blocks(MAMBA) == [4, 5]
    run.finish()
    # Every draft block passed by the store frontier is either offered for
    # store or counted as skipped, exactly once.
    assert scheduler.num_skipped_draft_blocks == 12 - len(expected_draft)
    req_status = scheduler._req_status.get("r1")
    if req_status is not None:
        assert not any(req_status.draft_deferred.values())


def test_window_late_store_reads_only_held_blocks():
    """Block 7 belongs to the window before 160 but was computed in the first
    chunk; it is stored in the step that hands off state 160. Blocks behind
    the window of the computed tokens are freed by the core and never read."""
    scheduler = _make("window")
    req = _request("r1", PROMPT)
    run = Run(scheduler, req)
    run.step(128)
    assert run.stored_blocks(DRAFT) == []
    deferred = scheduler._req_status["r1"].draft_deferred[DRAFT]
    assert deferred == set(range(8))
    meta = run.step(32, boundaries=(160,))
    # 128 computed tokens: the core freed draft blocks 0..4 (tokens < 84).
    assert scheduler.num_skipped_draft_blocks == 5
    assert run.stored_blocks(DRAFT) == [7, 8, 9]
    [job] = [
        job
        for job in meta.store_jobs.values()
        if job.transfer_spec[0].group_sizes[DRAFT]
    ]
    src, _ = job.transfer_spec
    draft_ids = src.block_ids.tolist()[-src.group_sizes[DRAFT] :]
    assert draft_ids == [307, 308, 309]
    # Draft source blocks are fenced against reuse like any SWA store.
    for bid in draft_ids:
        assert bid in scheduler._block_id_to_pending_jobs
    # A boundary arriving after its window was freed stores nothing stale.
    req.num_computed_tokens = 192
    run.step(8, boundaries=(64,))
    assert 1 not in run.stored_blocks(DRAFT)


def test_window_junction_without_state_handoff():
    """The window before a shared-prefix junction is kept even when no state
    hand-off has arrived (yet) for it."""
    run = Run(_make("window"), _request("r1", PROMPT, junction=64))
    run.step(96)
    assert run.stored_blocks(DRAFT) == [1, 2, 3]


def test_none_counts_once_across_host_tier_retry():
    scheduler = _make("none")
    run = Run(scheduler, _request("r1", PROMPT))
    from tests.v1.kv_connector.unit.offloading_connector.utils import (
        generate_store_output,
    )

    scheduler.manager.prepare_store.side_effect = [None]
    run.step(128)
    assert run.jobs == []
    assert scheduler.num_skipped_draft_blocks == 8
    scheduler.manager.prepare_store.side_effect = lambda keys, ctx: (
        generate_store_output(keys)
    )
    run.step(72)
    # The target blocks are stored on the retry; draft blocks are not recounted.
    assert run.stored_blocks(FULL) == list(range(6))
    assert scheduler.num_skipped_draft_blocks == 12


def test_window_preemption_drops_deferred_blocks():
    scheduler = _make("window")
    req = _request("r1", PROMPT)
    run = Run(scheduler, req)
    run.step(128)
    assert len(scheduler._req_status["r1"].draft_deferred[DRAFT]) == 8
    scheduler.build_connector_meta(
        _output(req, 0, preempted_req_ids={"r1"}, num_scheduled_tokens={})
    )
    assert scheduler.num_skipped_draft_blocks == 8
    # Resumed from scratch with new blocks: old block IDs are never read.
    req.num_computed_tokens = 0
    scheduler.get_num_new_matched_tokens(req, 0)
    assert scheduler._req_status["r1"].draft_frontier == {}
    meta = scheduler.build_connector_meta(
        _output(req, 160, new=_block_ids(PROMPT, base=1000), boundaries=(160,))
    )
    run.jobs.extend(meta.store_jobs.values())
    for job in meta.store_jobs.values():
        src, _ = job.transfer_spec
        if src.group_sizes[MAMBA]:
            continue  # the exact state hand-off source
        assert all(bid >= 1000 for bid in src.block_ids.tolist())
    assert run.stored_blocks(DRAFT) == [7, 8, 9]


# ------------------------------------------------------------------ lookup


def _tier(mode: str, junction: int = 160):
    manager = CPUOffloadingManager(256, cache_policy="lru")
    scheduler = _make(mode, manager=manager)
    run = _standard_run(scheduler, junction=junction, complete=True)
    run.finish()
    return scheduler, manager, run


def _sibling(scheduler, num_tokens: int = 240, req_id: str = "r2"):
    hashes = _hashes(PROMPT // DRAFT_BLOCK) + _hashes(
        (num_tokens - PROMPT) // DRAFT_BLOCK + 1, salt="x"
    )
    req = _request(req_id, num_tokens, hashes=hashes[: num_tokens // DRAFT_BLOCK])
    scheduler.on_new_request(req)
    return req


@pytest.mark.parametrize("mode", MODES)
def test_cpu_pool_contents_and_lookup(mode):
    scheduler, manager, run = _tier(mode)
    keys = scheduler._req_status  # r1 is gone once its stores completed
    assert "r1" not in keys
    req = _sibling(scheduler)
    state = scheduler._req_status["r2"]
    state.update_offload_keys()
    ctx = state.req_context
    draft_keys = state.group_states[DRAFT].offload_keys[:12]
    resident = [i for i, key in enumerate(draft_keys) if manager.lookup(key, ctx)]
    assert (
        resident
        == {
            "all": list(range(12)),
            "window": [7, 8, 9, 10, 11],
            "none": [],
        }[mode]
    )
    num_hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    # The draft window before 192 is stored in "all" and "window"; without
    # draft blocks no hit is possible.
    assert num_hit == {"all": 192, "window": 192, "none": 0}[mode]


def test_lookup_shrinks_to_complete_draft_window():
    scheduler, manager, _ = _tier("window")
    req = _sibling(scheduler)
    state = scheduler._req_status["r2"]
    state.update_offload_keys()
    draft_keys = state.group_states[DRAFT].offload_keys
    # Lose draft block 10 (inside the window before 192): back to 160.
    manager._policy.remove(draft_keys[10])
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 160
    # Lose block 7 as well: the window before 160 is incomplete too.
    manager._policy.remove(draft_keys[7])
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 0


# ---------------------------------------------------- demotion and metrics


@pytest.mark.parametrize("mode", MODES)
def test_extend_demotion_unaffected(mode):
    """superseded_state_rule "extend" demotes the same recurrent states in
    every mode and never a draft block."""
    from tests.v1.kv_connector.unit.offloading_connector.utils import (
        generate_store_output,
    )

    config = _vllm_config(mode)
    config.kv_transfer_config.kv_connector_extra_config["superseded_state_rule"] = (
        "extend"
    )
    spec = MockOffloadingSpec(config, _kv_config())
    scheduler = OffloadingConnectorScheduler(spec)
    scheduler.manager.prepare_store.side_effect = lambda keys, ctx: (
        generate_store_output(keys)
    )
    # The previous turn stored up to 192 (state and draft window there);
    # this turn hits at 192 and extends the conversation to 330 tokens.
    req = _sibling(scheduler, num_tokens=330)
    state = scheduler._req_status["r2"]
    state.update_offload_keys()
    groups = state.group_states
    present = set(groups[FULL].offload_keys[:6])
    present.add(groups[MAMBA].offload_keys[5])
    present.update(groups[DRAFT].offload_keys[9:12])
    scheduler.manager.lookup.side_effect = lambda key, ctx: key in present
    assert scheduler.get_num_new_matched_tokens(req, 0)[0] == 192
    assert [groups[g].num_hit_blocks for g in (FULL, MAMBA)] == [6, 6]
    state.stored_state_blocks = {MAMBA: {9}}
    scheduler._demote_superseded_states(state)
    [call] = scheduler.manager.demote.call_args_list
    demoted = call.args[0]
    # Superseded state 6 and the previous turn's end state 5 (the hit).
    mamba_keys = groups[MAMBA].offload_keys
    assert demoted == [mamba_keys[6], mamba_keys[5]]
    assert not set(demoted) & set(state.group_states[DRAFT].offload_keys)


@pytest.mark.parametrize("mode", MODES)
def test_counters_reconcile_with_stats(mode):
    run = _standard_run(_make(mode))
    run.finish()
    scheduler = run.scheduler
    stored = len(run.stored_blocks(DRAFT))
    assert stored + scheduler.num_skipped_draft_blocks == 12

    connector = object.__new__(OffloadingConnector)
    connector.connector_worker = None
    connector.connector_scheduler = scheduler
    scheduler.cpu_tier_eviction_deltas = lambda: {}
    stats = connector.get_kv_connector_stats()
    if mode == "all":
        assert stats is None
        return
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.reduce() == {CPU_DRAFT_SKIPPED_KEY: 12 - stored}
    # Reported once.
    assert connector.get_kv_connector_stats() is None


def test_prometheus_counter_for_skipped_draft_blocks():
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    registry = CollectorRegistry()

    def bind(cls):
        return lambda *a, **k: cls(*a, registry=registry, **k)

    prom = OffloadPromMetrics(
        SimpleNamespace(kv_transfer_config=None),
        {Gauge: bind(Gauge), Counter: bind(Counter), Histogram: bind(Histogram)},
        ["model_name", "engine"],
        {0: ["m", "0"]},
    )
    first = OffloadingConnectorStats()
    first.record_skipped_draft_blocks(5)
    merged = OffloadingConnectorStats(
        data={"GPU_to_CPU": [{"op_size": 8, "op_time": 0.1}]}
    ).aggregate(first)
    prom.observe(merged.data, engine_idx=0)
    second = OffloadingConnectorStats()
    second.record_skipped_draft_blocks(2)
    prom.observe(second.data, engine_idx=0)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_cpu_draft_blocks_skipped_total",
            {"model_name": "m", "engine": "0"},
        )
        == 7
    )


# --------------------------------------------------- partial prefix cache


@pytest.mark.parametrize("mode", MODES)
def test_partial_prefix_hit_resolution(mode):
    """A partial prefix-cache hit (aligned local hit 128, partial 170) is
    replaced by a longer external hit only when the tier can serve it; the
    draft window loaded is exactly the one stored for the boundary."""
    scheduler, _, _ = _tier(mode)
    req = _sibling(scheduler)
    null = KVCacheBlock(block_id=0, is_null=True)

    def blocks(num_tokens: int) -> KVCacheBlocks:
        n_t = num_tokens // TARGET_BLOCK
        n_d = num_tokens // DRAFT_BLOCK
        return KVCacheBlocks(
            (
                [KVCacheBlock(block_id=500 + i) for i in range(n_t)],
                [null] * (n_t - 1) + [KVCacheBlock(block_id=600)],
                [null] * (n_d - DRAFT_WINDOW_BLOCKS)
                + [KVCacheBlock(block_id=700 + i) for i in range(DRAFT_WINDOW_BLOCKS)],
            )
        )

    aligned_blocks = blocks(128)
    req.partial_prefix_entry = object()
    req.partial_prefix_fallback = (aligned_blocks, 128)
    ext, load_async = scheduler.get_num_new_matched_tokens(req, 128)
    partial_blocks = blocks(176)
    out_blocks, local, ext, load_async = Scheduler._resolve_partial_prefix_hit(
        req, partial_blocks, 170, ext, load_async
    )
    if mode == "none":
        assert (out_blocks, local, ext, load_async) == (partial_blocks, 170, 0, False)
        return
    assert out_blocks is aligned_blocks and local == 128
    assert ext == 64 and load_async
    scheduler.update_state_after_alloc(req, blocks(192), ext)
    state = scheduler._req_status["r2"]
    [(job_id, loaded)] = scheduler._current_batch_load_jobs.items()
    _, dst = loaded.transfer_spec
    assert list(dst.group_sizes) == [2, 1, DRAFT_WINDOW_BLOCKS]
    assert list(dst.block_indices) == [4, 5, 9]
    groups = state.group_states
    assert scheduler._jobs[job_id].keys == {
        *groups[FULL].offload_keys[4:6],
        groups[MAMBA].offload_keys[5],
        *groups[DRAFT].offload_keys[9:12],
    }
