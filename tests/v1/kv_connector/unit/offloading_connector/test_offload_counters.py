# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side offload counters:

- vllm:kv_offload_cpu_stored_blocks_total{kind}: completed CPU-tier stores by
  KV cache group kind (target / draft / state).
- vllm:kv_offload_lookup_shrink_tokens_total{cause}: how far sparse groups
  lower an admitted lookup below the full-attention reach.
- vllm:kv_offload_hit_draft_window_needed_total{needed}: whether an external
  hit leaves fewer tokens to prefill than the draft window.

Geometry as in test_cpu_draft_offload: full attention and Mamba ``align``
states on 32-token blocks, a drafter sliding window of 45 tokens on 16-token
blocks (3 offloaded blocks), 200-token prompts.
"""

import hashlib
import random
from types import SimpleNamespace

import pytest
import torch

from tests.v1.kv_connector.unit.offloading_connector.test_cpu_draft_offload import (
    DRAFT,
    DRAFT_BLOCK,
    DRAFT_WINDOW_BLOCKS,
    DRAFT_WINDOW_TOKENS,
    FULL,
    MAMBA,
    MODES,
    PROMPT,
    TARGET_BLOCK,
    Run,
    _block_ids,
    _cpu_manager,
    _hashes,
    _kv_config,
    _output,
    _request,
    _sibling,
    _standard_run,
    _vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    MockOffloadingSpec,
    generate_store_output,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.counters import (
    build_offload_counters,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    GroupOffloadConfig,
    OffloadingConnectorScheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)

pytestmark = pytest.mark.cpu_test


def _vllm_config_with_counters():
    config = _vllm_config("all")
    config.kv_transfer_config.kv_connector_extra_config["cpu.offload_counters"] = True
    return config


def _scheduler(
    mode="all",
    manager=None,
    *,
    eagle=False,
    speculative=True,
    kv_config=None,
    counters=True,
):
    config = _vllm_config(mode, speculative=speculative)
    if counters is not None:
        config.kv_transfer_config.kv_connector_extra_config["cpu.offload_counters"] = (
            counters
        )
    if eagle:
        config.speculative_config = SimpleNamespace(use_eagle_kv_cache=lambda: True)
    spec = MockOffloadingSpec(config, kv_config or _kv_config())
    if manager is not None:
        spec.manager = manager
    scheduler = OffloadingConnectorScheduler(spec)
    if manager is None:
        scheduler.manager.prepare_store.side_effect = lambda keys, ctx: (
            generate_store_output(keys)
        )
    return scheduler


def _complete(scheduler, job_ids):
    scheduler.update_connector_output(
        SimpleNamespace(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={jid: 1 for jid in job_ids}
            )
        )
    )


def _alloc_blocks(served: int) -> KVCacheBlocks:
    """GPU blocks of a request admitted with ``served`` cached tokens (local
    and external); the draft group holds only its window."""
    null = KVCacheBlock(block_id=0, is_null=True)
    n_t = cdiv(served, TARGET_BLOCK)
    n_d = cdiv(served, DRAFT_BLOCK)
    return KVCacheBlocks(
        (
            [KVCacheBlock(block_id=500 + i) for i in range(n_t)],
            [null] * max(n_t - 1, 0) + [KVCacheBlock(block_id=600)],
            [null] * max(n_d - DRAFT_WINDOW_BLOCKS, 0)
            + [KVCacheBlock(block_id=700 + i) for i in range(min(n_d, 3))],
        )
    )


def _admit(scheduler, req, num_external: int):
    """The core admits ``req`` with ``num_external`` tokens from the tier."""
    local = scheduler._req_status[req.request_id].num_locally_computed_tokens
    blocks = (
        _alloc_blocks(local + num_external)
        if num_external
        else KVCacheBlocks(([], [], []))
    )
    scheduler.update_state_after_alloc(req, blocks, num_external)


def _counters(scheduler):
    return scheduler._offload_counters


def _stats_connector(scheduler):
    connector = object.__new__(OffloadingConnector)
    connector.connector_worker = None
    connector.connector_scheduler = scheduler
    # Isolate the new counters from the other scheduler-side ones.
    scheduler.cpu_tier_eviction_deltas = lambda: {}
    scheduler.skipped_draft_blocks_delta = lambda: 0
    return connector


# ------------------------------------------ stored blocks: classification


def _kinds(scheduler):
    return _counters(scheduler).kind_by_group


def test_stored_kind_classification():
    # DFlash-style drafter: layers numbered after the target's.
    assert _kinds(_scheduler()) == {FULL: "target", MAMBA: "state", DRAFT: "draft"}
    # The same classification for every cpu.draft_offload value.
    for mode in MODES:
        assert _kinds(_scheduler(mode))[DRAFT] == "draft"
    # A target model's own sliding window is target KV.
    target_swa = _kv_config("model.layers.2.self_attn.attn")
    assert _kinds(_scheduler(kv_config=target_swa))[DRAFT] == "target"
    # Without speculative decoding there is no draft group.
    assert _kinds(_scheduler(speculative=False))[DRAFT] == "target"
    # An EAGLE/MTP group flagged as such with a sliding window is draft KV.
    eagle_swa = _kv_config("mtp.self_attn.attn", is_eagle_group=True)
    assert _kinds(_scheduler(kv_config=eagle_swa))[DRAFT] == "draft"
    # An MTP group with full attention is stored and looked up like the
    # target's full attention: target.
    mtp_full = KVCacheConfig(
        num_blocks=64,
        kv_cache_tensors=[],
        kv_cache_groups=[
            *_kv_config().kv_cache_groups[:2],
            KVCacheGroupSpec(
                ["mtp.layers.4.self_attn.attn"],
                FullAttentionSpec(
                    block_size=TARGET_BLOCK,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
                is_eagle_group=True,
            ),
        ],
    )
    assert _kinds(_scheduler(kv_config=mtp_full)) == {
        FULL: "target",
        MAMBA: "state",
        DRAFT: "target",
    }


def test_shrink_causes_and_draft_window():
    counters = _counters(_scheduler())
    assert counters.shrink_cause_by_group == {
        MAMBA: "recurrent_state",
        DRAFT: "draft_window",
    }
    assert counters.draft_window_tokens == DRAFT_WINDOW_TOKENS
    counters = _counters(_scheduler(speculative=False))
    assert counters.shrink_cause_by_group[DRAFT] == "sliding_window"
    assert counters.draft_window_tokens is None


# ----------------------------------------------------------- stored blocks


@pytest.mark.parametrize("grouped", [False, True], ids=["unified", "grouped"])
@pytest.mark.parametrize("mode", MODES)
def test_stored_blocks_by_kind(mode, grouped):
    manager = _cpu_manager(grouped)
    scheduler = _scheduler(mode, manager)
    run = _standard_run(scheduler, complete=True)
    run.finish()
    stored = _counters(scheduler).stored_blocks
    draft = {"all": 12, "window": 5, "none": 0}[mode]
    expected = {"target": 6, "state": 2, "draft": draft}
    assert {k: stored[k] for k in expected} == expected
    # Every store completed and no reset: the manager-side total (counted
    # when a store is prepared) is the same.
    cpu_stored, _ = scheduler._cpu_tier_counters()
    assert sum(stored.values()) == cpu_stored
    # Draft blocks passed by the store frontier: stored or skipped.
    assert stored["draft"] + scheduler.num_skipped_draft_blocks == 12
    assert scheduler.offload_counter_deltas()["cpu_stored_blocks"] == {
        k: v for k, v in expected.items() if v
    }


def test_stored_blocks_count_completed_stores_only():
    manager = _cpu_manager(grouped=False)
    scheduler = _scheduler("all", manager)
    run = Run(scheduler, _request("r1", PROMPT))
    meta = run.step(128)
    assert len(meta.store_jobs) == 1
    # Prepared but in flight: the manager counts it, the stored counter does not.
    assert sum(_counters(scheduler).stored_blocks.values()) == 0
    assert scheduler._cpu_tier_counters()[0] == 4 + 8
    # A store discarded by a cache reset is never counted.
    scheduler.reset_cache()
    _complete(scheduler, meta.store_jobs)
    assert sum(_counters(scheduler).stored_blocks.values()) == 0
    meta = run.step(72)
    _complete(scheduler, meta.store_jobs)
    stored = _counters(scheduler).stored_blocks
    # After the reset every block is offered again (stored now).
    assert stored == {"target": 6, "draft": 12}
    # Keys already resident are not stored again.
    run2 = Run(scheduler, _request("r2", PROMPT))
    assert not run2.step(200).store_jobs
    assert stored == {"target": 6, "draft": 12}


# ---------------------------------------------------------- lookup shrink


def _lookup(present, *, num_computed=0, eagle=False, kv_config=None, tokens=PROMPT):
    """Look up a fresh request against a host tier holding ``present``
    ({group: block indices}); return (scheduler, request, hit, shrink)."""
    scheduler = _scheduler(eagle=eagle, kv_config=kv_config)
    req = _request("r", tokens)
    scheduler.on_new_request(req)
    state = scheduler._req_status["r"]
    state.update_offload_keys()
    keys = {
        state.group_states[g].offload_keys[b]
        for g, blocks in present.items()
        for b in blocks
    }
    scheduler.manager.lookup.side_effect = lambda key, ctx: key in keys
    hit, _ = scheduler.get_num_new_matched_tokens(req, num_computed)
    return scheduler, req, hit, state.lookup_shrink


ALL = {FULL: range(6), MAMBA: range(6), DRAFT: range(12)}


def test_lookup_without_shrink():
    scheduler, req, hit, shrink = _lookup(ALL)
    # The request length minus one, on the page grid.
    assert hit == 192
    assert shrink == {}
    _admit(scheduler, req, hit)
    assert _counters(scheduler).lookup_shrink_tokens == {}
    assert "lookup_shrink_tokens" not in scheduler.offload_counter_deltas()


@pytest.mark.parametrize(
    ("present", "num_computed", "hit", "expected"),
    [
        # State only at 128: the recurrent group cuts the hit by two pages.
        ({**ALL, MAMBA: [3]}, 0, 128, {"recurrent_state": 64}),
        # The same after a local GPU hit of 64 tokens.
        ({**ALL, MAMBA: [3]}, 64, 64, {"recurrent_state": 64}),
        # Draft block 10 missing: the last complete window ends at 160.
        ({**ALL, DRAFT: [*range(10), 11]}, 0, 160, {"draft_window": 32}),
        # The last complete window ends at 176, inside a page: 16 tokens for
        # the window, 16 for rounding down to the page grid.
        ({**ALL, DRAFT: range(7, 11)}, 0, 160, {"draft_window": 16, "page_align": 16}),
        # No state at all: no hit, everything goes to the recurrent state.
        ({**ALL, MAMBA: []}, 0, 0, {"recurrent_state": 192}),
        # No draft window at all.
        ({**ALL, DRAFT: []}, 0, 0, {"draft_window": 192}),
    ],
)
def test_lookup_shrink_by_cause(present, num_computed, hit, expected):
    scheduler, req, got, shrink = _lookup(present, num_computed=num_computed)
    assert got == hit
    assert shrink == expected
    # Reconciles with reach (192) minus served.
    assert sum(shrink.values()) == 192 - (num_computed + hit)
    _admit(scheduler, req, hit)
    assert _counters(scheduler).lookup_shrink_tokens == expected


def test_lookup_shrink_eagle_state_shift():
    # With EAGLE/MTP the state is restored one page before the reach.
    scheduler, req, hit, shrink = _lookup(ALL, eagle=True)
    assert hit == 160
    assert shrink == {"page_align": 32}


def test_lookup_shrink_target_sliding_window():
    _, _, hit, shrink = _lookup(
        {**ALL, DRAFT: [*range(10), 11]},
        kv_config=_kv_config("model.layers.2.self_attn.attn"),
    )
    assert hit == 160
    assert shrink == {"sliding_window": 32}


def test_lookup_without_reach_counts_nothing():
    # The full-attention groups cannot serve a block: no reach, no shrink.
    _, _, hit, shrink = _lookup({FULL: [], MAMBA: range(6), DRAFT: range(12)})
    assert hit == 0
    assert shrink == {}
    # A model without full attention has no reach: shrink is not tracked.
    kv_config = _kv_config()
    sparse_only = [
        GroupOffloadConfig(MAMBA, 32, 32, 2, 1),
        GroupOffloadConfig(DRAFT, 16, 16, 1, 3, draft_window_tokens=45),
    ]
    counters = build_offload_counters(sparse_only, kv_config)
    assert counters.kind_by_group == {MAMBA: "state", DRAFT: "draft"}
    assert counters.new_lookup() is None
    assert _counters(_scheduler()).new_lookup() is not None


def test_lookup_counted_once_per_admission():
    scheduler, req, hit, shrink = _lookup({**ALL, MAMBA: [3]})
    # The core asks again when the request could not be allocated.
    scheduler.get_num_new_matched_tokens(req, 0)
    _admit(scheduler, req, hit)
    # A request admitted after an async load comes back with no new lookup.
    _admit(scheduler, req, 0)
    assert _counters(scheduler).lookup_shrink_tokens == {"recurrent_state": 64}


def test_deferred_lookup_counts_nothing():
    scheduler, req, _, _ = _lookup(ALL)
    state = scheduler._req_status["r"]
    in_flight = state.group_states[MAMBA].offload_keys[5]
    lookup = scheduler.manager.lookup.side_effect
    scheduler.manager.lookup.side_effect = lambda key, ctx: (
        None if key == in_flight else lookup(key, ctx)
    )
    hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert hit is None
    assert state.lookup_shrink is None


@pytest.mark.parametrize("eagle", [False, True], ids=["plain", "eagle"])
@pytest.mark.parametrize("seed", range(40))
def test_lookup_shrink_reconciles_random(seed, eagle):
    """Random host-tier contents: the causes add up to the full-attention
    reach minus the served hit, and tracking does not change the lookup."""
    rng = random.Random(seed)
    num_computed = rng.choice([0, 0, 32, 64])
    present = {
        g: [b for b in range(n) if rng.random() < (0.9 if g == FULL else 0.6)]
        for g, n in ((FULL, 6), (MAMBA, 6), (DRAFT, 12))
    }
    scheduler, req, hit, shrink = _lookup(
        present, num_computed=num_computed, eagle=eagle
    )
    calls = scheduler.manager.lookup.call_args_list
    # The full-attention lookup starts after the local hit.
    prefix = num_computed // TARGET_BLOCK
    while prefix in present[FULL]:
        prefix += 1
    reach = min(192, prefix * TARGET_BLOCK)
    if reach - num_computed < TARGET_BLOCK:
        assert shrink == {}
    else:
        assert sum(shrink.values()) == reach - (num_computed + hit)
        assert set(shrink) <= {"draft_window", "recurrent_state", "page_align"}
    # The same lookup with the counters disabled.
    off_scheduler = _scheduler(eagle=eagle, counters=False)
    off_req = _request("r", PROMPT)
    off_scheduler.on_new_request(off_req)
    off_state = off_scheduler._req_status["r"]
    off_state.update_offload_keys()
    keys = {
        off_state.group_states[g].offload_keys[b]
        for g, blocks in present.items()
        for b in blocks
    }
    off_scheduler.manager.lookup.side_effect = lambda key, ctx: key in keys
    off_hit, _ = off_scheduler.get_num_new_matched_tokens(off_req, num_computed)
    assert off_hit == hit
    assert off_scheduler.manager.lookup.call_args_list == calls
    assert off_req.shared_prefix_boundary == req.shared_prefix_boundary


# ---------------------------------------------------- draft window needed


@pytest.mark.parametrize(
    ("tokens", "needed"),
    [
        # 230 tokens, hit 192: 38 tokens left < 45-token window.
        (230, "true"),
        # 240 tokens, hit 192: 48 tokens left.
        (240, "false"),
    ],
)
def test_hit_draft_window_needed(tokens, needed):
    scheduler, req, hit, _ = _lookup(ALL, tokens=tokens)
    assert hit == 192
    _admit(scheduler, req, hit)
    assert _counters(scheduler).hit_draft_window_needed == {needed: 1}
    assert scheduler.offload_counter_deltas()["hit_draft_window_needed"] == {needed: 1}


def test_hit_draft_window_not_counted_without_external_hit():
    scheduler, req, hit, _ = _lookup({**ALL, MAMBA: []})
    assert hit == 0
    _admit(scheduler, req, 0)
    assert _counters(scheduler).hit_draft_window_needed == {}


def test_hit_draft_window_not_counted_without_draft_group():
    # The same geometry as a target model's own sliding window.
    scheduler, req, hit, _ = _lookup(
        ALL, kv_config=_kv_config("model.layers.2.self_attn.attn")
    )
    assert hit == 192
    _admit(scheduler, req, hit)
    assert _counters(scheduler).hit_draft_window_needed == {}


# -------------------------------------------------- stats and Prometheus


def test_counters_reported_once_through_stats():
    manager = _cpu_manager(grouped=False)
    scheduler = _scheduler("window", manager)
    _standard_run(scheduler, complete=True).finish()
    req = _sibling(scheduler, 230)
    hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    assert hit == 192
    _admit(scheduler, req, hit)
    # Shares only 128 tokens: no draft window stored there, so no hit.
    other = _request("r3", 230, hashes=_hashes(8) + _hashes(6, salt="y"))
    scheduler.on_new_request(other)
    assert scheduler.get_num_new_matched_tokens(other, 0)[0] == 0
    connector = _stats_connector(scheduler)
    stats = connector.get_kv_connector_stats()
    assert isinstance(stats, OffloadingConnectorStats)
    assert stats.reduce() == {
        "cpu_stored_blocks_target": 6,
        "cpu_stored_blocks_state": 2,
        "cpu_stored_blocks_draft": 5,
        "hit_draft_window_needed_true": 1,
    }
    # Reported once; the lookup of r3 counts when r3 is admitted.
    assert connector.get_kv_connector_stats() is None
    _admit(scheduler, other, 0)
    assert connector.get_kv_connector_stats().reduce() == {
        "lookup_shrink_tokens_draft_window": 128
    }
    assert connector.get_kv_connector_stats() is None


def test_lookup_shrink_reported_through_stats():
    scheduler, req, hit, _ = _lookup({**ALL, DRAFT: range(7, 11)})
    _admit(scheduler, req, hit)
    stats = _stats_connector(scheduler).get_kv_connector_stats()
    assert stats.reduce() == {
        "lookup_shrink_tokens_draft_window": 16,
        "lookup_shrink_tokens_page_align": 16,
        # Hit 160 of 200 tokens: 40 left to prefill, inside the window.
        "hit_draft_window_needed_true": 1,
    }


def test_prometheus_labelled_counters():
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

    registry = CollectorRegistry()

    def bind(cls):
        return lambda *a, **k: cls(*a, registry=registry, **k)

    prom = OffloadPromMetrics(
        _vllm_config_with_counters(),
        {Gauge: bind(Gauge), Counter: bind(Counter), Histogram: bind(Histogram)},
        ["model_name", "engine"],
        {0: ["m", "0"]},
    )
    first = OffloadingConnectorStats()
    first.record_labelled_counts("cpu_stored_blocks", {"target": 6, "state": 2})
    first.record_labelled_counts("lookup_shrink_tokens", {"recurrent_state": 64})
    merged = OffloadingConnectorStats(
        data={"GPU_to_CPU": [{"op_size": 8, "op_time": 0.1}]}
    ).aggregate(first)
    prom.observe(merged.data, engine_idx=0)
    second = OffloadingConnectorStats()
    second.record_labelled_counts("cpu_stored_blocks", {"target": 1, "draft": 3})
    second.record_labelled_counts("hit_draft_window_needed", {"true": 1})
    prom.observe(second.data, engine_idx=0)

    def sample(name, **labels):
        return registry.get_sample_value(
            name, {"model_name": "m", "engine": "0", **labels}
        )

    assert sample("vllm:kv_offload_cpu_stored_blocks_total", kind="target") == 7
    assert sample("vllm:kv_offload_cpu_stored_blocks_total", kind="state") == 2
    assert sample("vllm:kv_offload_cpu_stored_blocks_total", kind="draft") == 3
    assert (
        sample("vllm:kv_offload_lookup_shrink_tokens_total", cause="recurrent_state")
        == 64
    )
    assert sample("vllm:kv_offload_hit_draft_window_needed_total", needed="true") == 1
    assert merged.reduce()["cpu_stored_blocks_target"] == 6


def test_log_line_reports_totals(caplog_vllm):
    scheduler, req, hit, _ = _lookup({**ALL, MAMBA: [3]})
    _admit(scheduler, req, hit)
    scheduler._counters_logged_at = -1e9
    with caplog_vllm.at_level("INFO"):
        scheduler._maybe_log_counters()
    [line] = [
        r.getMessage()
        for r in caplog_vllm.records
        if r.getMessage().startswith("Offload counters")
    ]
    assert "stored_blocks_by_kind=target:0,draft:0,state:0" in line
    assert (
        "lookup_shrink_tokens=draft_window:0,recurrent_state:64,"
        "sliding_window:0,page_align:0" in line
    )
    assert "hit_draft_window_needed=true:0,false:1" in line


# ------------------------------------------------- no behaviour change


def _norm(value):
    """Deterministic, process-independent form of a call argument/result."""
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, bytes):
        return value.hex()
    if isinstance(value, (set, frozenset)):
        return sorted(_norm(v) for v in value)
    if isinstance(value, (list, tuple)):
        return [_norm(v) for v in value]
    if isinstance(value, dict):
        return sorted((_norm(k), _norm(v)) for k, v in value.items())
    fields = {}
    for name in (
        "keys_to_store",
        "store_spec",
        "evicted_keys",
        "block_ids",
        "group_sizes",
        "block_indices",
        "offload_keys",
    ):
        if hasattr(value, name):
            attr = getattr(value, name)
            if hasattr(attr, "tolist"):
                attr = attr.tolist()
            fields[name] = _norm(attr)
    if fields:
        return [type(value).__name__, sorted(fields.items())]
    return type(value).__name__  # ReqContext and the like


class _Recorder:
    """Wrap an offloading manager and log every call with its result."""

    def __init__(self, inner, log):
        self._inner = inner
        self._log = log

    def __getattr__(self, name):
        attr = getattr(self._inner, name)
        if name.startswith("_") or not callable(attr):
            return attr

        def call(*args, **kwargs):
            result = attr(*args, **kwargs)
            if name == "take_events":
                result = list(result)
            self._log.append((name, _norm(args), _norm(kwargs), _norm(result)))
            return result

        return call


def _meta_signature(meta):
    def jobs(job_map):
        return [
            (jid, job.req_id, _norm(list(job.transfer_spec)))
            for jid, job in sorted(job_map.items())
        ]

    return (
        jobs(meta.load_jobs),
        jobs(meta.store_jobs),
        sorted(meta.jobs_to_flush),
    )


def _trace(mode: str, grouped: bool, eagle: bool, counters: bool):
    """A multi-request scenario; returns everything the scheduler emits or
    asks the CPU tier: store and load jobs, flushes, lookup results, every
    manager call and the final pool contents."""
    log: list = []
    manager = _Recorder(_cpu_manager(grouped), log)
    scheduler = _scheduler(mode, manager, eagle=eagle, counters=counters)
    out: list = []

    # r1: chunked prefill; r2 looks up while r1's first stores are in flight.
    r1 = _request("r1", PROMPT, junction=160)
    run = Run(scheduler, r1)
    metas = [run.step(128), run.step(32, boundaries=(160,))]
    r2 = _sibling(scheduler, 240, "r2")
    out.append(("r2 deferred", scheduler.get_num_new_matched_tokens(r2, 0)))
    for meta in metas:
        _complete(scheduler, meta.store_jobs)
    metas += [run.step(32, boundaries=(192,)), run.step(8)]
    for meta in metas[2:]:
        _complete(scheduler, meta.store_jobs)
    run.finish()
    out += [("r1", _meta_signature(m)) for m in metas]

    # r2: external hit, load, then the rest of its prefill.
    hit, load_async = scheduler.get_num_new_matched_tokens(r2, 0)
    out.append(("r2 hit", hit, load_async, r2.shared_prefix_boundary))
    _admit(scheduler, r2, hit or 0)
    meta = scheduler.build_connector_meta(_output(r2, 0))
    out.append(("r2 load", _meta_signature(meta)))
    _complete(scheduler, meta.load_jobs)
    r2.num_computed_tokens = hit or 0
    meta = scheduler.build_connector_meta(
        _output(
            r2,
            240 - r2.num_computed_tokens,
            new=_block_ids(240, base=2000),
            boundaries=(224,),
        )
    )
    out.append(("r2 store", _meta_signature(meta)))
    _complete(scheduler, meta.store_jobs)
    r2.is_finished.return_value = True
    out.append(("r2 finished", scheduler.request_finished(r2)))

    # r3: shares only the first 128 tokens; no state there.
    r3 = _request("r3", 200, hashes=_hashes(8) + _hashes(4, salt="z"))
    scheduler.on_new_request(r3)
    hit, _ = scheduler.get_num_new_matched_tokens(r3, 0)
    out.append(("r3 hit", hit, r3.shared_prefix_boundary))
    _admit(scheduler, r3, hit or 0)

    # r4: 230 tokens after a reset of the tier: nothing to hit.
    scheduler.reset_cache()
    r4 = _sibling(scheduler, 230, "r4")
    out.append(("r4 hit", scheduler.get_num_new_matched_tokens(r4, 0)))
    out.append(("skipped", scheduler.num_skipped_draft_blocks))
    out.append(("demoted", scheduler.num_demoted_states))
    return out, log


TRACE_CASES = [
    (mode, grouped, eagle)
    for mode in MODES
    for grouped in (False, True)
    for eagle in (False, True)
]


@pytest.mark.parametrize(("mode", "grouped", "eagle"), TRACE_CASES)
def test_counters_do_not_change_behaviour(mode, grouped, eagle):
    """Jobs, lookups and every CPU-tier call are identical with the counters
    disabled through the deployment setting."""
    with_counters = _trace(mode, grouped, eagle, counters=True)
    without = _trace(mode, grouped, eagle, counters=False)
    assert with_counters == without
    digest = hashlib.sha256(repr(with_counters).encode()).hexdigest()
    print(f"TRACE {mode} grouped={grouped} eagle={eagle} {digest}")
