# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU protocol, single-slot ownership and fused integer accounting."""

import contextlib
import os
from types import SimpleNamespace as NS
from unittest.mock import Mock

import msgspec
import pytest
import torch
from prometheus_client import REGISTRY

from vllm.v1.spec_decode.lookup_metrics import LookupStatsFrontend, LookupStatsProm
from vllm.v1.spec_decode.lookup_stats import (
    ZERO,
    LookupStatsAccumulator,
    LookupStatsMessage,
    LookupStatsRegistration,
    LookupStatsSnapshot,
    enabled,
    initialize_owner,
    register_frontend,
)
from vllm.v1.worker.gpu.spec_decode.dflash2.native_lookup import _native_fuse_kernel
from vllm.v1.worker.gpu.spec_decode.dflash2.stats import (
    LookupStatsExporter,
    create_exporter,
)

pytest_plugins = ("tests.v1.spec_decode.test_dflash2_lookup_scheduler",)
pytestmark = pytest.mark.skip_global_cleanup
DEVICE = os.environ.get("W21_TEST_DEVICE", "cpu")
REG = LookupStatsRegistration(0, "engine-0/rank-0", "first-generation", 0)


@pytest.mark.parametrize("value", [0, 1, 2**63, 2**64 - 1])
def test_capture_zero_assertion_uses_cpu_integers(value):
    from tests.v1.spec_decode.test_dflash2_lookup_stats_cuda import (
        assert_zero_counters,
    )

    class DeviceCounters:
        def __init__(self):
            self.readbacks = 0

        def any(self):
            raise NotImplementedError('"or_cuda" not implemented for UInt64')

        def cpu(self):
            self.readbacks += 1
            return torch.tensor([value] + [0] * 17, dtype=torch.uint64)

    counters = DeviceCounters()
    if value:
        with pytest.raises(AssertionError):
            assert_zero_counters(counters)
    else:
        assert_zero_counters(counters)
    assert counters.readbacks == 1


def vector(eligible=4, candidate=2, gate=2, adopted=1, length=5):
    result = [eligible, candidate, gate, adopted] + [0] * 14
    bucket = next(
        i
        for i, bound in enumerate(
            (1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128, float("inf"))
        )
        if length <= bound
    )
    result[4 + bucket] = candidate
    result[-1] = candidate * length
    return tuple(result)


def message(producer=100, sequence=7, covered=100, totals=None, timestamp=10.0):
    return LookupStatsMessage(
        REG.owner,
        REG.generation,
        producer,
        None
        if sequence is None
        else LookupStatsSnapshot(
            sequence, covered, vector() if totals is None else totals, timestamp
        ),
    )


@pytest.mark.parametrize("sequence", [None, 6, 7])
def test_independent_watermarks_idle_and_recovery(sequence):
    state = LookupStatsAccumulator(REG)
    assert state.merge(message()) == vector()
    assert state.pending == 0
    assert state.merge(message(101, sequence)) == ZERO
    assert (state.producer, state.sequence, state.covered, state.timestamp) == (
        101,
        7,
        100,
        10.0,
    ), "F3_PENDING_WATERMARK"
    assert state.pending == 1, "F3_PENDING_WATERMARK"
    # No timer or idle RPC is required to retain this evidence indefinitely.
    assert state.merge(message(99, 5)) == ZERO
    assert state.producer == 101
    assert state.merge(message(101, 8, 101, vector(8, 4, 4, 2), 20.0)) == vector()
    assert not state.pending
    assert state.timestamp == 20.0


@pytest.mark.parametrize(
    "field,value",
    [
        ("owner", "other"),
        ("generation", "retired"),
        ("producer", True),
        ("producer", -1),
    ],
)
def test_untrusted_messages_cannot_register_or_advance(field, value):
    state = LookupStatsAccumulator(REG)
    data = msgspec.to_builtins(message())
    data[field] = value
    state.merge(
        msgspec.convert(data, type=LookupStatsMessage, strict=False)
        if field != "producer"
        else LookupStatsMessage(
            data["owner"], data["generation"], value, message().snapshot
        )
    )
    assert state.producer == state.sequence == 0, "F3_UNTRUSTED_IDENTITY"
    assert state.rejected == 1


@pytest.mark.parametrize(
    "snapshot",
    [
        LookupStatsSnapshot(1, 101, vector(), 10.0),
        LookupStatsSnapshot(1, 100, ZERO[:-1], 10.0),
        LookupStatsSnapshot(1, 100, vector(candidate=5), 10.0),
        LookupStatsSnapshot(1, 100, vector(), float("nan")),
    ],
)
def test_invalid_snapshot_keeps_independently_advanced_producer(snapshot):
    state = LookupStatsAccumulator(REG)
    state.merge(LookupStatsMessage(REG.owner, REG.generation, 100, snapshot))
    assert state.producer == 100 and state.pending
    assert state.sequence == 0 and state.rejected == 1


@pytest.mark.parametrize(
    "log_stats,collect,mode,effective",
    [
        (True, True, "gpu_head", True),
        (False, True, "gpu_head", False),
        (True, False, "gpu_head", False),
        (True, True, "off", False),
        (True, True, "legacy", False),
    ],
)
def test_switch_owner_and_disabled_resources(
    monkeypatch, log_stats, collect, mode, effective
):
    config = NS(
        speculative_config=NS(
            resolved_dflash2_lookup=NS(mode=mode, collect_stats=collect)
        ),
        parallel_config=NS(
            data_parallel_rank=0,
            world_size=4,
            tensor_parallel_size=4,
            prefill_context_parallel_size=1,
        ),
    )
    assert enabled(config, log_stats) is effective, "F3_EFFECTIVE_SWITCH"
    state = initialize_owner(config, log_stats)
    assert (state is not None) is effective
    monkeypatch.setattr(
        torch, "zeros", Mock(side_effect=AssertionError("off allocation"))
    )
    config._lookup_stats_rank = 1
    assert create_exporter(config, torch.device("cpu")) is None
    if not effective:
        config._lookup_stats_rank = 0
        assert create_exporter(config, torch.device("cpu")) is None


class Event:
    def __init__(self):
        self.ready = False
        self.queries = 0
        self.records = 0

    def record(self, stream):
        self.records += 1

    def query(self):
        self.queries += 1
        return self.ready


@pytest.fixture
def exporter(monkeypatch):
    import vllm.v1.worker.gpu.spec_decode.dflash2.stats as module

    monkeypatch.setattr(module.torch.cuda, "Event", Event)
    monkeypatch.setattr(
        module.torch.cuda, "Stream", lambda device: NS(wait_event=Mock())
    )
    monkeypatch.setattr(
        module.torch.cuda, "current_stream", lambda device: "production"
    )
    monkeypatch.setattr(
        module.torch.cuda, "stream", lambda stream: contextlib.nullcontext()
    )
    original = torch.empty
    monkeypatch.setattr(
        module.torch,
        "empty",
        lambda *a, **kw: original(
            *a, **{k: v for k, v in kw.items() if k != "pin_memory"}
        ),
    )
    return LookupStatsExporter(REG, torch.device("cpu"))


def test_single_slot_busy_skip_detach_and_stream_order(exporter):
    exporter.counters.copy_(torch.tensor(vector(), dtype=torch.uint64))
    exporter.submitted()
    assert exporter.service(force=True).snapshot is None, "F3_BUSY_SLOT"
    assert exporter.busy and exporter.snapshot_covered == 1
    assert exporter.snapshot_ready.records == exporter.export_done.records == 1
    exporter.export_stream.wait_event.assert_called_once_with(exporter.snapshot_ready)
    exporter.counters.copy_(torch.tensor(vector(8, 4, 4, 2), dtype=torch.uint64))
    exporter.submitted()
    assert exporter.service(force=True).snapshot is None, "F3_BUSY_SLOT"
    assert exporter.snapshot_ready.records == 1
    exporter.export_done.ready = True
    first = exporter.service(force=True)
    assert first.snapshot.totals == vector()
    assert first.snapshot.covered == 1 and first.producer == 2
    assert exporter.busy and exporter.snapshot_covered == 2
    second = exporter.service()
    assert second.snapshot.totals == vector(8, 4, 4, 2)
    assert first.snapshot.totals == vector()  # immutable after slot overwrite
    assert second.snapshot.covered == 2 and not exporter.busy
    assert exporter.cpu_receipt()[1]
    assert 3 * exporter.counters.numel() * exporter.counters.element_size() <= 65536


def test_cpu_receipt_never_queries_or_waits(exporter):
    exporter.submitted()
    exporter.service(force=True)
    exporter.export_done.query = Mock(
        side_effect=AssertionError("F3_CPU_RECEIPT_QUERY")
    )
    assert not exporter.cpu_receipt()[1]
    exporter.lock.acquire()
    try:
        assert not exporter.cpu_receipt()[1]
        assert not exporter.receipt_callback_finished, "F3_CALLBACK_NOT_FINISHED"
    finally:
        exporter.lock.release()


def test_expired_drain_does_no_device_operations(exporter):
    exporter.submitted()
    exporter.export_done.query = Mock(side_effect=AssertionError("expired query"))
    assert not exporter.drain(0.0)[1]
    assert not exporter.busy and exporter.snapshot_ready.records == 0


def test_frontend_registered_roundtrip_and_prometheus():
    config = NS()
    register_frontend(config, REG)
    front = LookupStatsFrontend(config)
    prom = LookupStatsProm(config, ["model_name", "engine"], {0: ["toy", "0"]})
    try:
        encoded = msgspec.msgpack.encode(message())
        decoded = msgspec.msgpack.decode(encoded, type=LookupStatsMessage)
        assert front.observe(decoded, 0) == prom.observe(decoded, 0) == vector()
        prom.observe(message(101, 7), 0)
        prom.observe(message(102, None), 0)
        assert (
            REGISTRY.get_sample_value(
                "vllm:dflash2_lookup_stats_pending",
                {"model_name": "toy", "engine": "0"},
            )
            == 1
        )
        assert (
            REGISTRY.get_sample_value(
                "vllm:dflash2_lookup_eligible_rounds_total",
                {"model_name": "toy", "engine": "0"},
            )
            == 4
        )
        assert (
            REGISTRY.get_sample_value(
                "vllm:dflash2_lookup_match_length_bucket",
                {"model_name": "toy", "engine": "0", "le": "5.0"},
            )
            == 2
        )
        assert prom.states[0].timestamp == 10.0
        register_frontend(config, LookupStatsRegistration(0, REG.owner, "rebuilt", 0))
        prom.observe(message(100000), 0)
        assert prom.states[0].producer == 0
    finally:
        from vllm.v1.metrics.prometheus import unregister_vllm_metrics

        unregister_vllm_metrics()


@pytest.mark.parametrize("recording", [False, True])
def test_fused_counts_exact_independent_recomputation(recording):
    device = torch.device(DEVICE)
    lengths = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 13, 16, 17, 32, 33, 64, 65, 128, 129]
    # One launch per synthetic round. Only the admitted row counts.
    counters = torch.zeros(18, dtype=torch.uint64, device=device)
    expected = [0] * 18
    for i, length in enumerate(lengths):
        valid = (i % 7) + 1
        agrees = i % 3 != 0
        admitted = recording and i % 5 != 0
        draft = torch.full((2, 7), 9, dtype=torch.int64, device=device)
        lookup = torch.full_like(draft, 9 if agrees else 8)
        matches = torch.tensor([length, 999], dtype=torch.int32, device=device)
        valids = torch.tensor([valid, 7], dtype=torch.int32, device=device)
        eligible = torch.tensor([admitted, False], dtype=torch.int32, device=device)
        use = torch.zeros((2, 7), dtype=torch.int32, device=device)
        _native_fuse_kernel[(2,)](
            draft,
            7,
            lookup,
            7,
            matches,
            valids,
            eligible,
            use,
            NMIN=1,
            counters=counters,
            COLLECT_STATS=True,
            num_warps=1,
        )
        if admitted:
            expected[0] += 1
            if length >= 1:
                expected[1] += 1
                expected[2] += agrees
                expected[3] += agrees and valid >= 2
                bucket = sum(
                    length > b for b in (1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128)
                )
                expected[4 + bucket] += 1
                expected[-1] += length
    assert counters.tolist() == expected, "F3_COUNTER_VECTOR"


def test_pending_across_engine_scheduler_wire_and_actual_frontend(
    chain_factory, monkeypatch
):
    import asyncio

    from vllm.v1.engine import EngineCoreOutputs
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.engine.output_processor import OutputProcessor
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.spec_decode.lookup_stats import merge_output

    chain = chain_factory()
    config = chain.config
    register_frontend(config, REG)
    front = LookupStatsFrontend(config)
    core = NS(lookup_stats=LookupStatsAccumulator(REG))

    async def check():
        queue: asyncio.Queue[EngineCoreOutputs] = asyncio.Queue()

        async def get_output():
            return await queue.get()

        observed = asyncio.Event()

        def record(*, scheduler_stats, **kwargs):
            front.observe(scheduler_stats.lookup_stats, 0)
            observed.set()

        llm = object.__new__(AsyncLLM)
        llm.output_handler = None
        llm.engine_core = NS(get_output_async=get_output, shutdown=lambda **kw: None)
        llm.output_processor = OutputProcessor(None, log_stats=False)
        llm.log_stats = True
        llm.logger_manager = NS(record=record)
        llm.renderer = NS(stat_mm_cache=lambda: None, shutdown=lambda: None)
        llm._mm_shadow_hits = {}
        llm._run_output_handler()
        try:
            for item in (
                message(),
                message(101, 7),
                message(101, None),
                message(101, 8, 101, vector(8, 4, 4, 2), 20.0),
            ):
                output = ModelRunnerOutput([], {}, lookup_stats=item)
                merge_output(core, output)
                scheduled = chain.scheduler.schedule()
                result = chain.scheduler.update_from_output(scheduled, output)
                assert result[0].scheduler_stats.lookup_stats is item
                wire = msgspec.msgpack.decode(
                    msgspec.msgpack.encode(result[0]), type=EngineCoreOutputs
                )
                observed.clear()
                await queue.put(wire)
                await asyncio.wait_for(observed.wait(), 1.0)
                state = front.states[0]
                assert (
                    state.producer,
                    state.covered,
                    state.sequence,
                    state.pending,
                    state.timestamp,
                ) == (
                    core.lookup_stats.producer,
                    core.lookup_stats.covered,
                    core.lookup_stats.sequence,
                    core.lookup_stats.pending,
                    core.lookup_stats.timestamp,
                )
            assert not state.pending and state.totals == vector(8, 4, 4, 2)
        finally:
            llm.output_handler.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await llm.output_handler

    asyncio.run(check())


@pytest.mark.parametrize(
    "collect,log_stats", [(False, False), (False, True), (True, False), (True, True)]
)
def test_effective_switch_preserves_complete_proposals_q_and_rng(collect, log_stats):
    from tests.v1.spec_decode.test_dflash2_native_lookup import fixture_components
    from tests.v1.spec_decode.test_w21_native_distribution import (
        interpreter_math,
        make_selector_trial,
    )

    def run(on):
        with interpreter_math(torch.device(DEVICE)):
            spec, _, _ = make_selector_trial(3, torch.device(DEVICE))
        first = int(spec.draft_tokens[0, 0])
        native, states, mapping, draft, _, _, _, batch, sampled = fixture_components(
            [4, 5, first, 8, 4, 5]
        )
        native.stats = (
            NS(counters=torch.zeros(18, dtype=torch.uint64, device=DEVICE))
            if on
            else None
        )
        draft.copy_(spec.draft_tokens[:2])
        rng = torch.get_rng_state().clone()
        native.prepare(batch, sampled, real_run=True)
        native.apply(
            states,
            mapping,
            draft,
            2,
            spec.draft_logits,
            spec._cached_candidate_ids,
            spec._cached_candidate_scores,
        )
        assert torch.equal(torch.get_rng_state(), rng)
        return (
            draft,
            spec.draft_logits,
            spec._cached_candidate_ids,
            spec._cached_candidate_scores,
            spec.seeds,
            spec.sample_pos,
        )

    baseline = run(False)
    observed = run(collect and log_stats)
    for expected, actual in zip(baseline, observed):
        assert torch.equal(expected, actual)


@pytest.mark.skipif(DEVICE != "cpu", reason="CPU delayed-slot fault injection")
def test_slot_busy_survives_real_scheduler_cancel_and_one_two_switch(
    exporter, chain_factory
):
    from tests.v1.spec_decode.test_dflash2_native_lookup import reference_lookup
    from vllm.v1.request import RequestStatus

    chain = chain_factory()
    chain.spec._native_lookup.stats = exporter
    chain.add("one")
    expected = [0] * 18

    def step():
        scheduled, batch = chain.schedule()
        record = chain.execute(scheduled, batch)
        if (
            batch.num_reqs == 1
            and not batch.is_prefilling_np[0]
            and int(record.counts[0]) > 0
        ):
            slot = int(batch.idx_mapping[0])
            length = int(chain.runner.req_states.total_len.gpu[slot])
            history = chain.runner.req_states.all_token_ids.gpu[slot, :length].tolist()
            _, match, valid = reference_lookup(history, 5, 16, 65536)
            expected[0] += 1
            if match >= 5 and valid >= 1:
                expected[1] += 1
                expected[2] += 1
                expected[3] += valid >= 2
                expected[
                    4
                    + sum(
                        match > bound
                        for bound in (1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128)
                    )
                ] += 1
                expected[-1] += match
        exporter.submitted()
        exporter.service(force=True)
        assert exporter.counters.tolist() == expected
        return batch

    step()
    step()
    first = exporter.pinned.clone()
    chain.add("two")
    assert step().num_reqs == 2
    assert step().num_reqs == 2
    chain.scheduler.finish_requests("two", RequestStatus.FINISHED_ABORTED)
    assert step().num_reqs == 1
    chain.add("replacement")
    assert step().num_reqs == 2
    assert torch.equal(exporter.pinned, first)
    assert exporter.snapshot_ready.records == 1


def test_no_forward_carries_metadata_without_mutating_shared_empty(exporter):
    from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
    from vllm.v1.worker.gpu.spec_decode.dflash2.stats import attach_empty_output

    runner = NS(speculator=NS(_native_lookup=NS(stats=exporter)))
    exporter.submitted()
    result = attach_empty_output(runner, EMPTY_MODEL_RUNNER_OUTPUT)
    assert result.lookup_stats is not None, "F3_ZERO_TOKEN_METADATA"
    assert result.lookup_stats.producer == 1
    assert EMPTY_MODEL_RUNNER_OUTPUT.lookup_stats is None
    assert result is not EMPTY_MODEL_RUNNER_OUTPUT


def test_gpu_activity_after_snapshot_invalidates_cpu_quiescence(exporter):
    assert not exporter.quiescent
    exporter.submitted()
    exporter.service(force=True)
    exporter.export_done.ready = True
    exporter.service()
    assert exporter.quiescent
    exporter.begin_activity()
    assert not exporter.quiescent
    assert not exporter.cpu_receipt()[1]


@pytest.mark.parametrize(
    "expired,wrong_generation", [(True, False), (False, True), (False, False)]
)
def test_worker_explicit_deadline_identity_and_reply_check(
    monkeypatch, expired, wrong_generation
):
    import vllm.v1.worker.gpu.spec_decode.dflash2.stats as module

    deadline = 10.0
    clock = Mock(side_effect=[11.0] if expired else [9.0, 11.0])
    monkeypatch.setattr(module, "time", NS(monotonic=clock))
    drain = Mock(return_value=(message(), True))
    worker = NS(
        rank=0,
        vllm_config=NS(_lookup_stats_registration=REG),
        model_runner=NS(speculator=NS(_native_lookup=NS(stats=NS(drain=drain)))),
    )
    result = module.drain_worker(
        worker,
        stats_deadline_monotonic=deadline,
        expected_generation="wrong" if wrong_generation else REG.generation,
        expected_owner=REG.owner,
    )
    assert not result.quiescent
    if expired or wrong_generation:
        assert drain.call_count == 0, "F3_EXPIRED_DRAIN"
    else:
        drain.assert_called_once_with(deadline)
        assert result.error == "deadline"


def test_inflight_snapshot_cannot_prove_later_gpu_activity_quiet(exporter):
    exporter.submitted()
    exporter.service(force=True)
    exporter.begin_activity()
    exporter.export_done.ready = True
    exporter.service()
    assert exporter.covered == exporter.producer
    assert not exporter.quiescent, "F3_STALE_ACTIVITY"
    assert not exporter.cpu_receipt()[1]


def test_logging_zero_denominator_and_effective_adoption():
    config = NS()
    register_frontend(config, REG)
    logger = LookupStatsFrontend(config)
    capture = Mock()
    logger.observe(message(0, 1, 0, ZERO, 10.0), 0)
    logger.log(capture)
    text = capture.call_args.args[0] % capture.call_args.args[1:]
    assert "no samples" in text and "last_export=10.000" in text
    logger.observe(message(), 0)
    logger.observe(message(101, None), 0)
    logger.log(capture)
    text = capture.call_args.args[0] % capture.call_args.args[1:]
    assert "adoption=25.00%" in text and "pending=1" in text
    assert "eligible=4 candidate=2 gate=2 adopted=1" in text


def test_active_service_and_cpu_receipt_have_no_blocking_gpu_calls():
    import ast
    import inspect
    import textwrap

    for function in (LookupStatsExporter.service, LookupStatsExporter.cpu_receipt):
        tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
        forbidden = {"synchronize", "result", "wait"}
        if function is LookupStatsExporter.cpu_receipt:
            forbidden |= {"query", "copy_"}
        calls = [
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        ]
        assert not forbidden.intersection(calls)


@pytest.mark.parametrize("scenario", ["idle", "pending", "completed"])
def test_query_measurement_exercises_actual_event_states(exporter, scenario):
    from tests.v1.spec_decode.test_dflash2_lookup_stats_cuda import (
        CountedEvent,
        timed_service,
    )

    exporter.export_done = CountedEvent(exporter.export_done)
    state = LookupStatsAccumulator(REG)
    if scenario != "idle":
        exporter.submitted()
        exporter.service(force=True)
        exporter.export_done.event.ready = scenario == "completed"
    label, cpu, wall = timed_service(exporter, state)
    assert label == scenario, "F4_EVENT_STATE"
    assert exporter.export_done.queries == (scenario != "idle"), "F4_EVENT_QUERY_COUNT"
    assert cpu >= 0 and wall >= 0
    assert (state.sequence > 0) is (scenario == "completed"), "F4_COMPLETED_PAYLOAD"


@pytest.mark.parametrize("force", [False, True])
def test_normal_service_does_not_build_shutdown_checkpoint(exporter, force):
    exporter.begin_activity()
    exporter.service(force=force)
    assert exporter.snapshot_ready.records == 0, "F2_ACTIVE_MUST_NOT_CHECKPOINT"
    assert exporter.export_done.queries == 0
    assert exporter.producer == exporter.covered == 0
    assert not exporter.quiescent


@pytest.mark.parametrize("capture", [False, True])
def test_profile_masks_native_counter_updates(capture):
    from tests.v1.spec_decode.test_dflash2_native_lookup import fixture_components

    native, states, mapping, draft, logits, ids, scores, batch, sampled = (
        fixture_components([4, 5, 9, 8, 4, 5])
    )
    counters = torch.zeros(18, dtype=torch.uint64, device=DEVICE)
    native.stats = NS(counters=counters)
    native.prepare(batch, sampled, real_run=capture)
    if capture:
        native.reset_for_capture()
    native.apply(states, mapping, draft, 2, logits, ids, scores)
    assert counters.cpu().tolist() == [0] * 18, "PROFILE_COUNTERS_MUST_STAY_ZERO"
