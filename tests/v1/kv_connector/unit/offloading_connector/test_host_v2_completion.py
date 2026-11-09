# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU stream/event substitutes around the actual page ownership output and drain
paths."""

import ast
import inspect
import textwrap
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.worker import (
    OffloadingConnectorWorker,
)
from vllm.v1.kv_offload.host.page_table import (
    HostKVStep,
    HostPageTable,
    PageHandle,
    SealTicket,
)
from vllm.v1.kv_offload.worker.worker import TransferResult
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.gpu.async_utils import AsyncOutput
from vllm.v1.worker.gpu.kv_connector import ActiveKVConnector


def connector(monkeypatch, events, *, fail_seal=False):
    ticket = SealTicket(1, 1, 1, PageHandle(1, 0), b"key", 1)
    result = ActiveKVConnector.__new__(ActiveKVConnector)
    result._disabled = False
    result._host_step = HostKVStep(1, [], [ticket])
    result._host_sealed = result._host_seal_failed = False
    result._host_event = None
    result._host_indices = []

    def seal(*_):
        events.append("seal-d2h")
        if fail_seal:
            raise ValueError("copy failed, context usable")

    result._host_store = SimpleNamespace(
        seal=seal,
        fence=lambda: SimpleNamespace(
            synchronize=lambda: events.append("host-visible")
        ),
    )
    result.kv_connector = SimpleNamespace(
        wait_for_save=lambda: None,
        get_finished=lambda _: (set(), set()),
        get_block_ids_with_load_errors=lambda: set(),
        get_kv_connector_stats=lambda: None,
        get_kv_connector_kv_cache_events=lambda: None,
        build_connector_worker_meta=lambda: None,
        clear_connector_metadata=lambda: None,
    )
    monkeypatch.setattr(
        "vllm.distributed.get_world_group", lambda: SimpleNamespace(rank_in_group=0)
    )
    return result, ticket


def test_seal_before_async_dependency_and_last_output_contains_receipt(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    events: list[str] = []
    conn, ticket = connector(monkeypatch, events)
    # Check the production entry point places seal before the sampling branch
    # and before AsyncOutput constructs its dependency on the main stream.
    tree = ast.parse(textwrap.dedent(inspect.getsource(GPUModelRunner.sample_tokens)))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    seal_line = next(
        n.lineno
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "seal_host_step"
    )
    async_line = next(
        n.lineno
        for n in calls
        if isinstance(n.func, ast.Name) and n.func.id == "AsyncOutput"
    )
    assert seal_line < async_line
    conn.seal_host_step()

    class Event:
        def __init__(self, **_):
            pass

        def record(self, _):
            events.append("output-event")

        def synchronize(self):
            events.append("output-ready")

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "set_stream", lambda _: None)
    main = object()
    copy = SimpleNamespace(
        wait_stream=lambda stream: events.append("output-waits-main")
    )
    model_output = ModelRunnerOutput(
        req_ids=[], req_id_to_index={}, sampled_token_ids=[], prompt_logprobs_dict={}
    )
    output = AsyncOutput(
        model_output,
        SimpleNamespace(
            sampled_token_ids=torch.tensor([[7]]),
            logprobs_tensors=None,
            num_nans=None,
        ),
        torch.tensor([1]),
        main,
        copy,
    )
    model_output.kv_connector_output = conn.post_forward(set())
    final = output.get_output()
    meta = final.kv_connector_output.kv_connector_worker_meta
    assert meta.seal_results[ticket] == (1, 0, 0)
    assert meta.completed_steps == {1: 1}
    assert (
        events.index("seal-d2h")
        < events.index("output-waits-main")
        < events.index("output-event")
    )
    assert events.index("host-visible") < events.index("output-ready")
    assert conn._host_step is None  # No idle poll or next request.


@pytest.mark.parametrize("seal_failure", [False, True])
def test_non_sampling_branch_seals_and_reports_classified_results(
    monkeypatch, seal_failure
):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner

    events: list[str] = []
    conn, ticket = connector(monkeypatch, events, fail_seal=seal_failure)
    state = SimpleNamespace(
        input_batch=object(),
        attn_metadata=None,
        slot_mappings_by_layer=None,
        hidden_states=None,
        aux_hidden_states=None,
        finished_req_ids=set(),
    )
    runner = SimpleNamespace(
        execute_model_state=state,
        _sm70_v2_mtp_profile_pending=None,
        kv_connector=conn,
        host_history=conn._host_store,
        is_last_pp_rank=False,
        pp_handler=SimpleNamespace(receive=lambda _: True),
        postprocess_num_computed_tokens=lambda _: None,
        eplb=SimpleNamespace(step=lambda **_: None),
    )
    output = GPUModelRunner.sample_tokens(runner, None)
    assert output.kv_connector_output.kv_connector_worker_meta.seal_results[ticket] == (
        1,
        int(seal_failure),
        0,
    )
    assert events[0] == "seal-d2h" and "host-visible" in events


def test_no_forward_completion_retires_step_and_marks_unexecuted_ticket(monkeypatch):
    events: list[str] = []
    conn, ticket = connector(monkeypatch, events)
    output = conn.post_forward(set(), wait_for_save=False)
    assert output.kv_connector_worker_meta.seal_results[ticket] == (1, 1, 1)
    assert output.kv_connector_worker_meta.completed_steps == {1: 1}


def test_metadata_or_deduplicates_rank_and_failure_masks():
    ticket = SealTicket(1, 1, 1, PageHandle(1, 0), b"key", 1)
    first = OffloadingWorkerMetadata(
        seal_results={ticket: (1, 0, 0)}, completed_steps={1: 1}
    )
    duplicate = first.aggregate(first)
    assert duplicate.seal_results[ticket] == (1, 0, 0)
    last = OffloadingWorkerMetadata(
        seal_results={ticket: (2, 2, 0)}, completed_steps={1: 2}
    )
    final = duplicate.aggregate(last)
    assert final.seal_results[ticket] == (3, 2, 0)
    assert final.completed_steps == {1: 3}


def test_control_retry_preserves_final_step_and_job_receipts(monkeypatch):
    events: list[str] = []
    conn, ticket = connector(monkeypatch, events)
    attempts: list[int] = []

    def drain():
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("load not drained")
        return set(), {"request"}, OffloadingWorkerMetadata(completed_jobs={9: 1})

    conn.kv_connector.connector_worker = SimpleNamespace(drain=drain)
    conn.seal_host_step()
    with pytest.raises(RuntimeError, match="load not drained"):
        conn.drain_host_kv()
    result = conn.drain_host_kv()
    assert result.kv_connector_worker_meta.seal_results[ticket] == (1, 0, 0)
    assert result.kv_connector_worker_meta.completed_steps == {1: 1}
    assert result.kv_connector_worker_meta.completed_jobs == {9: 1}
    assert conn.drain_host_kv() is result
    assert len(attempts) == 2
    conn.close_host_kv()
    assert conn._host_store is None and not conn._host_indices


@pytest.mark.parametrize(
    "load,store", [(False, False), (True, False), (False, True), (True, True)]
)
def test_worker_control_drains_unfinished_load_and_last_delayed_store(load, store):
    worker = OffloadingConnectorWorker(
        SimpleNamespace(
            vllm_config=SimpleNamespace(
                kernel_config=SimpleNamespace(qsa_cross_group_hit_check=True)
            )
        )
    )
    events: list[str] = []
    pending: list[int] = []
    finished: list[TransferResult] = []
    if load:
        worker._load_jobs[1] = "request"
        pending.append(1)
    if store:
        worker._unsubmitted_store_jobs.append((2, (None, None)))

    def submit(job_id, _):
        events.append("submit-store")
        pending.append(job_id)

    def drain():
        events.append("drain")
        finished.extend(TransferResult(job_id=job, success=True) for job in pending)
        pending.clear()

    def get_finished():
        result = list(finished)
        finished.clear()
        return result

    worker._submit_transfer = submit
    worker.worker.handlers = [SimpleNamespace(drain=drain, get_finished=get_finished)]
    _, receiving, meta = worker.drain()
    assert not pending and not worker._unsubmitted_store_jobs
    assert receiving == ({"request"} if load else set())
    expected = ({1: 1} if load else {}) | ({2: 1} if store else {})
    assert (meta.completed_jobs if meta else {}) == expected
    if store:
        assert events.index("submit-store") < events.index("drain")


@pytest.mark.parametrize(
    "failure", [None, "rank-error", "missing-rank", "duplicate-rank", "gap"]
)
def test_core_waits_for_all_rank_drains_before_executor_shutdown(monkeypatch, failure):
    from vllm.v1.engine import core

    events: list[str] = []
    table = HostPageTable(2, 2)
    scheduler = SimpleNamespace(
        host_history_pool=SimpleNamespace(host_pages=table, _jobs={}),
        current_step=int(failure == "gap"),
        update_from_output=lambda *args: events.append("batch-result"),
        shutdown=lambda: events.append("scheduler-shutdown-close-unlink"),
    )
    receipts: list[tuple[int, None, str | None]] = [(0, None, None), (1, None, None)]
    if failure == "rank-error":
        receipts[1] = (1, None, "unretired DMA")
    elif failure == "missing-rank":
        receipts.pop()
    elif failure == "duplicate-rank":
        receipts[1] = receipts[0]

    def rpc(method):
        assert method == "drain_host_kv"
        events.append("rank-drains")
        return receipts

    engine = core.EngineCore.__new__(core.EngineCore)
    engine.scheduler = scheduler
    engine.model_executor = SimpleNamespace(
        collective_rpc=rpc, shutdown=lambda: events.append("executor-shutdown")
    )
    engine.structured_output_manager = SimpleNamespace(clear_backend=lambda: None)
    future: Future[object] = Future()
    future.set_result(object())
    engine.batch_queue = [(future, object(), Future())]
    monkeypatch.setattr(
        core, "cleanup_dist_env_and_memory", lambda: events.append("cleanup")
    )
    if failure is not None:
        with pytest.raises(RuntimeError, match="retaining resources"):
            engine.shutdown()
        assert events == ["batch-result", "rank-drains"]
    else:
        engine.shutdown()
        assert events == [
            "batch-result",
            "rank-drains",
            "executor-shutdown",
            "scheduler-shutdown-close-unlink",
            "cleanup",
        ]
        engine.shutdown()
        assert events.count("rank-drains") == 1
