# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU teardown contracts; these doubles do not establish NCCL/GPU timing."""

import multiprocessing
import os
import subprocess
import sys
import time
import weakref
from contextlib import contextmanager, nullcontext
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.distributed import parallel_state as ps
from vllm.distributed import shutdown as teardown
from vllm.distributed.device_communicators import pynccl
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.v1.executor import multiproc_executor as mp


@pytest.fixture
def wrappers(monkeypatch):
    instances: weakref.WeakSet[CUDAGraphWrapper] = weakref.WeakSet()
    monkeypatch.setattr(CUDAGraphWrapper, "_all_instances", instances)
    return instances


def retained_wrapper(instances, released):
    class Graph:
        def __del__(self):
            released.set()

    wrapper = CUDAGraphWrapper.__new__(CUDAGraphWrapper)
    wrapper.concrete_cudagraph_entries = {"shape": Graph()}
    wrapper.semantic_variant_graph_pools = {-1: object()}
    instances.add(wrapper)
    # The returned reference represents a Dynamo cache, independent of runner.
    return wrapper


def coordinator(name, order):
    group = ps.GroupCoordinator.__new__(ps.GroupCoordinator)
    group.device_group = name + ".nccl"
    group.cpu_group = name + ".gloo"
    group.device_communicator = SimpleNamespace(
        destroy=lambda: order.append(name + ".custom")
    )
    group.mq_broadcaster = None
    return group


@pytest.mark.parametrize("rank", range(4))
def test_worker_releases_cached_graphs_before_all_groups(monkeypatch, wrappers, rank):
    released = Event()
    cached_wrapper = retained_wrapper(wrappers, released)
    order: list[str] = []
    names = ("tp", "dcp", "pcp", "pp", "dp", "ep", "eplb", "world")
    # Include optional EPLB too: initialization, not rank-local iteration,
    # decides whether it exists. Every rank must traverse this same order.
    for name in names:
        group = coordinator(name, order)
        group.ranks = (
            [0, 1, 2, 3] if name != "dcp" else [rank // 2 * 2 + i for i in range(2)]
        )
        monkeypatch.setattr(ps, "_" + name.upper(), group)
    monkeypatch.setattr(ps, "context_isolated", lambda: False)
    monkeypatch.setattr(ps.torch.distributed, "is_initialized", lambda: True)

    def destroy(group=None):
        assert released.is_set(), "finalize is waiting for a retained CUDA graph"
        if group is not None:
            assert group.split(".")[0] + ".custom" in order
        order.append(group or "default")

    monkeypatch.setattr(ps.torch.distributed, "destroy_process_group", destroy)
    worker = mp.WorkerProc.__new__(mp.WorkerProc)
    worker.rpc_broadcast_mq = SimpleNamespace(shutdown=lambda: order.append("rpc"))
    worker.worker_response_mq = SimpleNamespace(
        shutdown=lambda: order.append("response")
    )
    worker.worker = SimpleNamespace(shutdown=lambda: order.append("runner.retired"))
    worker.shutdown()
    expected = ["rpc", "response", "runner.retired"]
    expected += [
        name + suffix for name in names for suffix in (".custom", ".nccl", ".gloo")
    ]
    assert order == expected + ["default"]
    assert not cached_wrapper.concrete_cudagraph_entries
    assert not cached_wrapper.semantic_variant_graph_pools
    assert all(getattr(ps, "_" + name.upper()) is None for name in names)


@pytest.mark.parametrize("rank", range(4))
def test_real_tp4_dcp2_initialization_has_the_same_teardown_order(monkeypatch, rank):
    import vllm.config

    order: list[str] = []
    membership: dict[str, list[list[int]]] = {}
    names = ("tp", "dcp", "pcp", "pp", "dp", "ep", "eplb")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=1, enable_elastic_ep=False, enable_eplb=False
        ),
        model_config=SimpleNamespace(is_moe=True),
    )
    monkeypatch.setattr(vllm.config, "get_current_vllm_config", lambda: config)
    for name in names:
        monkeypatch.setattr(ps, "_" + name.upper(), None)
    monkeypatch.setattr(ps, "_WORLD", SimpleNamespace(local_rank=rank))
    monkeypatch.setattr(ps, "context_isolated", lambda: False)
    monkeypatch.setattr(ps.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(ps.torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(ps.torch.distributed, "get_world_size", lambda: 4)

    def build(group_ranks, local_rank, backend, group_name, **kwargs):
        membership[group_name] = group_ranks
        group = coordinator(group_name, order)
        group.ranks = next(ranks for ranks in group_ranks if rank in ranks)
        group.rank_in_group = group.ranks.index(rank)
        return group

    monkeypatch.setattr(ps, "init_model_parallel_group", build)
    monkeypatch.setattr(
        ps.torch.distributed, "destroy_process_group", lambda pg: order.append(pg)
    )
    ps.initialize_model_parallel(
        tensor_model_parallel_size=4,
        decode_context_model_parallel_size=2,
        backend="nccl",
    )
    assert membership == {
        "tp": [[0, 1, 2, 3]],
        "dcp": [[0, 1], [2, 3]],
        "pcp": [[0], [1], [2], [3]],
        "pp": [[0], [1], [2], [3]],
        "dp": [[0], [1], [2], [3]],
        "ep": [[0, 1, 2, 3]],
    }
    ps.destroy_model_parallel()
    assert order == [
        name + suffix for name in names[:-1] for suffix in (".custom", ".nccl", ".gloo")
    ]


def test_finalize_double_blocks_until_graph_is_released(wrappers):
    """Negative control: runner reference deletion alone cannot unblock it."""
    released, entered, finished = Event(), Event(), Event()
    cache = retained_wrapper(wrappers, released)
    runner = SimpleNamespace(model=cache)
    runner.model = None

    def finalize():
        entered.set()
        released.wait()
        finished.set()

    thread = Thread(target=finalize, daemon=True)
    thread.start()
    try:
        assert entered.wait(1)
        assert not finished.wait(0.05), "negative control did not block"
        CUDAGraphWrapper.clear_all_graphs()
        assert finished.wait(1), "graph cleanup did not unblock finalize"
        assert not cache.concrete_cudagraph_entries
    finally:
        released.set()
        thread.join(1)


def test_custom_communicator_keeps_rendezvous_alive_until_closed(monkeypatch):
    order: list[str] = []
    group = coordinator("tp", order)
    communicator = CudaCommunicator.__new__(CudaCommunicator)

    def release(name):
        assert hasattr(group, "device_group") and hasattr(group, "cpu_group")
        order.append(name)

    # Keep an external reference to custom AR, as a captured callable can do.
    custom_ar = SimpleNamespace(close=lambda: release("custom_ar"))
    communicator.pynccl_comm = SimpleNamespace(destroy=lambda: release("pynccl"))
    communicator.ca_comm = custom_ar
    communicator.fi_ar_comm = SimpleNamespace(destroy=lambda: release("flashinfer"))
    communicator.all2all_manager = SimpleNamespace(destroy=lambda: release("all2all"))
    group.device_communicator = communicator
    monkeypatch.setattr(
        ps.torch.distributed, "destroy_process_group", lambda pg: order.append(pg)
    )
    group.destroy()
    assert order == [
        "pynccl",
        "custom_ar",
        "flashinfer",
        "all2all",
        "tp.nccl",
        "tp.gloo",
    ]
    assert communicator.ca_comm is None
    assert group.device_communicator is None
    # An external owner must not delay explicit close; repeated cleanup is safe.
    assert custom_ar is not None
    group.destroy()
    assert len(order) == 6


def test_worker_shutdown_returns_with_nonblocking_finalize_double(
    monkeypatch, wrappers
):
    """The production shutdown method must finish, not just log completion."""
    released, finished = Event(), Event()
    cache = retained_wrapper(wrappers, released)
    errors = []

    def destroy():
        # Model the NCCL persistent-graph wait without any CUDA calls.
        released.wait()

    monkeypatch.setattr(mp, "destroy_model_parallel", destroy)
    monkeypatch.setattr(mp, "destroy_distributed_environment", lambda: None)
    worker = mp.WorkerProc.__new__(mp.WorkerProc)
    worker.rpc_broadcast_mq = worker.worker_response_mq = None
    worker.worker = SimpleNamespace(shutdown=lambda: None)

    def shutdown():
        try:
            worker.shutdown()
        except Exception as error:
            errors.append(error)
        finally:
            finished.set()

    thread = Thread(target=shutdown, daemon=True)
    thread.start()
    try:
        assert finished.wait(1), "worker blocked in finalize after runner shutdown"
        assert not errors
        assert not cache.concrete_cudagraph_entries
    finally:
        released.set()
        thread.join(1)


def _shutdown_child(ready):
    released = Event()
    instances: weakref.WeakSet[CUDAGraphWrapper] = weakref.WeakSet()
    cache = retained_wrapper(instances, released)
    worker = mp.WorkerProc.__new__(mp.WorkerProc)
    worker.rpc_broadcast_mq = worker.worker_response_mq = None
    worker.worker = SimpleNamespace(shutdown=lambda: None)
    with (
        patch.object(CUDAGraphWrapper, "_all_instances", instances),
        patch.object(mp, "destroy_model_parallel", released.wait),
        patch.object(mp, "destroy_distributed_environment", lambda: None),
    ):
        ready.send("ready")
        worker.shutdown()
        assert not cache.concrete_cudagraph_entries
    ready.close()


def test_spawned_worker_exits_zero_without_forced_termination():
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    process = context.Process(target=_shutdown_child, args=(writer,))
    process.start()
    writer.close()
    try:
        assert reader.poll(15), "CPU child did not finish importing"
        assert reader.recv() == "ready"
        # Use the real executor's unchanged 4+4 s policy, excluding import time.
        mp.MultiprocExecutor._ensure_worker_termination([process])
        process.join(2)
        assert process.exitcode == 0, "worker needed a termination/kill signal"
    finally:
        # Only this test's exact child is eligible for cleanup.
        if process.is_alive():
            process.kill()
            process.join(2)
        reader.close()
        process.close()


def bare_worker(shutdown, retired=False):
    worker = mp.WorkerProc.__new__(mp.WorkerProc)
    worker.rpc_broadcast_mq = worker.worker_response_mq = None
    worker.worker = SimpleNamespace(
        shutdown=shutdown,
        model_runner=SimpleNamespace(_shutdown_gpu_retired=retired),
    )
    return worker


def cuda_communicator(pynccl_comm=None, ca_comm=None, all2all_manager=None):
    communicator = CudaCommunicator.__new__(CudaCommunicator)
    communicator.pynccl_comm = pynccl_comm
    communicator.ca_comm = ca_comm
    communicator.fi_ar_comm = None
    communicator.all2all_manager = all2all_manager
    return communicator


def test_contract_custom_close_error_continues_and_remains_failed(monkeypatch):
    from vllm.distributed.device_communicators import custom_all_reduce

    order: list[str] = []

    def free_buffer(pointer):
        order.append("ca.free.error")
        raise RuntimeError("injected custom AR cudaFree failure")

    monkeypatch.setattr(
        custom_all_reduce,
        "ops",
        SimpleNamespace(
            dispose=lambda pointer: order.append("ca.dispose"),
            free_shared_buffer=free_buffer,
        ),
    )
    custom_ar = custom_all_reduce.CustomAllreduce.__new__(
        custom_all_reduce.CustomAllreduce
    )
    custom_ar.disabled = False
    custom_ar._ptr = 1
    custom_ar.rank = 0
    custom_ar.meta_ptrs = [101]
    custom_ar.buffer_ptrs = [102]
    comm = cuda_communicator(
        ca_comm=custom_ar,
        all2all_manager=SimpleNamespace(
            destroy=lambda: order.append("all2all"),
        ),
    )
    group = coordinator("tp", order)
    group.device_communicator = comm
    monkeypatch.setattr(ps.torch.distributed, "destroy_process_group", order.append)
    with pytest.raises(RuntimeError, match="custom AR"):
        group.destroy()
    assert order == ["ca.dispose", "ca.free.error", "all2all", "tp.nccl", "tp.gloo"]
    assert group.device_communicator is comm
    assert comm.ca_comm is custom_ar and comm.all2all_manager is None
    assert custom_ar._ptr == 0  # This does not prove all buffers were freed.
    with pytest.raises(teardown.ShutdownError, match="custom AR"):
        group.destroy()
    assert len(order) == 5  # Never double-free an unknown partial allocation.


@pytest.mark.parametrize("retired", [False, True])
def test_contract_runner_error_uses_retirement_proof(monkeypatch, wrappers, retired):
    released = Event()
    cache = retained_wrapper(wrappers, released)
    order: list[str] = []

    def fail():
        worker.worker.model_runner._shutdown_gpu_retired = retired
        raise RuntimeError("injected connector cleanup error")

    monkeypatch.setattr(mp, "destroy_model_parallel", lambda: order.append("model"))
    monkeypatch.setattr(
        mp, "destroy_distributed_environment", lambda: order.append("world")
    )
    worker = bare_worker(fail, retired=True)
    with pytest.raises(RuntimeError, match="connector"):
        worker.shutdown()
    assert released.is_set() is retired
    assert bool(cache.concrete_cudagraph_entries) is not retired
    assert order == (["model", "world"] if retired else [])


@pytest.mark.parametrize("delays", [(0, 0, 0), (None, None, None), (0.04, 0.04, 1.0)])
def test_contract_pynccl_uses_one_deadline_and_requires_completion(
    monkeypatch, wrappers, delays
):
    release = Event()
    comms = []
    order: list[str] = []
    monkeypatch.setattr(teardown, "SHUTDOWN_TIMEOUT_S", 0.2)
    monkeypatch.setattr(
        pynccl.torch.accelerator, "device_index", lambda *a: nullcontext()
    )
    for name in ("TP", "DCP", "PCP", "PP", "DP", "EP", "EPLB"):
        monkeypatch.setattr(ps, "_" + name, None)
    for name, delay in zip(("TP", "DCP", "EP"), delays):
        comm = pynccl.PyNcclCommunicator.__new__(pynccl.PyNcclCommunicator)
        comm.available = True
        comm.disabled = False
        comm.device = SimpleNamespace(index=0)
        comm.comm = object()
        comm.nccl = SimpleNamespace(
            ncclCommAbort=lambda *a, delay=delay: release.wait(delay)
        )
        group = coordinator(name.lower(), order)
        group.device_communicator = cuda_communicator(comm)
        monkeypatch.setattr(ps, "_" + name, group)
        comms.append(comm)
    monkeypatch.setattr(ps, "context_isolated", lambda: False)
    monkeypatch.setattr(ps.torch.distributed, "destroy_process_group", order.append)
    monkeypatch.setattr(
        mp, "destroy_distributed_environment", lambda: order.append("world")
    )
    started = time.monotonic()
    try:
        if any(delays):
            with pytest.raises(
                teardown.ShutdownIncompleteError, match="abort incomplete"
            ):
                bare_worker(lambda: None).shutdown()
            assert time.monotonic() - started < 0.5
            assert "world" not in order
            pending = next(
                c
                for c in comms
                if hasattr(c, "_abort_thread") and c._abort_thread.is_alive()
            )
            assert pending.available and pending.disabled
            assert ps._TP is not None or ps._EP is not None
            assert not any("ep." in call for call in order)
        else:
            # All None delays block; zero delays return immediately.
            if delays[0] is None:
                with pytest.raises(
                    teardown.ShutdownIncompleteError, match="abort incomplete"
                ):
                    bare_worker(lambda: None).shutdown()
                assert time.monotonic() - started < 0.5
                assert not order
                assert ps._TP.device_communicator.pynccl_comm is comms[0]
                assert comms[0].available and comms[0].disabled
            else:
                bare_worker(lambda: None).shutdown()
                assert all(
                    not c.available and not c._abort_thread.is_alive() for c in comms
                )
                assert order[-1] == "world"
    finally:
        release.set()
        for comm in comms:
            if hasattr(comm, "_abort_thread"):
                comm._abort_thread.join(1)
                assert not comm._abort_thread.is_alive()


def test_pynccl_abort_error_and_retry_are_not_success(monkeypatch):
    calls = []

    def abort(handle):
        calls.append(handle)
        raise RuntimeError("lost peer")

    comm = pynccl.PyNcclCommunicator.__new__(pynccl.PyNcclCommunicator)
    comm.available, comm.disabled = True, False
    comm.device, comm.comm = SimpleNamespace(index=0), object()
    comm.nccl = SimpleNamespace(ncclCommAbort=abort)
    monkeypatch.setattr(
        pynccl.torch.accelerator, "device_index", lambda *a: nullcontext()
    )
    for _ in range(2):
        with pytest.raises(teardown.ShutdownIncompleteError, match="lost peer"):
            comm.destroy()
    assert len(calls) == 1 and comm.available


def test_cleanup_errors_do_not_skip_later_groups_or_world(monkeypatch, wrappers):
    order: list[str] = []

    def fail():
        order.append("tp.failed")
        raise RuntimeError("injected close failure")

    bad = SimpleNamespace(destroy=fail)
    for name in ("TP", "DCP", "PCP", "PP", "DP", "EP", "EPLB", "WORLD"):
        monkeypatch.setattr(
            ps, "_" + name, bad if name == "TP" else coordinator(name.lower(), order)
        )
    monkeypatch.setattr(ps, "context_isolated", lambda: False)
    monkeypatch.setattr(ps.torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        ps.torch.distributed,
        "destroy_process_group",
        lambda pg=None: order.append(pg or "default"),
    )
    with pytest.raises(teardown.ShutdownError, match="close failure"):
        bare_worker(lambda: None).shutdown()
    assert ps._TP is bad and ps._DCP is None and ps._WORLD is None
    assert order[0] == "tp.failed" and order[-1] == "default"


@pytest.mark.parametrize("present", [(), ("TP",), ("TP", "DCP"), ("EP",)])
def test_partial_group_inventory_cleanup(monkeypatch, present):
    order: list[str] = []
    for name in ("TP", "DCP", "PCP", "PP", "DP", "EP", "EPLB"):
        monkeypatch.setattr(
            ps,
            "_" + name,
            coordinator(name.lower(), order) if name in present else None,
        )
    monkeypatch.setattr(ps, "context_isolated", lambda: False)
    monkeypatch.setattr(ps.torch.distributed, "destroy_process_group", order.append)
    ps.destroy_model_parallel()
    assert order == [
        n.lower() + suffix
        for n in ("TP", "DCP", "EP")
        if n in present
        for suffix in (".custom", ".nccl", ".gloo")
    ]


def test_worker_clears_breakable_registry(monkeypatch, wrappers):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    wrapper = BreakableCUDAGraphWrapper.__new__(BreakableCUDAGraphWrapper)
    wrapper.entries = {"shape": object()}
    monkeypatch.setattr(
        BreakableCUDAGraphWrapper, "_all_instances", weakref.WeakSet([wrapper])
    )
    monkeypatch.setattr(mp, "destroy_model_parallel", lambda: None)
    monkeypatch.setattr(mp, "destroy_distributed_environment", lambda: None)
    bare_worker(lambda: None).shutdown()
    assert not wrapper.entries


def test_queue_cleanup_error_does_not_abandon_runner_or_groups(monkeypatch, wrappers):
    order: list[str] = []

    def fail():
        raise RuntimeError("queue close failed")

    worker = bare_worker(lambda: order.append("runner"))
    queue = SimpleNamespace(shutdown=fail)
    worker.rpc_broadcast_mq = queue
    worker.worker_response_mq = SimpleNamespace(
        shutdown=lambda: order.append("response")
    )
    monkeypatch.setattr(mp, "destroy_model_parallel", lambda: order.append("model"))
    monkeypatch.setattr(
        mp, "destroy_distributed_environment", lambda: order.append("world")
    )
    with pytest.raises(teardown.ShutdownError, match="queue close failed"):
        worker.shutdown()
    assert order == ["response", "runner", "model", "world"]
    assert worker.rpc_broadcast_mq is queue and worker.worker_response_mq is None


def test_graph_release_error_attempts_other_registry_then_isolates(
    monkeypatch, wrappers
):
    from vllm.compilation.breakable_cudagraph import BreakableCUDAGraphWrapper

    order: list[str] = []

    def fail():
        raise RuntimeError("graph close failed")

    monkeypatch.setattr(CUDAGraphWrapper, "clear_all_graphs", fail)
    monkeypatch.setattr(
        BreakableCUDAGraphWrapper, "clear_all_graphs", lambda: order.append("breakable")
    )
    monkeypatch.setattr(mp, "destroy_model_parallel", lambda: order.append("unsafe"))
    with pytest.raises(teardown.ShutdownIncompleteError, match="graph close failed"):
        bare_worker(lambda: None).shutdown()
    assert order == ["breakable"]


def _failed_entrypoint_child(ready, mode, events=None):
    """Use the real entrypoint/guard; no native calls or peer processes."""
    entry = mp.WorkerProc.worker_main
    real_shutdown = mp.WorkerProc.shutdown
    real_guard = teardown.process_shutdown_guard

    def record(event):
        if events is not None:
            with open(events, "a") as stream:
                stream.write(event + "\n")

    class Owner:
        def __del__(self):
            record("unsafe.destructor")
            Event().wait()

    @contextmanager
    def guard():
        record("guard.enter")
        with real_guard():
            yield
        record("guard.leave")

    class Worker:
        READY_STR = "READY"

        def __init__(self, *args, **kwargs):
            if mode in ("constructor", "constructor_owner"):
                if mode == "constructor_owner":
                    self.owner = Owner()
                ready.send({"status": "CONSTRUCTING"})
                raise RuntimeError("partial construction")
            self.rpc_broadcast_mq = None

            def wait_until_ready():
                if mode == "handshake":
                    raise RuntimeError("peer lost during handshake")

            self.worker_response_mq = SimpleNamespace(
                export_handle=lambda: None,
                wait_until_ready=wait_until_ready,
                shutdown=lambda: None,
            )
            self.peer_response_handles = []

            def shutdown():
                if mode == "retirement":
                    raise RuntimeError("retirement unconfirmed")
                if mode == "synchronize":
                    Event().wait()

            self.worker = SimpleNamespace(shutdown=shutdown)

        def monitor_death_pipe(self, death_pipe, shutdown_requested):
            if mode == "cancelled":
                shutdown_requested.set()

        def worker_busy_loop(self):
            if mode == "runtime_owner":
                local_owner = Owner()
                assert local_owner is not None
                raise RuntimeError("peer lost with a traceback-owned resource")
            if mode in ("runtime", "cancelled"):
                raise RuntimeError("cancelled" if mode == "cancelled" else "peer lost")

        shutdown = real_shutdown

    def destroy_groups():
        if mode == "finalize":
            Event().wait()

    with (
        patch.object(mp, "WorkerProc", Worker),
        patch.object(mp, "set_worker_net_device", lambda *a: None),
        patch.object(mp, "maybe_init_worker_tracer", lambda **kw: None),
        patch.object(mp, "start_watch_dog", lambda *a: None),
        patch.object(mp, "get_watch_dog", lambda: SimpleNamespace(stop=lambda: None)),
        patch.object(mp, "destroy_model_parallel", destroy_groups),
        patch.object(mp, "destroy_distributed_environment", lambda: None),
        patch.object(teardown, "process_shutdown_guard", guard),
    ):
        entry(
            local_rank=0,
            rank=0,
            ready_pipe=ready,
            inherited_fds=[],
            vllm_config=SimpleNamespace(
                parallel_config=SimpleNamespace(numa_bind=False),
                watchdog_config=None,
            ),
        )


@pytest.mark.parametrize(
    "mode",
    [
        "constructor",
        "constructor_owner",
        "handshake",
        "runtime",
        "runtime_owner",
        "cancelled",
        "retirement",
        "synchronize",
        "finalize",
    ],
)
def test_terminal_worker_preserves_failure_state_and_deadline(mode, tmp_path):
    context = multiprocessing.get_context("spawn")
    reader, writer = context.Pipe(duplex=False)
    events = tmp_path / "terminal.events"
    process = context.Process(
        target=_failed_entrypoint_child, args=(writer, mode, str(events))
    )
    process.start()
    writer.close()
    signals = []

    def signal_process(name, action):
        signals.append(name)
        action()

    observed = SimpleNamespace(
        is_alive=process.is_alive,
        terminate=lambda: signal_process("TERM", process.terminate),
        kill=lambda: signal_process("KILL", process.kill),
    )
    try:
        assert reader.poll(15), "CPU child import did not complete"
        assert reader.recv()["status"] in ("READY", "CONSTRUCTING")
        started = time.monotonic()
        mp.MultiprocExecutor._ensure_worker_termination([observed])
        process.join(1)
        assert process.exitcode == (0 if mode == "cancelled" else 1)
        assert not signals
        assert time.monotonic() - started < 4
        assert "guard.enter" in events.read_text().splitlines()
        assert "unsafe.destructor" not in events.read_text().splitlines()
    finally:
        if process.is_alive():
            process.kill()
            process.join(2)
        reader.close()
        process.close()


@pytest.mark.parametrize("expire", [False, True])
@pytest.mark.parametrize("full_pipe", [False, True])
def test_guard_preserves_shared_stderr_and_exits_under_backpressure(expire, full_pipe):
    reader, writer = os.pipe()
    child = None
    try:
        if full_pipe:
            os.set_blocking(writer, False)
            try:
                while True:
                    os.write(writer, b"x" * 4096)
            except BlockingIOError:
                pass
            os.set_blocking(writer, True)
        # Redirect only after imports: import warnings are outside the guard.
        code = """
import os, sys, threading
from vllm.distributed import shutdown
shutdown.PROCESS_SHUTDOWN_TIMEOUT_S = 0.1
os.dup2(int(sys.argv[1]), 2)
with shutdown.process_shutdown_guard():
    if sys.argv[2] == 'True':
        threading.Event().wait(30)
"""
        child = subprocess.Popen(
            [sys.executable, "-c", code, str(writer), str(expire)],
            pass_fds=(writer,),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        assert child.wait(timeout=20) == (1 if expire else 0)
        assert os.get_blocking(writer), "guard changed shared file status flags"
    finally:
        if child is not None and child.poll() is None:
            child.kill()
            child.wait(timeout=3)
        os.close(writer)
        os.close(reader)


@pytest.mark.parametrize("interrupt", [None, "raise", "before", "after"])
def test_partial_close_error_survives_later_interruption(monkeypatch, interrupt):
    from vllm.distributed.device_communicators import custom_all_reduce

    calls = []
    exhausted = False

    def free(pointer):
        nonlocal exhausted
        calls.append("ca.free.failed")
        exhausted = interrupt == "before"
        raise RuntimeError("partial custom AR free failed")

    monkeypatch.setattr(
        custom_all_reduce,
        "ops",
        SimpleNamespace(
            dispose=lambda ptr: calls.append("ca.dispose"), free_shared_buffer=free
        ),
    )
    ar = custom_all_reduce.CustomAllreduce.__new__(custom_all_reduce.CustomAllreduce)
    ar.disabled, ar._ptr, ar.rank = False, 1, 0
    ar.meta_ptrs, ar.buffer_ptrs = [101], [102]

    def later_destroy():
        nonlocal exhausted
        calls.append("all2all.destroy")
        if interrupt == "raise":
            raise teardown.ShutdownIncompleteError("later release incomplete")
        exhausted = interrupt == "after"

    monkeypatch.setattr(
        teardown, "remaining_shutdown_time", lambda: 0.0 if exhausted else 3.0
    )
    comm = CudaCommunicator.__new__(CudaCommunicator)
    comm.pynccl_comm = comm.fi_ar_comm = None
    comm.ca_comm = ar
    manager = SimpleNamespace(destroy=later_destroy)
    comm.all2all_manager = manager
    expected = teardown.ShutdownIncompleteError if interrupt else teardown.ShutdownError
    with pytest.raises(expected):
        comm.destroy()
    before_retry = list(calls)
    with pytest.raises(teardown.ShutdownError, match="partial custom AR free failed"):
        comm.destroy()
    assert comm.ca_comm is ar and ar._ptr == 0
    assert comm._destroy_errors and calls == before_retry
    assert comm.all2all_manager is (manager if interrupt else None)
