# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU teardown contracts; these doubles do not establish NCCL/GPU timing."""

import multiprocessing
import weakref
from threading import Event, Thread
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.compilation.cuda_graph import CUDAGraphWrapper
from vllm.distributed import parallel_state as ps
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
