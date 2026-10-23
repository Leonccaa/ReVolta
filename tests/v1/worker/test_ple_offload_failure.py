# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU process-failure and CUDA-context contract regressions for PLE."""

import ctypes
import multiprocessing
import os
import signal
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from tests.utils import set_lazy_env
from vllm.model_executor.layers.ple_offload_layer import ple_offload_enabled
from vllm.models.qwen4_exp.common.ple import PLERemotePlacement, plan_ple_placement
from vllm.models.qwen4_exp.nvidia import ple_layer
from vllm.v1.ple_offload import worker


def test_resident_production_config_does_not_enable_process_offload(monkeypatch):
    for name in ("VLLM_PLE_CPU_OFFLOAD", "VLLM_PLE_DISK_OFFLOAD"):
        set_lazy_env(monkeypatch, name, None)
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", "0")
    config = SimpleNamespace(
        kernel_config=SimpleNamespace(ple_disk_cascade_active=False)
    )
    assert not ple_offload_enabled(config)


def _fake_offload(ready, trigger, mode, death_reader, death_writer):
    death_writer.close()
    if mode == "startup":
        os._exit(17)
    ready.send({"status": "READY", "layer_names": []})
    ready.close()
    if mode == "close":
        try:
            death_reader.recv()
        except EOFError:
            return
    if mode == "immediate":
        os._exit(17)
    trigger.wait()
    if mode == "signal":
        os.kill(os.getpid(), signal.SIGKILL)
    raise RuntimeError("injected PLE service failure")


def _fake_gpu_side(started, trigger, mode, error_path):
    # No CUDA calls: an Event wait models a host thread blocked on GPU work.
    # The real lifecycle code must fail even when that thread never returns.
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    death_reader, death_writer = context.Pipe(duplex=False)
    proc = context.Process(
        target=_fake_offload, args=(writer, trigger, mode, death_reader, death_writer)
    )
    proc.start()
    writer.close()
    death_reader.close()
    handle = worker.PleOffloadWorkerHandle(proc, death_writer, reader)

    def record_error(message, *args, **kwargs):
        import traceback

        with open(error_path, "a") as output:
            output.write(message % args + "\n" + traceback.format_exc())

    worker.logger.exception = record_error
    started.send("waiting")
    try:
        worker.PleOffloadWorker.wait_for_ready(handle)
    except RuntimeError as error:
        with open(error_path, "a") as output:
            output.write(str(error))
        os._exit(1)
    if mode in ("service", "signal"):
        started.send("serving")
    if mode == "close":
        handle.close()
        handle.close()
        return
    threading.Event().wait(60)


@pytest.mark.parametrize("mode", ["startup", "immediate", "service", "signal"])
def test_gpu_side_fails_promptly_when_offload_exits(tmp_path, mode):
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    trigger = context.Event()
    error_path = tmp_path / "failure.txt"
    gpu = context.Process(
        target=_fake_gpu_side, args=(writer, trigger, mode, error_path)
    )
    gpu.start()
    writer.close()
    try:
        assert reader.poll(5), "fake GPU worker failed to start"
        assert reader.recv() == "waiting"
        if mode in ("service", "signal"):
            assert reader.poll(5), "PLE worker did not become ready"
            assert reader.recv() == "serving"
        start = time.monotonic()
        trigger.set()
        gpu.join(timeout=3)
        assert not gpu.is_alive(), "GPU side hung after PLE worker exit"
        assert time.monotonic() - start < 3
        assert gpu.exitcode == 1
        message = error_path.read_text()
        expected_code = (
            -signal.SIGKILL if mode == "signal" else (1 if mode == "service" else 17)
        )
        assert "PLE offload worker exited" in message
        assert f"exit code {expected_code}" in message
    finally:
        if gpu.is_alive():
            gpu.kill()
        gpu.join(timeout=5)
        reader.close()


def test_intentional_close_does_not_fail_gpu_side(tmp_path):
    context = multiprocessing.get_context("fork")
    reader, writer = context.Pipe(duplex=False)
    trigger = context.Event()
    gpu = context.Process(
        target=_fake_gpu_side,
        args=(writer, trigger, "close", tmp_path / "failure.txt"),
    )
    gpu.start()
    writer.close()
    try:
        assert reader.poll(5)
        reader.recv()
        trigger.set()
        gpu.join(timeout=3)
        assert gpu.exitcode == 0
    finally:
        if gpu.is_alive():
            gpu.kill()
        gpu.join(timeout=5)
        reader.close()


@pytest.mark.parametrize("num_tokens", [1, 5])
def test_resident_only_cascade_h2d_selects_each_target_device(monkeypatch, num_tokens):
    # Zero device rows and zero disk rows still produce a nonempty result.
    # Exercise the real all-resident worker lookup, not a fake empty tensor.
    layer = ple_layer.Qwen4ExpNGramEmbedding.__new__(ple_layer.Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(layer)
    layer._cascade = True
    placement = plan_ple_placement(
        total_rows=32,
        row_bytes=2,
        host_budget_bytes=64,
        vram_budget_bytes=0,
        disk_allowed=True,
    )
    assert placement.vram_rows == placement.disk_rows == 0
    assert placement.host_rows == 32
    layer._remote_placements = [PLERemotePlacement(0, 32, placement.local_rows)]
    layer._disk_segments = []
    layer.head_dim = 2
    layer.embedding_dim = 4
    layer.compute_ngram_ids = lambda *args: torch.zeros(num_tokens, 2, dtype=torch.long)
    monkeypatch.setattr(ple_layer, "is_offload_process", lambda: True)
    monkeypatch.setattr(torch.Tensor, "is_pinned", lambda self: True)
    current = [3]
    copies = []
    signals = []

    @contextmanager
    def device_guard(index):
        previous = current[0]
        current[0] = index
        try:
            yield
        finally:
            current[0] = previous

    monkeypatch.setattr(torch.accelerator, "device_index", device_guard)

    class Destination:
        is_cuda = True
        dtype = torch.uint8

        def __init__(self, rank, storage):
            self.device = torch.device("cuda", rank)
            self.storage = storage

        def __getitem__(self, slices):
            return Destination(self.device.index, self.storage[slices])

        def is_contiguous(self):
            return self.storage.is_contiguous()

        def numel(self):
            return self.storage.numel()

        def data_ptr(self):
            return self.storage.data_ptr()

    def copy(destination, source, size, stream):
        rank = int(stream) - 100
        if current[0] != rank:
            return (SimpleNamespace(value=1),)
        assert int(destination) != 0 and source != 0 and size == num_tokens * 4
        ctypes.memmove(int(destination), source, size)
        copies.append(rank)
        return (SimpleNamespace(value=0),)

    monkeypatch.setattr(worker.cuda_driver, "cuMemcpyHtoDAsync", copy)

    def target(rank):
        def signal_done(stream):
            assert current[0] == rank
            signals.append(rank)

        return worker.PleOffloadOutputTarget(
            tp_rank=rank,
            gpu_output_buffer=Destination(
                rank, torch.full((8, 4), 99, dtype=torch.uint8)
            ),
            sem=SimpleNamespace(wait_reset=lambda _: None, signal=signal_done),
            copy_stream=SimpleNamespace(
                cuda_stream=100 + rank, synchronize=lambda: None
            ),
        )

    runner = worker.PleOffloadRunner.__new__(worker.PleOffloadRunner)
    runner._clamp_input_ids = False
    runner._layers = {"ple": layer}
    targets = [target(rank) for rank in range(4)]
    runner._worker_targets = {0: {"ple": targets}}
    runner._pinned_bufs = {0: {"ple": torch.full((8, 4), 77, dtype=torch.uint8)}}
    runner._input_bufs = {
        0: worker.PleOffloadInputBuffers(
            torch.ones(num_tokens, dtype=torch.int32),
            torch.tensor([0, num_tokens]),
            None,
        )
    }
    runner._handle_requests([worker.PleOffloadRequest(0, num_tokens, 1)])
    assert copies == signals == [0, 1, 2, 3]
    assert current[0] == 3
    for output in targets:
        assert torch.count_nonzero(output.gpu_output_buffer.storage[:num_tokens]) == 0
        assert torch.all(output.gpu_output_buffer.storage[num_tokens:] == 99)
