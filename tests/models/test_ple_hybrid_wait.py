# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for hybrid-local and cascade-remote PLE synchronization."""

import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tests.utils import set_lazy_env
from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
from vllm.model_executor.layers import ple_offload_layer
from vllm.model_executor.layers.ple_offload_layer import CpuGpuSemaphore
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
from vllm.v1.ple_offload.connector import PleOffloadConnector

pytestmark = pytest.mark.skip_global_cleanup


@pytest.fixture
def kernel_warmup():
    # GPUModelRunner is only a type annotation in warmup.py. Load the real
    # warmup code without importing its GPU-only runner dependency on CPU hosts.
    path = Path(ple_offload_layer.__file__).parents[2] / "v1/worker/gpu/warmup.py"
    with patch.dict(
        sys.modules,
        {"vllm.v1.worker.gpu.model_runner": SimpleNamespace(GPUModelRunner=object)},
    ):
        return runpy.run_path(str(path))["warmup_kernels"]


class _Embedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, ids, remote_rows=None):
        self.calls.append(remote_rows)
        local = ids[..., None].float()
        if remote_rows is None:
            return local
        # A tiny cascade with resident IDs below 2 and remote IDs from 2 on.
        return torch.where(ids[..., None] < 2, local, remote_rows[..., None])


@pytest.fixture
def lane(monkeypatch):
    """Use real layer/connector/semaphore methods with CPU buffers and driver stubs."""
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_DUAL_COMPILE", "1")
    monkeypatch.setattr(ple_offload_layer, "_offload_worker_flag", False)
    layer = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    torch.nn.Module.__init__(layer)
    layer.ngram_heads = 1
    layer.layer_name = "ple"
    layer.ngram_embedding = _Embedding()
    flag = torch.zeros(1, dtype=torch.int32)
    sem = CpuGpuSemaphore.from_ipc_tensor(flag)
    layer.setup_cross_process_offload(torch.full((64, 1), -1.0), sem)
    events = []
    stream = SimpleNamespace(cuda_stream=17)

    def write(_, pointer, value, flags):
        assert pointer == flag.data_ptr() and flags == 0
        events.append(("write", value))
        flag.fill_(value)
        return (SimpleNamespace(value=0),)

    def wait(_, pointer, value, flags):
        assert pointer == flag.data_ptr()
        wait_flags = ple_offload_layer.cuda_driver.CUstreamWaitValue_flags
        assert flags == wait_flags.CU_STREAM_WAIT_VALUE_EQ.value
        events.append(("wait", value))
        # Fail immediately instead of letting an orphaned GPU wait hang pytest.
        assert flag.item() == value, "PLE wait has no completed remote producer"
        return (SimpleNamespace(value=0),)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: stream)
    monkeypatch.setattr(ple_offload_layer.cuda_driver, "CUstream", lambda x: x)
    monkeypatch.setattr(ple_offload_layer.cuda_driver, "CUdeviceptr", lambda x: x)
    monkeypatch.setattr(ple_offload_layer.cuda_driver, "cuStreamWriteValue32", write)
    monkeypatch.setattr(ple_offload_layer.cuda_driver, "cuStreamWaitValue32", wait)
    monkeypatch.setattr(
        torch.ops.vllm, "ple_offload_wait", ple_offload_layer._ple_offload_wait_impl
    )
    monkeypatch.setattr(
        torch.ops.vllm,
        "qwen4_exp_compute_ple_ngram_ids",
        lambda ids, qsl, ctx, out, name: out.copy_(ids[:, None]),
    )
    connector = PleOffloadConnector.__new__(PleOffloadConnector)
    connector.device = torch.device("cpu")
    connector._layers = {"ple": layer}

    def produce(num_reqs, num_tokens):
        assert flag.item() == CpuGpuSemaphore.RESET_VALUE
        events.append(("request", num_tokens))
        layer._gpu_output_buffer[:num_tokens].fill_(42)
        sem.signal(stream)

    connector._launch = Mock(side_effect=produce)

    def forward(num_tokens, full):
        ids = torch.arange(num_tokens)
        with sm70_decode_graph_compilation(full):
            return layer(
                torch.zeros(num_tokens, 1),
                ids,
                torch.tensor([0, num_tokens]),
                torch.zeros(1, 1, dtype=torch.long),
            )

    return SimpleNamespace(
        layer=layer, connector=connector, flag=flag, events=events, forward=forward
    )


@pytest.mark.parametrize("full", [False, True], ids=["prefill", "FULL-decode"])
@pytest.mark.parametrize("mode", ["hybrid", "cascade", "whole-table", "local"])
def test_only_remote_consumers_wait(monkeypatch, lane, mode, full):
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", str(int(mode == "hybrid")))
    lane.layer._keeps_local_tables = mode == "cascade"
    lane.layer._is_cpu_offloaded = mode != "local"
    needs_remote = mode in ("cascade", "whole-table") or (mode == "hybrid" and not full)
    if mode != "local":
        lane.connector.prepare_forward(1, 5, dummy_run=False, use_local_model=full)
    output = lane.forward(5, full)
    assert lane.connector._launch.call_count == int(needs_remote)
    assert lane.events.count(("wait", 1)) == int(needs_remote)
    assert ("write", 0) not in lane.events  # Waiting never releases the buffer.
    expected = torch.arange(5).float()[:, None]
    if mode == "cascade":
        expected[2:] = 42
        assert len(lane.layer.ngram_embedding.calls) == 1
        assert lane.layer.ngram_embedding.calls[0] is not None
    elif needs_remote:
        expected.fill_(42)
        assert not lane.layer.ngram_embedding.calls
    else:
        assert lane.layer.ngram_embedding.calls == [None]
    torch.testing.assert_close(output, expected)
    lane.connector.release_outputs()
    assert lane.flag.item() == 0
    assert lane.events.count(("write", 0)) == int(mode != "local")


@pytest.mark.parametrize("full", [False, True])
@pytest.mark.parametrize("mode", ["hybrid", "cascade", "whole-table"])
def test_dummy_signal_and_release_keep_flag_lifecycle(monkeypatch, lane, mode, full):
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", str(int(mode == "hybrid")))
    lane.layer._keeps_local_tables = mode == "cascade"
    lane.connector.prepare_forward(1, 5, dummy_run=True, use_local_model=full)
    assert lane.flag.item() == 1
    assert torch.count_nonzero(lane.layer._gpu_output_buffer[:5]) == 0
    assert torch.all(lane.layer._gpu_output_buffer[5:] == -1)
    lane.forward(5, full)
    assert lane.flag.item() == 1
    lane.connector._launch.assert_not_called()
    assert lane.events == [("write", 1)] + (
        [] if mode == "hybrid" and full else [("wait", 1)]
    )
    lane.connector.release_outputs()
    assert lane.flag.item() == 0
    assert lane.events[-1] == ("write", 0)


@pytest.mark.parametrize("steps", [4, 7], ids=["MTP4", "DFlash2-K7"])
@pytest.mark.parametrize("num_reqs", [1, 4])
def test_speculative_capture_release_then_kernel_warmup(
    monkeypatch, lane, kernel_warmup, steps, num_reqs
):
    set_lazy_env(monkeypatch, "VLLM_SM70_QWEN38_HYBRID_PLE", "1")
    lane.layer._keeps_local_tables = False
    # The runner signals once before capture and resets after all captures.
    lane.connector.signal_dummy_outputs(64)
    lane.forward((steps + 1) * num_reqs, full=True)
    capture_events = lane.events.copy()
    lane.connector.release_outputs()
    assert lane.flag.item() == 0
    lane.events.clear()
    kv_disabled: list[bool] = []
    runner = SimpleNamespace(
        num_speculative_steps=steps,
        scheduler_config=SimpleNamespace(
            max_num_seqs=num_reqs, max_num_batched_tokens=64
        ),
        max_model_len=64,
        compilation_config=SimpleNamespace(static_forward_context={}),
        kv_cache_config=SimpleNamespace(kv_cache_groups=[], num_blocks=64),
        is_pooling_model=False,
        is_last_pp_rank=True,
        model_config=SimpleNamespace(get_vocab_size=lambda: 64),
        kv_connector=SimpleNamespace(set_disabled=kv_disabled.append),
    )
    outputs = []
    samples: list[object] = []

    def execute(output):
        outputs.append(output)
        num_tokens = output.total_num_scheduled_tokens
        if not num_tokens:
            return
        # CPU facade for MRV2's dispatch: the uniform speculative decode
        # matches a FULL capture; prefill exceeds its per-request query length.
        full = bool(output.scheduled_spec_decode_tokens)
        lane.connector.prepare_forward(
            num_reqs, num_tokens, False, use_local_model=full
        )
        result = lane.forward(num_tokens, full)
        if full:
            assert lane.flag.item() == 0  # No producer is needed for this decode.
            torch.testing.assert_close(result[:, 0], torch.arange(num_tokens).float())
        lane.connector.release_outputs()

    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: None)
    kernel_warmup(runner, execute, samples.append)
    assert capture_events == [("write", 1)]
    assert [o.total_num_scheduled_tokens for o in outputs] == [
        (steps + 2) * num_reqs,
        (steps + 1) * num_reqs,
        0,
    ]
    assert all(
        len(tokens) == steps
        for tokens in outputs[1].scheduled_spec_decode_tokens.values()
    )
    assert len(samples) == 2
    assert kv_disabled == [True, False]
    lane.connector._launch.assert_called_once_with(num_reqs, (steps + 2) * num_reqs)
    assert lane.events == [
        ("request", (steps + 2) * num_reqs),
        ("write", 1),
        ("wait", 1),
        ("write", 0),
        ("write", 0),
    ]
