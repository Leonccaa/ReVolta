# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run only in an approved GPU window, with the candidate native wheel."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("vllm.v1.attention.kv_cache_view", reason="v1 storage API absent")

from vllm import _custom_ops as ops
from vllm.v1.attention.kv_cache_view import KVCacheView, bind_planar_kv_cache
from vllm.v1.attention.kv_codecs import INT8
from vllm.v1.attention.ops.int8_kv import reference_int8_kv_write
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from vllm.v1.worker.utils import KVBlockZeroer

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def make_cache(device, heads=1, member=1):
    page = 848640 * heads
    raw = torch.full((3 * page,), 77, dtype=torch.int8, device=device)
    carrier = bind_planar_kv_cache(
        raw,
        num_blocks=3,
        scheduler_block_size=816,
        kernel_block_size=16,
        num_heads=heads,
        head_size=256,
        page_size_bytes=424320 * heads,
        member_index=member,
        member_count=2,
    )
    return raw, carrier, KVCacheView.bind(carrier, INT8, 256)


def write(k, v, view, slots):
    ignored = torch.tensor(float("nan"), device=k.device)
    ops.reshape_and_cache_flash(
        k, v, view.key, view.value, slots, "int8", ignored, ignored, **view.scale_kwargs
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("heads", [1, 3])
def test_native_writer_exact_bytes_and_row_batch_invariance(dtype, heads):
    torch.manual_seed(142)
    # Non-contiguous input head strides; graph padding adds two unwritten rows.
    k = torch.randn((10, heads * 2, 256), dtype=dtype, device="cuda")[:, ::2, :]
    v = k.flip(-1)
    k[0].zero_()
    v[0].zero_()
    k[1].fill_(0.5)
    k[1, :, 0] = 127.0
    k[1, :, 1] = 1.5
    k[1, :, 2] = 2.5
    k[2].fill_(65504)
    v[2].fill_(-65504)
    k[3, 0, 0] = float("inf")
    v[4, 0, 0] = float("nan")
    k[5].fill_(1e-6)
    v[5].fill_(-1e-6)
    slots = torch.tensor([15, 16, 815, 816, 1631, 1632, -1, 2448], device="cuda")
    raw, _, actual = make_cache("cuda", heads)
    oracle, _, expected = make_cache("cpu", heads)
    reference_int8_kv_write(k.cpu(), v.cpu(), expected, slots.cpu())
    write(k, v, actual, slots)
    assert torch.equal(raw.cpu(), oracle)  # all payload, scales, peers and padding
    raw.fill_(77)
    for i in range(slots.numel()):
        write(k[i : i + 1], v[i : i + 1], actual, slots[i : i + 1])
    assert torch.equal(raw.cpu(), oracle)


def test_native_writer_graph_replay_changes_slots_and_values():
    raw, _, view = make_cache("cuda")
    k = torch.randn(8, 1, 256, dtype=torch.float16, device="cuda")
    v = k.clone()
    slots = torch.tensor([15, 16, 815, 816, 1631, 1632, 17, 18], device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        write(k, v, view, slots)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        write(k, v, view, slots)
    for seed in [0, 1, 9]:
        torch.manual_seed(seed)
        k.normal_()
        v.normal_()
        slots.copy_(torch.randperm(2448, device="cuda")[:8])
        raw.fill_(77)
        oracle, _, expected = make_cache("cpu")
        reference_int8_kv_write(k.cpu(), v.cpu(), expected, slots.cpu())
        graph.replay()
        assert torch.equal(raw.cpu(), oracle)


def test_native_writer_zeroer_and_cpu_restore_full_owner():
    raw, carrier, view = make_cache("cuda")
    k = torch.randn(8, 1, 256, dtype=torch.float16, device="cuda")
    v = k.clone()
    slots = torch.tensor([15, 16, 815, 816, 1631, 1632, 17, 18], device="cuda")
    write(k, v, view, slots)
    saved = raw.cpu()

    class Backend:
        @staticmethod
        def get_kv_cache_block_dim(*a, **kw):
            return 0

    spec = FullAttentionSpec(
        block_size=816,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PLANAR,
    )
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        backend=Backend,
        layer_names=["member1"],
    )
    zeroer = KVBlockZeroer(torch.device("cuda"), pin_memory=False)
    zeroer.init_meta(
        [group], [16], "int8", set(), {"member1": SimpleNamespace(kv_cache=carrier)}
    )
    zeroer.zero_block_ids([1, 1])
    actual = raw.cpu().view(3, 848640)
    assert torch.count_nonzero(actual[1]) == 0
    assert torch.equal(actual[0], saved.view(3, 848640)[0])
    assert torch.equal(actual[2], saved.view(3, 848640)[2])
    raw.copy_(saved.to("cuda"))
    rebound = KVCacheView.bind(carrier, INT8, 256)
    assert rebound.key_scale.data_ptr() == view.key_scale.data_ptr()
    assert torch.equal(raw.cpu(), saved)
