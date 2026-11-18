# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""INT8 planar KV with CPU offload: the connector must not force HND."""

from types import SimpleNamespace

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)
from vllm.v1.attention.backends import flash_attn
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.kv_cache_view import KVCacheView
from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache


def config(cache_dtype: str):
    return SimpleNamespace(cache_config=SimpleNamespace(cache_dtype=cache_dtype))


@pytest.mark.parametrize("cache_dtype", ["auto", "float16", "fp8", "fp8_e4m3"])
def test_existing_formats_keep_hnd(cache_dtype):
    assert OffloadingConnector.get_required_kvcache_layout(config(cache_dtype)) == "HND"


def test_int8_planar_leaves_default_layout():
    assert OffloadingConnector.get_required_kvcache_layout(config("int8")) is None


def reshape(monkeypatch, layout):
    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: layout)
    spec = FullAttentionSpec(
        block_size=816,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PLANAR,
    )
    raw = torch.zeros(3 * spec.page_size_bytes, dtype=torch.int8)
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        backend=FlashAttentionBackend,
        layer_names=["a"],
    )
    return _reshape_kv_cache([group], {"a": raw}, "int8", [16], {}, None)


def test_planar_int8_binds_under_connector_layout(monkeypatch):
    # Production shape: OffloadingConnector + INT8 KV. The default layout the
    # connector now leaves in place lets the planar binding succeed.
    layout = OffloadingConnector.get_required_kvcache_layout(config("int8")) or "NHD"
    out = reshape(monkeypatch, layout)
    assert out["a"].shape[0] == 3 * 816 // 16


def test_planar_int8_rejects_hnd(monkeypatch):
    # The startup failure this fixes: HND forced by the connector.
    with pytest.raises(ValueError, match="block-major NHD"):
        reshape(monkeypatch, "HND")


@pytest.mark.parametrize("kernel_block", [16, 64, 832])
@pytest.mark.parametrize("runner", ["v1", "v2"])
def test_planar_e4m3_binders_use_actual_kernel_block(monkeypatch, kernel_block, runner):
    codec = FP8_E4M3_PER_TOKEN_HEAD
    layout = OffloadingConnector.get_required_kvcache_layout(config(codec.name))
    assert layout is None
    monkeypatch.setattr(flash_attn, "get_kv_cache_layout", lambda: "NHD")
    spec = FullAttentionSpec(
        block_size=832,
        num_kv_heads=1,
        head_size=256,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.FP8_E4M3_PLANAR,
    )
    raw = torch.full((3 * spec.page_size_bytes,), 71, dtype=torch.int8)
    group = SimpleNamespace(
        kv_cache_group_id=0,
        kv_cache_spec=spec,
        backend=FlashAttentionBackend,
        layer_names=["a"],
    )
    if runner == "v2":
        out = _reshape_kv_cache(
            [group], {"a": raw}, codec.name, [kernel_block], {}, None
        )
    else:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner

        owner = SimpleNamespace(
            _kv_cache_spec_attn_group_iterator=lambda: iter([group]),
            runner_only_attn_layers=set(),
            cache_config=config(codec.name).cache_config,
        )
        out = GPUModelRunner._reshape_kv_cache_tensors(
            owner, {"a": raw}, [kernel_block]
        )
    view = KVCacheView.bind(out["a"], codec, 256)
    assert view.key.shape == (3 * 832 // kernel_block, kernel_block, 1, 256)
    assert view.key.dtype == torch.uint8
    assert view.value.storage_offset() == kernel_block * 256
    assert view.key_scale.data_ptr() - raw.data_ptr() == kernel_block * 512
    assert view.value_scale.data_ptr() - raw.data_ptr() == kernel_block * 516
    assert torch.all(raw == 71)
