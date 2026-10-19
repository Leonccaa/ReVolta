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
