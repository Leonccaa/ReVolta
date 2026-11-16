# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Physical v2 stride descriptors and byte-exact logical payload transport."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.kv_cache_view import (
    KVPhysicalPage,
    bind_attention_cache,
    pack_page_payload,
    padded_page_stride,
    scatter_page_payload,
)
from vllm.v1.kv_cache_interface import (
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
)
from vllm.v1.kv_offload.tiering.fs.revolta import _persistent_kv_tensors
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache
from vllm.v1.worker.utils import AttentionGroup

pytestmark = pytest.mark.cpu_test


class Backend:
    @staticmethod
    def get_builder_cls():
        return SimpleNamespace(uses_physical_block_table=True)

    @staticmethod
    def get_kv_cache_shape(n, block, heads, dim, **kwargs):
        return (n, block, heads, dim)

    @staticmethod
    def get_kv_cache_stride_order():
        return (0, 1, 2, 3)


def test_padding_minimum_and_rejection():
    assert padded_page_stride(53248, 2048, element_bytes=2) == 53248
    assert padded_page_stride(100352, 8192, element_bytes=2) == 106496
    # A slot alignment alone is insufficient for subdivided FP16 elements.
    assert padded_page_stride(96, 128, subdivisions=3, element_bytes=2) == 384
    assert padded_page_stride(96, 64, subdivisions=2, subpage_payload_bytes=80) == 192
    with pytest.raises(ValueError, match="No supported"):
        padded_page_stride(100352, 8192, max_stride_bytes=100352)
    with pytest.raises(ValueError, match="divide"):
        padded_page_stride(97, 64, subdivisions=2)
    with pytest.raises(ValueError, match="uniformly"):
        KVPhysicalPage(128).kernel_stride(3, 32)
    with pytest.raises(ValueError, match="payload"):
        KVPhysicalPage(128).kernel_stride(2, 80)
    with pytest.raises(ValueError, match="uniformly"):
        bind_attention_cache(
            torch.empty(384, dtype=torch.uint8),
            shape=(3, 2, 2, 1, 8),
            dtype=torch.float16,
            page_size_bytes=128,
            scheduler_block_size=6,
            kernel_block_size=2,
        )
    with pytest.raises(ValueError, match="element aligned"):
        bind_attention_cache(
            torch.empty(130, dtype=torch.uint8),
            shape=(2, 2, 3, 1, 4),
            dtype=torch.float16,
            page_size_bytes=130,
            scheduler_block_size=6,
            kernel_block_size=3,
        )


def test_49_padded_pages_and_ring_use_explicit_description():
    spec = MLAAttentionSpec(
        block_size=1568,
        compress_ratio=4,
        num_kv_heads=1,
        head_size=128,
        dtype=torch.float16,
    )
    stride, blocks = 106496, 49
    assert spec.page_size_bytes == 100352
    assert stride * blocks // spec.page_size_bytes == 52  # Old divisor is wrong.
    desc = KVCacheTensor(
        stride * blocks, ["index", "ring"], num_blocks=blocks, page_stride_bytes=stride
    )
    raw = torch.zeros(desc.size, dtype=torch.int8)
    ring = MambaSpec(
        block_size=16,
        shapes=((2, 3),),
        dtypes=(torch.float16,),
        page_size_padded=spec.page_size_bytes,
    )
    groups = [
        AttentionGroup(Backend, ["index"], spec, 0),
        AttentionGroup(Backend, ["ring"], ring, 1),
    ]
    # Avoid hybrid stride adaptation in this isolated binder test: the index
    # and the ring branches each bind exactly the same physical description.
    results = []
    for group in groups:
        result = _reshape_kv_cache(
            [group],
            {name: raw for name in desc.shared_by},
            "auto",
            [16, 16],
            {},
            tensor_descriptions={name: desc for name in desc.shared_by},
        )
        view = result[group.layer_names[0]]
        results.append(view[0] if isinstance(view, list) else view)
    for view in results:
        assert view.shape[0] == blocks
        assert view.stride(0) * view.element_size() == stride
        assert view.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    assert spec.page_size_padded is None
    with pytest.raises(ValueError, match="description"):
        _reshape_kv_cache(
            [groups[0]],
            {"index": raw},
            "auto",
            [16],
            {},
            tensor_descriptions={"index": replace(desc, num_blocks=None)},
        )


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16])
def test_interior_padding_roundtrip_preserves_every_logical_byte(dtype):
    item = dtype.itemsize
    raw = torch.full((3 * 128,), 0xA5, dtype=torch.uint8)
    view = bind_attention_cache(
        raw,
        shape=(6, 2, 3, 1, 8 // item),
        dtype=dtype,
        page_size_bytes=128,
        scheduler_block_size=6,
        kernel_block_size=3,
    )
    # P=96, S=128, subdivisions=2: [0,48) and [64,112) in each page.
    payload = torch.arange(96, dtype=torch.uint8)
    # Include arbitrary float bit patterns (NaN/sign bits), not numeric casts.
    if item == 2:
        payload[1::2] = 0xFF
        frame = torch.empty(97, dtype=torch.uint8)
        frame[1:] = payload
        payload = frame[1:]  # Unaligned framing must not change logical bytes.
    scatter_page_payload(payload, view, 1, 2)
    expected = torch.full_like(raw, 0xA5)
    expected[128:176] = payload[:48]
    expected[192:240] = payload[48:]
    assert torch.equal(raw, expected)
    packed = pack_page_payload(view, 1, 2)
    assert torch.equal(packed, payload)
    assert not torch.equal(raw[128:224], packed)
    packed.zero_()
    assert torch.equal(raw, expected)  # Pack does not alias the source.
    scatter_page_payload(payload, view, 2, 2)
    assert torch.equal(pack_page_payload(view, 2, 2), payload)
    assert torch.equal(pack_page_payload(view, 1, 2), payload)


def test_payload_follows_inner_strides_and_ring_offsets():
    raw = torch.full((256,), 0x5A, dtype=torch.uint8)
    view = raw.as_strided((4, 2, 3), (64, 16, 2), storage_offset=8)
    payload = torch.arange(12, dtype=torch.uint8)
    scatter_page_payload(payload, view, 1, 2)
    assert torch.equal(pack_page_payload(view, 1, 2), payload)
    permuted = view.movedim(0, 1)
    assert torch.equal(pack_page_payload(permuted, 1, 2, block_dim=1), payload)
    touched = {
        8 + p * 64 + i * 16 + j * 2 for p in (2, 3) for i in range(2) for j in range(3)
    }
    assert all(int(raw[i]) == 0x5A for i in range(256) if i not in touched)
    scatter_page_payload(payload, permuted, 0, 2, block_dim=1)
    assert torch.equal(pack_page_payload(view, 0, 2), payload)
    assert torch.equal(pack_page_payload(view, 1, 2), payload)
    with pytest.raises(ValueError):
        scatter_page_payload(payload[:3], view, 1, 2)


def test_ordinary_persistent_identity_omits_both_absent_fields():
    tensor = KVCacheTensor(128, ["a"])
    cfg = SimpleNamespace(kv_cache_tensors=[tensor], block_segments=None)
    [(type_name, fields)] = _persistent_kv_tensors(cfg)
    assert type_name == "vllm.v1.kv_cache_interface.KVCacheTensor"
    assert fields == (("packed_members", None), ("shared_by", ("a",)), ("size", 128))
    cfg.kv_cache_tensors = [replace(tensor, num_blocks=2, page_stride_bytes=64)]
    assert _persistent_kv_tensors(cfg) != [(type_name, fields)]
    # Existing split GPU layouts already serialized num_blocks, but had no
    # page_stride_bytes field. Keep that namespace stable as well.
    cfg.block_segments = object()
    cfg.kv_cache_tensors = [replace(tensor, num_blocks=2)]
    [(split_name, split_fields)] = _persistent_kv_tensors(cfg)
    assert split_name == type_name
    assert dict(split_fields) == {**dict(fields), "num_blocks": 2}
