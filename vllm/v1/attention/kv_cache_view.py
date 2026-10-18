# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One byte-layout contract for ordinary KV owners and their borrowed views.

Adapted from 1CatAI/1Cat-vLLM #1179. No allocator or lifetime is owned here.
A scheduler page can contain several interleaved layer members. A kernel page
is the smaller unit readers address; padding must divide uniformly over it.
"""

from dataclasses import dataclass
from math import prod

import torch

from vllm.v1.attention.kv_codecs import KVCodec

INT8_KERNEL_TOKENS = 16


def align_up(size: int, alignment: int = 16) -> int:
    if size < 0 or alignment <= 0:
        raise ValueError("Invalid byte size or alignment")
    return (size + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class KVPageLayout:
    block_size: int
    num_heads: int
    head_size: int
    head_size_v: int
    element_bytes: int = 1
    planar_scales: bool = True

    def __post_init__(self) -> None:
        if (
            min(
                self.block_size,
                self.num_heads,
                self.head_size,
                self.head_size_v,
                self.element_bytes,
            )
            <= 0
        ):
            raise ValueError("KV page dimensions must be positive")

    def _region(self, size: int) -> int:
        return align_up(size) if self.planar_scales else size

    @property
    def value_offset(self) -> int:
        return self._region(
            self.block_size * self.num_heads * self.head_size * self.element_bytes
        )

    @property
    def key_scale_offset(self) -> int:
        return self.value_offset + self._region(
            self.block_size * self.num_heads * self.head_size_v * self.element_bytes
        )

    @property
    def value_scale_offset(self) -> int:
        return self.key_scale_offset + (
            align_up(self.block_size * self.num_heads * 4) if self.planar_scales else 0
        )

    @property
    def page_size_bytes(self) -> int:
        return self.value_scale_offset + (
            align_up(self.block_size * self.num_heads * 4) if self.planar_scales else 0
        )

    def scheduler_page_stride(self, padded_bytes: int | None) -> int:
        if padded_bytes is None:
            return self.page_size_bytes
        if padded_bytes < self.page_size_bytes:
            raise ValueError("Padded KV page is smaller than its regions")
        if self.planar_scales and padded_bytes % 16:
            raise ValueError("Planar pages require 16-byte alignment")
        return padded_bytes


@dataclass(frozen=True)
class KVPhysicalPage:
    """An opaque owner page; recurrent/side owners use this contract too."""

    member_page_bytes: int
    member_count: int = 1
    member_index: int = 0

    def __post_init__(self) -> None:
        if (
            self.member_page_bytes <= 0
            or not 0 <= self.member_index < self.member_count
        ):
            raise ValueError("Invalid physical KV page")

    @property
    def page_size_bytes(self) -> int:
        return self.member_page_bytes * self.member_count

    def io_frame_bytes(self, alignment: int = 4096, header_bytes: int = 0) -> int:
        """Disk framing budget only; never use this as the GPU page stride."""
        if header_bytes < 0:
            raise ValueError("Invalid frame header length")
        return align_up(self.page_size_bytes + header_bytes, alignment)

    def raw_pages(self, tensor: torch.Tensor, num_blocks: int) -> torch.Tensor:
        storage = tensor.untyped_storage()
        if storage.nbytes() != num_blocks * self.page_size_bytes:
            raise ValueError("Physical KV allocation does not match its page budget")
        return torch.empty(0, dtype=torch.int8, device=tensor.device).set_(
            storage, 0, (num_blocks, self.page_size_bytes), (self.page_size_bytes, 1)
        )

    def transfer_size(
        self,
        tensor: torch.Tensor,
        num_blocks: int,
        real_bytes: int,
        *,
        planar: bool,
        padded: bool = False,
    ) -> int:
        # Uniformly subdivided padding is inside a scheduler page. A contiguous
        # real_bytes prefix would truncate the last payload/scale, including a
        # dense member in a mixed INT8 plan. Legacy tail padding stays unchanged.
        interior_padding = (
            padded
            and self.member_page_bytes > real_bytes
            and tensor.ndim == 5
            and tensor.shape[1] == 2
            and tensor.shape[0] > num_blocks
        )
        if self.member_count > 1 or planar or interior_padding:
            return self.page_size_bytes
        return real_bytes

    def kernel_stride(self, subdivisions: int, minimum: int) -> int:
        if subdivisions <= 0 or self.member_page_bytes % subdivisions:
            raise ValueError("Physical padding must divide uniformly over kernel pages")
        stride = self.member_page_bytes // subdivisions
        if stride < minimum:
            raise ValueError("Kernel page cannot hold its payload/scales")
        return stride

    @staticmethod
    def zeroer_segment(
        tensor: torch.Tensor, block_dim: int, subdivisions: int
    ) -> tuple[int, int]:
        """W6 semantics: member 1 must resolve to member 0's physical base."""
        stride = tensor.stride(block_dim) * tensor.element_size()
        offset = tensor.storage_offset() * tensor.element_size() % stride
        if stride % 4:
            raise ValueError("KV zeroer needs 4-byte page alignment")
        return tensor.data_ptr() - offset, stride * subdivisions


def physical_owner_page_size(member_page_sizes: list[int]) -> int:
    if not member_page_sizes or len(set(member_page_sizes)) != 1:
        raise ValueError("Packed KV members must have equal physical page sizes")
    return KVPhysicalPage(member_page_sizes[0], len(member_page_sizes)).page_size_bytes


def packed_kv_members(kv_cache_config) -> dict[str, tuple[int, int]]:
    return {
        name: (index, len(tensor.packed_members))
        for tensor in kv_cache_config.kv_cache_tensors
        if tensor.packed_members
        for index, name in enumerate(tensor.packed_members)
    }


def bind_attention_cache(
    raw: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    page_size_bytes: int,
    scheduler_block_size: int,
    kernel_block_size: int,
    block_dim: int = 0,
    member_index: int = 0,
    member_count: int = 1,
) -> torch.Tensor:
    """Dense carrier, preserving legacy FP16/E4M3 strides and offsets exactly."""
    owner = KVPhysicalPage(page_size_bytes, member_count, member_index)
    item = torch.empty((), dtype=dtype, device="meta").element_size()
    if scheduler_block_size % kernel_block_size:
        raise ValueError("Kernel blocks must divide logical storage blocks")
    subdivisions = scheduler_block_size // kernel_block_size
    strides = list(torch.empty(shape, device="meta").stride())
    minimum = prod(shape[:block_dim] + shape[block_dim + 1 :]) * item
    if member_count > 1 and block_dim != 0:
        raise ValueError("Packed pages require a leading physical block dimension")
    kernel_bytes = owner.kernel_stride(subdivisions, minimum)
    if kernel_bytes % item:
        raise ValueError("KV page stride is not element aligned")
    if kernel_bytes != minimum and block_dim != 0:
        raise ValueError("Padded pages require a leading physical block dimension")
    if member_count > 1 or kernel_bytes != minimum:
        strides[block_dim] = kernel_bytes * member_count // item
    return raw.view(dtype).as_strided(
        shape,
        tuple(strides),
        raw.storage_offset() * raw.element_size() // item
        + kernel_bytes * member_index // item,
    )


def bind_planar_kv_cache(
    raw: torch.Tensor,
    *,
    num_blocks: int,
    scheduler_block_size: int,
    kernel_block_size: int,
    num_heads: int,
    head_size: int,
    page_size_bytes: int,
    member_index: int = 0,
    member_count: int = 1,
) -> torch.Tensor:
    if (
        kernel_block_size != INT8_KERNEL_TOKENS
        or scheduler_block_size % kernel_block_size
    ):
        raise ValueError("INT8 v1 uses 16-token kernel pages dividing scheduler blocks")
    layout = KVPageLayout(kernel_block_size, num_heads, head_size, head_size)
    owner = KVPhysicalPage(page_size_bytes, member_count, member_index)
    subdivisions = scheduler_block_size // kernel_block_size
    kernel_bytes = owner.kernel_stride(subdivisions, layout.page_size_bytes)
    if kernel_bytes % 16:
        raise ValueError("Physical padding must preserve 16-byte kernel alignment")
    # Capacity carrier only. Its final width is not a mathematical head size.
    row = head_size + 4
    return raw.view(torch.int8).as_strided(
        (num_blocks * subdivisions, 2, kernel_block_size, num_heads, row),
        (
            kernel_bytes * member_count,
            kernel_block_size * num_heads * row,
            num_heads * row,
            row,
            1,
        ),
        raw.storage_offset() * raw.element_size() + kernel_bytes * member_index,
    )


@dataclass(frozen=True)
class KVPageMapping:
    owner_page_span: int
    subpage_tokens: int
    owner_tokens: int

    @classmethod
    def bind_payload(cls, key, value, subpage_tokens: int) -> "KVPageMapping":
        if (
            key.ndim != 4
            or key.shape != value.shape
            or key.dtype != value.dtype
            or key.device != value.device
            or subpage_tokens <= 0
            or key.shape[1] % subpage_tokens
        ):
            raise ValueError("Invalid KV subpage geometry")
        for payload in (key, value):
            if (
                payload.stride(3) != 1
                or payload.stride(2) != payload.shape[3]
                or payload.stride(1) != payload.shape[2] * payload.shape[3]
                or payload.stride(0) * payload.element_size() % 16
                or payload.stride(0)
                < payload.shape[1] * payload.shape[2] * payload.shape[3]
            ):
                raise ValueError("KV subpages need aligned contiguous payload rows")
        if key.stride(0) != value.stride(0):
            raise ValueError("K/V owner strides disagree")
        return cls(key.shape[1] // subpage_tokens, subpage_tokens, key.shape[1])


@dataclass(frozen=True)
class KVCacheView:
    codec: KVCodec
    key: torch.Tensor
    value: torch.Tensor
    key_scale: torch.Tensor | None
    value_scale: torch.Tensor | None
    page_stride_bytes: int
    page_size_bytes: int

    @property
    def scale_kwargs(self) -> dict[str, torch.Tensor]:
        if not self.codec.per_token_head:
            return {}
        if self.key_scale is None or self.value_scale is None:
            raise ValueError("Missing KV scale planes")
        return {"k_scale_cache": self.key_scale, "v_scale_cache": self.value_scale}

    def subpage_mapping(self, page_tokens: int) -> KVPageMapping:
        return KVPageMapping.bind_payload(self.key, self.value, page_tokens)

    @classmethod
    def bind(cls, cache: torch.Tensor, codec: KVCodec, head_size: int) -> "KVCacheView":
        if cache.ndim != 5 or cache.shape[1] != 2 or cache.stride(-1) != 1:
            raise ValueError("Expected [page, 2, token, head, capacity] KV carrier")
        if cache.dtype != codec.storage_dtype:
            raise ValueError("KV carrier dtype disagrees with codec")
        n, _, b, h, stored_dim = cache.shape
        item = cache.element_size()
        stride = cache.stride(0) * item
        base = cache.storage_offset() * item

        def scale_view(offset, strides):
            if any(x % 4 for x in (offset, *strides)):
                raise ValueError("FP32 KV scales require aligned addresses")
            return torch.empty(0, dtype=torch.float32, device=cache.device).set_(
                cache.untyped_storage(),
                offset // 4,
                (n, b, h),
                tuple(x // 4 for x in strides),
            )

        if codec.layout == "planar_scales":
            if b != INT8_KERNEL_TOKENS:
                raise ValueError("INT8 v1 requires a 16-token kernel-page carrier")
            layout = KVPageLayout(b, h, head_size, head_size, item)
            if (
                base % 16
                or stride % 16
                or stride < layout.page_size_bytes
                or stored_dim < head_size + 4
                or (
                    n
                    and base + (n - 1) * stride + layout.page_size_bytes
                    > cache.untyped_storage().nbytes()
                )
            ):
                raise ValueError("Carrier does not reserve aligned KV regions")
            strides = (cache.stride(0), h * head_size, head_size, 1)
            key = cache.as_strided((n, b, h, head_size), strides)
            value = cache.as_strided(
                (n, b, h, head_size), strides, (base + layout.value_offset) // item
            )
            scale_strides = (stride, h * 4, 4)
            return cls(
                codec,
                key,
                value,
                scale_view(base + layout.key_scale_offset, scale_strides),
                scale_view(base + layout.value_scale_offset, scale_strides),
                stride,
                layout.page_size_bytes,
            )
        key, value = cache.unbind(1)
        if codec.layout == "inline_scale":
            if stored_dim != head_size + 4 // item:
                raise ValueError("Inline KV carrier must reserve FP32 scales")
            scale_strides = tuple(x * item for x in key.stride()[:3])
            return cls(
                codec,
                key[..., :head_size],
                value[..., :head_size],
                scale_view(base + head_size * item, scale_strides),
                scale_view(
                    value.storage_offset() * item + head_size * item, scale_strides
                ),
                stride,
                2 * b * h * stored_dim * item,
            )
        if stored_dim != head_size:
            raise ValueError("Dense KV head dimension disagrees with codec")
        return cls(codec, key, value, None, None, stride, 2 * b * h * head_size * item)


def kv_storage_metadata(kv_cache_config, configured_dtype: str) -> dict:
    """Versioned persistent identity for C's FileMapper.persistent_layout.

    Logical prefix hashes stay unchanged inside an engine. A persistent cache
    additionally needs its model/revision/YaRN/config identity (owned by tiering).
    This describes every physical tensor and every member, including side/state
    owners, without retaining a tensor or a storage pointer.
    """
    from vllm.v1.attention.kv_codecs import resolve_kv_codec

    owners = {}
    for group in kv_cache_config.kv_cache_groups:
        group_spec = group.kv_cache_spec
        specs = getattr(group_spec, "kv_cache_specs", {})
        for name in group.layer_names:
            spec = specs.get(name, group_spec)
            entry = {
                "spec": type(spec).__name__,
                "block_size": spec.block_size,
                "storage_block_size": spec.storage_block_size,
                "dcp_sharded": spec.dcp_sharded,
                "page_bytes": spec.page_size_bytes,
            }
            if hasattr(spec, "layout_cache_dtype"):
                codec = resolve_kv_codec(spec.layout_cache_dtype(configured_dtype))
                entry["codec"] = list(codec.cache_identity) if codec else None
                entry["dtype"] = str(spec.dtype)
                entry["heads"] = spec.num_kv_heads
                entry["head_size"] = spec.head_size
                entry["head_size_v"] = getattr(spec, "head_size_v", spec.head_size)
                if codec is not None and codec.layout == "planar_scales":
                    layout = KVPageLayout(
                        INT8_KERNEL_TOKENS,
                        spec.num_kv_heads,
                        spec.head_size,
                        entry["head_size_v"],
                    )
                    entry["kernel_page"] = {
                        "tokens": INT8_KERNEL_TOKENS,
                        "alignment": 16,
                        "region_offsets": [
                            0,
                            layout.value_offset,
                            layout.key_scale_offset,
                            layout.value_scale_offset,
                        ],
                        "region_bytes": layout.page_size_bytes,
                        "member_stride": spec.page_size_bytes
                        // (spec.storage_block_size // INT8_KERNEL_TOKENS),
                    }
            else:
                entry["shapes"] = list(spec.shapes)
                entry["dtypes"] = [str(dtype) for dtype in spec.dtypes]
            owners[name] = entry
    return {
        "version": "ordinary-kv-owner-v1",
        "owners": owners,
        "tensors": [
            {
                "bytes_per_block": t.size // kv_cache_config.num_blocks,
                "shared_by": list(t.shared_by),
                "packed_members": list(t.packed_members or []),
            }
            for t in kv_cache_config.kv_cache_tensors
        ],
    }


def has_planar_kv_cache(kv_cache_config) -> bool:
    """Target and draft may choose their storage encoding independently."""
    from vllm.v1.kv_cache_interface import AttentionSpec, KVQuantMode

    return any(
        isinstance(spec, AttentionSpec)
        and spec.kv_quant_mode == KVQuantMode.INT8_PLANAR
        for group in kv_cache_config.kv_cache_groups
        for spec in getattr(
            group.kv_cache_spec, "kv_cache_specs", {"owner": group.kv_cache_spec}
        ).values()
    )


def split_dense_kv_cache(
    kv_cache: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | list[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    if isinstance(kv_cache, (list, tuple)):
        if len(kv_cache) != 2:
            raise ValueError(
                f"Unexpected KV cache tuple/list length {len(kv_cache)}; expected 2"
            )
        return kv_cache[0], kv_cache[1]

    if kv_cache.ndim < 2:
        raise ValueError(
            f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
            "expected dimension 2 at axis 0 or 1"
        )

    # Standard vLLM paged KV layout is [num_blocks, 2, block_size, heads, dim].
    # Prefer axis 1 so num_blocks == 2 does not get mistaken for K/V.
    if kv_cache.shape[1] == 2:
        return kv_cache.unbind(1)
    if kv_cache.shape[0] == 2:
        return kv_cache.unbind(0)

    raise ValueError(
        f"Unexpected KV cache shape {tuple(kv_cache.shape)}; "
        "expected dimension 2 at axis 0 or 1"
    )
