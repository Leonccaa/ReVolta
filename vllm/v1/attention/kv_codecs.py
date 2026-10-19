# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordinary KV formats, adapted from 1CatAI/1Cat-vLLM #1179.

RN32 oracle/capability additions follow #1194 (snapshot b4b6c47992).
Grouped MLA, history and indexer representations are intentionally excluded.

These descriptors describe storage, independently of weight quantization and
reader arithmetic. Existing FP16/FP8 readers retain their own implementations;
reader unification is a separate change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True, eq=False)
class KVCodec:
    """One KV-cache storage format."""

    # Canonical name, also the ``kv_cache_dtype`` string native operators take.
    name: str
    # ``--kv-cache-dtype`` spellings that resolve to this codec.
    aliases: tuple[str, ...]
    # Tensor dtype of the paged K/V cache.
    storage_dtype: torch.dtype
    # Element dtype the stored bytes are reinterpreted as, if they are packed.
    element_dtype: torch.dtype | None = None
    # Whether reading the cache needs scales.
    quantized: bool = False
    native_id: int | None = None
    encoding_version: int = 1
    scale_granularity: Literal["none", "tensor", "token_head"] = "none"
    scale_dtype: torch.dtype | None = None
    rounding: Literal["none", "nearest_even", "truncate"] = "none"
    layout: Literal["dense", "inline_scale", "planar_scales"] = "dense"
    representation: str = "ordinary"
    scale_encoding: Literal["none", "fp32", "ue8m0"] = "none"

    @property
    def cache_identity(self) -> tuple[object, ...]:
        """Versioned storage identity; equal byte widths are not compatible."""
        return (self.name, self.encoding_version, self.layout)

    @property
    def per_token_head(self) -> bool:
        return self.scale_granularity == "token_head"

    def stores(self, key_cache: torch.Tensor, value_cache: torch.Tensor) -> bool:
        """Whether both cache tensors use this codec's storage dtype."""
        return (
            key_cache.dtype == self.storage_dtype
            and value_cache.dtype == self.storage_dtype
        )

    def supported_by(self, operation) -> bool:
        """Query installed bindings; legacy callables support tensor-scale KV only."""
        if operation is None:
            return False
        if self.representation not in getattr(
            operation, "kv_cache_representations", ("ordinary",)
        ):
            return False
        formats = getattr(operation, "kv_cache_codecs", None)
        if formats is None:
            return self in (FP16, FP8_E4M3, FP8_E5M2)
        return self.name in formats

    @property
    def fp32_attention_partials(self) -> bool:
        """Accumulation contract shared with native KVCodecTraits."""
        return self.per_token_head or self.element_dtype == torch.float8_e4m3fn

    def dequantize(
        self,
        cache: torch.Tensor,
        scale: float | torch.Tensor,
        out_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Reference dequantization of contiguous cache bytes."""
        if not self.quantized:
            return cache
        assert self.element_dtype is not None
        if self.per_token_head:
            if not isinstance(scale, torch.Tensor) or scale.shape != cache.shape[:-1]:
                raise ValueError("Per-token-head decoding needs one scale per row")
            # Keep the scale multiplication in FP32. Casting a scale or an
            # unscaled INT8 product to FP16 can overflow for finite inputs.
            decoded = cache.view(self.element_dtype).float() * scale[..., None]
            if self.layout == "planar_scales":
                # Even an FP32 scale rounded from FLT_MAX/127 can reconstruct
                # just above FLT_MAX. Saturate finite rows at the destination
                # boundary; preserve the NaN scale used for invalid rows.
                limit = torch.finfo(out_dtype).max
                decoded = torch.where(
                    torch.isfinite(scale[..., None]),
                    decoded.clamp(-limit, limit),
                    decoded,
                )
            return decoded.to(out_dtype)
        return cache.view(self.element_dtype).to(out_dtype) * scale

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Reference dynamic INT8 encoder; not used by the execution hot path.

        A non-finite row receives a NaN scale, propagating invalid inputs to
        its readers instead of silently converting them into finite values.
        Legacy truncation retains its original multiplication order.
        """
        if self.element_dtype != torch.int8 or not self.per_token_head:
            raise ValueError(f"{self.name} has no dynamic INT8 reference encoder")
        x = values.float()
        absmax = x.abs().amax(dim=-1)
        if self.rounding == "nearest_even":
            # Match CUDA/Triton RN FP32 division. Torch may strength-reduce a
            # scalar division to multiplication by a rounded reciprocal.
            scale = (absmax.double() / 127.0).float().clamp_min(1e-6)
            finite = torch.isfinite(x).all(dim=-1)
            scale = torch.where(finite, scale, float("nan"))
            q = (x.double() / scale[..., None].double()).float().round()
            q = torch.where(finite[..., None], q, 0.0)
        else:
            scale = (absmax / 127.0).clamp_min(1e-6)
            q = (x * scale.reciprocal()[..., None]).trunc()
        return q.clamp(-128, 127).to(torch.int8).reshape_as(values), scale

    def __repr__(self) -> str:
        return f"KVCodec({self.name})"


# The unquantized 16-bit cache. Native operators name it "auto".
FP16 = KVCodec("auto", ("auto", "float16"), torch.float16, native_id=0)
BF16 = KVCodec("bfloat16", ("bfloat16",), torch.bfloat16)
FP8_E4M3 = KVCodec(
    "fp8_e4m3",
    ("fp8", "fp8_e4m3"),
    torch.uint8,
    torch.float8_e4m3fn,
    quantized=True,
    native_id=1,
    scale_granularity="tensor",
    scale_dtype=torch.float32,
    scale_encoding="fp32",
    rounding="nearest_even",
)
FP8_E5M2 = KVCodec(
    "fp8_e5m2",
    ("fp8_e5m2",),
    torch.uint8,
    torch.float8_e5m2,
    quantized=True,
    native_id=2,
    scale_granularity="tensor",
    scale_dtype=torch.float32,
    scale_encoding="fp32",
    rounding="nearest_even",
)
INT8 = KVCodec(
    "int8",
    ("int8",),
    torch.int8,
    torch.int8,
    quantized=True,
    native_id=3,
    scale_granularity="token_head",
    scale_dtype=torch.float32,
    scale_encoding="fp32",
    rounding="nearest_even",
    layout="planar_scales",
)
INT8_PER_TOKEN_HEAD = KVCodec(
    "int8_per_token_head",
    ("int8_per_token_head",),
    torch.int8,
    torch.int8,
    quantized=True,
    native_id=3,
    scale_granularity="token_head",
    scale_dtype=torch.float32,
    scale_encoding="fp32",
    rounding="truncate",
    layout="inline_scale",
)
FP8_PER_TOKEN_HEAD = KVCodec(
    "fp8_per_token_head",
    ("fp8_per_token_head",),
    torch.uint8,
    torch.float8_e4m3fn,
    quantized=True,
    scale_granularity="token_head",
    scale_dtype=torch.float32,
    scale_encoding="fp32",
    rounding="nearest_even",
    layout="inline_scale",
)

KV_CODECS: tuple[KVCodec, ...] = (
    FP16,
    BF16,
    FP8_E4M3,
    FP8_E5M2,
    INT8,
    INT8_PER_TOKEN_HEAD,
    FP8_PER_TOKEN_HEAD,
)
_BY_ALIAS = {alias: codec for codec in KV_CODECS for alias in codec.aliases}


def resolve_kv_codec(kv_cache_dtype: str | None) -> KVCodec | None:
    return None if kv_cache_dtype is None else _BY_ALIAS.get(kv_cache_dtype)


def canonical_kv_cache_dtype(kv_cache_dtype: str) -> str:
    codec = resolve_kv_codec(kv_cache_dtype)
    return kv_cache_dtype if codec is None else codec.name
