# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only lossless metadata/address oracle, independent of vLLM imports.

Run with .venv/bin/python -m pytest --confcutdir=tests/quantization
-o addopts='' tests/quantization/test_sm70_awq_int4_zero_cpu.py
Optional real checkpoints: W3_CHECKPOINT=/read/only/checkpoint (safetensors).
No CUDA API is called, including device discovery. Peak working set < 2 GiB.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "awq_int4_zero", ROOT / "vllm/model_executor/layers/quantization/awq_int4_zero.py"
)
PACK = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PACK)
ORDER = (0, 4, 1, 5, 2, 6, 3, 7)


def old_metadata(scales, zeros):
    bits = scales.astype("<f2").view("<u2")
    return np.stack((bits & 255, bits >> 8, zeros), -1).astype("u1")


def unpack(packed):
    """Independent inverse, not a reshape of the Python pack implementation."""
    shape = (*packed.shape[:-2], packed.shape[-2] * 2, 3)
    indices = np.arange(np.prod(shape[:-1])).reshape(shape[:-1])
    flat = packed.reshape(-1)
    pair, odd = indices // 2 * 5, indices % 2
    return np.stack(
        (
            flat[pair + odd * 2],
            flat[pair + odd * 2 + 1],
            (flat[pair + 4] >> (odd * 4)) & 15,
        ),
        -1,
    ).astype("u1")


def stats_words(records):
    bits = records[..., 0].astype("u4") | (records[..., 1].astype("u4") << 8)
    scale = bits.astype("<u2").view("<f2")
    # __hneg(z), __hmul(-z,s), not -(z*s): retain the sign of zero.
    with np.errstate(over="ignore", invalid="ignore"):
        bias = -records[..., 2].astype("f2").astype("f8") * scale.astype("f8")
        bias = bias.astype("<f2")
    return bits | (bias.view("<u2").astype("u4") << 16)


def read_kernel(packed, experts, groups, tiles, cols, n):
    """Literal new CUDA pointer arithmetic, including expert byte stride."""
    g = packed.shape[1]
    expert_stride = g * n * 5 // 2
    index = groups * n + tiles * 32 + cols
    base = experts * expert_stride + (index >> 1) * 5
    odd = index & 1
    flat = packed.reshape(-1)
    return np.stack(
        (
            flat[base + odd * 2],
            flat[base + odd * 2 + 1],
            (flat[base + 4] >> (odd * 4)) & 15,
        ),
        -1,
    ).astype("u1")


def read_old_cooperative(records):
    """Literal 24-word/32-lane reader for every contiguous N32 tile."""
    words = records.reshape(-1, 96).view("<u4").astype("u8")
    byte = np.arange(32) * 3
    shift = ((byte & 3) * 8).astype("u8")
    low = words[:, byte >> 2] >> shift
    high = words[:, (byte >> 2) + (shift > 8)]
    low |= np.where(shift > 8, high << (32 - shift), 0)
    return (low & 0xFFFFFF).reshape(records.shape[:-1])


def check_bank(records):
    packed = PACK.pack_int4_zero(torch.from_numpy(records)).numpy()
    np.testing.assert_array_equal(unpack(packed), records)
    e, g, n, _ = records.shape
    assert packed.nbytes == e * g * n * 5 // 2
    # All lanes, tiles, groups and experts, including final byte of final expert.
    lane = np.arange(32)
    col = ((lane >> 2) & 3) * 8 + (lane & 3) + ((lane & 16) != 0) * 4
    expert = np.arange(e)[:, None, None, None]
    group = np.arange(g)[None, :, None, None]
    tile = np.arange(n // 32)[None, None, :, None]
    actual = read_kernel(packed, expert, group, tile, col, n)
    expected = records.reshape(e, g, n // 32, 32, 3)[..., col, :]
    np.testing.assert_array_equal(actual, expected)
    # Uint8 scalar and cooperative readers return exactly the same 24 bits.
    raw = records[..., 0].astype("u4") | (records[..., 1].astype("u4") << 8)
    raw |= records[..., 2].astype("u4") << 16
    np.testing.assert_array_equal(read_old_cooperative(records), raw)
    np.testing.assert_array_equal(stats_words(actual), stats_words(expected))
    return packed


@pytest.mark.parametrize("e,k,n", [(3, 64, 32), (512, 2560, 320), (512, 160, 2560)])
def test_random_full_banks(e, k, n):
    rng = np.random.default_rng(k + n)
    scales = rng.integers(0, 0x7C00, size=(e, k // 32, n), dtype="u2").view("f2")
    zeros = rng.integers(0, 16, size=scales.shape, dtype="u1")
    check_bank(old_metadata(scales, zeros))


@pytest.mark.parametrize(
    "k,n,split", [(2560, 320, 16), (2560, 320, 8), (2560, 320, 4), (160, 2560, 1)]
)
def test_qpn_m1_and_grouped_k16_loops(k, n, split):
    rng = np.random.default_rng(k + n + split)
    records = old_metadata(
        np.ones((2, k // 32, n), dtype="f2"),
        rng.integers(0, 16, (2, k // 32, n), dtype="u1"),
    )
    packed = PACK.pack_int4_zero(torch.from_numpy(records)).numpy()
    # M1 split=16 loops warp*10..warp*10+10; grouped pairs steps of two.
    # Both reuse metadata for two K16 halves. W2 loops all ten K16 steps.
    steps = k // 16 // split
    for warp in range(split):
        for s in range(steps):
            group = warp * steps + s
            columns = np.arange(n)
            actual = read_kernel(packed, 1, group // 2, columns // 32, columns % 32, n)
            np.testing.assert_array_equal(actual, records[1, group // 2])


def test_selected_expert_expansion_and_pointer_tags():
    rng = np.random.default_rng(71)
    records = old_metadata(
        np.ones((512, 5, 2560), dtype="f2"),
        rng.integers(0, 16, (512, 5, 2560), dtype="u1"),
    )
    packed = PACK.pack_int4_zero(torch.from_numpy(records)).numpy()
    # Nonmonotonic indices, repeated expert, trailing empty-group id, extremes.
    indices = np.array([511, 0, 257, 511, 1, 256, 0])
    expanded = unpack(packed[indices])
    np.testing.assert_array_equal(expanded, records[indices])
    base = 0x100000
    stride = 5 * 2560 * 5 // 2
    pointers = (base + indices * stride) | 3
    np.testing.assert_array_equal(pointers & ~3, base + indices * stride)
    np.testing.assert_array_equal(pointers & 3, np.full(len(indices), 3))
    # Restored 3B expert rows retain a 4-byte-aligned base for the old reader.
    assert (records[0].nbytes % 4) == 0


def test_finite_half_scales_all_zeros_all_codes():
    # All 63,488 finite half bit patterns, all 16 zero points and codes.
    # float64 computes products/sums exactly before the native FP16 rounding.
    bits = np.arange(65536, dtype="u2")
    scales = bits[(bits & 0x7C00) != 0x7C00].view("f2").reshape(1, -1)
    for z in range(16):
        records = old_metadata(scales, np.full(scales.shape, z, dtype="u1"))
        packed = PACK.pack_int4_zero(torch.from_numpy(records)).numpy()
        restored = unpack(packed)
        old, new = stats_words(records), stats_words(restored)
        np.testing.assert_array_equal(old, new)
        for q in range(16):

            def dequant(words, q=q):
                s = (words & 65535).astype("<u2").view("f2").astype("f8")
                b = (words >> 16).astype("<u2").view("f2").astype("f8")
                with np.errstate(over="ignore", invalid="ignore"):
                    return (q * s + b).astype("f2").view("u2")

            np.testing.assert_array_equal(dequant(old), dequant(new))
    # This is a material rounding boundary, not an algebraic simplification.
    s = np.float16(0.001)
    b = np.float16(-15 * np.float64(s))
    fused = np.float16(14 * np.float64(s) + np.float64(b))
    simplified = np.float16(-np.float64(s))
    assert fused.view("u2") != simplified.view("u2")


def test_all_scale_payload_bits_roundtrip():
    # No arithmetic on scale payload: signed zeros, NaNs and infinities survive.
    scales = np.arange(65536, dtype="u2").view("f2").reshape(1, -1)
    records = old_metadata(scales, np.arange(65536, dtype="u2").reshape(1, -1) % 16)
    packed = PACK.pack_int4_zero(torch.from_numpy(records)).numpy()
    np.testing.assert_array_equal(unpack(packed), records)


@pytest.mark.parametrize(
    "shape,dtype",
    [((2, 31, 3), torch.uint8), ((2, 32, 4), torch.uint8), ((2, 32, 3), torch.int32)],
)
def test_reject_unsupported_metadata(shape, dtype):
    with pytest.raises(ValueError):
        PACK.pack_int4_zero(torch.zeros(shape, dtype=dtype))


@pytest.mark.parametrize("layer,expert", [(0, 0), (23, 257), (47, 511)])
def test_real_checkpoint_tp4_all_ranks(layer, expert):
    checkpoint = os.getenv("W3_CHECKPOINT")
    if not checkpoint:
        pytest.skip("set W3_CHECKPOINT for read-only real expert validation")
    from safetensors import safe_open

    path = Path(checkpoint) / f"f1-layer-{layer:02d}.safetensors"
    with safe_open(path, framework="np") as handle:
        prefix = f"model.language_model.layers.{layer}.mlp.experts.{expert}."
        meta, codes = {}, {}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            z = handle.get_tensor(prefix + proj + ".qzeros").view("u4")
            s = handle.get_tensor(prefix + proj + ".scales")
            q = handle.get_tensor(prefix + proj + ".qweight").view("u4")

            def awq_unpack(words):
                return (
                    np.stack([(words >> (4 * i)) & 15 for i in ORDER], -1)
                    .reshape(words.shape[0], -1)
                    .astype("u1")
                )

            meta[proj] = (s, awq_unpack(z))
            codes[proj] = awq_unpack(q)
        for rank in range(4):
            start, end = rank * 160, (rank + 1) * 160
            # W13 concatenation then interleaving: gate0,up0,gate1,up1,...
            s13 = np.stack(
                [meta[p][0][:, start:end] for p in ("gate_proj", "up_proj")], -1
            ).reshape(80, 320)
            z13 = np.stack(
                [meta[p][1][:, start:end] for p in ("gate_proj", "up_proj")], -1
            ).reshape(80, 320)
            q13 = np.stack(
                [codes[p][:, start:end] for p in ("gate_proj", "up_proj")], -1
            ).reshape(2560, 320)
            s2, z2 = meta["down_proj"]
            for scale, zero, q in (
                (s13, z13, q13),
                (
                    s2[rank * 5 : rank * 5 + 5],
                    z2[rank * 5 : rank * 5 + 5],
                    codes["down_proj"][start:end],
                ),
            ):
                records = old_metadata(scale, zero)[None]
                packed = check_bank(records)
                old = stats_words(records)[0]
                new = stats_words(unpack(packed))[0]
                for word in (old, new):
                    sc = (word & 65535).astype("<u2").view("f2")
                    bias = (word >> 16).astype("<u2").view("f2")
                    group = np.arange(q.shape[0]) // 32
                    decoded = (
                        (
                            q.astype("f8") * sc[group].astype("f8")
                            + bias[group].astype("f8")
                        )
                        .astype("f2")
                        .view("u2")
                    )
                    if word is old:
                        expected = decoded
                    else:
                        np.testing.assert_array_equal(decoded, expected)


def test_flag_defaults_off_and_explicit_on(monkeypatch):
    import ast

    tree = ast.parse((ROOT / "vllm/envs.py").read_text())
    entry = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "VLLM_SM70_AWQ_MOE_INT4_ZERO"
                ):
                    entry = value.args[0]
    assert isinstance(entry, ast.Lambda)
    getter = eval(compile(ast.Expression(entry), "<env getter>", "eval"), {"os": os})
    monkeypatch.delenv("VLLM_SM70_AWQ_MOE_INT4_ZERO", raising=False)
    assert getter() is False
    monkeypatch.setenv("VLLM_SM70_AWQ_MOE_INT4_ZERO", "1")
    assert getter() is True
    monkeypatch.setenv("VLLM_SM70_AWQ_MOE_INT4_ZERO", "0")
    assert getter() is False
