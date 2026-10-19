# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest
import torch

csf = pytest.importorskip("vllm.model_executor.model_loader.csf_scales")
ref_path = Path(__file__).resolve().parents[3] / "tools/fp4_local_reference.py"
# The tools directory is mounted alongside tests in CPU qualification.
spec = importlib.util.spec_from_file_location("fp4_oracle", ref_path)
assert spec is not None and spec.loader is not None
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)


def encode(raw):
    rows, cols = raw.shape
    base = raw.to(torch.int16).amin(-1, keepdim=True).clamp(max=110)
    codes = raw.to(torch.int16) - base
    outlier = codes > 15
    codes[outlier] = 0
    packed = (codes[:, ::2] | (codes[:, 1::2] << 4)).byte()
    fixed = torch.cat(
        (base.byte().reshape(rows // 16, 16), packed.reshape(rows // 16, -1)), 1
    )
    pos = outlier.flatten().nonzero().flatten()
    words = (raw.flatten()[pos].long() << 24) | pos
    return fixed, words.to(torch.uint32)


@pytest.mark.parametrize("shape", [(16, 2), (640, 160), (2560, 40)])
def test_codec_bitwise_reference(shape):
    generator = torch.Generator().manual_seed(18)
    raw = torch.randint(48, 64, shape, generator=generator, dtype=torch.uint8)
    raw.reshape(-1)[::31] = 126
    fixed, exceptions = encode(raw)
    actual = csf.decode_nvfp4_csf_scale(fixed, exceptions).view(torch.uint8)
    assert torch.equal(actual, raw)
    assert torch.equal(actual, ref.csf_reference_bytes(fixed, exceptions))


def test_stream_reverse_order_and_passthrough_identity():
    fixed, ex = encode(torch.full((16, 4), 56, dtype=torch.uint8))
    untouched = torch.randn(3)
    items = [
        ("x", untouched),
        ("s.nvfp4_csf_exceptions", ex),
        ("s.nvfp4_csf_fixed", fixed),
    ]
    out = dict(csf.decode_csf_scale_streams(items))
    assert out["x"] is untouched
    assert out["s"].shape == (16, 4)


@pytest.mark.parametrize(
    "damage", ["bounds", "duplicate", "order", "nan", "base", "shape"]
)
def test_reject_corrupt_streams(damage):
    fixed, ex = encode(torch.full((16, 4), 56, dtype=torch.uint8))
    if damage == "bounds":
        ex = torch.tensor([64], dtype=torch.uint32)
    elif damage == "duplicate":
        ex = torch.tensor([1, 1], dtype=torch.uint32)
    elif damage == "order":
        ex = torch.tensor([2, 1], dtype=torch.uint32)
    elif damage == "nan":
        ex = torch.tensor([127 << 24], dtype=torch.uint32)
    elif damage == "base":
        fixed[0, 0] = 241
    else:
        fixed = fixed[:, :16]
    with pytest.raises(ValueError):
        csf.decode_nvfp4_csf_scale(fixed, ex)


def test_unpaired_and_duplicate_streams_fail():
    fixed, _ = encode(torch.full((16, 4), 56, dtype=torch.uint8))
    item = ("s.nvfp4_csf_fixed", fixed)
    for items in ([item], [item, item]):
        with pytest.raises(ValueError):
            list(csf.decode_csf_scale_streams(items))


def test_indexed_subdirectory_and_missing_shard(tmp_path):
    import json

    from vllm.config.load import LoadConfig
    from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

    shard = tmp_path / "tensors/w.safetensors"
    shard.parent.mkdir()
    from safetensors.torch import save_file

    save_file({"x": torch.ones(1)}, str(shard))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "tensors/w.safetensors"}})
    )
    loader = DefaultModelLoader(LoadConfig(load_format="safetensors"))
    result = loader._prepare_weights(str(tmp_path), None, None, False, None)
    assert result[1] == [str(shard)]
    shard.unlink()
    with pytest.raises(FileNotFoundError):
        loader._prepare_weights(str(tmp_path), None, None, False, None)
