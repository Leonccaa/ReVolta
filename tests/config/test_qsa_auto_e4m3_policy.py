# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Automatic QSA cache selection and SM70 MTP draft defaults."""

import json
import os
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from vllm.config import KernelConfig
from vllm.config.speculative_kv import validate_speculative_kv_cache_dtype
from vllm.models.qwen4_exp.common.kv_policy import (
    resolve_qsa_auto_cache_dtype,
    resolve_qsa_int8_aliases,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa
from vllm.v1.attention.kv_codecs import FP8_E4M3_PER_TOKEN_HEAD


def _checkpoint(tmp_path, scales=None):
    if scales is None:
        scales = {
            f"model.layers.{owner}.self_attn.{kind}_scale": torch.tensor(0.125)
            for owner in (0, 2)
            for kind in ("k", "v")
        }
    save_file(scales, tmp_path / "scales.safetensors")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: "scales.safetensors" for name in scales}})
    )
    return SimpleNamespace(
        model=str(tmp_path),
        dtype=torch.float16,
        hf_text_config=SimpleNamespace(
            indexer_n_heads=3,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=512,
            indexer_compress_ratio=4,
            layer_types=["full_attention", "linear_attention", "full_attention"],
        ),
    )


def _config(model):
    return SimpleNamespace(
        kernel_config=KernelConfig(),
        model_config=model,
        cache_config=SimpleNamespace(
            cache_dtype="auto",
            cache_dtype_from_checkpoint=False,
            calculate_kv_scales=False,
        ),
        speculative_config=None,
        parallel_config=SimpleNamespace(
            prefill_context_parallel_size=1, tensor_parallel_size=1
        ),
    )


@pytest.mark.parametrize("target_dtype", ["int8", "int8_per_token_head"])
@pytest.mark.parametrize("draft_dtype", [None, "int8", "int8_per_token_head"])
def test_qsa_int8_spellings_share_the_internal_format(
    tmp_path, target_dtype, draft_dtype
):
    cfg = _config(_checkpoint(tmp_path))
    cfg.cache_config.cache_dtype = target_dtype
    cfg.speculative_config = SimpleNamespace(method="mtp", kv_cache_dtype=draft_dtype)
    resolve_qsa_int8_aliases(cfg)
    assert cfg.cache_config.cache_dtype == "int8"
    assert cfg.speculative_config.kv_cache_dtype == (draft_dtype and "int8")


def test_non_qsa_int8_per_token_head_retains_its_format(tmp_path):
    cfg = _config(_checkpoint(tmp_path))
    cfg.model_config.hf_text_config = SimpleNamespace()
    cfg.cache_config.cache_dtype = "int8_per_token_head"
    resolve_qsa_int8_aliases(cfg)
    assert cfg.cache_config.cache_dtype == "int8_per_token_head"


@pytest.mark.parametrize("tp", [1, 2, 4, 8])
def test_auto_admission_uses_capability(tmp_path, monkeypatch, tp):
    cfg = _config(_checkpoint(tmp_path))
    cfg.parallel_config.tensor_parallel_size = tp
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)
    before = dict(os.environ)
    assert resolve_qsa_auto_cache_dtype(cfg)
    assert cfg.cache_config.cache_dtype == FP8_E4M3_PER_TOKEN_HEAD.name
    assert cfg.kernel_config.qsa_auto_e4m3_active
    assert cfg.kernel_config.qsa_auto_e4m3_reason is None
    assert dict(os.environ) == before


@pytest.fixture
def sm70(monkeypatch):
    monkeypatch.setattr(qsa.current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(qsa.current_platform, "is_device_capability", lambda _: True)
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)


@pytest.mark.parametrize(
    "target,host,hybrid,explicit,expected",
    [
        ("auto", False, True, None, "float16"),
        ("auto", True, True, None, None),
        ("auto", False, True, "int8", "int8"),
        ("auto", True, True, "float16", "float16"),
        ("fp8_per_token_head", False, True, None, "float16"),
        ("fp8_per_token_head", True, True, None, None),
        ("fp8_e4m3", False, True, None, "float16"),
        ("float16", False, True, None, None),
        ("auto", False, False, None, None),
    ],
)
def test_sm70_qsa_mtp_defaults(
    sm70, tmp_path, target, host, hybrid, explicit, expected
):
    model = _checkpoint(tmp_path)
    model.use_mla = False
    cfg = _config(model)
    cfg.is_speculative_draft = False
    cfg.use_v2_model_runner = True
    cfg.parallel_config.pipeline_parallel_size = 1
    cfg.parallel_config.decode_context_parallel_size = 1
    cfg.scheduler_config = SimpleNamespace(disable_hybrid_kv_cache_manager=not hybrid)
    cfg.kv_transfer_config = None
    cfg.cache_config.cache_dtype = target
    cfg.kernel_config.long_context_kv.enabled = host
    cfg.speculative_config = SimpleNamespace(
        method="mtp", kv_cache_dtype=explicit, draft_model_config=model
    )

    assert resolve_qsa_auto_cache_dtype(cfg) == (target == "auto")
    assert cfg.cache_config.cache_dtype == ("int8" if target == "auto" else target)
    assert not cfg.kernel_config.qsa_auto_e4m3_active
    validate_speculative_kv_cache_dtype(cfg)
    assert cfg.speculative_config.kv_cache_dtype == expected


@pytest.mark.parametrize(
    "mode",
    ["disabled", "float16", "fp8", "fp8_e4m3", "runtime_scales", "pcp", "capability"],
)
def test_unqualified_route_retains_cache_dtype(tmp_path, monkeypatch, mode):
    cfg = _config(_checkpoint(tmp_path))
    monkeypatch.setattr(
        qsa,
        "qsa_e4m3_capability_reason",
        lambda dtype: "unsupported dtype" if mode == "capability" else None,
    )
    if mode == "disabled":
        cfg.kernel_config.qsa_auto_e4m3 = False
    if mode in ("float16", "fp8", "fp8_e4m3"):
        cfg.cache_config.cache_dtype = mode
    if mode == "runtime_scales":
        cfg.cache_config.calculate_kv_scales = True
    if mode == "pcp":
        cfg.parallel_config.prefill_context_parallel_size = 2
    previous = cfg.cache_config.cache_dtype
    assert not resolve_qsa_auto_cache_dtype(cfg)
    assert cfg.cache_config.cache_dtype == previous
    assert cfg.kernel_config.qsa_auto_e4m3_reason


def test_without_qsa_metadata_keeps_auto(tmp_path, monkeypatch):
    cfg = _config(_checkpoint(tmp_path))
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)
    cfg.model_config.hf_text_config.indexer_n_heads = 0
    assert not resolve_qsa_auto_cache_dtype(cfg)
    cfg.model_config = None
    assert not resolve_qsa_auto_cache_dtype(cfg)


def test_speculative_draft_keeps_its_dtype_and_target_policy(tmp_path, monkeypatch):
    # DFlash builds its drafter from the target VllmConfig with
    # is_speculative_draft=True and cache_dtype "auto" (FP16 on SM70).
    target = _config(_checkpoint(tmp_path))
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)
    assert resolve_qsa_auto_cache_dtype(target)
    draft = SimpleNamespace(**vars(target))
    draft.is_speculative_draft = True
    draft.cache_config = SimpleNamespace(**vars(target.cache_config))
    draft.cache_config.cache_dtype = "auto"
    assert not resolve_qsa_auto_cache_dtype(draft)
    assert draft.cache_config.cache_dtype == "auto"
    assert target.kernel_config.qsa_auto_e4m3_active
    assert target.kernel_config.qsa_auto_e4m3_reason is None


def test_diagnostics_do_not_change_inactive_kernel_hash():
    policy = KernelConfig()
    original = policy.compute_hash()
    policy.qsa_auto_e4m3 = False
    policy.qsa_auto_e4m3_reason = "unsupported dtype"
    assert policy.compute_hash() == original
    policy.qsa_auto_e4m3_active = True
    assert policy.compute_hash() != original


@pytest.mark.parametrize("method", [None, "mtp", "dflash"])
@pytest.mark.parametrize("has_scales", [False, True])
def test_checkpoint_selects_planar(tmp_path, monkeypatch, method, has_scales):
    scales = None if has_scales else {"model.embed_tokens.weight": torch.ones(1)}
    cfg = _config(_checkpoint(tmp_path, scales))
    cfg.speculative_config = SimpleNamespace(method=method) if method else None
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda dtype: None)
    assert resolve_qsa_auto_cache_dtype(cfg)
    assert cfg.cache_config.cache_dtype == FP8_E4M3_PER_TOKEN_HEAD.name
    assert not cfg.cache_config.cache_dtype_from_checkpoint
    assert cfg.kernel_config.qsa_auto_e4m3_active
    assert cfg.kernel_config.qsa_auto_e4m3_reason is None


def test_planar_dtype_is_exposed_by_existing_cli_and_qsa_backend():
    from typing import get_args

    from vllm.config import CacheConfig
    from vllm.config.cache import CacheDType
    from vllm.engine.arg_utils import EngineArgs
    from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAFlashAttentionBackend
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend
    from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

    dtype = FP8_E4M3_PER_TOKEN_HEAD.name
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = EngineArgs.from_cli_args(parser.parse_args(["--kv-cache-dtype", dtype]))
    assert args.kv_cache_dtype == CacheConfig(cache_dtype=dtype).cache_dtype == dtype
    assert dtype == "fp8_per_token_head"
    assert get_args(CacheDType).count(dtype) == 1
    assert "fp8_e4m3_per_token_head" not in get_args(CacheDType)
    choices = parser._option_string_actions["--kv-cache-dtype"].choices
    assert dtype in choices and "fp8_e4m3_per_token_head" not in choices
    assert get_kv_quant_mode(dtype) == KVQuantMode.FP8_E4M3_PLANAR
    assert not TritonAttentionBackend.supports_kv_cache_dtype(dtype)
    assert dtype in Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
