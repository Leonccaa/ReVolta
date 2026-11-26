# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU admission, override, provenance and initialization-order coverage."""

import ast
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm import envs

SHARED_EXPERT_OVERLAP = "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP"

# Independent contract: do not derive expected values from implementation tables.
COMMON = {
    "VLLM_SM70_QWEN38_WIDE_M": "1",
    "VLLM_SM70_PLE_PREFILL_CONV_CHUNK": "1",
    "VLLM_SM70_QWEN38_GDN_NO_PACK": "1",
    "VLLM_SM70_GREEDY_VERIFY": "1",
    "VLLM_SM70_SKIP_REJECTED_BATCH_PACKS": "1",
    "VLLM_SM70_QSA_SHARED_KEY": "1",
    "VLLM_SM70_QSA_REQUEST_SEGMENTS": "1",
    "VLLM_SM70_QSA_COMPACT_TOPK": "1",
    "VLLM_SM70_PLE_NGRAM_BATCH": "1",
    "VLLM_SM70_PLE_INPUT_PREPARE": "1",
    "VLLM_PLE_ASYNC_REQUEST_INDICES": "1",
    "VLLM_EARLY_MODEL_INPUTS": "1",
    "VLLM_SM70_QWEN38_HYBRID_PLE": "0",
    "VLLM_USE_AOT_COMPILE": "0",
    "VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS": "1",
    "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP": "0",
    "VLLM_SM70_CAR_VMM_SAFE": "1",
}
AWQ = {
    "VLLM_SM70_AWQ_QWEN38_QPN_M1": "1",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN": "1",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS": "40",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT": "8",
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_PACK_ROWS": "24",
    "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": "1",
    "VLLM_V2_CUDAGRAPH_MEM_MIB": "700",
}
FP16 = (
    "VLLM_SM70_QWEN38_FP16_GEMV",
    "VLLM_SM70_QWEN38_FUSED_GDN_INPUT_FP16",
    "VLLM_SM70_QWEN38_FUSED_HC_FP16",
)


def model(quantization="awq", dtype=torch.float16):
    return SimpleNamespace(
        quantization=quantization,
        dtype=dtype,
        hf_config=SimpleNamespace(
            quantization_config={"quant_method": "awq", "bits": 4, "group_size": 32},
        ),
        hf_text_config=SimpleNamespace(
            hidden_size=2560,
            moe_intermediate_size=640,
            num_experts=512,
            num_experts_per_tok=10,
            indexer_n_heads=4,
            indexer_kv_heads=1,
            indexer_head_dim=128,
            indexer_budget=2048,
            indexer_compress_ratio=4,
            layer_types=["linear_attention", "full_attention"],
        ),
    )


@pytest.fixture
def profile(monkeypatch):
    # Load the installed tree, including on the baseline, using isolated state.
    # Baseline lacks the feature and explicitly skips only these new tests.
    spec = importlib.util.find_spec("vllm.revolta_profile")
    if spec is None:
        pytest.skip("baseline has no ReVolta profile")
    assert spec is not None and spec.loader is not None
    monkeypatch.setattr(os, "environ", {})
    envs.disable_envs_cache()
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    envs.disable_envs_cache()


@pytest.mark.parametrize(
    "quantization",
    ["awq", "modelopt_fp4", "modelopt_mixed", "compressed-tensors", None],
)
@pytest.mark.parametrize("mode", ["pex_p2p", "nvlink", "no_p2p"])
def test_defaults_by_format_and_interconnect(profile, quantization, mode):
    assert profile.apply_revolta_profile(
        model(quantization), 4, is_sm70=True, interconnect_mode=mode
    )
    expected = dict(COMMON)
    if mode == "no_p2p":
        expected.pop("VLLM_SM70_CAR_VMM_SAFE")
    if quantization == "awq":
        expected.update(AWQ)
    elif quantization == "modelopt_fp4":
        expected.update(
            {
                "VLLM_SM70_MTP_Q8_PREFILL_MMQ": "1",
                "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE": "1",
                "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE": "1",
                "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE": "1",
                "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "0",
                "VLLM_SM70_NVFP4_TURBOMIND": "1",
                "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL": "1",
                "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": "1",
                "VLLM_V2_CUDAGRAPH_MEM_MIB": "700",
            }
        )
    assert {
        k: v
        for k, v in os.environ.items()
        if k not in ("PYTEST_CURRENT_TEST", "VLLM_REVOLTA_PROFILE_PROVENANCE")
    } == expected
    assert not set(FP16) & set(os.environ)
    assert not any(name.startswith("PYTORCH_") for name in os.environ)


@pytest.mark.parametrize("name,value", list(COMMON.items()) + list(AWQ.items()))
def test_every_explicit_override_survives(profile, monkeypatch, name, value):
    override = "0" if value == "1" else "1"
    monkeypatch.setenv(name, override)
    assert profile.apply_revolta_profile(model(), 4, is_sm70=True)
    assert os.environ[name] == override
    assert json.loads(profile.profile_audit(enabled=True))["values"][name] == {
        "value": override,
        "source": "environment",
    }


@pytest.mark.parametrize("name,value", list(COMMON.items()) + list(AWQ.items()))
def test_every_default_has_profile_provenance(profile, name, value):
    assert profile.apply_revolta_profile(model(), 4, is_sm70=True)
    # A second config initialization must not relabel profile entries as env.
    assert profile.apply_revolta_profile(model(), 4, is_sm70=True)
    row = json.loads(profile.profile_audit(enabled=True))["values"][name]
    assert row == {"value": value, "source": "profile"}


@pytest.mark.parametrize("tp", [1, 2, 8])
def test_other_tp_does_not_mutate_environment(profile, tp):
    assert not profile.apply_revolta_profile(model(), tp, is_sm70=True)
    assert set(os.environ) <= {"PYTEST_CURRENT_TEST"}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_non_fp16_does_not_mutate_environment(profile, dtype):
    assert not profile.apply_revolta_profile(model(dtype=dtype), 4, is_sm70=True)
    assert set(os.environ) <= {"PYTEST_CURRENT_TEST"}


def test_non_sm70_and_missing_model(profile):
    assert not profile.apply_revolta_profile(model(), 4, is_sm70=False)
    assert not profile.apply_revolta_profile(None, 4, is_sm70=True)
    assert set(os.environ) <= {"PYTEST_CURRENT_TEST"}


@pytest.mark.parametrize(
    "field",
    [
        "indexer_n_heads",
        "indexer_kv_heads",
        "indexer_head_dim",
        "indexer_budget",
        "indexer_compress_ratio",
        "layer_types",
    ],
)
@pytest.mark.parametrize("bad", [None, 0])
def test_incomplete_qsa_does_not_match_model_name(profile, field, bad):
    config = model()
    config.architectures = ["Qwen4ExpForConditionalGeneration"]
    setattr(config.hf_text_config, field, bad)
    assert not profile.apply_revolta_profile(config, 4, is_sm70=True)
    assert set(os.environ) <= {"PYTEST_CURRENT_TEST"}


@pytest.mark.parametrize(
    "layer_type", ["full_attention", "qwen_sparse_attention", "indexed_attention"]
)
def test_qsa_sparse_layer_spelling_and_unrelated_name(profile, layer_type):
    config = model()
    config.architectures = ["AnotherQSAModel"]
    config.hf_text_config.layer_types = [layer_type]
    assert profile.apply_revolta_profile(config, 4, is_sm70=True)


def test_sources_after_upstream_auto_defaults(profile, monkeypatch):
    # A separate import takes the environment snapshot before auto defaults.
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT", "16")
    spec = importlib.util.find_spec("vllm.revolta_profile")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.apply_revolta_profile(model(), 4, is_sm70=True)
    monkeypatch.setenv(FP16[0], "1")  # downstream upstream policy
    monkeypatch.setenv("NCCL_GRAPH_REGISTER", "0")  # env_override
    audit = json.loads(module.profile_audit(enabled=True))
    assert audit["enabled"]
    assert len(audit["values"]) == 49
    assert audit["values"][FP16[0]]["source"] == "upstream_default"
    assert audit["values"]["NCCL_GRAPH_REGISTER"]["source"] == "upstream_default"
    assert audit["values"]["VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT"] == {
        "value": "16",
        "source": "environment",
    }
    assert audit["values"]["VLLM_SM70_QSA_INDEXER_CUBLAS"]["value"] == "1"
    assert "\n" not in module.profile_audit(enabled=True)


def inherited_profile():
    """Fresh module state, as in a spawned child inheriting only environ."""
    spec = importlib.util.find_spec("vllm.revolta_profile")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("enabled", [True, False])
def test_audit_provenance_survives_child_and_grandchild(profile, monkeypatch, enabled):
    # An explicit value identical to the profile must remain environment.
    explicit = "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT"
    monkeypatch.setenv(explicit, "8")
    parent = inherited_profile()
    parent.apply_revolta_profile(model(), 4 if enabled else 2, is_sm70=True)
    monkeypatch.setenv(FP16[0], "1")
    monkeypatch.setenv("NCCL_GRAPH_REGISTER", "0")
    expected = parent.profile_audit(enabled=enabled)
    for _ in range(2):
        child = inherited_profile()
        child.apply_revolta_profile(model(), 4 if enabled else 2, is_sm70=True)
        assert child.profile_audit(enabled=enabled) == expected
    rows = json.loads(expected)["values"]
    assert rows[explicit] == {"value": "8", "source": "environment"}
    assert rows[FP16[0]] == {"value": "1", "source": "upstream_default"}
    assert rows["NCCL_GRAPH_REGISTER"]["source"] == "upstream_default"
    if enabled:
        assert rows["VLLM_SM70_GREEDY_VERIFY"]["source"] == "profile"


def test_child_can_start_after_apply_before_audit(profile, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_QSA_SHARED_KEY", "1")
    profile.apply_revolta_profile(model(), 4, is_sm70=True)
    child = inherited_profile()
    child.apply_revolta_profile(model(), 4, is_sm70=True)
    assert child.profile_audit(enabled=True) == profile.profile_audit(enabled=True)


@pytest.mark.parametrize("name", ["VLLM_SM70_GREEDY_VERIFY", FP16[0]])
def test_changed_inherited_value_is_environment(profile, monkeypatch, name):
    profile.apply_revolta_profile(model(), 4, is_sm70=True)
    monkeypatch.setenv(FP16[0], "1")
    profile.profile_audit(enabled=True)
    monkeypatch.setenv(name, "0")
    child = inherited_profile()
    child.apply_revolta_profile(model(), 4, is_sm70=True)
    assert json.loads(child.profile_audit(enabled=True))["values"][name] == {
        "value": "0",
        "source": "environment",
    }


@pytest.mark.parametrize(
    "value", ["broken", "[]", '{"version": 999}', '{"version": 1}']
)
def test_invalid_provenance_does_not_change_defaults(profile, monkeypatch, value):
    monkeypatch.setenv("VLLM_REVOLTA_PROFILE_PROVENANCE", value)
    child = inherited_profile()
    child.apply_revolta_profile(model(), 4, is_sm70=True)
    assert os.environ["VLLM_SM70_GREEDY_VERIFY"] == "1"
    assert (
        json.loads(child.profile_audit(enabled=True))["values"][
            "VLLM_SM70_GREEDY_VERIFY"
        ]["source"]
        == "profile"
    )


def test_provenance_never_changes_compile_factors(profile, monkeypatch):
    # Keep actual serving variables constant; only the diagnostic changes.
    from vllm.config import utils  # noqa: F401

    monkeypatch.setattr(envs, "environment_variables", {"VLLM_PORT": lambda: 8000})
    monkeypatch.setenv("VLLM_REVOLTA_PROFILE_PROVENANCE", "parent audit")
    before = envs.compile_factors()
    monkeypatch.setenv("VLLM_REVOLTA_PROFILE_PROVENANCE", "child audit")
    assert envs.compile_factors() == before
    assert "VLLM_REVOLTA_PROFILE_PROVENANCE" not in before


def test_invalid_interconnect_fails_before_writing(profile):
    with pytest.raises(ValueError, match="interconnect"):
        profile.apply_revolta_profile(model(), 4, is_sm70=True, interconnect_mode="bad")
    assert set(os.environ) <= {"PYTEST_CURRENT_TEST"}


def test_no_p2p_keeps_explicit_custom_ar_setting(profile, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_CAR_VMM_SAFE", "1")
    profile.apply_revolta_profile(model(), 4, is_sm70=True, interconnect_mode="no_p2p")
    assert os.environ["VLLM_SM70_CAR_VMM_SAFE"] == "1"


def test_qpn_explicit_profile_keeps_fail_closed(profile):
    from vllm.model_executor.layers.quantization.awq_qpn_sm70 import initialize_qpn_m1

    profile.apply_revolta_profile(model(), 4, is_sm70=True)
    with pytest.raises(RuntimeError, match="native-g32 TP4"):
        initialize_qpn_m1(SimpleNamespace(), shape_contract=False)


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 3072),
        ("moe_intermediate_size", 704),
        ("moe_intermediate_size", 512),
        ("num_experts", 256),
        ("num_experts_per_tok", 8),
        ("hidden_size", None),
    ],
)
@pytest.mark.parametrize("method", [None, "dflash"])
def test_other_awq_shapes_leave_tuned_defaults_unset(profile, field, value, method):
    from vllm.model_executor.layers.quantization.awq_qpn_sm70 import initialize_qpn_m1

    target = model()
    setattr(target.hf_text_config, field, value)
    assert profile.apply_revolta_profile(
        target, 4, is_sm70=True, speculative_method=method
    )
    assert not (set(AWQ) | {TAIL_GRAPHS}) & set(os.environ)
    assert os.environ["VLLM_SM70_QSA_SHARED_KEY"] == "1"
    assert not initialize_qpn_m1(SimpleNamespace(), shape_contract=False)


@pytest.mark.parametrize(
    "metadata",
    [None, {}, {"bits": 4, "group_size": 128}, {"bits": 8, "group_size": 32}],
)
def test_non_native_g32_or_unknown_quantization_uses_generic_awq(profile, metadata):
    target = model()
    target.hf_config.quantization_config = metadata
    assert profile.apply_revolta_profile(target, 4, is_sm70=True)
    assert not set(AWQ) & set(os.environ)


@pytest.mark.parametrize("owner", ["hf_config", "hf_text_config", "model_arch_config"])
@pytest.mark.parametrize("legacy_keys", [False, True])
def test_production_awq_metadata_locations_and_aliases(profile, owner, legacy_keys):
    target = model()
    del target.hf_config.quantization_config
    metadata = (
        {"w_bit": 4, "q_group_size": 32}
        if legacy_keys
        else {"bits": 4, "group_size": 32}
    )
    if not hasattr(target, owner):
        setattr(target, owner, SimpleNamespace())
    getattr(target, owner).quantization_config = metadata
    assert profile.apply_revolta_profile(target, 4, is_sm70=True)
    assert all(os.environ.get(name) == value for name, value in AWQ.items())


def test_unsupported_awq_shape_keeps_explicit_fail_closed_request(profile, monkeypatch):
    from vllm.model_executor.layers.quantization.awq_qpn_sm70 import initialize_qpn_m1

    target = model()
    target.hf_text_config.num_experts = 256
    monkeypatch.setenv("VLLM_SM70_AWQ_QWEN38_QPN_M1", "1")
    assert profile.apply_revolta_profile(target, 4, is_sm70=True)
    with pytest.raises(RuntimeError, match="native-g32 TP4"):
        initialize_qpn_m1(SimpleNamespace(), shape_contract=False)


@pytest.mark.parametrize(
    "is_sm70,dtype",
    [(True, torch.float16), (False, torch.float16), (True, torch.bfloat16)],
)
@pytest.mark.parametrize("override", [None, "0"])
def test_upstream_still_owns_fp16_projection_gates(
    profile, monkeypatch, is_sm70, dtype, override
):
    from vllm.config.vllm import _apply_sm70_qwen38_decode_defaults

    target = model(dtype=dtype)
    target.architectures = ["Qwen4ExpForConditionalGeneration"]
    target.multimodal_config = SimpleNamespace(language_model_only=False)
    cfg = SimpleNamespace(
        model_config=target,
        speculative_config=None,
        parallel_config=SimpleNamespace(enable_expert_parallel=False, enable_dbo=False),
    )
    if override is not None:
        for name in FP16:
            monkeypatch.setenv(name, override)
    profile.apply_revolta_profile(target, 4, is_sm70=is_sm70)
    assert all(os.environ.get(name) == override for name in FP16)
    _apply_sm70_qwen38_decode_defaults(cfg, is_sm70=is_sm70)
    expected = override
    if override is None and is_sm70 and dtype == torch.float16:
        expected = "1"
    assert all(os.environ.get(name) == expected for name in FP16)


@pytest.mark.parametrize("caps", [[], [(8, 0)], [(7, 0), (8, 0)], [(7, 0)] * 4])
def test_hardware_detection_without_cuda_context(profile, monkeypatch, caps):
    import vllm.platforms

    platform = SimpleNamespace(
        is_cuda=lambda: True,
        device_count=lambda: len(caps),
        is_device_capability=lambda cap, device_id: caps[device_id] == cap,
    )
    monkeypatch.setattr(vllm.platforms, "current_platform", platform)
    assert profile.all_visible_devices_sm70() == bool(
        caps and all(c == (7, 0) for c in caps)
    )
    platform.is_cuda = lambda: False
    assert not profile.all_visible_devices_sm70()


def run_engineargs_early_policy(profile, monkeypatch, apply, **engine_kwargs):
    """Execute EngineArgs' real sequence from the profile import to MTP defaults."""
    import vllm.revolta_profile as installed
    from vllm.config import CompilationConfig
    from vllm.engine.arg_utils import EngineArgs

    monkeypatch.setattr(installed, "apply_revolta_profile", apply)
    monkeypatch.setattr(
        installed, "frontend_interconnect_mode", profile.frontend_interconnect_mode
    )
    monkeypatch.setattr(installed, "all_visible_devices_sm70", lambda: True)
    # Execute the real early sequence, including the real MTP default method.
    import vllm.engine.arg_utils as arg_utils

    monkeypatch.setattr(arg_utils, "get_model_path", lambda path, _revision: path)
    monkeypatch.setattr(
        arg_utils,
        "current_platform",
        SimpleNamespace(
            is_cuda_alike=lambda: True,
            get_device_capability=lambda: SimpleNamespace(major=7, minor=0),
        ),
    )
    assert arg_utils.__file__ is not None
    tree = ast.parse(Path(arg_utils.__file__).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineArgs"
    )
    fn = next(
        n
        for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == "create_engine_config"
    )
    start = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.ImportFrom) and n.module == "vllm.revolta_profile"
    )
    end = next(
        i
        for i, n in enumerate(fn.body)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Attribute)
        and n.value.func.attr == "_maybe_apply_sm70_mtp_defaults"
    )
    args = EngineArgs(
        model="unused",
        tensor_parallel_size=4,
        max_num_seqs=8,
        compilation_config=CompilationConfig(),
        speculative_config={"method": "mtp", "num_speculative_tokens": 4},
        **engine_kwargs,
    )
    target = model()
    target.hf_text_config.mtp_num_hidden_layers = 1
    target.is_hybrid = True
    target.runner_type = "generate"
    args._check_feature_supported = Mock()
    exec(
        compile(
            ast.Module(body=fn.body[start : end + 1], type_ignores=[]),
            "early_policy",
            "exec",
        ),
        {
            "self": args,
            "model_config": target,
            "usage_context": SimpleNamespace(name="OPENAI_API_SERVER"),
        },
    )
    return args


def test_engineargs_profile_precedes_actual_mtp_defaults(profile, monkeypatch):
    args = run_engineargs_early_policy(
        profile, monkeypatch, profile.apply_revolta_profile
    )
    assert args.compilation_config.fast_moe_cold_start is None
    assert args.compilation_config.cudagraph_capture_sizes is None


def test_hybrid_serving_environment_overrides(profile, monkeypatch):
    overrides = {
        "VLLM_SM70_QSA_GROUPED_LPT": "0",
        "REVOLTA_SKINNY_MPLAN": "0",
        "VLLM_SM70_DFLASH2_FUSED_QKV_PACK": "0",
        "VLLM_SM70_AWQ_MOE_INT4_ZERO": "0",
    }
    for name, value in overrides.items():
        monkeypatch.setenv(name, value)
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, hybrid_serving=True, speculative_method="dflash"
    )
    assert all(os.environ.get(name) == value for name, value in overrides.items())
    assert "VLLM_SM70_GDN_SKINNY_MPLAN" not in os.environ


@pytest.mark.parametrize(("method", "qkv_pack"), [("dflash", "1"), ("mtp", None)])
def test_hybrid_serving_environment_defaults(profile, method, qkv_pack):
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, hybrid_serving=True, speculative_method=method
    )
    assert os.environ["VLLM_SM70_QSA_GROUPED_LPT"] == "1"
    assert os.environ["VLLM_SM70_GDN_SKINNY_MPLAN"] == "1"
    assert os.environ["VLLM_SM70_AWQ_MOE_INT4_ZERO"] == "1"
    assert os.environ.get("VLLM_SM70_DFLASH2_FUSED_QKV_PACK") == qkv_pack


def hybrid_serving_args(
    profile,
    monkeypatch,
    tmp_path,
    method,
    flags,
    spec_options,
    *,
    active=True,
    serving=True,
    indexer_budget=2048,
    dflash_selector=True,
    shorthand=False,
):
    from transformers import Qwen3Config

    from vllm import revolta_profile
    from vllm.config import speculative
    from vllm.engine import arg_utils
    from vllm.platforms import current_platform
    from vllm.platforms.cpu import CpuPlatform
    from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    Qwen3NextConfig(
        architectures=["Qwen3NextForCausalLM"], num_key_value_heads=2
    ).save_pretrained(tmp_path / "target")
    argv = [
        "--model",
        str(tmp_path / "target"),
        "--dtype",
        "half",
        "--tensor-parallel-size",
        "4",
        "--distributed-executor-backend",
        "mp",
        *flags,
    ]
    if method is not None:
        raw = {
            "method": method,
            "num_speculative_tokens": 7 if method == "dflash" else 4,
            **spec_options,
        }
        if method == "dflash":
            dflash_config = {"block_size": 8}
            draft_arch = "DFlashDraftModel"
            if dflash_selector:
                dflash_config["selector_top_k"] = 16
                draft_arch = "DFlash2DraftModel"
            Qwen3Config(
                architectures=[draft_arch],
                num_attention_heads=4,
                num_key_value_heads=2,
                dflash_config=dflash_config,
            ).save_pretrained(tmp_path / "draft")
            raw["model"] = str(tmp_path / "draft")
        if shorthand:
            argv.extend(
                (
                    "--spec-method",
                    raw.pop("method"),
                    "--spec-model",
                    raw.pop("model"),
                    "--spec-tokens",
                    str(raw.pop("num_speculative_tokens")),
                )
            )
        if raw:
            argv.extend(("--speculative-config", json.dumps(raw)))

    monkeypatch.setattr(arg_utils, "current_platform", CpuPlatform())
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(revolta_profile, "all_visible_devices_sm70", lambda: active)
    monkeypatch.setattr(
        revolta_profile, "apply_revolta_profile", profile.apply_revolta_profile
    )
    monkeypatch.setattr(
        speculative,
        "ModelConfig",
        lambda **kwargs: arg_utils.ModelConfig(**(kwargs | {"quantization": None})),
    )
    parser = arg_utils.EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = arg_utils.EngineArgs.from_cli_args(parser.parse_args(argv))
    target = args.create_model_config()
    target.quantization = "awq"
    target.hf_text_config.update(vars(model().hf_text_config))
    target.hf_text_config.indexer_budget = indexer_budget
    target.hf_config.quantization_config = model().hf_config.quantization_config
    monkeypatch.setattr(args, "create_model_config", lambda: target)

    context = UsageContext.OPENAI_API_SERVER if serving else UsageContext.LLM_CLASS
    return args, context


MTP_DSD = [(1, 3, 4), (4, 8, 3)]
DFLASH_DSD = [(1, 3, 7), (4, 4, 3), (5, 8, 4)]
DFLASH_LOOKUP = (5, 16, 65536, 1)


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("method", "flags", "spec_options", "active", "serving", "expected"),
    [
        # Expected: sequences, tokens, prefix cache, DSD, lookup, long threshold.
        ("mtp", [], {}, True, True, (8, 8192, True, MTP_DSD, None, 4896)),
        (None, [], {}, True, True, (8, 8192, True, None, None, 4896)),
        (
            "mtp",
            ["--no-enable-prefix-caching"],
            {},
            True,
            True,
            (8, 8192, False, None, None, 4896),
        ),
        (
            "mtp",
            [
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                "4096",
                "--long-prefill-token-threshold",
                "0",
            ],
            {"num_speculative_tokens_per_batch_size": [[1, 4, 2]]},
            True,
            True,
            (4, 4096, True, [(1, 4, 2)], None, 0),
        ),
        (
            "mtp",
            [],
            {"num_speculative_tokens": 3},
            True,
            True,
            (8, 8192, True, None, None, 4896),
        ),
        (
            "dflash",
            [],
            {},
            True,
            True,
            (8, 8192, True, DFLASH_DSD, DFLASH_LOOKUP, 4896),
        ),
        (
            "dflash",
            [],
            {"dflash2_lookup": {"enabled": False}},
            True,
            True,
            (8, 8192, True, DFLASH_DSD, None, 4896),
        ),
        (
            "dflash",
            [],
            {
                "num_speculative_tokens_per_batch_size": None,
                "dflash2_lookup": {
                    "enabled": True,
                    "policy": "longest",
                    "nmin": 5,
                    "nmax": 10,
                    "max_requests": 2,
                },
            },
            True,
            True,
            (8, 8192, True, None, (5, 10, 65536, 2), 4896),
        ),
        (
            "dflash",
            [],
            {"num_speculative_tokens": 4},
            True,
            True,
            (8, 8192, True, None, None, 4896),
        ),
        ("mtp", [], {}, True, False, (1024, 16384, False, None, None, 0)),
        ("mtp", [], {}, False, True, (512, 8192, False, None, None, 0)),
    ],
)
def test_engineargs_revolta_hybrid_serving_defaults(
    profile,
    monkeypatch,
    tmp_path,
    method,
    flags,
    spec_options,
    active,
    serving,
    expected,
):
    from vllm.engine import arg_utils

    args, context = hybrid_serving_args(
        profile,
        monkeypatch,
        tmp_path,
        method,
        flags,
        spec_options,
        active=active,
        serving=serving,
    )
    monkeypatch.setattr(arg_utils, "VllmConfig", SimpleNamespace)
    config = args.create_engine_config(usage_context=context)
    scheduler = config.scheduler_config
    spec = config.speculative_config
    lookup = getattr(spec, "dflash2_lookup", None)
    assert (
        scheduler.max_num_seqs,
        scheduler.max_num_batched_tokens,
        config.cache_config.enable_prefix_caching,
        getattr(spec, "num_speculative_tokens_per_batch_size", None),
        (lookup.nmin, lookup.nmax, lookup.search_window, lookup.max_requests)
        if lookup is not None and lookup.enabled
        else None,
        scheduler.long_prefill_token_threshold,
    ) == expected


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("source", ["cli", "yaml"])
def test_hybrid_serving_preserves_disabled_prefill(
    profile, monkeypatch, tmp_path, source
):
    from vllm.engine import arg_utils

    flags = ["--max-model-len", "16384"]
    if source == "yaml":
        config_file = tmp_path / "serve.yaml"
        config_file.write_text(
            "enable-prefix-caching: false\n"
            "enable-chunked-prefill: false\n"
            "async-scheduling: false\n"
        )
        flags.extend(("--config", str(config_file)))
    else:
        flags.extend(
            (
                "--no-enable-prefix-caching",
                "--no-enable-chunked-prefill",
                "--no-async-scheduling",
            )
        )
    args, context = hybrid_serving_args(
        profile, monkeypatch, tmp_path, "mtp", flags, {}
    )
    monkeypatch.setattr(arg_utils, "VllmConfig", SimpleNamespace)
    config = args.create_engine_config(usage_context=context)
    scheduler = config.scheduler_config
    assert scheduler.max_num_batched_tokens == 16384
    assert scheduler.enable_chunked_prefill is False
    assert scheduler.async_scheduling is False
    assert config.cache_config.enable_prefix_caching is False


@pytest.fixture
def sm70_serving_resolution(monkeypatch):
    from vllm.config import VllmConfig
    from vllm.platforms import current_platform

    # Run real config resolution with the GPU capability checks stubbed.
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr(VllmConfig, "_get_quantization_config", lambda *_: None)
    monkeypatch.setattr(current_platform, "check_and_update_config", lambda _: None)
    monkeypatch.setattr(
        "vllm.config.vllm._any_participating_device_is_capability",
        lambda _, capability: capability == (7, 0),
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("active", [True, False])
def test_hybrid_serving_attention_backend(profile, monkeypatch, tmp_path, active):
    from vllm.engine import arg_utils
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    args, context = hybrid_serving_args(
        profile, monkeypatch, tmp_path, None, [], {}, active=active
    )
    monkeypatch.setattr(arg_utils, "VllmConfig", SimpleNamespace)
    config = args.create_engine_config(usage_context=context)
    expected = AttentionBackendEnum.FLASH_ATTN_V100 if active else None
    assert config.attention_config.backend is expected


@pytest.mark.skip_global_cleanup
def test_dflash2_shorthand_matches_json(
    profile, monkeypatch, tmp_path, sm70_serving_resolution
):
    from vllm.engine import arg_utils
    from vllm.platforms.interface import DeviceCapability

    resolved = []
    for shorthand in (False, True):
        args, context = hybrid_serving_args(
            profile, monkeypatch, tmp_path, "dflash", [], {}, shorthand=shorthand
        )
        platform = arg_utils.current_platform
        monkeypatch.setattr(platform, "is_cuda_alike", lambda: True)
        monkeypatch.setattr(
            platform,
            "get_device_capability",
            lambda device_id=0: DeviceCapability(7, 0),
        )
        spec = args.create_engine_config(usage_context=context).speculative_config
        assert spec.draft_sample_method == "probabilistic"
        assert spec.attention_backend.name == "FLASH_ATTN_V100"
        resolved.append(spec.to_worker_payload())
    assert resolved[0] == resolved[1]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("method", "flags", "spec_options", "budget", "expected"),
    [
        ("mtp", [], {}, 2048, (False, False, True)),
        ("mtp", ["--async-scheduling"], {}, 2048, (False, True, False)),
        ("mtp", [], {"gdn_state_replay": False}, 2048, (False, True, False)),
        ("dflash", [], {}, 2048, (True, False, True)),
        ("dflash", [], {}, 8192, (False, False, True)),
        (
            "dflash",
            ["--kernel-config", '{"qsa_dcp_index_shard":false}'],
            {"sm70_dflash2": {"spec_state_replay": False}},
            2048,
            (False, False, False),
        ),
        (
            "dflash",
            [],
            {"sm70_dflash2": {"fused_gdn_verify": True}},
            2048,
            (True, False, False),
        ),
    ],
)
def test_hybrid_serving_index_sharding_and_replay(
    profile,
    monkeypatch,
    tmp_path,
    sm70_serving_resolution,
    method,
    flags,
    spec_options,
    budget,
    expected,
):
    args, context = hybrid_serving_args(
        profile,
        monkeypatch,
        tmp_path,
        method,
        ["--decode-context-parallel-size", "2", *flags],
        spec_options,
        indexer_budget=budget,
    )
    config = args.create_engine_config(usage_context=context)
    assert (
        config.kernel_config.qsa_dcp_index_shard,
        config.scheduler_config.async_scheduling,
        config.speculative_config.gdn_spec_state_replay,
    ) == expected
    assert config.cache_config.mamba_cache_mode == "align"


@pytest.mark.skip_global_cleanup
def test_hybrid_serving_mtp_pp2_keeps_fixed_width(
    profile, monkeypatch, tmp_path, sm70_serving_resolution
):
    from vllm.model_executor.models.registry import ModelRegistry

    # The local metadata fixture stands in for a PP-capable target and draft.
    monkeypatch.setattr(ModelRegistry, "is_pp_supported_model", lambda *_: True)
    args, context = hybrid_serving_args(
        profile, monkeypatch, tmp_path, "mtp", ["--pipeline-parallel-size", "2"], {}
    )
    spec = args.create_engine_config(usage_context=context).speculative_config
    assert spec.num_speculative_tokens == 4
    assert spec.num_speculative_tokens_per_batch_size is None


# I1 supplied these values explicitly in its speculative configuration.
I1_DFLASH2_POLICY = {
    "context_kv_graph": True,
    "context_pipeline": True,
    "verify_fastpath": True,
    "fused_gdn_metadata": True,
    "fused_smallq_metadata": True,
    "grouped_smallq_metadata": True,
    "spec_state_replay": True,
}


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("active", "serving", "selector", "overrides", "environment", "expected"),
    [
        (True, True, True, {}, {}, I1_DFLASH2_POLICY),
        (
            True,
            True,
            True,
            {"context_pipeline": False},
            {
                "VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH": "0",
                "VLLM_SM70_DFLASH2_FP32_LOGITS": "1",
            },
            I1_DFLASH2_POLICY | {"context_pipeline": False, "context_kv_graph": False},
        ),
        (
            True,
            True,
            True,
            {"context_kv_graph": False},
            {},
            I1_DFLASH2_POLICY | {"context_kv_graph": False, "context_pipeline": False},
        ),
        (
            True,
            True,
            True,
            {},
            {"VLLM_SM70_DFLASH2_CONTEXT_KV_GRAPH": "0"},
            I1_DFLASH2_POLICY | {"context_kv_graph": False, "context_pipeline": False},
        ),
        (
            True,
            True,
            True,
            {"context_kv_graph": False, "context_pipeline": True},
            {},
            I1_DFLASH2_POLICY | {"context_kv_graph": False},
        ),
        (True, True, False, {}, {}, {"spec_state_replay": True}),
        (True, False, True, {}, {}, {}),
        (False, True, True, {}, {}, {}),
    ],
)
def test_dflash2_serving_verifier_defaults_match_i1(
    profile,
    monkeypatch,
    tmp_path,
    sm70_serving_resolution,
    active,
    serving,
    selector,
    overrides,
    environment,
    expected,
):
    from vllm.config.sm70_dflash2 import Sm70DFlash2Config

    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    args, context = hybrid_serving_args(
        profile,
        monkeypatch,
        tmp_path,
        "dflash",
        [],
        {"sm70_dflash2": overrides} if overrides else {},
        active=active,
        serving=serving,
        dflash_selector=selector,
    )
    spec = args.create_engine_config(usage_context=context).speculative_config
    policy = spec.sm70_dflash2

    reference = Sm70DFlash2Config(**expected)
    reference.resolve(qualified=False)
    assert policy.qualified is False
    assert policy.graph_options() == reference.graph_options()
    assert policy.spec_state_replay is reference.spec_state_replay
    worker_policy = spec.to_worker_payload()["derived"]["sm70_dflash2"]
    assert all(worker_policy[name] == value for name, value in expected.items())


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("connector", "override", "expected"),
    [
        ("NixlConnector", None, (None, None)),
        ("OffloadingConnector", None, ("expandable_segments:True",) * 2),
        (None, "expandable_segments:False", (None, "expandable_segments:False")),
    ],
)
def test_hybrid_serving_allocator_with_connector(
    profile,
    monkeypatch,
    tmp_path,
    sm70_serving_resolution,
    connector,
    override,
    expected,
):
    flags: list[str] = []
    if connector:
        config = {"kv_connector": connector, "kv_role": "kv_both"}
        flags.extend(("--kv-transfer-config", json.dumps(config)))
    if override:
        monkeypatch.setenv("PYTORCH_CUDA_ALLOC_CONF", override)
    args, context = hybrid_serving_args(
        profile, monkeypatch, tmp_path, "mtp", flags, {}
    )
    args.create_engine_config(usage_context=context)
    assert (
        os.environ.get("PYTORCH_ALLOC_CONF"),
        os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
    ) == expected


@pytest.mark.parametrize(
    ("quantization", "method", "source", "overrides", "expected"),
    [
        ("awq", "dflash", "cli", {}, (1024, 1, 0)),
        ("modelopt_fp4", "mtp", "yaml", {}, (1024, 1, 0)),
        ("awq", "qwen4_exp_mtp", "json", {}, (1024, 1, 0)),
        ("awq", None, "json", {}, (0, 1, 250)),
        ("awq", "mtp", "json", {"enable_chunked_prefill": False}, (0, 1, 0)),
        (
            "awq",
            "dflash",
            "json",
            {
                "prefill_pressure_token_budget": 0,
                "prefill_pressure_threshold": 2,
                "mixed_prefill_step_latency_ms": 250,
            },
            (0, 2, 250),
        ),
    ],
)
def test_engineargs_prefill_pressure_profile(
    profile,
    monkeypatch,
    tmp_path,
    quantization,
    method,
    source,
    overrides,
    expected,
):
    import yaml
    from transformers import Qwen3Config

    from vllm import revolta_profile
    from vllm.config import speculative
    from vllm.engine import arg_utils
    from vllm.platforms.cpu import CpuPlatform
    from vllm.platforms.interface import DeviceCapability
    from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    model_dir = tmp_path / "target"
    Qwen3NextConfig(architectures=["Qwen3NextForCausalLM"]).save_pretrained(model_dir)
    raw = {"method": method, "num_speculative_tokens": 4}
    if method == "dflash":
        draft = tmp_path / "draft"
        Qwen3Config(
            architectures=["DFlash2DraftModel"],
            num_attention_heads=4,
            num_key_value_heads=4,
            dflash_config={"selector_top_k": 16, "block_size": 5},
        ).save_pretrained(draft)
        raw["model"] = str(draft)

    argv = [
        "--model",
        str(model_dir),
        "--dtype",
        "float16",
        "--tensor-parallel-size",
        "4",
        "--distributed-executor-backend",
        "mp",
    ]
    if source == "yaml":
        config = tmp_path / "serve.yaml"
        config.write_text(yaml.safe_dump({"spec-method": method, "spec-tokens": 4}))
        argv.extend(["--config", str(config)])
    elif source == "cli":
        argv.extend(
            [
                "--spec-method",
                method,
                "--spec-tokens",
                "4",
                "--spec-model",
                raw["model"],
            ]
        )
    elif method is not None:
        argv.extend(["--speculative-config", json.dumps(raw)])

    monkeypatch.setattr(arg_utils, "current_platform", CpuPlatform())
    monkeypatch.setattr(revolta_profile, "all_visible_devices_sm70", lambda: True)
    monkeypatch.setattr(
        revolta_profile, "apply_revolta_profile", profile.apply_revolta_profile
    )
    # Keep real speculative argument resolution and scheduler construction.
    # The local checkpoint configs stand in for quantized weights.
    monkeypatch.setattr(
        speculative,
        "ModelConfig",
        lambda **kwargs: arg_utils.ModelConfig(**(kwargs | {"quantization": None})),
    )
    parser = arg_utils.EngineArgs.add_cli_args(FlexibleArgumentParser())
    args = arg_utils.EngineArgs.from_cli_args(parser.parse_args(argv))
    for name, value in overrides.items():
        setattr(args, name, value)
    target = args.create_model_config()
    target.quantization = quantization
    target.hf_text_config.update(vars(model(quantization).hf_text_config))
    legacy_mtp_graphs = method == "mtp" and source == "json"
    if legacy_mtp_graphs:
        monkeypatch.setenv("VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS", "0")
        monkeypatch.setattr(arg_utils.current_platform, "is_cuda_alike", lambda: True)
        monkeypatch.setattr(
            arg_utils.current_platform,
            "get_device_capability",
            lambda: DeviceCapability(7, 0),
        )
        args.max_num_seqs = 8
    monkeypatch.setattr(args, "create_model_config", lambda: target)
    monkeypatch.setattr(arg_utils, "VllmConfig", SimpleNamespace)

    config = args.create_engine_config()
    assert getattr(config.speculative_config, "method", None) == (
        "mtp" if method == "qwen4_exp_mtp" else method
    )
    scheduler = config.scheduler_config
    assert (
        scheduler.prefill_pressure_token_budget,
        scheduler.prefill_pressure_threshold,
        scheduler.mixed_prefill_step_latency_ms,
    ) == expected
    if legacy_mtp_graphs:
        sizes = config.compilation_config.cudagraph_capture_sizes
        assert sizes is not None and max(sizes) == 40


@pytest.mark.parametrize(
    ("requested", "disabled", "expected"),
    [
        (None, False, "pex_p2p"),
        ("nccl", False, "no_p2p"),
        ("nvlink", False, "nvlink"),
        ("auto", True, "no_p2p"),
    ],
)
def test_engineargs_profile_receives_frontend_interconnect(
    profile, monkeypatch, requested, disabled, expected
):
    if requested is not None:
        monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    seen = []

    def apply(*args, **kwargs):
        seen.append(kwargs["interconnect_mode"])
        return profile.apply_revolta_profile(*args, **kwargs)

    run_engineargs_early_policy(
        profile, monkeypatch, apply, disable_custom_all_reduce=disabled
    )
    assert seen == [expected]
    assert ("VLLM_SM70_CAR_VMM_SAFE" in os.environ) == (expected != "no_p2p")


AUTO_LOGGED = "auto (resolved by workers)"
INTERCONNECT_CASES = [
    (None, False, "pex_p2p", AUTO_LOGGED),
    ("auto", False, "pex_p2p", AUTO_LOGGED),
    ("pex", False, "pex_p2p", "pex_p2p"),
    (" NCCL ", False, "no_p2p", "no_p2p"),
    ("nvlink", False, "nvlink", "nvlink"),
    ("nvlink", True, "no_p2p", "no_p2p"),
    ("auto", True, "no_p2p", "no_p2p"),
    ("bogus", False, "pex_p2p", "pending (resolved by workers)"),
]


@pytest.mark.parametrize(
    ("requested", "disabled", "expected", "logged"), INTERCONNECT_CASES
)
def test_frontend_interconnect_mode_and_audit(
    profile, monkeypatch, requested, disabled, expected, logged
):
    # The frontend knows only explicit choices; workers resolve auto/pex.
    if requested is not None:
        monkeypatch.setenv("VLLM_SM70_INTERCONNECT", requested)
    mode = profile.frontend_interconnect_mode(disable_custom_all_reduce=disabled)
    assert mode == expected
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, interconnect_mode=mode
    )
    assert ("VLLM_SM70_CAR_VMM_SAFE" in os.environ) == (expected != "no_p2p")
    audit = json.loads(profile.profile_audit(enabled=True, interconnect_mode=mode))
    # Name a mode only for an explicit choice: on an NVLink machine an auto
    # policy must not read "pex_p2p" while the workers log nvlink.
    assert audit["interconnect_mode"] == logged
    assert audit["interconnect_policy"] == ("auto" if requested is None else requested)
    assert len(audit["values"]) == 49


def _applied_profile(profile, monkeypatch, requested, disabled):
    with monkeypatch.context() as m:
        m.setattr(os, "environ", {})
        envs.disable_envs_cache()
        profile._PROFILE_VALUES.clear()
        profile._INITIAL_ENV.clear()
        if requested is not None:
            os.environ["VLLM_SM70_INTERCONNECT"] = requested
        mode = profile.frontend_interconnect_mode(disable_custom_all_reduce=disabled)
        assert profile.apply_revolta_profile(
            model(), 4, is_sm70=True, interconnect_mode=mode
        )
        audit = json.loads(profile.profile_audit(enabled=True, interconnect_mode=mode))
        environ = dict(os.environ)
    envs.disable_envs_cache()
    return mode, audit, environ


def test_audit_log_text_does_not_change_applied_values(profile, monkeypatch):
    # The logged mode is the only field that depends on whether the frontend
    # knows the interconnect: auto, pex and nvlink apply the same values and
    # sources, and so do the two ways of reaching no_p2p.
    by_mode: dict[str, list] = {}
    for requested, disabled, expected, logged in INTERCONNECT_CASES:
        mode, audit, environ = _applied_profile(
            profile, monkeypatch, requested, disabled
        )
        assert (mode, audit.pop("interconnect_mode")) == (expected, logged)
        audit.pop("interconnect_policy")
        environ.pop("VLLM_SM70_INTERCONNECT", None)
        applied = "no_p2p" if expected == "no_p2p" else "p2p"
        by_mode.setdefault(applied, []).append((audit, environ))
    for rows in by_mode.values():
        assert all(row == rows[0] for row in rows)
    assert by_mode["p2p"][0][0] != by_mode["no_p2p"][0][0]


def test_vllmconfig_profile_and_audit_receive_frontend_interconnect():
    # Both VllmConfig call sites must pass the same frontend assumption.
    import vllm.config.vllm as config_module

    assert config_module.__file__ is not None
    tree = ast.parse(Path(config_module.__file__).read_text())
    calls = [
        (node.func.id, node)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("apply_revolta_profile", "profile_audit")
    ]
    # The second profile call refines the graph reserve once the KV dtype
    # is resolved.
    assert sorted(name for name, _ in calls) == [
        "apply_revolta_profile",
        "apply_revolta_profile",
        "profile_audit",
    ]
    for _, call in calls:
        value = next(k.value for k in call.keywords if k.arg == "interconnect_mode")
        assert isinstance(value, ast.Call)
        assert isinstance(value.func, ast.Name)
        assert value.func.id == "frontend_interconnect_mode"
        (keyword,) = value.keywords
        assert keyword.arg == "disable_custom_all_reduce"
        assert ast.unparse(keyword.value) == (
            "self.parallel_config.disable_custom_all_reduce"
        )


def test_audit_logs_disabled_profile(profile):
    assert not profile.apply_revolta_profile(model(), 2, is_sm70=True)
    audit = json.loads(profile.profile_audit(enabled=False))
    assert not audit["enabled"]
    assert all(row["source"] == "upstream_default" for row in audit["values"].values())


GRAPH_RESERVE = "VLLM_V2_CUDAGRAPH_MEM_MIB"
TAIL_GRAPHS = "VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS"


@pytest.mark.parametrize(
    ("quantization", "method", "expected"),
    [
        ("awq", None, "700"),
        ("awq", "mtp", "700"),
        ("awq", "dflash", "1200"),
        ("modelopt_fp4", "mtp", "700"),
        ("modelopt_fp4", "dflash", "1200"),
    ],
)
def test_graph_reserve_follows_the_speculative_method(
    profile, quantization, method, expected
):
    assert profile.apply_revolta_profile(
        model(quantization), 4, is_sm70=True, speculative_method=method
    )
    assert os.environ[GRAPH_RESERVE] == expected
    row = json.loads(profile.profile_audit(enabled=True))["values"][GRAPH_RESERVE]
    assert row == {"value": expected, "source": "profile"}


@pytest.mark.parametrize(
    ("quantization", "method", "kv", "expected"),
    [
        ("awq", "dflash", None, "1200"),
        ("awq", "dflash", "auto", "1200"),
        ("awq", "dflash", "fp8_e4m3", "1200"),
        ("awq", "dflash", "fp8_per_token_head", "1000"),
        ("awq", "dflash", "int8", "1000"),
        ("awq", "mtp", "fp8_per_token_head", "600"),
        ("awq", "mtp", "int8", "600"),
        ("awq", "mtp", "fp8_e4m3", "700"),
        ("awq", None, "fp8_per_token_head", "700"),
        ("modelopt_fp4", "mtp", "fp8_per_token_head", "600"),
        ("modelopt_fp4", "mtp", "auto", "700"),
        ("modelopt_fp4", "dflash", "fp8_per_token_head", "1024"),
        ("modelopt_fp4", "dflash", "int8", "1024"),
        ("modelopt_fp4", "dflash", "fp8_e4m3", "1200"),
    ],
)
def test_graph_reserve_follows_the_kv_dtype(
    profile, quantization, method, kv, expected
):
    assert profile.apply_revolta_profile(
        model(quantization),
        4,
        is_sm70=True,
        speculative_method=method,
        kv_cache_dtype=kv,
    )
    assert os.environ[GRAPH_RESERVE] == expected


def test_resolved_kv_dtype_refines_the_reserve(profile):
    # EngineArgs and the first VllmConfig call run before the KV dtype is
    # resolved; the second VllmConfig call lowers only the profile value.
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="dflash"
    )
    assert os.environ[GRAPH_RESERVE] == "1200"
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="dflash",
        kv_cache_dtype="fp8_per_token_head",
    )
    assert os.environ[GRAPH_RESERVE] == "1000"
    assert os.environ[TAIL_GRAPHS] == "0"


def test_explicit_reserve_wins_over_the_kv_tier(profile, monkeypatch):
    monkeypatch.setenv(GRAPH_RESERVE, "1300")
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="dflash",
        kv_cache_dtype="int8",
    )
    assert os.environ[GRAPH_RESERVE] == "1300"


@pytest.mark.parametrize("override", [None, "620"])
def test_mtp_dynamic_schedule_refines_the_planar_kv_reserve(
    profile, monkeypatch, override
):
    if override is not None:
        monkeypatch.setenv(GRAPH_RESERVE, override)
    target = model()
    assert profile.apply_revolta_profile(
        target,
        4,
        is_sm70=True,
        speculative_method="mtp",
        kv_cache_dtype="fp8_per_token_head",
    )
    assert os.environ[GRAPH_RESERVE] == (override or "600")
    assert profile.apply_revolta_profile(
        target,
        4,
        is_sm70=True,
        speculative_method="mtp",
        kv_cache_dtype="fp8_per_token_head",
        num_speculative_tokens_per_batch_size=[(1, 3, 4), (4, 8, 3)],
    )
    assert os.environ[GRAPH_RESERVE] == (override or "705")


@pytest.mark.parametrize(
    ("quantization", "num_drafts", "schedule", "expected"),
    [
        ("awq", 4, [(1, 3, 4), (4, 8, 3)], "705"),
        ("awq", 5, [(1, 3, 5), (4, 8, 4)], "800"),
        # The verify graphs reach the global draft count whatever the
        # schedule lists, so a K5 model with a K4 schedule still needs 800.
        ("awq", 5, [(1, 8, 4)], "800"),
        ("modelopt_fp4", 4, [(1, 3, 4), (4, 8, 3)], "705"),
        ("modelopt_fp4", 5, [(1, 3, 5), (4, 8, 4)], "800"),
    ],
)
def test_mtp_dynamic_schedule_widens_the_reserve_at_five_drafts(
    profile, quantization, num_drafts, schedule, expected
):
    # NVFP4 + MTP5 planar DSD measured 0.73 GiB per rank on TP4.
    assert profile.apply_revolta_profile(
        model(quantization),
        4,
        is_sm70=True,
        speculative_method="mtp",
        kv_cache_dtype="fp8_per_token_head",
        num_speculative_tokens_per_batch_size=schedule,
        num_speculative_tokens=num_drafts,
    )
    assert os.environ[GRAPH_RESERVE] == expected


def test_wide_mtp_schedule_keeps_the_explicit_reserve(profile, monkeypatch):
    monkeypatch.setenv(GRAPH_RESERVE, "750")
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="mtp",
        kv_cache_dtype="fp8_per_token_head",
        num_speculative_tokens_per_batch_size=[(1, 3, 5), (4, 8, 4)],
        num_speculative_tokens=5,
    )
    assert os.environ[GRAPH_RESERVE] == "750"


@pytest.mark.parametrize(
    ("quantization", "latency_ms", "expected"),
    [
        ("awq", 0, "1000"),
        ("awq", 250, "1150"),
        ("awq", 500, "1150"),
        # The target adds prefill-width graphs only to the AWQ DFlash planar
        # pool we measured (1094/1094/1074/1094 MiB); every other tier is
        # unchanged until measured.
        ("modelopt_fp4", 0, "1024"),
        ("modelopt_fp4", 250, "1024"),
    ],
)
def test_graph_reserve_follows_the_mixed_prefill_latency_target(
    profile, quantization, latency_ms, expected
):
    assert profile.apply_revolta_profile(
        model(quantization),
        4,
        is_sm70=True,
        speculative_method="dflash",
        kv_cache_dtype="fp8_per_token_head",
        mixed_prefill_step_latency_ms=latency_ms,
    )
    assert os.environ[GRAPH_RESERVE] == expected


def test_mixed_prefill_target_only_applies_to_planar_dflash(profile):
    # Non-planar AWQ DFlash keeps 1200 (measured pool about 1.13 GiB), and
    # MTP's planar tier ignores the target.
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="dflash",
        kv_cache_dtype="fp8_e4m3",
        mixed_prefill_step_latency_ms=250,
    )
    assert os.environ[GRAPH_RESERVE] == "1200"
    os.environ.pop(GRAPH_RESERVE)
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="mtp",
        kv_cache_dtype="int8",
        mixed_prefill_step_latency_ms=250,
    )
    assert os.environ[GRAPH_RESERVE] == "600"


def test_mixed_prefill_target_refines_the_awq_dflash_planar_reserve(profile):
    # EngineArgs and the first VllmConfig call run before the scheduler target
    # is passed; the second VllmConfig call raises only the profile value.
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="dflash",
        kv_cache_dtype="int8",
    )
    assert os.environ[GRAPH_RESERVE] == "1000"
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="dflash",
        kv_cache_dtype="int8",
        mixed_prefill_step_latency_ms=250,
    )
    assert os.environ[GRAPH_RESERVE] == "1150"
    row = json.loads(profile.profile_audit(enabled=True))["values"][GRAPH_RESERVE]
    assert row == {"value": "1150", "source": "profile"}


def test_dflash_leaves_every_other_default_alone(profile):
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="dflash"
    )
    dflash = dict(os.environ)
    assert dflash[SHARED_EXPERT_OVERLAP] == "1"
    os.environ.clear()
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="mtp"
    )
    assert os.environ[SHARED_EXPERT_OVERLAP] == "0"
    changed = {
        k
        for k in set(dflash) | set(os.environ)
        if k not in ("PYTEST_CURRENT_TEST", "VLLM_REVOLTA_PROFILE_PROVENANCE")
        and dflash.get(k) != os.environ.get(k)
    }
    assert changed == {GRAPH_RESERVE, TAIL_GRAPHS, SHARED_EXPERT_OVERLAP}


def test_later_call_refines_only_the_profile_value(profile, monkeypatch):
    # EngineArgs without the method, then VllmConfig with the resolved one.
    assert profile.apply_revolta_profile(model(), 4, is_sm70=True)
    assert os.environ[GRAPH_RESERVE] == "700"
    assert os.environ[SHARED_EXPERT_OVERLAP] == "0"
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="dflash"
    )
    assert os.environ[GRAPH_RESERVE] == "1200"
    assert os.environ[SHARED_EXPERT_OVERLAP] == "1"
    row = json.loads(profile.profile_audit(enabled=True))["values"][GRAPH_RESERVE]
    assert row == {"value": "1200", "source": "profile"}


@pytest.mark.parametrize(
    ("allowed", "override"), [(False, None), (True, "0"), (False, "1")]
)
def test_awq_dflash_shared_expert_overlap_gate(profile, monkeypatch, allowed, override):
    if override is not None:
        monkeypatch.setenv(SHARED_EXPERT_OVERLAP, override)
    assert profile.apply_revolta_profile(
        model(),
        4,
        is_sm70=True,
        speculative_method="dflash",
        allow_shared_expert_overlap=allowed,
    )
    row = json.loads(profile.profile_audit(enabled=True))["values"][
        SHARED_EXPERT_OVERLAP
    ]
    assert row == {
        "value": override or "0",
        "source": "environment" if override is not None else "profile",
    }


def test_explicit_graph_reserve_wins_over_dflash(profile, monkeypatch):
    monkeypatch.setenv(GRAPH_RESERVE, "900")
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="dflash"
    )
    assert os.environ[GRAPH_RESERVE] == "900"
    row = json.loads(profile.profile_audit(enabled=True))["values"][GRAPH_RESERVE]
    assert row == {"value": "900", "source": "environment"}


@pytest.mark.parametrize(
    ("quantization", "method", "expected"),
    [
        ("awq", None, None),
        ("awq", "mtp", None),
        ("awq", "dflash", "0"),
        ("modelopt_fp4", "mtp", None),
        ("modelopt_fp4", "dflash", "0"),
    ],
)
def test_tail_graphs_are_off_only_for_dflash(profile, quantization, method, expected):
    assert profile.apply_revolta_profile(
        model(quantization), 4, is_sm70=True, speculative_method=method
    )
    assert os.environ.get(TAIL_GRAPHS) == expected


def test_explicit_tail_graphs_win_over_dflash(profile, monkeypatch):
    monkeypatch.setenv(TAIL_GRAPHS, "1")
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="dflash"
    )
    assert os.environ[TAIL_GRAPHS] == "1"


def test_engineargs_profile_receives_the_speculative_method(profile, monkeypatch):
    seen = []

    def apply(*args, **kwargs):
        seen.append(kwargs["speculative_method"])
        return profile.apply_revolta_profile(*args, **kwargs)

    run_engineargs_early_policy(profile, monkeypatch, apply)
    assert seen == ["mtp"]
    assert os.environ[GRAPH_RESERVE] == "700"


def test_vllmconfig_profile_receives_the_speculative_method():
    import vllm.config.vllm as config_module

    assert config_module.__file__ is not None
    tree = ast.parse(Path(config_module.__file__).read_text())
    first, refine = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_revolta_profile"
    ]
    for call in (first, refine):
        value = next(k.value for k in call.keywords if k.arg == "speculative_method")
        assert ast.unparse(value) == (
            "getattr(self.speculative_config, 'method', None)"
        )
    refined = {k.arg: ast.unparse(k.value) for k in refine.keywords}
    assert refined["kv_cache_dtype"] == "self.cache_config.cache_dtype"
    schedule_field = "num_speculative_tokens_per_batch_size"
    assert refined[schedule_field] == (
        f"getattr(self.speculative_config, '{schedule_field}', None)"
    )
    assert refined["mixed_prefill_step_latency_ms"] == (
        "self.scheduler_config.mixed_prefill_step_latency_ms"
    )
    assert "kv_cache_dtype" not in {k.arg for k in first.keywords}
    assert "mixed_prefill_step_latency_ms" not in {k.arg for k in first.keywords}


def test_speculative_draft_keeps_target_graph_reserve(profile, monkeypatch, tmp_path):
    from transformers import Qwen3Config

    from vllm import revolta_profile
    from vllm.config import CacheConfig, ModelConfig, VllmConfig, replace

    def apply(_model, _tp, **kwargs):
        return profile.apply_revolta_profile(
            model(),
            4,
            is_sm70=True,
            speculative_method="mtp",
            kv_cache_dtype=kwargs.get("kv_cache_dtype"),
        )

    monkeypatch.setattr(revolta_profile, "apply_revolta_profile", apply)
    apply_kernel = Mock(wraps=revolta_profile.apply_revolta_kernel_profile)
    monkeypatch.setattr(revolta_profile, "apply_revolta_kernel_profile", apply_kernel)
    Qwen3Config(architectures=["Qwen3ForCausalLM"]).save_pretrained(tmp_path)
    target = VllmConfig(
        model_config=ModelConfig(
            model=str(tmp_path), dtype="float16", skip_tokenizer_init=True
        ),
        cache_config=CacheConfig(cache_dtype="int8"),
    )
    assert os.environ[GRAPH_RESERVE] == "600"

    draft = replace(
        target,
        is_speculative_draft=True,
        cache_config=replace(target.cache_config, cache_dtype="float16"),
    )
    assert draft.cache_config.cache_dtype == "float16"
    assert os.environ[GRAPH_RESERVE] == "600"
    apply_kernel.assert_called_once()
