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

# Independent contract: do not derive expected values from implementation tables.
COMMON = {
    "VLLM_SM70_QWEN38_WIDE_M": "1",
    "VLLM_SM70_PLE_PREFILL_CONV_CHUNK": "1",
    "VLLM_SM70_QWEN38_GDN_NO_PACK": "1",
    "VLLM_SM70_GREEDY_VERIFY": "1",
    "VLLM_SM70_SKIP_REJECTED_BATCH_PACKS": "1",
    "VLLM_SM70_QSA_SHARED_KEY": "1",
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
        hf_text_config=SimpleNamespace(
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


def test_qsa_sparse_layer_spelling_and_unrelated_name(profile):
    config = model()
    config.architectures = ["AnotherQSAModel"]
    config.hf_text_config.layer_types = ["qwen_sparse_attention"]
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
    assert len(audit["values"]) == 45
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


def test_engineargs_profile_precedes_actual_mtp_defaults(profile, monkeypatch):
    import vllm.revolta_profile as installed
    from vllm.config.compilation import CompilationConfig
    from vllm.engine.arg_utils import EngineArgs

    monkeypatch.setattr(
        installed, "apply_revolta_profile", profile.apply_revolta_profile
    )
    monkeypatch.setattr(installed, "all_visible_devices_sm70", lambda: True)
    # Execute the real early sequence, including the real MTP default method.
    import vllm.engine.arg_utils as arg_utils

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
        if isinstance(n, ast.Expr)
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
    )
    target = model()
    target.hf_text_config.mtp_num_hidden_layers = 1
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
    assert args.compilation_config.fast_moe_cold_start is None
    assert args.compilation_config.cudagraph_capture_sizes is None


def test_audit_logs_disabled_profile(profile):
    assert not profile.apply_revolta_profile(model(), 2, is_sm70=True)
    audit = json.loads(profile.profile_audit(enabled=False))
    assert not audit["enabled"]
    assert all(row["source"] == "upstream_default" for row in audit["values"].values())
