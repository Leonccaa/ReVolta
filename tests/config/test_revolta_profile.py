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
    elif quantization == "modelopt_fp4":
        expected.update(
            {
                "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE": "1",
                "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE": "1",
                "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE": "1",
                "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": "0",
                "VLLM_SM70_NVFP4_TURBOMIND": "1",
                "VLLM_SM70_FP8_TURBOMIND": "1",
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
    assert len(audit["values"]) == 46
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


def run_engineargs_early_policy(profile, monkeypatch, apply, **engine_kwargs):
    """Execute EngineArgs' real sequence from the profile import to MTP defaults."""
    import vllm.revolta_profile as installed
    from vllm.config.compilation import CompilationConfig
    from vllm.engine.arg_utils import EngineArgs

    monkeypatch.setattr(installed, "apply_revolta_profile", apply)
    monkeypatch.setattr(
        installed, "frontend_interconnect_mode", profile.frontend_interconnect_mode
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
        **engine_kwargs,
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
    return args


def test_engineargs_profile_precedes_actual_mtp_defaults(profile, monkeypatch):
    args = run_engineargs_early_policy(
        profile, monkeypatch, profile.apply_revolta_profile
    )
    assert args.compilation_config.fast_moe_cold_start is None
    assert args.compilation_config.cudagraph_capture_sizes is None


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
    assert len(audit["values"]) == 46


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
    assert sorted(name for name, _ in calls) == [
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
        ("modelopt_fp4", "dflash", "700"),
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


def test_dflash_leaves_every_other_default_alone(profile):
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="dflash"
    )
    dflash = dict(os.environ)
    os.environ.clear()
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="mtp"
    )
    changed = {
        k
        for k in set(dflash) | set(os.environ)
        if k not in ("PYTEST_CURRENT_TEST", "VLLM_REVOLTA_PROFILE_PROVENANCE")
        and dflash.get(k) != os.environ.get(k)
    }
    assert changed == {GRAPH_RESERVE, TAIL_GRAPHS}


def test_later_call_refines_only_the_profile_value(profile, monkeypatch):
    # EngineArgs without the method, then VllmConfig with the resolved one.
    assert profile.apply_revolta_profile(model(), 4, is_sm70=True)
    assert os.environ[GRAPH_RESERVE] == "700"
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, speculative_method="dflash"
    )
    assert os.environ[GRAPH_RESERVE] == "1200"
    row = json.loads(profile.profile_audit(enabled=True))["values"][GRAPH_RESERVE]
    assert row == {"value": "1200", "source": "profile"}


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
        ("modelopt_fp4", "dflash", None),
    ],
)
def test_tail_graphs_are_off_only_for_awq_dflash(
    profile, quantization, method, expected
):
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
    (call,) = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "apply_revolta_profile"
    ]
    value = next(k.value for k in call.keywords if k.arg == "speculative_method")
    assert ast.unparse(value) == "getattr(self.speculative_config, 'method', None)"
