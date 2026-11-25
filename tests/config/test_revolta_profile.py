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


@pytest.mark.parametrize("allocator", ["PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"])
def test_hybrid_serving_environment_overrides(profile, monkeypatch, allocator):
    overrides = {
        allocator: "expandable_segments:False",
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
    assert "NCCL_GRAPH_REGISTER" not in os.environ
    assert {
        name
        for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF")
        if name in os.environ
    } == {allocator}


def test_hybrid_serving_allocator_leaves_initialized_cuda_alone(profile, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    assert profile.apply_revolta_profile(
        model(), 4, is_sm70=True, hybrid_serving=True, speculative_method="dflash"
    )
    assert "PYTORCH_ALLOC_CONF" not in os.environ
    assert "PYTORCH_CUDA_ALLOC_CONF" not in os.environ


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    ("method", "active", "serving", "shortcut", "flags", "spec_options"),
    [
        ("mtp", True, True, False, [], {}),
        ("mtp", True, True, True, [], {}),
        (None, True, True, False, [], {}),
        ("mtp", True, True, False, ["--no-enable-prefix-caching"], {}),
        (
            "mtp",
            True,
            True,
            False,
            ["--enable-prefix-caching", "--mamba-cache-mode", "all"],
            {},
        ),
        (
            "mtp",
            True,
            True,
            False,
            [
                "--max-num-seqs",
                "4",
                "--max-num-batched-tokens",
                "4096",
                "--no-async-scheduling",
            ],
            {"num_speculative_tokens_per_batch_size": [[1, 4, 2]]},
        ),
        ("mtp", True, True, False, [], {"num_speculative_tokens": 3}),
        ("mtp", True, True, False, ["--async-scheduling"], {}),
        ("mtp", True, True, False, [], {"gdn_state_replay": False}),
        ("dflash", True, True, False, [], {}),
        (
            "dflash",
            True,
            True,
            False,
            [
                "--long-prefill-token-threshold",
                "0",
                "--kernel-config",
                '{"qsa_dcp_index_shard":false}',
            ],
            {"sm70_dflash2": {"spec_state_replay": False}},
        ),
        (
            "dflash",
            True,
            True,
            False,
            [],
            {"sm70_dflash2": {"fused_gdn_verify": True}},
        ),
        (
            "dflash",
            True,
            True,
            False,
            ["--decode-context-parallel-size", "1"],
            {},
        ),
        (
            "dflash",
            True,
            True,
            False,
            ["--async-scheduling"],
            {"dflash2_lookup": {"enabled": False}},
        ),
        ("dflash", True, True, False, [], {"dflash2_lookup": None}),
        (
            "dflash",
            True,
            True,
            False,
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
        ),
        ("dflash", True, True, False, [], {"num_speculative_tokens": 4}),
        ("mtp", True, False, False, [], {}),
        ("mtp", False, True, False, [], {}),
    ],
)
def test_engineargs_revolta_hybrid_serving_defaults(
    profile,
    monkeypatch,
    tmp_path,
    method,
    active,
    serving,
    shortcut,
    flags,
    spec_options,
):
    from transformers import Qwen3Config

    from vllm import revolta_profile
    from vllm.config import speculative
    from vllm.config.speculative_kv import dflash_draft_cache_dtype
    from vllm.engine import arg_utils
    from vllm.models.qwen4_exp.config import Qwen4ExpConfig
    from vllm.models.qwen4_exp.nvidia.ops import qsa
    from vllm.platforms import current_platform
    from vllm.platforms.cpu import CpuPlatform
    from vllm.platforms.interface import DeviceCapability
    from vllm.usage.usage_lib import UsageContext
    from vllm.utils.argparse_utils import FlexibleArgumentParser
    from vllm.v1.attention.backends import fa_utils

    model_dir = tmp_path / "target"
    Qwen4ExpConfig(
        architectures=["Qwen4ExpForCausalLM"],
        text_config={
            **vars(model().hf_text_config),
            "num_hidden_layers": 2,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "mtp_num_hidden_layers": 1,
            "max_position_embeddings": 524288,
        },
    ).save_pretrained(model_dir)
    raw = {
        "method": method,
        "num_speculative_tokens": 7 if method == "dflash" else 4,
        **spec_options,
    }
    if method == "dflash":
        draft = tmp_path / "draft"
        Qwen3Config(
            architectures=["DFlash2DraftModel"],
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=524288,
            dflash_config={
                "selector_top_k": 16,
                "block_size": raw["num_speculative_tokens"] + 1,
                "target_feature_reduction": "hc_mean",
            },
        ).save_pretrained(draft)
        raw["model"] = str(draft)

    platform = CpuPlatform()
    monkeypatch.setattr(platform, "is_cpu", lambda: False)
    monkeypatch.setattr(platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(platform, "get_device_total_memory", lambda: 32 * 1024**3)
    monkeypatch.setattr(
        platform, "get_device_capability", lambda: DeviceCapability(7, 0)
    )
    monkeypatch.setattr(arg_utils, "current_platform", platform)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(
        current_platform,
        "get_device_capability",
        lambda *args, **kwargs: DeviceCapability(7, 0),
    )
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda cap, **kwargs: cap in (70, (7, 0)),
    )
    monkeypatch.setattr(current_platform, "support_static_graph_mode", lambda: True)
    monkeypatch.setattr(current_platform, "check_and_update_config", lambda _: None)
    monkeypatch.setattr(
        "vllm.config.vllm._any_participating_device_is_capability",
        lambda _, cap: cap == (7, 0),
    )
    monkeypatch.setattr("vllm.config.vllm.HAS_TRITON", True)
    monkeypatch.setattr(
        arg_utils.VllmConfig, "_get_quantization_config", lambda *_: None
    )
    monkeypatch.setattr(qsa, "qsa_e4m3_capability_reason", lambda _: None)
    monkeypatch.setattr(fa_utils, "is_flash_attn_varlen_func_available", lambda: False)
    monkeypatch.setattr(revolta_profile, "all_visible_devices_sm70", lambda: active)
    monkeypatch.setattr(
        revolta_profile, "apply_revolta_profile", profile.apply_revolta_profile
    )
    # Local checkpoint configs stand in for the quantized target and draft.
    monkeypatch.setattr(
        speculative,
        "ModelConfig",
        lambda **kwargs: arg_utils.ModelConfig(**(kwargs | {"quantization": None})),
    )
    parser = arg_utils.EngineArgs.add_cli_args(FlexibleArgumentParser())
    spec_flags = []
    if shortcut:
        spec_flags = ["--spec-method", method, "--spec-tokens", "4"]
    elif method is not None:
        spec_flags = ["--speculative-config", json.dumps(raw)]
    args = arg_utils.EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--model",
                str(model_dir),
                "--dtype",
                "float16",
                "--tensor-parallel-size",
                "4",
                "--decode-context-parallel-size",
                "2",
                "--gpu-memory-utilization",
                "0.956",
                "--distributed-executor-backend",
                "mp",
                *spec_flags,
                *flags,
            ]
        )
    )
    target = args.create_model_config()
    target.quantization = "awq"
    target.hf_config.quantization_config = model().hf_config.quantization_config
    assert target.is_hybrid
    if "all" in flags:
        # Exercise an explicit mode supported by the model validator.
        monkeypatch.setattr(
            type(target), "supports_mamba_prefix_caching", property(lambda self: True)
        )
    monkeypatch.setattr(args, "create_model_config", lambda: target)

    context = UsageContext.OPENAI_API_SERVER if serving else UsageContext.LLM_CLASS
    config = args.create_engine_config(usage_context=context)
    assert envs.VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS is active
    spec = config.speculative_config
    assert getattr(spec, "method", None) == method
    if method == "mtp" and active:
        assert "draft_sample_method" not in args.speculative_config
        assert "attention_backend" not in args.speculative_config
    enabled = active and serving
    dcp = 1 if "--decode-context-parallel-size" in flags else 2
    assert config.kernel_config.qsa_dcp_index_shard is (
        enabled and method == "dflash" and dcp == 2 and "--kernel-config" not in flags
    )
    for name in (
        "VLLM_SM70_QSA_GROUPED_LPT",
        "VLLM_SM70_GDN_SKINNY_MPLAN",
        "VLLM_SM70_AWQ_MOE_INT4_ZERO",
    ):
        assert os.environ.get(name) == ("1" if enabled else None)
    assert os.environ.get("VLLM_SM70_DFLASH2_FUSED_QKV_PACK") == (
        "1" if enabled and method == "dflash" else None
    )
    for name in ("PYTORCH_ALLOC_CONF", "PYTORCH_CUDA_ALLOC_CONF"):
        assert os.environ.get(name) == ("expandable_segments:True" if enabled else None)
    assert os.environ.get("NCCL_GRAPH_REGISTER") == ("0" if enabled else None)
    prefix = (serving or target.is_prefix_caching_supported) and (
        "--no-enable-prefix-caching" not in flags
    )
    assert config.cache_config.enable_prefix_caching is prefix
    assert config.cache_config.mamba_cache_mode == (
        "all" if "all" in flags else "align" if prefix else "none"
    )
    scheduler = config.scheduler_config
    assert scheduler.long_prefill_token_threshold == (
        4896 if enabled and "--long-prefill-token-threshold" not in flags else 0
    )
    if enabled:
        assert (scheduler.max_num_seqs, scheduler.max_num_batched_tokens) == (
            (4, 4096) if "--max-num-seqs" in flags else (8, 8192)
        )
        assert config.cache_config.cache_dtype == "int8"
        assert (
            config.parallel_config.tensor_parallel_size,
            config.parallel_config.decode_context_parallel_size,
        ) == (4, dcp)
        assert config.use_v2_model_runner
        assert config.model_config.max_model_len == 524288
        assert config.cache_config.gpu_memory_utilization == 0.956
        assert config.cache_config.prefix_cache_retention_interval == 0
        assert scheduler.enable_chunked_prefill
    else:
        assert scheduler.max_num_seqs == 256
        assert scheduler.max_num_batched_tokens == (2048 if serving else 8192)
    dsd = spec_options.get("num_speculative_tokens_per_batch_size")
    if enabled and not (
        spec_options.keys()
        & {"num_speculative_tokens", "num_speculative_tokens_per_batch_size"}
    ):
        if method == "dflash":
            dsd = [(1, 3, 7), (4, 4, 3), (5, 8, 4)]
        elif method == "mtp" and prefix and "all" not in flags:
            dsd = [(1, 3, 4), (4, 8, 3)]
    assert getattr(spec, "num_speculative_tokens_per_batch_size", None) == (
        [tuple(row) for row in dsd] if dsd is not None else None
    )
    mtp_replay = spec_options.get(
        "gdn_state_replay", True if enabled and prefix and "all" not in flags else None
    )
    default_sync = (method == "dflash" and dsd is not None) or (
        method == "mtp" and mtp_replay is True
    )
    assert scheduler.async_scheduling is (
        "--async-scheduling" in flags
        or ("--no-async-scheduling" not in flags and not default_sync)
    )
    if method == "dflash":
        assert dflash_draft_cache_dtype(config) == "auto"
        assert spec.draft_model_config.dtype == torch.float16
        policy_options = spec_options.get("sm70_dflash2", {})
        assert spec.sm70_dflash2.spec_state_replay is policy_options.get(
            "spec_state_replay",
            True if enabled and not policy_options.get("fused_gdn_verify") else None,
        )
        if policy_options.get("fused_gdn_verify"):
            assert spec.sm70_dflash2.fused_gdn_verify
            assert not spec.gdn_spec_state_replay
        lookup = spec.resolved_dflash2_lookup
        if (spec_options.get("dflash2_lookup") or {}).get("enabled", False):
            assert (lookup.mode, lookup.nmax, lookup.max_requests) == (
                "gpu_head",
                10,
                2,
            )
        elif (
            "dflash2_lookup" not in spec_options and raw["num_speculative_tokens"] == 7
        ):
            assert (lookup.mode, lookup.policy, lookup.nmin, lookup.nmax) == (
                "gpu_head",
                "longest",
                5,
                16,
            )
            assert (
                lookup.search_window,
                lookup.gate,
                lookup.collect_stats,
                lookup.max_requests,
            ) == (65536, "first_token", True, 1)
        else:
            assert lookup.mode == "off"
    elif method == "mtp":
        assert spec.kv_cache_dtype is None  # Inherits target INT8.
        assert spec.gdn_state_replay is mtp_replay
        assert spec.gdn_spec_state_replay is (
            mtp_replay is True and not scheduler.async_scheduling
        )
    if enabled and not flags and not spec_options:
        assert config.compilation_config.compile_ranges_endpoints == [8193]
        if method == "mtp":
            assert scheduler.prefill_pressure_token_budget == 448
            assert scheduler.mixed_prefill_step_latency_ms == 0
            assert os.environ[GRAPH_RESERVE] == "705"
            assert config.compilation_config.cudagraph_capture_sizes == list(
                range(5, 41, 5)
            )
        elif method == "dflash":
            assert scheduler.prefill_pressure_token_budget == 448
            assert scheduler.mixed_prefill_step_latency_ms == 0
            assert os.environ[GRAPH_RESERVE] == "1000"
            assert spec.gdn_spec_state_replay
            assert spec.gdn_deferred_commit_metadata is False
            assert config.compilation_config.cudagraph_capture_sizes == [
                1,
                2,
                4,
                8,
                9,
                16,
                18,
                24,
                32,
                40,
                48,
                56,
                64,
            ]


@pytest.mark.parametrize(
    ("quantization", "method", "source", "overrides", "expected"),
    [
        ("awq", "dflash", "cli", {}, (448, 1, 0)),
        ("modelopt_fp4", "mtp", "yaml", {}, (448, 1, 0)),
        ("awq", "qwen4_exp_mtp", "json", {}, (448, 1, 0)),
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
    assert "kv_cache_dtype" not in {k.arg for k in first.keywords}


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
