# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.config.kernel import KernelConfig


@pytest.mark.parametrize("policy", ["sm70_skinny_moe", "fused_fp16_aux_gemv"])
def test_unused_policy_does_not_partition_compile_cache(policy):
    enabled = KernelConfig()
    disabled = KernelConfig(**{policy: False})
    assert enabled.compute_hash() == disabled.compute_hash()


@pytest.mark.parametrize("policy", ["sm70_skinny_moe", "fused_fp16_aux_gemv"])
def test_loaded_policy_keeps_enabled_and_disabled_graphs_distinct(policy):
    enabled = KernelConfig()
    disabled = KernelConfig(**{policy: False})
    setattr(enabled, f"{policy}_applicable", True)
    setattr(disabled, f"{policy}_applicable", True)
    assert enabled.compute_hash() != disabled.compute_hash()


@pytest.mark.parametrize("enabled", [False, True])
def test_junction_evidence_default_and_toggle_keep_compile_identity(
    enabled, monkeypatch
):
    import vllm.config.kernel as kernel

    config = KernelConfig()
    config.long_context_kv.enabled = enabled
    assert config.long_context_kv.enable_junction_evidence
    before = config.compute_hash()
    config.long_context_kv.enable_junction_evidence = False
    assert config.compute_hash() == before
    original = kernel.get_hash_factors

    def old_fields(*args, **kwargs):
        factors = original(*args, **kwargs)
        if "long_context_kv" not in factors:
            return factors
        name, items = factors["long_context_kv"]
        factors["long_context_kv"] = (
            name,
            tuple(item for item in items if item[0] != "enable_junction_evidence"),
        )
        return factors

    monkeypatch.setattr(kernel, "get_hash_factors", old_fields)
    assert config.compute_hash() == before
