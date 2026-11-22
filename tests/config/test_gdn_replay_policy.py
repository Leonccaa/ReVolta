# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Replay/packed precedence through real model and engine configuration (CPU)."""

import json
import os
from dataclasses import asdict

import pytest

from vllm import envs
from vllm.config import (
    CacheConfig,
    DeviceConfig,
    KernelConfig,
    ModelConfig,
    ParallelConfig,
    SchedulerConfig,
    SpeculativeConfig,
    VllmConfig,
)
from vllm.config.sm70_dflash2 import Sm70DFlash2Config

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


@pytest.fixture
def engine_config(monkeypatch, tmp_path):
    import vllm.config.vllm as config_module
    from vllm.platforms import current_platform

    monkeypatch.setattr(config_module.logger, "info_once", config_module.logger.info)
    monkeypatch.setattr(config_module.logger, "propagate", True)
    import logging

    monkeypatch.setattr(logging.getLogger("vllm"), "propagate", True)

    # Only device discovery is simulated; policy qualification and both config
    # constructors run normally, with local configuration-only checkpoints.
    monkeypatch.setattr(os, "environ", dict(os.environ))
    for name in list(os.environ):
        if name.startswith("VLLM_"):
            monkeypatch.delenv(name)
    for name in envs.environment_variables:
        monkeypatch.delitem(vars(envs), name, raising=False)
    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_TARGET_DEVICE", "cpu")
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    monkeypatch.setenv("VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", "0")
    monkeypatch.setenv("VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", "1")
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    monkeypatch.setattr(current_platform, "device_count", lambda: 4)
    monkeypatch.setattr(
        current_platform,
        "is_device_capability",
        lambda capability, device_id=0: capability in (70, (7, 0)),
    )

    def build(
        hidden_size=5120,
        *,
        method="dflash",
        replay=None,
        packed=None,
        long_context=False,
        gdn_replay=None,
        replay_graph=None,
        legacy_graph=True,
        dynamic_schedule=None,
        dp_size=1,
    ):
        target = tmp_path / "target"
        target.mkdir(exist_ok=True)
        (target / "config.json").write_text(
            json.dumps(
                {
                    "architectures": [
                        "Qwen4ExpForConditionalGeneration"
                        if hidden_size == 2560
                        else "Qwen3_5ForConditionalGeneration"
                    ],
                    "model_type": "qwen4_exp" if hidden_size == 2560 else "qwen3_5",
                    "text_config": {
                        "model_type": "qwen4_exp_text"
                        if hidden_size == 2560
                        else "qwen3_5_text",
                        "hidden_size": hidden_size,
                        "num_hidden_layers": 4,
                        "num_attention_heads": 24,
                        "num_key_value_heads": 2 if hidden_size == 2560 else 4,
                        "head_dim": 256,
                        "max_position_embeddings": 524288,
                        "vocab_size": 128,
                        "mtp_num_hidden_layers": 1,
                        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
                    },
                }
            )
        )
        draft = tmp_path / "draft"
        draft.mkdir(exist_ok=True)
        (draft / "config.json").write_text(
            json.dumps(
                {
                    "architectures": ["DFlash2DraftModel"],
                    "model_type": "qwen3",
                    "hidden_size": hidden_size,
                    "num_hidden_layers": 1,
                    "num_attention_heads": 24,
                    "num_key_value_heads": 4,
                    "head_dim": 256,
                    "max_position_embeddings": 524288,
                    "vocab_size": 128,
                    "dflash_config": {
                        "selector_top_k": 16,
                        "target_feature_reduction": "hc_mean",
                    },
                }
            )
        )
        model = ModelConfig(model=str(target), dtype="float16", max_model_len=524288)
        parallel = ParallelConfig(
            tensor_parallel_size=4,
            decode_context_parallel_size=2 if hidden_size == 2560 else 1,
            distributed_executor_backend="mp",
            data_parallel_size=dp_size,
        )
        spec = (
            None
            if method is None
            else SpeculativeConfig(
                target_model_config=model,
                target_parallel_config=parallel,
                model=str(draft) if method == "dflash" else None,
                method=method,
                num_speculative_tokens=7 if method == "dflash" else 4,
                num_speculative_tokens_per_batch_size=dynamic_schedule,
                draft_sample_method="greedy",
                gdn_state_replay=gdn_replay,
                gdn_state_replay_graph=replay_graph,
                sm70_dflash2=Sm70DFlash2Config(
                    spec_state_replay=replay,
                    spec_state_replay_graph=legacy_graph,
                    fused_gdn_verify=packed,
                ),
            )
        )
        return VllmConfig(
            kernel_config=KernelConfig(
                long_context_kv={"enabled": long_context, "host_pool_gib": 128}
            ),
            model_config=model,
            speculative_config=spec,
            parallel_config=parallel,
            device_config=DeviceConfig(device="cpu"),
            cache_config=CacheConfig(
                enable_prefix_caching=True, mamba_cache_mode="align"
            ),
            scheduler_config=SchedulerConfig(
                max_num_seqs=8,
                max_num_batched_tokens=8192,
                max_model_len=524288,
                is_encoder_decoder=model.is_encoder_decoder,
                async_scheduling=False,
            ),
        )

    yield build
    envs.disable_envs_cache()


def test_dynamic_mtp_uses_fixed_k_with_data_parallelism(engine_config):
    schedule = [[1, 3, 4], [4, 8, 3]]
    single = engine_config(2560, method="mtp", dynamic_schedule=schedule)
    assert single.speculative_config.num_speculative_tokens_per_batch_size == [
        (1, 3, 4),
        (4, 8, 3),
    ]
    parallel = engine_config(2560, method="mtp", dynamic_schedule=schedule, dp_size=2)
    assert parallel.speculative_config.num_speculative_tokens_per_batch_size is None
    assert parallel.num_speculative_tokens == 4


def test_dynamic_mtp_explicit_empty_schedule_is_invalid(engine_config):
    with pytest.raises(ValueError, match="must not be empty"):
        engine_config(2560, method="mtp", dynamic_schedule=[])


def test_r1_threshold_worker_transport(engine_config):
    from pydantic import TypeAdapter

    from vllm.config import replace

    default = engine_config(
        2560, method="mtp", dynamic_schedule=[[1, 8, 4]]
    ).speculative_config
    assert default.exact_end_draft_threshold is None
    assert "exact_end_draft_threshold" not in TypeAdapter(
        SpeculativeConfig
    ).dump_python(default)
    spec = replace(default, exact_end_draft_threshold=2)
    rebuilt = SpeculativeConfig.from_worker_payload(
        json.loads(json.dumps(spec.to_worker_payload())),
        target_model_config=spec.target_model_config,
        target_parallel_config=spec.target_parallel_config,
    )
    assert rebuilt.exact_end_draft_threshold == 2
    assert rebuilt.compute_hash() == default.compute_hash()


def test_explicit_replay_overrides_qualified_packed_default(engine_config):
    config = engine_config(replay=True)
    spec = config.speculative_config
    assert spec is not None
    assert spec.sm70_dflash2.qualified
    assert spec.gdn_spec_state_replay
    assert spec.sm70_dflash2.fused_gdn_verify is False


@pytest.mark.parametrize(
    "hidden_size,qualified,replay,packed",
    [
        (2560, False, True, False),
        (5120, True, False, True),
    ],
)
def test_default_replay_yields_only_to_enabled_packed_verifier(
    engine_config, caplog, hidden_size, qualified, replay, packed
):
    # Exercise the actual release default through both config constructors.
    spec = engine_config(hidden_size).speculative_config
    assert spec is not None
    assert spec.sm70_dflash2.resolved
    assert spec.sm70_dflash2.qualified is qualified
    assert spec.gdn_spec_state_replay is replay
    assert spec.sm70_dflash2.fused_gdn_verify is packed
    if packed:
        assert "spec_state_replay=None gives precedence" in caplog.text


def test_explicit_replay_logs_why_the_packed_default_was_disabled(
    engine_config, caplog
):
    engine_config(replay=True)
    assert "Disabling the automatic sm70_dflash2.fused_gdn_verify" in caplog.text
    assert "spec_state_replay=True was explicitly requested" in caplog.text


@pytest.mark.parametrize("legacy", [False, True])
def test_explicit_replay_and_explicit_packed_are_a_configuration_error(
    engine_config, monkeypatch, legacy
):
    if legacy:
        monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY", "1")
    with pytest.raises(
        ValueError, match=r"spec_state_replay=True.*fused_gdn_verify=True"
    ):
        engine_config(replay=True, packed=None if legacy else True)


@pytest.mark.parametrize(
    "replay,packed,expected",
    [
        (False, None, False),
        (False, False, False),
        (False, True, False),
        (True, False, True),
        (None, False, True),  # the ordinary route replays by default
        (None, True, False),
    ],
)
def test_explicit_choices_are_respected(engine_config, replay, packed, expected):
    spec = engine_config(replay=replay, packed=packed).speculative_config
    assert spec is not None
    assert spec.gdn_spec_state_replay is expected
    if packed is not None:
        assert spec.sm70_dflash2.fused_gdn_verify is packed


@pytest.mark.parametrize("method", [None, "mtp"])
@pytest.mark.parametrize("replay", [None, False, True])
def test_replay_disabled_keeps_compilation_configuration(engine_config, method, replay):
    config = engine_config(
        2560,
        method=method,
        replay=replay,
        packed=True if replay is True else None,
        gdn_replay=False,
    )
    spec = config.speculative_config
    assert spec is None or not spec.gdn_spec_state_replay
    if spec is not None:
        assert spec.sm70_dflash2.fused_gdn_verify is (replay is True)
    # Optional machine-readable evidence for paired baseline/branch runs.
    receipt_dir = os.environ.get("REPLAY_POLICY_RECEIPTS")
    if receipt_dir:
        from pathlib import Path

        payload = {
            "compilation": asdict(config.compilation_config),
            "graph_options": spec.sm70_dflash2.graph_options() if spec else None,
            "spec_hash": spec.compute_hash() if spec else None,
            "config_hash": config.compute_hash(),
            "environment": {
                k: v for k, v in os.environ.items() if k.startswith("VLLM_")
            },
        }
        Path(receipt_dir).mkdir(parents=True, exist_ok=True)
        Path(receipt_dir, f"{method}-{replay}.json").write_text(
            json.dumps(payload, sort_keys=True, default=str, indent=2) + "\n"
        )


def test_explicit_packed_false_wins_over_legacy_packed_true(engine_config, monkeypatch):
    monkeypatch.setenv("VLLM_SM70_DFLASH2_FUSED_GDN_VERIFY", "1")
    spec = engine_config(replay=True, packed=False).speculative_config
    assert spec is not None
    assert spec.gdn_spec_state_replay
    assert spec.sm70_dflash2.fused_gdn_verify is False


@pytest.mark.parametrize(
    "replay,expected,states", [(None, True, 29), (True, True, 29), (False, False, 71)]
)
def test_long_context_replay_default_and_capacity(
    engine_config, caplog, replay, expected, states, monkeypatch
):
    import vllm.config.vllm as config_module
    from tests.v1.core.long_context_kv_utils import production_config, production_specs
    from vllm.v1.core.kv_cache_utils import (
        get_kv_cache_groups,
        get_long_context_geometry,
    )

    monkeypatch.setattr(
        config_module.logger, "warning_once", config_module.logger.warning
    )
    config = engine_config(2560, replay=replay, long_context=True)
    assert config.speculative_config.gdn_spec_state_replay is expected
    assert config.speculative_config.sm70_dflash2.spec_state_replay is expected
    geometry_config = production_config()
    groups = get_kv_cache_groups(
        geometry_config,
        production_specs(
            speculative_blocks=config.speculative_config.num_speculative_state_blocks()
        ),
    )
    assert (
        get_long_context_geometry(geometry_config, groups).state_blocks_per_request
        == states
    )
    assert ("explicitly configured" in caplog.text) is (replay is False)


def test_long_context_replay_overrides_automatic_packed_default(engine_config):
    spec = engine_config(long_context=True).speculative_config
    assert spec.sm70_dflash2.qualified
    assert spec.gdn_spec_state_replay
    assert spec.sm70_dflash2.fused_gdn_verify is False


def test_long_context_replay_rejects_explicit_packed_conflict(engine_config):
    with pytest.raises(ValueError, match="cannot be enabled together"):
        engine_config(long_context=True, packed=True)


@pytest.mark.parametrize("replay", [None, False, True])
def test_mtp_replay_defaults_on_for_supported_route(engine_config, replay):
    spec = engine_config(2560, method="mtp", gdn_replay=replay).speculative_config
    assert spec.gdn_spec_state_replay is (replay is not False)
    assert spec.num_speculative_state_blocks() == (4 if replay is False else 0)
    # DFlash defaults do not become MTP input policy.
    assert spec.sm70_dflash2.spec_state_replay is None


@pytest.mark.parametrize("graph", [None, False, True])
def test_mtp_replay_worker_transport_and_graph_policy(engine_config, graph):
    spec = engine_config(
        2560, method="mtp", gdn_replay=True, replay_graph=graph
    ).speculative_config
    assert spec.gdn_spec_state_replay
    assert spec.gdn_replay_graph_enabled() is (graph is not False)
    rebuilt = SpeculativeConfig.from_worker_payload(
        json.loads(json.dumps(spec.to_worker_payload())),
        target_model_config=spec.target_model_config,
        target_parallel_config=spec.target_parallel_config,
    )
    assert rebuilt.gdn_state_replay is True
    assert rebuilt.gdn_spec_state_replay
    assert rebuilt.gdn_replay_graph_enabled() is (graph is not False)
    assert rebuilt.compute_hash() == spec.compute_hash()
    from vllm.config import replace

    copied = replace(spec, gdn_state_replay_graph=graph)
    assert copied.gdn_state_replay is True
    assert copied.gdn_spec_state_replay
    assert copied.gdn_replay_graph_enabled() is (graph is not False)


@pytest.mark.parametrize("replay", [False, True])
def test_mtp_resolver_drives_real_gdn_spec_and_padded_state_table(
    engine_config, replay
):
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.attention.backends.utils import mamba_get_block_table_tensor

    config = engine_config(2560, method="mtp", gdn_replay=replay)
    config.cache_config.mamba_block_size = 16
    layer = SimpleNamespace(
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        is_kv_cache_tp_replicated=False,
        get_state_shape=lambda: ((7, 2560), (12, 128, 128)),
        get_state_dtype=lambda: (torch.float16, torch.float32),
    )
    spec = MambaBase.get_kv_cache_spec(layer, config)
    assert spec.num_speculative_blocks == (0 if replay else 4)
    assert spec.num_replayed_speculative_blocks == (4 if replay else 0)
    assert (
        spec.max_memory_usage_bytes(config)
        == (2 if replay else 6) * spec.page_size_bytes
    )
    blocks = torch.arange(24, dtype=torch.int32).reshape(2, 12) + 1
    table = mamba_get_block_table_tensor(blocks, torch.tensor([17, 33]), spec, "align")
    assert table.shape == (2, 5)
    assert table[:, 0].ge(0).all()
    assert table[:, 1:].eq(-1).all() if replay else table.ge(0).all()


@pytest.mark.parametrize("replay", [None, False, True])
@pytest.mark.parametrize("graph", [False, True])
def test_target_replay_survives_real_draft_construction(
    engine_config, monkeypatch, replay, graph
):
    from types import SimpleNamespace

    import torch

    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
    from vllm.v1.worker.gpu.model_states import default as default_state
    from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState

    config = engine_config(
        2560,
        method="mtp",
        gdn_replay=replay,
        replay_graph=graph,
        legacy_graph=not graph,
    )
    spec = config.speculative_config
    replay = replay is not False
    depth = 4
    assert not spec.draft_model_config.is_hybrid
    policy = asdict(spec.sm70_dflash2)
    config.cache_config.mamba_block_size = 16
    layer = SimpleNamespace(
        mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
        is_kv_cache_tp_replicated=False,
        get_state_shape=lambda: ((7, 2560), (12, 128, 128)),
        get_state_dtype=lambda: (torch.float16, torch.float32),
    )

    def check_target():
        assert spec.gdn_spec_state_replay is replay, "target replay overwritten"
        assert asdict(spec.sm70_dflash2) == policy, "target verifier policy overwritten"
        resolved = MambaBase.get_kv_cache_spec(layer, config)
        assert resolved.num_speculative_blocks == (0 if replay else depth)
        assert resolved.num_replayed_speculative_blocks == (depth if replay else 0)
        assert resolved.max_memory_usage_bytes(config) // resolved.page_size_bytes == (
            2 if replay else 2 + depth
        )
        # Run the real target state constructor after the draft, as MRV2 does.
        # Only rotary setup is unrelated to the recurrent-state contract.
        monkeypatch.setattr(default_state, "get_rope_state", lambda *a, **kw: None)
        state = MambaHybridModelState(
            config, torch.nn.Module(), None, torch.device("cpu")
        )
        assert state._spec_state_replay is replay
        assert state._spec_replay_graphs.enabled is (replay and graph)
        return state

    initial_state = check_target()
    for _ in range(2):
        from vllm.v1.attention.backends import fa_utils

        # Device discovery is simulated; no native attention runs on CPU.
        monkeypatch.setattr(
            fa_utils, "is_flash_attn_varlen_func_available", lambda: False
        )
        from vllm.models.qwen4_exp.nvidia.mtp import _make_draft_vllm_config

        draft = _make_draft_vllm_config(config, mtp_start_layer_idx=4)
        assert draft.is_speculative_draft
        assert draft.model_config is spec.draft_model_config
        assert draft.speculative_config is spec
        assert initial_state._spec_state_replay is replay
        assert initial_state._spec_replay_graphs.enabled is (replay and graph)
        check_target()


def test_mtp_replay_disables_packed_verifier(engine_config, caplog):
    spec = engine_config(
        2560, method="mtp", gdn_replay=True, packed=True
    ).speculative_config
    assert spec.gdn_spec_state_replay
    assert not spec.sm70_dflash2.fused_gdn_verify
    assert "GDN state replay is enabled for MTP" in caplog.text


@pytest.mark.parametrize("method", ["mtp", "dflash"])
@pytest.mark.parametrize("graph", [None, False, True])
def test_replay_graph_override_and_default(engine_config, method, graph):
    spec = engine_config(
        2560,
        method=method,
        gdn_replay=True if method == "mtp" else None,
        replay=True if method == "dflash" else None,
        replay_graph=graph,
        legacy_graph=False,
    ).speculative_config
    assert spec.gdn_replay_graph_enabled() is (
        graph if graph is not None else method == "mtp"
    )
    replay_hash = spec.compute_hash()
    spec.gdn_state_replay_graph = not spec.gdn_replay_graph_enabled()
    assert spec.compute_hash() != replay_hash
    spec.gdn_spec_state_replay = False
    inactive_hash = spec.compute_hash()
    spec.gdn_state_replay_graph = not spec.gdn_state_replay_graph
    assert spec.compute_hash() == inactive_hash


@pytest.mark.parametrize("replay", [None, False, True])
def test_dflash_keeps_legacy_replay_input(engine_config, replay):
    spec = engine_config(2560, replay=False, gdn_replay=replay).speculative_config
    assert not spec.gdn_spec_state_replay
    assert spec.sm70_dflash2.spec_state_replay is False


@pytest.mark.parametrize("method", ["mtp", "dflash"])
def test_default_replay_fields_are_absent_from_serialization(engine_config, method):
    from pydantic import TypeAdapter

    spec = engine_config(2560, method=method).speculative_config
    adapter = TypeAdapter(SpeculativeConfig)
    for payload in (
        adapter.dump_python(spec),
        json.loads(adapter.dump_json(spec, fallback=str)),
    ):
        assert "gdn_state_replay" not in payload
        assert "gdn_state_replay_graph" not in payload


@pytest.mark.parametrize(
    "condition",
    ["hybrid", "sm70", "mrv2", "align", "async", "pp", "core_op", "mixed_qkv"],
)
@pytest.mark.parametrize("replay", [None, True])
def test_mtp_replay_uses_existing_platform_conditions(
    engine_config, monkeypatch, condition, replay
):
    from vllm.config.vllm import _resolve_gdn_spec_state_replay
    from vllm.platforms import current_platform

    config = engine_config(2560, method="mtp", gdn_replay=replay)
    assert config.speculative_config.gdn_spec_state_replay
    if condition == "hybrid":
        monkeypatch.setattr(
            type(config.model_config), "is_hybrid", property(lambda _: False)
        )
    elif condition == "sm70":
        monkeypatch.setattr(
            current_platform, "is_device_capability", lambda *a, **k: False
        )
    elif condition == "mrv2":
        monkeypatch.setattr(
            type(config), "use_v2_model_runner", property(lambda _: False)
        )
    elif condition == "align":
        config.cache_config.mamba_cache_mode = "none"
    elif condition == "async":
        config.scheduler_config.async_scheduling = True
    elif condition == "pp":
        config.parallel_config.pipeline_parallel_size = 2
    elif condition == "core_op":
        monkeypatch.setenv("VLLM_SM70_QWEN_GDN_SPEC_CORE_OP", "1")
    else:
        monkeypatch.setenv("VLLM_SM70_FUSED_SIGMOID_MIXED_QKV", "0")
    assert not _resolve_gdn_spec_state_replay(config)
