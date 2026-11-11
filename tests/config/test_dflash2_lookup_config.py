# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""W21 configuration lifecycle, without loading weights or invoking a GPU."""

import copy
import json
import os
import pickle
from dataclasses import fields
from types import SimpleNamespace

import cloudpickle
import pytest
from pydantic import TypeAdapter, ValidationError
from transformers import Qwen3Config

from vllm import envs
from vllm.config import ModelConfig, ParallelConfig, SpeculativeConfig, VllmConfig
from vllm.config.dflash2_lookup import LOOKUP_ENV_PREFIX, DFlash2LookupConfig
from vllm.config.speculative import uses_adaptive_dflash_lookup
from vllm.engine.arg_utils import EngineArgs
from vllm.platforms import current_platform
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup
HEAD = {"enabled": True, "policy": "longest", "nmin": 3, "nmax": 16}
ENV = "VLLM_DFLASH2_LOOKUP_NSTRONG"


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    envs.disable_envs_cache()
    for name in list(os.environ):
        if name.startswith(LOOKUP_ENV_PREFIX):
            monkeypatch.delenv(name)
    yield
    envs.disable_envs_cache()


@pytest.fixture
def checkpoint(tmp_path, monkeypatch):
    for name, arch in (("target", "Qwen3ForCausalLM"), ("draft", "DFlash2DraftModel")):
        Qwen3Config(
            architectures=[arch],
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=1,
            num_key_value_heads=1,
            head_dim=64,
            vocab_size=128,
            max_position_embeddings=512,
            dflash_config={"selector_top_k": 16, "block_size": 8},
        ).save_pretrained(tmp_path / name)
    # Configuration capability only. No device operations are invoked.
    monkeypatch.setattr(current_platform, "is_cuda", lambda: True)
    target = ModelConfig(
        model=str(tmp_path / "target"),
        dtype="float16",
        max_model_len=512,
        skip_tokenizer_init=True,
    )
    parallel = ParallelConfig()
    return {
        "method": "dflash",
        "model": str(tmp_path / "draft"),
        "num_speculative_tokens": 7,
        "target_model_config": target,
        "target_parallel_config": parallel,
    }


def build(checkpoint, **kwargs):
    return SpeculativeConfig(**(checkpoint | kwargs))


@pytest.mark.parametrize(
    "new",
    [
        None,
        {},
        {"enabled": False},
        {"enabled": False, "policy": "longest", "nmin": 3, "nmax": 16},
    ],
)
@pytest.mark.parametrize(
    "legacy,width,mode",
    [(False, 7, "off"), (True, 7, "legacy_host"), (True, 15, "legacy_gpu")],
)
def test_disabled_matrix(checkpoint, new, legacy, width, mode):
    spec = build(
        checkpoint,
        dflash2_lookup=new,
        ngram_assist=legacy,
        num_speculative_tokens=width,
    )
    assert spec.resolved_dflash2_lookup.mode == mode
    assert spec.resolved_dflash2_lookup.source.root == (
        "null" if new is None else "object"
    )
    assert spec.prompt_lookup_min == (5 if legacy else 0)
    assert uses_adaptive_dflash_lookup(spec) == (mode == "legacy_gpu")


@pytest.mark.parametrize(
    "legacy",
    [
        {},
        {"ngram_assist": False},
        {"ngram_assist": True},
        {"prompt_lookup_min": 3},
        {"prompt_lookup_max": 16},
    ],
)
def test_absent_matrix_and_source_before_normalization(checkpoint, legacy):
    spec = build(checkpoint, **legacy)
    source = spec.resolved_dflash2_lookup.source
    assert source.root == "absent"
    assert source.prompt_fields == tuple(
        name for name in ("prompt_lookup_min", "prompt_lookup_max") if name in legacy
    )
    assert source.ngram_assist == legacy.get("ngram_assist")
    assert spec.resolved_dflash2_lookup.mode == (
        "legacy_host" if legacy.get("ngram_assist") else "off"
    )


@pytest.mark.parametrize(
    "legacy",
    [
        {},
        {"ngram_assist": False},
        {"prompt_lookup_min": None, "prompt_lookup_max": None},
    ],
)
@pytest.mark.parametrize("policy", ["longest"])
def test_new_enabled_accepts_only_inert_legacy(checkpoint, legacy, policy):
    spec = build(checkpoint, dflash2_lookup=HEAD | {"policy": policy}, **legacy)
    assert spec.resolved_dflash2_lookup.mode == "gpu_head"
    assert spec.resolved_dflash2_lookup.nmin == 3
    assert not uses_adaptive_dflash_lookup(spec)
    assert spec.prompt_lookup_min == 0


@pytest.mark.parametrize(
    "legacy",
    [{"ngram_assist": True}, {"prompt_lookup_min": 3}, {"prompt_lookup_max": 16}],
)
def test_new_enabled_conflicts_before_prompt_fields_are_erased(checkpoint, legacy):
    with pytest.raises(ValueError, match="conflicts.*explicit legacy"):
        build(checkpoint, dflash2_lookup=HEAD, **legacy)


@pytest.mark.parametrize("new", [None, {}, {"enabled": False}, HEAD])
def test_explicit_env_and_disabled_legacy_routing(checkpoint, monkeypatch, new):
    monkeypatch.setenv(ENV, "6")
    if new == HEAD:
        with pytest.raises(ValueError, match=ENV):
            build(checkpoint, dflash2_lookup=new)
    else:
        spec = build(checkpoint, dflash2_lookup=new)
        assert spec.resolved_dflash2_lookup.mode == "off"
        assert spec.resolved_dflash2_lookup.source.environment == ((ENV, "6"),)


@pytest.mark.parametrize("enabled", [True, False])
def test_expansion_is_rejected_even_when_disabled(checkpoint, enabled):
    with pytest.raises(ValueError, match="expand.enabled=true is not supported"):
        build(
            checkpoint,
            dflash2_lookup=HEAD | {"enabled": enabled, "expand": {"enabled": True}},
        )


@pytest.mark.parametrize(
    "values",
    [
        {"nmin": True},
        {"nmax": False},
        {"nmin": "3"},
        {"search_window": True},
        {"nmin": 0},
        {"nmax": 0},
        {"nmin": 20, "nmax": 16},
        {"nmax": 513},
        {"policy": "other"},
        {"gate": "strong"},
        {"enabled": "false"},
        {"collect_stats": 1},
        {"expand": {"other": False}},
        {"unknown": 1},
    ],
)
@pytest.mark.parametrize("enabled", [True, False])
def test_invalid_format_is_rejected_even_when_disabled(checkpoint, values, enabled):
    with pytest.raises(ValueError):
        build(checkpoint, dflash2_lookup=HEAD | {"enabled": enabled} | values)


@pytest.mark.parametrize("missing", ["policy", "nmin", "nmax"])
def test_enabling_requires_explicit_policy_and_bounds(checkpoint, missing):
    raw = HEAD.copy()
    del raw[missing]
    with pytest.raises(ValueError, match="requires explicit"):
        build(checkpoint, dflash2_lookup=raw)


def test_checkpoint_geometry_capability_and_cuda(checkpoint, monkeypatch):
    with pytest.raises(ValueError, match="K=D=7"):
        build(checkpoint, dflash2_lookup=HEAD, num_speculative_tokens=15)
    path = os.path.join(checkpoint["model"], "config.json")
    with open(path) as stream:
        raw = json.load(stream)
    raw["dflash_config"]["block_size"] = 5
    with open(path, "w") as stream:
        json.dump(raw, stream)
    with pytest.raises(ValueError, match="K=D=7"):
        build(checkpoint, dflash2_lookup=HEAD)
    raw["dflash_config"]["selector_top_k"] = 0
    with open(path, "w") as stream:
        json.dump(raw, stream)
    with pytest.raises(ValueError, match="selector capability"):
        build(checkpoint, dflash2_lookup=HEAD)


def test_cuda_capability_rejection(checkpoint, monkeypatch):
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    with pytest.raises(ValueError, match="requires CUDA"):
        build(checkpoint, dflash2_lookup=HEAD)


def test_other_methods_and_original_ngram_validation():
    spec = SpeculativeConfig(
        method="ngram",
        num_speculative_tokens=7,
        prompt_lookup_min=3,
        prompt_lookup_max=16,
        dflash2_lookup={},
    )
    assert spec.prompt_lookup_min == 3
    assert spec.resolved_dflash2_lookup.mode == "off"
    with pytest.raises(ValueError, match="method='dflash'"):
        SpeculativeConfig(method="ngram", num_speculative_tokens=7, dflash2_lookup=HEAD)
    with pytest.raises(ValueError):
        SpeculativeConfig(
            method="ngram",
            num_speculative_tokens=7,
            prompt_lookup_min=16,
            prompt_lookup_max=3,
        )


def test_concurrency_and_startup_guard(checkpoint):
    spec = build(
        checkpoint, dflash2_lookup=HEAD, dflash_concurrency_drafts={4: 3, 5: 4}
    )
    spec.require_lookup_runtime()
    with pytest.raises(ValueError, match="without ngram_assist"):
        build(checkpoint, ngram_assist=True, dflash_concurrency_drafts={4: 3})


@pytest.mark.parametrize(
    "new,legacy,width",
    [
        (None, False, 7),
        ({}, False, 7),
        ({"enabled": False}, True, 7),
        ({"enabled": False}, True, 15),
        (HEAD, False, 7),
    ],
)
@pytest.mark.parametrize("transport", ["json", "pickle", "cloudpickle"])
def test_worker_lifecycle_is_idempotent_and_ignores_worker_env(
    checkpoint, monkeypatch, new, legacy, width, transport
):
    spec = build(
        checkpoint,
        dflash2_lookup=new,
        ngram_assist=legacy,
        num_speculative_tokens=width,
    )
    expected = spec.resolved_dflash2_lookup
    expected_hash = spec.compute_hash()
    payload = spec.to_worker_payload()
    monkeypatch.setenv(ENV, "27")
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", "0")
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_POLICY", "research")
    for _ in range(2):
        if transport == "json":
            spec = SpeculativeConfig.from_worker_payload(
                json.loads(json.dumps(spec.to_worker_payload())),
                target_model_config=checkpoint["target_model_config"],
                target_parallel_config=checkpoint["target_parallel_config"],
            )
        else:
            codec = pickle if transport == "pickle" else cloudpickle
            spec = codec.loads(codec.dumps(spec))
        assert spec.resolved_dflash2_lookup == expected
        assert spec.compute_hash() == expected_hash
        assert spec.to_worker_payload() == payload
        assert spec.lookup_env(ENV) == 6
        assert spec.lookup_env("VLLM_DFLASH2_LOOKUP_ADAPTIVE") is True
        factors = envs.compile_factors(spec)
        assert factors[ENV] == 6
        assert "VLLM_DFLASH2_LOOKUP_POLICY" not in factors


def test_source_captures_python_positional_arguments(checkpoint):
    names = [field.name for field in fields(SpeculativeConfig) if field.init]
    values = checkpoint | {"dflash2_lookup": HEAD, "ngram_assist": False}
    end = names.index("ngram_assist") + 1
    defaults = {
        name: field.default
        for name, field in SpeculativeConfig.__pydantic_fields__.items()
    }
    args = [values.pop(name, defaults[name]) for name in names[:end]]
    spec = SpeculativeConfig(*args, **values)
    assert spec.resolved_dflash2_lookup.source.ngram_assist is False
    assert spec.resolved_dflash2_lookup.source.prompt_fields == ()


@pytest.mark.parametrize("shorthand", [True, False])
def test_real_cli_parser_normalize_serialize_worker_rebuild(checkpoint, shorthand):
    raw = {"dflash2_lookup": HEAD, "ngram_assist": False}
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    argv = [
        "--model",
        checkpoint["target_model_config"].model,
        "--speculative-config",
        json.dumps(
            raw
            if shorthand
            else raw
            | {
                "method": "dflash",
                "model": checkpoint["model"],
                "num_speculative_tokens": 7,
            }
        ),
    ]
    if shorthand:
        argv += [
            "--spec-method",
            "dflash",
            "--spec-model",
            checkpoint["model"],
            "--spec-tokens",
            "7",
        ]
    args = EngineArgs.from_cli_args(parser.parse_args(argv))
    spec = args.create_speculative_config(
        checkpoint["target_model_config"], checkpoint["target_parallel_config"]
    )
    source = spec.resolved_dflash2_lookup.source
    assert source.keys == tuple(sorted(HEAD))
    assert source.system_fields == (
        "enforce_eager",
        "target_model_config",
        "target_parallel_config",
    )
    rebuilt = SpeculativeConfig.from_worker_payload(
        json.loads(json.dumps(spec.to_worker_payload())),
        target_model_config=checkpoint["target_model_config"],
        target_parallel_config=checkpoint["target_parallel_config"],
    )
    assert rebuilt.resolved_dflash2_lookup == spec.resolved_dflash2_lookup
    assert rebuilt.to_worker_payload() == spec.to_worker_payload()


def test_trusted_schema_and_cli_injection_rejection(checkpoint):
    spec = build(checkpoint, dflash2_lookup=HEAD)
    with pytest.raises(ValueError):
        build(
            checkpoint,
            resolved_dflash2_lookup=spec.resolved_dflash2_lookup.model_dump(),
        )
    with pytest.raises(ValueError):
        build(checkpoint, dflash2_lookup=HEAD | {"resolved": True})
    for field, value in (
        ("schema_version", 2),
        ("schema_version", True),
        ("admission_version", 2),
        ("admission_version", True),
        ("verify_tokens", 15),
        ("nmin", "3"),
        ("mode", "off"),
    ):
        payload = copy.deepcopy(spec.to_worker_payload())
        payload["lookup"][field] = value
        with pytest.raises(ValueError):
            SpeculativeConfig.from_worker_payload(
                payload,
                target_model_config=checkpoint["target_model_config"],
                target_parallel_config=checkpoint["target_parallel_config"],
            )
    with pytest.raises(ValidationError):
        spec.resolved_dflash2_lookup.mode = "off"


def test_legacy_payload_upgrade_once(checkpoint, monkeypatch):
    raw = {
        name: value
        for name, value in checkpoint.items()
        if not name.startswith("target_")
    } | {"ngram_assist": True, "num_speculative_tokens": 15}
    monkeypatch.setenv(ENV, "9")
    upgraded = SpeculativeConfig.from_worker_payload(
        raw,
        target_model_config=checkpoint["target_model_config"],
        target_parallel_config=checkpoint["target_parallel_config"],
    )
    monkeypatch.setenv(ENV, "27")
    rebuilt = SpeculativeConfig.from_worker_payload(
        upgraded.to_worker_payload(),
        target_model_config=checkpoint["target_model_config"],
        target_parallel_config=checkpoint["target_parallel_config"],
    )
    assert rebuilt.lookup_env(ENV) == 9
    assert rebuilt.resolved_dflash2_lookup == upgraded.resolved_dflash2_lookup


def test_active_hash_ignores_provenance_and_disabled_presets(checkpoint):
    baseline = build(checkpoint)
    for raw in (None, {}, {"enabled": False}, HEAD | {"enabled": False}):
        assert (
            build(checkpoint, dflash2_lookup=raw).compute_hash()
            == baseline.compute_hash()
        )
        assert envs.compile_factors(
            build(checkpoint, dflash2_lookup=raw)
        ) == envs.compile_factors(baseline)
    active = build(checkpoint, dflash2_lookup=HEAD)
    assert active.compute_hash() != baseline.compute_hash()
    assert (
        build(checkpoint, dflash2_lookup=HEAD, ngram_assist=False).compute_hash()
        == active.compute_hash()
    )
    for patch in (
        {"nmin": 4},
        {"nmax": 17},
        {"search_window": 1024},
    ):
        assert (
            build(checkpoint, dflash2_lookup=HEAD | patch).compute_hash()
            != active.compute_hash()
        )
    stats_off = build(checkpoint, dflash2_lookup=HEAD | {"collect_stats": False})
    assert stats_off.compute_hash() == active.compute_hash()
    assert envs.compile_factors(stats_off) == envs.compile_factors(active)
    assert stats_off.resolved_dflash2_lookup.collect_stats is False
    options = active.resolved_dflash2_lookup.graph_options()
    assert options[0] == active.resolved_dflash2_lookup.schema_version
    assert options[-1] == active.resolved_dflash2_lookup.admission_version


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"ngram_assist": True},
        {"ngram_assist": True, "num_speculative_tokens": 15},
        {"dflash2_lookup": {}},
        {"dflash2_lookup": HEAD},
    ],
)
def test_aot_and_backend_actual_cache_entrypoints_use_fixed_factors(
    checkpoint, monkeypatch, values
):
    from vllm.compilation import backends, caching

    spec = build(checkpoint, **values)
    original = envs.compile_factors
    calls = []

    class FactorsObserved(Exception):
        pass

    def observe(config=None):
        calls.append(original(config))
        assert config is spec
        raise FactorsObserved

    monkeypatch.setattr(envs, "compile_factors", observe)
    config = SimpleNamespace(speculative_config=spec)
    backend = SimpleNamespace(vllm_config=config, _log_compilation_config=lambda: None)
    for value in ("7", "29"):
        monkeypatch.setenv(ENV, value)
        with pytest.raises(FactorsObserved):
            caching.aot_compile_hash_factors(config)
        with pytest.raises(FactorsObserved):
            backends.VllmBackend.__call__(backend, None, [])
    assert calls[0] == calls[1] == calls[2] == calls[3]
    assert calls[0][ENV] == 6


def test_typed_configuration_preserves_only_explicit_keys(checkpoint):
    from vllm.config.dflash2_lookup import DFlash2LookupExpandConfig

    lookup = DFlash2LookupConfig(**HEAD)
    spec = build(checkpoint, dflash2_lookup=lookup)
    assert spec.resolved_dflash2_lookup.source.keys == tuple(sorted(HEAD))
    assert spec.resolved_dflash2_lookup.source.expand_keys == ()
    lookup = DFlash2LookupConfig(
        **HEAD, expand=DFlash2LookupExpandConfig(enabled=False)
    )
    spec = build(checkpoint, dflash2_lookup=lookup)
    assert spec.resolved_dflash2_lookup.source.expand_keys == ("enabled",)


def test_typed_configuration_cannot_bypass_expansion_rejection(checkpoint):
    lookup = DFlash2LookupConfig(enabled=False)
    lookup.expand.enabled = True
    with pytest.raises(ValueError, match="expand.enabled=true is not supported"):
        build(checkpoint, dflash2_lookup=lookup)


def test_all_positional_fields_skip_non_init_fields(checkpoint):
    names = [
        name
        for name, field in SpeculativeConfig.__pydantic_fields__.items()
        if field.init is not False
    ]
    defaults = {
        name: field.default
        for name, field in SpeculativeConfig.__pydantic_fields__.items()
    }
    values = checkpoint | {"dflash2_lookup": HEAD}
    # Supply the default factory field explicitly, as a normal positional caller must.
    from vllm.config.sm70_dflash2 import Sm70DFlash2Config

    values["sm70_dflash2"] = Sm70DFlash2Config()
    args = [values.get(name, defaults[name]) for name in names]
    spec = SpeculativeConfig(*args)
    assert spec.resolved_dflash2_lookup.mode == "gpu_head"
    assert spec.resolved_dflash2_lookup.source.keys == tuple(sorted(HEAD))


@pytest.mark.parametrize("legacy", [False, True])
def test_previous_normalized_payload_upgrade(checkpoint, legacy):
    spec = build(checkpoint, ngram_assist=legacy)
    old = TypeAdapter(SpeculativeConfig).dump_python(spec)
    old.pop("dflash2_lookup", None)
    rebuilt = SpeculativeConfig.from_worker_payload(
        old,
        target_model_config=checkpoint["target_model_config"],
        target_parallel_config=checkpoint["target_parallel_config"],
    )
    assert rebuilt.compute_hash() == spec.compute_hash()
    assert rebuilt.prompt_lookup_min == spec.prompt_lookup_min
    assert rebuilt.resolved_dflash2_lookup.mode == spec.resolved_dflash2_lookup.mode


def test_json_preserves_existing_resolved_policy(checkpoint):
    from vllm.config.sm70_dflash2 import Sm70DFlash2Config

    policy = Sm70DFlash2Config(context_pipeline=False)
    policy.resolve(qualified=True)
    spec = build(checkpoint, sm70_dflash2=policy)
    spec.gdn_spec_state_replay = True
    rebuilt = SpeculativeConfig.from_worker_payload(
        json.loads(json.dumps(spec.to_worker_payload())),
        target_model_config=checkpoint["target_model_config"],
        target_parallel_config=checkpoint["target_parallel_config"],
    )
    assert rebuilt.gdn_spec_state_replay
    assert rebuilt.sm70_dflash2 == spec.sm70_dflash2
    assert rebuilt.compute_hash() == spec.compute_hash()
    assert rebuilt.to_worker_payload() == spec.to_worker_payload()


@pytest.mark.parametrize("transport", ["pickle", "cloudpickle"])
def test_real_worker_process_receives_resolved_config(
    checkpoint, monkeypatch, transport
):
    import subprocess
    import sys

    import vllm

    spec = build(checkpoint, ngram_assist=True, num_speculative_tokens=15)
    monkeypatch.setenv(ENV, "29")
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", "0")
    codec = pickle if transport == "pickle" else cloudpickle
    script = """
import json
import sys
import vllm
vllm.__path__[:] = json.loads(sys.argv[1])
from vllm.platforms.cpu import CpuPlatform
import vllm.platforms
vllm.platforms._current_platform = CpuPlatform()
from vllm.v1.serial_utils import pickle, cloudpickle
from vllm import envs
codec = pickle if sys.argv[2] == "pickle" else cloudpickle
spec = codec.loads(sys.stdin.buffer.read())
assert spec.lookup_env("VLLM_DFLASH2_LOOKUP_NSTRONG") == 6
assert spec.lookup_env("VLLM_DFLASH2_LOOKUP_ADAPTIVE") is True
assert envs.compile_factors(spec)["VLLM_DFLASH2_LOOKUP_NSTRONG"] == 6
print("WORKER_PAYLOAD=" + json.dumps(spec.to_worker_payload(), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", script, json.dumps(list(vllm.__path__)), transport],
        input=codec.dumps(spec),
        capture_output=True,
        check=True,
    )
    line = next(
        line
        for line in completed.stdout.decode().splitlines()
        if line.startswith("WORKER_PAYLOAD=")
    )
    assert json.loads(line.split("=", 1)[1]) == spec.to_worker_payload()


def test_trusted_rebuild_does_not_capture_worker_env(checkpoint, monkeypatch):
    import vllm.config.speculative as implementation

    spec = build(checkpoint)

    def fail(*args, **kwargs):
        raise AssertionError("worker must not capture raw lookup inputs again")

    monkeypatch.setattr(implementation, "capture_lookup_source", fail)
    monkeypatch.setattr(implementation, "lookup_env_factors", fail)
    rebuilt = SpeculativeConfig.from_worker_payload(
        spec.to_worker_payload(),
        target_model_config=checkpoint["target_model_config"],
        target_parallel_config=checkpoint["target_parallel_config"],
    )
    assert rebuilt.resolved_dflash2_lookup == spec.resolved_dflash2_lookup


def test_legacy_env_getter_cache_and_invalid_values(checkpoint, monkeypatch):
    envs.enable_envs_cache()
    assert envs.VLLM_DFLASH2_LOOKUP_NSTRONG == 6
    monkeypatch.setenv(ENV, "9")
    spec = build(checkpoint, ngram_assist=True, num_speculative_tokens=15)
    assert spec.lookup_env(ENV) == 9
    assert envs.compile_factors(spec)[ENV] == 9
    monkeypatch.setenv(ENV, "invalid")
    invalid = build(checkpoint, ngram_assist=True, num_speculative_tokens=15)
    with pytest.raises(ValueError, match="Invalid legacy lookup"):
        invalid.lookup_env(ENV)


@pytest.mark.parametrize("legacy,width", [(False, 7), (True, 7), (True, 15)])
def test_full_engine_and_aot_factors_match_when_new_interface_is_off(
    checkpoint, monkeypatch, legacy, width
):
    from vllm.compilation.caching import aot_compile_hash_factors

    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(envs, "VLLM_USE_MEGA_AOT_ARTIFACT", False)
    raw = {
        name: value
        for name, value in checkpoint.items()
        if not name.startswith("target_")
    } | {"ngram_assist": legacy, "num_speculative_tokens": width}

    def create(new):
        values = dict(raw)
        if new is not None:
            values["dflash2_lookup"] = new
        return EngineArgs(
            model=checkpoint["target_model_config"].model,
            dtype="float16",
            skip_tokenizer_init=True,
            enforce_eager=True,
            max_model_len=512,
            max_num_batched_tokens=512,
            max_num_seqs=2,
            speculative_config=values,
        ).create_engine_config()

    baseline = create(None)
    for new in ({}, {"enabled": False}, HEAD | {"enabled": False}):
        closed = create(new)
        assert closed.compute_hash() == baseline.compute_hash()
        assert aot_compile_hash_factors(closed) == aot_compile_hash_factors(baseline)
        assert envs.compile_factors(closed.speculative_config) == envs.compile_factors(
            baseline.speculative_config
        )


@pytest.mark.parametrize("adaptive", [False, True])
def test_worker_env_cannot_change_legacy_async_admission(
    checkpoint, monkeypatch, adaptive
):
    from vllm.config import DeviceConfig, SchedulerConfig

    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", str(int(adaptive)))
    spec = build(checkpoint, ngram_assist=True, num_speculative_tokens=15)
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", str(int(not adaptive)))

    def create():
        return VllmConfig(
            model_config=checkpoint["target_model_config"],
            parallel_config=checkpoint["target_parallel_config"],
            speculative_config=spec,
            device_config=DeviceConfig(device="cpu"),
            scheduler_config=SchedulerConfig(
                max_model_len=512,
                is_encoder_decoder=False,
                max_num_seqs=2,
                max_num_batched_tokens=512,
                async_scheduling=True,
            ),
        )

    if adaptive:
        with pytest.raises(ValueError, match="incompatible with adaptive"):
            create()
    else:
        # CPU platform policy applies its scheduler override after admission.
        accepted = create()
        assert (
            accepted.speculative_config.lookup_env("VLLM_DFLASH2_LOOKUP_ADAPTIVE")
            is False
        )


@pytest.mark.parametrize("legacy,width", [(False, 7), (True, 7), (True, 15)])
@pytest.mark.parametrize(
    "new_options",
    [
        {},
        {"dflash2_lookup": None},
        {"dflash2_lookup": {}},
        {"dflash2_lookup": {"enabled": False}},
    ],
)
def test_project_replace_preserves_resolved_state_and_rebuilds_changes(
    checkpoint, monkeypatch, legacy, width, new_options
):
    from vllm.config.utils import replace

    spec = build(
        checkpoint, ngram_assist=legacy, num_speculative_tokens=width, **new_options
    )
    spec.gdn_spec_state_replay = True
    expected = spec.to_worker_payload()
    monkeypatch.setenv(ENV, "29")
    monkeypatch.setenv("VLLM_DFLASH2_LOOKUP_ADAPTIVE", "0")
    for _ in range(2):
        cloned = replace(spec)
        assert cloned is not spec
        assert cloned.resolved_dflash2_lookup == spec.resolved_dflash2_lookup
        assert cloned.to_worker_payload() == expected
        assert cloned.compute_hash() == spec.compute_hash()
        assert cloned.gdn_spec_state_replay
        assert cloned.lookup_env(ENV) == 6
        spec = cloned
    cloned = replace(spec, num_speculative_tokens=15 if legacy else 7)
    assert cloned.lookup_env(ENV) == 6
    assert cloned.resolved_dflash2_lookup.source == spec.resolved_dflash2_lookup.source
    assert cloned.resolved_dflash2_lookup.mode == ("legacy_gpu" if legacy else "off")
    assert cloned.resolved_dflash2_lookup is not spec.resolved_dflash2_lookup


def test_replace_ngram_uses_normalized_field_values():
    from vllm.config.utils import replace

    spec = SpeculativeConfig(method="ngram", num_speculative_tokens=7)
    copied = replace(spec)
    assert copied.resolved_dflash2_lookup == spec.resolved_dflash2_lookup
    changed = replace(spec, prompt_lookup_min=3)
    assert (changed.prompt_lookup_min, changed.prompt_lookup_max) == (3, 5)
    again = replace(changed, num_speculative_tokens=4)
    assert (
        again.resolved_dflash2_lookup.source == changed.resolved_dflash2_lookup.source
    )
    assert (again.prompt_lookup_min, again.prompt_lookup_max) == (3, 5)
    rebuilt = SpeculativeConfig.from_worker_payload(again.to_worker_payload())
    assert (rebuilt.prompt_lookup_min, rebuilt.prompt_lookup_max) == (3, 5)


@pytest.mark.parametrize("new", [None, {}, {"enabled": False}])
@pytest.mark.parametrize("legacy", [False, True])
def test_config_traversal_and_logging_accept_private_state(checkpoint, new, legacy):
    from dataclasses import asdict

    from vllm.config.utils import replace
    from vllm.logging_utils.dump_input import prepare_object_to_dump

    spec = build(checkpoint, dflash2_lookup=new, ngram_assist=legacy)
    public = asdict(spec)
    assert "_lookup_resolved" not in public
    assert "_lookup_input" not in public
    assert "SpeculativeConfig" in repr(spec)
    assert "SpeculativeConfig" in prepare_object_to_dump(spec)
    assert copy.copy(spec).resolved_dflash2_lookup == spec.resolved_dflash2_lookup
    assert copy.deepcopy(spec).resolved_dflash2_lookup == spec.resolved_dflash2_lookup
    if spec.dflash2_lookup is not None:
        child = replace(spec.dflash2_lookup, policy="longest")
        assert child.policy == "longest"
        assert replace(child.expand).enabled is False


@pytest.mark.parametrize("codec", [pickle, cloudpickle], ids=["pickle", "cloudpickle"])
@pytest.mark.parametrize(
    "corruption",
    [
        "lookup_type",
        "expand_type",
        "expand_enabled",
        "bool_bound",
        "negative_window",
        "inverted_bounds",
        "oversized_bound",
        "invalid_policy",
        "invalid_gate",
    ],
)
def test_pickle_restore_revalidates_nested_configuration(
    checkpoint, monkeypatch, codec, corruption
):
    spec = build(checkpoint, dflash2_lookup={"enabled": False})
    lookup = spec.dflash2_lookup
    if corruption == "lookup_type":
        spec.dflash2_lookup = 42
    elif corruption == "expand_type":
        lookup.expand = 42
    elif corruption == "expand_enabled":
        lookup.expand.enabled = True
    elif corruption == "bool_bound":
        lookup.nmin = True
    elif corruption == "negative_window":
        lookup.search_window = -1
    elif corruption == "inverted_bounds":
        lookup.nmin, lookup.nmax = 5, 3
    elif corruption == "oversized_bound":
        lookup.nmax = 513
    elif corruption == "invalid_policy":
        lookup.policy = "other"
    else:
        lookup.gate = "other"
    monkeypatch.setenv(ENV, "29")
    with pytest.raises(ValueError):
        codec.loads(codec.dumps(spec))


def test_failed_setstate_preserves_existing_instance(checkpoint):
    spec = build(checkpoint, dflash2_lookup={})
    state = dict(spec.__dict__)
    state["dflash2_lookup"] = 42
    expected = spec.to_worker_payload()
    with pytest.raises(ValueError):
        spec.__setstate__(state)
    assert spec.to_worker_payload() == expected


@pytest.mark.parametrize("new", [None, {}, {"enabled": False}])
@pytest.mark.parametrize("old", ["host", "gpu", "prompt", "env"])
def test_disabled_interface_logs_source_and_effective_mode(
    checkpoint, monkeypatch, caplog, new, old
):
    import logging

    values = {"dflash2_lookup": new}
    if old in ("host", "gpu"):
        values.update(
            ngram_assist=True, num_speculative_tokens=15 if old == "gpu" else 7
        )
    elif old == "prompt":
        values["prompt_lookup_min"] = 3
    else:
        monkeypatch.setenv(ENV, "6")
    with caplog.at_level(logging.WARNING, logger="vllm.config.dflash2_lookup"):
        spec = build(checkpoint, **values)
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "vllm.config.dflash2_lookup"
        and "effective mode=" in record.getMessage()
    ]
    assert len(messages) == 1
    source = spec.resolved_dflash2_lookup.source
    assert f"source={source.root}" in messages[0]
    assert f"keys={source.keys}" in messages[0]
    expected_mode = {
        "host": "legacy_host",
        "gpu": "legacy_gpu",
        "prompt": "off",
        "env": "off",
    }[old]
    assert f"effective mode={expected_mode}" in messages[0]


def test_absent_new_interface_has_no_disabled_mode_warning(checkpoint, caplog):
    build(checkpoint, ngram_assist=True)
    assert not any(
        "effective mode=" in record.getMessage() for record in caplog.records
    )


@pytest.mark.parametrize("ladder", [None, {4: 3, 5: 4}])
def test_native_runtime_accepts_concurrency_ladder(checkpoint, ladder):
    spec = build(checkpoint, dflash2_lookup=HEAD, dflash_concurrency_drafts=ladder)
    spec.require_lookup_runtime()
    baseline = build(checkpoint, dflash2_lookup=HEAD)
    assert spec.compute_hash() == baseline.compute_hash()
    assert envs.compile_factors(spec) == envs.compile_factors(baseline)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("policy", ["nearest", "other", "", 1, False])
def test_unprovided_policy_fails_at_configuration(checkpoint, enabled, policy):
    with pytest.raises(ValueError, match="policy .*not provided in this release"):
        build(checkpoint, dflash2_lookup=HEAD | {"enabled": enabled, "policy": policy})


@pytest.mark.parametrize("codec", [pickle, cloudpickle])
def test_worker_restore_rejects_unprovided_policy(checkpoint, codec):
    spec = build(checkpoint, dflash2_lookup=HEAD)
    state = spec.__getstate__()
    state["dflash2_lookup"].policy = "nearest"
    with pytest.raises(ValueError, match="not provided in this release"):
        restored = object.__new__(SpeculativeConfig)
        restored.__setstate__(codec.loads(codec.dumps(state)))


@pytest.mark.parametrize("replay", [False, True])
def test_native_kv_geometry_remains_seven(checkpoint, replay):
    spec = build(checkpoint, dflash2_lookup=HEAD)
    # The engine derives this flag after selecting the target GDN policy.
    spec.gdn_spec_state_replay = replay
    config = object.__new__(VllmConfig)
    config.speculative_config = spec
    assert spec.num_speculative_tokens == 7
    assert spec.num_speculative_state_tokens() == 7
    assert config.num_lookahead_tokens == 8
    assert spec.num_speculative_state_blocks() == (0 if replay else 7)
    assert not uses_adaptive_dflash_lookup(spec)


@pytest.mark.parametrize("async_scheduling", [False, None, True])
@pytest.mark.parametrize("ladder", [None, {4: 3, 5: 4}])
@pytest.mark.parametrize("padded", [False, True])
def test_native_engine_uses_existing_async_rules(
    checkpoint, monkeypatch, async_scheduling, ladder, padded
):
    from vllm.config import SchedulerConfig

    spec = build(
        checkpoint,
        dflash2_lookup=HEAD,
        dflash_concurrency_drafts=ladder,
        disable_padded_drafter_batch=padded,
    )
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(envs, "VLLM_USE_MEGA_AOT_ARTIFACT", False)
    # Test the engine decision before the CPU hardware adapter disables async.
    monkeypatch.setattr(
        current_platform, "check_and_update_config", lambda config: None
    )
    kwargs = dict(
        model_config=checkpoint["target_model_config"],
        parallel_config=checkpoint["target_parallel_config"],
        speculative_config=spec,
        scheduler_config=SchedulerConfig(
            max_model_len=512,
            max_num_seqs=2,
            max_num_batched_tokens=512,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
    )
    if async_scheduling is True and (ladder or padded):
        message = (
            "ignores dflash_concurrency_drafts"
            if ladder
            else "disable_padded_drafter_batch"
        )
        with pytest.raises(ValueError, match=message):
            VllmConfig(**kwargs)
    else:
        config = VllmConfig(**kwargs)
        assert config.scheduler_config.async_scheduling is (
            async_scheduling is not False and not ladder and not padded
        )


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_rejects_nearest_in_this_release(checkpoint, enabled):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    raw = {
        "method": "dflash",
        "model": checkpoint["model"],
        "num_speculative_tokens": 7,
        "dflash2_lookup": HEAD | {"enabled": enabled, "policy": "nearest"},
    }
    args = EngineArgs.from_cli_args(
        parser.parse_args(
            [
                "--model",
                checkpoint["target_model_config"].model,
                "--speculative-config",
                json.dumps(raw),
            ]
        )
    )
    with pytest.raises(ValueError, match="not provided in this release"):
        args.create_speculative_config(
            checkpoint["target_model_config"], checkpoint["target_parallel_config"]
        )


@pytest.mark.parametrize("async_scheduling", [None, True])
def test_native_async_respects_executor_capability(
    checkpoint, monkeypatch, async_scheduling
):
    from vllm.config import SchedulerConfig
    from vllm.v1.executor.abstract import Executor

    class SyncExecutor:
        @staticmethod
        def supports_async_scheduling():
            return False

    spec = build(checkpoint, dflash2_lookup=HEAD)
    monkeypatch.setattr(current_platform, "is_cuda", lambda: False)
    monkeypatch.setattr(
        current_platform, "check_and_update_config", lambda config: None
    )
    monkeypatch.setattr(Executor, "get_class", lambda config: SyncExecutor)
    kwargs = dict(
        model_config=checkpoint["target_model_config"],
        parallel_config=checkpoint["target_parallel_config"],
        speculative_config=spec,
        scheduler_config=SchedulerConfig(
            max_model_len=512,
            max_num_batched_tokens=512,
            is_encoder_decoder=False,
            async_scheduling=async_scheduling,
        ),
    )
    if async_scheduling:
        with pytest.raises(ValueError, match="does not support async scheduling yet"):
            VllmConfig(**kwargs)
    else:
        assert VllmConfig(**kwargs).scheduler_config.async_scheduling is False


def test_model_and_engine_config_copy_rebuild_derived_state(checkpoint):
    from dataclasses import asdict

    from vllm.config.utils import replace
    from vllm.logging_utils.dump_input import prepare_object_to_dump

    model = checkpoint["target_model_config"]
    model._temporary_runtime_state = object()
    cloned = replace(model, seed=41)
    assert cloned.seed == 41
    assert cloned.model == model.model
    assert cloned.model_arch_config == model.model_arch_config
    assert not hasattr(cloned, "_temporary_runtime_state")
    engine = VllmConfig(model_config=model)
    assert engine.speculative_config is None
    copied = replace(engine, model_config=cloned)
    assert copied.model_config.seed == 41
    assert copied.speculative_config is None
    assert "_temporary_runtime_state" not in asdict(cloned)
    assert "ModelConfig" in prepare_object_to_dump(cloned)
