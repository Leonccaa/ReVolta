# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReVolta sends its own anonymous usage report, on by default and silent."""

import json
import types

import pytest
import requests

import vllm.envs as envs
from vllm.usage import usage_lib
from vllm.v1 import utils as v1_utils

REVOLTA_COLLECTOR = "https://ai.19b.net/api/usage"

OPT_OUT_VARS = (
    "VLLM_NO_USAGE_STATS",
    "VLLM_DO_NOT_TRACK",
    "DO_NOT_TRACK",
)

# The field names the collector stores (see the payload contract in
# ops/w20/day-20261111/qwen-usage-client/PAYLOAD.md). A report may only carry
# names from this set.
CONTRACT_FIELDS = {
    # Identity / heartbeat
    "uuid",
    "log_time",
    "source",
    # Environment / hardware
    "provider",
    "num_cpu",
    "cpu_type",
    "cpu_family_model_stepping",
    "total_memory",
    "architecture",
    "platform",
    "xpu_runtime",
    "cuda_runtime",
    "gpu_count",
    "gpu_type",
    "gpu_memory_per_device",
    "env_var_json",
    # Engine / model
    "model_architecture",
    "vllm_version",
    "context",
    # Configuration summary
    "dtype",
    "block_size",
    "gpu_memory_utilization",
    "kv_cache_memory_bytes",
    "quantization",
    "kv_cache_dtype",
    "enable_lora",
    "enable_prefix_caching",
    "enforce_eager",
    "disable_custom_all_reduce",
    "tensor_parallel_size",
    "data_parallel_size",
    "pipeline_parallel_size",
    "enable_expert_parallel",
    "all2all_backend",
    "kv_connector",
    # ReVolta specifics
    "revolta_version",
    "max_model_len",
    "enable_long_context_kv",
    "main_kv_placement",
    "decode_context_parallel_size",
    "speculative_method",
    "num_speculative_tokens",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch, tmp_path):
    """No opt-out set, config root in a temp dir, notice flag per test."""
    envs.disable_envs_cache()
    for name in OPT_OUT_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("VLLM_USAGE_STATS_SERVER", raising=False)
    monkeypatch.delenv("VLLM_USAGE_SOURCE", raising=False)
    monkeypatch.setenv("VLLM_CONFIG_ROOT", str(tmp_path))
    monkeypatch.setattr(usage_lib, "_notice_logged", False)


def test_enabled_by_default(clean_env):
    assert usage_lib.is_usage_stats_enabled() is True


@pytest.mark.parametrize("name", OPT_OUT_VARS)
def test_every_opt_out_switch_disables(name, monkeypatch):
    monkeypatch.setenv(name, "1")
    assert usage_lib.is_usage_stats_enabled() is False


def test_opt_out_value_other_than_one_keeps_it_on(monkeypatch):
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "0")
    assert usage_lib.is_usage_stats_enabled() is True


def test_one_opt_out_is_enough_among_several(monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "1")
    assert usage_lib.is_usage_stats_enabled() is False


def test_do_not_track_file_disables(monkeypatch, tmp_path):
    (tmp_path / "do_not_track").touch()
    assert usage_lib.is_usage_stats_enabled() is False


def test_no_do_not_track_file_keeps_it_on(tmp_path):
    assert usage_lib.is_usage_stats_enabled() is True


def test_default_server_is_the_revolta_collector(clean_env):
    assert envs.VLLM_USAGE_STATS_SERVER == REVOLTA_COLLECTOR


def test_server_can_be_overridden(monkeypatch):
    monkeypatch.setenv("VLLM_USAGE_STATS_SERVER", "https://collector.example/api/usage")
    assert envs.VLLM_USAGE_STATS_SERVER == "https://collector.example/api/usage"


class _FakeSession:
    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def post(self, url, data=None, headers=None, timeout=None, json=None):
        self.calls.append(
            {
                "url": url,
                "data": data,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        if self.error is not None:
            raise self.error
        return object()


@pytest.fixture
def fake_client(monkeypatch):
    session = _FakeSession()
    connection = types.SimpleNamespace(get_sync_client=lambda: session)
    monkeypatch.setattr("vllm.connections.global_http_connection", connection)
    return session


def test_upload_uses_the_server_and_a_short_timeout(fake_client):
    usage_lib.usage_message._send_to_server({"uuid": "x"})

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["url"] == REVOLTA_COLLECTOR
    assert call["timeout"] == usage_lib._USAGE_STATS_TIMEOUT
    assert call["timeout"] <= 5.0


def test_upload_sends_one_json_object(fake_client):
    usage_lib.usage_message._send_to_server({"uuid": "x", "log_time": 1})

    call = fake_client.calls[0]
    assert call["headers"]["Content-Type"] == "application/json"
    assert json.loads(call["data"]) == {"uuid": "x", "log_time": 1}


@pytest.mark.parametrize(
    "error",
    [
        requests.exceptions.ConnectionError("down"),
        requests.exceptions.ConnectTimeout("slow"),
        requests.exceptions.ReadTimeout("slow"),
        ValueError("unknown url type"),
        RuntimeError("unexpected"),
    ],
)
def test_upload_failure_is_silent(fake_client, error):
    """Network and server errors must not raise into the engine thread."""
    fake_client.error = error
    usage_lib.usage_message._send_to_server({"uuid": "x"})
    assert len(fake_client.calls) == 1  # one attempt, no retry storm


def test_empty_server_sends_nothing(monkeypatch, fake_client):
    monkeypatch.setenv("VLLM_USAGE_STATS_SERVER", "")
    usage_lib.usage_message._send_to_server({"uuid": "x"})
    assert fake_client.calls == []


def test_worker_swallows_collection_errors(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("platform probe failed")

    monkeypatch.setattr(usage_lib.UsageMessage, "_report_usage_once", boom)
    # Must not raise: an exception in a bare thread prints a traceback.
    usage_lib.usage_message._report_usage_worker(
        "Arch", usage_lib.UsageContext.ENGINE_CONTEXT, {}
    )


def test_file_write_failure_is_silent(monkeypatch, tmp_path):
    # A file where the configuration directory should be makes the write fail.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory")
    monkeypatch.setenv("VLLM_CONFIG_ROOT", str(blocker))
    usage_lib.usage_message._write_to_file({"uuid": "x"})


def _stub_config(**overrides):
    model = types.SimpleNamespace(
        dtype="torch.float16",
        quantization="awq",
        enforce_eager=False,
        max_model_len=262144,
    )
    cache = types.SimpleNamespace(
        block_size=128,
        gpu_memory_utilization=0.9,
        kv_cache_memory_bytes=None,
        cache_dtype="fp8",
        enable_prefix_caching=True,
    )
    parallel = types.SimpleNamespace(
        tensor_parallel_size=4,
        data_parallel_size=1,
        pipeline_parallel_size=1,
        decode_context_parallel_size=2,
        enable_expert_parallel=False,
        all2all_backend="naive",
        disable_custom_all_reduce=False,
    )
    long_context_kv = types.SimpleNamespace(enabled=True, main_kv_placement="host")
    kernel = types.SimpleNamespace(long_context_kv=long_context_kv)
    speculative = types.SimpleNamespace(method="dflash2", num_speculative_tokens=4)
    config = types.SimpleNamespace(
        model_config=model,
        cache_config=cache,
        parallel_config=parallel,
        kernel_config=kernel,
        speculative_config=speculative,
        lora_config=None,
        kv_transfer_config=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


@pytest.fixture
def capture_report(monkeypatch):
    captured = {}

    def record(model_architecture, usage_context, extra_kvs=None):
        captured["model_architecture"] = model_architecture
        captured["context"] = usage_context
        captured.update(extra_kvs or {})

    monkeypatch.setattr(v1_utils.usage_message, "report_usage", record)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.get_architecture_class_name",
        lambda model_config: "StubForCausalLM",
    )
    return captured


def test_revolta_fields_are_present(capture_report):
    v1_utils.report_usage_stats(_stub_config())

    assert capture_report["tensor_parallel_size"] == 4
    assert capture_report["decode_context_parallel_size"] == 2
    assert capture_report["enable_long_context_kv"] is True
    assert capture_report["main_kv_placement"] == "host"
    assert capture_report["speculative_method"] == "dflash2"
    assert capture_report["num_speculative_tokens"] == 4
    assert capture_report["max_model_len"] == 262144
    assert capture_report["kv_cache_dtype"] == "fp8"
    assert capture_report["model_architecture"] == "StubForCausalLM"


def test_no_path_or_host_strings_in_the_report(capture_report):
    config = _stub_config()
    config.kv_transfer_config = types.SimpleNamespace(
        kv_connector="/opt/private/connectors/CustomConnector"
    )
    config.model_config.quantization = "/models/private/awq-checkpoint"

    v1_utils.report_usage_stats(config)

    for key, value in capture_report.items():
        if isinstance(value, str):
            assert "/" not in value, f"{key} carries a path: {value!r}"
            assert "\\" not in value, f"{key} carries a path: {value!r}"
    # The offending values are dropped rather than sent.
    assert capture_report["kv_connector"] is None
    assert capture_report["quantization"] is None


def test_report_without_optional_configs_still_works(capture_report):
    config = _stub_config(kernel_config=None, speculative_config=None)

    v1_utils.report_usage_stats(config)

    assert capture_report["enable_long_context_kv"] is False
    assert capture_report["main_kv_placement"] is None
    assert capture_report["speculative_method"] is None
    assert capture_report["num_speculative_tokens"] is None


def test_broken_config_never_raises(capture_report):
    # A configuration missing every sub-config must not break the startup.
    v1_utils.report_usage_stats(types.SimpleNamespace())
    assert capture_report == {}


def test_disabled_reports_nothing(capture_report, monkeypatch):
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "1")
    # The config is never read: the function returns before touching it.
    v1_utils.report_usage_stats(vllm_config=None)
    assert capture_report == {}


def test_env_payload_keeps_booleans_and_drops_paths(monkeypatch):
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "20,20")
    payload = {
        name: usage_lib._env_value(name)
        for name in usage_lib._USAGE_ENV_VARS_TO_COLLECT
    }
    assert "VLLM_USE_MODELSCOPE" in payload
    assert payload["VLLM_PP_LAYER_PARTITION"] == "20,20"

    # A path-valued member of the allowlist is dropped instead of sent.
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "/srv/models/partition.txt")
    assert usage_lib._env_value("VLLM_PP_LAYER_PARTITION") is None

    # A bare file name is dropped as well.
    monkeypatch.setenv("VLLM_PP_LAYER_PARTITION", "partition.json")
    assert usage_lib._env_value("VLLM_PP_LAYER_PARTITION") is None


def test_env_payload_only_holds_known_switches(monkeypatch):
    payload = {
        name: usage_lib._env_value(name)
        for name in usage_lib._USAGE_ENV_VARS_TO_COLLECT
    }
    assert set(payload) <= set(usage_lib._USAGE_ENV_VARS_TO_COLLECT)
    for value in payload.values():
        assert isinstance(value, (bool, int)) or value is None


def test_source_label_never_leaks_a_path(monkeypatch):
    monkeypatch.setenv("VLLM_USAGE_SOURCE", "production")
    assert usage_lib._source_label() == "production"

    monkeypatch.setenv("VLLM_USAGE_SOURCE", "/data/revolta/rc3")
    assert usage_lib._source_label() == "custom"


def test_full_report_matches_the_collector_contract(monkeypatch):
    """The report may only carry field names the collector stores."""
    message = usage_lib.UsageMessage()
    captured = {}
    # Patch on the class: an instance attribute would show up in the report.
    monkeypatch.setattr(
        usage_lib.UsageMessage,
        "_send_to_server",
        lambda self, data: captured.update(data),
    )
    monkeypatch.setattr(
        usage_lib.UsageMessage, "_write_to_file", lambda self, data: None
    )

    message._report_usage_once(
        "StubForCausalLM",
        usage_lib.UsageContext.ENGINE_CONTEXT,
        {"tensor_parallel_size": 4, "max_model_len": 262144},
    )

    assert set(captured) <= CONTRACT_FIELDS
    assert captured["model_architecture"] == "StubForCausalLM"
    assert captured["context"] == "ENGINE_CONTEXT"
    assert captured["tensor_parallel_size"] == 4
    assert captured["max_model_len"] == 262144
    # The version field separates a ReVolta report from an upstream vLLM one.
    assert captured["revolta_version"]
    assert captured["vllm_version"] == captured["revolta_version"]
    # A canonical UUID is what the collector deduplicates on.
    assert len(captured["uuid"]) == 36


def test_report_payload_is_json_serializable(monkeypatch):
    message = usage_lib.UsageMessage()
    captured = {}
    monkeypatch.setattr(
        usage_lib.UsageMessage,
        "_send_to_server",
        lambda self, data: captured.update(data),
    )
    monkeypatch.setattr(
        usage_lib.UsageMessage, "_write_to_file", lambda self, data: None
    )

    message._report_usage_once(
        "StubForCausalLM",
        usage_lib.UsageContext.ENGINE_CONTEXT,
        {"speculative_method": "dflash2"},
    )

    assert json.loads(json.dumps(captured)) == captured


def test_notice_is_logged_once_per_process(monkeypatch):
    notices = []
    fake = types.SimpleNamespace(
        info=lambda msg, *args: notices.append(msg % args),
        debug=lambda *args: None,
    )
    monkeypatch.setattr(usage_lib, "logger", fake)
    monkeypatch.setattr(
        usage_lib.usage_message, "_report_usage_worker", lambda *a, **k: None
    )

    usage_lib.usage_message.report_usage("Arch", usage_lib.UsageContext.ENGINE_CONTEXT)
    usage_lib.usage_message.report_usage("Arch", usage_lib.UsageContext.ENGINE_CONTEXT)

    assert len(notices) == 1
    text = notices[0]
    assert REVOLTA_COLLECTOR in text
    assert "VLLM_NO_USAGE_STATS=1" in text
    assert "DO_NOT_TRACK" in text
    assert "docs/usage-stats.md" in text
    assert "No prompts" in text


def test_no_notice_when_collection_is_off(monkeypatch):
    notices = []
    fake = types.SimpleNamespace(
        info=lambda msg, *args: notices.append(msg % args),
        debug=lambda *args: None,
    )
    monkeypatch.setattr(usage_lib, "logger", fake)
    monkeypatch.setenv("VLLM_NO_USAGE_STATS", "1")

    v1_utils.report_usage_stats(vllm_config=None)

    assert notices == []
