# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import datetime
import json
import os
import platform
import time
from enum import Enum
from pathlib import Path
from threading import Thread
from typing import Any
from uuid import uuid4

import cpuinfo
import psutil
import regex as re
import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.platform_utils import cuda_get_device_properties
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

# One report is a single best-effort POST on a daemon thread. The timeout only
# bounds that thread; the engine never waits for it. Failures are logged at
# debug level and dropped: no retry, so an unreachable collector costs one
# request per interval instead of a retry storm.
_USAGE_STATS_TIMEOUT = 5.0
_USAGE_STATS_INTERVAL = 600

# Path of the user-facing documentation, relative to the repository root.
_USAGE_STATS_DOCS = "docs/usage-stats.md"

_notice_logged = False

_GLOBAL_RUNTIME_DATA = dict[str, str | int | bool]()

# Environment variables whose values go into ``env_var_json``. Each entry was
# reviewed one by one: four are booleans and one (VLLM_PP_LAYER_PARTITION) is
# a numeric partition string such as "20,20". The screening in ``_env_value``
# keeps only such values, so a file path can never reach the report.
_USAGE_ENV_VARS_TO_COLLECT = [
    "VLLM_USE_MODELSCOPE",
    "VLLM_USE_FLASHINFER_SAMPLER",
    "VLLM_PP_LAYER_PARTITION",
    "VLLM_USE_TRITON_AWQ",
    "VLLM_ENABLE_V1_MULTIPROCESSING",
]

# A value that may be sent as an environment switch: no separator, no space,
# no dot, so it cannot be a path or a file name.
_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_,+-]{1,64}$")


def _config_home() -> str:
    return envs.VLLM_CONFIG_ROOT


def _usage_stats_json_path() -> str:
    return os.path.join(_config_home(), "usage_stats.json")


def _do_not_track_path() -> str:
    return os.path.join(_config_home(), "do_not_track")


def set_runtime_usage_data(key: str, value: str | int | bool) -> None:
    """Set global usage data that will be sent with every usage heartbeat."""
    _GLOBAL_RUNTIME_DATA[key] = value


def is_usage_stats_enabled() -> bool:
    """Whether ReVolta sends its anonymous usage report.

    ReVolta collects its own statistics (see ``docs/usage-stats.md``), so this
    is on unless one of the following turns it off:

    - ``REVOLTA_NO_USAGE_STATS=1`` (the ReVolta switch)
    - ``VLLM_NO_USAGE_STATS=1``
    - ``VLLM_DO_NOT_TRACK=1`` or ``DO_NOT_TRACK=1``
    - a ``do_not_track`` file in ``VLLM_CONFIG_ROOT``
    """
    if (
        envs.REVOLTA_NO_USAGE_STATS
        or envs.VLLM_NO_USAGE_STATS
        or envs.VLLM_DO_NOT_TRACK
    ):
        return False
    return not os.path.exists(_do_not_track_path())


def _log_usage_notice() -> None:
    """Tell the user once per process that anonymous statistics are sent."""
    global _notice_logged
    if _notice_logged:
        return
    _notice_logged = True
    server = envs.VLLM_USAGE_STATS_SERVER
    target = f"to {server}" if server else "nowhere (VLLM_USAGE_STATS_SERVER is empty)"
    logger.info(
        "ReVolta sends anonymous usage statistics %s: hardware, platform, "
        "CUDA runtime, ReVolta version, model architecture and a configuration "
        "summary (parallelism, KV cache, speculative decoding, max model "
        "length). No prompts, outputs, model paths, host names or IP addresses "
        "are collected. Turn it off with REVOLTA_NO_USAGE_STATS=1 (also "
        "VLLM_NO_USAGE_STATS=1, VLLM_DO_NOT_TRACK=1, DO_NOT_TRACK=1 or a "
        "do_not_track file in VLLM_CONFIG_ROOT). See %s.",
        target,
        _USAGE_STATS_DOCS,
    )


def _get_current_timestamp_ns() -> int:
    return int(datetime.datetime.now(datetime.timezone.utc).timestamp() * 1e9)


def _detect_cloud_provider() -> str:
    # Try detecting through vendor file
    vendor_files = [
        "/sys/class/dmi/id/product_version",
        "/sys/class/dmi/id/bios_vendor",
        "/sys/class/dmi/id/product_name",
        "/sys/class/dmi/id/chassis_asset_tag",
        "/sys/class/dmi/id/sys_vendor",
    ]
    # Mapping of identifiable strings to cloud providers
    cloud_identifiers = {
        "amazon": "AWS",
        "microsoft corporation": "AZURE",
        "google": "GCP",
        "oraclecloud": "OCI",
    }

    for vendor_file in vendor_files:
        path = Path(vendor_file)
        if path.is_file():
            file_content = path.read_text().lower()
            for identifier, provider in cloud_identifiers.items():
                if identifier in file_content:
                    return provider

    # Try detecting through environment variables
    env_to_cloud_provider = {
        "RUNPOD_DC_ID": "RUNPOD",
    }
    for env_var, provider in env_to_cloud_provider.items():
        if os.environ.get(env_var):
            return provider

    return "UNKNOWN"


def _env_value(name: str) -> Any:
    """Value of one collected environment variable, screened for local data.

    Only booleans, numbers and short switch-like strings survive: a value
    that could be a path, a file name or a URL is dropped instead of sent.
    """
    value = getattr(envs, name)
    if isinstance(value, (bool, int)):
        return value
    if isinstance(value, str) and _ENV_VALUE_RE.match(value):
        return value
    return None


def sanitize_usage_value(value: Any) -> Any:
    """Keep a value only if it cannot carry a path, URL or host name.

    Numbers and booleans pass through. Strings must be short identifiers (a
    method or class name); anything with a path separator, a URL scheme or a
    space becomes ``None``, which the collector drops. Free-form operator
    settings therefore never leak a local path into the report.
    """
    if isinstance(value, str) and (
        len(value) > 64 or any(ch in value for ch in ("/", "\\", ":", " "))
    ):
        return None
    return value


class UsageContext(str, Enum):
    UNKNOWN_CONTEXT = "UNKNOWN_CONTEXT"
    LLM_CLASS = "LLM_CLASS"
    API_SERVER = "API_SERVER"
    OPENAI_API_SERVER = "OPENAI_API_SERVER"
    OPENAI_BATCH_RUNNER = "OPENAI_BATCH_RUNNER"
    ENGINE_CONTEXT = "ENGINE_CONTEXT"


class UsageMessage:
    """Collect platform information and send it to the usage stats server."""

    def __init__(self) -> None:
        # NOTE: vLLM's server _only_ support flat KV pair.
        # Do not use nested fields.

        self.uuid = str(uuid4())

        # Environment Information
        self.provider: str | None = None
        self.num_cpu: int | None = None
        self.cpu_type: str | None = None
        self.cpu_family_model_stepping: str | None = None
        self.total_memory: int | None = None
        self.architecture: str | None = None
        self.platform: str | None = None
        self.xpu_runtime: str | None = None
        self.cuda_runtime: str | None = None
        self.gpu_count: int | None = None
        self.gpu_type: str | None = None
        self.gpu_memory_per_device: int | None = None
        self.env_var_json: str | None = None

        # vLLM Information
        self.model_architecture: str | None = None
        self.vllm_version: str | None = None
        self.context: str | None = None

        # ReVolta Information. The distribution version reported here is the
        # ReVolta release, so this field is what separates a ReVolta report
        # from an upstream vLLM report with the same base.
        self.revolta_version: str | None = None

        # Metadata
        self.log_time: int | None = None
        self.source: str | None = None

    def report_usage(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any] | None = None,
    ) -> None:
        _log_usage_notice()
        # One choke point for configuration values: everything coming from the
        # engine configuration is screened here, before the report is built.
        # Hardware and platform fields are collected verbatim.
        kvs = {
            key: sanitize_usage_value(value) for key, value in (extra_kvs or {}).items()
        }
        t = Thread(
            target=self._report_usage_worker,
            args=(model_architecture, usage_context, kvs),
            daemon=True,
        )
        t.start()

    def _report_usage_worker(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        # Nothing in here may reach the user: a broken collector, a read-only
        # configuration directory or an unexpected platform all stop reporting
        # instead of raising in (or delaying) the engine.
        try:
            self._report_usage_once(model_architecture, usage_context, extra_kvs)
            self._report_continuous_usage()
        except Exception as e:  # noqa: BLE001 - telemetry is best effort
            logger.debug("ReVolta usage reporting stopped: %s", type(e).__name__)

    def _report_tpu_inference_usage(self) -> bool:
        try:
            from tpu_inference import tpu_info, utils

            self.gpu_count = tpu_info.get_num_chips()
            self.gpu_type = tpu_info.get_tpu_type()
            self.gpu_memory_per_device = utils.get_device_hbm_limit()
            self.cuda_runtime = "tpu_inference"
            return True
        except Exception:
            return False

    def _report_usage_once(
        self,
        model_architecture: str,
        usage_context: UsageContext,
        extra_kvs: dict[str, Any],
    ) -> None:
        # Platform information
        from vllm.platforms import current_platform

        if current_platform.is_cuda_alike():
            self.gpu_count = current_platform.device_count()
            self.gpu_type, self.gpu_memory_per_device = cuda_get_device_properties(
                0, ("name", "total_memory")
            )
        if current_platform.is_cuda():
            self.cuda_runtime = torch.version.cuda
        if current_platform.is_xpu():
            self.xpu_runtime = torch.version.xpu
            self.gpu_count = torch.xpu.device_count()
            self.gpu_type = torch.xpu.get_device_name(0)
            self.gpu_memory_per_device = torch.xpu.get_device_properties(0).total_memory
        if current_platform.is_tpu():  # noqa: SIM102
            if not self._report_tpu_inference_usage():
                logger.debug("Failed to collect TPU information")
        self.provider = _detect_cloud_provider()
        self.architecture = platform.machine()
        self.platform = platform.platform()
        self.total_memory = psutil.virtual_memory().total

        info = cpuinfo.get_cpu_info()
        self.num_cpu = info.get("count", None)
        self.cpu_type = info.get("brand_raw", "")
        self.cpu_family_model_stepping = ",".join(
            [
                str(info.get("family", "")),
                str(info.get("model", "")),
                str(info.get("stepping", "")),
            ]
        )

        # vLLM information
        self.context = usage_context.value
        self.vllm_version = VLLM_VERSION
        self.revolta_version = VLLM_VERSION
        self.model_architecture = model_architecture

        # Environment variables
        self.env_var_json = json.dumps(
            {
                env_var: value
                for env_var in _USAGE_ENV_VARS_TO_COLLECT
                if (value := _env_value(env_var)) is not None
            }
        )

        # Metadata
        self.log_time = _get_current_timestamp_ns()
        self.source = _source_label()

        data = vars(self)
        if extra_kvs:
            data.update(extra_kvs)

        self._write_to_file(data)
        self._send_to_server(data)

    def _report_continuous_usage(self):
        """Report usage every 10 minutes.

        This helps us to collect more data points for uptime of vLLM usages.
        This function can also help send over performance metrics over time.
        """
        while True:
            time.sleep(_USAGE_STATS_INTERVAL)
            data = {
                "uuid": self.uuid,
                "log_time": _get_current_timestamp_ns(),
            }
            data.update(_GLOBAL_RUNTIME_DATA)

            self._write_to_file(data)
            self._send_to_server(data)

    def _send_to_server(self, data: dict[str, Any]) -> None:
        server = envs.VLLM_USAGE_STATS_SERVER
        if not server:
            return
        try:
            from vllm.connections import global_http_connection

            payload = json.dumps(data, default=str)
            global_http_client = global_http_connection.get_sync_client()
            global_http_client.post(
                server,
                data=payload,
                headers={"Content-Type": "application/json"},
                timeout=_USAGE_STATS_TIMEOUT,
            )
        except Exception as e:  # noqa: BLE001 - telemetry is best effort
            # Silently ignore unless we are using debug log; one attempt only.
            logger.debug("ReVolta usage report dropped: %s", type(e).__name__)

    def _write_to_file(self, data: dict[str, Any]) -> None:
        try:
            path = _usage_stats_json_path()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            Path(path).touch(exist_ok=True)
            with open(path, "a") as f:
                f.write(json.dumps(data, default=str))
                f.write("\n")
        except Exception as e:  # noqa: BLE001 - telemetry is best effort
            logger.debug("ReVolta usage log write skipped: %s", type(e).__name__)


def _source_label() -> str:
    """``VLLM_USAGE_SOURCE`` reduced to a plain label.

    The variable is free-form, so anything that looks like a path or a URL is
    reported as ``"custom"`` instead of being sent.
    """
    source = envs.VLLM_USAGE_SOURCE
    label = sanitize_usage_value(source) if isinstance(source, str) else None
    return label or "custom"


usage_message = UsageMessage()
