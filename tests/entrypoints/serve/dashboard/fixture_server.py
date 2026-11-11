# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""Synthetic CPU fixture for browser checks. Not an inference server."""

import argparse
import socket
import time
from types import SimpleNamespace
from typing import TypedDict

import uvicorn
from fastapi import FastAPI, Request
from prometheus_client import CollectorRegistry
from prometheus_client.parser import text_string_to_metric_families
from test_routes import dashboard


class FixtureState(TypedDict):
    start: float
    mode: str
    frozen: int
    auth: bool


state: FixtureState = {
    "start": time.monotonic(),
    "mode": "normal",
    "frozen": 0,
    "auth": False,
}
scrapes = 0


def native_metrics():
    mode = state["mode"]
    if mode == "missing-all":
        return ""
    step = (
        state["frozen"]
        if mode == "idle"
        else int((time.monotonic() - state["start"]) / 2)
    )
    model = "example/cpu-fixture"
    if mode == "escaped-label":
        model = "example/<img src=x onerror=alert(1)>"
    values: dict[str, float] = {
        "prompt_tokens_total": 1000000 + 3000 * step,
        "generation_tokens_total": 48000 + 144 * step,
        "prompt_tokens_cached_total": 800000 + 2400 * step,
        "request_success_total": 100 + step,
        "num_requests_running": 0 if mode == "idle" else 2,
        "num_requests_waiting": 0,
        "num_preemptions_total": 0,
        "kv_cache_usage_perc": 0.45,
        "request_prefill_time_seconds_sum": 8 + 0.45 * step,
        "request_prefill_time_seconds_count": 100 + step,
        "request_prefill_kv_computed_tokens_sum": 200000 + 600 * step,
        "request_prefill_kv_computed_tokens_count": 100 + step,
        "time_to_first_token_seconds_sum": 50 + 0.5 * step,
        "time_to_first_token_seconds_count": 100 + step,
        "inter_token_latency_seconds_sum": 600 + 2 * step,
        "inter_token_latency_seconds_count": 48000 + 144 * step,
        "request_time_per_output_token_seconds_sum": 2 + 0.02 * step,
        "request_time_per_output_token_seconds_count": 100 + step,
        "request_queue_time_seconds_sum": 4 + 0.04 * step,
        "request_queue_time_seconds_count": 100 + step,
        "e2e_request_latency_seconds_sum": 300 + 3 * step,
        "e2e_request_latency_seconds_count": 100 + step,
        "spec_decode_num_accepted_tokens_total": 20000 + 8 * step,
        "spec_decode_num_draft_tokens_total": 100000 + 10 * step,
        "spec_decode_num_drafts_total": 10000 + 2 * step,
    }
    if mode == "missing-optional":
        values = {
            key: value
            for key, value in values.items()
            if key
            in {
                "prompt_tokens_total",
                "generation_tokens_total",
                "request_success_total",
                "num_requests_running",
                "num_requests_waiting",
                "kv_cache_usage_perc",
            }
        }
    label = 'model_name="' + model + '",engine="0"'
    lines = [f"vllm:{key}{{{label}}} {value}" for key, value in values.items()]
    if mode == "missing-optional":
        return "\n".join(lines) + "\n"
    value: float
    for source, value in {
        "local_compute": 200000 + 600 * step,
        "local_cache_hit": 400000 + 1200 * step,
        "external_kv_transfer": 400000 + 1200 * step,
    }.items():
        lines.append(
            f'vllm:prompt_tokens_by_source_total{{{label},source="{source}"}} {value}'
        )
    for metric, value in {
        "unified_region_bytes": 128 * 1024**3,
        "unified_allocated_main_bytes": 56 * 1024**3 + step * 1048576,
        "unified_allocated_state_bytes": 6 * 1024**3,
        "unified_allocated_draft_bytes": 2 * 1024**3,
        "unified_ssd_bytes": 96 * 1024**3 + step * 2097152,
        "unified_prefetch_verified_bytes": 32 * 1024**3 + step * 1048576,
        "unified_committed_bytes": 64 * 1024**3 + step * 2097152,
        "gpu_hot_capacity_slots": 1000,
        "gpu_hot_valid_slots": 630,
        "store_snapshot_unix_s": time.time() - (300 if mode == "stale-store" else 0),
    }.items():
        lines.append(f'vllm:store_state{{{label},metric="{metric}"}} {value}')
    for event, increment in {
        "prompt_tokens_gpu": 600,
        "prompt_tokens_host_reuse": 800,
        "prompt_tokens_host_load": 200,
        "prompt_tokens_ssd": 800,
        "prompt_tokens_compute": 600,
        "gpu_hot_wanted_groups_total": 10,
        "gpu_hot_miss_groups_total": 2,
    }.items():
        lines.append(
            f'vllm:store_events_total{{{label},event="{event}"}} '
            f"{increment * (100 + step)}"
        )
    for bound in ("0.1", "0.5", "1.0", "+Inf"):
        count = 0 if bound == "0.1" else 100 + step
        lines.append(
            f'vllm:time_to_first_token_seconds_bucket{{{label},le="{bound}"}} {count}'
        )
    return "\n".join(lines) + "\n"


class FixtureCollector:
    def collect(self):
        global scrapes
        scrapes += 1
        if state["mode"] == "slow":
            time.sleep(0.4)
        yield from text_string_to_metric_families(native_metrics())


def make_fixture():
    app = FastAPI()
    app.state.args = SimpleNamespace(enable_dashboard=True)
    app.state.engine_client = SimpleNamespace(errored=state["mode"] == "offline")
    registry = CollectorRegistry()
    registry.register(FixtureCollector())
    keys = ["cpu-test-key"] if state["auth"] else []
    dashboard.initialize_dashboard(app.state.args)
    app.state.args._dashboard_started_at = time.time() - 3600
    dashboard.attach_router(app, registry, keys, version="0.0.0+cpu-fixture")
    return app


gateway = FastAPI()


@gateway.post("/__fixture")
async def control(request: Request):
    global fixture
    command = await request.json()
    if "mode" in command:
        mode = command["mode"]
        if mode == "restart":
            state["start"] = time.monotonic()
            mode = "normal"
        if mode == "idle":
            state["frozen"] = int((time.monotonic() - state["start"]) / 2)
        state["mode"] = mode
        fixture.state.engine_client.errored = mode == "offline"
    rebuild = command.get("mode") == "restart" or "auth" in command
    if "auth" in command:
        state["auth"] = bool(command["auth"])
    if rebuild:
        fixture = make_fixture()
        gateway.routes[-1].app = fixture
    return {"scrapes": scrapes, "mode": state["mode"]}


fixture = make_fixture()
gateway.mount("/", fixture)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=0)
    options = parser.parse_args()
    server_socket = socket.socket()
    server_socket.bind(("127.0.0.1", options.port))
    print(server_socket.getsockname()[1], flush=True)
    config = uvicorn.Config(gateway, log_level="warning")
    uvicorn.Server(config).run(sockets=[server_socket])
