# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""Exercise the real instrumentation attachment with a CPU Prometheus registry.

Only the engine-owned registry provider and package bootstrap are replaced; the
actual instrumentator, dashboard routes, exporter and ASGI stack are executed.
"""

import importlib.util
import sys
import types
import unittest
from argparse import Namespace
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter
from test_routes import ROOT, dashboard


class DashboardWiringTest(unittest.TestCase):
    def client(self, enabled, cli_keys=None, env_key=None):
        registry = CollectorRegistry()
        Counter("vllm:prompt_tokens", "Input", registry=registry).inc(123)
        vllm = types.ModuleType("vllm")
        vars(vllm)["envs"] = Namespace(VLLM_API_KEY=env_key)
        provider = types.ModuleType("vllm.v1.metrics.prometheus")
        vars(provider)["get_prometheus_registry"] = lambda: registry
        version = types.ModuleType("vllm.version")
        vars(version)["__version__"] = "0.0.0+cpu-wiring"
        spec = importlib.util.spec_from_file_location(
            "instrumentator_metrics",
            ROOT / "vllm/entrypoints/serve/instrumentator/metrics.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        app = FastAPI()
        app.state.args = Namespace(enable_dashboard=enabled, api_key=cli_keys)
        with patch.dict(
            sys.modules,
            {
                "vllm": vllm,
                "vllm.v1.metrics.prometheus": provider,
                "vllm.entrypoints.serve.dashboard.api_router": dashboard,
                "vllm.version": version,
            },
        ):
            spec.loader.exec_module(module)
            module.attach_router(app)
        return TestClient(app)

    def test_enabled_real_attachment_and_existing_scrape(self):
        with self.client(True) as client:
            self.assertEqual(client.get("/dashboard/").status_code, 200)
            native = client.get("/metrics")
            browser = client.get("/dashboard/metrics")
            self.assertEqual(native.status_code, 200)
            self.assertEqual(browser.status_code, 200)
            self.assertIn("vllm:prompt_tokens_total 123.0", native.text)
            self.assertIn("vllm:prompt_tokens_total 123.0", browser.text)

    def test_disabled_preserves_existing_scrape(self):
        with self.client(False) as client:
            self.assertEqual(client.get("/dashboard/").status_code, 404)
            self.assertEqual(client.get("/dashboard/metrics").status_code, 404)
            self.assertEqual(client.get("/metrics").status_code, 200)

    def test_cli_key_takes_priority_over_environment_key(self):
        with self.client(True, ["cli-test-key"], "env-test-key") as client:
            self.assertEqual(client.get("/dashboard/metrics").status_code, 401)
            self.assertEqual(
                client.get(
                    "/dashboard/metrics",
                    headers={"Authorization": "Bearer env-test-key"},
                ).status_code,
                401,
            )
            self.assertEqual(
                client.get(
                    "/dashboard/metrics",
                    headers={"Authorization": "Bearer cli-test-key"},
                ).status_code,
                200,
            )
            self.assertEqual(client.get("/metrics").status_code, 200)

    def test_environment_key_protects_dashboard_data(self):
        with self.client(True, env_key="env-test-key") as client:
            self.assertEqual(client.get("/dashboard/config").status_code, 401)
            response = client.get(
                "/dashboard/config", headers={"Authorization": "Bearer env-test-key"}
            )
            self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main()
