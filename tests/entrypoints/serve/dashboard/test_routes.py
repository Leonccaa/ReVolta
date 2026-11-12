# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""CPU-only ASGI tests, runnable without importing the CUDA engine or conftest.

Run: python -m unittest discover -s tests/entrypoints/serve/dashboard -p 'test_*.py'
"""

import ast
import importlib.util
import logging
import unittest
from argparse import Namespace
from copy import deepcopy
from http import HTTPStatus
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry, Counter
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[4]
SPEC = importlib.util.spec_from_file_location(
    "dashboard_router", ROOT / "vllm/entrypoints/serve/dashboard/api_router.py"
)
assert SPEC is not None and SPEC.loader is not None
dashboard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dashboard)


def extract(names, path, namespace):
    """Load selected top-level definitions without importing the engine stack."""
    tree = ast.parse(path.read_text())
    found = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
    }
    assert set(names) <= set(found), f"Missing {sorted(set(names) - set(found))}"
    body: list[ast.stmt] = [found[name] for name in names]
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)


def api_server_exception_chain_app(keys=()):
    """Dashboard routes under the real API server HTTPException handler.

    `vllm.entrypoints.openai.server_utils` and its error DTOs are loaded with
    their original source; only the OpenAI model base class is replaced by
    pydantic's BaseModel so no engine or CUDA import is needed.
    """
    namespace = {
        "BaseModel": BaseModel,
        "HTTPStatus": HTTPStatus,
        "HTTPException": HTTPException,
        "Request": Request,
        "JSONResponse": JSONResponse,
        "OpenAIBaseModel": BaseModel,
        "logger": logging.getLogger("dashboard-test"),
        "sanitize_message": str,
    }
    extract(
        ["ErrorInfo", "ErrorResponse"],
        ROOT / "vllm/entrypoints/openai/engine/protocol.py",
        namespace,
    )
    extract(
        ["http_exception_handler"],
        ROOT / "vllm/entrypoints/openai/server_utils.py",
        namespace,
    )
    app = FastAPI()
    app.state.args = Namespace(enable_dashboard=True, log_error_stack=False)
    # api_server registers the handler with exactly this call.
    app.exception_handler(HTTPException)(namespace["http_exception_handler"])
    registry = CollectorRegistry()
    Counter("vllm:prompt_tokens", "Input", registry=registry).inc(100)
    dashboard.attach_router(app, registry, keys, version="0.0.0+cpu-chain")
    return app


def make_app(enabled=True, keys=(), root_path="", args=None):
    app = FastAPI(root_path=root_path)
    app.state.args = args or Namespace(enable_dashboard=enabled)
    registry = CollectorRegistry()
    Counter("vllm:prompt_tokens", "Input", registry=registry).inc(100)
    Counter("vllm:generation_tokens", "Output", registry=registry).inc(10)
    dashboard.attach_router(app, registry, keys, version="0.0.0+cpu-test")
    return app


class DashboardRoutesTest(unittest.TestCase):
    def test_disabled_has_no_dashboard_routes(self):
        with TestClient(make_app(enabled=False)) as client:
            for path in (
                "/dashboard",
                "/dashboard/",
                "/dashboard/config",
                "/dashboard/metrics",
                "/dashboard/assets/dashboard.mjs",
            ):
                self.assertEqual(client.get(path).status_code, 404)

    def test_default_missing_args_has_no_routes(self):
        app = FastAPI()
        dashboard.attach_router(app, CollectorRegistry(), version="dev")
        self.assertEqual(TestClient(app).get("/dashboard/").status_code, 404)

    def test_enabled_normal_browser_entrypoint_and_packaged_assets(self):
        with TestClient(make_app()) as client:
            response = client.get("/dashboard")
            self.assertEqual(response.status_code, 200)
            self.assertTrue(str(response.url).endswith("/dashboard/"))
            self.assertIn("assets/dashboard.mjs", response.text)
            self.assertIn(
                "script-src 'self'", response.headers["Content-Security-Policy"]
            )
            for asset in dashboard.ASSETS:
                result = client.get("/dashboard/assets/" + asset)
                self.assertEqual(result.status_code, 200)
                self.assertEqual(result.headers["Cache-Control"], "no-cache")
                expected = "text/css" if asset.endswith(".css") else "text/javascript"
                self.assertTrue(result.headers["Content-Type"].startswith(expected))

    def test_auth_uses_bearer_keys_without_embedding_secrets_in_shell(self):
        with TestClient(make_app(keys=("test-primary", "test-secondary"))) as client:
            for path in ("/dashboard/config", "/dashboard/metrics"):
                for headers in (
                    {},
                    {"Authorization": "Bearer wrong"},
                    {"Authorization": "Basic test-primary"},
                ):
                    response = client.get(path, headers=headers)
                    self.assertEqual(response.status_code, 401)
                    self.assertEqual(response.headers["Cache-Control"], "no-store")
                for key in ("test-primary", "test-secondary"):
                    response = client.get(
                        path, headers={"Authorization": "bEaReR " + key}
                    )
                    self.assertEqual(response.status_code, 200)
                    self.assertNotIn(key, response.text)
            self.assertNotIn("test-primary", client.get("/dashboard/").text)

    def test_static_assets_support_conditional_browser_caching(self):
        with TestClient(make_app()) as client:
            first = client.get("/dashboard/assets/dashboard.mjs")
            cached = client.get(
                "/dashboard/assets/dashboard.mjs",
                headers={"If-None-Match": first.headers["ETag"]},
            )
            self.assertEqual(cached.status_code, 304)
            self.assertEqual(cached.content, b"")
            self.assertEqual(cached.headers["Cache-Control"], "no-cache")

    def test_metric_registry_passes_through_and_metadata_uses_actual_version(self):
        with TestClient(make_app()) as client:
            config = client.get("/dashboard/config")
            metrics = client.get("/dashboard/metrics")
            self.assertEqual(config.json()["version"], "0.0.0+cpu-test")
            self.assertEqual(
                config.json()["instance_id"], metrics.headers["X-ReVolta-Instance-Id"]
            )
            self.assertGreater(config.json()["started_at"], 0)
            self.assertIn("vllm:prompt_tokens_total 100.0", metrics.text)
            self.assertIn("vllm:generation_tokens_total 10.0", metrics.text)
            self.assertTrue(
                metrics.headers["Content-Type"].startswith("text/plain; version=")
            )
            self.assertEqual(metrics.headers["Cache-Control"], "no-store")

    def test_initialization_is_shared_across_worker_arg_copies(self):
        args = Namespace(enable_dashboard=True)
        dashboard.initialize_dashboard(args)
        worker_args = deepcopy(args)
        with (
            TestClient(make_app(args=args)) as a,
            TestClient(make_app(args=worker_args)) as b,
        ):
            self.assertEqual(
                a.get("/dashboard/config").json(), b.get("/dashboard/config").json()
            )
        with TestClient(make_app()) as c:
            self.assertNotEqual(
                c.get("/dashboard/config").json()["instance_id"],
                args._dashboard_instance_id,
            )

    def test_reverse_proxy_root_path_and_relative_assets(self):
        with TestClient(make_app(root_path="/inference-proxy")) as client:
            redirect = client.get("/inference-proxy/dashboard", follow_redirects=False)
            self.assertTrue(
                redirect.headers["Location"].endswith("/inference-proxy/dashboard/")
            )
            self.assertEqual(
                client.get("/inference-proxy/dashboard/assets/metrics.mjs").status_code,
                200,
            )

    def test_openai_server_exception_chain_keeps_rejection_headers(self):
        app = api_server_exception_chain_app(keys=("chain-test-key",))
        with TestClient(app) as client:
            for path in ("/dashboard/config", "/dashboard/metrics"):
                denied = client.get(path)
                self.assertEqual(denied.status_code, 401)
                self.assertEqual(denied.headers["WWW-Authenticate"], "Bearer")
                self.assertEqual(denied.headers["Cache-Control"], "no-store")
                self.assertEqual(
                    denied.json()["error"]["code"], HTTPStatus.UNAUTHORIZED.value
                )
            for path in ("/dashboard/config", "/dashboard/metrics"):
                accepted = client.get(
                    path, headers={"Authorization": "Bearer chain-test-key"}
                )
                self.assertEqual(accepted.status_code, 200)
                self.assertEqual(accepted.headers["Cache-Control"], "no-store")

    def test_openai_server_exception_chain_keeps_dead_engine_headers(self):
        app = api_server_exception_chain_app()
        app.state.engine_client = Namespace(errored=True)
        with TestClient(app) as client:
            response = client.get("/dashboard/metrics")
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["Cache-Control"], "no-store")
            self.assertEqual(response.json()["error"]["code"], 503)

    def test_asset_allowlist_blocks_arbitrary_files(self):
        with TestClient(make_app()) as client:
            for asset in ("api_router.py", "index.html", "%2E%2E%2Fapi_router.py"):
                self.assertEqual(
                    client.get("/dashboard/assets/" + asset).status_code, 404
                )

    def test_dead_engine_returns_unavailable_without_engine_rpc(self):
        app = make_app()
        app.state.engine_client = Namespace(errored=True)
        with TestClient(app) as client:
            self.assertEqual(client.get("/dashboard/metrics").status_code, 503)

    def test_no_request_payload_or_schema_routes_added(self):
        app = make_app()
        for route in app.routes:
            if getattr(route, "path", "").startswith("/dashboard"):
                self.assertEqual(route.methods, {"GET"})
        self.assertNotIn("dashboard", TestClient(app).get("/openapi.json").text)


if __name__ == "__main__":
    unittest.main()
