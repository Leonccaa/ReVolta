# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""Opt-in static dashboard. No sampling task, engine RPC or history on server."""

import hashlib
import secrets
import time
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest

STATIC = Path(__file__).with_name("static")
ASSETS = {"dashboard.mjs", "metrics.mjs", "history.mjs", "dashboard.css"}
HEADERS = {
    "Cache-Control": "no-cache",
    "X-Content-Type-Options": "nosniff",
    "Content-Security-Policy": (
        "default-src 'self'; script-src 'self'; "
        "style-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self'; object-src 'none'; base-uri 'self'; "
        "frame-ancestors 'self'; form-action 'none'"
    ),
}


def initialize_dashboard(args: Namespace) -> None:
    """Initialize once before API workers fork, sharing one service identity.

    The time is service startup, including model loading, not a GPU timer.
    The fallback in attach_router also supports embedded single-worker apps.
    """
    if not hasattr(args, "_dashboard_instance_id"):
        args._dashboard_instance_id = uuid4().hex
        args._dashboard_started_at = time.time()


def attach_router(
    app: FastAPI,
    registry: CollectorRegistry,
    api_keys: Sequence[str] = (),
    *,
    version: str,
) -> None:
    args = getattr(app.state, "args", None)
    if args is None or not getattr(args, "enable_dashboard", False):
        return
    initialize_dashboard(args)
    static_files = StaticFiles(directory=STATIC)
    token_hashes = [hashlib.sha256(key.encode()).digest() for key in api_keys]
    data_headers = {
        "Cache-Control": "no-store",
        "X-Content-Type-Options": "nosniff",
        "X-ReVolta-Instance-Id": args._dashboard_instance_id,
        "X-ReVolta-Started-At": str(args._dashboard_started_at),
    }

    def authorize(request: Request) -> None:
        if not token_hashes:
            return
        scheme, _, token = request.headers.get("Authorization", "").partition(" ")
        digest = hashlib.sha256(token.encode()).digest()
        matches = [secrets.compare_digest(digest, key) for key in token_hashes]
        if scheme.lower() != "bearer" or not any(matches):
            raise HTTPException(
                status_code=401,
                detail="Unauthorized",
                headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
            )

    @app.get("/dashboard", include_in_schema=False)
    async def dashboard_redirect(request: Request) -> RedirectResponse:
        return RedirectResponse(request.url_for("dashboard_index"))

    @app.get("/dashboard/", include_in_schema=False)
    async def dashboard_index() -> FileResponse:
        # The shell contains no metrics and lets browsers enter a key in memory.
        return FileResponse(STATIC / "index.html", headers=HEADERS)

    @app.get("/dashboard/assets/{asset}", include_in_schema=False)
    async def dashboard_asset(asset: str, request: Request) -> Response:
        if asset not in ASSETS:
            raise HTTPException(status_code=404)
        media_type = "text/css" if asset.endswith(".css") else "text/javascript"
        response = await static_files.get_response(asset, request.scope)
        response.headers.update(HEADERS)
        response.headers["Content-Type"] = media_type
        return response

    @app.get(
        "/dashboard/config", dependencies=[Depends(authorize)], include_in_schema=False
    )
    def dashboard_config() -> JSONResponse:
        return JSONResponse(
            {
                "instance_id": args._dashboard_instance_id,
                "started_at": args._dashboard_started_at,
                "version": version,
            },
            headers=data_headers,
        )

    @app.get(
        "/dashboard/metrics", dependencies=[Depends(authorize)], include_in_schema=False
    )
    def dashboard_metrics() -> Response:
        # Reuse the ordinary scrape registry; a sync endpoint runs collection in
        # FastAPI's thread pool. Do not contact EngineCore or maintain a sampler.
        engine = getattr(app.state, "engine_client", None)
        if engine is not None and getattr(engine, "errored", False):
            raise HTTPException(status_code=503, detail="Engine unavailable")
        return Response(
            generate_latest(registry),
            headers={**data_headers, "Content-Type": CONTENT_TYPE_LATEST},
        )
