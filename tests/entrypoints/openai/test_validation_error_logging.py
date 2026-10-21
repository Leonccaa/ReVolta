# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import logging
from argparse import Namespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.testclient import TestClient

from vllm.entrypoints.openai.api_server import build_app
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.entrypoints.openai.server_utils import (
    http_exception_handler,
    validation_exception_handler,
)
from vllm.exceptions import VLLMValidationError


@pytest.fixture(autouse=True)
def capture_server_logger(caplog, monkeypatch):
    logger = logging.getLogger("vllm.entrypoints.openai.server_utils")
    monkeypatch.setattr(logger, "propagate", False)
    logger.addHandler(caplog.handler)
    yield
    logger.removeHandler(caplog.handler)


@pytest.mark.parametrize("failure", ["pydantic", "http"])
@pytest.mark.parametrize("log_error_stack", [False, True])
def test_bad_request_logs_one_warning_without_traceback(
    failure, log_error_stack, caplog
):
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=log_error_stack)
    app.exception_handler(HTTPException)(http_exception_handler)
    app.exception_handler(RequestValidationError)(validation_exception_handler)

    @app.post("/v1/completions")
    async def completion(request: CompletionRequest):
        raise HTTPException(400, detail="Invalid request parameter\nsecond line")

    payload = {"model": "m", "prompt": "test", "n": 0 if failure == "pydantic" else 1}
    with caplog.at_level(
        logging.WARNING, logger="vllm.entrypoints.openai.server_utils"
    ):
        response = TestClient(app).post("/v1/completions", json=payload)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == 400
    records = [
        r for r in caplog.records if r.name == "vllm.entrypoints.openai.server_utils"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is None
    assert "\n" not in records[0].getMessage()


def test_server_http_error_retains_traceback(caplog):
    app = FastAPI()
    app.state.args = Namespace(log_error_stack=True)
    app.exception_handler(HTTPException)(http_exception_handler)

    @app.get("/failure")
    async def failure():
        raise HTTPException(500, detail="server error")

    with caplog.at_level(logging.ERROR, logger="vllm.entrypoints.openai.server_utils"):
        response = TestClient(app).get("/failure")
    assert response.status_code == 500
    assert any(r.exc_info for r in caplog.records)


@pytest.mark.parametrize(
    "error_type", [ValueError, TypeError, OverflowError, VLLMValidationError]
)
def test_parameter_error_does_not_escape_asgi(error_type, caplog):
    args = Namespace(
        disable_fastapi_docs=True,
        enable_offline_docs=False,
        root_path="",
        allowed_origins=["*"],
        allow_credentials=False,
        allowed_methods=["*"],
        allowed_headers=["*"],
        api_key=None,
        enable_request_id_headers=False,
        middleware=[],
        log_error_stack=True,
    )
    app = build_app(args, supported_tasks=())

    @app.get("/invalid-parameter")
    async def invalid_parameter():
        raise error_type("n must be 1 when using greedy sampling")

    with caplog.at_level(
        logging.WARNING, logger="vllm.entrypoints.openai.server_utils"
    ):
        # Default raise_server_exceptions=True proves no exception reaches
        # uvicorn after the 400 response (the old catch-all handler re-raises).
        response = TestClient(app).get("/invalid-parameter")
    assert response.status_code == 400
    assert response.json()["error"] == {
        "message": "n must be 1 when using greedy sampling",
        "type": "BadRequestError",
        "code": 400,
        "param": None,
    }
    records = [
        r for r in caplog.records if r.name == "vllm.entrypoints.openai.server_utils"
    ]
    assert len(records) == 1
    assert records[0].levelno == logging.WARNING
    assert records[0].exc_info is None
