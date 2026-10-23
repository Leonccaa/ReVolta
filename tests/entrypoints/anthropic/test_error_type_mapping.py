# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Anthropic ``error.type`` translation (issue #57997)."""

import asyncio
import json
from http import HTTPStatus
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vllm.entrypoints.anthropic.api_router import (
    attach_router,
    translate_error_response,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (400, "invalid_request_error"),
        (401, "authentication_error"),
        (403, "permission_error"),
        (404, "not_found_error"),
        (413, "request_too_large"),
        (418, "invalid_request_error"),
        (422, "invalid_request_error"),
        (429, "rate_limit_error"),
        (500, "api_error"),
        (501, "api_error"),
        (503, "overloaded_error"),
    ],
)
def test_error_type_is_translated_to_anthropic_vocabulary(
    status_code: int, expected_type: str
):
    response = translate_error_response(
        ErrorResponse(
            error=ErrorInfo(
                message="boom",
                # Deliberately use an internal type name to prove it is ignored.
                type="NotFoundError",
                code=status_code,
            )
        )
    )
    assert response.status_code == status_code
    assert json.loads(response.body) == {
        "type": "error",
        "error": {"type": expected_type, "message": "boom"},
    }


def _convert(generator) -> list[str]:
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    # The converter's error paths do not touch instance attributes, so
    # __new__ bypasses __init__ safely here.
    converter = AnthropicServingMessages.__new__(AnthropicServingMessages)

    async def collect() -> list[str]:
        return [e async for e in converter.message_stream_converter(generator)]

    return asyncio.run(collect())


def _sse_error_payloads(events: list[str]) -> list[dict]:
    error_events = [e for e in events if e.startswith("event: error")]
    payloads = []
    for event in error_events:
        payload_line = [
            line for line in event.splitlines() if line.startswith("data: ")
        ][0]
        payloads.append(json.loads(payload_line[len("data: ") :]))
    return payloads


@pytest.mark.parametrize(
    ("status_code", "expected_type"),
    [
        (400, "invalid_request_error"),
        (404, "not_found_error"),
        (429, "rate_limit_error"),
        (503, "overloaded_error"),
        (500, "api_error"),
    ],
)
def test_stream_error_event_uses_translated_type(status_code: int, expected_type: str):
    """The SSE error path must translate ``error.type`` the same way as the
    non-streaming path, reusing the in-band ErrorResponse status code."""
    error = ErrorResponse(
        error=ErrorInfo(message="boom", type="InternalError", code=status_code)
    )

    async def fake_generator():
        # The chat completion stream reports failures in-band like this.
        yield f"data: {json.dumps(error.model_dump())}\n\n"
        yield "data: [DONE]\n\n"

    events = _convert(fake_generator())
    payloads = _sse_error_payloads(events)
    assert len(payloads) == 1
    assert payloads[0]["type"] == "error"
    assert payloads[0]["error"]["type"] == expected_type
    assert payloads[0]["error"]["message"] == "boom"
    assert not any(e.startswith("event: message_stop") for e in events)


def test_stream_mid_stream_exception_maps_to_api_error():
    """An exception raised mid-stream (not an ErrorResponse) surfaces as
    ``api_error`` in the SSE error event."""

    async def failing_generator():
        raise RuntimeError("kaboom")
        yield  # pragma: no cover - make this an async generator

    events = _convert(failing_generator())
    payloads = _sse_error_payloads(events)
    assert len(payloads) == 1
    assert payloads[0]["error"]["type"] == "api_error"
    assert payloads[0]["error"]["message"] == "kaboom"


def test_stream_invalid_data_format_maps_to_api_error():
    async def bad_generator():
        yield "not an sse data line"

    events = _convert(bad_generator())
    payloads = _sse_error_payloads(events)
    assert len(payloads) == 1
    assert payloads[0]["error"] == {
        "type": "api_error",
        "message": "Invalid data format received",
    }


def _client_with_failing_handler() -> TestClient:
    app = FastAPI()
    attach_router(app)
    handler = MagicMock()
    handler.create_messages = AsyncMock(side_effect=RuntimeError("exploded"))
    handler.count_tokens = AsyncMock(side_effect=RuntimeError("exploded"))
    app.state.anthropic_serving_messages = handler
    return TestClient(app)


_REQUEST_BODY = {
    "model": "test-model",
    "max_tokens": 16,
    "messages": [{"role": "user", "content": "hi"}],
}


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_unhandled_handler_exception_maps_to_api_error(path: str):
    """Unexpected handler exceptions keep HTTP 500 and use ``api_error``."""
    body = dict(_REQUEST_BODY)
    if path.endswith("count_tokens"):
        body.pop("max_tokens")
    response = _client_with_failing_handler().post(path, json=body)
    assert response.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert response.json() == {
        "type": "error",
        "error": {"type": "api_error", "message": "exploded"},
    }
