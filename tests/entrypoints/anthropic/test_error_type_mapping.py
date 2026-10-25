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
from pydantic import BaseModel, Field, ValidationError

from vllm.entrypoints.anthropic.api_router import (
    attach_router,
    translate_error_response,
)
from vllm.entrypoints.openai.engine.protocol import ErrorInfo, ErrorResponse
from vllm.exceptions import VLLMNotFoundError, VLLMValidationError
from vllm.renderers.params import TokenizeParams

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


def _client_with_handler(handler: MagicMock) -> TestClient:
    app = FastAPI()
    attach_router(app)
    app.state.anthropic_serving_messages = handler
    return TestClient(app)


def _client_with_failing_handler(error: Exception | None = None) -> TestClient:
    if error is None:
        error = RuntimeError("exploded")
    handler = MagicMock()
    handler.create_messages = AsyncMock(side_effect=error)
    handler.count_tokens = AsyncMock(side_effect=error)
    return _client_with_handler(handler)


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


@pytest.mark.parametrize(
    ("path", "stream"),
    [
        ("/v1/messages", False),
        ("/v1/messages", True),
        ("/v1/messages/count_tokens", False),
    ],
)
def test_overlong_prompt_returns_bad_request(path: str, stream: bool):
    """Exercise the real token-length validator without a tokenizer or engine.

    The serving layer raises this exception before returning a stream, too.
    """
    params = TokenizeParams(
        max_total_tokens=8,
        max_output_tokens=0 if path.endswith("count_tokens") else 1,
    )
    with pytest.raises(VLLMValidationError) as exc:
        params.apply_post_tokenization(None, {"prompt_token_ids": list(range(9))})

    body = dict(_REQUEST_BODY)
    body["max_tokens"] = 1
    if path.endswith("count_tokens"):
        body.pop("max_tokens")
    else:
        body["stream"] = stream
    with _client_with_failing_handler(exc.value) as client:
        response = client.post(path, json=body)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json() == {
        "type": "error",
        "error": {"type": "invalid_request_error", "message": str(exc.value)},
    }
    assert "maximum context length is 8 tokens" in str(exc.value)


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
def test_conversion_validation_error_returns_bad_request(path: str):
    """Adapted from vllm-project/vllm#52246; retain Anthropic error types."""

    class StubRequest(BaseModel):
        stop: list[str] = Field(max_length=4)

    with pytest.raises(ValidationError) as exc:
        StubRequest(stop=["a"] * 6)
    body = dict(_REQUEST_BODY)
    if path.endswith("count_tokens"):
        body.pop("max_tokens")
    with _client_with_failing_handler(exc.value) as client:
        response = client.post(path, json=body)
    assert response.status_code == HTTPStatus.BAD_REQUEST
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "at most 4 items" in response.json()["error"]["message"]


@pytest.mark.parametrize("path", ["/v1/messages", "/v1/messages/count_tokens"])
@pytest.mark.parametrize("raised", [True, False])
def test_unknown_model_returns_not_found(path: str, raised: bool):
    message = "The model `missing-model` does not exist."
    if raised:
        client = _client_with_failing_handler(VLLMNotFoundError(message))
    else:
        error = ErrorResponse(
            error=ErrorInfo(
                message=message, type="NotFoundError", code=HTTPStatus.NOT_FOUND
            )
        )
        handler = MagicMock()
        handler.create_messages = AsyncMock(return_value=error)
        handler.count_tokens = AsyncMock(return_value=error)
        client = _client_with_handler(handler)
    body = dict(_REQUEST_BODY, model="missing-model")
    if path.endswith("count_tokens"):
        body.pop("max_tokens")
    with client:
        response = client.post(path, json=body)
    assert response.status_code == HTTPStatus.NOT_FOUND
    assert response.json() == {
        "type": "error",
        "error": {"type": "not_found_error", "message": message},
    }


@pytest.mark.parametrize("error", [ValueError("late validation"), RuntimeError("boom")])
def test_mid_stream_exception_keeps_http_ok_and_api_error(error: Exception):
    """Once SSE starts, the router's pre-stream exception mapping cannot run."""
    from vllm.entrypoints.anthropic.serving import AnthropicServingMessages

    async def upstream():
        chunk = {
            "id": "chatcmpl-test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": {"role": "assistant"}}],
        }
        yield f"data: {json.dumps(chunk)}\n\n"
        chunk["choices"] = [{"index": 0, "delta": {"content": "hello"}}]
        yield f"data: {json.dumps(chunk)}\n\n"
        raise error

    converter = AnthropicServingMessages.__new__(AnthropicServingMessages)
    handler = MagicMock()
    handler.create_messages = AsyncMock(
        return_value=converter.message_stream_converter(upstream())
    )
    with _client_with_handler(handler) as client:
        response = client.post("/v1/messages", json=dict(_REQUEST_BODY, stream=True))
    assert response.status_code == HTTPStatus.OK
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: message_start" in response.text
    assert "hello" in response.text
    assert _sse_error_payloads(response.text.split("\n\n")) == [
        {
            "type": "error",
            "error": {"type": "api_error", "message": str(error)},
        }
    ]
    assert "event: message_stop" not in response.text
