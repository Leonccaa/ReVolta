# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exercise real routes, SSE serialization and Starlette disconnect handling."""

import asyncio
import json
from contextlib import AsyncExitStack
from types import SimpleNamespace

import anyio
import pytest
from fastapi import FastAPI
from pydantic import BaseModel

from vllm.entrypoints.anthropic.api_router import router as messages_router
from vllm.entrypoints.openai.chat_completion.api_router import router as chat_router
from vllm.entrypoints.openai.completion.api_router import router as completion_router
from vllm.entrypoints.openai.responses.api_router import router as responses_router

pytestmark = [pytest.mark.asyncio, pytest.mark.skip_global_cleanup]
COMMENT = b": keep-alive\n\n"
ROUTES = ["chat", "completion", "responses", "retrieve", "messages"]


class _Event(BaseModel):
    type: str = "response.output_text.delta"
    delta: str


def _request(route):
    if route == "chat":
        return "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}
    if route == "completion":
        return "/v1/completions", {"prompt": "hi"}
    if route == "responses":
        return "/v1/responses", {"input": "hi"}
    if route == "retrieve":
        return "/v1/responses/resp_test", None
    return "/v1/messages", {
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 4,
    }


async def _run(route, interval, scenario="fast", harmony=False):
    app = FastAPI()
    for router in [chat_router, completion_router, responses_router, messages_router]:
        app.include_router(router)
    if interval is not None:
        app.state.args = SimpleNamespace(sse_keep_alive_interval=interval)
    state = SimpleNamespace(closed=False, aborted=False, comments=0)
    started = asyncio.Event()
    first_data = asyncio.Event()
    stalled = asyncio.Event()
    chunks = ["first 中文\n", "second", "done"]
    is_responses = route in ("responses", "retrieve")

    async def generate():
        try:
            async with AsyncExitStack() as stack:
                if harmony:
                    stack.enter_context(anyio.CancelScope())
                for i, text in enumerate(chunks):
                    if scenario == "disconnect_idle" or (
                        scenario == "disconnect_between" and i == 1
                    ):
                        stalled.set()
                        await asyncio.Event().wait()
                    if scenario in ("idle", "error_frame", "exception"):
                        await asyncio.sleep(0.03)
                    if scenario == "exception" and i == 1:
                        raise RuntimeError("stream failed")
                    if scenario == "error_frame" and i == 1:
                        text = '{"error":{"message":"stream failed"}}'
                    if is_responses:
                        yield _Event(delta=text)
                    else:
                        yield f"data: {text}\n\n"
        except (asyncio.CancelledError, GeneratorExit):
            state.aborted = True
            raise
        finally:
            state.closed = True

    async def handler(*args, **kwargs):
        return generate()

    serving = SimpleNamespace(
        use_harmony=harmony,
        create_chat_completion=handler,
        create_completion=handler,
        create_responses=handler,
        retrieve_responses=handler,
        create_messages=handler,
    )
    app.state.openai_serving_chat = serving
    app.state.openai_serving_completion = serving
    app.state.openai_serving_responses = serving
    app.state.anthropic_serving_messages = serving
    path, payload = _request(route)
    body = (
        json.dumps({"model": "test", "stream": True, **payload}).encode()
        if payload
        else b""
    )
    sent_request = False
    output = []
    status = []

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": body, "more_body": False}
        if scenario.startswith("disconnect"):
            await started.wait()
            if scenario == "disconnect_between":
                await first_data.wait()
            await stalled.wait()
            # Ensure anext is pending before asking Starlette to cancel it.
            await asyncio.sleep(0.005)
            return {"type": "http.disconnect"}
        await asyncio.Event().wait()

    async def send(message):
        if message["type"] == "http.response.start":
            status.append(message["status"])
            started.set()
        elif message["type"] == "http.response.body":
            chunk = message["body"]
            output.append(chunk)
            if chunk == COMMENT:
                state.comments += 1
            elif chunk:
                first_data.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST" if payload else "GET",
        "path": path,
        "raw_path": path.encode(),
        "root_path": "",
        "query_string": b"stream=true" if not payload else b"",
        "headers": [(b"content-type", b"application/json")],
        "scheme": "http",
        "server": ("test", 80),
        "client": ("test", 1234),
    }
    failure = None
    try:
        await asyncio.wait_for(app(scope, receive, send), 2)
    except RuntimeError as exc:
        failure = exc
    assert status == [200]
    assert state.closed
    # Let cancelled route listeners finish before checking for leaked tasks.
    await asyncio.sleep(0)
    pending = [
        t
        for t in asyncio.all_tasks()
        if t is not asyncio.current_task() and not t.done()
    ]
    assert pending == []
    return b"".join(output), state, failure


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("interval", [None, 0, 30, 0.01])
async def test_sse_bytes_unchanged_without_idle(route, interval):
    expected, _, _ = await _run(route, None)
    body, state, error = await _run(route, interval)
    assert body == expected
    assert not state.comments
    assert error is None


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("scenario", ["idle", "error_frame", "exception"])
async def test_idle_adds_only_comments_and_preserves_errors(route, scenario):
    expected, _, expected_error = await _run(route, None, scenario)
    body, state, error = await _run(route, 0.01, scenario)
    assert state.comments > 0
    assert body.replace(COMMENT, b"") == expected
    assert type(error) is type(expected_error)
    if error:
        assert str(error) == str(expected_error)


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("scenario", ["disconnect_idle", "disconnect_between"])
@pytest.mark.parametrize("interval", [0, 0.01])
async def test_disconnect_closes_and_aborts_foreground_stream(
    route, scenario, interval
):
    _, state, _ = await _run(route, interval, scenario)
    assert state.aborted


@pytest.mark.parametrize("route", ["responses", "retrieve"])
async def test_harmony_is_not_wrapped_and_cancel_scope_stays_in_one_task(route):
    expected, _, _ = await _run(route, 0, "idle", harmony=True)
    body, state, error = await _run(route, 0.001, "idle", harmony=True)
    assert body == expected
    assert not state.comments
    assert error is None
