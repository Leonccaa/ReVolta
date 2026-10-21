# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check bytes and disconnect cleanup over a real loopback Uvicorn connection."""

import asyncio
import socket

import pytest
import uvicorn
from starlette.responses import StreamingResponse

from vllm.entrypoints.openai.sse_keep_alive import with_sse_keep_alive

pytestmark = [pytest.mark.asyncio, pytest.mark.skip_global_cleanup]


@pytest.mark.parametrize("interval", [0, 0.01])
async def test_uvicorn_delivers_comments_and_disconnect_closes_stream(interval):
    finalized = asyncio.Event()

    async def generate():
        try:
            yield "data: ready\n\n"
            await asyncio.Event().wait()
        finally:
            finalized.set()

    async def app(scope, receive, send):
        await StreamingResponse(
            with_sse_keep_alive(generate(), interval),
            media_type="text/event-stream",
        )(scope, receive, send)

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    server = uvicorn.Server(
        uvicorn.Config(app, lifespan="off", access_log=False, log_level="error")
    )
    serving = asyncio.create_task(server.serve(sockets=[listener]))
    writer = None

    async def communicate():
        nonlocal writer
        while not server.started:
            await asyncio.sleep(0.001)
        reader, writer = await asyncio.open_connection(
            "127.0.0.1", listener.getsockname()[1]
        )
        writer.write(b"GET / HTTP/1.1\r\nHost: localhost\r\n\r\n")
        await writer.drain()
        headers = await reader.readuntil(b"\r\n\r\n")
        assert headers.startswith(b"HTTP/1.1 200")

        async def chunk():
            size = int((await reader.readline()).strip(), 16)
            body = await reader.readexactly(size)
            assert await reader.readexactly(2) == b"\r\n"
            return body

        assert await chunk() == b"data: ready\n\n"
        if interval:
            assert await chunk() == b": keep-alive\n\n"
        writer.close()
        await writer.wait_closed()
        await finalized.wait()

    try:
        await asyncio.wait_for(communicate(), 3)
    finally:
        if writer is not None:
            writer.close()
        server.should_exit = True
        await asyncio.wait_for(serving, 3)
        listener.close()
