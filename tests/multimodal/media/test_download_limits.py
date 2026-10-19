# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import gzip
import ssl
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import MagicMock

import pytest

from vllm import envs
from vllm.connections import HTTPConnection
from vllm.multimodal.media import MediaConnector

pytestmark = pytest.mark.cpu_test
LIMIT = 1024 * 1024


@pytest.fixture(params=[False, True], ids=["http", "https"])
def media_server(request, tmp_path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/redirect":
                self.send_response(302)
                self.send_header("Location", "/large")
                self.end_headers()
                return
            body = b"x" * (LIMIT + 1) if self.path != "/small" else b"ok"
            self.send_response(200)
            if self.path == "/gzip":
                body = gzip.compress(body)
                self.send_header("Content-Encoding", "gzip")
            if self.path in ("/large", "/small"):
                self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                for i in range(0, len(body), 4096):
                    self.wfile.write(body[i : i + 4096])
            except (BrokenPipeError, ConnectionResetError, ssl.SSLError):
                pass

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    scheme = "http"
    if request.param:
        key, cert = tmp_path / "key.pem", tmp_path / "cert.pem"
        subprocess.run(
            [
                "openssl",
                "req",
                "-x509",
                "-newkey",
                "rsa:2048",
                "-nodes",
                "-keyout",
                str(key),
                "-out",
                str(cert),
                "-days",
                "1",
                "-subj",
                "/CN=localhost",
                "-addext",
                "subjectAltName=IP:127.0.0.1",
            ],
            check=True,
            capture_output=True,
        )
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert, key)
        server.socket = context.wrap_socket(server.socket, server_side=True)
        scheme = "https"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f"{scheme}://127.0.0.1:{server.server_port}",
            (ssl.create_default_context(cafile=str(cert)) if request.param else None),
            str(cert) if request.param else True,
        )
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("path", ["large", "unknown", "gzip", "redirect", "small"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_download_limits_local_server(monkeypatch, media_server, path, asynchronous):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1, raising=False)
    url, ssl_context, verify = media_server
    connection = HTTPConnection()
    connection.get_sync_client().verify = verify

    async def fetch():
        import aiohttp

        connection._async_client = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=ssl_context)
        )
        try:
            return await connection.async_get_bytes(f"{url}/{path}")
        finally:
            await connection._async_client.close()

    try:
        if path == "small":
            assert (
                asyncio.run(fetch())
                if asynchronous
                else connection.get_bytes(f"{url}/{path}")
            ) == b"ok"
        else:
            with pytest.raises(ValueError, match="maximum|exceed|VLLM_MAX_MEDIA"):
                if asynchronous:
                    asyncio.run(fetch())
                else:
                    connection.get_bytes(f"{url}/{path}")
    finally:
        connection.get_sync_client().close()


def test_bounded_reader_never_materializes_body():
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {}
    response.raw.stream.return_value = iter([b"abcd", b"efgh", b"unread"])
    connection = HTTPConnection()
    connection.get_response = MagicMock(return_value=response)
    with pytest.raises(ValueError, match="maximum|exceed"):
        connection.get_bytes("http://127.0.0.1/media", max_bytes=4)
    assert next(response.raw.stream.return_value) == b"unread"
    assert connection.get_response.call_args.kwargs["stream"] is True


def test_connector_passes_download_limit(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB", 1, raising=False)
    connection = MagicMock(spec=HTTPConnection)
    connection.get_bytes.side_effect = ValueError("maximum response size exceeded")
    with pytest.raises(ValueError):
        MediaConnector(connection=connection).fetch_image("http://127.0.0.1/media")
    assert connection.get_bytes.call_args.kwargs["max_bytes"] == LIMIT


def test_size_error_maps_to_bad_request():
    from vllm.connections import MediaDownloadSizeExceededError
    from vllm.entrypoints.utils import create_error_response

    response = create_error_response(MediaDownloadSizeExceededError(1024, 1025))
    assert response.error.code == 400
    assert "VLLM_MAX_MEDIA_DOWNLOAD_SIZE_MB" in response.error.message


def test_size_failure_is_not_retried(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MEDIA_FETCH_MAX_RETRIES", 3, raising=False)
    connection = HTTPConnection()
    response = MagicMock()
    response.__enter__.return_value = response
    response.headers = {"Content-Length": "5"}
    connection.get_response = MagicMock(return_value=response)
    with pytest.raises(ValueError, match="maximum|exceed"):
        connection.get_bytes("http://127.0.0.1/media", max_bytes=4)
    connection.get_response.assert_called_once()
    response.__exit__.assert_called_once()


def test_cancelled_stream_closes_response():
    async def run():
        started = asyncio.Event()
        closed = []

        class Stream:
            async def iter_chunked(self, _):
                yield b"a"
                started.set()
                await asyncio.Event().wait()

        class Response:
            content_length = None
            headers = {}
            content = Stream()

            def raise_for_status(self):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                closed.append(True)

        connection = HTTPConnection()

        async def response(*args, **kwargs):
            return Response()

        connection.get_async_response = response
        task = asyncio.create_task(
            connection.async_get_bytes("http://127.0.0.1/media", max_bytes=4)
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert closed == [True]

    asyncio.run(run())
