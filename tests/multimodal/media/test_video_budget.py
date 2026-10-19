# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pybase64
import pytest

from vllm import envs
from vllm.connections import HTTPConnection
from vllm.multimodal.media import ImageMediaIO, MediaConnector, VideoMediaIO
from vllm.multimodal.video import VideoBackend, VideoSourceMetadata

pytestmark = pytest.mark.cpu_test


def test_video_budget_precedes_frame_allocation(monkeypatch):
    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 6399, raising=False
    )
    cap = MagicMock()
    cap.get.return_value = 10
    monkeypatch.setattr(VideoBackend, "open_video_capture", lambda _: cap)
    monkeypatch.setattr(
        VideoBackend, "get_video_metadata", lambda _: VideoSourceMetadata(16, 1, 16)
    )
    read = MagicMock(
        return_value=(np.zeros((16, 10, 10, 3), dtype=np.uint8), list(range(16)))
    )
    monkeypatch.setattr(VideoBackend, "read_frames", read)
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        VideoMediaIO(ImageMediaIO(), num_frames=16).load_bytes(b"encoded")
    read.assert_not_called()


def test_default_budget_covers_sixteen_4k_frames():
    from vllm.multimodal.media.budget import MediaBudget

    MediaBudget().charge_video(16, 3840, 2160)


@pytest.fixture
def encoded_video(tmp_path):
    import cv2

    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 4.0, (32, 32))
    assert writer.isOpened()
    for i in range(16):
        writer.write(np.full((32, 32, 3), i * 12, dtype=np.uint8))
    writer.release()
    return path


@pytest.fixture
def video_http(encoded_video):
    data = encoded_video.read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/video.avi"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


@pytest.mark.parametrize("source", ["file", "data", "http"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_sixteen_frames_match_unbudgeted_reference(
    encoded_video, source, asynchronous, video_http
):
    import asyncio

    data = encoded_video.read_bytes()
    expected, expected_meta = VideoBackend.load_bytes(data, num_frames=16)
    connection = HTTPConnection()
    connector = MediaConnector(
        connection=connection,
        allowed_local_media_path=str(encoded_video.parent),
        media_io_kwargs={"video": {"num_frames": 16}},
    )
    url = (
        encoded_video.as_uri()
        if source == "file"
        else ("data:video/avi;base64," + pybase64.b64encode(data).decode())
    )
    if source == "http":
        url = video_http

    async def fetch():
        try:
            return await connector.fetch_video_async(url)
        finally:
            if connection._async_client is not None:
                await connection._async_client.close()

    try:
        frames, meta = (
            asyncio.run(fetch()) if asynchronous else connector.fetch_video(url)
        )
    finally:
        connection.get_sync_client().close()
    assert frames.shape[0] == 16
    assert frames.tobytes() == expected.tobytes()
    assert meta == expected_meta


def test_image_and_video_share_decoded_budget(monkeypatch, encoded_video):
    from PIL import Image

    monkeypatch.setattr(
        envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 65536, raising=False
    )
    connector = MediaConnector(
        allowed_local_media_path=str(encoded_video.parent),
        media_io_kwargs={"video": {"num_frames": 16}},
    )
    connector.fetch_video(encoded_video.as_uri())
    out = BytesIO()
    Image.new("RGB", (1, 1)).save(out, "PNG")
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        connector.fetch_image(
            "data:image/png;base64," + pybase64.b64encode(out.getvalue()).decode()
        )


def test_request_cannot_switch_to_unbounded_decoder():
    from vllm.multimodal.video import VIDEO_LOADER_REGISTRY, VideoLoader

    called = []

    @VIDEO_LOADER_REGISTRY.register("unbounded_test_backend")
    class Unbounded(VideoLoader):
        @classmethod
        def load_bytes(cls, data, **kwargs):
            called.append(True)
            raise AssertionError("must reject before entering unbounded decoder")

    with pytest.raises(ValueError, match="pre-decode budget"):
        VideoMediaIO(ImageMediaIO(), video_backend="unbounded_test_backend").load_bytes(
            b"input"
        )
    assert not called
