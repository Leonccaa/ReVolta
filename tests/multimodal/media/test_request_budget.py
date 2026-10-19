# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from unittest.mock import MagicMock

import numpy as np
import pybase64 as base64
import pytest
from PIL import Image

from vllm import envs
from vllm.config import MultiModalConfig
from vllm.multimodal.media import ImageMediaIO, MediaConnector, VideoMediaIO

pytestmark = pytest.mark.cpu_test


def png(width=10, height=10):
    out = BytesIO()
    Image.new("RGB", (width, height), (20, 40, 60)).save(out, "PNG")
    return out.getvalue()


def data_url(data):
    return "data:image/png;base64," + base64.b64encode(data).decode()


def test_safe_modality_defaults_and_explicit_override():
    config = MultiModalConfig(limit_per_prompt={"video": 1})
    assert config.get_limit_per_prompt("image") == 16
    assert config.get_limit_per_prompt("video") == 1
    assert (
        MultiModalConfig(
            limit_per_prompt={"image": {"width": 512}}
        ).get_limit_per_prompt("image")
        == 16
    )
    assert (
        MultiModalConfig(
            limit_per_prompt={"video": {"num_frames": 16}}
        ).get_limit_per_prompt("video")
        == 1
    )
    assert (
        MultiModalConfig(limit_per_prompt={"image": 24}).get_limit_per_prompt("image")
        == 24
    )
    assert MultiModalConfig(language_model_only=True).get_limit_per_prompt("image") == 0


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("source", ["file", "data", "http", "cache"])
def test_aggregate_encoded_bytes(monkeypatch, tmp_path, source, asynchronous):
    data = png()
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_BYTES", len(data), raising=False)
    path = tmp_path / "image.png"
    path.write_bytes(data)
    connection = MagicMock()

    def fetch(*args, **kwargs):
        charge = kwargs.get("on_chunk")
        if charge:
            charge(len(data))
        return data

    connection.get_bytes.side_effect = fetch

    async def async_fetch(*args, **kwargs):
        return fetch(*args, **kwargs)

    connection.async_get_bytes.side_effect = async_fetch
    connector = MediaConnector(
        connection=connection, allowed_local_media_path=str(tmp_path)
    )
    if source == "file":
        url = path.as_uri()
    elif source == "data":
        url = data_url(data)
    else:
        url = "http://127.0.0.1/image"
        if source == "cache":
            connector._media_cache_dir = str(tmp_path)
            connector._media_cache_ttl_secs = 3600
            connector._media_cache_path(url).write_bytes(data)

    async def run():
        await connector.fetch_image_async(url)
        with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_BYTES"):
            await connector.fetch_image_async(url)

    if asynchronous:
        asyncio.run(run())
    else:
        connector.fetch_image(url)
        with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_BYTES"):
            connector.fetch_image(url)


def test_image_budget_rejects_before_raster_allocation(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 399, raising=False)
    # PNG opening parses headers without allocating the raster.
    original = Image.Image.load
    loaded = []

    def load(image, *args, **kwargs):
        loaded.append(image.size)
        return original(image, *args, **kwargs)

    data = png()
    monkeypatch.setattr(Image.Image, "load", load)
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        MediaConnector().fetch_image(data_url(data))
    assert not loaded


def test_request_decoded_bytes_are_shared(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 400, raising=False)
    connector = MediaConnector()
    connector.fetch_image(data_url(png()))
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        connector.fetch_image(data_url(png()))
    # A failed request cannot poison the budget of the next request.
    MediaConnector().fetch_image(data_url(png()))


def test_parallel_decodes_share_atomic_budget(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 400, raising=False)
    connector = MediaConnector()

    def load(_):
        try:
            connector.fetch_image(data_url(png()))
            return "ok"
        except ValueError:
            return "rejected"

    with ThreadPoolExecutor(max_workers=4) as pool:
        assert list(pool.map(load, range(8))).count("ok") == 1


def test_item_budget_is_before_fetch(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_ITEMS", 1, raising=False)
    connector = MediaConnector()
    connector.fetch_image(data_url(png()))
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_ITEMS"):
        connector.fetch_image("http://127.0.0.1:1/must-not-connect")


def test_pixel_limit_precedes_load(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 99, raising=False)
    with pytest.raises(ValueError, match="VLLM_MAX_IMAGE_PIXELS"):
        ImageMediaIO().load_bytes(png())


def test_request_cannot_raise_server_pixel_limit(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 99, raising=False)
    connector = MediaConnector(media_io_kwargs={"image": {"max_image_pixels": 1000}})
    with pytest.raises(ValueError, match="VLLM_MAX_IMAGE_PIXELS"):
        connector.fetch_image(data_url(png()))


def test_request_cannot_raise_video_frame_limit(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_VIDEO_FRAMES", 16, raising=False)
    with pytest.raises(ValueError, match="VLLM_MAX_VIDEO_FRAMES"):
        VideoMediaIO(ImageMediaIO(), num_frames=17)


def test_jpeg_sequence_counts_decoded_frames(monkeypatch):
    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 400, raising=False)
    frame = base64.b64encode(png()).decode()
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        MediaConnector().fetch_video("data:video/jpeg;base64," + frame + "," + frame)


def test_zero_budget_explicitly_disables_limit(monkeypatch):
    for name in (
        "VLLM_MAX_MEDIA_REQUEST_BYTES",
        "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES",
        "VLLM_MAX_MEDIA_REQUEST_ITEMS",
        "VLLM_MAX_IMAGE_PIXELS",
    ):
        monkeypatch.setattr(envs, name, 0, raising=False)
    connector = MediaConnector()
    for _ in range(33):
        connector.fetch_image(data_url(png()))


def test_all_message_parsers_share_request_budget(monkeypatch):
    from vllm.entrypoints.chat_utils import MultiModalItemTracker

    monkeypatch.setattr(envs, "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES", 400, raising=False)
    config = MagicMock()
    config.multimodal_config = MultiModalConfig()
    config.allowed_local_media_path = ""
    config.allowed_media_domains = []
    tracker = MultiModalItemTracker(config)
    # Avoid model loading: exercise the actual parser and its shared connector.
    tracker.add = MagicMock(return_value="<image>")
    tracker.create_parser().parse_image(data_url(png()))
    with pytest.raises(ValueError, match="VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"):
        tracker.create_parser().parse_image(data_url(png()))


def test_budgeted_image_tensor_identical():
    data = png(64, 32)
    expected = np.asarray(Image.open(BytesIO(data)).convert("RGB"))
    actual = MediaConnector().fetch_image(data_url(data))
    np.testing.assert_array_equal(np.asarray(actual), expected)
