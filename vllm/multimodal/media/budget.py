# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Request-scoped media accounting, shared by all message parsers and IO threads."""

from threading import Lock

from PIL import Image

from vllm import envs


class MediaBudget:
    def __init__(self) -> None:
        self._lock = Lock()
        self._used = {"items": 0, "encoded": 0, "decoded": 0}
        self._limits = {
            "items": (
                "VLLM_MAX_MEDIA_REQUEST_ITEMS",
                envs.VLLM_MAX_MEDIA_REQUEST_ITEMS,
            ),
            "encoded": (
                "VLLM_MAX_MEDIA_REQUEST_BYTES",
                envs.VLLM_MAX_MEDIA_REQUEST_BYTES,
            ),
            "decoded": (
                "VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES",
                envs.VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES,
            ),
        }
        for name, limit in self._limits.values():
            if limit < 0:
                raise ValueError(f"{name} must be non-negative")

    def charge(self, kind: str, amount: int) -> None:
        if amount < 0:
            raise ValueError("Invalid negative media size")
        name, limit = self._limits[kind]
        with self._lock:
            total = self._used[kind] + amount
            if limit and total > limit:
                raise ValueError(
                    f"Media request exceeds {name}={limit} (at least {total})"
                )
            self._used[kind] = total

    def charge_encoded(self, amount: int) -> None:
        self.charge("encoded", amount)

    def charge_image(self, image: Image.Image) -> None:
        pixels = image.width * image.height
        limit = envs.VLLM_MAX_IMAGE_PIXELS
        if limit and pixels > limit:
            raise ValueError(f"Image exceeds VLLM_MAX_IMAGE_PIXELS={limit}: {pixels}")
        # Four bytes per pixel covers RGB storage and alpha-bearing source modes.
        # This is retained-raster accounting, not a process RSS ceiling.
        self.charge("decoded", pixels * 4)

    def charge_video(self, count: int, width: int, height: int) -> None:
        limit = envs.VLLM_MAX_VIDEO_FRAMES
        if limit and count > limit:
            raise ValueError(f"Video exceeds VLLM_MAX_VIDEO_FRAMES={limit}: {count}")
        pixels = width * height
        if width <= 0 or height <= 0:
            raise ValueError("Video must report positive frame dimensions")
        if envs.VLLM_MAX_IMAGE_PIXELS and pixels > envs.VLLM_MAX_IMAGE_PIXELS:
            raise ValueError(f"Video frame exceeds VLLM_MAX_IMAGE_PIXELS: {pixels}")
        self.charge("decoded", count * pixels * 4)
