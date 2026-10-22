# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pybase64
from PIL import Image

from vllm import envs

from ..video import VIDEO_LOADER_REGISTRY, VideoBackend
from .base import MediaIO
from .image import ImageMediaIO


class VideoMediaIO(MediaIO[tuple[npt.NDArray, dict[str, Any]]]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        # fps and num_frames interact with each other, so if either is
        # overridden at request time, wipe the other from defaults to
        # avoid unintuitive cross-field interactions.
        if runtime_kwargs:
            if "num_frames" in runtime_kwargs and "fps" not in runtime_kwargs:
                merged.pop("fps", None)
            elif "fps" in runtime_kwargs and "num_frames" not in runtime_kwargs:
                merged.pop("num_frames", None)
        return merged

    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 16,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        limit = envs.VLLM_MAX_VIDEO_FRAMES
        if limit and num_frames > limit:
            raise ValueError(
                f"Video exceeds VLLM_MAX_VIDEO_FRAMES={limit}: {num_frames}"
            )
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.

        # Allow per-request override of video backend via kwargs.
        # This enables users to specify a different backend than the
        # global VLLM_VIDEO_LOADER_BACKEND env var, e.g.:
        #   --media-io-kwargs '{"video": {"video_backend": "torchcodec"}}'
        video_loader_backend = (
            kwargs.pop("video_backend", None) or envs.VLLM_VIDEO_LOADER_BACKEND
        )
        self.kwargs = kwargs
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(self, data: bytes) -> tuple[npt.NDArray, dict[str, Any]]:
        kwargs = dict(self.kwargs)
        precharged = 0

        def charge_frames(count: int, width: int, height: int) -> None:
            nonlocal precharged
            self.image_io.budget.charge_video(count, width, height)
            precharged = count * width * height * 4

        if isinstance(self.video_loader, VideoBackend):
            # Set internally after request kwargs, so clients cannot replace it.
            kwargs["frame_budget"] = charge_frames
        elif envs.VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES:
            raise ValueError(
                "Video backend cannot enforce a pre-decode budget; choose opencv "
                "or explicitly disable VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES"
            )
        result = self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **kwargs
        )
        frames = result[0] if isinstance(result, tuple) else result
        if frames.ndim == 4:
            count, height, width, _ = frames.shape
            if not precharged:
                charge_frames(count, width, height)
            if frames.nbytes > precharged:
                self.image_io.budget.charge("decoded", frames.nbytes - precharged)
        return result

    def load_base64(
        self, media_type: str, data: str
    ) -> tuple[npt.NDArray, dict[str, Any]]:
        if media_type.lower() == "video/jpeg":
            load_frame = partial(
                self.image_io.load_base64,
                "image/jpeg",
            )

            if self.num_frames > 0:
                frame_parts = data.split(",", self.num_frames)[: self.num_frames]
            elif self.num_frames == 0:
                raise ValueError("num_frames must be greater than 0 or -1")
            else:
                limit = envs.VLLM_MAX_VIDEO_FRAMES
                if limit and data.count(",") + 1 > limit:
                    raise ValueError(f"Video exceeds VLLM_MAX_VIDEO_FRAMES={limit}")
                frame_parts = data.split(",")

            limit = envs.VLLM_MAX_VIDEO_FRAMES
            if limit and len(frame_parts) > limit:
                raise ValueError(f"Video exceeds VLLM_MAX_VIDEO_FRAMES={limit}")
            frames = np.stack(
                [np.asarray(load_frame(frame_data)) for frame_data in frame_parts]
            )
            total = int(frames.shape[0])
            fps = float(self.kwargs.get("fps", 1))

            # validate and extract frames_indices
            frames_indices = self.kwargs.get("frames_indices")
            if frames_indices is not None:
                if not (
                    isinstance(frames_indices, list)
                    and all(isinstance(i, int) for i in frames_indices)
                ):
                    raise ValueError("frames_indices must be a list of integers")
                if len(frames_indices) != total:
                    raise ValueError(
                        f"frames_indices length ({len(frames_indices)}) must "
                        f"match number of frames sent ({total})"
                    )
            else:
                frames_indices = list(range(total))

            # validate and extract total_num_frames
            total_num_frames = self.kwargs.get("total_num_frames", total)
            if not isinstance(total_num_frames, int) or total_num_frames < 1:
                raise ValueError("total_num_frames must be a positive integer")
            if total_num_frames < total:
                raise ValueError(
                    f"total_num_frames ({total_num_frames}) must be >= "
                    f"number of frames sent ({total})"
                )

            # validate and extract duration
            duration = self.kwargs.get("duration")
            if duration is not None:
                if not isinstance(duration, (int, float)) or duration < 0:
                    raise ValueError("duration must be a non-negative number")
            else:
                duration = total_num_frames / fps if fps > 0 else 0.0

            metadata = {
                "total_num_frames": total_num_frames,
                "fps": fps,
                "duration": duration,
                "video_backend": "jpeg_sequence",
                "frames_indices": frames_indices,
                "do_sample_frames": self.kwargs.get("do_sample_frames", False),
            }
            return frames, metadata

        return self.load_bytes(pybase64.b64decode(data))

    def load_file(self, filepath: Path) -> tuple[npt.NDArray, dict[str, Any]]:
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
