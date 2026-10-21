# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest

from vllm.multimodal.video import (
    OpenCVVideoBackendMixin,
    PyAVVideoBackendMixin,
    VideoLoader,
)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("backend", ["opencv", "pyav"])
def test_unknown_fps_uses_consistent_fallback(fps, backend):
    from vllm.multimodal import video

    if backend == "opencv":
        values = {video.cv2.CAP_PROP_FRAME_COUNT: 48, video.cv2.CAP_PROP_FPS: fps}
        source = OpenCVVideoBackendMixin.get_video_metadata(
            SimpleNamespace(get=values.__getitem__)
        )
    else:
        stream = SimpleNamespace(frames=48, average_rate=fps, duration=0)
        source = PyAVVideoBackendMixin.get_metadata(
            SimpleNamespace(streams=SimpleNamespace(video=[stream]))
        )
    assert source.original_fps == 24.0
    assert source.duration == 2.0
    metadata = VideoLoader.create_hf_metadata(source, [0, 24], backend)
    assert [idx / metadata["fps"] for idx in metadata["frames_indices"]] == [0, 1]
    assert metadata["duration"] == 2.0


@pytest.mark.parametrize("fps", [23.976, 24.0, 30.0, 60.0])
def test_valid_fps_is_preserved(fps):
    stream = SimpleNamespace(frames=48, average_rate=fps, duration=2, time_base=1)
    source = PyAVVideoBackendMixin.get_metadata(
        SimpleNamespace(streams=SimpleNamespace(video=[stream]))
    )
    assert source.original_fps == fps
    assert source.duration == 2
