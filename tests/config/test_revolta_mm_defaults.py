# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from vllm import envs
from vllm.config.multimodal import MultiModalConfig
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.models.qwen4_exp.config import Qwen4ExpConfig
from vllm.models.qwen4_exp.nvidia.model import Qwen4ExpProcessingInfo
from vllm.multimodal.media import ImageMediaIO, VideoMediaIO
from vllm.multimodal.processing import InputProcessingContext


@pytest.fixture
def info_factory(tmp_path):
    tokens = [
        "[UNK]",
        "<|image_pad|>",
        "<|video_pad|>",
        "<|vision_start|>",
        "<|vision_end|>",
    ]
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel(dict(zip(tokens, range(5))), "[UNK]")),
        additional_special_tokens=tokens[1:],
    )
    tokenizer.save_pretrained(tmp_path)
    image_config = dict(
        size={"longest_edge": 16777216, "shortest_edge": 65536},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        image_mean=[0.5] * 3,
        image_std=[0.5] * 3,
        processor_class="Qwen3VLProcessor",
        image_processor_type="Qwen2VLImageProcessorFast",
    )
    video_config = image_config | dict(
        size={"longest_edge": 25165824, "shortest_edge": 4096},
        video_processor_type="Qwen3VLVideoProcessor",
    )
    video_config.pop("image_processor_type")
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(image_config))
    (tmp_path / "video_preprocessor_config.json").write_text(json.dumps(video_config))
    config = Qwen4ExpConfig(
        vision_config=dict(patch_size=16, spatial_merge_size=2, temporal_patch_size=2),
        image_token_id=1,
        video_token_id=2,
        vision_start_token_id=3,
        vision_end_token_id=4,
    )

    def make_info(kwargs=None):
        mm_config = MultiModalConfig(mm_processor_kwargs=kwargs)
        model = SimpleNamespace(
            model=str(tmp_path),
            revision=None,
            trust_remote_code=False,
            hf_config=config,
            dtype=torch.float16,
            get_multimodal_config=lambda: mm_config,
        )
        return Qwen4ExpProcessingInfo(InputProcessingContext(model, tokenizer))

    return make_info


def test_modality_defaults():
    config = MultiModalConfig()
    assert [config.get_limit_per_prompt(m) for m in ("image", "video", "audio")] == [
        8,
        1,
        1,
    ]
    assert config.get_limit_per_prompt("other") == 999
    explicit = MultiModalConfig(limit_per_prompt={"image": 16, "audio": 16})
    assert [explicit.get_limit_per_prompt(m) for m in ("image", "video", "audio")] == [
        16,
        1,
        16,
    ]
    assert MultiModalConfig(language_model_only=True).get_limit_per_prompt("image") == 0


@pytest.mark.parametrize("kwargs", [None, {}, {"do_rescale": True}])
def test_unset_image_size(info_factory, kwargs):
    info = info_factory(kwargs)
    assert info.get_image_processor().size == {
        "longest_edge": 1048576,
        "shortest_edge": 65536,
    }


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"size": {"longest_edge": 4194304, "shortest_edge": 65536}}, 4194304),
        ({"max_pixels": 4194304}, 4194304),
        ({"min_pixels": 32768}, 16777216),
        ({"size": {"longest_edge": 16777216, "shortest_edge": 65536}}, 16777216),
        ({"do_resize": False}, 16777216),
        (
            {
                "images_kwargs": {
                    "size": {"longest_edge": 4194304, "shortest_edge": 65536}
                }
            },
            4194304,
        ),
    ],
)
def test_operator_size_wins(info_factory, kwargs, expected):
    info = info_factory(kwargs)
    assert info.get_image_processor().size["longest_edge"] == expected
    image = Image.new("RGB", (2048, 2048))
    result = info.ctx.call_hf_processor(
        info.get_hf_processor(),
        dict(text="<|image_pad|>", images=[image]),
    )
    assert result.image_grid_thw.tolist() == [[1, 128, 128]]


@pytest.mark.parametrize(
    "width,height,grid,tokens",
    [
        (2048, 2048, [1, 64, 64], 1024),
        (4032, 3024, [1, 54, 72], 972),
        (1920, 1080, [1, 48, 84], 1008),
        (1024, 1024, [1, 64, 64], 1024),
        (800, 600, [1, 38, 50], 475),
    ],
)
def test_real_image_grid(info_factory, width, height, grid, tokens):
    info = info_factory({})
    processor = info.get_hf_processor()
    # Nonuniform pixels also exercise the bitwise comparison below.
    image = Image.fromarray(
        np.random.default_rng(42).integers(0, 256, (height, width, 3), dtype=np.uint8)
    )
    result = processor(text="<|image_pad|>", images=[image], return_tensors="pt")
    assert result.image_grid_thw.tolist() == [grid]
    assert result.pixel_values.shape == (tokens * 4, 1536)
    assert grid[1] * grid[2] * 16**2 <= 1048576
    assert (
        info.get_num_image_tokens(
            image_width=width,
            image_height=height,
            image_processor=processor.image_processor,
            mm_kwargs={},
        )
        == tokens
    )
    if width * height <= 1048576:
        old = info_factory({"max_pixels": 16777216}).get_hf_processor()
        old_result = old(text="<|image_pad|>", images=[image], return_tensors="pt")
        assert torch.equal(result.pixel_values, old_result.pixel_values)
        assert torch.equal(result.input_ids, old_result.input_ids)


def test_dummy_profile(info_factory):
    info = info_factory({})
    size = info.get_image_size_with_most_features()
    assert tuple(size) == (1024, 1024)
    assert info.get_max_image_tokens() == 1024
    dummy = Qwen3VLDummyInputsBuilder(info).get_dummy_mm_data(
        seq_len=524288,
        mm_counts={"image": 1},
        mm_options={},
    )
    assert dummy["image"][0].size == (1024, 1024)
    assert info_factory({"max_pixels": 16777216}).get_max_image_tokens() == 16384


def test_video_unchanged_and_cached_processor_isolation(info_factory):
    info = info_factory({})
    p = info.get_hf_processor()
    raw = info.ctx.get_hf_processor(type(p), use_fast=True)
    assert raw.image_processor.size["longest_edge"] == 16777216
    assert p.video_processor is raw.video_processor
    assert p.video_processor.size == {"longest_edge": 25165824, "shortest_edge": 4096}
    video = np.zeros((2, 1080, 1920, 3), dtype=np.uint8)
    expected = raw.video_processor(
        videos=[video], do_sample_frames=False, return_tensors="pt"
    )
    actual = p.video_processor(
        videos=[video], do_sample_frames=False, return_tensors="pt"
    )
    assert torch.equal(actual.pixel_values_videos, expected.pixel_values_videos)
    assert torch.equal(actual.video_grid_thw, expected.video_grid_thw)
    assert info.get_max_video_tokens(524288, {"video": 1}) == 12288


def test_request_override_is_still_possible_in_this_lane(info_factory):
    info = info_factory({})
    assert (
        info.get_hf_processor(max_pixels=4194304).image_processor.size["longest_edge"]
        == 4194304
    )
    result = info.ctx.call_hf_processor(
        info.get_hf_processor(max_pixels=4194304),
        dict(text="<|image_pad|>", images=[Image.new("RGB", (2048, 2048))]),
        dict(max_pixels=4194304),
    )
    assert result.image_grid_thw.tolist() == [[1, 128, 128]]
    assert info.get_image_processor().size["longest_edge"] == 1048576


def test_video_default_and_decode_limits():
    assert VideoMediaIO(ImageMediaIO()).num_frames == 16
    assert VideoMediaIO(ImageMediaIO(), num_frames=32).num_frames == 32
    assert envs.VLLM_MAX_IMAGE_PIXELS == 16777216
    assert envs.VLLM_MAX_VIDEO_FRAMES == 64


@pytest.mark.parametrize(
    "kwargs,expected,source",
    [
        ({}, 1048576, "default"),
        ({"max_pixels": 4194304}, 4194304, "operator"),
    ],
)
def test_image_budget_log(info_factory, monkeypatch, kwargs, expected, source):
    from vllm.models.qwen4_exp.nvidia import model

    logs = []
    monkeypatch.setattr(
        model.logger, "info_once", lambda fmt, *args: logs.append(fmt % args)
    )
    info_factory(kwargs).get_hf_processor()
    assert logs == [
        f"Qwen4Exp image max_pixels={expected} source={source} do_resize=True"
    ]


def test_text_tokenization_unchanged(info_factory):
    info = info_factory({})
    processor = info.get_hf_processor()
    raw = info.ctx.get_hf_processor(type(processor), use_fast=True)
    new = processor(text="example text", return_tensors="pt")
    old = raw(text="example text", return_tensors="pt")
    assert torch.equal(new.input_ids, old.input_ids)


def test_small_image_alignment_boundary(info_factory):
    # HF alignment can cross the new area budget even for a smaller source.
    image = Image.new("RGB", (977, 1073))
    new = info_factory({}).get_hf_processor()(
        text="<|image_pad|>", images=[image], return_tensors="pt"
    )
    old = info_factory({"max_pixels": 16777216}).get_hf_processor()(
        text="<|image_pad|>", images=[image], return_tensors="pt"
    )
    assert image.width * image.height <= 1048576
    assert old.image_grid_thw.tolist() == [[1, 68, 62]]
    assert new.image_grid_thw.tolist() == [[1, 66, 60]]
