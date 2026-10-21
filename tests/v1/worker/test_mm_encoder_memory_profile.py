# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for MRv2's replicated encoder memory profile.

Only patch metadata and output shapes are simulated, not CUDA memory usage.
The budget, batching, item assignment and profile_run are the real code.
"""

import weakref
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from transformers import BatchFeature

from vllm.config import MultiModalConfig
from vllm.model_executor.models import vision as vision_module
from vllm.model_executor.models.qwen3_vl import (
    Qwen3VLDummyInputsBuilder,
    Qwen3VLProcessingInfo,
    Qwen3VLVideoProcessor,
)
from vllm.model_executor.models.vision import get_load_balance_assignment
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import MultiModalFieldConfig, MultiModalKwargsItems
from vllm.v1.worker.gpu import model_runner
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner

pytestmark = pytest.mark.skip_global_cleanup

IMAGE_TOKENS = 16384
HIDDEN_SIZE = 2560
TP = 4
CACHE_BYTES = IMAGE_TOKENS * HIDDEN_SIZE * 2 * TP


def test_qwen_maximum_dummy_image_and_video_shapes(monkeypatch):
    # The pinned checkpoint uses p16, merge 2 and a 4096**2 pixel limit.
    # Run its real resize/token arithmetic without downloading a checkpoint.
    ctx = SimpleNamespace(get_merged_mm_kwargs=lambda kwargs: kwargs)
    info = Qwen3VLProcessingInfo(ctx)
    vision = SimpleNamespace(patch_size=16, spatial_merge_size=2, temporal_patch_size=2)
    processor = SimpleNamespace(size={"shortest_edge": 65536, "longest_edge": 16777216})
    monkeypatch.setattr(
        info, "get_hf_config", lambda: SimpleNamespace(vision_config=vision)
    )
    monkeypatch.setattr(info, "get_image_processor", lambda: processor)
    video_processor = Qwen3VLVideoProcessor(
        size={"shortest_edge": 4096, "longest_edge": 25165824},
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
    )
    monkeypatch.setattr(info, "get_video_processor", lambda: video_processor)
    size, tokens = info._get_vision_info(
        image_width=4096,
        image_height=4096,
        image_processor=processor,
        mm_kwargs={},
    )
    assert size.width == size.height == 4096
    assert info.get_image_size_with_most_features() == size
    assert info.get_max_image_tokens() == tokens == IMAGE_TOKENS
    assert (size.height // 16) * (size.width // 16) == 65536
    assert info.get_max_video_tokens(524288, {"video": 1}) == 12288
    dummy = Qwen3VLDummyInputsBuilder(info).get_dummy_mm_data(
        524288, {"image": 1, "video": 1}, {}
    )
    assert dummy["image"][0].size == (4096, 4096)
    video, _ = dummy["video"][0]
    assert video.shape == (2, 4096, 3072, 3)


class Registry:
    def __init__(self, tokens=None, limits=None):
        self.tokens = tokens if tokens is not None else {"image": IMAGE_TOKENS}
        self.limits = limits if limits is not None else dict.fromkeys(self.tokens, 4)
        self.calls = []
        self.processor = SimpleNamespace(
            info=SimpleNamespace(
                supported_mm_limits=dict.fromkeys(self.tokens),
                allowed_mm_limits=self.limits,
                get_mm_max_tokens_per_item=lambda **_: self.tokens,
            )
        )

    def processor_only_cache_from_config(self, config):
        self.calls.append("processor_cache")
        return None

    def create_processor(self, config, cache=None):
        return self.processor

    def get_dummy_mm_inputs(self, config, mm_counts, processor=None):
        assert processor is self.processor
        assert list(mm_counts.values()) == [1]
        modality = next(iter(mm_counts))
        self.calls.append(modality)
        patches = self.tokens[modality] * 4
        # One scalar per patch keeps the CPU fixture small. The production
        # processor's 1536 pixel features per patch are not materialized here.
        return {
            "mm_kwargs": MultiModalKwargsItems.from_hf_inputs(
                BatchFeature(
                    {
                        "pixels": torch.empty(patches, 1),
                        "grid": torch.tensor([[1, 256, patches // 256]]),
                    }
                ),
                {
                    "pixels": MultiModalFieldConfig.flat_from_sizes(
                        modality, torch.tensor([patches])
                    ),
                    "grid": MultiModalFieldConfig.batched(modality),
                },
            )
        }


def config(mode="data", **mm_kwargs):
    mm_config = MultiModalConfig(mm_encoder_tp_mode=mode, **mm_kwargs)
    return SimpleNamespace(
        parallel_config=SimpleNamespace(tensor_parallel_size=TP),
        model_config=SimpleNamespace(
            max_model_len=524288,
            multimodal_config=mm_config,
            get_multimodal_config=lambda: mm_config,
        ),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=8192,
            max_num_seqs=8,
            max_num_encoder_input_tokens=8192,
            encoder_cache_size=8192,
            disable_chunked_mm_input=False,
            enable_chunked_prefill=True,
        ),
    )


def runner_fixture(monkeypatch, cfg, registry, rank=0, **runner_kwargs):
    allocations = []
    original_empty = torch.empty

    def empty(*args, **kwargs):
        tensor = original_empty(*args, **kwargs)
        if args == (IMAGE_TOKENS * TP, HIDDEN_SIZE) and "device" in kwargs:
            allocations.append(weakref.ref(tensor))
        return tensor

    monkeypatch.setattr(torch, "empty", empty)
    events: list[Any] = []
    assignments = []

    def embed_multimodal(pixels, grid):
        # Full encoder-cache reservation must coexist with encoder activations.
        assert allocations and allocations[0]() is not None
        sizes = grid.prod(-1).tolist()
        _, _, per_rank = get_load_balance_assignment(sizes, 4)
        assignments.append(per_rank)
        events.append(("encoder", per_rank[rank], len(sizes)))
        return [torch.empty(n // 4, HIDDEN_SIZE, dtype=torch.float16) for n in sizes]

    encoder = object.__new__(EncoderRunner)
    encoder.model = SimpleNamespace(embed_multimodal=embed_multimodal)
    encoder.hidden_size = HIDDEN_SIZE
    encoder.dtype = torch.float16
    encoder.device = torch.device("cpu")

    def dummy_run(num_tokens, **kwargs):
        events.append(("decoder", num_tokens, kwargs))
        if cfg.model_config.multimodal_config.mm_encoder_tp_mode == "data" and not (
            cfg.model_config.multimodal_config.skip_mm_profiling
        ):
            assert allocations
            cache = allocations[0]()
            assert cache is not None and cache.nbytes == CACHE_BYTES
        return torch.ones(8, 1), torch.ones(8, 1)

    runner = SimpleNamespace(
        supports_mm_inputs=True,
        is_first_pp_rank=True,
        is_last_pp_rank=True,
        model_config=cfg.model_config,
        vllm_config=cfg,
        model_state=SimpleNamespace(encoder_runner=encoder),
        max_num_tokens=8192,
        pooling_runner=None,
        _dummy_run=dummy_run,
        _dummy_sampler_run=lambda _: events.append("sampler"),
    )
    runner.__dict__.update(runner_kwargs)
    monkeypatch.setattr(model_runner, "MULTIMODAL_REGISTRY", registry)
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda: events.append("sync"))
    return runner, encoder, events, allocations, assignments


@pytest.mark.parametrize("rank", [0, 1, 2, 3])
def test_data_profiles_indivisible_image_and_cache_with_full_prefill(monkeypatch, rank):
    cfg, registry = config(), Registry()
    budget = MultiModalBudget(cfg, registry)
    assert budget.encoder_compute_budget == budget.encoder_cache_size == IMAGE_TOKENS
    assert budget.mm_max_items_per_batch == {"image": 1}
    registry.calls.clear()
    runner, _, events, allocations, assignments = runner_fixture(
        monkeypatch, cfg, registry, rank=rank
    )

    model_runner.GPUModelRunner.profile_run(runner)

    assert assignments == [[65536, 0, 0, 0]]
    assert events == [
        ("encoder", 65536 if rank == 0 else 0, 1),
        ("decoder", 8192, {"skip_attn": True, "is_profile": True}),
        "sampler",
        "sync",
    ]
    assert registry.calls == ["processor_cache", "image"]
    assert len(allocations) == 1 and allocations[0]() is None


@pytest.mark.parametrize("quantization", ["awq", "standard_nvfp4"])
def test_weight_tp_keeps_profile_calls_and_extra_reserved_bytes_unchanged(
    monkeypatch, quantization
):
    cfg, registry = config("weights"), Registry()
    cfg.model_config.quantization = quantization
    runner, _, events, allocations, assignments = runner_fixture(
        monkeypatch, cfg, registry
    )
    model_runner.GPUModelRunner.profile_run(runner)
    assert events == [
        ("decoder", 8192, {"skip_attn": True, "is_profile": True}),
        "sampler",
        "sync",
    ]
    assert registry.calls == assignments == allocations == []
    # Zero additional bytes and no changed decoder/sampler inputs, also on
    # the unmodified base. Actual GPU KV and numerical parity need a GPU gate.
    assert sum(ref().nbytes for ref in allocations) == 0


def test_skip_mm_profiling_keeps_explicit_opt_out(monkeypatch):
    cfg, registry = config(skip_mm_profiling=True), Registry()
    runner, _, events, allocations, _ = runner_fixture(monkeypatch, cfg, registry)
    model_runner.GPUModelRunner.profile_run(runner)
    assert events[0][0] == "decoder"
    assert registry.calls == allocations == []


def test_data_profiles_every_enabled_modality_without_resetting_peaks(monkeypatch):
    cfg = config()
    registry = Registry({"image": IMAGE_TOKENS, "video": IMAGE_TOKENS // 2})
    runner, _, events, allocations, assignments = runner_fixture(
        monkeypatch, cfg, registry
    )

    def reset_peak(*args, **kwargs):
        pytest.fail("Encoder diagnostics must not reset the startup profile peak")

    monkeypatch.setenv("VLLM_SM70_LOG_MEMORY_PEAKS", "1")
    monkeypatch.setattr(torch.accelerator, "reset_peak_memory_stats", reset_peak)
    model_runner.GPUModelRunner.profile_run(runner)
    # MultiModalBudget discovers modalities through a set. Both towers must
    # run once; their iteration order is not part of the profiling contract.
    assert registry.calls[0] == "processor_cache"
    assert sorted(registry.calls[1:]) == ["image", "video"]
    assert ("encoder", 65536, 1) in events
    assert ("encoder", 32768, 2) in events
    assert sorted(assignments) == [[32768, 32768, 0, 0], [65536, 0, 0, 0]]
    assert allocations[0]() is None


def test_disabled_towers_do_not_allocate_an_encoder_cache(monkeypatch):
    cfg, registry = config(), Registry(limits={"image": 0})
    _, encoder, _, allocations, _ = runner_fixture(monkeypatch, cfg, registry)
    assert encoder.profile(cfg, registry) is None
    assert allocations == []
    assert registry.calls == ["processor_cache"]


def test_embedding_only_mode_reserves_cache_without_running_a_tower(monkeypatch):
    cfg = config(enable_mm_embeds=True)
    registry = Registry(limits={"image": 0})
    _, encoder, events, allocations, _ = runner_fixture(monkeypatch, cfg, registry)
    cache = encoder.profile(cfg, registry)
    assert cache.nbytes == CACHE_BYTES == 335544320
    assert events == []
    assert len(allocations) == 1


def test_data_encoder_output_retains_padded_all_gather_storage(monkeypatch):
    class Vision(torch.nn.Module):
        spatial_merge_size = 2
        out_hidden_size = 8

        def forward(self, pixels, grid):
            return torch.empty(pixels.shape[0] // 4, self.out_hidden_size)

    monkeypatch.setattr(
        vision_module, "get_tensor_model_parallel_world_size", lambda: TP
    )
    monkeypatch.setattr(vision_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        vision_module,
        "tensor_model_parallel_all_gather",
        lambda tensor, dim: torch.cat([tensor] * TP, dim=dim),
    )
    (output,) = vision_module.run_dp_sharded_mrope_vision_model(
        Vision(), torch.empty(16, 1), [[1, 4, 4]], rope_type="rope_3d"
    )
    assert output.shape == (4, 8)
    assert output.untyped_storage().nbytes() == TP * output.nbytes


@pytest.mark.parametrize("supports_mm,is_first", [(False, True), (True, False)])
def test_text_or_non_first_pp_rank_keeps_original_profile(
    monkeypatch, supports_mm, is_first
):
    cfg, registry = config("weights"), Registry()
    runner, _, events, allocations, _ = runner_fixture(
        monkeypatch,
        cfg,
        registry,
        supports_mm_inputs=supports_mm,
        is_first_pp_rank=is_first,
    )
    model_runner.GPUModelRunner.profile_run(runner)
    assert events[0][0] == "decoder"
    assert registry.calls == allocations == []
