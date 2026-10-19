# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check encoder TP shapes against the resolved quantizer before allocation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class EncoderTPLinear:
    prefix: str
    input_size: int
    output_sizes: tuple[int, ...]
    split: Literal["row", "column"]

    def invalid_reason(self, tp: int, alignment: int) -> str | None:
        dimensions = (self.input_size,) if self.split == "row" else self.output_sizes
        if any(size % tp for size in dimensions):
            return (
                f"{self.prefix}: {self.split} dimensions {dimensions} "
                f"do not divide TP{tp}"
            )
        local_input = self.input_size // tp if self.split == "row" else self.input_size
        if local_input % alignment:
            return (
                f"{self.prefix}: {self.split} input {self.input_size}, TP{tp}, "
                f"local input {local_input} is not a multiple of quantization "
                f"alignment {alignment}"
            )
        return None


def qwen3_vision_tp_linears(vision, prefix: str = "model.visual"):
    """Mirror the column/row splits in Qwen3_VisionTransformer (no weights)."""
    hidden = vision.hidden_size
    intermediate = vision.intermediate_size
    for index in range(vision.depth):
        block = f"{prefix}.blocks.{index}"
        yield EncoderTPLinear(f"{block}.attn.qkv", hidden, (hidden,) * 3, "column")
        yield EncoderTPLinear(f"{block}.attn.proj", hidden, (hidden,), "row")
        yield EncoderTPLinear(
            f"{block}.mlp.linear_fc1", hidden, (intermediate,), "column"
        )
        yield EncoderTPLinear(f"{block}.mlp.linear_fc2", intermediate, (hidden,), "row")
    merged = hidden * vision.spatial_merge_size**2
    mergers = [f"{prefix}.merger"] + [
        f"{prefix}.deepstack_merger_list.{index}"
        for index, _ in enumerate(getattr(vision, "deepstack_visual_indexes", ()))
    ]
    for merger in mergers:
        yield EncoderTPLinear(f"{merger}.linear_fc1", merged, (merged,), "column")
        yield EncoderTPLinear(
            f"{merger}.linear_fc2", merged, (vision.out_hidden_size,), "row"
        )


def resolve_mm_encoder_tp_mode(config: "VllmConfig") -> None:
    model = config.model_config
    if model is None or config.is_speculative_draft or config.quant_config is None:
        return
    mm = model.multimodal_config
    tp = config.parallel_config.tensor_parallel_size
    if (
        mm is None
        or mm.language_model_only
        or mm.mm_encoder_tp_mode == "data"
        or tp == 1
    ):
        return

    from vllm.model_executor.models.config import MODELS_CONFIG_MAP

    architecture_config = MODELS_CONFIG_MAP.get(model.architecture)
    get_shapes = getattr(architecture_config, "get_mm_encoder_tp_linear_shapes", None)
    if get_shapes is None:
        return
    for layer in get_shapes(model):
        alignment = config.quant_config.get_linear_input_alignment(layer.prefix)
        if alignment == 1:
            continue
        reason = layer.invalid_reason(tp, alignment)
        if reason is None:
            continue
        # Replication must itself be valid; otherwise it cannot fix the checkpoint.
        replicated_reason = layer.invalid_reason(1, alignment)
        if replicated_reason is not None:
            raise ValueError(f"Invalid quantized vision layer: {replicated_reason}")
        suggestion = "Use --mm-encoder-tp-mode data to replicate the vision encoder."
        if model._mm_encoder_tp_mode_explicit:
            raise ValueError(
                f"--mm-encoder-tp-mode weights is invalid: {reason}. {suggestion}"
            )
        if not model._model_info.supports_multimodal_encoder_tp_data:
            raise ValueError(f"{reason}. This encoder does not support data TP mode.")
        mm.mm_encoder_tp_mode = "data"
        logger.info_once("Using mm_encoder_tp_mode=data: %s.", reason)
        return
