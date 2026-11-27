# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only Qwen4Exp MTP (Multi-Token Predictor) model.

The MTP draft model reuses the Qwen4Exp backbone (PLE/HC/MoE) but:
  - drops all multi-modal handling (text-only),
  - forces PLE off while keeping the main model's HC stream count,
  - fuses the backbone hidden and the new-token embedding via
    ``residual_linear_shared`` (fc_embedding + shared fc_hidden) instead of
    the ``Linear(2H, H)`` + repeat used by other MTP variants,
  - emits TWO hidden streams per step (scheme A): a single stream [T, H]
    (final-mixer collapsed, fed to the LM head) and a pre-final-mixer
    multi stream [T, hc_count*H] (fed to the next draft step).
"""

from collections.abc import Iterable

import regex as re
import torch
from torch import nn

import vllm.envs as envs
from vllm.compilation.decorators import support_torch_compile
from vllm.compilation.sm70_decode_graph import is_sm70_decode_graph_compiling
from vllm.config import SpeculativeConfig, VllmConfig, replace, set_current_vllm_config
from vllm.distributed import get_pp_group, tensor_model_parallel_all_gather
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.utils import configure_quant_config
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    get_draft_quant_config,
    make_empty_intermediate_tensors_factory,
    maybe_fuse_shared_experts,
    maybe_prefix,
)
from vllm.platforms import current_platform
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.configs.qwen4_exp import (
    Qwen4ExpTextConfig,
)

from ..common.qsa_cache import qsa_draft_dcp_sharded
from .hyperconnection import GatedResidual, HyperConnectionConfig

try:
    from .low_latency_gemm import enable_qwen4_exp_low_latency_gemm
except ModuleNotFoundError as exc:
    # The Blackwell-only CuTe DSL helper is absent from the 1Cat SM70 tree.
    if exc.name != "vllm.model_executor.kernels.linear.cute_dsl":
        raise

    def enable_qwen4_exp_low_latency_gemm(
        module: nn.Module, dtype: torch.dtype
    ) -> None:
        del module, dtype


from .model import (
    _HC_WEIGHTS_MAPPER,
    _QWEN3_5_WEIGHTS_MAPPER,
    _QWEN4_EXP_IGNORED_MISSING_SUFFIXES,
    Qwen4ExpDecoderLayer,
    Qwen4ExpMixtureOfExperts,
    _make_qwen38_decode_compile_config,
)
from .sm70_fp16_gemv import enable_qwen38_sm70_fp16_gemv
from .sm70_fp16_hc import enable_qwen38_sm70_fp16_fused_hc
from .sm70_mtp_head import MTPQPN8LogitsProcessor

logger = init_logger(__name__)


def _remap_ignored_layers(
    ignored_layers: list[str],
    mtp_start_layer_idx: int,
) -> list[str]:
    remapped: list[str] = []
    for name in ignored_layers:
        if name.startswith("mtp."):
            new_name = re.sub(
                r"(?<=\.layers\.)\d+",
                lambda m: str(mtp_start_layer_idx + int(m.group(0))),
                name,
            )
            remapped.append(new_name)
        else:
            remapped.append(name)
    return remapped


def _remap_quantized_layers(
    quantized_layers: dict[str, dict],
    mtp_start_layer_idx: int,
) -> dict[str, dict]:
    """Map checkpoint MTP layer indices to standalone draft indices."""
    return {
        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: layer_info
        for name, layer_info in quantized_layers.items()
    }


def _remap_mtp_weight_name(name: str) -> str | None:
    """Map Qwen4Exp checkpoint paths into the standalone draft model."""

    for checkpoint_prefix in (
        "model.language_model.",
        "language_model.",
    ):
        if name.startswith(checkpoint_prefix):
            name = name.removeprefix(checkpoint_prefix)
            break

    if name.startswith("embed_tokens."):
        name = f"model.{name}"
    if name.startswith("model.mtp."):
        name = name.removeprefix("model.")
    if name.startswith("mtp.shared_head.head."):
        return name.replace("mtp.shared_head.head.", "lm_head.", 1)
    if name.startswith("model.shared_head.head."):
        return name.replace("model.shared_head.head.", "lm_head.", 1)
    if name.startswith("shared_head.head."):
        return name.replace("shared_head.head.", "lm_head.", 1)
    if name.startswith("model.lm_head."):
        return name.removeprefix("model.")
    if name.startswith("mtp."):
        return name.replace("mtp.", "model.", 1)
    if name.startswith("model.embed_tokens.") or name.startswith("lm_head."):
        return name
    return None


def _validate_mtp_expert_weights_loaded(
    model: nn.Module,
    loaded_weights: set[str],
) -> None:
    """Reject a silently incomplete Qwen4Exp MTP routed-expert load."""

    required_suffixes = (
        ".mlp.experts.w13_weight",
        ".mlp.experts.w2_weight",
    )
    required = {
        name for name, _ in model.named_parameters() if name.endswith(required_suffixes)
    }
    missing = required - loaded_weights
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(
            "Qwen4Exp MTP routed-expert checkpoint weights were not loaded: "
            f"{missing_names}. Check fused/per-expert checkpoint mappings."
        )


def _mtp_fp8_experts_supported(
    draft_vllm_config: VllmConfig,
    draft_quant_config: QuantizationConfig,
    speculative_config: SpeculativeConfig,
    exact_sm70: bool,
    online_fp8: bool,
) -> bool:
    """Whether the SM70 FP8 MTP expert method can serve this draft.

    Checkpoint FP8 experts are allowed under pipeline parallelism: the V2
    runner builds the speculator only on the last pipeline rank, so the
    drafter is stage-local and the FP8 expert padding follows that stage's
    tensor-parallel size. Online conversion keeps the single-stage limit
    until it is validated under pipeline parallelism as well.
    """
    return (
        exact_sm70
        and (
            not online_fp8
            or draft_vllm_config.parallel_config.pipeline_parallel_size == 1
        )
        and draft_vllm_config.model_config.dtype == torch.float16
        and draft_quant_config.get_name()
        in ("awq", "modelopt_fp4", "modelopt_mixed", "fp8")
        and not draft_vllm_config.parallel_config.enable_expert_parallel
        and speculative_config.rejection_sample_method == "standard"
    )


def _get_mtp_quant_config(
    vllm_config: VllmConfig, mtp_start_layer_idx: int
) -> QuantizationConfig | None:
    draft_quant_config = get_draft_quant_config(vllm_config)

    # inject packed and ignored modules to the quantization config of draft model
    if draft_quant_config is not None:
        configure_quant_config(draft_quant_config, Qwen4ExpMTP)
        ignored_layers = getattr(draft_quant_config, "ignored_layers", None)
        if ignored_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "ignored_layers",
                _remap_ignored_layers(ignored_layers, mtp_start_layer_idx),
            )
        exclude_modules = getattr(draft_quant_config, "exclude_modules", None)
        if exclude_modules:
            setattr(  # noqa: B010
                draft_quant_config,
                "exclude_modules",
                _remap_ignored_layers(exclude_modules, mtp_start_layer_idx),
            )
        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)
        if quantized_layers:
            setattr(  # noqa: B010
                draft_quant_config,
                "quantized_layers",
                _remap_quantized_layers(quantized_layers, mtp_start_layer_idx),
            )

    return draft_quant_config


def _mtp_expert_prefixes(config, mtp_start_layer_idx: int) -> set[str]:
    return {
        f"mtp.layers.{mtp_start_layer_idx + index}.mlp.experts"
        for index in range(getattr(config, "mtp_num_hidden_layers", 1))
    }


def resolve_mtp_expert_default(vllm_config: VllmConfig) -> None:
    """Default SM70 Qwen4Exp MTP routed experts to Q8_0 when the draft
    checkpoint stores them as floating-point or block-FP8 tensors."""
    from vllm.model_executor.layers.quantization.awq import AWQConfig
    from vllm.model_executor.layers.quantization.awq_marlin import AWQMarlinConfig
    from vllm.model_executor.layers.quantization.modelopt import (
        ModelOptMixedPrecisionConfig,
        ModelOptQuantConfigBase,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        is_layer_skipped,
    )

    from .mtp_fp8_experts import checkpoint_fp8_prefixes
    from .mtp_q8_experts import supports_q8_0_expert_shape

    spec = vllm_config.speculative_config
    assert spec is not None and spec.draft_model_config is not None
    draft = spec.draft_model_config
    parallel = vllm_config.parallel_config
    # RoutedExperts shards the expert intermediate size across TP, DP and PCP.
    expert_tp = (
        parallel.tensor_parallel_size
        * parallel.data_parallel_size
        * parallel.prefill_context_parallel_size
    )
    if (
        draft.dtype != torch.float16
        or spec.rejection_sample_method != "standard"
        or (parallel.enable_expert_parallel and expert_tp > 1)
    ):
        return
    config = draft.hf_text_config
    local_width, remainder = divmod(config.moe_intermediate_size, expert_tp)
    if remainder or not supports_q8_0_expert_shape(
        config.hidden_size, local_width, config.num_experts_per_tok
    ):
        return

    start = config.num_hidden_layers
    quant = _get_mtp_quant_config(vllm_config, start)
    prefixes = _mtp_expert_prefixes(config, start)
    if quant is not None:
        remaining = prefixes - checkpoint_fp8_prefixes(quant, prefixes)
        if isinstance(quant, ModelOptQuantConfigBase):
            remaining = {
                prefix for prefix in remaining if not quant.is_layer_excluded(prefix)
            }
            if isinstance(quant, ModelOptMixedPrecisionConfig):
                remaining = {
                    prefix
                    for prefix in remaining
                    if quant._resolve_quant_algo(prefix) is not None
                }
        elif isinstance(quant, (AWQConfig, AWQMarlinConfig)):
            remaining = {
                prefix
                for prefix in remaining
                if not is_layer_skipped(
                    prefix, quant.modules_to_not_convert, skip_with_substr=True
                )
            }
        if remaining:
            return
    spec.mtp_expert_quantization = "q8_0"
    logger.info_once("Defaulting SM70 Qwen4Exp MTP routed experts to Q8_0.")


def _make_draft_vllm_config(
    vllm_config: VllmConfig,
    mtp_start_layer_idx: int,
) -> VllmConfig:
    """Ensure that the draft model config is set in the vLLM config."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.draft_model_config is None:
        raise ValueError("speculative_config.draft_model_config must be set")

    draft_quant_config = _get_mtp_quant_config(vllm_config, mtp_start_layer_idx)
    draft_vllm_config = replace(
        vllm_config,
        model_config=speculative_config.draft_model_config,
        is_speculative_draft=True,
    )
    # VllmConfig post-init derives the target quant config, so restore the
    # independently resolved draft quant config after replacement.
    from vllm.model_executor.layers.quantization.sm70_turbomind import (
        is_exact_sm70_cuda_platform,
    )

    from .mtp_fp8_experts import MTPExpertQuantConfig, checkpoint_fp8_prefixes

    expert_format = getattr(speculative_config, "mtp_expert_quantization", None)
    online_fp8 = expert_format == "fp8"
    exact_sm70 = is_exact_sm70_cuda_platform()
    checkpoint_prefixes = set()
    if draft_quant_config is not None and exact_sm70:
        config = draft_vllm_config.model_config.hf_text_config
        prefixes = _mtp_expert_prefixes(config, mtp_start_layer_idx)
        checkpoint_prefixes = checkpoint_fp8_prefixes(draft_quant_config, prefixes)
    if expert_format in ("q8_0", "fp16"):
        if not exact_sm70 or draft_vllm_config.model_config.dtype != torch.float16:
            raise ValueError(f"MTP {expert_format} requires SM70 with FP16 activations")
        # Block-FP8 checkpoint experts are dequantized in load_weights.
        draft_quant_config = MTPExpertQuantConfig(
            draft_quant_config, checkpoint_prefixes, expert_format=expert_format
        )
    elif online_fp8 or checkpoint_prefixes:
        if draft_quant_config is None or not _mtp_fp8_experts_supported(
            draft_vllm_config,
            draft_quant_config,
            speculative_config,
            exact_sm70,
            online_fp8,
        ):
            raise ValueError(
                "MTP FP8 experts require SM70, FP16, an AWQ/ModelOpt/FP8 draft "
                "checkpoint, no expert parallelism, standard rejection sampling, "
                "and pipeline-parallel size 1 for online conversion"
            )
        draft_quant_config = MTPExpertQuantConfig(
            draft_quant_config,
            checkpoint_prefixes,
            quantize_unquantized=online_fp8,
        )
    draft_vllm_config.quant_config = draft_quant_config
    return draft_vllm_config


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class Qwen4ExpMultiTokenPredictor(nn.Module):
    hf_to_vllm_mapper = _QWEN3_5_WEIGHTS_MAPPER | _HC_WEIGHTS_MAPPER

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()

        model_config = vllm_config.model_config
        config: Qwen4ExpTextConfig = model_config.hf_text_config

        self.config = config
        self.vocab_size = config.vocab_size

        self.mtp_start_layer_idx = config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "mtp_num_hidden_layers", 1)

        self.hidden_size = config.hidden_size
        self.hc_count = config.hc_count

        self.embed_tokens = VocabParallelEmbedding(self.vocab_size, self.hidden_size)
        draft_vllm_config = _make_draft_vllm_config(
            vllm_config,
            self.mtp_start_layer_idx,
        )
        self.fp8_mtp_checkpoint_prefixes = {
            f"model.layers.{int(name.split('.')[2]) - self.mtp_start_layer_idx}"
            ".mlp.experts"
            for name in getattr(
                draft_vllm_config.quant_config, "checkpoint_prefixes", ()
            )
        }
        self.fp8_mtp_tp_size = draft_vllm_config.parallel_config.tensor_parallel_size
        self.mtp_expert_format = getattr(
            draft_vllm_config.quant_config, "expert_format", None
        )
        # Phase 2: the drafter shares the target cache_config; remember the main
        # KV cache dtype so load_weights can finalize the draft QSA E4M3 scales.
        self._kv_cache_dtype = draft_vllm_config.cache_config.cache_dtype
        with set_current_vllm_config(draft_vllm_config, prefix=prefix):
            # residual_linear_shared fusion: fc_embedding projects the token
            # embedding, fc_hidden (shared across HC branches) projects the
            # backbone hidden; the embedding is added as a residual to every
            # branch (see mtp_residual_linear_shared.md).
            self.fc_embedding = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=False,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_embedding",
            )
            self.fc_hidden = ColumnParallelLinear(
                self.hidden_size,
                self.hidden_size,
                gather_output=False,
                bias=False,
                return_bias=False,
                quant_config=draft_vllm_config.quant_config,
                prefix=f"{prefix}.fc_hidden",
            )
            dcp_local_indices_buffer: torch.Tensor | None = None
            dcp_partial_output_buffer: torch.Tensor | None = None
            dcp_partial_lse_buffer: torch.Tensor | None = None
            if qsa_draft_dcp_sharded(draft_vllm_config):
                parallel_config = draft_vllm_config.parallel_config
                if parallel_config.decode_context_parallel_size != 2:
                    raise NotImplementedError("Qwen4Exp QSA supports DCP1 or DCP2")
                # Draft graphs capture their own addresses; do not borrow the
                # target's workspaces or overwrite the draft's top-k selection.
                max_tokens = draft_vllm_config.scheduler_config.max_num_batched_tokens
                dcp_local_indices_buffer = torch.empty(
                    max_tokens,
                    int(config.indexer_budget) + int(config.indexer_compress_ratio) - 1,
                    dtype=torch.int32,
                )
                gathered_heads = (
                    int(config.num_attention_heads)
                    // parallel_config.tensor_parallel_size
                    * 2
                )
                dcp_partial_output_buffer = torch.empty(
                    max_tokens,
                    gathered_heads,
                    int(config.head_dim),
                    dtype=torch.float32,
                )
                dcp_partial_lse_buffer = torch.empty(
                    max_tokens, gathered_heads, dtype=torch.float32
                )
                self.dcp_local_indices_buffer = dcp_local_indices_buffer
                self.dcp_partial_output_buffer = dcp_partial_output_buffer
                self.dcp_partial_lse_buffer = dcp_partial_lse_buffer
            self.layers = nn.ModuleList(
                Qwen4ExpDecoderLayer(
                    draft_vllm_config,
                    layer_type="full_attention",
                    prefix=f"{prefix}.layers.{self.mtp_start_layer_idx + idx}",
                    dcp_local_indices_buffer=dcp_local_indices_buffer,
                    dcp_partial_output_buffer=dcp_partial_output_buffer,
                    dcp_partial_lse_buffer=dcp_partial_lse_buffer,
                )
                for idx in range(self.num_mtp_layers)
            )

        self.pre_fc_norm_embedding = GemmaRMSNorm(
            self.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_fc_norm_hidden = GemmaRMSNorm(
            self.hidden_size * self.hc_count, eps=config.rms_norm_eps
        )
        # HC final mixer collapses the multi stream into [T, H] for the LM head.
        hc_config = HyperConnectionConfig(
            hc_count=config.hc_count,
            hidden_size=config.hidden_size,
            params_dtype=model_config.dtype,
            hc_lowrank=config.hc_lowrank,
            rms_norm_eps=config.rms_norm_eps,
            hc_per_branch_norm=True,
        )
        self.hyper_connection_mixer = GatedResidual(
            hc_config,
            use_combine=False,
            prefix=maybe_prefix(prefix, "hyper_connection_mixer"),
        )
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states"], self.hidden_size * self.hc_count
        )

    def _iter_qsa_attentions(self):
        """Yield MTP attention modules that own a QSA indexer."""

        for layer in self.layers:
            attention = getattr(layer, "self_attn", None)
            if (
                attention is not None
                and getattr(attention, "indexer", None) is not None
            ):
                yield attention

    def set_skip_topk(self, skip: bool) -> None:
        """Select on MTP step 0 and reuse its QSA indices on later steps."""

        for attention in self._iter_qsa_attentions():
            attention.indexer.skip_topk = skip

    def compact_topk_indices(self, row_indices: torch.Tensor) -> None:
        """Keep each request's target-aligned step-0 sparse-index row."""

        num_rows = row_indices.numel()
        for attention in self._iter_qsa_attentions():
            buffer = attention.topk_indices_buffer
            selected = buffer.index_select(0, row_indices)
            buffer[:num_rows].copy_(selected)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    @property
    def supports_kv_only_prefill(self) -> bool:
        if self.num_mtp_layers != 1:
            return False
        layer = self.layers[0]
        return layer.ple is None and hasattr(layer.self_attn, "forward_kv_only")

    def forward_kv_only(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> bool:
        if not self.supports_kv_only_prefill:
            return False
        layer = self.layers[0]
        hidden_states = self._project_inputs(
            self.embed_input_ids(input_ids), hidden_states
        )
        _, block_input, _ = layer.attn_hyper_connection.mix(hidden_states)
        layer.self_attn.forward_kv_only(positions, block_input)
        return True

    def _project_inputs(
        self, inputs_embeds: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        inputs_embeds = self.fc_embedding(self.pre_fc_norm_embedding(inputs_embeds))
        # Normalize the whole pre-final-mixer row across all HC streams.
        hidden_states = hidden_states.view(
            hidden_states.shape[0], self.hc_count, self.hidden_size
        )
        hidden_states = self.pre_fc_norm_hidden(hidden_states.flatten(-2)).view_as(
            hidden_states
        )
        hidden_states = self.fc_hidden(hidden_states)
        # Add the embedding residual on the local feature shard before gather.
        return tensor_model_parallel_all_gather(
            inputs_embeds.unsqueeze(-2) + hidden_states
        ).flatten(-2)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        *,
        row_indices: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # The drafter is stage-local, so this module is always a complete
        # model: gpu_model_runner returns the IntermediateTensors on every
        # non-final pipeline rank before speculation is reached, and the
        # weights here are replicated rather than partitioned (embed_tokens,
        # fc_embedding, fc_hidden and every MTP layer are built on all ranks).
        # Branching on the TARGET model's pipeline position sent the final
        # rank into the "receive from the previous stage" path and asserted on
        # intermediate tensors that nobody sends -- with the fullgraph AOT
        # compile of 1.5.0 that is a hard compile error, so no k > 0 boots at
        # all under pipeline parallelism.
        assert hidden_states is not None
        if inputs_embeds is None:
            assert input_ids is not None
            inputs_embeds = self.embed_input_ids(input_ids)
        hidden_states = self._project_inputs(inputs_embeds, hidden_states)

        current_step_idx = spec_step_idx % self.num_mtp_layers
        layer = self.layers[current_step_idx]
        hidden_states, block_output, injection = layer(
            hidden_states=hidden_states,
            prev_block_output=None,
            prev_injection=None,
            positions=positions,
            input_ids=None,
            query_start_loc=None,
            ngram_context=None,
            row_indices=row_indices,
        )
        # Last PP rank finalize. Keep both:
        #   (A) sample_hidden_states [T, H]  -> single stream for the LM head
        #   (B) multi_hidden [T, hc_count*H] -> pre-final-mixer multi stream
        #       for the next draft step (zero extra compute, just kept).
        multi_hidden, sample_hidden_states, _ = (
            self.hyper_connection_mixer.combine_and_mix(
                hidden_states, block_output, injection
            )
        )
        return sample_hidden_states, multi_hidden

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        weights = maybe_fuse_shared_experts(
            weights,
            n_routed_experts=getattr(self.config, "num_experts", 0) or 0,
            n_shared_experts=1,
            ckpt_prefix="mlp.shared_expert",
        )
        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        return loader.load_weights(weights, mapper=self.hf_to_vllm_mapper)


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    }
)
class _Qwen4ExpMTPDecodeGraphModel(nn.Module):
    """Small-shape compiled view sharing all drafter parameters and state."""

    def __init__(
        self,
        *,
        target_model: Qwen4ExpMultiTokenPredictor,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__()
        object.__setattr__(self, "_target_model", target_model)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        return self._target_model.forward(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx,
        )


@support_torch_compile(
    # As on the target, selection between the two compiled backbones must
    # remain outside the first, prefill-specialized compiled wrapper.
    enable_if=lambda cfg: not envs.VLLM_SM70_QWEN38_DUAL_COMPILE,
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
        "hidden_states": 0,
    },
)
class Qwen4ExpMTP(nn.Module, SupportsPP, Qwen4ExpMixtureOfExperts):
    # A BF16-named shard may contain only KV scales, with the MTP body in
    # ordinary shards. Select the full checkpoint, including compact BF16
    # layouts; skip_checkpoint_weight filters target tensors before reading
    # them in the standard mapped/direct safetensors iterator.
    allow_patterns_overrides = [
        "*.safetensors",
        "*.bin",
        "*.pt",
    ]

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
        "input_mix_weight_down_block_inject": [
            "input_mix_weight_down",
            "block_inject_weight",
            "_input_mix_padding",
        ],
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config: Qwen4ExpTextConfig = vllm_config.model_config.hf_text_config
        self.vllm_config = vllm_config
        cache_config = vllm_config.cache_config
        if cache_config.mamba_cache_mode == "all":
            raise NotImplementedError(
                "Qwen4ExpMTP currently does not support 'all' prefix caching, "
                "please use '--mamba-cache-mode=align' instead"
            )

        self.quant_config = vllm_config.quant_config

        super().__init__()
        self.config = config
        self.model = Qwen4ExpMultiTokenPredictor(
            vllm_config=vllm_config,
            prefix=maybe_prefix(prefix, "mtp"),
        )
        if vllm_config.kernel_config.sm70_mtp_draft_qpn8:
            from vllm.model_executor.layers.quantization.sm70_mtp_draft_qpn8 import (
                mark_mtp_draft_qpn8,
            )

            # fc_hidden projects each HC branch separately for every token.
            max_capture_tokens = (
                vllm_config.compilation_config.max_cudagraph_capture_size or 0
            )
            mark_mtp_draft_qpn8(
                self.model, max_capture_rows=max_capture_tokens * config.hc_count
            )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)
        self.make_empty_intermediate_tensors = (  # type: ignore[method-assign]
            self.model.make_empty_intermediate_tensors
        )
        self.set_moe_parameters(self.model.layers)
        enable_qwen4_exp_low_latency_gemm(self, vllm_config.model_config.dtype)
        config_dtype = vllm_config.model_config.dtype
        enable_qwen38_sm70_fp16_gemv(self, config_dtype, vllm_config)
        enable_qwen38_sm70_fp16_fused_hc(self, config_dtype, vllm_config)
        object.__setattr__(self, "_sm70_decode_graph_model", None)

    def prepare_sm70_mtp_draft_head(self) -> None:
        weight = self.lm_head.weight
        n, k = weight.shape
        if (
            not current_platform.is_device_capability(70)
            or weight.dtype != torch.float16
            or n % 32
            or k % 128
        ):
            return
        # The draft-only pack only replaces greedy top-1 selection; full
        # logits still use the shared FP16 head.
        self.logits_processor = MTPQPN8LogitsProcessor(self.config.vocab_size, weight)
        logger.info_once("SM70 greedy MTP draft QPN8 head enabled.")

    def prepare_sm70_decode_graph_model(self) -> bool:
        if not envs.VLLM_SM70_QWEN38_DUAL_COMPILE:
            return False
        if self._sm70_decode_graph_model is None:
            decode_config = _make_qwen38_decode_compile_config(self.vllm_config)
            with set_current_vllm_config(decode_config):
                decode_model = _Qwen4ExpMTPDecodeGraphModel(
                    target_model=self.model, vllm_config=decode_config
                )
            object.__setattr__(self, "_sm70_decode_graph_model", decode_model)
            logger.info_once(
                "Prepared shared-weight SM70 Qwen3.8 MTP decode compiler; "
                "supported draft shapes reuse the common FP16 GEMV/HC routes."
            )
        return True

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    @property
    def supports_last_row_prefill(self) -> bool:
        return self.model.supports_kv_only_prefill

    def forward_last_rows(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        row_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.forward(
            input_ids, positions, hidden_states, row_indices=row_indices
        )

    def forward_kv_only(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> bool:
        return self.model.forward_kv_only(input_ids, positions, hidden_states)

    def forward(  # type: ignore[override]
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor] | IntermediateTensors:
        backbone = self.model
        if envs.VLLM_SM70_QWEN38_DUAL_COMPILE and is_sm70_decode_graph_compiling():
            backbone = self._sm70_decode_graph_model
            if backbone is None:
                raise RuntimeError("SM70 Qwen3.8 MTP decode compiler was not prepared")
        return backbone(
            input_ids,
            positions,
            hidden_states,
            intermediate_tensors,
            inputs_embeds,
            spec_step_idx=spec_step_idx,
        )

    def compute_logits(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    def get_top_tokens(
        self, hidden_states: torch.Tensor, spec_step_idx: int = 0
    ) -> torch.Tensor:
        return self.logits_processor.get_top_tokens(self.lm_head, hidden_states)

    def skip_checkpoint_weight(self, name: str) -> bool:
        # The drafter ships inside its target's checkpoint; without this the
        # loader reads the whole target again only for load_weights to drop
        # everything but the MTP tensors.
        return _remap_mtp_weight_name(name) is None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        def remap_weight_names():
            for name, weight in weights:
                remapped_name = _remap_mtp_weight_name(name)
                if remapped_name is not None:
                    yield remapped_name, weight

        loader = AutoWeightsLoader(
            self,
            skip_substrs=["hyper_connection_mixer.block_inject_weight"],
            ignore_unexpected_suffixes=_QWEN4_EXP_IGNORED_MISSING_SUFFIXES.copy(),
        )
        from .mtp_fp8_checkpoint import prepare_mtp_fp8_checkpoint

        expert_format = self.model.mtp_expert_format
        if expert_format == "q8_0" and self.model.fp8_mtp_checkpoint_prefixes:
            logger.info_once(
                "Q8_0 MTP expert secondary quantization: dequantizing checkpoint "
                "block-FP8 experts before packing Q8_0. Prefer a BF16/FP16 source."
            )
        loaded_weights = loader.load_weights(
            prepare_mtp_fp8_checkpoint(
                remap_weight_names(),
                self.model.fp8_mtp_checkpoint_prefixes,
                tp_size=self.model.fp8_mtp_tp_size,
                num_experts=self.model.config.num_experts,
                dequantize=expert_format in ("q8_0", "fp16"),
            )
        )
        _validate_mtp_expert_weights_loaded(self, loaded_weights)
        # Phase 2: an E4M3 main KV cache makes the drafter's QSA layer E4M3 too.
        # Finalize its calibrated k/v scales strictly from the checkpoint (no
        # runtime fallback; missing names are listed). No-op for FP16/BF16 caches
        # and on the PLE offload process (is_offload_process()).
        from .model import _finalize_qsa_e4m3_scale_load

        _finalize_qsa_e4m3_scale_load(
            self,
            loaded_weights,
            self.model._kv_cache_dtype,
            require_calibrated_speculative_draft=True,
        )
        return loaded_weights


__all__ = ["Qwen4ExpMTP", "Qwen4ExpMultiTokenPredictor"]
