# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsMultiModal
from vllm.multimodal.encoder_budget import MultiModalBudget
from vllm.multimodal.inputs import MultiModalKwargsItem
from vllm.multimodal.registry import MultiModalRegistry
from vllm.multimodal.utils import group_and_batch_mm_kwargs
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.utils import sanity_check_mm_encoder_outputs

logger = init_logger(__name__)


class EncoderRunner:
    def __init__(
        self,
        model: SupportsMultiModal,
        max_num_tokens: int,
        hidden_size: int,
        encoder_cache: EncoderCache,
        dtype: torch.dtype,
        device: torch.device,
    ):
        self.model = model
        self.max_num_tokens = max_num_tokens
        self.hidden_size = hidden_size
        self.encoder_cache = encoder_cache
        self.dtype = dtype
        self.device = device

        self.inputs_embeds = torch.zeros(
            max_num_tokens, hidden_size, dtype=dtype, device=device
        )

    def profile(
        self, vllm_config: VllmConfig, mm_registry: MultiModalRegistry
    ) -> torch.Tensor | None:
        """Profile replicated encoders and retain cache space for the decoder.

        An item is indivisible in data TP mode: a single maximum-size image
        runs entirely on one rank. Use the real item assignment, without
        dividing the encoder budget by TP. KV allocation takes the minimum
        block count across ranks, so the busiest encoder rank limits every
        rank's KV pool. Weight TP profiling intentionally stays unchanged.
        """
        mm_config = vllm_config.model_config.get_multimodal_config()
        if mm_config.mm_encoder_tp_mode != "data" or mm_config.skip_mm_profiling:
            return None

        budget = MultiModalBudget(vllm_config, mm_registry)
        if budget.encoder_cache_size == 0:
            return None

        # Data TP may return views into a padded all-gather: one item's
        # backing storage can be TP times its logical size. Reserve that
        # full-cache upper bound on every rank, also during encoder execution,
        # and keep it alive through the max_num_batched_tokens decoder profile.
        encoder_cache = torch.empty(
            budget.encoder_cache_size
            * vllm_config.parallel_config.tensor_parallel_size,
            self.hidden_size,
            dtype=self.dtype,
            device=self.device,
        )
        for modality, num_items in budget.mm_max_items_per_batch.items():
            if num_items == 0:
                continue
            # Profile each enabled tower modality: the one with the most
            # output tokens need not have the largest activation peak.
            dummy_inputs = mm_registry.get_dummy_mm_inputs(
                vllm_config.model_config,
                mm_counts={modality: 1},
                processor=budget.processor,
            )
            dummy_item = dummy_inputs["mm_kwargs"][modality][0]
            assert dummy_item is not None
            logger.info(
                "Profiling data-parallel encoder with %d maximum-size %s "
                "items and %d encoder cache tokens (%d reserved bytes) per rank",
                num_items,
                modality,
                budget.encoder_cache_size,
                encoder_cache.nbytes,
            )
            for _, batch_size, kwargs in group_and_batch_mm_kwargs(
                [(modality, dummy_item)] * num_items,
                device=self.device,
                pin_memory=False,
            ):
                # Do not use execute_mm_encoder: its optional diagnostic
                # logging resets peak stats and would discard earlier peaks.
                outputs = self.model.embed_multimodal(**kwargs)
                sanity_check_mm_encoder_outputs(outputs, expected_num_items=batch_size)
                del outputs
            del kwargs, dummy_inputs, dummy_item
        budget.reset_cache()
        return encoder_cache

    def prepare_mm_inputs(
        self, scheduled_encoder_inputs: dict[str, list[int]]
    ) -> tuple[list[str], list[tuple[str, MultiModalKwargsItem]]]:
        mm_hashes: list[str] = []
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]] = []
        for req_id, encoder_input_ids in scheduled_encoder_inputs.items():
            mm_features = self.encoder_cache.mm_features[req_id]
            for mm_input_id in encoder_input_ids:
                mm_feature = mm_features[mm_input_id]
                if mm_feature.data is None:
                    continue
                mm_hashes.append(mm_feature.identifier)
                mm_kwargs.append((mm_feature.modality, mm_feature.data))

        return mm_hashes, mm_kwargs

    @torch.inference_mode()
    def execute_mm_encoder(
        self,
        mm_kwargs: list[tuple[str, MultiModalKwargsItem]],
    ) -> list[torch.Tensor]:
        log_peak = envs.VLLM_SM70_LOG_MEMORY_PEAKS
        if log_peak:
            torch.accelerator.synchronize()
            base = torch.accelerator.memory_allocated(self.device)
            torch.accelerator.reset_peak_memory_stats(self.device)
        encoder_outputs: list[torch.Tensor] = []
        for modality, num_items, mm_kwargs_batch in group_and_batch_mm_kwargs(
            mm_kwargs, device=self.device, pin_memory=False
        ):
            batch_outputs = self.model.embed_multimodal(**mm_kwargs_batch)
            sanity_check_mm_encoder_outputs(batch_outputs, expected_num_items=num_items)
            encoder_outputs.extend(batch_outputs)
        if log_peak:
            torch.accelerator.synchronize()
            peak = torch.accelerator.max_memory_allocated(self.device)
            held = torch.accelerator.memory_allocated(self.device)
            logger.info(
                "Encoder memory: %d items, peak +%.1f MiB, outputs held "
                "+%.1f MiB over %.1f MiB allocated",
                len(mm_kwargs),
                (peak - base) / 2**20,
                (held - base) / 2**20,
                base / 2**20,
            )
        return encoder_outputs

    def gather_mm_embeddings(
        self,
        req_ids: list[str],
        total_num_scheduled_tokens: int,
        num_scheduled_tokens: np.ndarray,
        query_start_loc: np.ndarray,
        prefill_lens: np.ndarray,
        computed_prefill_lens: np.ndarray,
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        is_prefilling = (computed_prefill_lens < prefill_lens).tolist()
        all_decode = not any(is_prefilling)
        if all_decode:
            # All decode requests, so no need to gather any embeddings.
            return [], torch.zeros(
                total_num_scheduled_tokens, dtype=torch.bool, device=self.device
            )

        query_start = computed_prefill_lens.tolist()
        query_end = (computed_prefill_lens + num_scheduled_tokens).tolist()

        mm_embeds: list[torch.Tensor] = []
        is_mm_embed = torch.zeros(
            total_num_scheduled_tokens, dtype=torch.bool, device="cpu"
        )
        for i, req_id in enumerate(req_ids):
            if not is_prefilling[i]:
                # OPTIMIZATION: Skip decode requests.
                continue

            mm_features = self.encoder_cache.mm_features[req_id]
            for mm_feature in mm_features:
                pos_info = mm_feature.mm_position
                start_pos = pos_info.offset
                num_encoder_tokens = pos_info.length

                if start_pos >= query_end[i]:
                    # The encoder output is not needed in this step.
                    break
                if start_pos + num_encoder_tokens <= query_start[i]:
                    # The encoder output is already processed and stored
                    # in the decoder's KV cache.
                    continue

                start_idx = max(query_start[i] - start_pos, 0)
                end_idx = min(query_end[i] - start_pos, num_encoder_tokens)
                assert start_idx < end_idx
                curr_embeds_start, curr_embeds_end = (
                    pos_info.get_embeds_indices_in_range(start_idx, end_idx)
                )
                # If there are no embeddings in the current range, we skip
                # gathering the embeddings.
                if curr_embeds_start == curr_embeds_end:
                    continue

                mm_hash = mm_feature.identifier
                encoder_output = self.encoder_cache.encoder_outputs.get(mm_hash, None)
                assert encoder_output is not None, f"Encoder cache miss for {mm_hash}."

                if (is_embed := pos_info.is_embed) is not None:
                    is_embed = is_embed[start_idx:end_idx]
                    mm_embeds_item = encoder_output[curr_embeds_start:curr_embeds_end]
                else:
                    mm_embeds_item = encoder_output[start_idx:end_idx]

                req_start_pos = query_start_loc[i] + start_pos - query_start[i]
                is_mm_embed[req_start_pos + start_idx : req_start_pos + end_idx] = (
                    True if is_embed is None else is_embed
                )
                mm_embeds.append(mm_embeds_item)

        return mm_embeds, is_mm_embed

    @torch.inference_mode()
    def get_inputs_embeds(
        self,
        input_ids: torch.Tensor,
        mm_embeds: list[torch.Tensor],
        is_mm_embed: torch.Tensor,
    ) -> torch.Tensor:
        x = self.model.embed_input_ids(
            input_ids, multimodal_embeddings=mm_embeds, is_multimodal=is_mm_embed
        )
        # Copy to the pre-allocated buffer for CUDA graphs.
        self.inputs_embeds[: x.shape[0]] = x
        return self.inputs_embeds
