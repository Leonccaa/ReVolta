# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams, envs
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    CrossAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    UniformTypeKVCacheSpecs,
    has_block_segments,
)
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

if TYPE_CHECKING:
    from vllm.models.qwen4_exp.nvidia.host_kv.history import HostHistoryStore
    from vllm.v1.kv_offload.host.page_table import HostKVStep


def _kernel_prefill_warmup_token_counts(
    model_runner: GPUModelRunner,
    default_prompt_len: int,
) -> tuple[int, ...]:
    """Collect per-request prefill sizes advertised by active kernel owners."""
    max_tokens = min(
        model_runner.scheduler_config.max_num_batched_tokens,
        model_runner.max_model_len,
    )
    token_counts = {default_prompt_len}
    static_context = model_runner.compilation_config.static_forward_context
    for layer in static_context.values():
        for token_count in getattr(layer, "kernel_warmup_prefill_token_counts", ()):
            if (
                isinstance(token_count, int)
                and not isinstance(token_count, bool)
                and default_prompt_len < token_count <= max_tokens
            ):
                token_counts.add(token_count)
    return tuple(sorted(token_counts))


def _host_kv_warmup_profiles(
    model_runner: GPUModelRunner,
    prompt_lengths: tuple[int, ...],
) -> list[tuple[int, int]]:
    """Eager prefill profiles that reach the host KV staged write path.

    ``GPUModelRunner._set_host_write_plan`` plans staged writes only for
    eagerly executed steps, and the plan kind depends on the batch: one
    bound request takes the runs plan (word writer kernel), any other
    prefill batch falls back to the stage kernel. Run one profile of each
    shape at the first token count ``dispatch`` can no longer route to a
    captured graph.
    """
    max_tokens = min(
        model_runner.scheduler_config.max_num_batched_tokens,
        model_runner.max_model_len,
    )
    manager = model_runner.cudagraph_manager
    captured = manager.max_captured_tokens if manager is not None else -1
    eager = max(captured, max(prompt_lengths, default=0)) + 1
    if eager > max_tokens:
        return []
    return [(eager, 1), (eager, 2)]


def _dflash_warmup_profiles(
    model_runner: GPUModelRunner,
    profiled_lengths: Sequence[int],
) -> list[tuple[int, int]]:
    """One extra prefill profile that compiles ``prepare_dflash_inputs``'
    capped tile bucket.

    ``_prepare_dflash_inputs_kernel`` specializes on
    ``BLOCK_SIZE = min(cap, next_power_of_2(query_len + num_query_per_req))``;
    every longer prefill reuses the capped variant. The default and advertised
    warmup prompt lengths stay below the cap, so the first long real prefill
    used to JIT-compile the capped tile during serving.
    """
    speculator = getattr(model_runner, "speculator", None)
    if speculator is None:
        return []
    from vllm.v1.worker.gpu.spec_decode.dflash.speculator import (
        PREPARE_DFLASH_INPUTS_TILE_CAP,
        DFlashSpeculator,
    )

    if not isinstance(speculator, DFlashSpeculator):
        return []
    max_tokens = min(
        model_runner.scheduler_config.max_num_batched_tokens,
        model_runner.max_model_len,
    )
    # Shortest per-request query length whose draft span reaches the cap.
    tile_prompt_len = (
        PREPARE_DFLASH_INPUTS_TILE_CAP // 2 + 1 - speculator.num_query_per_req
    )
    if tile_prompt_len > max_tokens or max(profiled_lengths) >= tile_prompt_len:
        return []
    return [(tile_prompt_len, 1)]


def _greedy_verify_warmup_enabled(model_runner: GPUModelRunner) -> bool:
    """Whether the SM70 greedy MTP verification fast path can run.

    It needs a draft-carrying decode step whose every request samples greedily
    without logprobs, penalties, bias or bad words, which the sampler-warmup
    parameters never satisfy; one greedy profile compiles the kernel instead.
    """
    if model_runner.num_speculative_steps <= 0 or model_runner.is_pooling_model:
        return False
    kernel_config = getattr(
        getattr(model_runner, "vllm_config", None), "kernel_config", None
    )
    return bool(
        getattr(kernel_config, "sm70_greedy_verify", False)
        or envs.VLLM_SM70_GREEDY_VERIFY
    )


def _host_kv_group_id(model_runner: GPUModelRunner) -> int:
    """KV cache group whose layers the host history store backs."""
    host_layers = set(model_runner.kv_cache_config.host_kv_layers)
    for group_id, group in enumerate(model_runner.kv_cache_config.kv_cache_groups):
        if host_layers & set(group.layer_names):
            return group_id
    raise AssertionError("host history store without a host KV cache group")


def _host_kv_warmup_step(
    store: "HostHistoryStore",
    step: int,
    batch: Sequence[tuple[str, int, int]],
    *,
    claim: Sequence[int] = (),
    donate: Sequence[int] = (),
    bindings: Sequence[int] = (),
) -> "HostKVStep":
    """One warmup step's scheduler-side host KV metadata: per-request
    (rows, G_q, class) planner input at the length after this step, plus
    this step's shared hot region roles and page bindings.

    Mirrors the scheduler's ``_host_kv_batch`` step metadata for the
    hand-built warmup batches; block IDs stand in for their own host pages.
    """
    from vllm.models.qwen4_exp.nvidia.host_kv.planning import history_groups
    from vllm.v1.kv_offload.host.page_table import HostKVStep

    s = store.shape
    requests = {
        req_id: (
            rows,
            history_groups(length, s.dcp_world_size, s.interleave),
            0 if rows <= s.decode_rows else 1,
        )
        for req_id, rows, length in batch
    }
    pages = store.host.shape[0]
    return HostKVStep(
        step,
        [(hid, hid) for hid in bindings if 0 < hid < pages],
        [],
        donate=list(donate),
        claim=list(claim),
        requests=requests,
    )


def _reserved_block_count(
    num_tokens: int,
    kv_cache_spec: KVCacheSpec,
    *,
    num_lookahead_tokens: int,
    max_model_len: int,
    max_encoder_len: int,
) -> int:
    """Match the scheduler's block reservation in hand-built warmup batches."""
    if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
        specs = tuple(kv_cache_spec.kv_cache_specs.values())
        assert specs
        kv_cache_spec = specs[0]
    if isinstance(kv_cache_spec, CircularBufferSpec):
        return 1
    if isinstance(kv_cache_spec, CrossAttentionSpec):
        return cdiv(max_encoder_len, kv_cache_spec.block_size)
    num_speculative_blocks = 0
    if isinstance(kv_cache_spec, MambaSpec):
        num_speculative_blocks = kv_cache_spec.num_speculative_blocks
        if kv_cache_spec.mamba_cache_mode == "align":
            return cdiv(num_tokens, kv_cache_spec.block_size) + num_speculative_blocks
    num_tokens = min(num_tokens + num_lookahead_tokens, max_model_len)
    return cdiv(num_tokens, kv_cache_spec.block_size) + num_speculative_blocks


def _warmup_block_counter(
    model_runner: GPUModelRunner,
) -> Callable[[int, KVCacheSpec], int]:
    def block_count(num_tokens: int, kv_cache_spec: KVCacheSpec) -> int:
        return _reserved_block_count(
            num_tokens,
            kv_cache_spec,
            num_lookahead_tokens=model_runner.vllm_config.num_lookahead_tokens,
            max_model_len=model_runner.max_model_len,
            max_encoder_len=getattr(model_runner.model_state, "max_encoder_len", 0),
        )

    return block_count


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    2 + num_spec_steps prompt tokens each. The second iteration simulates
    a decode step with all requests generating 1 + num_spec_steps tokens.

    In long-context host KV mode every iteration also carries scheduler-side
    host KV metadata (roles, page bindings, planner input) like the real
    scheduler's steps, so the step-head bookkeeping, the union windows and
    the staged write kernels are compiled before serving.

    With a DFlash speculative config, one extra single-request prefill
    compiles the draft-input kernel's capped tile bucket, and (with the SM70
    greedy verification switch on) one extra all-greedy prefill+draft-decode
    pair compiles the greedy MTP verification kernel, which no
    sampler-warmup batch can reach.
    """
    num_spec_steps = model_runner.num_speculative_steps
    # Use 1 + num_spec_steps + 1 tokens so the prefill batch's per-request
    # query length exceeds decode_query_len (= 1 + num_spec_steps), preventing
    # it from being misclassified as a uniform decode batch.
    default_prompt_len = 2 + num_spec_steps
    prompt_lengths = _kernel_prefill_warmup_token_counts(
        model_runner, default_prompt_len
    )

    host_store = getattr(model_runner, "host_history", None)
    host_group = _host_kv_group_id(model_runner) if host_store is not None else -1
    # The static hot pool carries no shared-region roles (see
    # HostHistoryStore.apply_roles).
    shared_hot_region = host_store is not None and host_store.shape.slots_per_block > 0
    profiles: list[tuple[int, int]] = [(length, 1) for length in prompt_lengths]
    if host_store is not None:
        profiles += _host_kv_warmup_profiles(model_runner, prompt_lengths)
    profiles += _dflash_warmup_profiles(
        model_runner, [length for length, _ in profiles] + [1 + num_spec_steps]
    )
    # Draft-carrying decode steps only JIT-compile the SM70 greedy MTP
    # verification kernel when every request samples greedily; run the
    # default profile once more with plain greedy parameters for that.
    greedy_profile_idx = -1
    if _greedy_verify_warmup_enabled(model_runner):
        greedy_profile_idx = len(profiles)
        profiles.append((default_prompt_len, 1))

    # Use a serving-shaped prefill before capture locks the workspace. The
    # unpadded batch still goes through the regular runner admission check.
    hc_sp_profile_idx = -1
    hc_sp_min_tokens = getattr(model_runner, "hc_sp_min_tokens", None)
    if hc_sp_min_tokens is not None:
        max_tokens = model_runner.scheduler_config.max_num_batched_tokens
        prompt_len = min(max_tokens, model_runner.max_model_len - 1 - num_spec_steps)
        if prompt_len > 0:
            profile_reqs = min(
                max_tokens // prompt_len, model_runner.scheduler_config.max_num_seqs
            )
            if prompt_len * profile_reqs >= hc_sp_min_tokens:
                hc_sp_profile_idx = len(profiles)
                profiles.append((prompt_len, profile_reqs))

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    block_count = _warmup_block_counter(model_runner)
    kv_cache_specs = [group.kv_cache_spec for group in kv_cache_groups]
    # Block IDs are numbered per block space: one space normally, a state and
    # a history space in long-context mode (KVCacheConfig.block_segments).
    segments = (
        model_runner.kv_cache_config.block_segments
        if has_block_segments(model_runner.kv_cache_config)
        else None
    )
    group_spaces = (
        list(segments.group_segments)
        if segments is not None
        else [0] * num_kv_cache_groups
    )
    space_blocks = (
        list(segments.num_blocks)
        if segments is not None
        else [model_runner.kv_cache_config.num_blocks]
    )

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None
    # Plain greedy parameters: the SM70 greedy verification fast path treats
    # any sampler feature (temperature, logprobs, penalties, bias) as a miss.
    greedy_sampling_params = SamplingParams(temperature=0.0)

    # Disable KV connector for all warmup runs. Kernel-advertised profiles run
    # with one request so adding an operator does not silently multiply startup
    # work by max_num_seqs.
    model_runner.kv_connector.set_disabled(True)
    try:
        for profile_idx, (prompt_len, profile_reqs) in enumerate(profiles):
            prompt_token_ids = list(range(prompt_len))
            decode_len = prompt_len + 1 + num_spec_steps
            prefill_block_counts = [
                block_count(prompt_len, spec) for spec in kv_cache_specs
            ]
            decode_block_counts = [
                block_count(decode_len, spec) for spec in kv_cache_specs
            ]
            decode_block_deltas = [
                d - p for d, p in zip(decode_block_counts, prefill_block_counts)
            ]
            space_blocks_per_req = [0] * len(space_blocks)
            for space, count in zip(group_spaces, decode_block_counts):
                space_blocks_per_req[space] += count
            num_reqs = min(
                model_runner.scheduler_config.max_num_seqs,
                model_runner.scheduler_config.max_num_batched_tokens
                // max(prompt_len, 1 + num_spec_steps),
                # Reserve block 0 (null block) and ensure enough blocks.
                *(
                    max(1, (blocks - 1) // per_req)
                    for blocks, per_req in zip(space_blocks, space_blocks_per_req)
                    if per_req
                ),
            )
            if profile_idx:
                num_reqs = min(num_reqs, profile_reqs)
            if profile_idx == hc_sp_profile_idx:
                assert hc_sp_min_tokens is not None
                if prompt_len * num_reqs < hc_sp_min_tokens or any(
                    num_reqs * per_req >= blocks
                    for blocks, per_req in zip(space_blocks, space_blocks_per_req)
                ):
                    continue
            if num_reqs <= 0:
                continue

            req_ids = [f"_warmup_{profile_idx}_{i}_" for i in range(num_reqs)]
            next_block_ids = [1] * len(space_blocks)
            profile_sampling_params = (
                greedy_sampling_params
                if profile_idx == greedy_profile_idx
                else sampling_params
            )

            def _alloc_blocks(
                num_blocks: int, group_id: int, next_ids: list[int] = next_block_ids
            ) -> list[int]:
                space = group_spaces[group_id]
                first = next_ids[space]
                next_ids[space] = first + num_blocks
                return list(range(first, first + num_blocks))

            new_reqs = [
                NewRequestData.from_request(
                    Request(
                        req_ids[i],
                        prompt_token_ids,
                        profile_sampling_params,
                        pooling_params,
                    ),
                    block_ids=tuple(
                        _alloc_blocks(n, g) for g, n in enumerate(prefill_block_counts)
                    ),
                    prefill_token_ids=prompt_token_ids,
                )
                for i in range(num_reqs)
            ]

            prefill_output = SchedulerOutput.make_empty()
            prefill_output.num_spec_tokens_to_schedule = num_spec_steps
            prefill_output.scheduled_new_reqs = new_reqs
            prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
            prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
            prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups
            prefill_claim = (
                [block_id for req in new_reqs for block_id in req.block_ids[host_group]]
                if host_store is not None
                else []
            )
            decode_claim: list[int] = []
            if host_store is not None:
                # New history IDs: claim their shared-region slots, bind a
                # host page and zero/begin the blocks like the scheduler's
                # first step for these requests.
                prefill_output.host_kv = _host_kv_warmup_step(
                    host_store,
                    profile_idx,
                    [(rid, prompt_len, prompt_len) for rid in req_ids],
                    claim=prefill_claim if shared_hot_region else [],
                    bindings=prefill_claim,
                )
                prefill_output.new_block_ids_to_zero = list(prefill_claim)
            worker_execute_model(prefill_output)

            if not model_runner.is_pooling_model:
                grammar_output = None
                if profile_idx == 0 and model_runner.is_last_pp_rank:
                    # Exercise the structured-output bitmask once; extra
                    # operator profiles only need the model path.
                    vocab_size = model_runner.model_config.get_vocab_size()
                    bitmask_width = (vocab_size + 31) // 32
                    grammar_bitmask = np.full(
                        (len(req_ids), bitmask_width),
                        fill_value=-1,
                        dtype=np.int32,
                    )
                    grammar_output = GrammarOutput(
                        structured_output_request_ids=req_ids,
                        grammar_bitmask=grammar_bitmask,
                    )
                worker_sample_tokens(grammar_output)

                cached_req_data = CachedRequestData.make_empty()
                cached_req_data.req_ids = list(req_ids)
                cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
                cached_req_data.num_output_tokens = [1] * num_reqs
                new_block = any(decode_block_deltas)
                cached_req_data.new_block_ids = [
                    (
                        tuple(
                            _alloc_blocks(n, g)
                            for g, n in enumerate(decode_block_deltas)
                        )
                        if new_block
                        else None
                    )
                    for _ in range(num_reqs)
                ]

                decode_output = SchedulerOutput.make_empty()
                decode_output.num_spec_tokens_to_schedule = num_spec_steps
                decode_output.scheduled_cached_reqs = cached_req_data
                decode_output.num_scheduled_tokens = {
                    req_id: 1 + num_spec_steps for req_id in req_ids
                }
                if num_spec_steps > 0:
                    decode_output.scheduled_spec_decode_tokens = {
                        req_id: [0] * num_spec_steps for req_id in req_ids
                    }
                decode_output.total_num_scheduled_tokens = sum(
                    decode_output.num_scheduled_tokens.values()
                )
                decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups
                if host_store is not None:
                    decode_claim = [
                        block_id
                        for new_ids in cached_req_data.new_block_ids or []
                        for block_id in (new_ids[host_group] if new_ids else ())
                    ]
                    decode_output.host_kv = _host_kv_warmup_step(
                        host_store,
                        profile_idx,
                        [
                            (rid, 1 + num_spec_steps, prompt_len + 1 + num_spec_steps)
                            for rid in req_ids
                        ],
                        claim=decode_claim if shared_hot_region else [],
                        bindings=decode_claim,
                    )
                    if decode_claim:
                        decode_output.new_block_ids_to_zero = list(decode_claim)

                worker_execute_model(decode_output)
                worker_sample_tokens(None)

            cleanup_output = SchedulerOutput.make_empty()
            cleanup_output.finished_req_ids = set(req_ids)
            if host_store is not None and shared_hot_region:
                # Request finish donates the IDs' slots back to the pool.
                cleanup_output.host_kv = _host_kv_warmup_step(
                    host_store,
                    profile_idx,
                    [],
                    donate=prefill_claim + decode_claim,
                )
            worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    if host_store is not None:
        hot_metrics = host_store.hot_metrics
        if hot_metrics is not None:
            # The serving connector's periodic hot-pool snapshot is otherwise
            # first JIT-compiled during serving; reset drops its warmup report.
            hot_metrics.harvest(0, host_store.slot_owner[0])
            hot_metrics.reset()
        # Warmup ran serving-shaped host steps; the first real step must
        # still perform the serving-start reset of the hot region.
        model_runner._host_history_live = False
    torch.accelerator.synchronize()
