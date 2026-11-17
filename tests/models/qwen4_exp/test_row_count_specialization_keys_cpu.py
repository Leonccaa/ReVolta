# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the ReVolta project
"""A row-count change must not produce a new Triton specialization key.

Cold compiles inside inference come from the specialization key, not from the
kernel body: an unspecialized scalar keeps the same key for every batch size,
while a ``tl.constexpr`` or an automatically specialized int gets a new entry
(the classes are ``== 1``, ``divisible by 16`` and ``other``) and therefore a
new compile. Triton builds that key before it touches a device, so this gate
computes it directly for the candidate and for the same kernel at the branch
point of this change over a sweep of row counts that spans all three classes.

Requirement: the candidate yields one key for every count; the baseline yields
more than one, which is what this change removes.
"""

import importlib
import inspect
import os
import subprocess
from pathlib import Path

import pytest
import torch

if os.environ.get("TRITON_INTERPRET") == "1":
    pytest.skip("needs the compiling JIT path", allow_module_level=True)

from vllm.triton_utils import triton  # noqa: E402

pytestmark = pytest.mark.skip_global_cleanup

ROOT = Path(__file__).resolve().parents[2]
# Branch point of this change: the "before" side of the comparison, so the gate
# keeps measuring the removal instead of comparing a commit with itself.
BASELINE = "4859f64df5"
# Triton builds the specialization key before it touches a device: the binder
# only asks the backend for the divisibility attribute, so no GPU is needed.
# The submodules are resolved through importlib because repository code imports
# Triton via vllm.triton_utils.
jit = importlib.import_module("triton.runtime.jit")
nvidia = importlib.import_module("triton.backends.nvidia.compiler")
BACKEND = nvidia.CUDABackend.__new__(nvidia.CUDABackend)
MockTensor = jit.MockTensor
assert triton is not None

# Row counts that span the three specialization classes: equal to one,
# divisible by 16 and neither.
COUNTS = (1, 2, 3, 15, 16, 17, 33)


def _module(rel):
    return importlib.import_module(rel.removesuffix(".py").replace("/", "."))


def head_kernel(rel, name):
    """The same kernel as of HEAD: the behaviour before this change."""
    source = subprocess.check_output(
        ["git", "show", f"{BASELINE}:{rel}"], cwd=ROOT, text=True
    )
    namespace = dict(vars(_module(rel)))
    exec(compile(source, str(ROOT / rel), "exec"), namespace)
    return namespace[name]


def keys(kernel, build):
    binder = jit.create_function_from_signature(
        inspect.signature(kernel.fn), kernel.params, BACKEND
    )
    found = set()
    for count in COUNTS:
        _, specialization, _ = binder(*build(count)[0], **build(count)[1])
        found.add(repr(specialization))
    return found


QSA = "vllm/models/qwen4_exp/nvidia/ops/qsa.py"
SEGMENTS = "vllm/models/qwen4_exp/nvidia/ops/qsa_request_segments.py"
NGRAM = "vllm/models/qwen4_exp/nvidia/sm70_ngram_input.py"
MARKS = "vllm/models/qwen4_exp/nvidia/host_kv/history_kernels.py"
FLASH = "vllm/v1/attention/backends/flash_attn_v100.py"
GDN = "vllm/v1/attention/backends/gdn_attn.py"
LOGPROB = "vllm/v1/worker/gpu/sample/logprob.py"
REJECT = "vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py"
GREEDY = "vllm/v1/worker/gpu/spec_decode/sm70_greedy_verify.py"
DFLASH = "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py"

I32 = MockTensor(torch.int32)
I64 = MockTensor(torch.int64)
F32 = MockTensor(torch.float32)
F16 = MockTensor(torch.float16)
U64 = MockTensor(torch.uint64)


def _pointers(count):
    # MockTensor always reports a 16-byte aligned pointer, so pointer
    # attributes stay constant across the sweep by construction.
    return I32


CASES = {
    (QSA, "_qsa_visible_blocks_kernel"): lambda n: (
        (I32, I64, I32, I32, n, 16),
        {"COMPRESS_RATIO": 4},
    ),
    (QSA, "_qsa_gather_single_request_keys_kernel"): lambda n: (
        (F16, I32, F16, I32, 128, 128, 1, 128, 128, n, 32),
        {
            "PAGE_SIZE": 16,
            "PAGE_TABLE_WIDTH": 8,
            "HEAD_DIM": 128,
            "BLOCK_D": 128,
        },
    ),
    (QSA, "_qsa_relu_headsum_visible_kernel"): lambda n: (
        (F32, I32, I32, F32, n, 1, 2051, n, 256, 11.31),
        {"NUM_HEADS": 4, "BLOCK_N": 256},
    ),
    (QSA, "_expand_qsa_indices_kernel"): lambda n: (
        (I32, I64, I32, I32, I32, 512, 1, 2054, 1, n, 64),
        {
            "BLOCK_TOPK": 512,
            "COMPRESS_RATIO": 4,
            "TOKEN_TOPK": 2048,
            "OUTPUT_WIDTH": 2051,
            "COLUMN_BLOCK": 256,
        },
    ),
    (QSA, "_qsa_merge_splitk_kernel"): lambda n: (
        (F32, F32, F16, F32, None, 4096, 128, 0, 0, n, 1.0),
        {
            "HEAD_DIM": 128,
            "NUM_QUERY_HEADS": 32,
            "NUM_SPLITS": 8,
            "BLOCK_SPLITS": 8,
            "KV_E4M3": False,
        },
    ),
    (QSA, "_index_shard_pack_or_map_kernel"): lambda n: (
        (I32, I32, I32, 512, 1, n, 1, 512, 512, 2, 0, True, 128),
        {},
    ),
    (
        SEGMENTS,
        "_build_segments",
    ): lambda n: (
        (I32, I64, I32, I32, I32, 1, 1, 1, n, 64, 4),
        {"MAX_ROWS": 4, "BLOCK_M": 64},
    ),
    (NGRAM, "_prepare_ngram"): lambda n: (
        (I32, I32, I32, I32, I32, I32, n, max(n, 4), 8, 64, 151645, 8),
        {},
    ),
    (MARKS, "mark_decode"): lambda n: (
        (
            I32,
            I32,
            I32,
            I32,
            I32,
            I32,
            I32,
            I32,
            2051,
            416,
            2051,
            32,
            1024,
            128,
            1,
            0,
            1,
            512,
            n,
        ),
        {},
    ),
    (MARKS, "mark_tail"): lambda n: (
        (I32, I32, I32, I32, I32, I32, 416, 32, 1024, 1, 0, 1),
        {"ROW_OFFSET": n},
    ),
    (FLASH, "_sm70_prepare_grouped_smallq_decode_metadata_kernel"): lambda n: (
        (U64, U64, U64, U64, I64, I64, I32, I32, I32, n, max(n - 1, 1)),
        {"REQ_BLOCK": 64, "BLOCK_COLS": 8},
    ),
    (GDN, "_dflash2_gdn_group_metadata_kernel"): lambda n: (
        (
            U64,
            U64,
            I64,
            I32,
            I32,
            I32,
            I32,
            I32,
            I32,
            MockTensor(torch.bool),
            I32,
            I32,
            I32,
            n,
            max(n, 8),
        ),
        {
            "WIDTH": 5,
            "PAD_ID": -1,
            "BLOCK": 256,
            "USE_STATE_START": False,
            "USE_SEQ_LEN_START": False,
            "MAMBA_BLOCK_SIZE": 128,
            "STATE_COLUMNS": 5,
        },
    ),
    (LOGPROB, "_topk_log_softmax_kernel"): lambda n: (
        (F32, F16, 151936, I64, n, 151936),
        {"BLOCK_SIZE": 1024, "PADDED_TOPK": 32},
    ),
    (QSA, "_qsa_mqa_paged_kernel"): lambda n: (
        (
            F16,  # q_ptr
            F16,  # k_cache_ptr
            I32,  # page_table_ptr
            I32,  # token_to_req_ptr
            I64,  # query_positions_ptr
            I32,  # sequence_lengths_ptr
            I32,  # visible_blocks_ptr
            F32,  # logits_ptr
            512,  # stride_q_row
            128,  # stride_q_head
            1,  # stride_q_dim
            4096,  # stride_cache_block
            256,  # stride_cache_token
            1,  # stride_cache_dim
            416,  # stride_table_req
            1,  # stride_table_page
            2051,  # stride_logits_row
            n,  # num_rows
            2051,  # num_columns
            2048,  # num_pages
            n,  # num_requests
            11.31,  # score_divisor
        ),
        {
            "PAGE_SIZE": 16,
            "PAGE_TABLE_WIDTH": 416,
            "NUM_HEADS": 4,
            "HEAD_DIM": 128,
            "BLOCK_N": 64,
            "BLOCK_D": 128,
            "TILES_PER_PROG": 1,
            "STAGES": 2,
            "MAX_N": 16,
            "COMPRESS_RATIO": 4,
        },
    ),
    (QSA, "_qsa_sparse_paged_gqa_splitk_kernel"): lambda n: (
        (
            F16,  # q_ptr
            F16,  # k_cache_ptr
            F16,  # v_cache_ptr
            I32,  # indices_ptr
            I32,  # block_table_ptr
            I32,  # token_to_req_ptr
            F32,  # partial_output_ptr
            F32,  # partial_lse_ptr
            F16,  # output_ptr
            F32,  # final_lse_ptr
            None,  # output_gate_ptr
            4096,  # stride_q_row
            128,  # stride_q_head
            4096,  # stride_k_block
            128,  # stride_k_token
            128,  # stride_k_head
            4096,  # stride_v_block
            128,  # stride_v_token
            128,  # stride_v_head
            2051,  # stride_indices_row
            416,  # stride_table_req
            4096,  # stride_output_row
            128,  # stride_output_head
            0,  # stride_output_gate_row
            0,  # stride_output_gate_head
            n,  # num_rows
            2048,  # num_cache_blocks
            n,  # num_requests
            1.0,  # k_scale
            1.0,  # v_scale
        ),
        {
            "TOPK": 2051,
            "PAGE_SIZE": 16,
            "PAGE_TABLE_WIDTH": 416,
            "GROUP_SIZE": 8,
            "HEAD_DIM": 128,
            "NUM_QUERY_HEADS": 32,
            "NUM_SPLITS": 8,
            "NUM_TILES": 33,
            "BLOCK_M": 8,
            "BLOCK_N": 64,
            "KV_E4M3": False,
        },
    ),
    (REJECT, "_rejection_kernel"): lambda n: (
        (
            I64,  # sampled_ptr
            5,  # sampled_stride
            I32,  # rejected_steps_ptr
            F32,  # rejected_logsumexp_ptr
            n * 2,  # rejected_logsumexp_plane_stride
            F32,  # target_logits_ptr
            151936,  # target_logits_stride
            I64,  # target_local_argmax_ptr
            19,  # target_local_argmax_stride
            F32,  # block_stats_ptr
            n * 19,  # block_stats_plane_stride
            19,  # block_stats_row_stride
            I64,  # draft_sampled_ptr
            F32,  # draft_logits_ptr
            0,  # draft_logits_stride_0
            0,  # draft_logits_stride_1
            I32,  # cu_num_logits_ptr
            I32,  # idx_mapping_ptr
            F32,  # temp_ptr
            I64,  # seed_ptr
            I32,  # pos_ptr
            None,  # synthetic_conditional_rates_ptr
            19,  # vocab_num_blocks
        ),
        {
            "PADDED_VOCAB_NUM_BLOCKS": 32,
            "HAS_DRAFT_LOGITS": False,
            "SYNTHETIC_MODE": False,
        },
    ),
}


@pytest.mark.parametrize("target", sorted(CASES))
def test_row_count_does_not_change_the_specialization_key(target):
    rel, name = target
    build = CASES[target]
    candidate = getattr(_module(rel), name)
    found = keys(candidate, build)
    assert len(found) == 1, f"{name} still recompiles per row count: {found}"


@pytest.mark.parametrize("target", sorted(CASES))
def test_the_baseline_kernel_did_specialize_on_the_row_count(target):
    rel, name = target
    build = CASES[target]
    before = head_kernel(rel, name)
    after = getattr(_module(rel), name)
    before_keys, after_keys = keys(before, build), keys(after, build)
    assert len(after_keys) == 1
    assert len(before_keys) >= len(after_keys), (
        f"{name}: {BASELINE} had {len(before_keys)} variant(s), candidate "
        f"{len(after_keys)}; the gate no longer measures a removal"
    )
    assert len(before_keys) > 1, (
        f"{name}: {BASELINE} already produced one key, so this kernel is no "
        "longer an example of the removal and should leave this gate"
    )


# ``_greedy_verify_kernel`` and ``_prepare_dflash_inputs_kernel`` also showed up
# as first-step cold compiles after a production restart. Neither takes a row or
# length argument: the batch size only selects the grid, and every integer
# argument is a per-configuration constant (speculative step count, block-table
# stride, cache geometry, vocabulary ids). There is no shape-driven variant to
# remove here, so the remaining cold compile is a warmup path gap, and these
# kernels belong to the warmup lane instead. The gate pins that reading: if a
# batch-derived scalar is ever added, the parameter list changes and this test
# fails.
SHAPE_INDEPENDENT = {
    (
        "vllm/v1/worker/gpu/spec_decode/sm70_greedy_verify.py",
        "_greedy_verify_kernel",
    ): [
        "sampled_ptr",
        "sampled_stride",
        "num_sampled_ptr",
        "target_ptr",
        "draft_ptr",
        "cu_num_logits_ptr",
    ],
    (
        "vllm/v1/worker/gpu/spec_decode/dflash/speculator.py",
        "_prepare_dflash_inputs_kernel",
    ): [
        "out_input_ids_ptr",
        "out_query_positions_ptr",
        "out_query_start_loc_ptr",
        "out_seq_lens_ptr",
        "out_query_slot_mapping_ptr",
        "out_context_positions_ptr",
        "out_context_slot_mapping_ptr",
        "out_sample_indices_ptr",
        "out_sample_pos_ptr",
        "out_sample_idx_mapping_ptr",
        "out_temperature_ptr",
        "out_seeds_ptr",
        "target_positions_ptr",
        "target_query_start_loc_ptr",
        "idx_mapping_ptr",
        "last_sampled_ptr",
        "next_prefill_tokens_ptr",
        "num_sampled_ptr",
        "num_rejected_ptr",
        "temperature_ptr",
        "seeds_ptr",
        "block_table_ptr",
        "block_table_stride",
        "parallel_drafting_token_id",
        "block_size",
        "num_query_per_req",
        "num_speculative_steps",
        "max_num_reqs",
        "max_num_tokens",
        "max_model_len",
        "cp_rank",
        "SAMPLE_FROM_ANCHOR",
        "PAD_SLOT_ID",
        "CP_SIZE",
        "CP_INTERLEAVE",
        "BLOCK_SIZE",
    ],
}


@pytest.mark.parametrize("target", sorted(SHAPE_INDEPENDENT))
def test_kernels_without_a_batch_shape_argument_stay_that_way(target):
    rel, name = target
    kernel = getattr(_module(rel), name)
    names = [p.name for p in kernel.params]
    assert names == SHAPE_INDEPENDENT[target]
    # Every non-pointer argument above is a per-configuration constant, so no
    # runtime specialization key can depend on the batch. A row/length count
    # added here would need do_not_specialize and this pinned list to change.
    for param in kernel.params:
        if param.name.endswith("_ptr") or param.is_constexpr:
            continue
        assert not param.name.startswith(("rows", "batch_", "len_")), param.name
