# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-maintained notes for the generated configuration reference.

tools/gen_env_docs.py renders docs/configuration.md#environment-variable-reference
from vllm/envs.py metadata (type, default, category, description) plus these
notes. Each entry: section, since (first release with the variable; "1Cat"
for variables inherited from 1Cat-vLLM), range, tradeoff (what the setting
trades, when to change it) and metrics (related /metrics series, or "").
Add a variable here to document it, then run `python tools/gen_env_docs.py`.
"""

SECTIONS = {
    "promoted": "Promoted research switches (0.2.0rc5)",
    "deployment": "Settings used by the reference deployment",
    "profile": "Values applied by the ReVolta profile",
}

PROFILE_NOTE = (
    "The ReVolta profile (`vllm/revolta_profile.py`) sets these on SM70 + TP4 + "
    "FP16 + QSA models when they are unset; explicit values win. The profile "
    "column shows the value it applies (AWQ-only and NVFP4-only entries apply to "
    "that checkpoint family only)."
)

_BITWISE = "Outputs are bitwise identical either way."

NOTES: dict[str, dict[str, str]] = {
    # ---- reference deployment ---------------------------------------------
    "VLLM_QWEN4EXP_PLE_HOST_GIB": dict(
        section="deployment",
        since="1Cat",
        range="GiB per TP rank; unset = automatic",
        tradeoff="Caps pinned host memory for the PLE table. Lower values leave "
        "host RAM for CPU KV offload but move more PLE rows to disk; the "
        "reference deployment uses 12.",
        metrics="",
    ),
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT": dict(
        section="deployment",
        since="1Cat",
        range="seconds",
        tradeoff="How long startup waits for the PLE offload worker. Raise it on "
        "slow storage; the reference deployment uses 1200.",
        metrics="",
    ),
    "VLLM_SM70_AWQ_MOE_INT4_ZERO": dict(
        section="deployment",
        since="0.2.0",
        range="0, 1",
        tradeoff="Packs AWQ zero points into nibbles: less resident metadata, "
        "lossless. Needs compact metadata and a native build with the packed "
        "kernels.",
        metrics="",
    ),
    "VLLM_SM70_DFLASH2_FUSED_QKV_PACK": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff="One Triton launch instead of three copies and a concat in the "
        "DFlash2 verifier. " + _BITWISE,
        metrics="",
    ),
    "VLLM_SM70_FLASH_ATTN_V100": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff="Default on (set in the image). 0 only to diagnose backend "
        "selection; prefer --attention-backend to choose a backend.",
        metrics="",
    ),
    "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff="Set in the image: VLLM_COMPILE + FULL_AND_PIECEWISE with small "
        "decode captures (the 0.0.3 graph policy).",
        metrics="",
    ),
    "VLLM_SM70_QWEN_GDN_FULL_FORWARD": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff="Diagnostic force-on of the opaque GDN full-forward boundary. "
        "The image sets 0; the automatic default arms it only for active MTP "
        "batches.",
        metrics="",
    ),
    "VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff="Diagnostic: compare the split GDN compile boundary with the "
        "full-forward guard. The image sets 0.",
        metrics="",
    ),
    "VLLM_SM70_QWEN_GDN_SPEC_CORE_OP": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff="Experimental narrower MTP boundary (only the recurrent core is "
        "opaque). The image sets 0 until it passes long-output quality.",
        metrics="",
    ),
    # ---- ReVolta profile ---------------------------------------------------
    "VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Opts out of the 1Cat SM70 MTP launch defaults; the ReVolta "
        "profile supplies its own.",
        metrics="",
    ),
    "VLLM_EARLY_MODEL_INPUTS": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Prepares model-state inputs before attention metadata in FULL "
        "graph steps, overlapping host work.",
        metrics="",
    ),
    "VLLM_PLE_ASYNC_REQUEST_INDICES": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Non-blocking PLE request-index transfers; removes a host wait "
        "per step.",
        metrics="",
    ),
    "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Shared-expert stream overlap. The profile sets 0 on TP4 SM70, "
        "where the second stream does not pay off.",
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Grouped AWQ MoE decode reads each active expert once for all "
        "its rows (2..48). Faster at 2..8 requests; reduction order differs "
        "from the TurboMind route.",
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS": dict(
        section="profile",
        since="0.2.0",
        range="2..48",
        tradeoff="Largest row count sent to the grouped route; above it rows use "
        "TurboMind. The profile uses 40 (8 requests x 5 tokens).",
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_PACK_ROWS": dict(
        section="profile",
        since="0.2.0",
        range="0 or 8..48 step 8",
        tradeoff="Caps rows per expert pack: fewer registers per CTA but hot "
        "experts are read more than once. Result unchanged.",
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT": dict(
        section="profile",
        since="0.2.0",
        range="4, 8, 16",
        tradeoff="W13 split-K warps per CTA. 16 matches the M1 arithmetic; 8 "
        "(profile) uses fewer threads and a different FP32 split association.",
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_QPN_M1": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Single-token W13/W2 QPN route; 0 restores the legacy route. "
        "Changes reduction order.",
        metrics="",
    ),
    "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Tunes TurboMind small-M AWQ shapes at startup: longer warmup, "
        "faster small batches.",
        metrics="",
    ),
    "VLLM_SM70_CAR_VMM_SAFE": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Copies non-exportable (VMM) inputs into graph allocations while "
        "capturing custom all-reduce graphs, so expandable segments can stay "
        "on. Only with P2P or NVLink.",
        metrics="",
    ),
    "VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Single-request DFlash2 verifier tail graphs. The AWQ DFlash "
        "profile sets 0 to keep the graph memory for the KV cache.",
        metrics="",
    ),
    "VLLM_SM70_GREEDY_VERIFY": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Greedy MTP verification from TP-local argmax pairs instead of "
        "full-vocabulary logits; only when every request is plain greedy.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Grouped native-NVFP4 decode; the profile keeps it off pending "
        "quality admission.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="One grouped TurboMind call for all routed experts in prefill.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="QPN for every live width 2..16 (NVFP4).",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Graph-safe raw-E4M3 grouped NVFP4 MoE for M = 2..40; needs "
        "RAW_SCALE storage.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Keeps E4M3 block-scale codes resident instead of a second FP16 "
        "copy: less memory, decode reconstructs exact values.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Tunes small NVFP4 shapes at startup.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_TURBOMIND": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="TurboMind route for NVFP4/MXFP4 dense layers; 0 only for diagnosis.",
        metrics="",
    ),
    "VLLM_SM70_PLE_INPUT_PREPARE": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="One fused launch for PLE n-gram context gathering (<=4 requests).",
        metrics="",
    ),
    "VLLM_SM70_PLE_NGRAM_BATCH": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Fused PLE n-gram id preprocessing for 2..32 rows.",
        metrics="",
    ),
    "VLLM_SM70_PLE_PREFILL_CONV_CHUNK": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Bounds PLE short-conv prefill buffers when several prefills "
        "share a step; needed at max_num_seqs=8. " + _BITWISE,
        metrics="",
    ),
    "VLLM_SM70_QSA_COMPACT_TOPK": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Coarse FP16 pre-selection before the exact QSA top-k for <=32 "
        "rows and long contexts; same indices.",
        metrics="",
    ),
    "VLLM_SM70_QSA_REQUEST_SEGMENTS": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Reads each 64-key tile once per request segment in "
        "multi-request decode. " + _BITWISE,
        metrics="",
    ),
    "VLLM_SM70_QSA_SHARED_KEY": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Shared-key FP16 indexer for M2..8 verification queries of one "
        "request.",
        metrics="",
    ),
    "VLLM_SM70_QWEN38_GDN_NO_PACK": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Skips the packed GDN weight copy (725.6 MiB per TP4 rank) and "
        "serves 2..48 rows with the skinny dual GEMM; frees memory for KV.",
        metrics="",
    ),
    "VLLM_SM70_QWEN38_HYBRID_PLE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Local pinned PLE shards for decode. The profile sets 0 "
        "(single PLE placement) on the reference deployment.",
        metrics="",
    ),
    "VLLM_SM70_QWEN38_WIDE_M": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Widens Qwen3.8 decode/verify routes from 4 to 8 requests (M up "
        "to 40); each VLLM_SM70_QWEN38_WIDE_M_* item can override it.",
        metrics="",
    ),
    "VLLM_SM70_SKIP_REJECTED_BATCH_PACKS": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Skips batch packs the precision policy never admits; saves "
        "about 199 MiB per rank.",
        metrics="",
    ),
    "VLLM_USE_AOT_COMPILE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="AOT compile reuse. The profile sets 0 on the reference deployment.",
        metrics="",
    ),
    "VLLM_V2_CUDAGRAPH_MEM_MIB": dict(
        section="profile",
        since="1Cat",
        range="MiB",
        tradeoff="CUDA graph memory reserved before sizing the KV cache (MRv2 "
        "does not profile it). Too low risks OOM at capture; too high wastes "
        "KV. Profile: 700 (AWQ, NVFP4), 1200 (AWQ + DFlash2).",
        metrics="",
    ),
}

# speculative_config.sm70_dflash2.* keys (not environment variables).
DFLASH2_NOTES: dict[str, str] = {
    "context_kv_graph": "Capture the drafter's context-KV writes in CUDA graphs "
    "per token count instead of running them eagerly.",
    "context_pipeline": "Pipeline the drafter context-KV refresh with the "
    "verifier; accepted slots are refreshed before each replay.",
    "verify_fastpath": "Fast metadata path for verify steps in the hybrid "
    "(GDN + attention) model state.",
    "fused_gdn_metadata": "Build the GDN verify metadata with one fused launch.",
    "fused_smallq_metadata": "Fused Flash-V100 metadata for small-query verify "
    "batches.",
    "grouped_smallq_metadata": "Grouped small-query Flash-V100 metadata across "
    "requests.",
    "spec_state_replay": "Replay the accepted prefix to rebuild recurrent state "
    "instead of storing every speculative state: more GPU KV capacity at a "
    "small per-step cost. Off by default since 0.1.4; see "
    "[State replay](dflash2.md#state-replay-optional).",
}
