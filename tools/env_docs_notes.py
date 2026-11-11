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
    "research": "Pregraph preparation research switches",
    "promoted": "Promoted research switches (0.2.0rc5)",
    "deployment": "Settings used by the reference deployment",
    "profile": "Values applied by the ReVolta profile",
    "dflash": "DFlash2 and DFlash draft settings",
    "retiring": "Deprecated, removed in 0.2.0rc6",
}

PROFILE_NOTE = (
    "The ReVolta profile (`vllm/revolta_profile.py`) sets these on SM70 + TP4 + "
    "FP16 + QSA models when they are unset; explicit values win. The profile "
    "column shows the value it applies (AWQ-only and NVFP4-only entries apply to "
    "that checkpoint family only)."
)

_BITWISE = "Outputs are bitwise identical either way."
_LOOKUP = (
    "Legacy ngram_assist only (default off). Kept for compatibility with no "
    "removal version set. New configurations use explicit dflash2_lookup "
    "settings; see the native history lookup guide. Legacy values and cache "
    "factors are fixed at configuration resolution. "
)

NOTES: dict[str, dict[str, str]] = {
    "REVOLTA_RESEARCH_PREGRAPH_GDN_COMMON_CACHE": dict(
        section="research",
        since="unreleased research",
        range="0, 1",
        tradeoff="Deployment scope, default off. MRV2 FULL replay only: retain "
        "one pure-speculative batch's common GDN metadata, guarded by all CPU "
        "geometry values and the live GPU offset buffer layout. Removes repeated "
        "host construction, mask H2D and token-index arange on a hit. Mixed and "
        "capture batches use the original builder. Accepted counts, state IDs "
        "and request mapping are never cached. GPU bitwise and throughput "
        "validation pending. Set 0 and restart to restore the original builder.",
        metrics="no dedicated counter; compare decode throughput "
        "and preparation traces",
    ),
    "REVOLTA_RESEARCH_PREGRAPH_REPLAY_VIEWS": dict(
        section="research",
        since="unreleased research",
        range="0, 1",
        tradeoff="Deployment scope, default off. MRV2 FULL replay only: share "
        "read-only state-ID and query-offset tensor views within each metadata "
        "group for one step. No cross-step reuse or change to kernels, storage, "
        "accepted counts or fences. GPU bitwise and throughput validation "
        "pending. Set 0 and restart to restore per-layer view construction.",
        metrics="no dedicated counter; compare decode throughput "
        "and preparation traces",
    ),
    "REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP": dict(
        section="deployment",
        since="0.2.0",
        range="0, 1",
        tradeoff="Research, engine startup only; default off. 1 requests clean "
        "checkpoint cache eviction before host KV allocation, after a world "
        "load barrier. Skips live mappings and PLE disk/cascade modes. May "
        "slow later reloads or other engines reading the same checkpoint. "
        "See startup logs for advised files/bytes and cgroup usage; no metric.",
        metrics="",
    ),
    "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB": dict(
        section="deployment",
        since="1Cat",
        range="non-negative GiB or unset",
        tradeoff="Unset reserves min(host limit / 4, 8 GiB), preserving the "
        "small-host margin without scaling it to tens of GiB. Explicit values "
        "override the cap. The warning-only preflight counts shared CPU KV "
        "storage separately; automatic placement also uses this reserve.",
        metrics="",
    ),
    # ---- promoted research switches -------------------------------------
    "VLLM_SM70_GDN_CPU_SPEC_QSL": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff="Removes two host synchronizations per step in mixed prefill + "
        "speculative batches; costs a host-side offset computation. " + _BITWISE,
        metrics="",
    ),
    "VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE": dict(
        section="promoted",
        since="0.2.0",
        range="path",
        tradeoff="Validation only: adds the synchronizations back while the file "
        "exists. Never set for timing runs.",
        metrics="",
    ),
    "VLLM_SM70_QSA_GROUPED_LPT": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff="Longest grouped page4 CTAs start first, which shortens the "
        "tail of uneven decode batches. " + _BITWISE,
        metrics="",
    ),
    "VLLM_SM70_GDN_SKINNY_MPLAN": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff="ng=2 tiling is 8-26% faster at M = 9..24 and 33..48 on V100; "
        "other M keep the fixed plan. Only with VLLM_SM70_QWEN38_GDN_NO_PACK. "
        + _BITWISE,
        metrics="",
    ),
    "REVOLTA_GDN_CPU_SPEC_QSL": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff="Deprecated alias of VLLM_SM70_GDN_CPU_SPEC_QSL; removed in "
        "0.2.0rc6. The new name wins when both are set.",
        metrics="",
    ),
    "REVOLTA_GDN_CPU_SPEC_QSL_CHECK_FILE": dict(
        section="promoted",
        since="0.2.0",
        range="path",
        tradeoff="Deprecated alias of VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE; "
        "removed in 0.2.0rc6.",
        metrics="",
    ),
    "REVOLTA_QSA_GROUPED_LPT": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff="Deprecated alias of VLLM_SM70_QSA_GROUPED_LPT; removed in 0.2.0rc6.",
        metrics="",
    ),
    "REVOLTA_SKINNY_MPLAN": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff="Deprecated alias of VLLM_SM70_GDN_SKINNY_MPLAN; removed in 0.2.0rc6.",
        metrics="",
    ),
    # ---- reference deployment ---------------------------------------------
    "VLLM_SM70_HOST_KV_STATIC_HOT_POOL": dict(
        section="deployment",
        since="0.2.0",
        range="0, 1",
        tradeoff="Long-context host placement only. 1 restores the separate "
        "static hot-page pool (history IDs capped by host pages); the default "
        "shares free history-ID index pages with the hot slots.",
        metrics="",
    ),
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
    # ---- DFlash2 / DFlash draft ------------------------------------------
    "VLLM_DFLASH2_LOOKUP_ADAPTIVE": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff=_LOOKUP + "Adaptive q8/q16 lookup width.",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_NSTRONG": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 1",
        tradeoff=_LOOKUP + "n-gram length that counts as a strong prompt match.",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_AGREE": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 0",
        tradeoff=_LOOKUP
        + "Leading tokens the lookup and the draft must agree on (0 = none).",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_NMIN_TAIL": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 1",
        tradeoff=_LOOKUP + "Smallest n-gram accepted for the lookup tail.",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_LONG_MIN": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 1",
        tradeoff=_LOOKUP + "Minimum length of a long match.",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_SEARCH": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 1",
        tradeoff=_LOOKUP + "Context searched for matches (default unbounded).",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_ENTRY_STREAK": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 1",
        tradeoff=_LOOKUP + "Consecutive hits before lookup mode starts.",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_STICKY": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 0",
        tradeoff=_LOOKUP + "Steps lookup mode stays on after a hit.",
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_CHEAP_CONTEXT": dict(
        section="dflash",
        since="1Cat",
        range="integer >= 0",
        tradeoff=_LOOKUP
        + "Context length below which lookup is always tried (0 = off).",
        metrics="",
    ),
    "VLLM_DFLASH_COMPACT_AUX_HIDDEN": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Concatenates the target auxiliary hidden states directly in the "
        "draft projection dtype (DFlash and DFlash2 drafts); avoids an extra "
        "full-precision copy.",
        metrics="",
    ),
    "VLLM_DFLASH_PROFILE": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug: DFlash/DFlash2 proposal timing log.",
        metrics="",
    ),
    "VLLM_DFLASH_PROFILE_LOG_INTERVAL": dict(
        section="dflash",
        since="1Cat",
        range="steps",
        tradeoff="Debug: interval of the timing log.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_TARGET_LOGITS": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug: log target logits (MRv2).",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION": dict(
        section="dflash",
        since="1Cat",
        range="position",
        tradeoff="Debug: first position of the target layer trace.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_STATE_TABLE": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug: log the GDN speculative state table.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_CONTEXT_KV": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug: log the draft context KV.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_PROPOSAL_STAGES": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug: DFlash v1 speculator (MRv2) proposal stages.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_TENSOR_DUMP_DIR": dict(
        section="dflash",
        since="1Cat",
        range="path",
        tradeoff="Debug: DFlash v1 speculator (MRv2) tensor dump directory.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_TENSOR_DUMP_LIMIT": dict(
        section="dflash",
        since="1Cat",
        range="count",
        tradeoff="Debug: maximum tensor dumps.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug: target layer trace (NVFP4 MoE).",
        metrics="",
    ),
    "VLLM_DFLASH_DISABLE_AUX_OUTPUTS": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug, MRv1 only: disable the auxiliary outputs for the draft.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_CORRUPTION": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug, MRv1 only: draft corruption diagnostics.",
        metrics="",
    ),
    "VLLM_DFLASH_DUMP_DRAFT_LOGITS": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug, MRv1 only: dump draft logits.",
        metrics="",
    ),
    "VLLM_DFLASH_DUMP_FIRST_PASS": dict(
        section="dflash",
        since="1Cat",
        range="0, 1",
        tradeoff="Debug, MRv1 DFlash (v1) only: dump the first draft pass.",
        metrics="",
    ),
    # ---- retiring in 0.2.0rc6 ---------------------------------------------
    "VLLM_DFLASH_DDTREE_ATTN_COMPACT_BATCH": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_COMPACT_DRAFTER_CONTEXT": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_CONV_KERNEL": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_DEBUG": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_ENGINE_PROFILE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_FORCE_MAMBA_COMPACT": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_FUSED_GDN": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_GPU_SAMPLER": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_LINEAR_GDN": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_MAMBA_COMPACT_BATCH": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_METADATA_PROFILE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE_LAYER": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE_MAX_REPORTS": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE_NODE_LIMIT": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_PROFILE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_QLA_GDN": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_SERIAL_GDN": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_SKIP_MAMBA_COMPACT": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_STOCHASTIC_TOPK_LOGITS": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_TRACE_JSONL": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_TRITON_SAMPLER": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_CONTEXT": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_TOPK": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DDTREE_WORKER_PROFILE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: DDTree is retired.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_COORD_TRACE": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: glm5next is unsupported.",
        metrics="",
    ),
    "VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT": dict(
        section="retiring",
        since="1Cat",
        range="-",
        tradeoff="Deprecated: glm5next is unsupported.",
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
    "small per-step cost. On by default in current development for supported "
    "ordinary DFlash2 routes; released 0.1.4–0.1.5 require explicit true. "
    "None retains an enabled packed verifier, and explicit false disables replay. See "
    "[State replay](dflash2.md#state-replay-optional).",
}
