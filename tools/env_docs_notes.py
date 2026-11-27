# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hand-maintained notes for the generated configuration reference.

tools/gen_env_docs.py renders docs/configuration.md#environment-variable-reference
from vllm/envs.py metadata (type, default, category, description) plus these
notes. Table sections need: section, since (first release with the variable;
"1Cat" for variables inherited from 1Cat-vLLM), range, tradeoff (what the
setting does and when to change it, in one or two sentences) and metrics
(related /metrics series, or ""). The compact sections (lookup, developer,
retiring) need only section and tradeoff; for retiring, tradeoff names the
replacement. Add a variable here to document it, then run
`python tools/gen_env_docs.py`.
"""

SECTIONS = {
    "promoted": "Kernel and path switches",
    "deployment": "Deployment settings",
    "profile": "Values applied by the ReVolta profile",
    "research": "Experimental switches (off by default)",
    "usage": "Anonymous usage statistics",
    "lookup": "DFlash2 legacy lookup",
    "developer": "Developer diagnostics",
    "retiring": "Deprecated variables",
}

# Sections rendered as a short list or table instead of the full table.
COMPACT_SECTIONS = {
    "lookup": (
        "The legacy `ngram_assist` lookup for DFlash2 (off by default). These "
        "variables still work, with no removal release set, and setting any of "
        "them turns off the profile's default native lookup. New setups use "
        "`dflash2_lookup`; see "
        "[Native history lookup](dflash2.md#native-history-lookup)."
    ),
    "developer": (
        "These variables log, dump or switch internal paths for development. "
        "Leave them unset when serving."
    ),
    "retiring": (
        "Still accepted in 0.2.0; setting one logs a deprecation warning. For a "
        "renamed variable the new name wins when both are set."
    ),
}

PROFILE_NOTE = (
    "The ReVolta profile (`vllm/revolta_profile.py`) sets these on SM70 + TP4 + "
    "FP16 + QSA models when they are unset; explicit values win. The profile "
    "column shows the value it applies (AWQ-only and NVFP4-only entries apply to "
    "that checkpoint family only)."
)

NOTES: dict[str, dict[str, str]] = {
    "VLLM_DFLASH_COMPACT_AUX_HIDDEN": dict(
        section="promoted",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Builds the DFlash and DFlash2 draft inputs from the target's hidden "
            "states without an extra full-precision copy. Leave on."
        ),
        metrics="",
    ),
    "VLLM_SM70_CHECKPOINT_CACHE_DROP": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "At startup, after every rank has loaded the model, drops the "
            "checkpoint files from the page cache before the host KV pools are "
            "allocated. Files still mapped and the PLE disk modes are skipped. Set "
            "0 if other processes read the same checkpoint and need it cached."
        ),
        metrics="",
    ),
    "VLLM_SM70_GDN_CPU_SPEC_QSL": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Set 1 to remove two host synchronizations per step in batches that mix"
            " prefill and speculative decoding; output is unchanged."
        ),
        metrics="",
    ),
    "VLLM_SM70_GDN_NATIVE_PREFILL": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Runs eligible GDN prefill chunks with the native SM70 kernel (Model "
            "Runner V2). It can change floating-point rounding. Set 0 to use the "
            "TileLang kernel."
        ),
        metrics="",
    ),
    "VLLM_SM70_HOST_DECODE_FAST": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Host KV only: fused decode kernels for the history in host memory. Set"
            " 0 to use the original kernels."
        ),
        metrics="",
    ),
    "VLLM_SM70_MTP_KV_ONLY_PREFILL": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Lets the MTP draft skip work during chunked prefill; see [What the "
            "profile turns on for "
            "MTP](speculative-decoding.md#what-the-profile-turns-on-for-mtp). It "
            "can change floating-point rounding. Set 0 for full draft prefill."
        ),
        metrics="",
    ),
    "VLLM_SM70_QSA_LOCAL_INPUTS_FUSED": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "One fused kernel for the index metadata of the DCP-sharded QSA scorer."
            " Set 0 to use the reference implementation."
        ),
        metrics="",
    ),
    "VLLM_SM70_QSA_SHORT_APPEND": dict(
        section="promoted",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Lets short prompt continuations on a long cached context use the "
            "faster cuBLAS QSA scorer. Near-tied block selection can differ from "
            "the paged scorer. Set 0 to keep the paged scorer."
        ),
        metrics="",
    ),
    "VLLM_1CAT_ADAPTIVE_MTP": dict(
        section="deployment",
        since="0.2.0",
        range="off/0/false/no, on/1/true/yes, auto",
        tradeoff=(
            "Adapts the draft width of a single decoding request to its recent "
            "acceptance. Needs a draft-width schedule and synchronous scheduling; "
            "`auto` acts only on SM70 with at least 5 draft tokens. See [Adaptive "
            "MTP depth](dynamic-speculative-decoding.md#adaptive-mtp-depth)."
        ),
        metrics="",
    ),
    "VLLM_PLE_OFFLOAD_READY_TIMEOUT": dict(
        section="deployment",
        since="1Cat",
        range="seconds",
        tradeoff=(
            "How long startup waits for the PLE offload worker. Raise it on slow "
            "storage; the reference deployment uses 1200."
        ),
        metrics="",
    ),
    "VLLM_QWEN4EXP_PLE_HOST_GIB": dict(
        section="deployment",
        since="1Cat",
        range="GiB per TP rank; unset = automatic",
        tradeoff=(
            "Per-rank cap on pinned host memory for the FP8 PLE table. Unset pins "
            "the whole table; with a cap, the remaining rows stay on the GPU."
        ),
        metrics="",
    ),
    "VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB": dict(
        section="deployment",
        since="1Cat",
        range="non-negative GiB or unset",
        tradeoff=(
            "Host memory that PLE placement keeps free. Unset reserves the smaller "
            "of a quarter of the host limit and 8 GiB."
        ),
        metrics="",
    ),
    "VLLM_QWEN_REASONING_EOS_MASK": dict(
        section="deployment",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "1 keeps a request from ending its turn while it is still inside its "
            "reasoning block; see [the reasoning end-of-turn "
            "mask](#chat-control-tokens-and-the-reasoning-end-of-turn-mask)."
        ),
        metrics="",
    ),
    "VLLM_SM70_AWQ_MOE_INT4_ZERO": dict(
        section="deployment",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Stores AWQ zero points as 4-bit values: less GPU memory, same results."
            " Under `vllm serve` the profile sets 1 for AWQ g32 checkpoints with "
            "compact metadata."
        ),
        metrics="",
    ),
    "VLLM_SM70_FLASH_ATTN_V100": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "The V100 FlashAttention path; set in the image. Set 0 only to diagnose"
            " backend selection; use `--attention-backend` to choose a backend."
        ),
        metrics="",
    ),
    "VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Set to 1 in the image: torch.compile with FULL_AND_PIECEWISE CUDA "
            "graphs and small decode capture sizes. Leave as is."
        ),
        metrics="",
    ),
    "VLLM_SM70_HOST_KV_STATIC_HOT_POOL": dict(
        section="deployment",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Host KV only: 1 restores the separate static hot-page pool; the "
            "default lets hot pages share GPU memory with unused index pages."
        ),
        metrics="",
    ),
    "VLLM_SM70_IDLE_CLOCK_SECONDS": dict(
        section="deployment",
        since="0.2.0",
        range="non-negative integer seconds",
        tradeoff=(
            "Locks exclusively assigned V100 GPUs at 405 MHz after this many idle "
            "seconds; 0 disables it. Needs NVML clock permission and one engine "
            "with TP1 or TP4 on one host. After an abnormal exit, reset the GPU "
            "clocks before reusing the GPUs."
        ),
        metrics="",
    ),
    "VLLM_SM70_MTP_PLE_CONV": dict(
        section="deployment",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Fused PLE state update for MTP and DFlash2 batches. Set 0 only to "
            "diagnose a suspected kernel problem."
        ),
        metrics="",
    ),
    "VLLM_SM70_PLE_GATE_NORM": dict(
        section="deployment",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Opt-in fusion of the PLE gate and norm in decode graphs. It can change"
            " FP16 rounding."
        ),
        metrics="",
    ),
    "VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Turns off the inherited 1Cat MTP launch defaults; the ReVolta profile "
            "supplies its own."
        ),
        metrics="",
    ),
    "VLLM_EARLY_MODEL_INPUTS": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Prepares model inputs earlier in CUDA graph steps, overlapping host work."
        ),
        metrics="",
    ),
    "VLLM_PLE_ASYNC_REQUEST_INDICES": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff="Removes a host wait per step for the PLE inputs.",
        metrics="",
    ),
    "VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Runs shared and routed experts on separate CUDA streams. The profile "
            "sets 1 for AWQ g32 with DFlash2 when expert parallelism and dual-batch"
            " overlap are off."
        ),
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Grouped AWQ MoE decode kernel for small batches: each active expert is"
            " read once per step."
        ),
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS": dict(
        section="profile",
        since="0.1.0",
        range="2..48",
        tradeoff=(
            "Largest batch, in rows, for the grouped AWQ MoE kernel; larger batches"
            " use TurboMind. DFlash2 uses its own row cap."
        ),
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_PACK_ROWS": dict(
        section="profile",
        since="0.1.0",
        range="0 or 8..48 step 8",
        tradeoff=(
            "Tuning of the grouped AWQ kernel (rows per expert pack). Keep the "
            "profile value."
        ),
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT": dict(
        section="profile",
        since="0.1.0",
        range="4, 8, 16",
        tradeoff=(
            "Tuning of the grouped AWQ kernel (warps per expert). Keep the profile "
            "value."
        ),
        metrics="",
    ),
    "VLLM_SM70_AWQ_QWEN38_QPN_M1": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Single-request AWQ MoE kernel. Set 0 for the older route.",
        metrics="",
    ),
    "VLLM_SM70_AWQ_TUNE_SMALL_SHAPES": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Tunes small AWQ batch shapes at startup: longer warm-up, faster small "
            "batches."
        ),
        metrics="",
    ),
    "VLLM_SM70_CAR_VMM_SAFE": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Lets custom all-reduce CUDA graphs work with expandable segments (P2P "
            "or NVLink only)."
        ),
        metrics="",
    ),
    "VLLM_SM70_DFLASH2_FUSED_QKV_PACK": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="One kernel for the Q/K/V packing of the DFlash2 verifier.",
        metrics="",
    ),
    "VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Single-request DFlash2 verifier tail graphs. The profile sets 0 for "
            "DFlash2 so that their memory goes to the KV cache; with 1, also raise "
            "`VLLM_V2_CUDAGRAPH_MEM_MIB`."
        ),
        metrics="",
    ),
    "VLLM_SM70_GDN_SKINNY_MPLAN": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Faster GDN projection plans for some batch sizes; needs "
            "`VLLM_SM70_QWEN38_GDN_NO_PACK`."
        ),
        metrics="",
    ),
    "VLLM_SM70_GREEDY_VERIFY": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff="Faster MTP verification when every request in the batch is greedy.",
        metrics="",
    ),
    "VLLM_SM70_MTP_GDN_NATIVE_VERIFY": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Native GDN verification for the [AWQ MTP4 wide "
            "verifier](#awq-mtp4-wide-verifier). Set 0 to disable."
        ),
        metrics="",
    ),
    "VLLM_SM70_MTP_Q8_PREFILL_MMQ": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Grouped kernel for large MTP draft prefill batches (above 64 rows). It"
            " uses more temporary GPU memory and can change rounding. Set 0 for the"
            " smaller per-row kernel."
        ),
        metrics="",
    ),
    "VLLM_SM70_NVFP4_MOE_GROUPED_DECODE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Alternative NVFP4 MoE decode kernel; needs "
            "`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=0`. The profile keeps it off "
            "because its quality is not validated."
        ),
        metrics="",
    ),
    "VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="One grouped call for all routed experts in NVFP4 prefill.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="NVFP4 MoE decode kernel for 2–16 rows.",
        metrics="",
    ),
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Grouped NVFP4 MoE decode for 2–40 rows; needs raw scales "
            "(`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=1`)."
        ),
        metrics="",
    ),
    "VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Keeps NVFP4 block scales in their compact E4M3 form instead of a "
            "second FP16 copy, which saves GPU memory. Set 0 for FP16 scales, for "
            "example to use dual-batch overlap or microbatching, which raw scales "
            "do not support."
        ),
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
        tradeoff=(
            "TurboMind kernels for NVFP4 and MXFP4 dense layers. Set 0 only to "
            "diagnose."
        ),
        metrics="",
    ),
    "VLLM_SM70_PLE_INPUT_PREPARE": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff="Fused PLE input preparation for up to 4 requests.",
        metrics="",
    ),
    "VLLM_SM70_PLE_NGRAM_BATCH": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff="Fused PLE n-gram preprocessing for 2–32 rows.",
        metrics="",
    ),
    "VLLM_SM70_PLE_PREFILL_CONV_CHUNK": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Bounds the PLE prefill buffers when several prefills share a step; "
            "needed at `--max-num-seqs 8`."
        ),
        metrics="",
    ),
    "VLLM_SM70_QSA_COMPACT_TOPK": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Faster QSA block selection for up to 32 rows at long contexts; it "
            "selects the same blocks."
        ),
        metrics="",
    ),
    "VLLM_SM70_QSA_GROUPED_LPT": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Starts the longest QSA work first, which shortens uneven decode batches."
        ),
        metrics="",
    ),
    "VLLM_SM70_QSA_REQUEST_SEGMENTS": dict(
        section="profile",
        since="0.2.0",
        range="0, 1",
        tradeoff="Reads the QSA keys once per request in multi-request decode.",
        metrics="",
    ),
    "VLLM_SM70_QSA_SHARED_KEY": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff="Faster QSA indexer for the 2–8 verification queries of one request.",
        metrics="",
    ),
    "VLLM_SM70_QWEN38_GDN_NO_PACK": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Skips a packed copy of the GDN weights (about 0.7 GiB per GPU), which "
            "frees memory for the KV cache."
        ),
        metrics="",
    ),
    "VLLM_SM70_QWEN38_HYBRID_PLE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Per-rank pinned PLE shards for decode. The profile sets 0 (one shared "
            "PLE placement)."
        ),
        metrics="",
    ),
    "VLLM_SM70_QWEN38_NATIVE_ROUTER_TOPK": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Native MoE router for 17–64 rows in the [AWQ MTP4 wide "
            "verifier](#awq-mtp4-wide-verifier). Set 0 to disable."
        ),
        metrics="",
    ),
    "VLLM_SM70_QWEN38_WIDE_M": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff=(
            "Extends the fast Qwen3.8 decode and verify kernels from 4 to 8 requests."
        ),
        metrics="",
    ),
    "VLLM_SM70_SKIP_REJECTED_BATCH_PACKS": dict(
        section="profile",
        since="0.1.0",
        range="0, 1",
        tradeoff="Skips weight packs that are never used, about 199 MiB per GPU.",
        metrics="",
    ),
    "VLLM_USE_AOT_COMPILE": dict(
        section="profile",
        since="1Cat",
        range="0, 1",
        tradeoff="Ahead-of-time compile reuse. The profile sets 0.",
        metrics="",
    ),
    "VLLM_V2_CUDAGRAPH_MEM_MIB": dict(
        section="profile",
        since="1Cat",
        range="MiB",
        tradeoff=(
            "CUDA graph memory reserved per GPU before the KV cache is sized. The "
            "profile picks the value by method, checkpoint and KV format; see [CUDA"
            " graph reserve](checkpoints.md#cuda-graph-reserve). Raise it if the "
            "startup log warns that the graphs took more than the reserve."
        ),
        metrics="",
    ),
    "VLLM_SM70_NVFP4_RAW_GROUPED_M48_V154": dict(
        section="research",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Opt-in NVFP4 MoE decode kernel for 41–64 rows. Needs raw grouped "
            "decode, Model Runner V2 and equal gate/up global scales."
        ),
        metrics="",
    ),
    "VLLM_SM70_NVFP4_RAW_GROUPED_NARROW_V154": dict(
        section="research",
        since="0.2.0",
        range="0, 1",
        tradeoff=(
            "Opt-in NVFP4 MoE decode kernel for 3–5 rows. Needs raw grouped decode,"
            " Model Runner V2 and equal gate/up global scales."
        ),
        metrics="",
    ),
    "VLLM_DO_NOT_TRACK": dict(
        section="usage",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Honours the cross-project Do Not Track convention (DO_NOT_TRACK counts"
            " too): 1 turns the usage report off."
        ),
        metrics="",
    ),
    "VLLM_NO_USAGE_STATS": dict(
        section="usage",
        since="1Cat",
        range="0, 1",
        tradeoff=(
            "Upstream opt-out, honoured as-is: 1 turns the usage report off. The Do"
            " Not Track variables and the do_not_track file have the same effect."
        ),
        metrics="",
    ),
    "VLLM_USAGE_SOURCE": dict(
        section="usage",
        since="1Cat",
        range="short label",
        tradeoff=(
            "Deployment label sent as `source` (`production` by default). A value "
            "with `/`, `\\`, `:`, a space or more than 64 characters is sent as "
            "`custom`. Do not put identifying text in it."
        ),
        metrics="",
    ),
    "VLLM_USAGE_STATS_SERVER": dict(
        section="usage",
        since="1Cat",
        range="URL, empty to disable sending",
        tradeoff=(
            "Where the report goes. ReVolta defaults to its own collector; an empty"
            " string keeps the local log file but sends nothing."
        ),
        metrics="",
    ),
    "VLLM_DFLASH2_LOOKUP_ADAPTIVE": dict(
        section="lookup",
        tradeoff="Adaptive q8/q16 lookup width.",
    ),
    "VLLM_DFLASH2_LOOKUP_NSTRONG": dict(
        section="lookup",
        tradeoff="n-gram length that counts as a strong prompt match.",
    ),
    "VLLM_DFLASH2_LOOKUP_AGREE": dict(
        section="lookup",
        tradeoff="Leading tokens the lookup and the draft must agree on (0 = none).",
    ),
    "VLLM_DFLASH2_LOOKUP_NMIN_TAIL": dict(
        section="lookup",
        tradeoff="Smallest n-gram accepted for the lookup tail.",
    ),
    "VLLM_DFLASH2_LOOKUP_LONG_MIN": dict(
        section="lookup",
        tradeoff="Minimum length of a long match.",
    ),
    "VLLM_DFLASH2_LOOKUP_SEARCH": dict(
        section="lookup",
        tradeoff="Context searched for matches (default unbounded).",
    ),
    "VLLM_DFLASH2_LOOKUP_ENTRY_STREAK": dict(
        section="lookup",
        tradeoff="Consecutive hits before lookup mode starts.",
    ),
    "VLLM_DFLASH2_LOOKUP_STICKY": dict(
        section="lookup",
        tradeoff="Steps lookup mode stays on after a hit.",
    ),
    "VLLM_DFLASH2_LOOKUP_CHEAP_CONTEXT": dict(
        section="lookup",
        tradeoff="Context length below which lookup is always tried (0 = off).",
    ),
    "VLLM_DFLASH_PROFILE": dict(
        section="developer",
        tradeoff="DFlash and DFlash2 proposal timing log and its interval",
    ),
    "VLLM_DFLASH_PROFILE_LOG_INTERVAL": dict(
        section="developer",
        tradeoff="DFlash and DFlash2 proposal timing log and its interval",
    ),
    "VLLM_DFLASH_DEBUG_CONTEXT_KV": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_PROPOSAL_STAGES": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_STATE_TABLE": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_TARGET_LOGITS": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_TENSOR_DUMP_DIR": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_TENSOR_DUMP_LIMIT": dict(
        section="developer",
        tradeoff="DFlash debug logs, traces and tensor dumps",
    ),
    "VLLM_DFLASH_DEBUG_CORRUPTION": dict(
        section="developer",
        tradeoff=(
            "DFlash diagnostics for Model Runner V1 only; no effect on Qwen3.8 "
            "Flash-Next, which runs Model Runner V2"
        ),
    ),
    "VLLM_DFLASH_DISABLE_AUX_OUTPUTS": dict(
        section="developer",
        tradeoff=(
            "DFlash diagnostics for Model Runner V1 only; no effect on Qwen3.8 "
            "Flash-Next, which runs Model Runner V2"
        ),
    ),
    "VLLM_DFLASH_DUMP_DRAFT_LOGITS": dict(
        section="developer",
        tradeoff=(
            "DFlash diagnostics for Model Runner V1 only; no effect on Qwen3.8 "
            "Flash-Next, which runs Model Runner V2"
        ),
    ),
    "VLLM_DFLASH_DUMP_FIRST_PASS": dict(
        section="developer",
        tradeoff=(
            "DFlash diagnostics for Model Runner V1 only; no effect on Qwen3.8 "
            "Flash-Next, which runs Model Runner V2"
        ),
    ),
    "VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE": dict(
        section="developer",
        tradeoff=(
            "validation file for `VLLM_SM70_GDN_CPU_SPEC_QSL`; do not set it for "
            "timing runs"
        ),
    ),
    "VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD": dict(
        section="developer",
        tradeoff="GDN compile-boundary diagnostics; the image sets 0",
    ),
    "VLLM_SM70_QWEN_GDN_FULL_FORWARD": dict(
        section="developer",
        tradeoff="GDN compile-boundary diagnostics; the image sets 0",
    ),
    "VLLM_SM70_QWEN_GDN_SPEC_CORE_OP": dict(
        section="developer",
        tradeoff="GDN compile-boundary diagnostics; the image sets 0",
    ),
    "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING": dict(
        section="developer",
        tradeoff=(
            "experimental MTP draft vocabulary shortlist, outside release "
            "acceptance; it needs a ranking file that ReVolta does not provide"
        ),
    ),
    "VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE": dict(
        section="developer",
        tradeoff=(
            "experimental MTP draft vocabulary shortlist, outside release "
            "acceptance; it needs a ranking file that ReVolta does not provide"
        ),
    ),
    "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE": dict(
        section="developer",
        tradeoff=(
            "experimental MTP draft vocabulary shortlist, outside release "
            "acceptance; it needs a ranking file that ReVolta does not provide"
        ),
    ),
    "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FULL_REFRESH_INTERVAL": dict(
        section="developer",
        tradeoff="Model Runner V1 only; setting one with Model Runner V2 is an error",
    ),
    "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FUSED_PROPOSAL": dict(
        section="developer",
        tradeoff="Model Runner V1 only; setting one with Model Runner V2 is an error",
    ),
    "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU": dict(
        section="developer",
        tradeoff="Model Runner V1 only; setting one with Model Runner V2 is an error",
    ),
    "VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_PREFILL_TOPK": dict(
        section="developer",
        tradeoff="Model Runner V1 only; setting one with Model Runner V2 is an error",
    ),
    "REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP": dict(
        section="retiring",
        tradeoff="`VLLM_SM70_CHECKPOINT_CACHE_DROP`",
    ),
    "REVOLTA_RESEARCH_HOST_DECODE_FAST": dict(
        section="retiring",
        tradeoff="`VLLM_SM70_HOST_DECODE_FAST`",
    ),
    "REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED": dict(
        section="retiring",
        tradeoff="`VLLM_SM70_QSA_LOCAL_INPUTS_FUSED`",
    ),
    "REVOLTA_QSA_GROUPED_LPT": dict(
        section="retiring",
        tradeoff="`VLLM_SM70_QSA_GROUPED_LPT`",
    ),
    "REVOLTA_SKINNY_MPLAN": dict(
        section="retiring",
        tradeoff="`VLLM_SM70_GDN_SKINNY_MPLAN`",
    ),
    "VLLM_DFLASH_DDTREE_ATTN_COMPACT_BATCH": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_COMPACT_DRAFTER_CONTEXT": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_CONV_KERNEL": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_DEBUG": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_ENGINE_PROFILE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_FORCE_MAMBA_COMPACT": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_FUSED_GDN": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_GPU_SAMPLER": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_LINEAR_GDN": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_MAMBA_COMPACT_BATCH": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_METADATA_PROFILE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE_LAYER": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE_MAX_REPORTS": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_PATH_PROBE_NODE_LIMIT": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_PROFILE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_QLA_GDN": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_SERIAL_GDN": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_SKIP_MAMBA_COMPACT": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_STOCHASTIC_TOPK_LOGITS": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_TRACE_JSONL": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_TRITON_SAMPLER": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_CONTEXT": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_TOPK": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DDTREE_WORKER_PROFILE": dict(
        section="retiring",
        tradeoff=(
            "no replacement: DDTree (the `dflash_ddtree` speculative method) is retired"
        ),
    ),
    "VLLM_DFLASH_DEBUG_COORD_TRACE": dict(
        section="retiring",
        tradeoff="no replacement: glm5next is outside the supported models",
    ),
    "VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT": dict(
        section="retiring",
        tradeoff="no replacement: glm5next is outside the supported models",
    ),
}

# speculative_config.sm70_dflash2.* keys (not environment variables).
DFLASH2_NOTES: dict[str, str] = {
    "context_kv_graph": (
        "Capture the drafter's context-KV writes in CUDA graphs per token count "
        "instead of running them eagerly."
    ),
    "context_pipeline": (
        "Pipeline the drafter context-KV refresh with the verifier; accepted "
        "slots are refreshed before each replay."
    ),
    "verify_fastpath": (
        "Fast metadata path for verify steps in the hybrid (GDN + attention) "
        "model state."
    ),
    "fused_gdn_metadata": ("Build the GDN verify metadata with one fused launch."),
    "fused_smallq_metadata": (
        "Fused Flash-V100 metadata for small-query verify batches."
    ),
    "grouped_smallq_metadata": (
        "Grouped small-query Flash-V100 metadata across requests."
    ),
    "sparse_target_rejection": (
        "Avoid gathering full-vocabulary target logits for eligible sampled "
        "DFlash2 rounds. On by default for the qualifying Flash-Next serving "
        "profile with a DFlash2 selector (AWQ and NVFP4). Greedy and unsupported "
        "sampling stay dense. Set false to use dense verification for all "
        "requests; the legacy VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION alias is "
        "used only when the config value is unset."
    ),
    "spec_state_replay": (
        "Replay the accepted prefix to rebuild recurrent state instead of storing"
        " every speculative state: more GPU KV capacity at a small per-step cost."
        " The serving profile turns it on for DFlash2 unless fused_gdn_verify is "
        "set explicitly; Host KV mode also turns it on. Explicit false disables "
        "replay. See [State replay](dflash2.md#state-replay)."
    ),
}
