# Configuration reference

This page explains the reference deployment settings, sampling defaults and resource budgets.
Use the [getting-started tutorial](getting-started.md) for the installation and launch sequence.

## Environment settings

The engine detects SM70, TP4, a QSA model (Qwen3.8 Flash-Next) and FP16, and then sets
the tuned kernel switches by itself (the "ReVolta profile"). Only these deployment
settings remain:

The complete reference exports are in [step 3 of the tutorial](getting-started.md#3-environment).

Use only the documented `VLLM_SM70_*` settings unless you are troubleshooting. Explicit
values override the profile.

## Interconnect

The engine selects the interconnect automatically and logs `SM70 interconnect=<mode>`.
P2P uses custom all-reduce for messages up to 80 KiB inside CUDA graphs; other
collectives use NCCL. On an SM70 TP4 group (under one PCIe switch, or on NVLink) the MoE
output sum (sum2) uses its own push kernel at the sizes that have one and otherwise adds
locally before the ordinary all-reduce. Override with `VLLM_SM70_INTERCONNECT=auto|pex|nccl|nvlink`
or disable custom all-reduce with `--disable-custom-all-reduce`.

## Launch options

- **`draft_sample_method`: `greedy` for MTP, `probabilistic` for DFlash2.** With MTP, a
  probabilistic draft corrupts output above `max_num_seqs=2` at temperature > 0
  (upstream 1Cat-vLLM issue #656), so MTP must stay `greedy`. For DFlash2, `probabilistic`
  is the recommended setting from 0.1.4; it was validated in the 0.1.4 release gate at one
  to eight concurrent requests, at temperature 1 with thinking, including tool calls and
  structured output. `greedy` still works with DFlash2. This setting is about drafting and
  draft verification only; clients can still use any sampling parameters.
- **`--watchdog-config` (opt-in stack-dump watchdog, from 0.1.4, #33).** JSON with three
  fields: `dump_dir` (directory for the dump files; the empty default disables the
  watchdog), `timeout` (seconds without progress after which the process counts as stuck,
  default 300, must be greater than 0) and `check_interval` (seconds between checks,
  default 10). Example:
  `--watchdog-config '{"timeout": 300, "check_interval": 10, "dump_dir": "/data/revolta/dumps"}'`.
  When enabled it runs a background thread in the engine core process (`engine_<dp_rank>`)
  and in each worker (`worker_<rank>`); if that process makes no progress for `timeout`
  seconds it appends a numbered dump of all its Python thread stacks to
  `<dump_dir>/VLLM_STACK_DUMP_for_<name>_<pid>.log`, at most once per stall, and logs
  `[Watchdog]Dumped stack to ...`. The engine core also dumps its stacks on SIGTERM.
  Waiting for incoming work does not count as a stall: the timeout window restarts when
  the process resumes. This is a diagnostic aid, not a recovery mechanism: the process is
  not restarted or killed.
- **`--shutdown-timeout 30` (ReVolta default since 0.2.0; upstream vLLM: 0).** On SIGTERM
  the engine lets in-flight requests run for up to this many seconds, then aborts the rest
  and answers them. With `0` the engine process is stopped at once: requests still in
  flight can stay unanswered until the container runtime kills the server, and CPU
  offload cleanup does not run (see [Support and known limits](known-limits.md)). Keep the
  container stop grace period (for example `docker stop --time 120`) above this value.
- **`--gpu-memory-utilization 0.95`** passed the full gate, including the multimodal
  stress test with 8 concurrent requests. Higher values ran out of memory in earlier
  multimodal stress tests. If you change concurrency or media limits, measure again.
- The scheduler settings (`--prefill-pressure-*`, `--long-prefill-token-threshold`,
  `--mixed-prefill-step-latency-ms`) and `ple_disk_cascade: false` are the tested
  values for this machine. They are deployment settings, not universal defaults.
- `cpu_bytes_to_use` is the total for all four workers (48 GiB = 12 GiB per worker).
- **Keep `--limit-mm-per-prompt` on V100.** With FlashAttention-V100, the engine allows
  1 image and no video per request without it, and a value that leaves out `"video"`
  disables video. 0.1.4 keeps this V100 default. `{"image": 16, "video": 1}`
  keeps the reference deployment's limits; Qwen3.8 Flash-Next takes images and video,
  not audio, so no audio limit is set. To use the generic image count instead, set
  `--limit-mm-per-prompt '{"image": 8, "video": 1}'` (the inherited generic default is
  8 images, 1 video and 1 audio item, and the audio field is left out here because this
  model has no audio input); a lower image count leaves more
  host memory for bursts of large images. Large images are resized
  either way (see [Support and known limits](known-limits.md)).
- `--structured-outputs-config '{"backend": "xgrammar"}'` pins the structured-output
  backend. With `auto`, the engine keeps the backend chosen for its first
  structured-output request.
- `--sse-keep-alive-interval 15` sends an SSE comment line (`: keep-alive`) after 15 idle
  seconds on streaming responses, so proxies with read timeouts keep the connection open
  during queueing and prefill. The default, 0, sends none.
- `TILELANG_CACHE_DIR` ([step 3](getting-started.md#3-environment)) keeps TileLang's compiled kernels on persistent storage.
  Without persistent cache storage, a recreated container compiles them again.
- Media limits and request budgets: see
  [multimodal input limits](../.github/MULTIMODAL_INPUTS.md).

## Optional native DFlash2 lookup

The `dflash2_lookup` object inside `--speculative-config` is disabled by default.
See [native history lookup](dflash2.md#native-history-lookup-optional) for the C64
recipe, scheduling limits, measurements and statistics.

| Field | Default | Accepted settings |
| --- | --- | --- |
| `enabled` | `false` | Boolean; enable explicitly |
| `policy` | unset | `longest`; required when enabled |
| `nmin`, `nmax` | unset | Positive integers with `nmin <= nmax`; required when enabled |
| `search_window` | `65536` | Positive token distance in committed history |
| `gate` | `first_token` | Strict agreement with the original first draft token |
| `collect_stats` | `true` | Also requires the engine's effective statistics switch |
| `max_requests` | `3` | Strict integer from 1 to 3; set to 1 for single-request admission |
| `expand.enabled` | `false` | Must remain false, even when lookup is disabled |

Native lookup requires a seven-token DFlash2 selector drafter and seven configured
speculative tokens. It attempts lookup only when every live row is an eligible
decode request, the batch contains at most `max_requests` requests, and the
existing concurrency ladder selects width seven. Otherwise the entire batch
bypasses lookup. Four or more requests always bypass lookup. Fixed seven without a ladder permits
async subject to existing constraints; a nonempty ladder rejects explicit async
and otherwise selects synchronous scheduling.

Raw input provenance is captured before normalization and preserved through worker
reconstruction. Worker environment changes do not reinterpret resolved lookup
settings. Enabled native lookup rejects conflicting legacy assist, prompt bounds
or lookup environment inputs; explicit disabled/null/empty inputs retain the
existing effective-mode warnings. Legacy inputs remain compatible without a
scheduled removal. Lookup configuration changes isolate active compile/graph
identities; provenance and the statistics switch are not model-forward factors.

## Sampling parameters

The server takes default sampling values from the checkpoint's `generation_config.json`
(`temperature` 1.0, `top_p` 0.95, `top_k` 20), but only for fields that a request
leaves out. Many chat clients send their own `temperature` and `top_p`, so check the
client settings. Qwen recommends:

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

- **Endless repetition:** Qwen suggests a `presence_penalty` between 0 and 2. Higher values
  can mix languages and slightly lower quality.
- `presence_penalty` can only be set per request. `--override-generation-config` sets
  server defaults for `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` and
  `max_new_tokens` only.
- `top_k` and `min_p` are not standard OpenAI fields. With the OpenAI Python SDK, pass them
  in `extra_body`.
- These are the model vendor's values. The [speed tests](performance.md) use
  `temperature=0`. The `draft_sample_method` in the launch command applies only to
  drafting and draft verification, not to these request settings.

## YaRN and native context

YaRN in vLLM is static: it applies the same scaling to every request, short or long.
Qwen's model card notes that this can lower quality on shorter texts and advises
enabling YaRN only when long contexts are needed. ReVolta's quality measurements
compare against a BF16 reference that also uses YaRN, so they do not show this
effect. If your requests stay within 262,144 tokens, do not make YaRN views: use
`--max-model-len 262144` and serve the checkpoint directory itself (the materialized
directory for Standard NVFP4 and FP4-Local). For the MTP draft, use the AWQ
checkpoint's own `mtp-fp8/`, the scale pack's `mtp-fp8/` for Standard NVFP4, or
FP4-Local's built-in MTP layer with no draft `model` field. The full acceptance ran
with the 512K views.

## System memory and storage

**128 GiB RAM is recommended** for the default 48 GiB CPU KV pool. The AWQ PLE
embedding table uses about 47.7 GiB of pinned host memory even with KV offload disabled.

| Container memory limit | KV offload | Result |
| --- | --- | --- |
| 128 GiB | Off, or 48 GiB CPU pool | Full acceptance passed |
| 80 GiB | Off | Memory stress passed; full acceptance not run |
| 64 GiB | Off | Out of memory during startup |
| 64 GiB, whole PLE table on disk (experimental) | Off, or 32 GiB CPU pool | Served; greedy output matched the pinned path (0.1.4) |
| 32 GiB, whole PLE table on disk (experimental) | Off | Served; greedy output matched the pinned path (0.1.4) |

With the default pinned PLE table, 80 GiB is the smallest tested limit that passed memory
stress, with KV offload off and without full acceptance. With less than 128 GiB,
reduce or disable the CPU KV pool. The experimental whole-table PLE disk option
([Support and known limits](known-limits.md)) keeps the table on a local SSD instead and
brings the tested minimum down to 32 GiB; single-request decoding is about 23–25% slower
and a cold first token takes 1.6–2.3 times as long.

Image and video preprocessing also uses host memory, in the API server process. In one
test, 8 concurrent requests with 8 different 2048-pixel images each used about 33 GB at
peak. With a 32 GiB CPU KV pool on a 125 GiB host, the kernel then killed a worker. The
same burst passed with a 16 GiB pool. If you expect bursts of large images or videos,
use a smaller CPU KV pool or limit the image size and the number of concurrent requests.

The AWQ checkpoint is 137 GB at the pinned revision:

| Part | Format | Size | Where it runs |
| --- | --- | ---: | --- |
| Routed experts | 4-bit AWQ, group size 32 | 74.9 GB | GPU |
| PLE n-gram tables | FP8 E4M3 | 51.3 GB | Pinned host memory |
| Other weights (attention, embeddings, `lm_head`, vision tower, shared experts) | 16-bit | 10.9 GB | GPU |
| Optional MTP draft (`mtp-fp8/`) | FP8 | 5.3 GB | GPU |

Allow extra disk space for compilation caches and the optional disk KV tier.

<!-- BEGIN GENERATED: tools/gen_env_docs.py -->
<!-- Edit tools/env_docs_notes.py, then run python tools/gen_env_docs.py -->

## Environment variable reference

Every variable below is read once per process (deployment scope). Columns:
type and default as registered in `vllm/envs.py`; *Since* is the first
release with the variable (1Cat = inherited from 1Cat-vLLM); *Category*
is configuration, tuning, experimental, debug or deprecated.

### Promoted research switches (0.2.0rc5)

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `REVOLTA_GDN_CPU_SPEC_QSL` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_GDN_CPU_SPEC_QSL; removed in 0.2.0rc6. The new name wins when both are set. |
| `REVOLTA_GDN_CPU_SPEC_QSL_CHECK_FILE` | str | `''` | path | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE; removed in 0.2.0rc6. |
| `REVOLTA_QSA_GROUPED_LPT` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_QSA_GROUPED_LPT; removed in 0.2.0rc6. |
| `REVOLTA_SKINNY_MPLAN` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_GDN_SKINNY_MPLAN; removed in 0.2.0rc6. |
| `VLLM_SM70_GDN_CPU_SPEC_QSL` | bool | `False` | 0, 1 | tuning | 0.2.0 | Removes two host synchronizations per step in mixed prefill + speculative batches; costs a host-side offset computation. Outputs are bitwise identical either way. |
| `VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE` | str | `''` | path | debug | 0.2.0 | Validation only: adds the synchronizations back while the file exists. Never set for timing runs. |
| `VLLM_SM70_GDN_SKINNY_MPLAN` | bool | `False` | 0, 1 | tuning | 0.2.0 | ng=2 tiling is 8-26% faster at M = 9..24 and 33..48 on V100; other M keep the fixed plan. Only with VLLM_SM70_QWEN38_GDN_NO_PACK. Outputs are bitwise identical either way. |
| `VLLM_SM70_QSA_GROUPED_LPT` | bool | `False` | 0, 1 | tuning | 0.2.0 | Longest grouped page4 CTAs start first, which shortens the tail of uneven decode batches. Outputs are bitwise identical either way. |

### Settings used by the reference deployment

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP` | bool | `False` | 0, 1 | experimental | 0.2.0 | Research, engine startup only; default off. 1 requests clean checkpoint cache eviction before host KV allocation, after a world load barrier. Skips live mappings and PLE disk/cascade modes. May slow later reloads or other engines reading the same checkpoint. See startup logs for advised files/bytes and cgroup usage; no metric. |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | float | `600.0` | seconds | configuration | 1Cat | How long startup waits for the PLE offload worker. Raise it on slow storage; the reference deployment uses 1200. |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | float \| None | `None` | GiB per TP rank; unset = automatic | configuration | 1Cat | Caps pinned host memory for the PLE table. Lower values leave host RAM for CPU KV offload but move more PLE rows to disk; the reference deployment uses 12. |
| `VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB` | float \| None | `None` | non-negative GiB or unset | configuration | 1Cat | Unset reserves min(host limit / 4, 8 GiB), preserving the small-host margin without scaling it to tens of GiB. Explicit values override the cap. The warning-only preflight counts shared CPU KV storage separately; automatic placement also uses this reserve. |
| `VLLM_SM70_AWQ_MOE_INT4_ZERO` | bool | `False` | 0, 1 | experimental | 0.2.0 | Packs AWQ zero points into nibbles: less resident metadata, lossless. Needs compact metadata and a native build with the packed kernels. |
| `VLLM_SM70_DFLASH2_FUSED_QKV_PACK` | bool | `False` | 0, 1 | configuration | 1Cat | One Triton launch instead of three copies and a concat in the DFlash2 verifier. Outputs are bitwise identical either way. |
| `VLLM_SM70_FLASH_ATTN_V100` | bool | `True` | 0, 1 | configuration | 1Cat | Default on (set in the image). 0 only to diagnose backend selection; prefer --attention-backend to choose a backend. |
| `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` | bool | `False` | 0, 1 | configuration | 1Cat | Set in the image: VLLM_COMPILE + FULL_AND_PIECEWISE with small decode captures (the 0.0.3 graph policy). |
| `VLLM_SM70_HOST_KV_STATIC_HOT_POOL` | bool | `False` | 0, 1 | configuration | 0.2.0 | Long-context host placement only. 1 restores the separate static hot-page pool (history IDs capped by host pages); the default shares free history-ID index pages with the hot slots. |
| `VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD` | bool | `False` | 0, 1 | configuration | 1Cat | Diagnostic: compare the split GDN compile boundary with the full-forward guard. The image sets 0. |
| `VLLM_SM70_QWEN_GDN_FULL_FORWARD` | bool | `False` | 0, 1 | configuration | 1Cat | Diagnostic force-on of the opaque GDN full-forward boundary. The image sets 0; the automatic default arms it only for active MTP batches. |
| `VLLM_SM70_QWEN_GDN_SPEC_CORE_OP` | bool | `False` | 0, 1 | experimental | 1Cat | Experimental narrower MTP boundary (only the recurrent core is opaque). The image sets 0 until it passes long-output quality. |

### Values applied by the ReVolta profile

The ReVolta profile (`vllm/revolta_profile.py`) sets these on SM70 + TP4 + FP16 + QSA models when they are unset; explicit values win. The profile column shows the value it applies (AWQ-only and NVFP4-only entries apply to that checkpoint family only).

| Name | Type | Default | Profile | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|---|
| `VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS` | bool | `False` | `1` | 0, 1 | configuration | 1Cat | Opts out of the 1Cat SM70 MTP launch defaults; the ReVolta profile supplies its own. |
| `VLLM_EARLY_MODEL_INPUTS` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Prepares model-state inputs before attention metadata in FULL graph steps, overlapping host work. |
| `VLLM_PLE_ASYNC_REQUEST_INDICES` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Non-blocking PLE request-index transfers; removes a host wait per step. |
| `VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP` | bool | `False` | `0` | 0, 1 | configuration | 1Cat | Shared-expert stream overlap. The profile sets 0 on TP4 SM70, where the second stream does not pay off. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN` | bool | `False` | `1` (AWQ) | 0, 1 | experimental | 0.2.0 | Grouped AWQ MoE decode reads each active expert once for all its rows (2..48). Faster at 2..8 requests; reduction order differs from the TurboMind route. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS` | int | `20` | `40` (AWQ) | 2..48 | tuning | 0.2.0 | Largest row count sent to the grouped route; above it rows use TurboMind. The profile uses 40 (8 requests x 5 tokens). |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_PACK_ROWS` | int | `0` | `24` (AWQ) | 0 or 8..48 step 8 | tuning | 0.2.0 | Caps rows per expert pack: fewer registers per CTA but hot experts are read more than once. Result unchanged. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT` | int | `16` | `8` (AWQ) | 4, 8, 16 | tuning | 0.2.0 | W13 split-K warps per CTA. 16 matches the M1 arithmetic; 8 (profile) uses fewer threads and a different FP32 split association. |
| `VLLM_SM70_AWQ_QWEN38_QPN_M1` | bool | `True` | `1` (AWQ) | 0, 1 | configuration | 1Cat | Single-token W13/W2 QPN route; 0 restores the legacy route. Changes reduction order. |
| `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES` | bool | `False` | `1` (AWQ) | 0, 1 | configuration | 1Cat | Tunes TurboMind small-M AWQ shapes at startup: longer warmup, faster small batches. |
| `VLLM_SM70_CAR_VMM_SAFE` | bool | `False` | `1` (P2P/NVLink) | 0, 1 | experimental | 0.2.0 | Copies non-exportable (VMM) inputs into graph allocations while capturing custom all-reduce graphs, so expandable segments can stay on. Only with P2P or NVLink. |
| `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` | bool | `True` | `0` (AWQ+DFlash2) | 0, 1 | configuration | 1Cat | Single-request DFlash2 verifier tail graphs. The AWQ DFlash profile sets 0 to keep the graph memory for the KV cache. |
| `VLLM_SM70_GREEDY_VERIFY` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Greedy MTP verification from TP-local argmax pairs instead of full-vocabulary logits; only when every request is plain greedy. |
| `VLLM_SM70_NVFP4_MOE_GROUPED_DECODE` | bool | `False` | `0` (NVFP4) | 0, 1 | experimental | 1Cat | Grouped native-NVFP4 decode; the profile keeps it off pending quality admission. |
| `VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | One grouped TurboMind call for all routed experts in prefill. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 1Cat | QPN for every live width 2..16 (NVFP4). |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 0.2.0 | Graph-safe raw-E4M3 grouped NVFP4 MoE for M = 2..40; needs RAW_SCALE storage. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 1Cat | Keeps E4M3 block-scale codes resident instead of a second FP16 copy: less memory, decode reconstructs exact values. |
| `VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | Tunes small NVFP4 shapes at startup. |
| `VLLM_SM70_NVFP4_TURBOMIND` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | TurboMind route for NVFP4/MXFP4 dense layers; 0 only for diagnosis. |
| `VLLM_SM70_PLE_INPUT_PREPARE` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | One fused launch for PLE n-gram context gathering (<=4 requests). |
| `VLLM_SM70_PLE_NGRAM_BATCH` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Fused PLE n-gram id preprocessing for 2..32 rows. |
| `VLLM_SM70_PLE_PREFILL_CONV_CHUNK` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Bounds PLE short-conv prefill buffers when several prefills share a step; needed at max_num_seqs=8. Outputs are bitwise identical either way. |
| `VLLM_SM70_QSA_COMPACT_TOPK` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Coarse FP16 pre-selection before the exact QSA top-k for <=32 rows and long contexts; same indices. |
| `VLLM_SM70_QSA_REQUEST_SEGMENTS` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Reads each 64-key tile once per request segment in multi-request decode. Outputs are bitwise identical either way. |
| `VLLM_SM70_QSA_SHARED_KEY` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Shared-key FP16 indexer for M2..8 verification queries of one request. |
| `VLLM_SM70_QWEN38_GDN_NO_PACK` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Skips the packed GDN weight copy (725.6 MiB per TP4 rank) and serves 2..48 rows with the skinny dual GEMM; frees memory for KV. |
| `VLLM_SM70_QWEN38_HYBRID_PLE` | bool | `False` | `0` | 0, 1 | configuration | 1Cat | Local pinned PLE shards for decode. The profile sets 0 (single PLE placement) on the reference deployment. |
| `VLLM_SM70_QWEN38_WIDE_M` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Widens Qwen3.8 decode/verify routes from 4 to 8 requests (M up to 40); each VLLM_SM70_QWEN38_WIDE_M_* item can override it. |
| `VLLM_SM70_SKIP_REJECTED_BATCH_PACKS` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Skips batch packs the precision policy never admits; saves about 199 MiB per rank. |
| `VLLM_USE_AOT_COMPILE` | bool | `False` | `0` | 0, 1 | configuration | 1Cat | AOT compile reuse. The profile sets 0 on the reference deployment. |
| `VLLM_V2_CUDAGRAPH_MEM_MIB` | float | `0.0` | `700` (AWQ), `1200` (AWQ+DFlash2), `700` (NVFP4) | MiB | configuration | 1Cat | CUDA graph memory reserved before sizing the KV cache (MRv2 does not profile it). Too low risks OOM at capture; too high wastes KV. Profile: 700 (AWQ, NVFP4), 1200 (AWQ + DFlash2). |

### DFlash2 and DFlash draft settings

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_DFLASH2_LOOKUP_ADAPTIVE` | bool | `True` | 0, 1 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Adaptive q8/q16 lookup width. |
| `VLLM_DFLASH2_LOOKUP_AGREE` | int | `Computed when unset: max(0, int('0'))` | integer >= 0 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Leading tokens the lookup and the draft must agree on (0 = none). |
| `VLLM_DFLASH2_LOOKUP_CHEAP_CONTEXT` | int | `Computed when unset: max(0, int('0'))` | integer >= 0 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Context length below which lookup is always tried (0 = off). |
| `VLLM_DFLASH2_LOOKUP_ENTRY_STREAK` | int | `Computed when unset: max(1, int('2'))` | integer >= 1 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Consecutive hits before lookup mode starts. |
| `VLLM_DFLASH2_LOOKUP_LONG_MIN` | int | `Computed when unset: max(1, int('6'))` | integer >= 1 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Minimum length of a long match. |
| `VLLM_DFLASH2_LOOKUP_NMIN_TAIL` | int | `Computed when unset: max(1, int('4'))` | integer >= 1 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Smallest n-gram accepted for the lookup tail. |
| `VLLM_DFLASH2_LOOKUP_NSTRONG` | int | `Computed when unset: max(1, int('6'))` | integer >= 1 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. n-gram length that counts as a strong prompt match. |
| `VLLM_DFLASH2_LOOKUP_SEARCH` | int | `Computed when unset: max(1, int(str(1 << 30)))` | integer >= 1 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Context searched for matches (default unbounded). |
| `VLLM_DFLASH2_LOOKUP_STICKY` | int | `Computed when unset: max(0, int('3'))` | integer >= 0 | experimental | 1Cat | Legacy ngram_assist only (default off). Kept for compatibility with no removal version set. New configurations use explicit dflash2_lookup settings; see the native history lookup guide. Legacy values and cache factors are fixed at configuration resolution. Steps lookup mode stays on after a hit. |
| `VLLM_DFLASH_COMPACT_AUX_HIDDEN` | bool | `True` | 0, 1 | configuration | 1Cat | Concatenates the target auxiliary hidden states directly in the draft projection dtype (DFlash and DFlash2 drafts); avoids an extra full-precision copy. |
| `VLLM_DFLASH_DEBUG_CONTEXT_KV` | bool | `False` | 0, 1 | debug | 1Cat | Debug: log the draft context KV. |
| `VLLM_DFLASH_DEBUG_CORRUPTION` | bool | `False` | 0, 1 | debug | 1Cat | Debug, MRv1 only: draft corruption diagnostics. |
| `VLLM_DFLASH_DEBUG_PROPOSAL_STAGES` | str \| None | `None` | 0, 1 | debug | 1Cat | Debug: DFlash v1 speculator (MRv2) proposal stages. |
| `VLLM_DFLASH_DEBUG_STATE_TABLE` | bool | `False` | 0, 1 | debug | 1Cat | Debug: log the GDN speculative state table. |
| `VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE` | str \| None | `None` | 0, 1 | debug | 1Cat | Debug: target layer trace (NVFP4 MoE). |
| `VLLM_DFLASH_DEBUG_TARGET_LOGITS` | str \| None | `None` | 0, 1 | debug | 1Cat | Debug: log target logits (MRv2). |
| `VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION` | str \| None | `None` | position | debug | 1Cat | Debug: first position of the target layer trace. |
| `VLLM_DFLASH_DEBUG_TENSOR_DUMP_DIR` | str \| None | `None` | path | debug | 1Cat | Debug: DFlash v1 speculator (MRv2) tensor dump directory. |
| `VLLM_DFLASH_DEBUG_TENSOR_DUMP_LIMIT` | str \| None | `None` | count | debug | 1Cat | Debug: maximum tensor dumps. |
| `VLLM_DFLASH_DISABLE_AUX_OUTPUTS` | bool | `False` | 0, 1 | debug | 1Cat | Debug, MRv1 only: disable the auxiliary outputs for the draft. |
| `VLLM_DFLASH_DUMP_DRAFT_LOGITS` | bool | `False` | 0, 1 | debug | 1Cat | Debug, MRv1 only: dump draft logits. |
| `VLLM_DFLASH_DUMP_FIRST_PASS` | bool | `False` | 0, 1 | debug | 1Cat | Debug, MRv1 DFlash (v1) only: dump the first draft pass. |
| `VLLM_DFLASH_PROFILE` | bool | `False` | 0, 1 | debug | 1Cat | Debug: DFlash/DFlash2 proposal timing log. |
| `VLLM_DFLASH_PROFILE_LOG_INTERVAL` | int | `32` | steps | debug | 1Cat | Debug: interval of the timing log. |

### Deprecated, removed in 0.2.0rc6

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_DFLASH_DDTREE_ATTN_COMPACT_BATCH` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_COMPACT_DRAFTER_CONTEXT` | bool | `True` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_CONV_KERNEL` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_DEBUG` | bool | `False` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE` | bool | `False` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_ENGINE_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_FORCE_MAMBA_COMPACT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_FUSED_GDN` | bool | `True` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_GPU_SAMPLER` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_LINEAR_GDN` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_MAMBA_COMPACT_BATCH` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_METADATA_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE_LAYER` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE_MAX_REPORTS` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE_NODE_LIMIT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_QLA_GDN` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_SERIAL_GDN` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_SKIP_MAMBA_COMPACT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_STOCHASTIC_TOPK_LOGITS` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_TRACE_JSONL` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN` | bool | `True` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT` | bool | `False` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_TRITON_SAMPLER` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_CONTEXT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_TOPK` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DDTREE_WORKER_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree is retired. |
| `VLLM_DFLASH_DEBUG_COORD_TRACE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: glm5next is unsupported. |
| `VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: glm5next is unsupported. |

### speculative_config.sm70_dflash2 keys

Pass these inside `--speculative-config '{..., "sm70_dflash2": {...}}'`
(see [DFlash2](dflash2.md)). `None` selects the model-qualified policy;
operator shape, dtype and native checks still decide each dispatch.

| Key | Type | Default | Effect and trade-off |
|---|---|---|---|
| `context_kv_graph` | bool \| None | `None` | Capture the drafter's context-KV writes in CUDA graphs per token count instead of running them eagerly. |
| `context_pipeline` | bool \| None | `None` | Pipeline the drafter context-KV refresh with the verifier; accepted slots are refreshed before each replay. |
| `fused_gdn_metadata` | bool \| None | `None` | Build the GDN verify metadata with one fused launch. |
| `fused_smallq_metadata` | bool \| None | `None` | Fused Flash-V100 metadata for small-query verify batches. |
| `grouped_smallq_metadata` | bool \| None | `None` | Grouped small-query Flash-V100 metadata across requests. |
| `spec_state_replay` | bool \| None | `None` | Replay the accepted prefix to rebuild recurrent state instead of storing every speculative state: more GPU KV capacity at a small per-step cost. On by default in current development for supported ordinary DFlash2 routes; released 0.1.4–0.1.5 require explicit true. None retains an enabled packed verifier, and explicit false disables replay. See [State replay](dflash2.md#state-replay-optional). |
| `verify_fastpath` | bool \| None | `None` | Fast metadata path for verify steps in the hybrid (GDN + attention) model state. |

<!-- END GENERATED: tools/gen_env_docs.py -->

## See also

- [Reference launch tutorial](getting-started.md)
- [Checkpoint preparation](checkpoints.md)
- [KV offload modes](kv-offload.md)

### Junction retention evidence

`kernel_config.long_context_kv.enable_junction_evidence` is a deployment boolean,
default `true` (introduced in 0.2, experimental). It applies when long-context
mode and `latest_only` are enabled. It requires registered content successors
for permanent junction retention. Set it to `false` to restore the legacy
latest-only policy. It does not change compilation or persisted page identity.
See [checkpoint retention](kv-offload.md#checkpoint-retention-evidence) for
migration rules, limits and counters.

## Host decode experiments

`REVOLTA_RESEARCH_HOST_DECODE_FAST=1` fuses three pairs of host-history decode kernels. `REVOLTA_RESEARCH_HOST_DECODE_FAST_ORDERED=1` (requires the parent switch) additionally keeps the fused mark kernel's tail claims ordered before selection claims, so refill pairing and cache evolution match the unfused baseline; it costs no extra kernel launches and no phase parallelism (ordering comes from per-phase mark bits and a second wanted list, not from scheduling), at the price of one extra int32-per-hot-slot scratch tensor allocated unconditionally. `REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED=1` fuses integer metadata conversion in the DCP-sharded shared-key scorer. All are experimental and default to `0`; GPU correctness and matched performance validation are required before deployment. Set either to `0` and restart to restore its original path. The switches act inside opaque QSA operators and do not change the torch compilation key or SSD cache identity.
