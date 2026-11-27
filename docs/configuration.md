# Configuration reference

This page is the reference for ReVolta 0.2.0 settings: what the ReVolta profile sets automatically, the launch options you may change, sampling defaults, memory budgets and every documented environment variable. For the installation and launch sequence, follow the [getting-started tutorial](getting-started.md). Its reference launch needs no hand-tuned values: the profile fills in the tuned defaults.

Terms used in the tables:

- **Scope.** *Deployment* settings are fixed at launch for the whole server: environment variables, command-line options and the fields of their JSON configs. *Request* settings are chosen per API request.
- **Status.** *Stable* settings are part of the supported forms. *Experimental* settings are off by default or outside release acceptance, and can change between releases. *Optional* features are off by default and do not affect serving while off. *Deprecated* settings still work and will be removed.
- **Since.** The first ReVolta release with the setting. *upstream* means inherited from vLLM; *1Cat* means inherited from 1Cat-vLLM.

## Environment settings

The engine turns on the ReVolta profile when all of these hold: every visible GPU is SM70 (V100), `--tensor-parallel-size 4`, the model has QSA selector metadata (Qwen3.8 Flash-Next) and `--dtype half`. The profile then sets tuned kernel switches and serving defaults for that server. A value you set yourself, as an environment variable or a launch option, always takes precedence.

To check it, look for `ReVolta profile {"enabled": true, ...}` in the startup log. The line lists the audited settings with their values and sources (`profile`, `environment` or `upstream_default`); not every setting the profile applies is in that list.

With the profile, the reference deployment needs only these environment settings. All are deployment scope.

| Name | Type | Default | Range | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `TILELANG_CACHE_DIR` | path | TileLang default, inside the container | directory | - | stable | Where TileLang keeps compiled kernels. Point it at persistent storage; otherwise a recreated container compiles them again. |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | float | `600` | seconds | 1Cat | stable | How long startup waits for the PLE worker. The reference deployment uses `1200`; raise it on slow storage. |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | float | unset (automatic) | GiB per TP rank | 1Cat | stable | Optional cap on pinned host memory for the FP8 PLE table. Rows above the cap stay on the GPU. See [System memory and storage](#system-memory-and-storage). |

The profile also sets these process settings, so you do not need to export them:

- `PYTORCH_CUDA_ALLOC_CONF` and `PYTORCH_ALLOC_CONF` to `expandable_segments:True` when neither is set. It skips this when a configured KV connector is not marked safe for expandable segments; the CPU offload connector `OffloadingConnector` is marked safe.
- `NCCL_GRAPH_REGISTER=0` whenever expandable segments are on and the variable is unset, also when you export the allocator setting yourself. Do not set it to `1` with expandable segments: NCCL graph registration assumes one physical allocation per buffer.
- `VLLM_SM70_AWQ_MOE_INT4_ZERO=1` for AWQ g32 checkpoints with compact MoE metadata (on by default), which stores the AWQ zero points as 4-bit values.

The full list of profile values is in [Values applied by the ReVolta profile](#values-applied-by-the-revolta-profile). Use other `VLLM_SM70_*` settings only for troubleshooting.

## ReVolta serving defaults

The profile applies these defaults to the OpenAI-compatible server (`vllm serve`) for the FP16 TP4 QSA model, with or without speculative decoding. All are deployment scope and new in 0.2.0. Explicit values take precedence.

| Option | Profile default | Effect |
| --- | --- | --- |
| `--enable-prefix-caching` | `true` | Reuses cached prompt prefixes. With speculative decoding, the hybrid model checks then select `--mamba-cache-mode align`. |
| `--max-num-seqs` | `8` | Maximum number of scheduled sequences. |
| `--max-num-batched-tokens` | `8192`, unless chunked prefill is disabled | Total token budget per scheduler step. |
| `--long-prefill-token-threshold` | `4896`, capped at `--max-model-len` | Caps one long prefill when other requests may need the step. |
| `--attention-backend` | `FLASH_ATTN_V100` | Selects the V100 FlashAttention backend for the target model. |
| `gdn_state_replay` in `--speculative-config` | `true` for MTP with prefix caching | Replays accepted GDN tokens instead of storing one state per draft position. |
| `num_speculative_tokens_per_batch_size` | `[[1,3,4],[4,8,3]]` for MTP with K=4 and prefix caching; `[[1,3,7],[4,4,3],[5,8,4]]` for DFlash2 with K=7 | Chooses the number of drafts from the number of scheduled requests. Each entry is `[first, last, drafts]`. See [Dynamic speculative decoding](dynamic-speculative-decoding.md). Other K values keep a fixed width. |
| `dflash2_lookup` | Enabled for the seven-token DFlash2 selector: `longest`, suffix 5–16, 65,536-token window, `first_token` gate, statistics on, one live request | Uses committed token history to supplement the neural proposal. See [Native history lookup](dflash2.md#native-history-lookup). |
| `sm70_dflash2` verifier options | `true` for the DFlash2 selector | Turns on `context_kv_graph`, `context_pipeline`, `verify_fastpath`, `fused_gdn_metadata`, `fused_smallq_metadata`, `grouped_smallq_metadata`, `sparse_target_rejection` and `spec_state_replay`. See [speculative_config.sm70_dflash2 keys](#speculative_configsm70_dflash2-keys). |
| `qsa_dcp_index_shard` in `--kernel-config` | `true` for DFlash2, and for MTP with Host KV (`long_context_kv.enabled`), when the model selects 512 index groups (Qwen3.8 Flash-Next) | Splits the target QSA indexer keys over all tensor-parallel ranks; it takes effect at DCP2. MTP with GPU KV keeps the replicated layout. It is part of the SSD page identity; see [Configure Host KV](kv-offload.md#configure-host-kv). |
| `VLLM_SM70_QSA_GROUPED_LPT`, `VLLM_SM70_GDN_SKINNY_MPLAN` | `1` | Longest-first grouped QSA work and row-count-based GDN input plans. |
| `VLLM_SM70_DFLASH2_FUSED_QKV_PACK` | `1` for DFlash2 | Fuses eligible post-convolution QKV packing. |

Omitted `--async-scheduling` is still resolved from the executor and the speculative settings. A dynamic schedule and MTP's GDN state replay select synchronous scheduling, which is the recommended setting. An explicit `--async-scheduling` keeps the existing compatibility checks. What async scheduling does to the GDN state depends on the deployment; see [DFlash2 scheduling](dflash2.md#scheduling).

### Mixed prefill scheduling

For AWQ or NVFP4 weights with MTP or DFlash2, the profile also sets these scheduler defaults. Explicit values take precedence.

| Option | Type | Profile default | Default without the profile | Since | Effect and trade-offs |
| --- | --- | ---: | ---: | --- | --- |
| `--prefill-pressure-token-budget` | integer tokens, ≥ 0 | `1024` with chunked prefill | `0` (off) | 0.1.0 | Caps the total prefill tokens in a step while decoding is runnable and at least `--prefill-pressure-threshold` prefills (default `1`) are running or waiting. Lower values protect decode latency; higher values finish long prompts sooner. |
| `--mixed-prefill-step-latency-ms` | float ms, ≥ 0 | `0` | `250` | 0.1.0 | `0` disables adaptive step latency control and keeps the standard CUDA graph capture sizes. A positive target adapts the prefill budget to measured GPU step time. |

### AWQ MTP4 wide verifier

The profile turns on a wider MTP verifier for one qualified form. All of these must hold:

- An AWQ g32 checkpoint with the Flash-Next shape (hidden size 2560, 512 experts, 10 experts per token, MoE intermediate size 640) on TP4.
- MTP with `"num_speculative_tokens": 4` and GDN state replay (on by default with prefix caching).
- DCP2, one data-parallel replica and synchronous scheduling.
- GPU KV: Host KV (`long_context_kv`) is off. A CPU KV pool or disk tier in `--kv-transfer-config` is allowed, so the [reference launch](getting-started.md#4-launch) qualifies.
- The interconnect resolves to PCIe switch P2P (`SM70 interconnect=pex_p2p`). NVLink and NCCL-only systems keep the standard verifier.

When all hold, it sets the following. Explicit values take precedence.

| Setting | Value | Effect |
| --- | --- | --- |
| `VLLM_SM70_MTP_GDN_NATIVE_VERIFY` | `1` | Native GDN verification and accepted-state replay for eligible batches. |
| `VLLM_SM70_QWEN38_NATIVE_ROUTER_TOPK` | `1` | Native router for 17–64 eligible FP16 router rows. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS` | `48` | Raises the grouped AWQ MoE row limit from 40 to 48. |
| `"sm70_gdn_input_v154"` in `--kernel-config` | `true` | Packed FP16 GDN input projections for 2–64 rows in FULL verification graphs, and the 3–4 row FULL decode route. It covers AWQ g32 and standard NVFP4 FP16 projections, not FP4-Local or NVFP4-QAD MXFP8 projections. |

These kernels change the reduction order, so floating-point rounding and near-tie token choices can differ from the standard verifier. To keep the standard verifier, set the three variables to `0`, `0` and `40` and add `"sm70_gdn_input_v154": false` to `--kernel-config`.

## Launch options

All options in this section are deployment scope.

### KV cache format

`--kv-cache-dtype` selects the target KV storage format.

| Value | Status | Storage format |
| --- | --- | --- |
| `auto` (default) | stable | Planar INT8 for the QSA target on SM70 (the reference setting). Planar E4M3 for an eligible QSA target on other CUDA devices. |
| `int8`, `int8_per_token_head` | stable | Planar INT8 with round-to-nearest-even and dynamic FP32 scales per token and head. For the QSA target, both names select the same format. |
| `fp8_per_token_head` | experimental | Planar E4M3FN with dynamic FP32 scales per token and head. No KV scale pack is needed. |
| `fp8`, `fp8_e4m3` | stable, frozen | Scalar E4M3 with calibrated per-layer K/V scales. It still works and is functionally tested, but it is no longer optimized (#348). On SM70 with FlashAttention-V100, `fp8` resolves to `fp8_e4m3`. Setup: [Scalar E4M3 KV](checkpoints.md#scalar-e4m3-kv-frozen). |
| `float16` | stable | Model-precision storage. Uses the most GPU memory per token. |

`auto` selects planar INT8 only when the model has QSA selector metadata, `--calculate-kv-scales` is not set and prefill context parallelism is 1. Otherwise the generic vLLM rules apply. Planar formats use four page regions `[K | V | sK | sV]`, not upstream vLLM's inline FP8 page layout. The old name `fp8_e4m3_per_token_head` is removed without an alias.

Draft KV:

- **MTP** inherits the target format, with GPU KV and with Host KV. To override it, set `"kv_cache_dtype"` in `--speculative-config`; `"float16"` keeps an FP16 draft with an 8-bit target, and the INT8 names are accepted when the target uses INT8. Host KV placement of the draft is described in [KV offload](kv-offload.md).
- **DFlash2** with the SM70 QSA target uses FP16 draft KV by default, with GPU KV and with Host KV.

### Speculative decoding

These fields go in `--speculative-config`.

| Field | Type | Default | Range | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `draft_sample_method` | string | `greedy` | `greedy`, `probabilistic` | 1Cat | stable | How the drafter picks proposals. MTP must stay `greedy`: a probabilistic MTP draft corrupts output above `max_num_seqs=2` at temperature above 0 (1CatAI/1Cat-vLLM#656). For DFlash2, `probabilistic` is recommended; it passed the 0.1.4 release gate at 1–8 concurrent requests, temperature 1 with thinking, tool calls and structured output. This setting affects drafting and draft verification only; clients can still use any sampling parameters. |
| `num_speculative_tokens_per_batch_size` | list of `[first, last, drafts]` | profile default above | inclusive ranges of the scheduled request count | 0.2.0 | stable | Chooses the number of drafts from the number of scheduled requests. See [Dynamic speculative decoding](dynamic-speculative-decoding.md). |
| `kv_cache_dtype` | string | unset (inherit) | see [KV cache format](#kv-cache-format) | upstream | stable | Draft KV format; see the draft KV notes above. |

DFlash2 options are in [DFlash2 speculative decoding](dflash2.md) and [speculative_config.sm70_dflash2 keys](#speculative_configsm70_dflash2-keys).

### Memory and capacity

| Option | Type | Reference value | Default | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `--gpu-memory-utilization` | float | `0.956` | `0.92` | upstream | stable | Share of each GPU's memory the engine may use; the rest of the reserve sizes the KV cache. Keep the GPUs free of other processes that use GPU memory. 0.2.0 acceptance runs at `0.956` without multimodal stress. Measure again if you raise concurrency or media limits. |
| `cpu_bytes_to_use` in `--kv-transfer-config` | integer bytes | `51539607552` (48 GiB) | none | upstream | stable | Size of the CPU KV pool, summed over all four workers (48 GiB = 12 GiB per worker). It needs host memory on top of the PLE table; see [System memory and storage](#system-memory-and-storage) and [KV offload](kv-offload.md). |

### Multimodal input

| Option | Type | Reference value | Default on V100 | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `--limit-mm-per-prompt` | JSON | `{"image": 16, "video": 1}` | 1 image, no video | upstream | stable | Media items per request. With FlashAttention-V100, the engine allows 1 image and no video when the option is omitted, and a value without `"video"` disables video. Qwen3.8 Flash-Next takes images and video, not audio, so no audio limit is set. `{"image": 8, "video": 1}` matches the generic vLLM image count; a lower image count leaves more host memory for bursts of large images. |

Large images are resized either way; see [Support and known limits](known-limits.md#multimodal-input). Media limits and request budgets are in [multimodal input limits](../.github/MULTIMODAL_INPUTS.md).

### API server

| Option | Type | Default | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- |
| `--enable-dashboard` | boolean | `false` | 0.2.0 | optional | Serves the built-in aggregate metrics page at `/dashboard/`. Off by default. Rate and window calculations run in the browser, with optional local history; no engine background collector is added. Dashboard data follows the API-key configuration. Tested with CPU-only suites; its cost while serving on GPUs has not been measured. See [Dashboard](dashboard.md). |
| `--structured-outputs-config` | JSON | `{"backend": "auto"}` | upstream | stable | `{"backend": "xgrammar"}` pins the structured-output backend. With `auto`, the engine keeps the backend it chose for the first structured-output request. |
| `--sse-keep-alive-interval` | integer seconds, ≥ 0 | `0` (off) | 0.1.2 | stable | Sends an SSE comment line (`: keep-alive`) after this many idle seconds on streaming responses, so proxies with read timeouts keep the connection open during queueing and prefill. The reference launch uses `15`. |
| `--sanitize-control-tokens` | boolean | `false` | 0.2.0 | stable | See [Chat control tokens](#chat-control-tokens-and-the-reasoning-end-of-turn-mask). |

### Chat control tokens and the reasoning end-of-turn mask

`--sanitize-control-tokens` (boolean, default `false`, deployment scope) encodes control tokens that appear inside chat messages, tool results or documents as plain text: the special tokens and the added tokens the chat template uses. Only the chat template can then emit them, so a user cannot inject a control token by writing it literally. It applies to the Hugging Face chat template path. A tokenizer that does not forward the option, for example the Mistral tokenizer, ignores it. The setting cannot be turned off per request. Because sanitizing happens while the rendered chat template is tokenized, a request with multimodal input and the `string` chat template content format is rejected with a validation error. See [#157](https://github.com/Leonccaa/ReVolta/issues/157).

`VLLM_QWEN_REASONING_EOS_MASK` (boolean, default `0`, deployment scope, since 0.2.0) suppresses the model's end-of-turn token while a request is still inside its reasoning block. It stops a Qwen chat model that writes the end-of-turn marker in its reasoning from ending the turn with reasoning only and no answer. Set `1` to enable it. It acts only when the request asks for reasoning content, uses the Qwen3 reasoning parser with thinking enabled, runs on Model Runner V2 and has a prompt that already contains an unclosed thinking start tag. A template that lets the model write the thinking start tag itself is not masked at the start of the request. See [#157](https://github.com/Leonccaa/ReVolta/issues/157).

### Shutdown and diagnostics

| Option | Type | Default | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- |
| `--shutdown-timeout` | integer seconds, ≥ 0 | `0` | upstream | stable | With `0`, a stop aborts in-flight requests and stops the engine at once, and the host KV SSD final flush is skipped. A positive value lets in-flight requests run for up to that many seconds (with host KV, that value minus 5 s), then aborts the rest. With the host KV SSD tier, the rest of that window is used for a best-effort final flush; pages not written are misses on the next start. The reference launch uses `30`. |
| `--shutdown-reap-timeout` | float seconds, ≥ 0 | `--shutdown-timeout` | 0.2.0 | stable | Extra wait for processes to exit after a worker or the local engine core reports that it has retired. Capped by `--shutdown-timeout`; `0` disables the wait. It does not change request draining or SSD flush deadlines. Missing reports still lead to signal escalation. |
| `--watchdog-config` | JSON | watchdog off | 0.1.4 | stable | Stack-dump watchdog (#33). Fields: `dump_dir` (empty default disables it), `timeout` (seconds without progress, default `300`, greater than 0) and `check_interval` (seconds between checks, default `10`). |

Keep the container stop grace period (for example `docker stop --time 120`) above `--shutdown-timeout` plus `--shutdown-reap-timeout`.

When the watchdog is enabled, a background thread runs in the engine core process (`engine_<dp_rank>`) and in each worker (`worker_<rank>`). If that process makes no progress for `timeout` seconds, it appends a numbered dump of all its Python thread stacks to `<dump_dir>/VLLM_STACK_DUMP_for_<name>_<pid>.log`, at most once per stall, and logs `[Watchdog]Dumped stack to ...`. The engine core also dumps its stacks on SIGTERM. Waiting for incoming work does not count as a stall. The watchdog only records stacks: it does not restart or kill the process. Example:

```bash
--watchdog-config '{"timeout": 300, "check_interval": 10, "dump_dir": "/data/revolta/dumps"}'
```

## Interconnect

The engine selects the interconnect automatically and each worker logs `SM70 interconnect=<mode>`: `pex_p2p` (all GPUs under one PCIe switch with P2P), `nvlink` or `no_p2p`. On a PCIe switch, custom all-reduce runs inside CUDA graphs and larger payloads use NCCL.

| Name | Type | Default | Range | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `VLLM_SM70_INTERCONNECT` | string | `auto` | `auto`, `pex`, `nccl`, `nvlink` | 0.1.0 | stable | `auto` detects NVLink, then a PCIe switch topology (PIX/PXB) with a fresh P2P check. `pex` skips the topology query but still checks P2P. `nccl` disables custom all-reduce. `nvlink` forces the fully connected path and still checks P2P. `--disable-custom-all-reduce` overrides all of them. |
| `VLLM_SM70_PCIE_SWITCH_CAR_MAX_BYTES` | integer bytes | `327680` (320 KiB) | ≥ 0 | 0.1.0 | stable | Largest payload the PCIe switch custom all-reduce handles; larger payloads use NCCL. The default covers 42–64 rows of width 2560 in FP16. `0` removes the byte cap but keeps the other custom all-reduce limits. |
| `VLLM_SM70_PCIE_SWITCH_CAR_GRAPH_ONLY` | boolean | `1` | `0`, `1` | 0.1.0 | stable | `1` uses custom all-reduce only inside CUDA graphs; eager calls use NCCL. With `0`, eager calls also use it, within the same byte cap. |
| `VLLM_SM70_NVLINK_HC` | boolean | `0` | `0`, `1` | 0.2.0 | experimental | Flash-Next HC sharding, fused all-reduce and HC, and the QSA DCP2 query exchange and output merge over NVLink. All six GPU pairs must have direct NVLink and native peer atomics; `VLLM_SM70_INTERCONNECT=nvlink` does not bypass this check. Asynchronous scheduling turns it off. The QSA route takes FP16 queries with six 256-element heads per rank and up to 64 rows. It can change floating-point rounding. |
| `VLLM_SM70_DCP2_P2P` | boolean | `0` | `0`, `1` | 0.2.0 | experimental | QSA DCP2 query and packed output/LSE exchange through the registered PCIe switch push buffers, with the A2A DCP backend (the Flash-Next default). It covers eager uniform decode and dedicated FULL decode graphs up to 53 rows; PIECEWISE graphs keep NCCL. In 0.2.0 tests with four GPUs and AWQ, it gave no stable gain in cold time to first token or decode, and fixed-prompt outputs matched the NCCL path token for token. Needs the native extension from the same build. |

## Optional MTP draft vocabulary (experimental)

Model Runner V2 can restrict MTP drafting to a static vocabulary shortlist, optionally with a global tail of recently used tokens. It is off by default and outside release acceptance. It needs a ranking file that matches the model vocabulary and TP size; this repository does not include a ranking file or a tool that builds one. All settings are deployment scope.

| Name | Type | Default | Range | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING` | path | unset (off) | ranking file (`.pt`) | 1Cat | experimental | Enables the static shortlist. Greedy and probabilistic proposals are mapped back to full-vocabulary token IDs. |
| `VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE` | integer | `0` | > 0 when enabled, for example `65536` | 1Cat | experimental | Shortlist size. |
| `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE` | integer | `0` (off) | 1–2048; 512 or 1024 to start | 1Cat | experimental | Model Runner V2 support since 0.2.0. Adds a global LRU tail T shared by all concurrent requests in the worker group, updated from accepted, recovered and bonus output tokens. Each rank holds T FP16 weight rows: `2*T*H` bytes, 2.5 MiB at T=512 and 5 MiB at T=1024 with H=2560. Probabilistic drafting adds `4*max_num_seqs*K*(B+TP*T)` bytes for processed logits, where B is the shortlist size. |

The tail requires `"method": "mtp"`, an enabled static shortlist, an FP16 LM head, no logits soft cap, scale 1, at least T eligible seed tokens after the shortlist and a model padding token outside the shortlist. Unsupported configurations raise `ValueError` at startup. `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU`, `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FUSED_PROPOSAL`, `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FULL_REFRESH_INTERVAL` and `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_PREFILL_TOPK` apply to Model Runner V1 only; setting any of them with Model Runner V2 is an error, even to zero.

To confirm activation, look for `Enabled static MTP draft vocabulary` and, with a tail, `Enabled global MTP draft vocabulary tail: base=... tail=... tp=... sink=... draft_sample=...` on every TP rank, with the configured sizes. Measure acceptance length, round time and throughput on your workload before choosing a size.

## Sampling parameters

These are request-scope settings. The server takes default sampling values from the checkpoint's `generation_config.json` (`temperature` 1.0, `top_p` 0.95, `top_k` 20), but only for fields that a request leaves out. Many chat clients send their own `temperature` and `top_p`, so check the client settings. Qwen recommends:

| Mode | `temperature` | `top_p` | `top_k` | `min_p` | `presence_penalty` | `repetition_penalty` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Thinking (default) | 1.0 | 0.95 | 20 | 0.0 | 0.0 | 1.0 |
| Non-thinking | 0.7 | 0.80 | 20 | 0.0 | 1.5 | 1.0 |

- **Endless repetition:** Qwen suggests a `presence_penalty` between 0 and 2. Higher values can mix languages and slightly lower quality.
- `presence_penalty` can only be set per request. `--override-generation-config` sets server defaults for `temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty` and `max_new_tokens` only.
- `top_k` and `min_p` are not standard OpenAI fields. With the OpenAI Python SDK, pass them in `extra_body`.
- These are the model vendor's values. The `draft_sample_method` in the launch command applies only to drafting and draft verification, not to these request settings.

## YaRN and native context

YaRN in vLLM is static: it applies the same scaling to every request, short or long. Qwen's model card notes that this can lower quality on shorter texts and advises enabling YaRN only when long contexts are needed. ReVolta's quality measurements compare against a BF16 reference that also uses YaRN, so they do not show this effect. Release acceptance runs with the 512K views.

If your requests stay within 262,144 tokens, do not make YaRN views: use `--max-model-len 262144` and serve the checkpoint directory itself (the materialized directory for Standard NVFP4 and FP4-Local). For the MTP draft, use the AWQ checkpoint's own `mtp-fp8/`, the scale pack's `mtp-fp8/` for Standard NVFP4, or FP4-Local's built-in MTP layer with no draft `model` field.

## System memory and storage

**128 GiB of RAM is recommended** for the default 48 GiB CPU KV pool. The AWQ PLE table uses about 47.7 GiB of pinned host memory even with KV offload disabled.

Automatic FP8 PLE placement pins the whole table in host memory when the disk cascade is inactive. Startup reports an error if available host memory cannot hold the table across local ranks plus the host reserve. Free host memory, or set `VLLM_QWEN4EXP_PLE_HOST_GIB` to a smaller per-rank cap to keep the remaining rows on the GPU. FP4-Local keeps its full NVFP4 PLE table in host memory, about 26.8 GiB across four ranks.

| Container memory limit | KV offload | Result | Release |
| --- | --- | --- | --- |
| 128 GiB | Off, or 48 GiB CPU pool | Full acceptance passed | 0.1.0 |
| 80 GiB | Off | Memory stress passed; full acceptance not run | 0.1.0 |
| 64 GiB | Off | Out of memory during startup | 0.1.0 |
| 64 GiB, whole PLE table on disk (experimental) | Off, or 32 GiB CPU pool | Served; greedy output matched the pinned path | 0.1.4 |
| 32 GiB, whole PLE table on disk (experimental) | Off | Served; greedy output matched the pinned path | 0.1.4 |

With the default pinned PLE table, 80 GiB is the smallest tested limit that passed memory stress, with KV offload off and without full acceptance. With less than 128 GiB, reduce or disable the CPU KV pool. The experimental whole-table PLE disk option ([Support and known limits](known-limits.md#memory-and-host-requirements)) keeps the table on a local SSD instead and brings the tested minimum down to 32 GiB. In 0.1.4, single-request decoding was about 23–25% slower with it and a cold first token took 1.6–2.3 times as long.

Image and video preprocessing also uses host memory, in the API server process. In a 0.1.1 test, 8 concurrent requests with 8 different 2048-pixel images each used about 33 GB at peak. With a 32 GiB CPU KV pool on a 125 GiB host, the kernel then killed a worker. The same burst passed with a 16 GiB pool. If you expect bursts of large images or videos, use a smaller CPU KV pool or limit the image size and the number of concurrent requests.

The AWQ checkpoint is 137 GB at the pinned revision:

| Part | Format | Size | Where it runs |
| --- | --- | ---: | --- |
| Routed experts | 4-bit AWQ, group size 32 | 74.9 GB | GPU |
| PLE n-gram tables | FP8 E4M3 | 51.3 GB | Pinned host memory |
| Other weights (attention, embeddings, `lm_head`, vision tower, shared experts) | 16-bit | 10.9 GB | GPU |
| Optional MTP draft (`mtp-fp8/`) | FP8 | 5.3 GB | GPU |

Allow extra disk space for compilation caches and the optional disk KV tier. Structured long-context Host KV settings (`kernel_config.long_context_kv.*`) are in [KV offload](kv-offload.md).

<!-- BEGIN GENERATED: tools/gen_env_docs.py -->
<!-- Edit tools/env_docs_notes.py, then run python tools/gen_env_docs.py -->

## Environment variable reference

Every variable below is read once per process (deployment scope). Columns: type and default as registered in `vllm/envs.py`; *Since* is the first release with the variable (1Cat = inherited from 1Cat-vLLM); *Category* is configuration, tuning, experimental, debug or deprecated.

### Kernel and path switches

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_SM70_CHECKPOINT_CACHE_DROP` | bool | `True` | 0, 1 | configuration | 0.2.0 | Default on; engine startup only. 1 requests clean checkpoint cache eviction before host KV allocation, after a world load barrier. Skips live mappings and PLE disk/cascade modes. May slow later reloads or other engines reading the same checkpoint. See startup logs for advised files/bytes and cgroup usage; no metric. |
| `VLLM_SM70_GDN_CPU_SPEC_QSL` | bool | `False` | 0, 1 | tuning | 0.2.0 | Removes two host synchronizations per step in mixed prefill + speculative batches; costs a host-side offset computation. Outputs are bitwise identical either way. |
| `VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE` | str | `''` | path | debug | 0.2.0 | Validation only: adds the synchronizations back while the file exists. Never set for timing runs. |
| `VLLM_SM70_GDN_NATIVE_PREFILL` | bool | `True` | 0, 1 | tuning | 0.2.0 | Use the native SM70 GDN kernel by default for eligible original TileLang chunks on Model Runner V2 only. It can change floating-point results and reserves shared GPU scratch space. Set 0 to use TileLang instead. Model Runner V1 is unaffected. |
| `VLLM_SM70_HOST_DECODE_FAST` | bool | `True` | 0, 1 | tuning | 0.2.0 | Default on. Long-context host KV decode only: fuses three pairs of host-history decode kernels. Set 0 and restart to restore the original kernels; ordinary GPU KV paths are unaffected. |
| `VLLM_SM70_MTP_KV_ONLY_PREFILL` | bool | `True` | 0, 1 | tuning | 0.2.0 | For eligible Flash-Next MTP batches with DP=1 and GPU or Host draft KV, eager incomplete prefills only write KV. If a prefill completes, the sampled-row path writes KV for all rows and computes outputs only for sampled rows outside FULL CUDA graph replay. Floating-point results can differ from the full-row path. Set 0 to use full-row draft prefill. |
| `VLLM_SM70_QSA_LOCAL_INPUTS_FUSED` | bool | `True` | 0, 1 | tuning | 0.2.0 | Default on. DCP-sharded shared-key scorer: one fused integer metadata conversion instead of the torch reference. Set 0 and restart to restore the torch reference. |
| `VLLM_SM70_QSA_SHORT_APPEND` | bool | `True` | 0, 1 | tuning | 0.2.0 | Short warm appends can use SM70 QSA cuBLAS scoring. The single-request row floor drops from 512 to 128 at 4096 aligned routing columns, and mixed batches can split requests with at least 128 rows at 16384 tokens or more. The score work threshold still applies. Near-tied block selection may differ from the paged scorer. Set 0 and restart to restore the original routing. |

### Deployment settings

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_1CAT_ADAPTIVE_MTP` | str | `off` | off/0/false/no, on/1/true/yes, auto | experimental | 0.2.0 | Opt in to adaptive depth for synchronous Flash-Next MRv2 MTP with a dynamic speculative decoding schedule. Explicit auto also requires SM70 and a configured maximum depth of at least five. Single-request decode may trade fewer draft steps for more target steps. |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | float | `600.0` | seconds | configuration | 1Cat | How long startup waits for the PLE offload worker. Raise it on slow storage; the reference deployment uses 1200. |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | float \| None | `None` | GiB per TP rank; unset = automatic | configuration | 1Cat | Caps pinned host memory for FP8 PLE. Automatic placement pins the full table when the disk cascade is inactive. An explicit cap leaves remaining rows on the GPU in this mode. |
| `VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB` | float \| None | `None` | non-negative GiB or unset | configuration | 1Cat | Unset reserves min(host limit / 4, 8 GiB), preserving the small-host margin without scaling it to tens of GiB. Explicit values override the cap. The warning-only preflight counts shared CPU KV storage separately; automatic placement also uses this reserve. |
| `VLLM_QWEN_REASONING_EOS_MASK` | bool | `False` | 0, 1 | configuration | 0.2.0 | Suppresses model EOS during visible Qwen chat reasoning in Model Runner V2. Requests still in reasoning use full-vocabulary sampling when compact DFlash2 verification would truncate logits. |
| `VLLM_SM70_AWQ_MOE_INT4_ZERO` | bool | `False` | 0, 1 | experimental | 0.1.0 | Packs AWQ zero points into nibbles: less resident metadata, lossless. Needs compact metadata and a native build with the packed kernels. Under vllm serve the ReVolta profile sets 1 for AWQ g32 checkpoints with compact metadata (on by default). |
| `VLLM_SM70_FLASH_ATTN_V100` | bool | `True` | 0, 1 | configuration | 1Cat | Default on (set in the image). 0 only to diagnose backend selection; prefer --attention-backend to choose a backend. |
| `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` | bool | `False` | 0, 1 | configuration | 1Cat | Set in the image: VLLM_COMPILE + FULL_AND_PIECEWISE with small decode captures (the 0.0.3 graph policy). |
| `VLLM_SM70_HOST_KV_STATIC_HOT_POOL` | bool | `False` | 0, 1 | configuration | 0.2.0 | Long-context host placement only. 1 restores the separate static hot-page pool (history IDs capped by host pages); the default shares free history-ID index pages with the hot slots. |
| `VLLM_SM70_IDLE_CLOCK_SECONDS` | int | `0` | non-negative integer seconds | configuration | 0.2.0 | Zero disables clock control. A positive value opts in for a single-host EngineCore process with exclusive V100 TP1 or TP4 GPUs. Requires NVML clock permissions and host-side reset by GPU UUID on abnormal exit before the GPUs can be reused. |
| `VLLM_SM70_MTP_PLE_CONV` | bool | `True` | 0, 1 | configuration | 1Cat | Fuses SM70 PLE speculative rollback, convolution, SiLU and state update for eligible MTP and DFlash2 batches. Set 0 to use the separate operations. |
| `VLLM_SM70_PLE_GATE_NORM` | bool | `False` | 0, 1 | experimental | 0.2.0 | Opt-in FP16 PLE gate and norm fusion in SM70 Flash-Next full decode graphs with dual compilation. The fused reduction can change FP16 results. |
| `VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD` | bool | `False` | 0, 1 | configuration | 1Cat | Diagnostic: compare the split GDN compile boundary with the full-forward guard. The image sets 0. |
| `VLLM_SM70_QWEN_GDN_FULL_FORWARD` | bool | `False` | 0, 1 | configuration | 1Cat | Diagnostic force-on of the opaque GDN full-forward boundary. The image sets 0; the automatic default arms it only for active MTP batches. |
| `VLLM_SM70_QWEN_GDN_SPEC_CORE_OP` | bool | `False` | 0, 1 | experimental | 1Cat | Experimental narrower MTP boundary (only the recurrent core is opaque). The image sets 0 until it passes long-output quality. |

### Values applied by the ReVolta profile

The ReVolta profile (`vllm/revolta_profile.py`) sets these on SM70 + TP4 + FP16 + QSA models when they are unset; explicit values win. The profile column shows the value it applies (AWQ-only and NVFP4-only entries apply to that checkpoint family only).

| Name | Type | Default | Profile | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|---|
| `VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS` | bool | `False` | `1` | 0, 1 | configuration | 1Cat | Opts out of the 1Cat SM70 MTP launch defaults; the ReVolta profile supplies its own. |
| `VLLM_EARLY_MODEL_INPUTS` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Prepares model-state inputs before attention metadata in FULL graph steps, overlapping host work. |
| `VLLM_PLE_ASYNC_REQUEST_INDICES` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Non-blocking PLE request-index transfers; removes a host wait per step. |
| `VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP` | bool | `False` | `0`, `1` (AWQ+DFlash2) | 0, 1 | configuration | 1Cat | Allows shared and routed experts to run on separate CUDA streams. The native g32 AWQ profile enables it for DFlash2 with expert parallelism and dual batch overlap disabled. An explicit 0 or 1 takes precedence. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN` | bool | `False` | `1` (AWQ) | 0, 1 | experimental | 0.1.0 | Grouped AWQ MoE decode reads each active expert once for all its rows (2..48). Faster at 2..8 requests; reduction order differs from the TurboMind route. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS` | int | `20` | `40` (AWQ), `48` (AWQ MTP4 wide verifier) | 2..48 | tuning | 0.1.0 | Largest row count sent to the grouped route; above it rows use TurboMind. The wide verifier profile raises the AWQ cap from 40 to 48. DFlash2 uses its dedicated row cap. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_PACK_ROWS` | int | `0` | `24` (AWQ) | 0 or 8..48 step 8 | tuning | 0.1.0 | Caps rows per expert pack: fewer registers per CTA but hot experts are read more than once. Result unchanged. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT` | int | `16` | `8` (AWQ) | 4, 8, 16 | tuning | 0.1.0 | W13 split-K warps per CTA. 16 matches the M1 arithmetic; 8 (profile) uses fewer threads and a different FP32 split association. |
| `VLLM_SM70_AWQ_QWEN38_QPN_M1` | bool | `True` | `1` (AWQ) | 0, 1 | configuration | 1Cat | Single-token W13/W2 QPN route; 0 restores the legacy route. Changes reduction order. |
| `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES` | bool | `False` | `1` (AWQ) | 0, 1 | configuration | 1Cat | Tunes TurboMind small-M AWQ shapes at startup: longer warmup, faster small batches. |
| `VLLM_SM70_CAR_VMM_SAFE` | bool | `False` | `1` (P2P/NVLink) | 0, 1 | experimental | 0.1.0 | Copies non-exportable (VMM) inputs into graph allocations while capturing custom all-reduce graphs, so expandable segments can stay on. Only with P2P or NVLink. |
| `VLLM_SM70_DFLASH2_FUSED_QKV_PACK` | bool | `False` | `1` (DFlash2, vllm serve) | 0, 1 | configuration | 1Cat | One Triton launch instead of three copies and a concat in the DFlash2 verifier. Outputs are bitwise identical either way. |
| `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` | bool | `True` | `0` (DFlash2) | 0, 1 | configuration | 1Cat | Single-request DFlash2 verifier tail graphs. The DFlash2 profile sets 0 to keep the graph memory for the KV cache. |
| `VLLM_SM70_GDN_SKINNY_MPLAN` | bool | `False` | `1` (vllm serve) | 0, 1 | tuning | 0.2.0 | ng=2 tiling is 8-26% faster at M = 9..24 and 33..48 on V100; other M keep the fixed plan. Only with VLLM_SM70_QWEN38_GDN_NO_PACK. Outputs are bitwise identical either way. |
| `VLLM_SM70_GREEDY_VERIFY` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Greedy MTP verification from TP-local argmax pairs instead of full-vocabulary logits; only when every request is plain greedy. |
| `VLLM_SM70_MTP_GDN_NATIVE_VERIFY` | bool | `False` | `1` (AWQ MTP4 wide verifier) | 0, 1 | configuration | 0.2.0 | Uses the native recurrent verifier and accepted-state replay in the AWQ MTP4 wide verifier profile. Set 0 to disable. |
| `VLLM_SM70_MTP_Q8_PREFILL_MMQ` | bool | `False` | `1` (NVFP4) | 0, 1 | configuration | 0.2.0 | Use grouped GGUF MMQ for Q8_0 MTP batches above 64 rows in chunks of up to 2048. MMQ changes accumulation order and increases temporary GPU memory usage. Set 0 to use the 64-row vector path. |
| `VLLM_SM70_NVFP4_MOE_GROUPED_DECODE` | bool | `False` | `0` (NVFP4) | 0, 1 | experimental | 1Cat | Prepared-scale native-NVFP4 decode for M = 2..64; set VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=0 (the profile defaults to 1). The profile keeps grouped decode off pending quality admission. |
| `VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | One grouped TurboMind call for all routed experts in prefill. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 1Cat | QPN for every live width 2..16 (NVFP4). |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 0.1.0 | Graph-safe raw-E4M3 grouped NVFP4 MoE for M = 2..40; needs RAW_SCALE storage. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 1Cat | Keeps E4M3 block-scale codes resident instead of a second FP16 copy: less memory, decode reconstructs exact values. |
| `VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | Tunes small NVFP4 shapes at startup. |
| `VLLM_SM70_NVFP4_TURBOMIND` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | TurboMind route for NVFP4/MXFP4 dense layers; 0 only for diagnosis. |
| `VLLM_SM70_PLE_INPUT_PREPARE` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | One fused launch for PLE n-gram context gathering (<=4 requests). |
| `VLLM_SM70_PLE_NGRAM_BATCH` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Fused PLE n-gram id preprocessing for 2..32 rows. |
| `VLLM_SM70_PLE_PREFILL_CONV_CHUNK` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Bounds PLE short-conv prefill buffers when several prefills share a step; needed at max_num_seqs=8. Outputs are bitwise identical either way. |
| `VLLM_SM70_QSA_COMPACT_TOPK` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Coarse FP16 pre-selection before the exact QSA top-k for <=32 rows and long contexts; same indices. |
| `VLLM_SM70_QSA_GROUPED_LPT` | bool | `False` | `1` (vllm serve) | 0, 1 | tuning | 0.2.0 | Longest grouped page4 CTAs start first, which shortens the tail of uneven decode batches. Outputs are bitwise identical either way. |
| `VLLM_SM70_QSA_REQUEST_SEGMENTS` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Reads each 64-key tile once per request segment in multi-request decode. Outputs are bitwise identical either way. |
| `VLLM_SM70_QSA_SHARED_KEY` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Shared-key FP16 indexer for M2..8 verification queries of one request. |
| `VLLM_SM70_QWEN38_GDN_NO_PACK` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Skips the packed GDN weight copy (725.6 MiB per TP4 rank) and serves 2..48 rows with the skinny dual GEMM; frees memory for KV. |
| `VLLM_SM70_QWEN38_HYBRID_PLE` | bool | `False` | `0` | 0, 1 | configuration | 1Cat | Local pinned PLE shards for decode. The profile sets 0 (single PLE placement) on the reference deployment. |
| `VLLM_SM70_QWEN38_NATIVE_ROUTER_TOPK` | bool | `False` | `1` (AWQ MTP4 wide verifier) | 0, 1 | experimental | 0.1.0 | Uses the native router for eligible 17–64 row FP16 logits in the AWQ MTP4 wide verifier profile. Set 0 to disable. |
| `VLLM_SM70_QWEN38_WIDE_M` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Widens Qwen3.8 decode/verify routes from 4 to 8 requests (M up to 40); each VLLM_SM70_QWEN38_WIDE_M_* item can override it. |
| `VLLM_SM70_SKIP_REJECTED_BATCH_PACKS` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Skips batch packs the precision policy never admits; saves about 199 MiB per rank. |
| `VLLM_USE_AOT_COMPILE` | bool | `False` | `0` | 0, 1 | configuration | 1Cat | AOT compile reuse. The profile sets 0 on the reference deployment. |
| `VLLM_V2_CUDAGRAPH_MEM_MIB` | float | `0.0` | `700` (AWQ), `700` (NVFP4), `1200` (DFlash2), `1000` (AWQ+DFlash2, planar KV), `1150` (AWQ+DFlash2, planar KV, positive mixed-prefill target), `1024` (NVFP4+DFlash2, planar KV), `600` (fixed MTP, planar KV), `705` (dynamic MTP, planar KV), `800` (dynamic MTP with 5+ drafts, planar KV) | MiB | configuration | 1Cat | CUDA graph memory reserved before sizing the KV cache (MRv2 does not profile it). Too low risks OOM at capture; too high wastes KV. Profile: 700 (AWQ, NVFP4), 1200 (DFlash2); with fp8_per_token_head or int8 KV, 600 (fixed MTP), 705 (dynamic MTP), 800 (dynamic MTP with 5 or more drafts), 1000 (AWQ + DFlash2; 1150 with a positive --mixed-prefill-step-latency-ms) and 1024 (NVFP4 + DFlash2). See [CUDA graph reserve](checkpoints.md#cuda-graph-reserve). |

### Experimental switches (off by default)

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_SM70_NVFP4_RAW_GROUPED_M48_V154` | bool | `False` | 0, 1 | experimental | 0.2.0 | Native raw-E4M3 grouped MoE for 41..64-row target decode on SM70 TP4 Qwen3.8 with equal interleaved gate/up global scales. Requires raw grouped decode and the V2 model runner. |
| `VLLM_SM70_NVFP4_RAW_GROUPED_NARROW_V154` | bool | `False` | 0, 1 | experimental | 0.2.0 | Native raw-E4M3 grouped MoE for 3-5 row target decode on SM70 TP4 Qwen3.8 with equal interleaved gate/up global scales. Requires raw grouped decode and the V2 model runner. |

### Anonymous usage statistics

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_DO_NOT_TRACK` | bool | `False` | 0, 1 | configuration | 1Cat | Honours the cross-project Do Not Track convention (DO_NOT_TRACK counts too): 1 turns the usage report off. |
| `VLLM_NO_USAGE_STATS` | bool | `False` | 0, 1 | configuration | 1Cat | Upstream opt-out, honoured as-is: 1 turns the usage report off. The Do Not Track variables and the do_not_track file have the same effect. |
| `VLLM_USAGE_SOURCE` | str | `'production'` | short label | configuration | 1Cat | Deployment label reported with the anonymous report (production by default). A value that looks like a path or a URL is reported as custom instead. |
| `VLLM_USAGE_STATS_SERVER` | str | `'https://usage.19b.net/api/'` | URL, empty to disable sending | configuration | 1Cat | Where the report goes. ReVolta defaults to its own collector; an empty string keeps the local log file but sends nothing. |

### DFlash2 legacy lookup and debug settings

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

### Deprecated variables

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `REVOLTA_QSA_GROUPED_LPT` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_QSA_GROUPED_LPT from the 0.2.0 pre-releases. Still accepted with a FutureWarning; the new name wins when both are set. It will be removed in a later release. |
| `REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_CHECKPOINT_CACHE_DROP from the 0.2.0 pre-releases. Still accepted with a FutureWarning; the new name wins when both are set. It will be removed in a later release. |
| `REVOLTA_RESEARCH_HOST_DECODE_FAST` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_HOST_DECODE_FAST from the 0.2.0 pre-releases. Still accepted with a FutureWarning; the new name wins when both are set. It will be removed in a later release. |
| `REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_QSA_LOCAL_INPUTS_FUSED from the 0.2.0 pre-releases. Still accepted with a FutureWarning; the new name wins when both are set. It will be removed in a later release. |
| `REVOLTA_SKINNY_MPLAN` | bool | `False` | 0, 1 | deprecated | 0.2.0 | Deprecated alias of VLLM_SM70_GDN_SKINNY_MPLAN from the 0.2.0 pre-releases. Still accepted with a FutureWarning; the new name wins when both are set. It will be removed in a later release. |
| `VLLM_DFLASH_DDTREE_ATTN_COMPACT_BATCH` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_COMPACT_DRAFTER_CONTEXT` | bool | `True` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_CONV_KERNEL` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_DEBUG` | bool | `False` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE` | bool | `False` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_ENGINE_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_FORCE_MAMBA_COMPACT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_FUSED_GDN` | bool | `True` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_GPU_SAMPLER` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_LINEAR_GDN` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_MAMBA_COMPACT_BATCH` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_METADATA_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE_LAYER` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE_MAX_REPORTS` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_PATH_PROBE_NODE_LIMIT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_QLA_GDN` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_SERIAL_GDN` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_SKIP_MAMBA_COMPACT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_STOCHASTIC_TOPK_LOGITS` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_TRACE_JSONL` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN` | bool | `True` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT` | bool | `False` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_TRITON_SAMPLER` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_CONTEXT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_TOPK` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DDTREE_WORKER_PROFILE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: DDTree (speculative method dflash_ddtree) is retired. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DEBUG_COORD_TRACE` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: glm5next is outside the supported models. Setting it logs a deprecation warning. |
| `VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT` | str \| None | `None` | - | deprecated | 1Cat | Deprecated: glm5next is outside the supported models. Setting it logs a deprecation warning. |

### speculative_config.sm70_dflash2 keys

Pass these inside `--speculative-config '{..., "sm70_dflash2": {...}}'` (see [DFlash2](dflash2.md)). `None` (unset) takes the built-in default when the DFlash2 verifier qualifies (Flash-Next on SM70 under the ReVolta serving profile), otherwise the value of the matching legacy `VLLM_SM70_DFLASH2_*` environment variable; shape, dtype and native checks still decide each dispatch.

| Key | Type | Default | Effect and trade-off |
|---|---|---|---|
| `context_kv_graph` | bool \| None | `None` | Capture the drafter's context-KV writes in CUDA graphs per token count instead of running them eagerly. |
| `context_pipeline` | bool \| None | `None` | Pipeline the drafter context-KV refresh with the verifier; accepted slots are refreshed before each replay. |
| `fused_gdn_metadata` | bool \| None | `None` | Build the GDN verify metadata with one fused launch. |
| `fused_smallq_metadata` | bool \| None | `None` | Fused Flash-V100 metadata for small-query verify batches. |
| `grouped_smallq_metadata` | bool \| None | `None` | Grouped small-query Flash-V100 metadata across requests. |
| `sparse_target_rejection` | bool \| None | `None` | Avoid gathering full-vocabulary target logits for eligible sampled DFlash2 rounds. On by default for the qualifying Flash-Next serving profile with a DFlash2 selector (AWQ and NVFP4). Greedy and unsupported sampling stay dense. Set false to use dense verification for all requests; the legacy VLLM_SM70_DFLASH2_SPARSE_TARGET_REJECTION alias is used only when the config value is unset. |
| `spec_state_replay` | bool \| None | `None` | Replay the accepted prefix to rebuild recurrent state instead of storing every speculative state: more GPU KV capacity at a small per-step cost. The serving profile turns it on for DFlash2 unless fused_gdn_verify is set explicitly; Host KV mode also turns it on. Explicit false disables replay. See [State replay](dflash2.md#state-replay). |
| `verify_fastpath` | bool \| None | `None` | Fast metadata path for verify steps in the hybrid (GDN + attention) model state. |

<!-- END GENERATED: tools/gen_env_docs.py -->

## See also

- [Reference launch tutorial](getting-started.md)
- [Checkpoints, KV scales and MTP drafts](checkpoints.md)
- [DFlash2 speculative decoding](dflash2.md)
- [Dynamic speculative decoding](dynamic-speculative-decoding.md)
- [KV offload modes](kv-offload.md)
- [Support and known limits](known-limits.md)
