# Configuration reference

This page is the reference for ReVolta 0.2.0 settings: what the ReVolta profile sets automatically, the launch options you may change, sampling defaults, memory budgets and every documented environment variable. For the installation and launch sequence, follow the [getting-started tutorial](getting-started.md). Its reference launch needs no hand-tuned values: the profile fills in the tuned defaults.

Terms used in the tables:

- **Scope.** *Deployment* settings are fixed at launch for the whole server: environment variables, command-line options and the fields of their JSON configs. *Request* settings are chosen per API request.
- **Status.** *Stable* settings are part of the supported forms. *Experimental* settings are off by default or outside release acceptance, and can change between releases. *Optional* features are off by default and do not affect serving while off. *Deprecated* settings still work and will be removed.
- **Since.** The first ReVolta release with the setting. *upstream* means inherited from vLLM; *1Cat* means inherited from 1Cat-vLLM.

## Environment settings

<a id="settings-used-by-the-reference-deployment"></a>

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
| `num_speculative_tokens_per_batch_size` | A schedule for MTP with K=4 (with prefix caching) and for DFlash2 with K=7; other K values keep a fixed width | Chooses the number of drafts from the number of scheduled requests. The schedules are in [Default schedules](dynamic-speculative-decoding.md#default-schedules). |
| `dflash2_lookup` and the `sm70_dflash2` verifier options | Native history lookup for single requests and the verifier fast paths, for the DFlash2 selector | See [What the profile sets](dflash2.md#what-the-profile-sets). |
| `qsa_dcp_index_shard` in `--kernel-config` | `true` for DFlash2, and for MTP with Host KV (`long_context_kv.enabled`), when the model selects 512 index groups (Qwen3.8 Flash-Next) | Splits the target QSA indexer keys over all tensor-parallel ranks; it takes effect at DCP2. MTP with GPU KV keeps the replicated layout. It is part of the SSD page identity; see [Configure Host KV](kv-offload.md#configure-host-kv). |
| `VLLM_SM70_QSA_GROUPED_LPT`, `VLLM_SM70_GDN_SKINNY_MPLAN` | `1` | Longest-first grouped QSA work and row-count-based GDN input plans. |
| `VLLM_SM70_DFLASH2_FUSED_QKV_PACK` | `1` for DFlash2 | Fuses eligible post-convolution QKV packing. |

Outside the fully connected [NVLink TP4 profile](#nvlink-tp4-auto-configuration), omitted `--async-scheduling` is resolved from the executor and the speculative settings. A dynamic schedule and MTP's GDN state replay select synchronous scheduling. An explicit `--async-scheduling` keeps the existing compatibility checks. What async scheduling does to the GDN state depends on the deployment; see [DFlash2 scheduling](dflash2.md#scheduling).

### NVLink TP4 auto-configuration

With `VLLM_SM70_INTERCONNECT=auto`, the SM70 profile selects NVLink defaults for AWQ or NVFP4 with MTP or DFlash2 when all six pairs among the four TP ranks have direct NVLink. This is a single-node, single-replica setup with pipeline parallelism 1. Explicit environment, CLI and JSON settings take precedence.

| Default | What it enables |
| --- | --- |
| `VLLM_SM70_NVLINK_HC=1` and `sm70_hc_ll_async: true` | HyperConnection low-latency (HC LL) communication, including with async scheduling. Workers still require P2P access and peer atomics for HC LL. |
| Asynchronous scheduling | Remove `--no-async-scheduling` from the launch command to use this default. |
| `--enable-hc-sp` | HyperConnection sequence parallelism for eligible prefill batches. |
| FP16 paged QSA readers | H12 (`sm70_qsa_paged_fp16_h12`) for eligible DCP2 decode with FP16 GPU main KV; H6 (`sm70_qsa_paged_fp16_h6`) for eligible DCP2 MTP decode with FP16 draft KV. |
| MTP sampling and draft schedule | `sm70_sparse_target_sampling: "device_top64"` and DSD `[[1, 8, 3]]` without changing the configured draft maximum. |

If the selected GPUs do not form a six-pair direct mesh, startup warns that they are not fully connected and does not apply the NVLink profile. Check `nvidia-smi topo -m` and select a fully connected group in `CUDA_VISIBLE_DEVICES`.

When HC LL is active, the final HyperConnection mixers in the target and MTP draft use the fused down/up projection for up to 64 rows. Larger batches and sequence-parallel prefill keep the generic projection.

The MTP draft FC residual add and all-gather can also run in one peer kernel. This is off by default; add `"sm70_nvlink_mtp_fc_peer": true` to `--kernel-config` to enable it on Model Runner V2 with fully connected NVLink TP4 and HC LL active. It works with AWQ or NVFP4 and GPU or Host KV, for up to 64 rows. It does not apply to DFlash2.

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

Pending merge: the ReVolta profile will enable the native paged QSA decode reader by default on SM70 TP4/DCP2 with INT8 main KV and Model Runner V2, on both PCIe and NVLink. This covers eligible target decode (H12) and eligible INT8 MTP draft decode (H6), including reads from the hot pool with Host KV. To turn it off explicitly, add `"sm70_qsa_paged_int8_h12": false` to `--kernel-config`. <!-- TODO(0.2.0-acceptance): confirm the INT8 reader change has merged and verify the default and opt-out in the release build -->

Draft KV:

- **MTP** inherits the target format, with GPU KV and with Host KV. To override it, set `"kv_cache_dtype"` in `--speculative-config`; `"float16"` keeps an FP16 draft with an 8-bit target, and the INT8 names are accepted when the target uses INT8. Host KV placement of the draft is described in [KV offload](kv-offload.md).
- **DFlash2** with the SM70 QSA target uses FP16 draft KV by default, with GPU KV and with Host KV.

### Speculative decoding

These fields go in `--speculative-config`.

| Field | Type | Default | Range | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `draft_sample_method` | string | `greedy` | `greedy`, `probabilistic` | 1Cat | stable | How the drafter picks proposals. MTP must stay `greedy`: a probabilistic MTP draft corrupts output above `max_num_seqs=2` at temperature above 0 (1CatAI/1Cat-vLLM#656). For DFlash2, `probabilistic` is recommended; it passed the 0.1.4 release gate at 1–8 concurrent requests, temperature 1 with thinking, tool calls and structured output. This setting affects drafting and draft verification only; clients can still use any sampling parameters. |
| `num_speculative_tokens_per_batch_size` | list of `[first, last, drafts]` | profile default ([Default schedules](dynamic-speculative-decoding.md#default-schedules)) | inclusive ranges of the scheduled request count | 0.2.0 | stable | Chooses the number of drafts from the number of scheduled requests. See [Dynamic speculative decoding](dynamic-speculative-decoding.md). |
| `kv_cache_dtype` | string | unset (inherit) | see [KV cache format](#kv-cache-format) | upstream | stable | Draft KV format; see the draft KV notes above. |

DFlash2 options are in [DFlash2 speculative decoding](dflash2.md) and [speculative_config.sm70_dflash2 keys](#speculative_configsm70_dflash2-keys).

### Memory and capacity

| Option | Type | Reference value | Default | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `--gpu-memory-utilization` | float | `0.956` | `0.92` | upstream | stable | Share of each GPU's memory the engine may use; the rest of the reserve sizes the KV cache. Keep the GPUs free of other processes that use GPU memory. 0.2.0 acceptance runs at `0.956` without multimodal stress. Measure again if you raise concurrency or media limits. |
| `cpu_bytes_to_use` in `--kv-transfer-config` | integer bytes | `51539607552` (48 GiB) | none | upstream | stable | Size of the CPU KV pool, summed over all four workers (48 GiB = 12 GiB per worker). It needs host memory on top of the PLE table; see [System memory and storage](#system-memory-and-storage) and [KV offload](kv-offload.md). |

### CUDA graph reserve

Model Runner V2 cannot measure CUDA graph memory before it sizes the KV cache, so the ReVolta profile reserves graph memory per GPU through `VLLM_V2_CUDAGRAPH_MEM_MIB`. The value depends on the speculative method, the draft schedule, the checkpoint and the KV format. Planar KV means `int8` (including `auto` on V100) or `fp8_per_token_head`.

| Speculative method | KV format | Reserve per GPU |
| --- | --- | ---: |
| MTP, fixed draft count | Planar | 600 MiB |
| MTP with a dynamic draft schedule, `num_speculative_tokens` up to 4 (the reference configuration) | Planar | 705 MiB |
| MTP with a dynamic draft schedule, `num_speculative_tokens` 5 or more | Planar | 800 MiB |
| MTP | Scalar E4M3 or FP16 | 700 MiB |
| None | Any | 700 MiB |
| DFlash2, AWQ | Planar | 1000 MiB (1150 MiB with a positive `--mixed-prefill-step-latency-ms`) |
| DFlash2, NVFP4 | Planar | 1024 MiB |
| DFlash2 | Scalar E4M3 or FP16 | 1200 MiB |

With DFlash2 the profile also turns off the single-request verifier tail graphs. An explicit `VLLM_V2_CUDAGRAPH_MEM_MIB` takes precedence.

If the startup log shows `CUDA graphs took … more than the reserve the KV cache was sized with`, set `VLLM_V2_CUDAGRAPH_MEM_MIB` to at least the value in the message, or lower `--gpu-memory-utilization`. Otherwise the KV cache uses memory that the graphs need, and a large mixed step can run out of GPU memory. A reserve that is too high wastes KV capacity.

### Multimodal input

| Option | Type | Reference value | Default on V100 | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- | --- |
| `--limit-mm-per-prompt` | JSON | `{"image": 16, "video": 1}` | 1 image, no video | upstream | stable | Media items per request. With FlashAttention-V100, the engine allows 1 image and no video when the option is omitted, and a value without `"video"` disables video. A lower image count leaves more host memory for bursts of large images. |

Image resizing, video frames and per-request media budgets are in [Multimodal input limits](multimodal.md).

### API server

| Option | Type | Default | Since | Status | Effect and trade-offs |
| --- | --- | --- | --- | --- | --- |
| `--enable-dashboard` | boolean | `false` | 0.2.0 | optional | Serves the built-in aggregate metrics page at `/dashboard/`. Off by default. Calculations run in the browser; the data routes follow the API key. See [Dashboard](dashboard.md). |
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
| `VLLM_SM70_NVLINK_HC` | boolean | `0` | `0`, `1` | 0.2.0 | experimental | Flash-Next HC sharding, fused all-reduce and HC, and the QSA DCP2 query exchange and output merge over NVLink. All six GPU pairs must have direct NVLink and native peer atomics; `VLLM_SM70_INTERCONNECT=nvlink` does not bypass this check. The [NVLink TP4 profile](#nvlink-tp4-auto-configuration) sets it to `1` and sets `sm70_hc_ll_async: true` to allow it with asynchronous scheduling. The QSA route takes FP16 queries with six 256-element heads per rank and up to 64 rows. It can change floating-point rounding. |
| `VLLM_SM70_DCP2_P2P` | boolean | `0` | `0`, `1` | 0.2.0 | experimental | QSA DCP2 query and packed output/LSE exchange through the registered PCIe switch push buffers, with the A2A DCP backend (the Flash-Next default). It covers eager uniform decode and dedicated FULL decode graphs up to 53 rows; PIECEWISE graphs keep NCCL. In 0.2.0 tests with four GPUs and AWQ, it gave no stable gain in cold time to first token or decode, and fixed-prompt outputs matched the NCCL path token for token. Needs the native extension from the same build. |

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

With the default pinned PLE table, 80 GiB is the smallest tested limit that passed memory stress, with KV offload off and without full acceptance. With less than 128 GiB, reduce or disable the CPU KV pool. For smaller hosts, the experimental option that keeps the whole PLE table on disk is described in [Whole PLE table on disk](known-limits.md#whole-ple-table-on-disk-experimental).

Image and video preprocessing also uses host memory, in the API server process. In a 0.1.1 test, 8 concurrent requests with 8 different 2048-pixel images each used about 33 GB at peak. With a 32 GiB CPU KV pool on a 125 GiB host, the kernel then killed a worker. The same burst passed with a 16 GiB pool. If you expect bursts of large images or videos, use a smaller CPU KV pool or limit the image size and the number of concurrent requests.

The AWQ checkpoint is 137 GB at the pinned revision:

| Part | Format | Size | Where it runs |
| --- | --- | ---: | --- |
| Routed experts | 4-bit AWQ, group size 32 | 74.9 GB | GPU |
| PLE n-gram tables | FP8 E4M3 | 51.3 GB | Pinned host memory |
| Other weights (attention, embeddings, `lm_head`, vision tower, shared experts) | 16-bit | 10.9 GB | GPU |
| Optional MTP draft (`mtp-fp8/`) | FP8 | 5.3 GB | GPU |

Allow extra disk space for compilation caches and the optional disk KV tier. Structured long-context Host KV settings (`kernel_config.long_context_kv.*`) are in [KV offload](kv-offload.md).

<a id="promoted-research-switches"></a><a id="pregraph-preparation-research-switches"></a><a id="host-decode-fast-path"></a><a id="dflash2-and-dflash-draft-settings"></a><a id="optional-mtp-draft-vocabulary-in-model-runner-v2"></a><a id="optional-mtp-draft-vocabulary-experimental"></a><a id="deprecated-removed-in-020rc6"></a>

<!-- BEGIN GENERATED: tools/gen_env_docs.py -->
<!-- Edit tools/env_docs_notes.py, then run python tools/gen_env_docs.py -->

## Environment variable reference

Every variable below is read once per process (deployment scope). Columns: type and default as registered in `vllm/envs.py`; *Since* is the first release with the variable (1Cat = inherited from 1Cat-vLLM); *Category* is configuration, tuning or experimental. Developer and deprecated variables are listed by name at the end.

### Kernel and path switches

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_DFLASH_COMPACT_AUX_HIDDEN` | bool | `True` | 0, 1 | configuration | 1Cat | Builds the DFlash and DFlash2 draft inputs from the target's hidden states without an extra full-precision copy. Leave on. |
| `VLLM_SM70_CHECKPOINT_CACHE_DROP` | bool | `True` | 0, 1 | configuration | 0.2.0 | At startup, after every rank has loaded the model, drops the checkpoint files from the page cache before the host KV pools are allocated. Files still mapped and the PLE disk modes are skipped. Set 0 if other processes read the same checkpoint and need it cached. |
| `VLLM_SM70_GDN_CPU_SPEC_QSL` | bool | `False` | 0, 1 | tuning | 0.2.0 | Set 1 to remove two host synchronizations per step in batches that mix prefill and speculative decoding; output is unchanged. |
| `VLLM_SM70_GDN_NATIVE_PREFILL` | bool | `True` | 0, 1 | tuning | 0.2.0 | Runs eligible GDN prefill chunks with the native SM70 kernel (Model Runner V2). It can change floating-point rounding. Set 0 to use the TileLang kernel. |
| `VLLM_SM70_HOST_DECODE_FAST` | bool | `True` | 0, 1 | tuning | 0.2.0 | Host KV only: fused decode kernels for the history in host memory. Set 0 to use the original kernels. |
| `VLLM_SM70_MTP_KV_ONLY_PREFILL` | bool | `True` | 0, 1 | tuning | 0.2.0 | Lets the MTP draft skip work during chunked prefill; see [MTP settings](speculative-decoding.md#mtp-settings-you-can-change). It can change floating-point rounding. Set 0 for full draft prefill. |
| `VLLM_SM70_QSA_LOCAL_INPUTS_FUSED` | bool | `True` | 0, 1 | tuning | 0.2.0 | One fused kernel for the index metadata of the DCP-sharded QSA scorer. Set 0 to use the reference implementation. |
| `VLLM_SM70_QSA_SHORT_APPEND` | bool | `True` | 0, 1 | tuning | 0.2.0 | Lets short prompt continuations on a long cached context use the faster cuBLAS QSA scorer. Near-tied block selection can differ from the paged scorer. Set 0 to keep the paged scorer. |

### Deployment settings

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_1CAT_ADAPTIVE_MTP` | str | `off` | off/0/false/no, on/1/true/yes, auto | experimental | 0.2.0 | Adapts the draft width of a single decoding request to its recent acceptance. Needs a draft-width schedule and synchronous scheduling; `auto` acts only on SM70 with at least 5 draft tokens. See [Adaptive MTP depth](dynamic-speculative-decoding.md#adaptive-mtp-depth). |
| `VLLM_PLE_OFFLOAD_READY_TIMEOUT` | float | `600.0` | seconds | configuration | 1Cat | How long startup waits for the PLE offload worker. Raise it on slow storage; the reference deployment uses 1200. |
| `VLLM_QWEN4EXP_PLE_HOST_GIB` | float \| None | `None` | GiB per TP rank; unset = automatic | configuration | 1Cat | Per-rank cap on pinned host memory for the FP8 PLE table. Unset pins the whole table; with a cap, the remaining rows stay on the GPU. |
| `VLLM_QWEN4EXP_PLE_HOST_RESERVE_GIB` | float \| None | `None` | non-negative GiB or unset | configuration | 1Cat | Host memory that PLE placement keeps free. Unset reserves the smaller of a quarter of the host limit and 8 GiB. |
| `VLLM_QWEN_REASONING_EOS_MASK` | bool | `False` | 0, 1 | configuration | 0.2.0 | 1 keeps a request from ending its turn while it is still inside its reasoning block; see [the reasoning end-of-turn mask](#chat-control-tokens-and-the-reasoning-end-of-turn-mask). |
| `VLLM_SM70_AWQ_MOE_INT4_ZERO` | bool | `False` | 0, 1 | experimental | 0.1.0 | Stores AWQ zero points as 4-bit values: less GPU memory, same results. Under `vllm serve` the profile sets 1 for AWQ g32 checkpoints with compact metadata. |
| `VLLM_SM70_FLASH_ATTN_V100` | bool | `True` | 0, 1 | configuration | 1Cat | The V100 FlashAttention path; set in the image. Set 0 only to diagnose backend selection; use `--attention-backend` to choose a backend. |
| `VLLM_SM70_FLASH_V100_0DOT3_COMPILE_GRAPH` | bool | `False` | 0, 1 | configuration | 1Cat | Set to 1 in the image: torch.compile with FULL_AND_PIECEWISE CUDA graphs and small decode capture sizes. Leave as is. |
| `VLLM_SM70_HOST_KV_STATIC_HOT_POOL` | bool | `False` | 0, 1 | configuration | 0.2.0 | Host KV only: 1 restores the separate static hot-page pool; the default lets hot pages share GPU memory with unused index pages. |
| `VLLM_SM70_IDLE_CLOCK_SECONDS` | int | `0` | non-negative integer seconds | configuration | 0.2.0 | Locks exclusively assigned V100 GPUs at 405 MHz after this many idle seconds; 0 disables it. Needs NVML clock permission and one engine with TP1 or TP4 on one host. After an abnormal exit, reset the GPU clocks before reusing the GPUs. |
| `VLLM_SM70_MTP_PLE_CONV` | bool | `True` | 0, 1 | configuration | 1Cat | Fused PLE state update for MTP and DFlash2 batches. Set 0 only to diagnose a suspected kernel problem. |
| `VLLM_SM70_PLE_GATE_NORM` | bool | `False` | 0, 1 | experimental | 0.2.0 | Opt-in fusion of the PLE gate and norm in decode graphs. It can change FP16 rounding. |

### Values applied by the ReVolta profile

The ReVolta profile (`vllm/revolta_profile.py`) sets these on SM70 + TP4 + FP16 + QSA models when they are unset; explicit values win. The profile column shows the value it applies (AWQ-only and NVFP4-only entries apply to that checkpoint family only).

| Name | Type | Default | Profile | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|---|
| `VLLM_1CAT_DISABLE_SM70_MTP_DEFAULTS` | bool | `False` | `1` | 0, 1 | configuration | 1Cat | Turns off the inherited 1Cat MTP launch defaults; the ReVolta profile supplies its own. |
| `VLLM_EARLY_MODEL_INPUTS` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Prepares model inputs earlier in CUDA graph steps, overlapping host work. |
| `VLLM_PLE_ASYNC_REQUEST_INDICES` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Removes a host wait per step for the PLE inputs. |
| `VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP` | bool | `False` | `0`, `1` (AWQ+DFlash2) | 0, 1 | configuration | 1Cat | Runs shared and routed experts on separate CUDA streams. The profile sets 1 for AWQ g32 with DFlash2 when expert parallelism and dual-batch overlap are off. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN` | bool | `False` | `1` (AWQ) | 0, 1 | experimental | 0.1.0 | Grouped AWQ MoE decode kernel for small batches: each active expert is read once per step. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS` | int | `20` | `40` (AWQ), `48` (AWQ MTP4 wide verifier) | 2..48 | tuning | 0.1.0 | Largest batch, in rows, for the grouped AWQ MoE kernel; larger batches use TurboMind. DFlash2 uses its own row cap. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_PACK_ROWS` | int | `0` | `24` (AWQ) | 0 or 8..48 step 8 | tuning | 0.1.0 | Tuning of the grouped AWQ kernel (rows per expert pack). Keep the profile value. |
| `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_SPLIT` | int | `16` | `8` (AWQ) | 4, 8, 16 | tuning | 0.1.0 | Tuning of the grouped AWQ kernel (warps per expert). Keep the profile value. |
| `VLLM_SM70_AWQ_QWEN38_QPN_M1` | bool | `True` | `1` (AWQ) | 0, 1 | configuration | 1Cat | Single-request AWQ MoE kernel. Set 0 for the older route. |
| `VLLM_SM70_AWQ_TUNE_SMALL_SHAPES` | bool | `False` | `1` (AWQ) | 0, 1 | configuration | 1Cat | Tunes small AWQ batch shapes at startup: longer warm-up, faster small batches. |
| `VLLM_SM70_CAR_VMM_SAFE` | bool | `False` | `1` (P2P/NVLink) | 0, 1 | experimental | 0.1.0 | Lets custom all-reduce CUDA graphs work with expandable segments (P2P or NVLink only). |
| `VLLM_SM70_DFLASH2_FUSED_QKV_PACK` | bool | `False` | `1` (DFlash2, vllm serve) | 0, 1 | configuration | 1Cat | One kernel for the Q/K/V packing of the DFlash2 verifier. |
| `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` | bool | `True` | `0` (DFlash2) | 0, 1 | configuration | 1Cat | Single-request DFlash2 verifier tail graphs. The profile sets 0 for DFlash2 so that their memory goes to the KV cache; with 1, also raise `VLLM_V2_CUDAGRAPH_MEM_MIB`. |
| `VLLM_SM70_GDN_SKINNY_MPLAN` | bool | `False` | `1` (vllm serve) | 0, 1 | tuning | 0.2.0 | Faster GDN projection plans for some batch sizes; needs `VLLM_SM70_QWEN38_GDN_NO_PACK`. |
| `VLLM_SM70_GREEDY_VERIFY` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Faster MTP verification when every request in the batch is greedy. |
| `VLLM_SM70_MTP_GDN_NATIVE_VERIFY` | bool | `False` | `1` (AWQ MTP4 wide verifier) | 0, 1 | configuration | 0.2.0 | Native GDN verification for the [AWQ MTP4 wide verifier](#awq-mtp4-wide-verifier). Set 0 to disable. |
| `VLLM_SM70_MTP_Q8_PREFILL_MMQ` | bool | `False` | `1` (NVFP4) | 0, 1 | configuration | 0.2.0 | Grouped kernel for large MTP draft prefill batches (above 64 rows). It uses more temporary GPU memory and can change rounding. Set 0 for the smaller per-row kernel. |
| `VLLM_SM70_NVFP4_MOE_GROUPED_DECODE` | bool | `False` | `0` (NVFP4) | 0, 1 | experimental | 1Cat | Alternative NVFP4 MoE decode kernel; needs `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=0`. The profile keeps it off because its quality is not validated. |
| `VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | One grouped call for all routed experts in NVFP4 prefill. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 1Cat | NVFP4 MoE decode kernel for 2–16 rows. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 0.1.0 | Grouped NVFP4 MoE decode for 2–40 rows; needs raw scales (`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=1`). |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE` | bool | `False` | `1` (NVFP4) | 0, 1 | experimental | 1Cat | Keeps NVFP4 block scales in their compact E4M3 form instead of a second FP16 copy, which saves GPU memory. Set 0 for FP16 scales, for example to use dual-batch overlap or microbatching, which raw scales do not support. |
| `VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | Tunes small NVFP4 shapes at startup. |
| `VLLM_SM70_NVFP4_TURBOMIND` | bool | `True` | `1` (NVFP4) | 0, 1 | configuration | 1Cat | TurboMind kernels for NVFP4 and MXFP4 dense layers. Set 0 only to diagnose. |
| `VLLM_SM70_PLE_INPUT_PREPARE` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Fused PLE input preparation for up to 4 requests. |
| `VLLM_SM70_PLE_NGRAM_BATCH` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Fused PLE n-gram preprocessing for 2–32 rows. |
| `VLLM_SM70_PLE_PREFILL_CONV_CHUNK` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Bounds the PLE prefill buffers when several prefills share a step; needed at `--max-num-seqs 8`. |
| `VLLM_SM70_QSA_COMPACT_TOPK` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Faster QSA block selection for up to 32 rows at long contexts; it selects the same blocks. |
| `VLLM_SM70_QSA_GROUPED_LPT` | bool | `False` | `1` (vllm serve) | 0, 1 | tuning | 0.2.0 | Starts the longest QSA work first, which shortens uneven decode batches. |
| `VLLM_SM70_QSA_REQUEST_SEGMENTS` | bool | `False` | `1` | 0, 1 | experimental | 0.2.0 | Reads the QSA keys once per request in multi-request decode. |
| `VLLM_SM70_QSA_SHARED_KEY` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Faster QSA indexer for the 2–8 verification queries of one request. |
| `VLLM_SM70_QWEN38_GDN_NO_PACK` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Skips a packed copy of the GDN weights (about 0.7 GiB per GPU), which frees memory for the KV cache. |
| `VLLM_SM70_QWEN38_HYBRID_PLE` | bool | `False` | `0` | 0, 1 | configuration | 1Cat | Per-rank pinned PLE shards for decode. The profile sets 0 (one shared PLE placement). |
| `VLLM_SM70_QWEN38_NATIVE_ROUTER_TOPK` | bool | `False` | `1` (AWQ MTP4 wide verifier) | 0, 1 | experimental | 0.1.0 | Native MoE router for 17–64 rows in the [AWQ MTP4 wide verifier](#awq-mtp4-wide-verifier). Set 0 to disable. |
| `VLLM_SM70_QWEN38_WIDE_M` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Extends the fast Qwen3.8 decode and verify kernels from 4 to 8 requests. |
| `VLLM_SM70_SKIP_REJECTED_BATCH_PACKS` | bool | `False` | `1` | 0, 1 | experimental | 0.1.0 | Skips weight packs that are never used, about 199 MiB per GPU. |
| `VLLM_USE_AOT_COMPILE` | bool | `False` | `0` | 0, 1 | configuration | 1Cat | Ahead-of-time compile reuse. The profile sets 0. |
| `VLLM_V2_CUDAGRAPH_MEM_MIB` | float | `0.0` | `700` (AWQ), `700` (NVFP4), `1200` (DFlash2), `1000` (AWQ+DFlash2, planar KV), `1150` (AWQ+DFlash2, planar KV, positive mixed-prefill target), `1024` (NVFP4+DFlash2, planar KV), `600` (fixed MTP, planar KV), `705` (dynamic MTP, planar KV), `800` (dynamic MTP with 5+ drafts, planar KV) | MiB | configuration | 1Cat | CUDA graph memory reserved per GPU before the KV cache is sized. The profile picks the value by method, checkpoint and KV format; see [CUDA graph reserve](#cuda-graph-reserve). Raise it if the startup log warns that the graphs took more than the reserve. |

### Experimental switches (off by default)

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_SM70_NVFP4_RAW_GROUPED_M48_V154` | bool | `False` | 0, 1 | experimental | 0.2.0 | Opt-in NVFP4 MoE decode kernel for 41–64 rows. Needs raw grouped decode, Model Runner V2 and equal gate/up global scales. |
| `VLLM_SM70_NVFP4_RAW_GROUPED_NARROW_V154` | bool | `False` | 0, 1 | experimental | 0.2.0 | Opt-in NVFP4 MoE decode kernel for 3–5 rows. Needs raw grouped decode, Model Runner V2 and equal gate/up global scales. |

### Anonymous usage statistics

| Name | Type | Default | Range | Category | Since | Effect and trade-off |
|---|---|---|---|---|---|---|
| `VLLM_DO_NOT_TRACK` | bool | `False` | 0, 1 | configuration | 1Cat | Honours the cross-project Do Not Track convention (DO_NOT_TRACK counts too): 1 turns the usage report off. |
| `VLLM_NO_USAGE_STATS` | bool | `False` | 0, 1 | configuration | 1Cat | Upstream opt-out, honoured as-is: 1 turns the usage report off. The Do Not Track variables and the do_not_track file have the same effect. |
| `VLLM_USAGE_SOURCE` | str | `'production'` | short label | configuration | 1Cat | Deployment label sent as `source` (`production` by default). A value with `/`, `\`, `:`, a space or more than 64 characters is sent as `custom`. Do not put identifying text in it. |
| `VLLM_USAGE_STATS_SERVER` | str | `'https://usage.19b.net/api/'` | URL, empty to disable sending | configuration | 1Cat | Where the report goes. ReVolta defaults to its own collector; an empty string keeps the local log file but sends nothing. |

### DFlash2 legacy lookup

The legacy `ngram_assist` lookup for DFlash2 (off by default). These variables still work, with no removal release set, and setting any of them turns off the profile's default native lookup. New setups use `dflash2_lookup`; see [Native history lookup](dflash2.md#native-history-lookup).

| Name | Default | Effect |
|---|---|---|
| `VLLM_DFLASH2_LOOKUP_ADAPTIVE` | `True` | Adaptive q8/q16 lookup width. |
| `VLLM_DFLASH2_LOOKUP_AGREE` | `0` | Leading tokens the lookup and the draft must agree on (0 = none). |
| `VLLM_DFLASH2_LOOKUP_CHEAP_CONTEXT` | `0` | Context length below which lookup is always tried (0 = off). |
| `VLLM_DFLASH2_LOOKUP_ENTRY_STREAK` | `2` | Consecutive hits before lookup mode starts. |
| `VLLM_DFLASH2_LOOKUP_LONG_MIN` | `6` | Minimum length of a long match. |
| `VLLM_DFLASH2_LOOKUP_NMIN_TAIL` | `4` | Smallest n-gram accepted for the lookup tail. |
| `VLLM_DFLASH2_LOOKUP_NSTRONG` | `6` | n-gram length that counts as a strong prompt match. |
| `VLLM_DFLASH2_LOOKUP_SEARCH` | `1073741824` | Context searched for matches (default unbounded). |
| `VLLM_DFLASH2_LOOKUP_STICKY` | `3` | Steps lookup mode stays on after a hit. |

### Developer diagnostics

These variables log, dump or switch internal paths for development. Leave them unset when serving.

- `VLLM_DFLASH_DEBUG_CONTEXT_KV`, `VLLM_DFLASH_DEBUG_PROPOSAL_STAGES`, `VLLM_DFLASH_DEBUG_STATE_TABLE`, `VLLM_DFLASH_DEBUG_TARGET_LAYER_TRACE`, `VLLM_DFLASH_DEBUG_TARGET_LOGITS`, `VLLM_DFLASH_DEBUG_TARGET_TRACE_MIN_POSITION`, `VLLM_DFLASH_DEBUG_TENSOR_DUMP_DIR`, `VLLM_DFLASH_DEBUG_TENSOR_DUMP_LIMIT`: DFlash debug logs, traces and tensor dumps
- `VLLM_DFLASH_DEBUG_CORRUPTION`, `VLLM_DFLASH_DISABLE_AUX_OUTPUTS`, `VLLM_DFLASH_DUMP_DRAFT_LOGITS`, `VLLM_DFLASH_DUMP_FIRST_PASS`: DFlash diagnostics for Model Runner V1 only; no effect on Qwen3.8 Flash-Next, which runs Model Runner V2
- `VLLM_DFLASH_PROFILE`, `VLLM_DFLASH_PROFILE_LOG_INTERVAL`: DFlash and DFlash2 proposal timing log and its interval
- `VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE`: validation file for `VLLM_SM70_GDN_CPU_SPEC_QSL`; do not set it for timing runs
- `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FULL_REFRESH_INTERVAL`, `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_FUSED_PROPOSAL`, `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_GPU_LRU`, `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_PREFILL_TOPK`: Model Runner V1 only; setting one with Model Runner V2 is an error
- `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_TAIL_SIZE`, `VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING`, `VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE`: experimental MTP draft vocabulary shortlist, outside release acceptance; it needs a ranking file that ReVolta does not provide
- `VLLM_SM70_QWEN_GDN_DISABLE_FULL_FORWARD`, `VLLM_SM70_QWEN_GDN_FULL_FORWARD`, `VLLM_SM70_QWEN_GDN_SPEC_CORE_OP`: GDN compile-boundary diagnostics; the image sets 0

### Deprecated variables

Still accepted in 0.2.0; setting one logs a deprecation warning. For a renamed variable the new name wins when both are set.

- `REVOLTA_QSA_GROUPED_LPT` → `VLLM_SM70_QSA_GROUPED_LPT`
- `REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP` → `VLLM_SM70_CHECKPOINT_CACHE_DROP`
- `REVOLTA_RESEARCH_HOST_DECODE_FAST` → `VLLM_SM70_HOST_DECODE_FAST`
- `REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED` → `VLLM_SM70_QSA_LOCAL_INPUTS_FUSED`
- `REVOLTA_SKINNY_MPLAN` → `VLLM_SM70_GDN_SKINNY_MPLAN`
- `VLLM_DFLASH_DDTREE_ATTN_COMPACT_BATCH`, `VLLM_DFLASH_DDTREE_COMPACT_DRAFTER_CONTEXT`, `VLLM_DFLASH_DDTREE_CONV_KERNEL`, `VLLM_DFLASH_DDTREE_DEBUG`, `VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD`, `VLLM_DFLASH_DDTREE_DISABLE_GDN_FAST_BUILD_CACHE`, `VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD`, `VLLM_DFLASH_DDTREE_ENABLE_GDN_FAST_BUILD_CACHE`, `VLLM_DFLASH_DDTREE_ENABLE_HYBRID_TREE_STATE`, `VLLM_DFLASH_DDTREE_ENGINE_PROFILE`, `VLLM_DFLASH_DDTREE_FAST_BUILD_DEBUG`, `VLLM_DFLASH_DDTREE_FORCE_MAMBA_COMPACT`, `VLLM_DFLASH_DDTREE_FUSED_GDN`, `VLLM_DFLASH_DDTREE_GDN_FAST_BUILD_TRITON`, `VLLM_DFLASH_DDTREE_GDN_SHARED_COMMON`, `VLLM_DFLASH_DDTREE_GPU_SAMPLER`, `VLLM_DFLASH_DDTREE_LINEAR_GDN`, `VLLM_DFLASH_DDTREE_MAMBA_COMPACT_BATCH`, `VLLM_DFLASH_DDTREE_METADATA_PROFILE`, `VLLM_DFLASH_DDTREE_PATH_PROBE`, `VLLM_DFLASH_DDTREE_PATH_PROBE_LAYER`, `VLLM_DFLASH_DDTREE_PATH_PROBE_MAX_REPORTS`, `VLLM_DFLASH_DDTREE_PATH_PROBE_NODE_LIMIT`, `VLLM_DFLASH_DDTREE_PROFILE`, `VLLM_DFLASH_DDTREE_QLA_GDN`, `VLLM_DFLASH_DDTREE_SERIAL_GDN`, `VLLM_DFLASH_DDTREE_SKIP_MAMBA_COMPACT`, `VLLM_DFLASH_DDTREE_STOCHASTIC_TOPK_LOGITS`, `VLLM_DFLASH_DDTREE_TARGET_FORWARD_NVTX`, `VLLM_DFLASH_DDTREE_TARGET_FORWARD_PROFILER_STEP`, `VLLM_DFLASH_DDTREE_TRACE_JSONL`, `VLLM_DFLASH_DDTREE_TRACE_KV_CACHE_DIFF`, `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN`, `VLLM_DFLASH_DDTREE_TRITON_BRANCH_ATTN_STRICT`, `VLLM_DFLASH_DDTREE_TRITON_SAMPLER`, `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE`, `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_CONTEXT`, `VLLM_DFLASH_DDTREE_VERIFY_ROW_TRACE_TOPK`, `VLLM_DFLASH_DDTREE_WORKER_PROFILE` → no replacement: DDTree (the `dflash_ddtree` speculative method) is retired
- `VLLM_DFLASH_DEBUG_COORD_TRACE`, `VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT` → no replacement: glm5next is outside the supported models

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
