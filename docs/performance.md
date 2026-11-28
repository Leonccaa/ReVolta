# Performance and measurements

<a id="performance"></a><a id="measurements"></a>

This page lists what ReVolta 0.2.0 measured: output speed, time to first token, KV cache capacity, prefix reuse and answer quality, for each tested configuration. It also explains how the values were measured and keeps a few earlier results that are still useful for comparison. Every value carries its version and test conditions; do not compare values across tables unless the conditions match.

For the supported hardware and configurations, see [Support and known limits](known-limits.md). To reproduce a configuration, start from [Getting started](getting-started.md).

## How we measure

### Test system

The PCIe acceptance tables below use one test system. The [NVLink development measurements](#nvlink-tp4-development-measurements) use a separate system.

- Four V100 32 GB PCIe GPUs, all under one PCIe switch with peer-to-peer (P2P) access, no NVLink.
- Power limit 200 W per GPU.
- Tensor parallel 4 with decode context parallel 2 (TP4 + DCP2).
- `--max-model-len 524288` on the 512K views built in [Getting started](getting-started.md#2-download-the-checkpoint-and-make-512k-views).
- `--max-num-seqs 8`, which the ReVolta serving profile sets when you leave it unset.

Each configuration below states its other settings.

### Benchmark method

<!-- TODO(0.2.0-acceptance): benchmark method for the speed and latency tables (prompt set, response length cap, greedy or sampled decoding and sampling parameters, thinking on or off, warm-up and number of repeats), all configurations -->

### What the columns mean

| Term | Meaning |
| --- | --- |
| C1, C4, C8 | One, four or eight requests running at the same time. |
| Output speed | Generated tokens of all running requests divided by wall-clock time, in tokens/s. It includes the time to first token. At C4 and C8 it is the total over all requests, not the speed of one request. |
| Cold TTFT | Time to first token for a prompt that is in no prefix cache. |
| Second-token latency | Time from the first to the second output token of the same request. |
| KV capacity | Tokens of context the KV cache can hold. With GPU KV, from the startup log line `GPU KV cache size: N tokens`. With Host KV, which does not log that line, the host pool tokens from the `long-context KV worker=N ... plan` line, minus one reserved page. |
| Capacity ratio | KV capacity divided by 524,288. With GPU KV it equals the `Maximum concurrency` value in the same startup log; with Host KV it is computed from the host pool tokens. It is a capacity ratio, not a request limit: at 2.5×, two full 512K requests fit, and many more shorter requests can run. With Host KV, the number of full-length requests is the smallest of the per-pool limits in the plan line. |
| Prompt tokens hit | Prompt tokens served from a prefix cache instead of being computed. |
| GSM8K, MMLU | Fixed subsets of the public GSM8K and MMLU sets, given as correct answers / questions. |

### Configurations

The 0.2.0 acceptance tested six configurations. All use TP4 + DCP2 on the test system above.

| Configuration | Weights | Speculative decoding | KV cache | CPU KV pool and disk tier | Other settings |
| --- | --- | --- | --- | --- | --- |
| C: production form | Uncensored AWQ g32 | DFlash2, 7 draft tokens, dynamic draft width (profile default) | GPU KV; INT8 target, FP16 draft | 98 GiB CPU KV pool and 350 GiB `revolta_fs` disk tier | `--gpu-memory-utilization 0.956` |
| D+E1: public default | Uncensored AWQ g32 | MTP, 4 draft tokens, dynamic draft width (profile default), greedy drafting | GPU KV; INT8 target and draft (`--kv-cache-dtype auto`) | 48 GiB CPU KV pool | Reference configuration in [Getting started](getting-started.md); `--gpu-memory-utilization 0.956` |
| E4M3 | As D+E1 | As D+E1 | GPU KV; calibrated E4M3 target and draft (`--kv-cache-dtype fp8_e4m3`) | As D+E1 | Functional check only: start, serve and answer quality |
| E2: Host KV | AWQ | MTP, 5 draft tokens | Host KV: main KV history in host memory <!-- TODO(0.2.0-acceptance): KV format and host pool size, E2 --> | <!-- TODO(0.2.0-acceptance): CPU and SSD tiers, E2 --> | QSA index sharding (`qsa_dcp_index_shard: true`) <!-- TODO(0.2.0-acceptance): remaining non-default settings, E2 --> |
| E3a: NVFP4 + MTP | <!-- TODO(0.2.0-acceptance): NVFP4 checkpoint variant, E3a --> | MTP, 5 draft tokens | GPU main KV; INT8 target and draft | CPU KV pool, no SSD tier <!-- TODO(0.2.0-acceptance): CPU KV pool size, E3a --> | <!-- TODO(0.2.0-acceptance): other non-default settings, E3a --> |
| E3b: NVFP4 + DFlash2 | <!-- TODO(0.2.0-acceptance): NVFP4 checkpoint variant, E3b --> | DFlash2 <!-- TODO(0.2.0-acceptance): draft tokens, E3b --> | <!-- TODO(0.2.0-acceptance): KV placement and target / draft format, E3b --> | CPU KV pool, no SSD tier <!-- TODO(0.2.0-acceptance): CPU KV pool size, E3b --> | <!-- TODO(0.2.0-acceptance): other non-default settings, E3b --> |

## Results

All values in this section were measured with ReVolta 0.2.0 on the test system above.

### Output speed

<a id="speed"></a>

Output speed in tokens/s.

| Configuration | C1 | C4 | C8 |
| --- | ---: | ---: | ---: |
| C: AWQ, DFlash2, GPU KV | <!-- TODO(0.2.0-acceptance): output tok/s at C1, C --> | <!-- TODO(0.2.0-acceptance): output tok/s at C4, C --> | <!-- TODO(0.2.0-acceptance): output tok/s at C8, C --> |
| D+E1: AWQ, MTP4, GPU KV | <!-- TODO(0.2.0-acceptance): output tok/s at C1, D+E1 --> | <!-- TODO(0.2.0-acceptance): output tok/s at C4, D+E1 --> | <!-- TODO(0.2.0-acceptance): output tok/s at C8, D+E1 --> |
| E2: Host KV, MTP | <!-- TODO(0.2.0-acceptance): output tok/s at C1, E2 --> | <!-- TODO(0.2.0-acceptance): output tok/s at C4, E2 --> | <!-- TODO(0.2.0-acceptance): output tok/s at C8, E2 --> |
| E3a: NVFP4, MTP5 | <!-- TODO(0.2.0-acceptance): output tok/s at C1, E3a --> | <!-- TODO(0.2.0-acceptance): output tok/s at C4, E3a --> | <!-- TODO(0.2.0-acceptance): output tok/s at C8, E3a --> |
| E3b: NVFP4, DFlash2 | <!-- TODO(0.2.0-acceptance): output tok/s at C1, E3b --> | <!-- TODO(0.2.0-acceptance): output tok/s at C4, E3b --> | <!-- TODO(0.2.0-acceptance): output tok/s at C8, E3b --> |

The E4M3 configuration was a functional check; its speed was not measured.

### Time to first token

One request at a time, cold prefix cache.

| Configuration | Cold TTFT, 8K prompt | Cold TTFT, 32K prompt | Second-token latency, 32K prompt |
| --- | ---: | ---: | ---: |
| D+E1: AWQ, MTP4, GPU KV | <!-- TODO(0.2.0-acceptance): cold TTFT at 8K, D+E1 --> | <!-- TODO(0.2.0-acceptance): cold TTFT at 32K, D+E1 --> | <!-- TODO(0.2.0-acceptance): second-token latency at 32K, D+E1 --> |
| E2: Host KV, MTP | <!-- TODO(0.2.0-acceptance): cold TTFT at 8K, E2 --> | <!-- TODO(0.2.0-acceptance): cold TTFT at 32K, E2 --> | <!-- TODO(0.2.0-acceptance): second-token latency at 32K, E2 --> |
| E3a: NVFP4, MTP5 | <!-- TODO(0.2.0-acceptance): cold TTFT at 8K, E3a --> | <!-- TODO(0.2.0-acceptance): cold TTFT at 32K, E3a --> | <!-- TODO(0.2.0-acceptance): second-token latency at 32K, E3a --> |

### GPU KV capacity

<a id="gpu-kv-capacity-by-method"></a>

KV capacity at `--max-model-len 524288`, read from the startup log.

| Configuration | Target / draft KV | Where | KV capacity, tokens | Capacity ratio |
| --- | --- | --- | ---: | ---: |
| C: AWQ, DFlash2 | INT8 / FP16 | GPU | <!-- TODO(0.2.0-acceptance): GPU KV capacity tokens, C --> | <!-- TODO(0.2.0-acceptance): capacity ratio at 512K, C --> |
| D+E1: AWQ, MTP4 | INT8 / INT8 | GPU | <!-- TODO(0.2.0-acceptance): GPU KV capacity tokens, D+E1 --> | <!-- TODO(0.2.0-acceptance): capacity ratio at 512K, D+E1 --> |
| E4M3: AWQ, MTP4 | E4M3 / E4M3 | GPU | <!-- TODO(0.2.0-acceptance): GPU KV capacity tokens, E4M3 --> | <!-- TODO(0.2.0-acceptance): capacity ratio at 512K, E4M3 --> |
| E2: Host KV, MTP | <!-- TODO(0.2.0-acceptance): target / draft KV format, E2 --> | Host pool | <!-- TODO(0.2.0-acceptance): host KV pool tokens, E2 --> | <!-- TODO(0.2.0-acceptance): capacity ratio at 512K, E2 --> |
| E3a: NVFP4, MTP5 | INT8 / INT8 | GPU | <!-- TODO(0.2.0-acceptance): GPU KV capacity tokens, E3a --> | <!-- TODO(0.2.0-acceptance): capacity ratio at 512K, E3a --> |
| E3b: NVFP4, DFlash2 | <!-- TODO(0.2.0-acceptance): target / draft KV format, E3b --> | GPU | <!-- TODO(0.2.0-acceptance): GPU KV capacity tokens, E3b --> | <!-- TODO(0.2.0-acceptance): capacity ratio at 512K, E3b --> |

### Coming back to a document

When a new request starts with a prompt the engine has seen before, such as a follow-up question about the same document, ReVolta reuses the stored KV instead of computing it again. The table gives the time to first token, one request at a time. "Second request" follows directly after the first; in C the prefix is still in the GPU prefix cache, and in E2 the main KV history is in host memory. "From CPU memory" follows after the prefix was evicted from GPU memory. "From SSD after a restart" follows an engine restart.

| Configuration | Prompt tokens | Recomputed | Second request | From CPU memory | From SSD after a restart | Prompt tokens hit |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C: AWQ, DFlash2 | 32,768 | <!-- TODO(0.2.0-acceptance): cold TTFT at 32K, C --> | <!-- TODO(0.2.0-acceptance): TTFT with GPU prefix hit at 32K, C --> | <!-- TODO(0.2.0-acceptance): TTFT from CPU tier at 32K, C --> | <!-- TODO(0.2.0-acceptance): TTFT from SSD tier after restart at 32K, C --> | <!-- TODO(0.2.0-acceptance): prompt tokens hit at 32K, C --> |
| C: AWQ, DFlash2 | 131,072 | <!-- TODO(0.2.0-acceptance): cold TTFT at 128K, C --> | <!-- TODO(0.2.0-acceptance): TTFT with GPU prefix hit at 128K, C --> | <!-- TODO(0.2.0-acceptance): TTFT from CPU tier at 128K, C --> | <!-- TODO(0.2.0-acceptance): TTFT from SSD tier after restart at 128K, C --> | <!-- TODO(0.2.0-acceptance): prompt tokens hit at 128K, C --> |
| E2: Host KV, MTP | 32,768 | <!-- TODO(0.2.0-acceptance): cold TTFT at 32K, E2 --> | <!-- TODO(0.2.0-acceptance): TTFT with Host KV hit at 32K, E2 --> | — | <!-- TODO(0.2.0-acceptance): TTFT from SSD after restart at 32K, E2 (or "not measured") --> | <!-- TODO(0.2.0-acceptance): prompt tokens hit at 32K, E2 --> |

How the tiers work and how to size them: [KV offload modes](kv-offload.md).

### Task accuracy

Correct answers / questions on fixed subsets of GSM8K and MMLU.

| Configuration | GSM8K | MMLU |
| --- | ---: | ---: |
| C: AWQ, DFlash2, GPU KV | <!-- TODO(0.2.0-acceptance): GSM8K correct/total, C --> | <!-- TODO(0.2.0-acceptance): MMLU correct/total, C --> |
| D+E1: AWQ, MTP4, GPU KV | <!-- TODO(0.2.0-acceptance): GSM8K correct/total, D+E1 --> | <!-- TODO(0.2.0-acceptance): MMLU correct/total, D+E1 --> |
| E4M3: AWQ, MTP4, E4M3 KV | <!-- TODO(0.2.0-acceptance): GSM8K correct/total, E4M3 --> | <!-- TODO(0.2.0-acceptance): MMLU correct/total, E4M3 --> |
| E3a: NVFP4, MTP5 | <!-- TODO(0.2.0-acceptance): GSM8K correct/total, E3a --> | <!-- TODO(0.2.0-acceptance): MMLU correct/total, E3a --> |
| E3b: NVFP4, DFlash2 | <!-- TODO(0.2.0-acceptance): GSM8K correct/total, E3b --> | <!-- TODO(0.2.0-acceptance): MMLU correct/total, E3b --> |

For how closely each quantized checkpoint follows a BF16 reference, see [Quality measurements](#quality-measurements).

### Speculative decoding comparison

To compare the two speculative methods in 0.2.0, read C against D+E1 for AWQ (DFlash2 against MTP4) and E3b against E3a for NVFP4 (DFlash2 against MTP5). The AWQ pair also differs in draft KV format and in the CPU KV pool and disk tier; see [Configurations](#configurations). Earlier per-prompt results are in [DFlash2 against MTP4 on PCIe (0.1.2)](#dflash2-against-mtp4-on-pcie-012); which method to choose is in [Choosing a method](speculative-decoding.md#choosing-a-method).

## NVLink TP4 development measurements

The test system had eight V100 SXM2 32 GB GPUs in two fully connected four-GPU groups, with a 275 W power limit per GPU. The run used one group (TP4 + DCP2), NVFP4 with MTP5, INT8 main KV, a 512K context limit, `--gpu-memory-utilization 0.956`, a 48 GiB CPU KV pool and multimodal inputs enabled. It used the [NVLink auto-configuration](configuration.md#nvlink-tp4-auto-configuration) with `--no-async-scheduling` removed. The integration development build (`0.2.0.dev9012`) also included native INT8 decode and prefill changes. These are not final release acceptance numbers or a measurement of the profile change in isolation.

The synthetic short-prompt decode test used one English story template at temperature 1.0. The table includes only runs with no compilation during inference and reports time per decode round, not output throughput.

| Concurrent requests | Before auto-configuration (ms/round) | Development build with auto-configuration (ms/round) |
| ---: | ---: | ---: |
| 1 | 26.51 | 16.33 |
| 4 | 35.52 | 23.95 |
| 8 | 46.28 | 33.36 |

In a separate 32K single-request decode test, the development build took 16.04 ms per round. Cold prefill time to first token on that build was 1.149 s at 8K, 4.222 s at 32K and 8.405 s at 64K.

## Kernel changes during serving

<a id="triton-compilation-during-serving"></a>

**Triton compilation.** Some SM70 kernels used to recompile during requests when a buffer slice changed pointer alignment or an integer argument changed divisibility by 16. In 0.2.0 these arguments no longer specialize on those properties, and the existing startup warmup compiles the remaining known variants covered by the fix. The kernel bodies are unchanged. This applies to AWQ and NVFP4 with MTP or DFlash2 on PCIe and NVLink; it does not eliminate all compilation during serving.

**MTP shared expert.** Eligible SM70 TP4 MTP verification graph batches of 2–64 rows use dedicated shared-expert kernels without a new setting. The gate/up projection is fused with SiLU. The down projection is fused with the expert gate through 8 rows; larger eligible batches keep the down GEMM and fuse the gate dot product and multiply. This applies to AWQ and NVFP4, not DFlash2 ([#371](https://github.com/Leonccaa/ReVolta/pull/371)).

**Row-wise INT8 QSA prefill.** On SM70 TP4/DCP2, eligible prefill batches with FP16 queries, planar INT8 KV and at least 256 rows automatically read each row's own selected tokens instead of the union for a group of eight rows. This works with GPU or Host KV, AWQ or NVFP4, and MTP or DFlash2. Decode-only batches, CUDA graph capture and FP16 KV do not use this path. In a single-V100 PCIe kernel benchmark (12 heads, head dimension 256, one DCP2 rank, causal top-512), median attention time per chunk went from 22.5 to 10.8 ms at the 8K chunk end and from 63.1 to 12.6 ms at 32K. These are kernel timings, not service TTFT ([#372](https://github.com/Leonccaa/ReVolta/pull/372)).

**FP8 PLE table reads.** The pinned-host and hybrid PLE gathers skip table reads for rows owned by another rank or another tier instead of repeatedly reading row zero over PCIe. The change is automatic, does not change the output and does not apply to NVFP4-format PLE tables. In a single-V100 PCIe kernel benchmark (96 rows, a TP4 ID mix and the first gather after a 300 µs idle gap), median gather time fell from 125–132 µs to 6.1–7.0 µs across pinned host tables of 3–12 GiB. This is a gather timing, not a decode-round or service result ([#373](https://github.com/Leonccaa/ReVolta/pull/373)).

## Quality measurements

<a id="quantization-quality"></a>

These measurements show how closely each quantized checkpoint follows a BF16 reference. They were published with 0.1.0 (the exact build was not recorded) and have not been repeated with 0.2.0. All use scalar E4M3 KV unless the KV column says otherwise.

Two BF16 references are used:

- **Its own BF16 parent:** the model the checkpoint was quantized from. For Uncensored AWQ that is the uncensored fine-tune; for the other checkpoints it is the base model. The [model table in the README](../README.md#models) uses these values.
- **The base BF16 model** for every format. Against this reference, the Uncensored rows include the effect of fine-tuning as well as quantization.

| Checkpoint | KV | BF16 reference | KL ↓ | Top-1 agreement ↑ |
| --- | --- | --- | ---: | ---: |
| Uncensored AWQ g32 | E4M3 | Its own parent | 0.0214 | 97.08% |
| Uncensored AWQ g32 | E4M3 | Base model | 0.0273 | 96.63% |
| Uncensored AWQ g32 | INT8 | Base model | 0.0267 | Within 0.1 percentage point of E4M3 |
| Base AWQ g32 | E4M3 | Base model (its parent) | 0.0207 | 97.06% |
| Standard NVFP4 | E4M3 | Base model (its parent) | 0.0303 | 96.36% |
| NVIDIA FP4-Local | E4M3 | Base model (its parent) | 0.0380 | 95.91% |

A separate comparison against the Uncensored checkpoint's own parent put FP16 KV within 0.001 KL of E4M3: on the short set KL 0.0220 against 0.0214 with top-1 97.08% for both, and on the long set 0.0239 against 0.0238. NVIDIA NVFP4-QAD has not been measured.

KL measures how much the next-token probabilities change. Top-1 agreement counts how often the most likely next token is the same. Both are distribution measurements, not task accuracy or a percentage of model ability retained.

Method:

- Teacher forcing, measuring `KL(quantized ‖ BF16)` and top-1 agreement.
- Short set: 66 English, Chinese and code records; 39,101 tokens; contexts up to 4K.
- Long set: three natural documents (English 32K, Chinese 64K and SQLite source 128K) with 14,336 sampled positions.
- KL uses the top-20 log-probabilities plus a tail bucket. In one checked quantization, this estimate was 7–10% below the full-vocabulary KL.
- The BF16 reference also uses YaRN, so these values do not show the effect of YaRN on short texts; see [YaRN and native context](configuration.md#yarn-and-native-context).

Compare results from other datasets or estimators separately.

## Earlier results

<a id="reference-system-mtp4"></a>

These results were measured with earlier releases and are kept because 0.2.0 has no comparable measurement. 0.2.0 changed the default KV format to INT8, the serving defaults and both speculative methods, so these values do not describe 0.2.0.

### Standard NVFP4 against AWQ (0.2.0rc12)

Paired measurements on 0.2.0rc12 development builds (same configuration and time window, 4 × V100 PCIe, 250 W per GPU, TP4 + DCP2): NVFP4 and Uncensored AWQ ran at similar speed with MTP4 at one to eight requests and with DFlash2 at four and eight requests. With DFlash2 at one request, NVFP4 was slower (90.4 against 97.8 tok/s). NVFP4 weights took 21.48 GiB per GPU and Uncensored AWQ weights 22.01 GiB; the raw NVFP4 scales (the profile default) saved about 1.69 GiB per GPU compared with FP16 scales. <!-- TODO(0.2.0-acceptance): C1/C4/C8 tok/s, E3a NVFP4+MTP5 and E3b NVFP4+DFlash2 against D+E1 AWQ+MTP4 -->

### NVLink community result (0.1.2)

<a id="nvlink-community-result"></a><a id="cold-prefill"></a>

[@jttdev](https://github.com/jttdev) measured **0.1.2 at a 200 W cap per GPU** on 2026-10-24 ([report in #2](https://github.com/Leonccaa/ReVolta/issues/2#issuecomment-6384639530)): 4 × V100-SXM2 32 GB with an NV2 NVLink mesh, Uncensored AWQ `5d0ef929`, 512K views, `--gpu-memory-utilization 0.95`, `--max-num-seqs 8` and a 16 GiB CPU KV pool. DFlash2 used the 0.1.2 recommended settings, including the explicit 1500 MiB graph reserve.

| Requests and prompt | MTP4, tok/s | DFlash2, tok/s | DFlash2 / MTP4 |
| --- | ---: | ---: | ---: |
| 1, English code | 180.4 | **234.3** | 1.30× |
| 1, Chinese | 117.1 | 114.3 | 0.98× |
| 1, English reasoning | 184.9 | 228.2 | 1.23× |
| 2, aggregate | 231.8 | 260.1 | 1.12× |
| 4, aggregate | 337.6 | **367.8** | 1.09× |
| 8, aggregate | **516.3** | 513.0 | 0.99× |

The tests used the same three prompts, 512-token responses, thinking off and `temperature=0`. Single-request values are decode rates; concurrent values are aggregate output. DFlash2's KV pool held 1,216,131 tokens (capacity ratio 2.32× at 512K), 10.2% more than MTP4's 1,103,764 (2.11×).

Cold prefill used MTP4, one request at a time, token-ID prompts from Python standard-library sources, a nonce to miss the prefix cache, and `max_tokens=1`. Throughput is computed prompt tokens divided by time to first token. The 8K and 131K values are medians of three runs; the other lengths have two runs each.

| Prompt tokens | NVLink prefill, tok/s |
| --- | ---: |
| 8,192 | **4,680** |
| 32,768 | 4,505 |
| 131,072 | **4,037** |
| 262,144 | 3,618 |
| 460,000 | 3,007 |

On the PCIe reference system, 0.1.2 with MTP4 at 250 W per GPU measured 3,382 tok/s at 8K and 3,016 tok/s at 131K (512K views, one request at a time, cold prefix cache). The NVLink values are about 1.38× and 1.34× these; the machines and power caps differ. PCIe decode speeds for the same release are in the next table.

### DFlash2 against MTP4 on PCIe (0.1.2)

<a id="dflash2-against-mtp4"></a>

The PCIe reference system (four V100 32 GB PCIe GPUs under one PCIe switch) with 0.1.2: Uncensored AWQ g32, E4M3 KV, TP4 + DCP2, the 0.1.2 recommended DFlash2 settings, greedy decoding, thinking off, 512-token responses. One request: decode speed after the first token. Four and eight requests with mixed prompts: total output. With one prompt per category, the per-category values are indicative.

| Requests and prompt | DFlash2, tok/s | MTP4, tok/s |
| --- | ---: | ---: |
| 1, English code | 201 | 163 |
| 1, English reasoning | 195 | 163 |
| 1, agent code | 181 | 158 |
| 1, Chinese explanation | 101 | 105 |
| 1, Chinese chat | 85 | 100 |
| 4, mixed | 238 | 236 |
| 8, mixed | 331 | 357 |

English code and reasoning gained the most; Chinese text was slower with DFlash2. With 0.1.4 on the same system, DFlash2 ran about 20% faster than MTP4 at four concurrent requests and about 9% faster at eight (512-token answers), and a replay of 90 agent sessions, one request at a time, put DFlash2 at 0.98 of the MTP4 speed, within run-to-run variation (0.1.5 release notes; 0.1.5 speed is unchanged from 0.1.4).

## See also

- [Checkpoints, KV scales and MTP drafts](checkpoints.md)
- [DFlash2 speculative decoding](dflash2.md)
- [Dynamic speculative decoding](dynamic-speculative-decoding.md)
- [KV offload modes](kv-offload.md)
- [Built-in engine dashboard](dashboard.md) for live metrics of a running service
