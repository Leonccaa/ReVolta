# Performance and measurements

This reference collects measured speed, cache capacity, readback and quantization results.
Read the [hardware scope](known-limits.md) first; each group keeps its own version and test conditions.

## Measurements

### Speed

Version and test conditions: the PCIe values use 0.1.2; the NVLink overview combines 0.1.2 and 0.1.1.
See [DFlash2 against MTP4](#dflash2-against-mtp4), [Cold prefill](#cold-prefill) and
[NVLink](#nvlink-community-result) for the per-group conditions.

Best measured output speed in tokens/s, 512-token responses. The breakdown by speculative
method (MTP4 and DFlash2) and prompt type is in [Performance](#performance).

| System | 1 request | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| 4 × V100 32 GB PCIe, one PCIe switch, P2P, no NVLink (our reference) | **201** | **275** | **370** | **509** |
| 4 × V100-SXM2 32 GB, NV2 NVLink mesh ([community result](https://github.com/Leonccaa/ReVolta/issues/2)) | **234** | **260** | **368** | **525** |

| Cold prefill, tok/s | 8K tokens | 131K tokens |
| --- | ---: | ---: |
| PCIe reference | 3,382 | 3,016 |
| NVLink community | 4,680 | 4,037 |

### Coming back to a document

No ReVolta version is specified for this readback measurement; its request and cache conditions follow below.

A cached 32K-token document answered a follow-up in **1.28 seconds from memory**,
compared with **10.06 seconds when recomputed**. Both produced identical output.

| Prefix location | Request time |
| --- | ---: |
| GPU | 1.23 s |
| Recomputed, offload disabled | 10.06 s |
| CPU memory | 1.28 s |
| CPU memory + filesystem cache | 1.60 s |

The request generated 64 tokens and reused 31,008 cached prompt tokens. A small GPU
pool forced eviction. The filesystem tier was tested on a network filesystem;
local SSD performance has not yet been measured.

### Quantization quality

No ReVolta version is specified for these measurements. The reference and
[quality measurement method](#quality-measurements) apply to the table below.

The default AWQ format keeps the model close to its BF16 parent on the measured
English, Chinese, and code samples.

| AWQ checkpoint | BF16 reference | KL ↓ | Top-1 agreement ↑ |
| --- | --- | ---: | ---: |
| Uncensored AWQ g32 | Its own parent | 0.0214 | 97.08% |
| Base AWQ g32 | Base model | 0.0207 | 97.06% |

KL measures changes in next-token probabilities. Top-1 agreement counts how often
the most likely next token matches. These are distribution measurements, not task
accuracy or a percentage of model ability retained. See [Quality measurements](#quality-measurements).

## GPU KV capacity by method

Measured on our 4x V100 PCIe deployment with TP4 + DCP2, `--gpu-memory-utilization 0.95`, `--max-model-len 524288` (512K) and `--max-num-seqs 8`. Capacity and maximum concurrency use the startup log's convention: the concurrency figure is the KV token capacity divided by 524,288. It is a capacity ratio; 2.94× fits two full-length requests. Eight shorter requests are a separate tested workload.

| Format | Measured version | Method | GPU KV capacity, tokens | Maximum concurrency at 512K |
| --- | --- | --- | ---: | ---: |
| Uncensored AWQ g32 + E4M3 KV | 0.1.3, in testing | DFlash2 recommended (state replay off) | **1,379,634** | **2.63×** |
| Uncensored AWQ g32 + E4M3 KV | 0.1.3, in testing | DFlash2 with state replay | 1,542,558 | 2.94× |
| Uncensored AWQ g32 + E4M3 KV | 0.1.2 | DFlash2 | 1,268,831 | 2.42× |
| Uncensored AWQ g32 + E4M3 KV | 0.1.2 | MTP4 | 1,153,143 | 2.20× |
| Uncensored AWQ g32 + INT8 KV | 0.1.2 | MTP4 | 1,140,072 | 2.17× |
| Uncensored AWQ g32 + INT8 KV | — | DFlash2 | Unsupported | — |
| Uncensored AWQ g32 + FP16 KV | 0.1.2 | MTP4 | 673,749 | 1.29× |
| Uncensored AWQ g32 + FP16 KV | 0.1.2 | DFlash2 | 781,584 | 1.49× |
| NVIDIA FP4-Local + E4M3 KV | 0.1.2 | MTP4 | 1,321,612 | 2.52× |
| NVIDIA FP4-Local + E4M3 KV | 0.1.2 | DFlash2 | 1,335,042 | 2.55× |
| Standard NVFP4 + E4M3 KV | 0.1.2 | MTP4 | 1,250,448 | 2.39× |
| Standard NVFP4 + E4M3 KV | 0.1.2 | DFlash2 | 1,337,745 | 2.55× |

The 0.1.3 AWQ DFlash2 measurement includes state replay (#21), sharded selector codebooks (#9), and the 1200 MiB graph reserve with tail graphs off (#23). The FP16 KV, FP4-Local and NVFP4 DFlash2 values were measured on 0.1.2; they have not been remeasured with 0.1.3's state replay. The 0.1.2 AWQ DFlash2 measurements use a 1500 MiB graph reserve. INT8 KV is not supported by the DFlash2 drafter; 0.1.3 rejects this combination at startup.

## Performance

### Reference system, MTP4

Release configuration: Uncensored AWQ g32, E4M3 KV, TP4 + DCP2, MTP4, on the reference system.

| Metric | 0.1.0 |
| --- | ---: |
| GPU KV capacity, tokens (maximum concurrency at 512K) | 1,140,505 (2.18×) |
| Total output throughput, C1 | 152.5 |
| Total output throughput, C2 | 167.2 |
| Total output throughput, C4 | 262.2 |
| Total output throughput, C8 | 383.6 |
| 211K-token retrieval (5 needles) | 5/5 |
| 512K boundary: 524,224 input + 64 output | Passed; one more input token is rejected |

**Method.** Each output test requests 512 tokens, with thinking off,
`temperature=0`, streaming, and MTP4 enabled. English code, Chinese explanation,
and English arithmetic prompts rotate across clients. Each concurrency level
has one warm-up and eight measured runs; the table reports the mean.

At C2, the two requests share the first prefill step only if the second one reaches
the engine within a sub-millisecond window. With the same code, this happened in
0 to 8 of the 8 measured repetitions, varying from run to run, and C2 ranged
from 159.7 to 173.7 tok/s; the table's 167.2 came from a run with 4 of 8.
Repetitions where both requests started together reached about 174 tok/s.

Total output throughput is all generated tokens divided by wall time, including
time to first token. It is an end-to-end rate, not a decode-only measurement.
Prefill uses a cold prefix cache and divides computed prompt tokens by time to
first token.

Version-to-version speed comparisons use the same machine, configuration, and test
window. Other context lengths, media workloads, and thinking-mode outputs can give
different results.

### Cold prefill

The PCIe reference measured 0.1.2 with MTP4 at 250 W per GPU: **3,382 tok/s at 8K** and **3,016 tok/s at 131K**. With DFlash2 on 0.1.2, the same window measured 3,120 and 3,093 tok/s. These use 512K views, one request at a time and a cold prefix cache; throughput is computed prompt tokens divided by time to first token. The Measurements table uses the MTP4 values so both systems use the same speculative method. See [NVLink](#nvlink-community-result) for the community method and longer prompts.

### DFlash2 against MTP4

Reference system, 0.1.2 recommended DFlash2 config, greedy, thinking off,
512-token responses. One request: decode speed after the first token. Four and eight
mixed requests: total output. With one prompt per category, the per-category values are
indicative.

| Requests and prompt | DFlash2, tok/s | MTP4, tok/s |
| --- | ---: | ---: |
| 1, English code | 201 | 163 |
| 1, English reasoning | 195 | 163 |
| 1, agent code | 181 | 158 |
| 1, Chinese explanation | 101 | 105 |
| 1, Chinese chat | 85 | 100 |
| 4, mixed | 238 | 236 |
| 8, mixed | 331 | 357 |

By prompt category, 8 prompts per category, aggregate output; measured with the 0.1.2
code before the final width ladder, with the same verify widths at 2 and 8 requests:

| Requests | DFlash2: code / English / Chinese | MTP4: code / English / Chinese |
| --- | ---: | ---: |
| 2 | 275 / 203 / 129 | 232 / 193 / 141 |
| 8 | 505 / 336 / 285 | 492 / 332 / 327 |

A later 0.1.2 DFlash2 measurement at 250 W, with the recommended configuration, recorded the following aggregate output rates. The fastest code values below supply the four- and eight-request PCIe entries in the overview; the one- and two-request entries come from the preceding tables. Values in the overview are rounded to whole tokens/s.

| Requests | Code | English | Chinese |
| --- | ---: | ---: | ---: |
| 4 | 370.0 | 245.2 | 214.0 |
| 8, run 1 | 497.3 | 343.9 | 288.2 |
| 8, run 2 | 508.7 | 340.1 | 294.8 |

English code and reasoning gain the most; Chinese text is slower with DFlash2. On a
replay of real agent sessions (one request at a time, sampling), the two were on par
(0.97× the MTP4 speed). See [DFlash2 speculative decoding](dflash2.md).

### State replay in 0.1.3

GDN speculative state replay (`sm70_dflash2.spec_state_replay`, on by default with DFlash2) adds about 7 ms per decoding step when the target samples (30.6 to 37.9 ms per step at one request on our 4x V100 PCIe deployment) and costs 2 to 3% of throughput with greedy decoding at one request (5 to 7% at eight). The reference deployment therefore turns it off with `"sm70_dflash2": {"spec_state_replay": false}`; outputs are unchanged either way.

With replay off, DFlash2 keeps the other 0.1.3 memory changes (sharded selector codebooks, verifier tail graphs off, 1200 MiB graph reserve): 1,379,634 KV tokens at 512K against 1,268,831 in 0.1.2. Replay adds about 11% more (1,542,558); at eight concurrent 96K-token requests it lowered peak KV use from 60% to 26%. MTP4 remains the default speculative method.

### NVLink (community result)

[@jttdev](https://github.com/jttdev) measured **0.1.2 at a 200 W cap per GPU** on 2026-10-24 ([report in #2](https://github.com/Leonccaa/ReVolta/issues/2#issuecomment-6384639530)): 4 × V100-SXM2 32 GB, NV2 mesh, Uncensored AWQ `5d0ef929`, 512K views, `--gpu-memory-utilization 0.95`, `--max-num-seqs 8` and a 16 GiB CPU KV pool. DFlash2 used the 0.1.2 recommended settings, including the explicit 1500 MiB graph reserve.

| Requests and prompt | MTP4, tok/s | DFlash2, tok/s | DFlash2 / MTP4 |
| --- | ---: | ---: | ---: |
| 1, English code | 180.4 | **234.3** | 1.30× |
| 1, Chinese | 117.1 | 114.3 | 0.98× |
| 1, English reasoning | 184.9 | 228.2 | 1.23× |
| 2, aggregate | 231.8 | 260.1 | 1.12× |
| 4, aggregate | 337.6 | **367.8** | 1.09× |
| 8, aggregate | **516.3** | 513.0 | 0.99× |

The tests use the same three prompts, 512-token responses, thinking off and `temperature=0`. Single-request values are decode rates; concurrent values are aggregate output. MTP4 reached about 516 tok/s at eight requests. DFlash2's pool held 1,216,131 tokens (2.32× maximum concurrency at 512K), **10.2% more** than MTP4's 1,103,764 (2.11×). NVLink retained a 9% DFlash2 gain at four requests; eight requests were level.

Cold prefill used MTP4, one request at a time, token-ID prompts from Python standard-library sources, a nonce to miss the prefix cache, and `max_tokens=1`. Throughput is computed prompt tokens / time to first token. The 8K and 131K values are medians of three runs; the other lengths have two runs each.

| Prompt tokens | NVLink prefill, tok/s |
| --- | ---: |
| 8,192 | **4,680** |
| 32,768 | 4,505 |
| 131,072 | **4,037** |
| 262,144 | 3,618 |
| 460,000 | 3,007 |

The community report's 1.33× prefill comparison at 8K and 131K used the earlier README PCIe figures of 3,510 / 3,020. Against the 0.1.2 PCIe measurements above (3,382 / 3,016 at 250 W), the ratios are about 1.38× / 1.34×; the machines and power caps differ.

The overview uses the 0.1.2 DFlash2 values for one, two and four requests. Its eight-request value remains the higher 0.1.1 MTP4 measurement below: 524.6, rounded to 525 tok/s.

Earlier 0.1.1 results:

[@jttdev](https://github.com/jttdev) measured 0.1.1 on 4 × V100-SXM2 32 GB with an NV2
NVLink mesh ([#2](https://github.com/Leonccaa/ReVolta/issues/2)): MTP4, the same three
prompts, base AWQ checkpoint, 512K views, `--max-num-seqs 8`, `--gpu-memory-utilization 0.95` and a 16 GiB CPU
KV pool (`cpu_bytes_to_use`). Each level runs 3 requests per stream.

| | 1 request: code / Chinese / reasoning (decode) | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| 0.1.1 | 178 / 123 / 191 | 231.3 | 329.5 | 524.6 |
| 0.1.0, same session | 179 / 123 / 191 | 231.0 | 329.0 | 519.2 |

- The greedy outputs of both releases were byte-identical.
- At 262K context, `--gpu-memory-utilization 0.925` and a 32 GiB CPU pool, four requests
  reached 338 without the prefill-pressure flags (333 with them).
- The CPU KV pool size affects eight requests: in that 262K setup, eight requests gave
  513 tokens/s with a 16 GiB pool and 497 with 32 GiB, about 3% less; one to four
  requests did not change. An earlier 8-request gap against a hand-built 1Cat-vLLM setup
  did not reproduce in later runs.
- The GPU KV pool at 0.95 and 512K is 1,095,050 tokens with 0.1.1 (2.09× maximum concurrency at 512K), against 1,100,286 with 0.1.0 (2.10×);
  0.1.1 creates the NCCL communicators before memory profiling, which probably explains
  the 0.5% difference.
- Two concurrent retrieval requests of about 490K tokens each found all three needles
  (0.1.0); 0.1.1 found 3 of 3 at 91K and 244K tokens.

## Quality measurements

No ReVolta version is specified for this quality dataset and method.

Quality uses teacher forcing, measuring `KL(quantized ‖ BF16)` and top-1 agreement.

- Short set: 66 English, Chinese, and code records; 39,101 tokens; contexts up to 4K.
- Long set: three natural documents—English 32K, Chinese 64K, and SQLite source
  128K—with 14,336 sampled positions.
- KL uses top-20 log-probabilities plus a tail bucket. In one checked quantization,
  this estimate was 7–10% below full-vocabulary KL.

These KL values are lower than published KLD figures for other quantizations of this
model mainly because only the routed experts are 4-bit: the PLE tables are FP8 and all
other weights stay 16-bit. For comparison, llama.cpp Q8_0 of this model measures about
0.02 against BF16 on English text. Only English, Chinese and code were measured.

The AWQ quality table near the top compares each checkpoint with its own BF16 parent.
The [format table](checkpoints.md#quantization-comparison) uses the base BF16 model throughout. Results from other datasets
or estimators should be compared separately.

## See also

- [Checkpoint and KV formats](checkpoints.md)
- [DFlash2 setup](dflash2.md)
- [KV offload modes](kv-offload.md)
