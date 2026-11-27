# Support and known limits

This reference lists what ReVolta 0.2.0 supports and tests, and the limits you may hit. Check it before you change hardware, checkpoints, the speculative method, the KV mode or request limits. To install and launch, follow the [getting-started tutorial](getting-started.md).

Measurements on this page come from earlier releases, which used scalar E4M3 KV, unless the text gives another release or format. The 0.2.0 reference launch uses planar INT8 KV (`--kv-cache-dtype auto`); see [KV cache format](configuration.md#kv-cache-format).

## Support matrix

ReVolta is built for four or more V100 32 GB GPUs and validated on four. We recommend and test two interconnects: NVLink, or all GPUs under one PCIe switch with P2P enabled.

| Hardware | Status |
| --- | --- |
| 4 × V100 32 GB PCIe under one PCIe switch (PIX/PXB in `nvidia-smi topo -m`), P2P enabled | Reference system, with a 128 GiB container memory limit. Release acceptance runs here. |
| 4 × V100-SXM2 32 GB, NVLink | Tested by the community: [@jttdev](https://github.com/jttdev) ([#2](https://github.com/Leonccaa/ReVolta/issues/2)): speed, retrieval to about 490K tokens, vision (0.1.2); also [#37](https://github.com/Leonccaa/ReVolta/issues/37) and [#38](https://github.com/Leonccaa/ReVolta/issues/38) |
| The reference system with P2P disabled | Correct output; about 30–55% of P2P throughput (0.1.0 release test) |
| V100 16 GB, TP2, TP8, other GPU counts | Not tested |

The release builds for SM70 only. Other GPU architectures, ROCm, CPU, XPU, TPU and GGUF are outside the build. There is no GPU CI: releases are checked on the reference system.

The supported model is Qwen3.8 Flash-Next. Checkpoints and their test status are in [Checkpoints](checkpoints.md#choose-a-checkpoint). The 0.2.0 release acceptance covers only Qwen3.8 Flash-Next; Hy4-flash has validation support (below). The supported serving forms run tensor parallel 4 with decode context parallel 2 (DCP2) and combine:

- **Weights:** AWQ g32 or NVFP4.
- **Speculative decoding:** MTP with dynamic speculative decoding (DSD), or DFlash2.
- **KV placement:** GPU KV, optionally with CPU and disk offload, or Host KV.

| KV mode | Status in 0.2.0 |
| --- | --- |
| GPU KV with the CPU KV pool | Stable; the [reference launch](getting-started.md#4-launch) uses a 48 GiB pool |
| `revolta_fs` disk tier below the CPU KV pool | Supported; the production form in the acceptance table below runs it. The on-disk format can change between releases, and pages are reused only by the same configuration; see [Disk tier](kv-offload.md#disk-tier-revolta_fs) |
| Host KV (Active Host KV) with MTP | Supported, off by default; in the 0.2.0 release acceptance (configuration E2 below) <!-- TODO(0.2.0-acceptance): acceptance result, E2 Host KV + MTP --> |
| Host KV with DFlash2 | Supported combination; not tested separately in the 0.2.0 release acceptance |

The modes are described in [KV offload](kv-offload.md).

0.2.0 release acceptance covers these configurations:

| Weights | Speculative decoding | KV | Tests | Result |
| --- | --- | --- | --- | --- |
| AWQ g32 | DFlash2, 7 drafts with DSD | GPU KV, INT8, CPU and disk offload | Full core tests (the maintainer's production form) | <!-- TODO(0.2.0-acceptance): acceptance result, C production form --> |
| AWQ g32 | MTP, 4 drafts with DSD | GPU KV, INT8, 48 GiB CPU pool | Reduced core tests (the [reference launch](getting-started.md#4-launch)) | <!-- TODO(0.2.0-acceptance): acceptance result, D+E1 public default AWQ + MTP4 --> |
| AWQ g32 | MTP, 4 drafts with DSD | GPU KV, scalar E4M3 | Functional check only | <!-- TODO(0.2.0-acceptance): functional check result, E4M3 --> |
| <!-- TODO(0.2.0-acceptance): weights and draft count, E2 Host KV + MTP --> | MTP | Host KV | Host KV path and offload readback | <!-- TODO(0.2.0-acceptance): acceptance result, E2 Host KV + MTP --> |
| NVFP4 | MTP, 5 drafts | GPU KV | Reduced core tests | <!-- TODO(0.2.0-acceptance): acceptance result, E3a NVFP4 + MTP5 --> |
| NVFP4 | DFlash2 | GPU KV | Reduced core tests | <!-- TODO(0.2.0-acceptance): acceptance result, E3b NVFP4 + DFlash2 --> |

Text-only Hy4-flash has validation support and is outside the 0.2.0 release acceptance; its status and limits are in [Hy4-flash](hy4-flash.md).

## Known limits

### Output reproducibility

A single greedy request is reproducible bit for bit with a fixed configuration. Sampled requests are reproducible only with a request `seed`; without one, each request gets a random seed. When several requests are prefilled together, the result can differ from the same request run alone, so greedy outputs at 2–8 concurrent requests can change between runs. The cause is the tensor-parallel all-reduce: the order in which it sums each value depends on where the request's tokens sit in the batch. Where expert routing or sparse-attention block selection is close to a tie, the rounding-level difference can grow. In 0.1.1 tests, the prompt NLL changed by less than 0.015 nat/token on most text, and by up to about 0.3 on highly repetitive or table-like text. Against a BF16 reference, a request run alone and the same request in a batch were equally close. Fresh compiler autotuning can also change outputs; restarts that reuse the compile cache do not.

### Memory and host requirements

- **GPU memory.** The reference launch uses `--gpu-memory-utilization 0.956`; see [Memory and capacity](configuration.md#memory-and-capacity). Retest memory use when you change concurrency or media limits.
- **Host memory.** By default the AWQ PLE table needs about 48 GiB of pinned host memory. 80 GiB is the smallest tested container limit that passed memory stress on that path, with KV offload off (0.1.0). Budgets are in [System memory and storage](configuration.md#system-memory-and-storage).
- **PLE cascade.** Cascade (`"ple_disk_cascade": true` in `--kernel-config`) spills the part of the PLE table that does not fit in host memory to the checkpoint files. It is not supported in 0.2.0, and the ReVolta profile keeps it off.
- **Sleep level 2.** Not supported on the reference system (128 GiB host, PLE tables in host memory): it needs more free host memory for the model buffers than is left. Sleep level 1 works.
- **Small KV pools.** Startup rejects a KV pool that cannot hold one maximum-length request.
- **First start.** With an empty compile cache, the reported GPU KV capacity can be one allocation unit lower than on later starts.

#### Whole PLE table on disk (experimental)

Instead of pinned host memory, the whole PLE table can stay as file-backed mappings of the checkpoint's safetensors files on a local SSD, read by a separate PLE process. This is a different path from cascade and is off by default. To try it:

- Set `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_DISK_OFFLOAD=1` and `VLLM_SM70_QWEN38_HYBRID_PLE=0`.
- Add `--safetensors-load-strategy lazy`.
- Keep `"ple_disk_cascade": false` in `--kernel-config`.
- CPU KV offload (`--kv-transfer-config`) can stay on, with a smaller pool.

Measured in 0.1.4 (Uncensored AWQ, DFlash2, E4M3 KV, 250 W, table on a local NVMe or SATA SSD): the engine started and served under 64 GiB and 32 GiB container memory limits, also with a 32 GiB CPU KV pool under 64 GiB, and greedy outputs matched the pinned-memory path. Single-request decoding was about 23–25% slower (101.5–104.9 against 135.5 tokens/s), and a cold first token took 1.6–2.3 times as long; once the table pages for a prompt had been read, the first token was within 2–8% of the pinned path. Peak memory of the serving container was about 57 GiB under the 64 GiB limit and 30 GiB under 32 GiB. Not measured: a machine with little physical RAM (the limits were set on the container) and long-running stability.

### Long context

Release retrieval was tested at 211K tokens, and the production build retrieved 20/20 needles at 489K (0.1.0). At 512K, the test covers capacity and the input/output boundary. <!-- TODO(0.2.0-acceptance): long-context result (32K / 128K), C production form -->

### Speculative decoding

- **CUDA graph reserve.** The profile sizes the graph reserve for the speculative method by itself ([values](configuration.md#cuda-graph-reserve)). If you set `VLLM_V2_CUDAGRAPH_MEM_MIB` yourself, keep it at or above the graph size the startup log reports; a smaller reserve can run out of GPU memory under concurrent long prompts.
- **Standard NVFP4 draft.** Use the Uncensored FP8 MTP draft, which the NVFP4 scale pack includes, as in validation. See [MTP drafts](checkpoints.md#mtp-drafts).
- **Structured output with MTP.** GPU tests produced valid JSON and tool outputs, also at the zero-draft step, where the grammar leaves no draft tokens.

### Checkpoints

- **NVFP4 scales.** The raw-scale path does not support different gate and up global scales. Both documented NVFP4 checkpoints use equal scales.
- **FP4-Local vision.** Keep the default vision-tower TP4. Forcing data parallelism removes the tested memory arrangement.

### API behaviour

- **Request-level media options.** Non-empty request-level `mm_processor_kwargs` and `media_io_kwargs` are rejected with HTTP 400 by default. Enable `--trust-request-mm-kwargs` only for trusted API clients: their image settings can raise preprocessing resource use.
- **Stop strings and thinking.** With thinking on, a `stop` string matches only after the reasoning ends, and the reasoning/content split uses the `</think>` token, not its spelling. A `stop` of `</think>` therefore does not end a request at the end of the reasoning. With thinking off, everything is content, also in non-streaming responses.
- **Tool calls.** In `qwen3_coder` tool calls, a string argument that contains the literal text `</function>` is cut at that point. String arguments are streamed as they are generated. `$ref` parameters can be treated as strings, and a failed `required` parse can still report `finish_reason="tool_calls"`. Stop tokens inside JSON strings and very large parallel-call deltas remain problematic; strict native XML structural tags are not supported.
- **Anthropic endpoint.** On `/v1/messages`, a response that `max_tokens` cuts inside a tool call streams the partial `tool_use` input, but a non-streaming response returns no `tool_use` block. `/v1/messages` and `/v1/messages/count_tokens` return HTTP 400 for a prompt longer than `--max-model-len` and HTTP 404 for an unknown model (#32); internal errors stay HTTP 500.
- **Chat template.** The model's own chat template rejects a system message that is not the first message (`System message must be at the beginning.`). Agent stacks that send several system messages need to merge them.
- **Image placeholder text.** Text that contains the literal image placeholder `<|image_pad|>` can take the place of a real image in the same request. Upstream vLLM and Hugging Face behave the same (vllm-project/vllm#57740), and ReVolta follows them. Remove such tokens from untrusted text.
- **Prompt log-probabilities.** Long prompts are processed in chunks; `prompt_logprobs=20` took about 2.3 times the ordinary prefill time (0.1.0).
- **Dashboard.** The [dashboard](dashboard.md) is optional and off by default (`--enable-dashboard`). It was tested with CPU-only suites; its cost while a model serves requests on GPUs has not been measured.

### Multimodal input

- On V100 with FlashAttention-V100, a launch without `--limit-mm-per-prompt` allows 1 image and no video per request, and a value that leaves out `"video"` disables video. Set the option and include `"video"`, as in the [reference launch](getting-started.md#4-launch); see [Multimodal input](configuration.md#multimodal-input). The model has no audio input.
- Images in earlier chat messages count toward the image limit.
- Large images are resized to at most 1,048,576 pixels with Hugging Face's resize logic. Small text can become blurry, and alignment near the area boundary can also change images whose source area is slightly below the limit. An empty `--mm-processor-kwargs '{}'` selects this default. To restore the checkpoint image budget, use `--mm-processor-kwargs '{"images_kwargs": {"size": {"longest_edge": 16777216, "shortest_edge": 65536}}}'`.
- Video sampling defaults to 16 frames. Download and decode ceilings are separate; see [multimodal input limits](../.github/MULTIMODAL_INPUTS.md).

## See also

- [Reference launch tutorial](getting-started.md)
- [Configuration reference](configuration.md)
- [Checkpoints, KV scales and MTP drafts](checkpoints.md)
- [Report an issue](../README.md#getting-help)
