# Support and known limits

This reference lists what ReVolta 0.2.0 supports and tests, and the limits you may hit. Check it before you change hardware, checkpoints, the speculative method, the KV mode or request limits. To install and launch, follow the [getting-started tutorial](getting-started.md).

Measurements on this page come from earlier releases, which used scalar E4M3 KV, unless the text gives another release or format. The 0.2.0 reference launch uses planar INT8 KV (`--kv-cache-dtype auto`); see [KV cache format](configuration.md#kv-cache-format).

## Support matrix

ReVolta is built for four or more V100 32 GB GPUs and validated on four. We recommend and test two interconnects: NVLink, or all GPUs under one PCIe switch with P2P enabled.

| Hardware | Status |
| --- | --- |
| 4 × V100 32 GB PCIe under one PCIe switch (PIX/PXB in `nvidia-smi topo -m`), P2P enabled | Reference system, with a 128 GiB container memory limit. Release acceptance runs here. |
| 4 × V100-SXM2 32 GB, NVLink | Tested by the community: [@jttdev](https://github.com/jttdev) ([#2](https://github.com/Leonccaa/ReVolta/issues/2)): speed, retrieval to about 490K tokens, vision; also [#37](https://github.com/Leonccaa/ReVolta/issues/37) and [#38](https://github.com/Leonccaa/ReVolta/issues/38) |
| The reference system with P2P disabled | Correct output; about 30–55% of P2P throughput (0.1.0) |
| V100 16 GB, TP2, TP8, other GPU counts | Not tested |

The release builds for SM70 only. Other GPU architectures, ROCm, CPU, XPU, TPU and GGUF are outside the build. There is no GPU CI: releases are checked on the reference system.

The supported model is Qwen3.8 Flash-Next. Checkpoints and their test status are in [Checkpoints](checkpoints.md#models). Other models have not been tested. The supported serving forms run tensor parallel 4 with decode context parallel 2 (DCP2) and combine:

- **Weights:** AWQ g32 or NVFP4.
- **Speculative decoding:** MTP with dynamic speculative decoding (DSD), or DFlash2.
- **KV placement:** GPU KV, optionally with CPU and disk offload, or Host KV.

0.2.0 release acceptance covers these configurations:

| Weights | Speculative decoding | KV | Tests | Result |
| --- | --- | --- | --- | --- |
| AWQ g32 | DFlash2, 7 drafts with DSD | GPU KV, INT8, CPU and disk offload | Full core tests (the maintainer's production form) | <!-- TODO(0.2.0-acceptance): acceptance result, C production form --> |
| AWQ g32 | MTP, 4 drafts with DSD | GPU KV, INT8, 48 GiB CPU pool | Reduced core tests (the [reference launch](getting-started.md#4-launch)) | <!-- TODO(0.2.0-acceptance): acceptance result, D+E1 public default AWQ + MTP4 --> |
| AWQ g32 | MTP, 4 drafts with DSD | GPU KV, scalar E4M3 | Functional check only | <!-- TODO(0.2.0-acceptance): functional check result, E4M3 --> |
| <!-- TODO(0.2.0-acceptance): weights and draft count, E2 Host KV + MTP --> | MTP | Host KV | Host KV path and offload readback | <!-- TODO(0.2.0-acceptance): acceptance result, E2 Host KV + MTP --> |
| NVFP4 | MTP, 5 drafts | GPU KV | Reduced core tests | <!-- TODO(0.2.0-acceptance): acceptance result, E3a NVFP4 + MTP5 --> |
| NVFP4 | DFlash2 | GPU KV | Reduced core tests | <!-- TODO(0.2.0-acceptance): acceptance result, E3b NVFP4 + DFlash2 --> |

Hy4-flash support is experimental and frozen: it was added in a 0.2.0 pre-release, and 0.2.0 acceptance does not cover it. It is text-only and needs exactly four V100 32 GB GPUs. Image and video input, the checkpoint's DFly2 speculative-decoding layers and pipeline parallelism are not supported. See [Hy4-flash](hy4-flash.md).

## Known limits

### Output reproducibility

A single request is reproducible bit for bit. When several requests are prefilled together, the result can differ from the same request run alone, so greedy outputs at 2–8 concurrent requests can change between runs. The cause is the tensor-parallel all-reduce: the order in which it sums each value depends on where the request's tokens sit in the batch. Where expert routing or sparse-attention block selection is close to a tie, the rounding-level difference can grow. In 0.1.1 tests, the prompt NLL changed by less than 0.015 nat/token on most text, and by up to about 0.3 on highly repetitive or table-like text. Against a BF16 reference, a request run alone and the same request in a batch were equally close. Fresh compiler autotuning can also change outputs; restarts that reuse the compile cache do not.

### Memory and host requirements

- **GPU memory.** The reference launch uses `--gpu-memory-utilization 0.956`; see [Memory and capacity](configuration.md#memory-and-capacity). Retest memory use when you change concurrency or media limits.
- **Host memory.** By default the AWQ PLE table needs about 48 GiB of pinned host memory. 80 GiB is the smallest tested container limit that passed memory stress on that path, with KV offload off (0.1.0). Budgets are in [System memory and storage](configuration.md#system-memory-and-storage).
- **PLE cascade.** Cascade (`"ple_disk_cascade": true` in `--kernel-config`) spills the part of the PLE table that does not fit in host memory to the checkpoint files. It is not supported in 0.2.0, and the ReVolta profile keeps it off. In earlier profiling with it enabled, it used about 4 GiB more GPU memory per card, which left room for about 345K context tokens.
- **Sleep level 2.** Not supported on the reference system (128 GiB host, PLE tables in host memory): it needs about 87 GiB of free host memory for the model buffers, while about 63 GiB is available (0.1.2). Sleep level 1 works.
- **Small KV pools.** Startup rejects a KV pool that cannot hold one maximum-length request.

#### Whole PLE table on disk (experimental)

Instead of pinned host memory, the whole PLE table can stay as file-backed mappings of the checkpoint's safetensors files, read by a separate PLE process. This is a different path from cascade and is off by default. It was first checked in 0.1.3 and measured in 0.1.4. To try it:

- Set `VLLM_PLE_CPU_OFFLOAD=1`, `VLLM_PLE_DISK_OFFLOAD=1` and `VLLM_SM70_QWEN38_HYBRID_PLE=0`.
- Add `--safetensors-load-strategy lazy`.
- Keep `"ple_disk_cascade": false` in `--kernel-config` (the checks also set `"ple_disk_release_pages": true`).
- CPU KV offload (`--kv-transfer-config`) can stay on; 0.1.4 measured it with a 32 GiB CPU pool.

Results:

- **0.1.3, Uncensored AWQ, E4M3 KV.** MTP4 served 131K-token prompts, 4 concurrent requests and images under both a 64 GiB and a 32 GiB container memory limit, and a 491K-token prompt under 64 GiB; DFlash2 also served under 64 GiB. Peak container memory was about 54 GiB under the 64 GiB limit; under 32 GiB, startup peaked at 16.5 GiB. The GPU KV pool was about 7% smaller (1,073,265 tokens with MTP4). Greedy outputs matched the pinned-memory path; one of eight sampled single-request outputs diverged at a near-tie, so the two paths are not bitwise identical. If the PLE process dies, the client gets an error after about 9 s and the engine exits after 11–13 s instead of hanging.
- **0.1.4, DFlash2 (recommended settings), Uncensored AWQ, E4M3 KV, 250 W, table on a local NVMe or SATA SSD.** Under a 64 GiB limit (NVMe and SATA), a 32 GiB limit (NVMe), and a 64 GiB limit with a 32 GiB CPU KV pool (NVMe and SATA), the engine started and served without running out of memory, and three 64-token greedy outputs matched the pinned-memory path exactly. Single-request decoding was 101.5–104.9 tokens/s against 135.5 with the pinned table, about 23–25% slower. A cold first token took 5.4–5.9 / 16.8–17.9 / 65.6–69.8 s at 8K / 32K / 131K prompt tokens, against 2.61 / 9.66 / 40.40 s, so 1.6–2.3 times as long. Once the table pages for a prompt had been read, the first token was within 2–3% of the pinned path (5–8% with the 32 GiB CPU KV pool). NVMe and SATA, and the 32 and 64 GiB limits, differed little. Peak memory of the serving container was about 57 GiB under the 64 GiB limit (62 GiB with the CPU KV pool) and 30 GiB under 32 GiB.
- **Not measured:** a machine with little physical RAM (the limits above were set on the container that runs the engine), and long-running stability on this path.

### Long context

Release retrieval was tested at 211K tokens, and the production build retrieved 20/20 needles at 489K (0.1.0). At 512K, the test covers capacity and the input/output boundary. <!-- TODO(0.2.0-acceptance): long-context result (32K / 128K), C production form -->

### Speculative decoding

- **DFlash2 CUDA graph reserve.** For DFlash2 the profile sets the graph reserve by itself: 1200 MiB with tail graphs off, or 1000 MiB (AWQ) and 1024 MiB (NVFP4) with the INT8 or `fp8_per_token_head` KV cache. If you set `VLLM_V2_CUDAGRAPH_MEM_MIB` yourself, keep it at or above the graph size the startup log reports; a smaller reserve can run out of GPU memory under concurrent long prompts. See [DFlash2 speculative decoding](dflash2.md).
- **Standard NVFP4 draft.** Use the Uncensored FP8 MTP draft, which the NVFP4 scale pack includes, as in validation. See [Checkpoints](checkpoints.md#standard-nvfp4-checkpoint-setup).
- **Structured output with MTP.** GPU tests produced 16/16 valid JSON and tool outputs (0.1.0). A later GPU test also reached the zero-draft step, where the grammar leaves no draft tokens (64 such steps): all outputs were valid, and repeated runs gave identical output.

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
