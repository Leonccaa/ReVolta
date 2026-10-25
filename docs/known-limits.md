# Support and known limits

This reference lists the tested support scope and known limitations. Check it before
using the [launch tutorial](getting-started.md) or changing hardware, formats or request limits.

The reference system uses four V100 32 GB PCIe cards behind one PCIe switch
(PIX/PXB in `nvidia-smi topo -m`), with P2P enabled and a 128 GiB container memory limit.

| Configuration | Test status |
| --- | --- |
| 4 × V100 32 GB PCIe, TP4 + DCP2, P2P | Full acceptance passed on the reference configuration |
| P2P disabled on the same machine | Correct output; about 30–55% of P2P throughput |
| 4 × V100-SXM2 32 GB, NVLink (NV2 mesh) | Community test by [@jttdev](https://github.com/jttdev) ([#2](https://github.com/Leonccaa/ReVolta/issues/2)): speed, retrieval to about 490K tokens, vision. Not part of our release gate |
| V100 16 GB, other GPU counts | Not yet tested |

**Full acceptance on the reference setup:** default Uncensored AWQ, Standard NVFP4,
and FP4-Local with their documented settings. Base AWQ has quality measurements.
80 GiB RAM has memory-stress results. Disk caching has readback results; its full
release gate passed with an 8 GiB CPU pool and a 40 GiB disk budget.

V100 16 GB, TP2/TP8, and other models have not been tested. NVLink has a community
test ([#2](https://github.com/Leonccaa/ReVolta/issues/2)) but is not part of our release gate.
This release builds SM70 only; other GPU architectures, ROCm, CPU, XPU, TPU, and
GGUF are outside the build.

- **Output reproducibility.** A single request is reproducible bit for bit. When several
  requests are prefilled together, the result can differ from the same request run alone,
  so greedy outputs at C2–C8 can change between runs. The cause is the tensor-parallel
  all-reduce: the order in which it sums each value depends on where the request's
  tokens sit in the batch. The rounding-level difference can then be amplified where
  expert routing or sparse-attention block selection is close to a tie. On most text
  the prompt NLL changes by less than 0.015 nat/token; highly repetitive or table-like
  text changed by up to about 0.3. Against a BF16 reference, a request run alone and the
  same request in a batch were equally close. Fresh compiler autotuning can also change
  outputs; restarts that reuse the compile cache do not.
- **Long context.** Release retrieval is tested at 211K. The previous production
  build retrieved 20/20 needles at 489K. At 512K, the test covers capacity and the
  input/output boundary.
- **Memory.** The default GPU utilization of 0.95 passed multimodal stress. Retest
  memory use when changing concurrency or media limits. The AWQ PLE table always
  needs about 48 GiB of host memory.
- **PLE cascade.** Profiling with `ple_disk_cascade=true` used about 4 GiB more
  GPU memory per 32 GB card. This left room for about 345K context tokens.
  Cascade is unsupported in 0.1.x. Keep it disabled.
- **FP4-Local vision.** Keep the default vision-tower TP4. Forcing data parallelism
  removes the tested memory arrangement. Non-empty request-level
  `mm_processor_kwargs` and `media_io_kwargs` are rejected with HTTP 400 by
  default. Only enable `--trust-request-mm-kwargs` for trusted API clients;
  their image settings can then raise preprocessing resource usage.
- **NVFP4 scales.** Different gate/up global scales are not supported by the
  raw-scale path. Both documented checkpoints use equal scales.
- **Structured output with MTP.** GPU tests produced 16/16 valid JSON/tool outputs. A
  later GPU test also reached the zero-draft step, where the grammar leaves no draft
  tokens (64 such steps): all outputs were valid, and repeated runs gave identical output.
- **Stop strings and thinking.** Since 0.1.1, with thinking on, a `stop` string matches
  only after the reasoning ends, and the reasoning/content split uses the `</think>`
  token, not its spelling. A `stop` of `</think>` therefore no longer ends a request at
  the end of the reasoning. With thinking off, everything is content, also in
  non-streaming responses.
- **Tool calls.** In `qwen3_coder` tool calls, a string argument that contains the
  literal text `</function>` is cut at that point. String arguments are streamed as
  they are generated.
- **Image placeholder text.** Text that contains the literal image placeholder
  `<|image_pad|>` can take the place of a real image in the same request. Upstream vLLM
  and HF behave the same (vllm-project/vllm#57740), and ReVolta follows them. Remove such tokens from
  untrusted text.
- **DFlash2.** Optional and off by default; see
  [DFlash2 speculative decoding](dflash2.md) for its settings
  and limits. From 0.1.3, the AWQ TP4 profile sizes the CUDA graph reserve for it (1200 MiB, tail graphs off).
  If you set the reserve yourself, keep it at or above the graph size the startup log
  reports; a smaller reserve can run out of GPU memory under concurrent long prompts.
- **Chat template.** The model's own chat template rejects a system message that is not
  the first message (`System message must be at the beginning.`). Agent stacks that send
  several system messages need to merge them.
- **Allocator.** Keep `NCCL_GRAPH_REGISTER=0` with expandable allocator segments.
- **Sleep level 2.** Not supported on the reference setup (128 GiB host, PLE tables in host
  memory): it needs about 87 GiB of free host memory for the model buffers while about
  63 GiB is available. Sleep level 1 works.
- **Small KV pools.** Startup rejects a pool that cannot hold one maximum-length request.
- **Prompt log-probabilities.** Long prompts are processed in chunks;
  `prompt_logprobs=20` measured about 2.3× ordinary prefill time.
- **Standard NVFP4 draft.** Use the FP8 `mtp-fp8/` draft from the Uncensored AWQ
  repository, as in validation.
- **Media.** On V100 with FlashAttention-V100, the default without `--limit-mm-per-prompt`
  is 1 image and no video, and a value that leaves out `"video"` disables video. This
  behaviour is kept. Set the option explicitly and include `"video"`: for example
  `'{"image": 8, "video": 1}'` (the generic image count) or
  `'{"image": 16, "video": 1}'` ([the reference launch](getting-started.md#4-launch)). The model has no audio input. On other hardware
  and backends, the defaults are 8 images, 1 video, and 1 audio item per whole request. Images in historical chat messages count
  toward the image limit. Qwen4Exp
  resizes large images to at most 1,048,576 pixels using HF's existing resize
  logic. Small text can become blurry; alignment near the area boundary can also
  change images whose source area is slightly below the limit.
  An empty `--mm-processor-kwargs '{}'`
  selects this default; restore the checkpoint image budget with
  `--mm-processor-kwargs '{"images_kwargs": {"size": {"longest_edge": 16777216, "shortest_edge": 65536}}}'`.
  Video sampling defaults to 16 frames. Download/decode ceilings remain separate.
  See [multimodal input limits](../.github/MULTIMODAL_INPUTS.md).

There is no GPU CI. Releases are checked on the reference machine.

## See also

- [Reference launch tutorial](getting-started.md)
- [Configuration reference](configuration.md)
- [Report an issue](../README.md#getting-help)
