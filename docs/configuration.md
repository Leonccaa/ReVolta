# Configuration reference

This page explains the reference deployment settings, sampling defaults and resource budgets.
Use the [getting-started tutorial](getting-started.md) for the installation and launch sequence.

## Environment settings

The engine detects SM70, TP4, a QSA model (Qwen3.8 Flash-Next) and FP16, and then sets
the tuned kernel switches by itself (the "ReVolta profile"). Only these deployment
settings remain:

The complete reference exports are in [step 3 of the tutorial](getting-started.md#3-environment).

Do not set other `VLLM_SM70_*` variables unless you are troubleshooting. Explicit
values override the profile.

## Interconnect

The engine selects the interconnect automatically and logs `SM70 interconnect=<mode>`.
P2P uses custom all-reduce for messages up to 80 KiB inside CUDA graphs; other
collectives use NCCL. Override with `VLLM_SM70_INTERCONNECT=auto|pex|nccl|nvlink`
or disable custom all-reduce with `--disable-custom-all-reduce`.

## Launch options

- **`draft_sample_method` must be `greedy`.** A probabilistic draft corrupts output
  above `max_num_seqs=2` at temperature > 0 (upstream 1Cat-vLLM issue #656). This
  setting is about draft verification only; clients can still use any sampling
  parameters.
- **`--gpu-memory-utilization 0.95`** passed the full gate, including the multimodal
  stress test with 8 concurrent requests. Higher values ran out of memory in earlier
  multimodal stress tests. If you change concurrency or media limits, measure again.
- The scheduler settings (`--prefill-pressure-*`, `--long-prefill-token-threshold`,
  `--mixed-prefill-step-latency-ms`) and `ple_disk_cascade: false` are the tested
  values for this machine. They are deployment settings, not universal defaults.
- `cpu_bytes_to_use` is the total for all four workers (48 GiB = 12 GiB per worker).
- **Keep `--limit-mm-per-prompt` on V100.** With FlashAttention-V100, the engine allows
  1 image and no video per request without it, and a value that leaves out `"video"`
  disables video. 0.1.3 keeps this V100 default. `{"image": 16, "video": 1}`
  keeps the reference deployment's limits; Qwen3.8 Flash-Next takes images and video,
  not audio, so no audio limit is set. To use the generic image count instead, set
  `--limit-mm-per-prompt '{"image": 8, "video": 1}'`; a lower image count leaves more
  host memory for bursts of large images. Large images are resized
  either way (see [Support and known limits](known-limits.md)).
- `--structured-outputs-config '{"backend": "xgrammar"}'` pins the structured-output
  backend. With `auto`, the engine keeps the backend chosen for its first
  structured-output request.
- `--sse-keep-alive-interval 15` sends an SSE comment line (`: keep-alive`) after 15 idle
  seconds on streaming responses, so proxies with read timeouts keep the connection open
  during queueing and prefill. The default, 0, sends none.
- `TILELANG_CACHE_DIR` ([step 3](getting-started.md#3-environment)) keeps TileLang's compiled kernels on persistent storage.
  Without it, a new container compiles them again.
- Media limits and request budgets: see
  [multimodal input limits](../.github/MULTIMODAL_INPUTS.md).

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
  `temperature=0`. The greedy `draft_sample_method` in the launch command applies only to
  draft verification, not to these request settings.

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

80 GiB is the smallest tested passing memory configuration. With less than 128 GiB,
reduce or disable the CPU KV pool.

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

## See also

- [Reference launch tutorial](getting-started.md)
- [Checkpoint preparation](checkpoints.md)
- [KV offload modes](kv-offload.md)
