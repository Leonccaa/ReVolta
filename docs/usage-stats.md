# Anonymous usage statistics

ReVolta sends a small anonymous report about the engine it is running, so that we can see which hardware, versions and configurations the deployment actually uses. The report contains no request data.

Sending is **on by default** and completely silent: a network failure, an unreachable collector or a read-only configuration directory never raises, never blocks the engine and never slows a request. Every error is dropped after one line of debug log, and a failed report is not retried — the next attempt is the next scheduled one.

## What is sent

One JSON object when the engine starts, then a short heartbeat every 10 minutes. The startup object describes the engine; the heartbeat only repeats the identifier and the timestamp.

| Field | Type | Meaning |
| --- | --- | --- |
| `uuid` | string | Random identifier generated at startup; it ties the heartbeats of one process together. It is not derived from the machine. |
| `provider` | string | Cloud provider detection, or `UNKNOWN`. |
| `num_cpu` | number | Number of CPU cores. |
| `cpu_type` | string | CPU brand string. |
| `cpu_family_model_stepping` | string | CPU family, model and stepping. |
| `total_memory` | number | Host memory size in bytes. |
| `architecture` | string | Machine architecture, for example `x86_64`. |
| `platform` | string | Platform description of the operating system. |
| `gpu_count` | number | Number of accelerators. |
| `gpu_type` | string | Accelerator name. |
| `gpu_memory_per_device` | number | Memory of one accelerator in bytes. |
| `cuda_runtime` | string | CUDA runtime version. |
| `xpu_runtime` | string | XPU runtime version, when the engine runs on XPU. |
| `revolta_version` | string | The ReVolta release this engine was built from. This is the field to group reports by. |
| `vllm_version` | string | The same value under the inherited field name. |
| `model_architecture` | string | Model class name, for example the architecture name in the checkpoint configuration. Not a path. |
| `context` | string | How the engine was started; `ENGINE_CONTEXT` for a served engine. |
| `env_var_json` | string | A serialized JSON object with a fixed allowlist of switches: `VLLM_USE_MODELSCOPE`, `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_USE_TRITON_AWQ` and `VLLM_ENABLE_V1_MULTIPROCESSING` (booleans), plus `VLLM_PP_LAYER_PARTITION` when it holds a numeric partition string such as `20,20`. |
| `source` | string | Deployment label from `VLLM_USAGE_SOURCE` (`production` by default). Anything that looks like a path or a URL is reported as `custom`. |
| `log_time` | number | Report time in nanoseconds since the epoch. |
| `dtype` | string | Model dtype. |
| `block_size` | number | KV cache block size. |
| `gpu_memory_utilization` | number | Fraction of device memory the KV cache may use. |
| `kv_cache_memory_bytes` | number | Explicit KV cache size, when one was requested. |
| `quantization` | string | Quantization method, for example `awq`. |
| `kv_cache_dtype` | string | KV cache dtype, for example `fp8`. |
| `enable_lora` | boolean | Whether LoRA adapters are enabled. |
| `enable_prefix_caching` | boolean | Whether prefix caching is enabled. |
| `enforce_eager` | boolean | Whether CUDA graph capture is disabled. |
| `disable_custom_all_reduce` | boolean | Whether the custom all-reduce is disabled. |
| `tensor_parallel_size` | number | Tensor parallelism. |
| `pipeline_parallel_size` | number | Pipeline parallelism. |
| `data_parallel_size` | number | Data parallelism. |
| `decode_context_parallel_size` | number | Decode-context parallelism. |
| `enable_expert_parallel` | boolean | Whether expert parallelism is enabled. |
| `all2all_backend` | string | All-to-all backend used by expert parallelism. |
| `kv_connector` | string | Name of the KV transfer connector, when one is configured. |
| `max_model_len` | number | Maximum model context length. |
| `enable_long_context_kv` | boolean | Whether the Host KV long-context mode is enabled. |
| `main_kv_placement` | string | Where the main KV history lives in long-context mode: `host` or `gpu`. |
| `speculative_method` | string | Speculative decoding method, when one is configured. |
| `num_speculative_tokens` | number | Number of drafted tokens per step, when speculative decoding is configured. |

The ReVolta part of the summary is the parallelism (including decode-context parallelism), the Host KV long-context mode and its main-KV placement, the speculative decoding method and its draft length, the context length and the KV cache dtype.

## What is never sent

Prompts, outputs, token counts of individual requests, model paths or repository names, served model names, host names, IP addresses, user names, environment variables outside the allowlist above, and any free-form string setting (paths, URLs, file names). Values that look like paths are dropped before sending, not redacted afterwards.

The report also goes to a local file, `usage_stats.json` in the directory given by `VLLM_CONFIG_ROOT`, with the same content. A write failure there is ignored the same way a send failure is.

## Where it goes

`https://ai.19b.net/api/usage` — the ReVolta collector. Point it elsewhere with `VLLM_USAGE_STATS_SERVER`; set that variable to an empty string to keep the local file and send nothing. Each attempt waits at most 5 seconds, on a background thread, with no retry.

## Turning it off

Any one of these disables collection and sending:

| Setting | Effect |
| --- | --- |
| `VLLM_NO_USAGE_STATS=1` | Upstream switch, honoured as-is: 1 turns the usage report off. |
| `VLLM_DO_NOT_TRACK=1` or `DO_NOT_TRACK=1` | [Do Not Track](https://contribute.ispr.io/what-do-not-track-dnt-is-and-how-to-enable-it-in-all-of-your-browsers/) convention, honoured as-is. |
| A `do_not_track` file in `VLLM_CONFIG_ROOT` | File-based opt-out (by default `~/.config/vllm/do_not_track`). |

The first engine start with collection on logs one `INFO` line saying that anonymous statistics are sent, what they contain, how to turn them off and where this page is.

## See also

- [Configuration](configuration.md#environment-variable-reference) for the switches above.
- [Known limits](known-limits.md) for what this release supports.
