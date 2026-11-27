# Anonymous usage statistics

ReVolta sends a small anonymous report about the engine it runs on: hardware, platform, versions and a summary of the engine configuration. We use it to see which hardware, versions and configurations deployments actually use. The report contains no prompts, outputs or other request data.

Sending is **on by default**. To turn it off, set `VLLM_NO_USAGE_STATS=1`; the other switches are in [Turning it off](#turning-it-off).

## When it is sent

- **At engine start and every 10 minutes.** The engine sends one report when it starts. A heartbeat follows every 10 minutes; it repeats the startup report with a new `log_time` and the engine uptime in `uptime_s`, so a lost startup report does not lose the configuration.
- **Once per engine.** Only the first GPU worker sends it, so a four-GPU tensor-parallel engine sends one report, not four.
- **In the background, never in the way.** A background thread sends the report. Each attempt waits at most 5 seconds. A network error, an unreachable collector, a read-only configuration directory or any other failure is logged at debug level and dropped. A failed report is not retried; the next attempt is the next heartbeat.
- **Announced in the log.** When collection is on, each engine start logs one `INFO` line that says statistics are sent and where, what they contain, how to turn them off and where this page is.
- **Kept in a local file.** Every report, startup and heartbeat, is also appended as one JSON line to `usage_stats.json` in the directory given by `VLLM_CONFIG_ROOT` (default `~/.config/vllm`, or `$XDG_CONFIG_HOME/vllm` when `XDG_CONFIG_HOME` is set). The file grows by one line per report and is not rotated. A write failure is ignored like a send failure.

## What is sent

One flat JSON object. Fields that do not apply are sent as `null`.

| Field | Type | Meaning |
| --- | --- | --- |
| `uuid` | string | Random identifier created when the engine process starts. It ties the heartbeats of one process together; a restart creates a new one. It is not derived from the machine. |
| `provider` | string | Cloud provider: `AWS`, `AZURE`, `GCP` or `OCI` when the machine's DMI vendor strings name one, `RUNPOD` when `RUNPOD_DC_ID` is set, otherwise `UNKNOWN`. |
| `num_cpu` | number | Number of logical CPUs. |
| `cpu_type` | string | CPU brand string. |
| `cpu_family_model_stepping` | string | CPU family, model and stepping, separated by commas. |
| `total_memory` | number | Host memory in bytes. |
| `architecture` | string | Machine architecture, for example `x86_64`. |
| `platform` | string | Operating system, kernel release and C library, for example `Linux-6.8.0-47-generic-x86_64-with-glibc2.39`. |
| `gpu_count` | number | Number of visible GPUs. |
| `gpu_type` | string | Name of the first GPU, for example `Tesla V100-PCIE-32GB`. |
| `gpu_memory_per_device` | number | Memory of the first GPU in bytes. |
| `gpu_topology` | string | Sent only on CUDA with at least two visible GPUs. A JSON-encoded array with one entry per GPU pair. Each entry has `gpus` (two zero-based CUDA ordinals) and `type` (`NVLink`, `PCIe` or `unknown`). When known, `path` is `PIX`, `PXB`, `PHB`, `NODE` or `SYS`, and `nvlink_count` is the number of direct NVLink links, given only when every link could be checked. Values that cannot be read stay `unknown`; nothing is guessed. The device UUIDs and PCI bus IDs used to build it are not sent. Heartbeats repeat the startup value. |
| `cuda_runtime` | string | CUDA runtime version, for example `12.8`. |
| `xpu_runtime` | string | XPU runtime version; `null` except on Intel XPU. |
| `revolta_version` | string | The ReVolta release the engine was built from. Group reports by this field. |
| `vllm_version` | string | The same value under the inherited field name. |
| `model_architecture` | string | Model class name the engine selected for the checkpoint. It is a class name, not a path or repository name. |
| `context` | string | How the engine was started. Always `ENGINE_CONTEXT` in this release. |
| `env_var_json` | string | A JSON object with a fixed allowlist of switches: `VLLM_USE_MODELSCOPE`, `VLLM_USE_FLASHINFER_SAMPLER`, `VLLM_USE_TRITON_AWQ` and `VLLM_ENABLE_V1_MULTIPROCESSING` (booleans, always present), plus `VLLM_PP_LAYER_PARTITION` only when it is set to a numeric partition such as `20,20`. |
| `source` | string | Deployment label from `VLLM_USAGE_SOURCE`: `production` by default, `production-docker-image` in the published Docker image. An empty value, or one that looks like a path or URL (see [screening](#what-is-never-sent)), is reported as `custom`. |
| `log_time` | number | Report time in nanoseconds since the Unix epoch. |
| `uptime_s` | number | Seconds since the startup report, on a monotonic clock; `0` in the startup report. |
| `dtype` | string | Model dtype, for example `torch.float16`. |
| `block_size` | number | KV cache block size. |
| `gpu_memory_utilization` | number | Fraction of each GPU's memory the engine may use (`--gpu-memory-utilization`). |
| `kv_cache_memory_bytes` | number | Explicit KV cache size per GPU, when one was requested. |
| `quantization` | string | Weight quantization method, for example `awq`. |
| `kv_cache_dtype` | string | KV cache dtype after `auto` is resolved, for example `int8` (the default on V100) or `fp8_e4m3`. |
| `enable_lora` | boolean | Whether LoRA adapters are enabled. |
| `enable_prefix_caching` | boolean | Whether prefix caching is enabled. |
| `enforce_eager` | boolean | Whether CUDA graph capture is disabled. |
| `disable_custom_all_reduce` | boolean | Whether the custom all-reduce is disabled. |
| `tensor_parallel_size` | number | Tensor parallel size. |
| `pipeline_parallel_size` | number | Pipeline parallel size. |
| `data_parallel_size` | number | Data parallel size. |
| `decode_context_parallel_size` | number | Decode context parallel size. |
| `enable_expert_parallel` | boolean | Whether expert parallelism is enabled. |
| `all2all_backend` | string | All-to-all backend setting for expert parallelism; sent even when expert parallelism is off. |
| `kv_connector` | string | Name of the KV transfer connector, for example `OffloadingConnector`, when one is configured. |
| `max_model_len` | number | Maximum context length. |
| `enable_long_context_kv` | boolean | Whether Host KV mode is enabled. |
| `main_kv_placement` | string | Host KV setting for where the main KV history lives: `host` or `gpu`. Sent even when Host KV is off (default `host`). |
| `speculative_method` | string | Speculative decoding method, for example `mtp` or `dflash` (DFlash2), when one is configured. |
| `num_speculative_tokens` | number | Draft tokens per step, when speculative decoding is configured. |

## What is never sent

- Prompts, outputs, and the token counts or timings of individual requests.
- Model paths, repository names and served model names.
- Host names, user names and IP addresses. The payload contains no address. Like any HTTPS request, the collector sees the network address the request comes from.
- GPU UUIDs, serial numbers and PCI bus IDs.
- Environment variables outside the allowlist in `env_var_json`. `RUNPOD_DC_ID` is only checked for presence to set `provider`; its value is not sent.
- Free-form settings such as paths, URLs and file names. Every configuration value is screened before the report is built: a string longer than 64 characters, or one that contains `/`, `\`, `:` or a space, is replaced by `null`. Allowlisted environment values must be booleans, numbers or short switch-like strings. Hardware and platform fields (`cpu_type`, `platform`, `gpu_type`) are sent as the system reports them.

## Where it goes

Reports go to the ReVolta collector at `https://usage.19b.net/api/` as an HTTPS `POST` with a JSON body. If you filter outbound traffic, this is the only host the report contacts.

Set `VLLM_USAGE_STATS_SERVER` to send the report to another URL. Set it to an empty string to send nothing while still writing the local `usage_stats.json` file.

## Turning it off

Any one of these turns collection off. With collection off, nothing is sent, nothing is written to `usage_stats.json`, and the `INFO` line is not logged.

| Setting | Notes |
| --- | --- |
| `VLLM_NO_USAGE_STATS=1` | The vLLM opt-out switch. |
| `VLLM_DO_NOT_TRACK=1` or `DO_NOT_TRACK=1` | The cross-tool Do Not Track convention. When `VLLM_DO_NOT_TRACK` is set to any non-empty value, `DO_NOT_TRACK` is not read. |
| A file named `do_not_track` in `VLLM_CONFIG_ROOT` | By default `~/.config/vllm/do_not_track`. The file's content does not matter. |

Only the exact value `1` turns collection off; `true` or `yes` do not.

In Docker, pass the variable to the container, for example `docker run -e VLLM_NO_USAGE_STATS=1 ...`. A `do_not_track` file on the host has no effect unless it is inside the container's `VLLM_CONFIG_ROOT`.

## See also

- [Configuration](configuration.md#environment-variable-reference) for the environment variables above.
- [Getting started](getting-started.md) for installation.
