# KV offload reference

This page lists every key and metric of the KV offload tiers and Host KV. How to choose and set up a tier is in [KV cache tiers and offload](kv-offload.md).

All keys below are per deployment and read at startup. Top-level keys of `kv_connector_extra_config` are not checked for spelling: a misspelled key is ignored. Unknown keys inside a disk tier entry, or inside an enabled `host_kv.ssd`, fail startup.

## Connector and CPU pool keys

Keys of `kv_connector_extra_config`:

| Key | Type | Default | Values | Effect and trade-off | Since | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `spec_name` | string | `CPUOffloadingSpec` | `TieringOffloadingSpec`, `CPUOffloadingSpec` | Offloading implementation. `TieringOffloadingSpec` is required for a disk tier and is what the reference launch uses. | vLLM | stable |
| `cpu_bytes_to_use` | int, bytes | none | > 0 | CPU pool size, summed over all workers. Required with GPU KV; rejected together with `host_store_gib` while the main K/V is on the host. | vLLM | stable |
| `eviction_policy` | string | `lru` | `lru`, `arc` | Eviction order of the CPU pool. Host KV requires `lru`. | vLLM | stable |
| `secondary_tiers` | list | `[]` | tier entries | Tiers below the CPU pool; see [Disk tier entry](#disk-tier-entry). Not used with Host KV. | vLLM; `revolta_fs` 0.2.0 | stable |
| `offload_prompt_only` | bool | `true` | | Offload only prompt blocks, not blocks generated while decoding. Suits reasoning models whose thinking is dropped before the next turn. | vLLM | stable |
| `demote_superseded_states` | bool | `true` | | See [Superseded recurrent states](kv-offload.md#superseded-recurrent-states). | 0.1.0 | stable |
| `superseded_state_rule` | string | `request` | `request`, `request_fixed`, `extend` | Which states count as superseded. `request`: the request's own states between its prefix hit and its last three blocks. `request_fixed`: the same, with the last block counted from the prompt instead of the output. `extend`: `request_fixed` plus the previous turn's end state once the request has stored a newer one. Other values are rejected at startup. | 0.2.0 | experimental |
| `mamba_state_slots_reference_tokens` | int | `--max-model-len` | > 0 | Request length used to size the recurrent-state slots of a grouped pool (MTP4). Smaller gives more state slots and fewer token slots, which helps many short sessions. The reference uses 65536. | 0.1.0 | stable |
| `cpu.draft_offload` | string | `all` | `all`, `window`, `none` | Which DFlash2 drafter blocks the CPU pool stores; see [DFlash2 drafter blocks](kv-offload.md#dflash2-drafter-blocks). `none` stores none, so DFlash2 gets no CPU prefix hits; it is only for measuring their cost with GPU KV, and Host KV with DFlash2 rejects it. Other values are rejected at startup. | 0.2.0 | experimental |
| `cpu.enable_dependency_eviction` | bool | `false` | | After a capacity eviction, also free CPU blocks that can provably no longer serve a hit. Only with a CPU-only LRU pool: any secondary tier, including a disk tier, disables it. No effect with Host KV. | 0.2.0 | experimental |
| `cpu.residency_stats` | bool | `false` | | Publish the `vllm:kv_offload_cpu_resident_blocks` gauge. No effect with Host KV. | 0.2.0 | experimental |
| `cpu.offload_counters` | bool | `false` | | Publish three diagnostic counters for development. Read-only: nothing depends on them. | 0.2.0 | experimental |
| `host_kv.ssd` | object | disabled | | SSD for Host KV; see [`host_kv.ssd` keys](#host_kvssd-keys). | 0.2.0 | stable |

The `cpu.*` booleans must be JSON booleans; other types are rejected at startup. Upstream keys not used by the reference configuration (`block_size`, `store_threshold`, `max_tracker_size`) keep their vLLM meaning; `TieringOffloadingSpec` and Host KV reject `store_threshold` of 2 or more.

## Disk tier entry

Keys of a `secondary_tiers` entry with `"type": "revolta_fs"`. An unknown key fails startup.

| Key | Type | Default | Values | Effect and trade-off | Since | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `type` | string | required | `revolta_fs` | `fs` is the upstream tier without verification; it rejects planar INT8 and per-token-head E4M3. | 0.2.0 | stable |
| `root_dir` | string | required | path | Dedicated directory for one engine, on a file system with user extended attributes. | vLLM | stable |
| `max_size_bytes` | int | required | > 0 | Budget for the cache page files under `root_dir`. | 0.1.0 | stable |
| `min_free_bytes` | int | 5% of the file system | ≥ 0; 0 disables | Free space kept on the file system. | 0.1.0 | stable |
| `max_files` | int | 1,000,000 | > 0 | Page count limit; bounds the in-memory index. | 0.1.0 | stable |
| `n_read_threads`, `n_write_threads` | int | 16, 16 | ≥ 0 each, at least 1 in total | I/O threads. The 0.1.4 readback measurement on a local NVMe SSD used one each. | vLLM | stable |
| `content_manifest_path` | string | none | path | Content manifest of the loaded checkpoint. Without it the tier is disabled. | 0.2.0 | stable |
| `content_verify` | string | `stat` | `stat`, `full` | `stat` checks size, modification time and inode of every listed file at startup (milliseconds). `full` also rehashes every file (about a minute per 100 GB) and catches a rewrite that restored the modification time. | 0.2.0 | stable |
| `mtime_update_interval_s` | float | `60` | ≥ 0; 0 = every use | Minimum time between modification-time refreshes of a used page; a due page refreshes its whole chain. The times order eviction after a restart: shorter is more precise and writes more metadata. | 0.2.0 | stable |
| `store_timeout_s` | float | `30` | ≥ 0; 0 = never | A store job, including its time in the queue, is reported failed after this time, so a stuck disk does not hold the CPU pool. | 0.2.0 | stable |
| `drain_timeout_s` | float | `30` | ≥ 0 | At shutdown, pending writes run for up to this time; the rest are dropped. Must fit in `--shutdown-timeout`. | 0.2.0 | stable |
| `max_io_retries` | int | `2` | ≥ 0 | Retries of a page read or write after a transient error (`EAGAIN`, `EINTR`, `EBUSY`, `ESTALE`, `ETIMEDOUT`, `EIO`). Checksum failures are not retried. | 0.2.0 | stable |
| `max_inflight_store_blocks` | int | `0` (auto) | ≥ 0 | Pages queued or being written above which new store jobs are skipped, so a slow disk does not hold the CPU pool. Auto is a quarter of the CPU pool's blocks, at least 64. | 0.2.0 | stable |
| `write_delay_s` | float | `300` | ≥ 0; 0 = write at once | See [Set the SSD write delay](kv-offload.md#set-the-ssd-write-delay). | 0.2.0 | stable |
| `write_warn_bytes_24h` | int | `1000000000000` | ≥ 0; 0 = no warning | Logs a warning at most once an hour when the submitted page bytes of the last 24 hours exceed it; writes continue. Not an SSD wear counter. | 0.2.0 | stable |
| `state_latest_only` | bool | `false` | | With delayed writes, write only a finished request's newest recurrent state (plus its junction and states other requests resume from). | 0.2.0 | stable |

## `host_kv.ssd` keys

Keys of the `host_kv.ssd` object in `kv_connector_extra_config` (Host KV only). With `enabled: true`, unknown keys fail startup.

| Key | Type | Default | Values | Effect and trade-off | Since | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `false` | | Turns the SSD on. Without it, nothing is created on disk. | 0.2.0 | stable |
| `root_dir` | string | required | path | As for the disk tier. | 0.2.0 | stable |
| `max_size_bytes` | int | required | > 0 | As for the disk tier. | 0.2.0 | stable |
| `min_free_bytes` | int | 5% of the file system | ≥ 0; 0 disables | As for the disk tier. | 0.2.0 | stable |
| `max_files` | int | 1,000,000 | > 0 | As for the disk tier. | 0.2.0 | stable |
| `content_manifest_path` | string | none | path | As for the disk tier; without it, SSD reuse is disabled. | 0.2.0 | stable |
| `content_verify` | string | `stat` | `stat`, `full` | As for the disk tier. | 0.2.0 | stable |
| `write_delay_s` | float | `300` | ≥ 0 | Delay before a published page is written. | 0.2.0 | stable |
| `write_bytes_per_second` | int | `67108864` (64 MiB/s) | > 0 | Rate limit for page writes, shared by the write threads. | 0.2.0 | stable |
| `write_threads` | int | `4` | > 0 | Concurrent page writes. | 0.2.0 | stable |
| `max_pending_blocks` | int | `64` | > 0 | Write jobs queued or running at once. | 0.2.0 | stable |
| `write_warn_bytes_24h` | int | `1000000000000` | ≥ 0 | As for the disk tier. | 0.2.0 | stable |
| `prefetch_max_inflight` | int | `16` | > 0 | Page reads from disk in flight, in total. | 0.2.0 | stable |
| `prefetch_max_per_request` | int | `8` | > 0 | Page reads from disk in flight per request. | 0.2.0 | stable |
| `prefetch_timeout_s` | float | `5` | ≥ 0 | Time budget per read window; a restore that needs several windows gets that many times this value. | 0.2.0 | stable |
| `shutdown_flush_timeout_s` | float | `10` | ≥ 0 | Upper limit of the final flush at shutdown, within `--shutdown-timeout` minus 5 s. | 0.2.0 | stable |
| `shutdown_io_grace_s` | float | `2` | ≥ 0 | Time for writes already started to finish at the end of the final flush. | 0.2.0 | stable |

## Host KV keys

Keys of `long_context_kv` in `--kernel-config` (all inert while `enabled` is `false`), plus related settings:

| Key | Type | Default | Values | Effect and trade-off | Since | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `enabled` | bool | `false` | | Turns Host KV mode on. | 0.2.0 | stable |
| `host_store_gib` | float | none | > 0 | Host store in GiB, summed over workers: main K/V, recurrent states and drafter pages. Set exactly one of `host_store_gib`, `host_pool_gib` and `host_pool_tokens`. | 0.2.0 | stable |
| `host_pool_gib`, `host_pool_tokens` | float, int | none | > 0 | Older sizing of the main K/V pool only; with `cpu_bytes_to_use` the two are added into one store. Use `host_store_gib` instead. | 0.2.0 | experimental |
| `main_kv_placement` | string | `host` | `host`, `gpu` | Leave at `host`. `gpu` is a comparison setting that keeps the main K/V on the GPU; a CPU pool behind the connector then needs its own `cpu_bytes_to_use` (or remove `--kv-transfer-config`). | 0.2.0 | experimental |
| `draft_main_placement` | string | `host` | `host`, `gpu` | See [MTP draft placement](kv-offload.md#mtp-draft-placement). | 0.2.0 | experimental |
| `hot_pool_gib` | float or `"auto"` | unset (auto) | > 0 | Guaranteed minimum of hot slots, from a per-GPU GiB budget; not an upper limit. | 0.2.0 | experimental |
| `hot_pool_ratio` | float | unset | (0, 1] | Guaranteed minimum of hot slots as a fraction of the host pool's tokens per rank. Not together with `hot_pool_gib`. | 0.2.0 | experimental |
| `prefill_staging_gib` | float | `0` | ≥ 0 | GPU memory per GPU reserved for staging history during appended prefill. | 0.2.0 | experimental |
| `state_blocks` | int | unset | > 0 | GPU state blocks (recurrent states, drafter windows, checkpoints). Unset reserves the worst case for `--max-num-seqs` requests. | 0.2.0 | experimental |
| `latest_only` | bool | `true` | | See [Checkpoint retention](kv-offload.md#checkpoint-retention). | 0.2.0 | experimental |
| `enable_junction_evidence` | bool | `true` | | See [Checkpoint retention](kv-offload.md#checkpoint-retention). | 0.2.0 | experimental |
| `superseded_keep_ssd` | bool | `false` | | See [Checkpoint retention](kv-offload.md#checkpoint-retention). | 0.2.0 | experimental |
| `kernel_config.qsa_dcp_index_shard` | bool | unset | `true`, `false` | Split the compressed index over all TP ranks at DCP2. Unset turns it on under the ReVolta profile for DFlash2, and for MTP in Host KV mode. Needed for MTP draft placement on the host at DCP2. Part of the SSD page identity. | 0.2.0 | stable |
| `speculative_config.gdn_state_replay` | bool | unset (on) | `true`, `false` | MTP recurrent-state replay; see [GDN state replay](speculative-decoding.md#gdn-state-replay). | 0.2.0 | stable |
| `speculative_config.gdn_state_replay_graph` | bool | unset (on for MTP) | `true`, `false` | Run the replay as a CUDA graph for full verify shapes. | 0.2.0 | stable |

## Metrics and log lines

| Name | Kind | Enabled by | Meaning |
| --- | --- | --- | --- |
| `vllm:external_prefix_cache_queries_total`, `vllm:external_prefix_cache_hits_total` | Prometheus counters | any connector | Prompt tokens looked up in, and served from, the offload tiers (CPU and disk together). |
| `vllm:kv_offload_total_bytes_total{transfer_type}`, `vllm:kv_offload_total_time_total{transfer_type}`, `vllm:kv_offload_size` | Prometheus | CPU offload | Bytes, time and transfer sizes for `GPU_to_CPU` and `CPU_to_GPU`. |
| `vllm:kv_offload_cpu_evicted_blocks_total{reason}` | Prometheus counter | CPU offload | CPU blocks evicted: `capacity`, `superseded` (after demotion), `reset`, `dependency`. |
| `vllm:kv_offload_cpu_draft_blocks_skipped_total` | Prometheus counter | `cpu.draft_offload` | DFlash2 drafter blocks not stored because of the setting. The series appears with the first skip. |
| `vllm:kv_offload_cpu_resident_blocks{kind,status}` | Prometheus gauge | `cpu.residency_stats` or `cpu.enable_dependency_eviction` | Occupied CPU slots by kind and usability, sampled at most once a minute. |
| `vllm:kv_offload_cpu_stored_blocks_total{kind}`, `vllm:kv_offload_lookup_shrink_tokens_total{cause}`, `vllm:kv_offload_hit_draft_window_needed_total{needed}` | Prometheus counters | `cpu.offload_counters` | Diagnostic counters for development. |
| `vllm:store_events_total{event}`, `vllm:store_state` | Prometheus | store metrics (on by default; `--no-enable-store-metrics` turns them off) | Scheduler store counters and gauges. With Host KV, host store and SSD events such as `host_ssd_evict_unwritten`; with a disk tier, `delayed_revolta_fs_dropped_evicted` and `delayed_revolta_fs_dropped_superseded`. |
| `Offload counters: ...` | log, at most once a minute | CPU offload | Cumulative CPU pool totals (demoted states, stored and evicted blocks, skipped drafter blocks). |
| `Delayed writes (revolta_fs)`, `SSD tier stats (revolta_fs)` | log, at most once a minute | disk tier with delayed writes | See [Set the SSD write delay](kv-offload.md#set-the-ssd-write-delay) and [Check that it works](kv-offload.md#4-check-that-it-works). |
| `long-context KV worker=N ...` | log, startup | Host KV | The Host KV capacity plan. |
| `Store stats: counters=... gauges=...` | log, at most once a minute | store metrics (on by default) | The same counters and gauges as `vllm:store_events_total` and `vllm:store_state`. |

## See also

- [KV cache tiers and offload](kv-offload.md): choose and set up a tier
- [Configuration reference](configuration.md): launch options and environment variables
- [Measured readback times](performance.md#coming-back-to-a-document)
