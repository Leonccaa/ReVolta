# KV cache tiers and offload

<a id="kv-offload-modes"></a>

This page explains where ReVolta keeps the KV cache and how to set up each tier: GPU only, CPU offload, a disk tier, and Host KV for long contexts. Every key and metric is listed in the [KV offload reference](kv-offload-reference.md).

Before you start, have a working [reference launch](getting-started.md) and check the [host memory budget](configuration.md#system-memory-and-storage).

Terms used on this page:

- **KV cache**: the attention keys and values of tokens already processed. With GPU KV, the GPU holds the KV cache of running requests; the **prefix cache** keeps it after a request finishes, so a later request with the same prompt start skips that part of prefill.
- **Offload**: when the GPU prefix cache evicts blocks, a lower **tier** keeps a copy: the **CPU KV pool** in host memory (the "CPU pool" below), then optionally the **`revolta_fs` disk tier** on a local SSD. A later request with the same prefix reads it back instead of computing it again.
- **Host KV** (Active Host KV): a long-context mode in which the main K/V history lives in host memory. The GPU keeps the compressed index, the recurrent state and a working set of hot pages.

## Choose a mode

| Mode | Status in 0.2.0 | Where evicted prefixes go | Extra host memory | Choose it when |
| --- | --- | --- | --- | --- |
| [GPU KV only](#gpu-kv-only) | Stable | Nowhere: they are computed again | None | Host memory is tight |
| [GPU KV + CPU offload](#cpu-offload-reference-configuration) | Stable; used by the reference launch | The CPU KV pool in `/dev/shm` | `cpu_bytes_to_use` (48 GiB in the reference) | Most deployments |
| [GPU KV + CPU + disk](#disk-tier-revolta_fs) | Supported; used by the production form in the 0.2.0 acceptance (configuration C) | The CPU KV pool, then the `revolta_fs` disk tier on a local SSD | The CPU pool, plus SSD space | Cached documents should survive a restart |
| [Host KV](#host-kv-long-context) | Supported; validated with MTP (acceptance configuration E2) | Main K/V lives in host memory; optional SSD | `host_store_gib` | Long contexts or many long requests do not fit in GPU KV |

All modes use the same model, KV format (INT8 by default) and speculative decoding settings. Only `--kv-transfer-config` and, for Host KV, `--kernel-config` change.

## GPU KV only

Remove `--kv-transfer-config` from the launch command. Evicted prefixes are computed again. Host memory then holds only the PLE table and the usual process memory; see [System memory and storage](configuration.md#system-memory-and-storage) for the tested limits.

To check the GPU pool, read the startup log:

- `GPU KV cache size: N tokens`: the tokens the GPU pool can hold.
- `Maximum concurrency for L tokens per request: X`: how many requests of the full `--max-model-len` fit at once.

For measured pool sizes by configuration, see [GPU KV capacity](performance.md#gpu-kv-capacity).

### Recurrent-state replay

<a id="gdn-state-replay"></a>

During speculative decoding, MTP and DFlash2 replay the accepted tokens into the GDN recurrent state, so each running request keeps two state blocks per GDN cache group instead of one per draft position. The saved GPU memory goes to the KV cache. Replay is on by default in the reference launch; requirements, settings and measured costs are in [GDN state replay](speculative-decoding.md#gdn-state-replay).

## CPU offload (reference configuration)

The reference launch uses this `--kv-transfer-config`:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 51539607552,
    "eviction_policy": "lru",
    "secondary_tiers": [],
    "demote_superseded_states": true,
    "mamba_state_slots_reference_tokens": 65536
  }
}
```

`cpu_bytes_to_use` is the pool size in bytes, summed over all workers (51539607552 bytes = 48 GiB). The other keys are explained in the [reference](kv-offload-reference.md#connector-and-cpu-pool-keys).

**Host memory.** The pool is a shared-memory file in `/dev/shm`. It is allocated and touched at startup, so the whole pool is resident from the start. In a container, make `--shm-size` larger than `cpu_bytes_to_use` (the reference container uses `--shm-size 128g`). 128 GiB of RAM is recommended for the 48 GiB pool; with less, make the pool smaller or turn offload off ([System memory and storage](configuration.md#system-memory-and-storage)). At startup, the line `Qwen4Exp PLE host preflight (...)` lists the PLE share, the CPU KV pool and the available memory; if it is logged as a warning, the two together may not fit.

**Capacity.** With MTP4, a 48 GiB pool held about 2.4 million tokens, about 21 KB per token (0.1.0, E4M3 KV). With INT8 KV in the 0.2.0 MTP4 test, the same 48 GiB grouped CPU pool had 1,455 token-page slots of 1,632 tokens each (2,374,560 tokens of page capacity), with separate slots for recurrent state. With DFlash2, the drafter uses a different cache block size, so every cache group takes pages of the full block size: a prefix needs about 50 KB per token in the CPU pool and on disk (measured in 0.1.4 with E4M3 KV: 1.84 GB for a 32K-token prefix, 6.36 GB for 131K). The same 48 GiB pool then holds about 1 million tokens (an estimate from the same 50 KB per token).

**Check that it works.** At startup, look for:

- `Creating offloading spec with name: TieringOffloadingSpec`.
- With MTP4: `Grouped CPU offload pool: N token slots, M state slots per Mamba group; ...`. With DFlash2: `Created TieringOffloadingManager with primary tier (lru, N blocks) and 0 secondary tier(s)`.
- `Created mmap file /dev/shm/vllm_offload_<id>.mmap (X GB)` (with MTP4, one `vllm_offload_<id>_g<group>.mmap` file per cache group), each followed shortly by `Removed mmap file ...`. The name is removed on purpose; the memory stays mapped.

While serving, the periodic stats line shows `External prefix cache hit rate: X%`. Prometheus has the same data as `vllm:external_prefix_cache_hits_total` and `vllm:external_prefix_cache_queries_total` (in tokens), and the transferred bytes as `vllm:kv_offload_total_bytes_total{transfer_type="GPU_to_CPU"}` and `{transfer_type="CPU_to_GPU"}`.

**Stopping and restarting.** The engine removes the pool's file name from `/dev/shm` as soon as every process has mapped it, so the kernel frees the memory when the last process exits, even after a kill (fixed in 0.2.0, #51). Only a crash during startup, before that point, can leave a `vllm_offload_*.mmap` file behind. Remove such files when no engine is running, or recreate the container.

### Superseded recurrent states

Each conversation turn stores a new recurrent state at its end. With `demote_superseded_states` (default `true`), when a request finishes, the states it made obsolete move to the front of the CPU pool's eviction order, so they are evicted before other conversations' reusable prefixes. Nothing is deleted, and a demoted state can still serve a hit while it is in the pool.

States at a shared-prefix junction are never demoted. `superseded_state_rule` (experimental) changes which states count as obsolete; the default suits most deployments, and the values are in the [reference](kv-offload-reference.md#connector-and-cpu-pool-keys).

### DFlash2 drafter blocks

The DFlash2 drafter has its own sliding-window KV cache. `cpu.draft_offload` selects which of its blocks the CPU pool stores. It has no effect with MTP, whose draft layer uses full attention; if it is set to another value than `all` with MTP, the scheduler logs that it has no effect.

- `all` (default): store every drafter block.
- `window`: store only the drafter blocks inside the window before each point where the request stores a recurrent state, and before its shared-prefix junction. Hits normally land on those points, because a hit also needs the recurrent state there. The skipped blocks leave room for other prefixes. If a needed window is incomplete, the hit falls back to an earlier point whose window is complete, and the rest is computed again.

<a id="disk-tier-revolta_fs-experimental"></a><a id="verified-disk-tier-revolta_fs-experimental"></a>

## Disk tier (`revolta_fs`)

The disk tier keeps evicted prefixes on a local SSD below the CPU pool. Pages survive an engine restart, so a returning document is read from disk instead of computed again; see [Coming back to a document](performance.md#coming-back-to-a-document) for readback times. Use the `revolta_fs` tier type: it stores the INT8 scale planes with each page and verifies every page before use. The upstream `fs` tier type rejects the default planar INT8 KV (and per-token-head E4M3) at startup and has no integrity check.

The disk tier is a disposable cache, not model storage. Losing it only means prefixes are computed again.

The disk tier is supported in 0.2.0, with two limits on reuse:

- **The on-disk format can change between releases.** A later release may start with an empty tier and ignore pages written by 0.2.0.
- **Pages are reused only by the same configuration.** Changing the checkpoint, KV format, parallel layout or another setting listed in [When pages are not reused](#when-pages-are-not-reused) also starts from an empty tier.

In both cases nothing breaks: the old pages miss, the prefixes are computed again, and the old files are evicted as the tier fills.

### 1. Create a content manifest

The tier binds its pages to the contents of the loaded checkpoint, not to its path. A content manifest records every file of the target and of a separately loaded draft. Create it once per checkpoint, with the same paths the engine loads, and keep it outside the model directories. The tool is in the repository, not in the container image; run it with the image's Python, for example:

```bash
mkdir -p /data/revolta/manifests
docker run --rm --entrypoint python3 \
  -v /data/revolta:/data/revolta \
  -v "$PWD/tools/kv_offload:/tools:ro" \
  ghcr.io/leonccaa/revolta:0.2.0 \
  /tools/make_content_manifest.py \
    --role target=/data/revolta/views/target-512k \
    --role draft=/data/revolta/views/draft-512k \
    --out /data/revolta/manifests/target-512k.json
```

Run it from a checkout of the ReVolta repository at the release you serve. Omit the `draft` role only when the draft is loaded from the target's own directory. Symbolic links (as in the 512K views) are followed. Hashing reads every file once.

Create a new manifest whenever the checkpoint changes. If a listed file changes, a file is added to a model directory, or the manifest is missing, the tier disables itself and the CPU tier keeps working.

### 2. Configure the tier

Replace the launch command's `--kv-transfer-config` with:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 17179869184,
    "eviction_policy": "lru",
    "secondary_tiers": [
      {
        "type": "revolta_fs",
        "content_manifest_path": "/data/revolta/manifests/target-512k.json",
        "root_dir": "/data/revolta/kv-cache",
        "max_size_bytes": 8589934592,
        "min_free_bytes": 17179869184,
        "max_files": 10000,
        "n_read_threads": 1,
        "n_write_threads": 1,
        "drain_timeout_s": 20
      }
    ],
    "demote_superseded_states": true,
    "mamba_state_slots_reference_tokens": 65536
  }
}
```

The example sets a 16 GiB CPU pool and an 8 GiB disk budget. Notes:

- `max_size_bytes` is required: the budget for the cache page files under `root_dir`.
- `min_free_bytes` is the free-space floor of the file system. The default is 5% of its capacity.
- Use one dedicated directory per engine on a local SSD. A lock file allows only one engine per directory.
- The file system must support user extended attributes (ext4 and XFS do); each page's checksum is stored there.
- Keep `.hash_seed.json` in the directory. Pages are reused after a restart only with the same seed, the same manifest content and the same configuration identity (see [When pages are not reused](#when-pages-are-not-reused)).

### 3. Allow time to drain at shutdown

Pending page writes are written at shutdown only if the engine gets time for it. The API server stops the engine `--shutdown-timeout` seconds after a stop request; with the default `0` it stops at once and pending writes are lost. The reference launch sets `--shutdown-timeout 30`: in-flight requests finish first, then pending writes get up to `drain_timeout_s` (20 s in the example) within what is left of the 30 s. Give the container or service supervisor more time than that before it force-stops the engine (the reference container uses `--stop-timeout 120`); see `--shutdown-timeout` in [Shutdown and diagnostics](configuration.md#shutdown-and-diagnostics).

### 4. Check that it works

- At startup: `revolta_fs enabled at <directory> (content <digest>)`. If the manifest or the loading setup does not qualify ([When the tier disables itself](#when-the-tier-disables-itself)), the log shows `revolta_fs disabled: <reason>` instead, and serving continues with the CPU tier only. Invalid tier settings, such as a missing or non-positive `max_size_bytes`, an empty `root_dir` or no I/O threads, stop startup with an error.
- While serving, at most once a minute: `Delayed writes (revolta_fs): {...}` and `SSD tier stats (revolta_fs): verified_load_pages=N verified_load_bytes=B ...`. `verified_load_pages` counts pages read back from disk after their checksum and identity were verified.
- At shutdown: `revolta_fs stats: {...}` with pages written, pages already present, pages verified, checksum failures, retries and write jobs skipped because the disk was behind.

The external prefix-cache hit counters combine CPU and disk hits, so they cannot give the disk's share of prompt tokens. The `verified_load_*` counts appear only in the tier's log line; the delayed-write drops are also exported as `vllm:store_events_total` (see [Set the SSD write delay](#set-the-ssd-write-delay)).

### Size the CPU pool for disk restores

A prefix restored from disk is first staged in the CPU pool. Two limits follow:

- The pool must be larger than the largest prefix you expect to restore from disk, or the request is computed again. At the per-token sizes [above](#cpu-offload-reference-configuration) (about 21 KB with MTP4 and about 50 KB with DFlash2, E4M3 KV), a 100K-token prefix needs more than 2 GiB with MTP4 and about 5 GB with DFlash2.
- With MTP4 the pool is grouped: each recurrent-state group gets its own slots, sized from `mamba_state_slots_reference_tokens`. If more requests restore from disk at the same time than there are state slots, the extra requests are treated as cache misses and computed again; their output is not affected. The startup line `Grouped CPU offload pool: N token slots, M state slots per Mamba group` gives the state slots; each restoring request needs two. With the reference launch on a 0.2.0 development build, that was 4 slots per group at 2 GiB and 72 at 48 GiB, about 0.73 requests' worth of states per GiB. Size the pool so that 0.73 × pool GiB covers the requests that may restore at once: about 11 GiB for the default 8 sequences, hence 16 GiB in the example. Or lower `mamba_state_slots_reference_tokens`: 16384 gives each state group about four times as many slots from the same budget, at the cost of fewer token slots (not measured).

### Set the SSD write delay

<a id="setting-the-ssd-write-delay"></a>

`write_delay_s` (default 300 s) delays writing a page to disk. A page that is replaced during the delay never reaches the disk, which saves SSD bandwidth and wear. The cost: with `revolta_fs`, a page that is evicted from the CPU pool or demoted before its deadline is not written at all. (The Host KV SSD tier tries one write before it evicts an unwritten page; see [Host KV with an SSD](#add-an-ssd-to-host-kv).)

Keep the delay well below the time the CPU pool takes to turn over:

```text
turnover ≈ pool capacity ÷ new KV write rate
new KV write rate ≈ KV bytes per token × prefill tokens per second
```

For example, at about 13.8 KB per token and about 40 MB/s of new KV with four-way prefill, a 98 GiB pool turns over in about 40 minutes, so 300 s is safe there. A 7 GiB pool turns over in about 3 minutes; 300 s would evict most pages before they are written, so use about 20–60 s there. The SSD must sustain more than the average new KV write rate.

Check the result in the `Delayed writes (revolta_fs)` log line: `dropped_evicted` counts pages dropped because they were evicted before their deadline, and `dropped_superseded` counts pages dropped because a newer version replaced them. A rising `dropped_evicted` means the delay is too long for the pool. The same drops appear as `vllm:store_events_total{event="delayed_revolta_fs_dropped_evicted"}` and `...dropped_superseded`. `written` and `written_bytes_24h` count submitted pages, which can still fail or be skipped; they are not proof of physical writes.

At shutdown, pending pages that are still in the pool are submitted newest first, within the drain time.

### When the tier disables itself

The tier then reads and writes nothing, logs `revolta_fs disabled: <reason>`, and the CPU tier works as before. Causes:

- `content_manifest_path` is missing or unreadable, a listed file changed (size, modification time or inode; with `content_verify: "full"`, also its contents), or a file was added to a model directory.
- The target or draft is not a local checkpoint covered by the manifest.
- An unsupported loading setup: a load format other than `auto` or `safetensors`, extra loader options or secondary weights, LoRA, `--calculate-kv-scales`, a custom callable `hf_overrides`, or FP8 multimodal encoder attention without a scale file.
- The Python package `google-crc32c` is not installed (the reason reads `No module named 'google_crc32c'`). Install it in the serving environment.
- The directory's file system does not support user extended attributes.

### When pages are not reused

Pages are bound to a full identity: the manifest content, the KV format and page layout, the target and draft settings (including quantization, dtype and RoPE), block size, tensor and decode context parallel sizes, the attention backend, the index sharding (`qsa_dcp_index_shard`) and the release's compatibility epoch. Planar INT8, planar E4M3 and scalar E4M3 use separate namespaces. If any of these changes, existing files simply miss and the prefixes are computed again. Old files still count against `max_size_bytes` and are evicted oldest first.

Each page carries a CRC32C of its contents bound to its prefix and the tier identity. A corrupted page, or an intact page that belongs to another prefix or layout, is deleted and computed again; it is never loaded. Pages are written to a temporary file, synced and then renamed.

Prefixes served from the GPU or CPU also refresh their disk pages, with the start of a prefix kept most recent, so a shared system prompt is not the first page evicted.

## Host KV (long context)

<a id="active-host-kv-development-configuration"></a>

In Host KV mode, the main K/V history of the QSA layers lives in a host memory store. The GPU keeps the compressed index, recurrent states, drafter windows and a pool of hot pages that the sparse attention reads. Long contexts and many concurrent long requests then fit even when their K/V does not fit in GPU memory. Prefix reuse works through the same host store, and an optional SSD tier keeps pages across restarts.

**Status:** supported in 0.2.0 and off by default. On PCIe, AWQ with INT8 Host KV and MTP5 passed startup, serving, concurrency and quality checks (configuration E2); SSD readback after restart was not measured. The synthetic short-prompt test measured 26.76, 36.17 and 52.30 ms per decode round at C1, C4 and C8. These are wall-clock round times, not kernel timings. Host KV with DFlash2 is supported but was not tested separately in 0.2.0.

### Requirements

- A QSA model with linear attention (Qwen3.8 Flash-Next), tensor parallelism with decode context parallelism of 1 or 2. No pipeline or data parallelism.
- MTP, DFlash2 or no speculative decoding.
- `--kv-transfer-config` with the built-in `OffloadingConnector`: it moves recurrent states and drafter pages. Do not set `cpu_bytes_to_use` or `secondary_tiers` in this mode.
- Main K/V in INT8 (the default), E4M3 or FP16.
- CUDA graph mode for mixed batches must not be `FULL`. The reference `FULL_AND_PIECEWISE` works.
- The decode batch must fit the host reader's 256 rows: `--max-num-seqs` times the verify width (draft tokens + 1), and the largest CUDA graph capture size. The reference values (8 sequences, MTP4) are well inside.
- Host memory for `host_store_gib` in `/dev/shm`, in addition to the PLE table. In a container, make `--shm-size` larger than the store.

At startup, the engine checks that every pool can hold one request of `--max-model-len`. If not, it refuses to start, names the pool that is short and the settings to change, and gives the largest `--max-model-len` that fits. A request that does not fit at run time waits (or is preempted under the usual rules); it is never rejected for lack of pool memory.

### Configure Host KV

Start from the reference launch and replace `--kernel-config` and `--kv-transfer-config`:

```bash
--kernel-config '{"ple_disk_cascade": false,
    "long_context_kv": {"enabled": true, "host_store_gib": 48}}' \
--kv-transfer-config '{"kv_connector": "OffloadingConnector", "kv_role": "kv_both"}' \
```

- `host_store_gib` is the host store in GiB, summed over all workers. It holds the main K/V pages, recurrent states and drafter pages. Size it to the memory left after the PLE table (about 48 GiB for AWQ), other processes and the page cache. The example uses the same 48 GiB as the reference CPU pool, which fits the 128 GiB reference host; the [capacity table](#host-kv-capacity) shows 98 GiB stores.
- `qsa_dcp_index_shard` (in `--kernel-config`) splits the compressed index over all tensor-parallel ranks at DCP2 and lets the MTP draft's main K/V move to the host store; see [MTP draft placement](#mtp-draft-placement). If unset, the ReVolta hybrid serving profile turns it on for DFlash2 and for MTP with `long_context_kv.enabled`, when the model selects 512 index groups, as Flash-Next does. An explicit value takes precedence. DCP1 does not use it. MTP with GPU KV keeps the replicated index and does not support sharding.
- The index layout is part of the SSD page identity. If a Host KV MTP deployment switches from a replicated index to a sharded one on upgrade, existing Host KV SSD pages miss and the prefixes are computed again.
- `main_kv_placement` and `draft_main_placement` default to `"host"`; there is no need to set them.

### Add an SSD to Host KV

Add `host_kv.ssd` to `kv_connector_extra_config`. Create a [content manifest](#1-create-a-content-manifest) first; include the MTP draft as the `draft` role unless it is loaded from the target's directory.

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "host_kv.ssd": {
      "enabled": true,
      "root_dir": "/data/revolta/kv-cache-host",
      "max_size_bytes": 375809638400,
      "content_manifest_path": "/data/revolta/manifests/target-512k.json"
    }
  }
}
```

The example sets a 350 GiB disk budget. The directory, file system and identity rules are those of the [disk tier](#disk-tier-revolta_fs). With `enabled: true`, unknown keys inside `host_kv.ssd` are rejected at startup; all keys are listed in the [reference](kv-offload-reference.md#host_kvssd-keys).

- **Writes.** A published host page is queued for a delayed write (`write_delay_s`, default 300 s; see [Set the SSD write delay](#set-the-ssd-write-delay)). Writes are rate-limited to `write_bytes_per_second` (default 64 MiB/s). Before the store evicts a page that is not yet on disk, it tries one immediate write; it skips that write when the write queue is full, the disk budget has no room, or the page cannot be written. Watch `host_ssd_evict_unwritten` and `host_ssd_evicted_persisted` in the `Store stats` log line, or `vllm:store_events_total{event="host_ssd_evict_unwritten"}`: a rising `host_ssd_evict_unwritten` means pages leave the store before they reach the disk.
- **Reads.** Pages needed for a prefix hit are read back from disk and verified before use; reads are bounded per request and in total (`prefetch_max_per_request`, `prefetch_max_inflight`, `prefetch_timeout_s`). A failed or corrupt read counts as a miss and the prefix is computed again.
- **Shutdown.** With a positive `--shutdown-timeout`, in-flight requests get that time minus 5 s, and what is left is used for a final flush of unwritten pages for at most `shutdown_flush_timeout_s` (default 10 s): first the recurrent-state checkpoints the store has guaranteed to persist, then the other checkpoints, most recently used first, then the remaining pages. Pages not written are misses on the next start. With the default `--shutdown-timeout 0` the final flush is skipped. The log reports `Host KV SSD final flush: candidates=... completed=... failed=... dropped_unsubmitted=...`.

### MTP draft placement

In Host KV mode the MTP draft shares the target's KV cache groups, block table and block hashes. CPU and SSD prefix hits restore the draft together with the target and its recurrent state.

`draft_main_placement` chooses where the draft's main K/V lives; the draft's index stays on the GPU in both cases.

- `"host"` (default): the draft's main K/V is one more layer of the host store, read through the hot pages like the target's. Each history block takes fewer GPU bytes, so the same GPU budget holds more long contexts. Cost (0.2.0rc11, four V100, sampling, one request): about 0.2 ms longer per decode round.
- `"gpu"`: the draft's main K/V stays in GPU memory. Draft forwards read only GPU memory, so single-request decoding is fastest, but fewer long contexts fit.

`"host"` takes effect only with `main_kv_placement: "host"`, a target and draft with the same KV format and page geometry, and, at DCP2, `qsa_dcp_index_shard` on (the Host KV default). Otherwise the draft stays on the GPU, the engine logs a warning, and the GPU holds fewer long contexts (compare the "draft K/V on GPU" and "draft K/V on host" rows in the [capacity table](#host-kv-capacity)). The two placements are different configurations of the same build; their outputs are not guaranteed to be bitwise equal, because the draft's proposals and the shapes of the verified batches can differ. Changing `qsa_dcp_index_shard` changes the host page layout: an SSD tier written with the other value is not read back.

With MTP, in any KV mode, a prefix hit stops one block early, as with any EAGLE-style drafter: a prompt of L tokens reuses at most floor((L − 1 − S) / S) × S tokens, for a block of S tokens.

### Hot pages

The hot-page pool is GPU memory that holds the host pages the sparse attention needs for the current step. It is sized automatically from the GPU KV budget left after the recurrent states, the history index and prefill staging, and it always holds each admitted request's working set. `hot_pool_gib` (per GPU) or `hot_pool_ratio` set a guaranteed minimum instead; a minimum that cannot coexist with the active working set is a startup error.

### Checkpoint retention

<a id="checkpoint-retention-evidence"></a>

A recurrent-state checkpoint lets a later request resume from the end of an earlier turn. With `latest_only` (default `true`), the host store keeps, for each conversation chain it has seen, the latest checkpoint, the point an identical resend would resume from, and branch points. Older checkpoints are released. A later request that needs an older state recomputes from the nearest remaining checkpoint, which can span much of the prompt. Main K/V pages and their SSD files are not deleted by this policy. Set `latest_only: false` to keep every checkpoint, at the cost of host memory.

- `enable_junction_evidence` (default `true`): keep a branch point only when a different continuation of the same prefix has been seen. Confirmed branch points survive eviction of their children. They survive a restart only when `host_kv.ssd` is enabled and their pages were written to disk before the engine stopped, with the same identity on the next start. It has no effect with `latest_only: false`.
- `superseded_keep_ssd` (default `false`): write superseded checkpoints to the SSD before releasing their host memory. It requires an enabled `host_kv.ssd`.

### Host KV capacity

These sizing examples are from 0.2.0rc15: Qwen3.8 Flash-Next AWQ, four 32 GB V100s, TP4 + DCP2, the compressed index split over all TP ranks, `max_num_seqs=8`, `--gpu-memory-utilization 0.956` and planar INT8 KV. The "Startup" row was measured at engine start on PCIe V100s; "Planner" rows are the engine planner's arithmetic with an estimated per-GPU budget.

In the final 0.2.0 PCIe acceptance, AWQ with MTP5 and a configured 98 GiB Host KV pool had 7,329,312 usable host tokens after excluding one reserved page. The startup plan allocated 14,588 usable GPU history blocks (4.98 GiB per GPU including the reserved block).

A host page row takes 13,248 bytes per token for DFlash2 and 14,352 bytes per token for MTP4 (which also stores the draft layer's K/V and index), with planar INT8 (0.2.0rc15). Divide the host store by these sizes for a first estimate of the tokens it holds. DFlash2 keeps its FP16 sliding-window drafter KV on the GPU.

| Configuration | Context per request | Host store | Full-context requests (limit) | Usable Host KV tokens | GPU-indexable history tokens | Basis |
| --- | ---: | ---: | --- | ---: | ---: | --- |
| DFlash2 | 524,288 | 98 GiB | 8 (GPU state, max sequences) | 7,940,608 | 26,625,664 | Startup |
| MTP4, draft K/V on GPU | 524,288 | 98 GiB | 8 (GPU state, max sequences) | 7,329,312 | 11.9 M | Planner |
| MTP4, draft K/V on host | 524,288 | 98 GiB | 8 (GPU state, max sequences) | 7,329,312 | 25.6 M | Planner |

The full-context limit is the smallest of the per-pool limits, assuming every request is at the listed length. GPU-indexable history is the address limit of history blocks shared by all requests. The startup row had a per-GPU KV budget of 6.34 GiB after weights, activations and CUDA graphs; the planner rows assume about 6.05 GiB, the DFlash2 budget minus the difference between the two methods' startup profiles. Use the budget in your own startup log when sizing a deployment.

<a id="kv-capacity-in-long-context-host-kv-mode"></a>

### Check that Host KV works

- At startup, each worker logs `long-context KV worker=N long-context KV plan (main K/V in the host pool): GPU KV budget ...`. The line gives the GPU budget split (state, hot pool, staging, history), the host pool in blocks, tokens and GiB, and the request limits per pool at `--max-model-len`.
- With MTP at DCP2 and `"qsa_dcp_index_shard": false`, a warning says the draft stays on the GPU.
- With `host_kv.ssd`, the disk tier logs `revolta_fs enabled at ...` or `revolta_fs disabled: <reason>`.
- While serving, the `Store stats: counters=... gauges=...` line (at most once a minute; `--no-enable-store-metrics` turns it off) and the Prometheus series `vllm:store_events_total{event=...}` and `vllm:store_state{...}` report host store and SSD activity. `host_ssd_verified_load_pages` and `host_ssd_verified_load_bytes` count pages read back from disk after verification.
- At shutdown: `Host KV SSD final flush: ...` and `Host KV SSD stats: ...`.

## See also

- [KV offload reference](kv-offload-reference.md): every key and metric
- [Reference launch tutorial](getting-started.md)
- [Configuration reference](configuration.md)
- [Measured readback times](performance.md#coming-back-to-a-document)
- [GDN state replay](speculative-decoding.md#gdn-state-replay)
- [Support and known limits](known-limits.md)
