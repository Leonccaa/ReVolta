# KV offload modes

Use this guide to configure prefix-cache offload to CPU memory and disk. Start with
[a working reference launch](getting-started.md) and check its [host memory budget](configuration.md#system-memory-and-storage).

See [Coming back to a document](performance.md#coming-back-to-a-document) for measured readback times.

The KV cache always lives on the GPUs first. Offload adds lower tiers that keep
evicted prefixes, so a later request with the same prefix reads them back instead of
running prefill again. All three modes were tested for correct readback: tokens
generated after readback matched the off-mode reference exactly.

**1. Off.** Remove `--kv-transfer-config`. This uses the least host memory with the default pinned PLE table (80 GiB passed memory stress; full acceptance was not run). Evicted prefixes are
computed again.

**2. CPU.** The quick start configuration:

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

`cpu_bytes_to_use` sets the pool size (total for all workers). With MTP4, a 48 GiB pool
holds about 2.4M tokens. With DFlash2, the drafter uses a different cache block size, so
every cache group takes pages of the full block size and a prefix needs about 50 KB per
token in the CPU pool and on disk (measured in 0.1.4: 1.84 GB for a 32K-token prefix,
6.36 GB for 131K), about 2.4 times MTP4's about 21 KB per token. `mamba_state_slots_reference_tokens` sets how many
recurrent-state slots each layer group gets. A smaller reference gives more slots,
which helps when many short sessions share the pool.

`demote_superseded_states` (default `true`) moves recurrent states that a finished request made obsolete to the front of the CPU pool's eviction order, so they are evicted before other conversations' reusable prefixes. Nothing is deleted. `superseded_state_rule` selects which states count as obsolete:

- `request` (default): the request's own states between its prefix hit and its last three blocks. The last block is counted including generated output.
- `request_fixed`: the same, but the last block is counted from the prompt. Recurrent states are only offloaded for prompt tokens, so counting output could demote the prompt-end states the next turn of the conversation resumes from (with outputs longer than about three blocks).
- `extend`: `request_fixed`, plus the state at the request's prefix hit once the request has stored a newer state past it. In a continued conversation this is the previous turn's end state, which the new turn has extended. A request that regenerates or edits the previous turn would need that state; it is only moved forward in the eviction order, so it is still used while it remains in the CPU pool or on the GPU.

States at a shared-prefix junction are never demoted. Under `request_fixed` and `extend`, the prefix-hit state of any other active request is not demoted either, and `extend` never demotes the request's own resend checkpoint (the state an identical resend, capped at the prompt length minus one, resumes from; with an EAGLE/MTP shift, the one before it as well). The scheduler logs `Offload counters: demoted_states=... cpu_stored_blocks=... cpu_evicted_blocks=... rejected_boundary_offers=...` every 60 seconds. `rejected_boundary_offers` counts recurrent states the CPU pool could not take when they were offered because all its state slots were in use by transfers; those states are not offloaded and are computed again on a later hit.

**3. CPU + disk.** Add a filesystem tier, typically on a local SSD:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 2147483648,
    "eviction_policy": "lru",
    "secondary_tiers": [
      {
        "type": "fs",
        "root_dir": "/path/to/nvme/revolta-kv",
        "max_size_bytes": 8589934592,
        "min_free_bytes": 17179869184,
        "max_files": 10000,
        "n_read_threads": 1,
        "n_write_threads": 1
      }
    ],
    "demote_superseded_states": true,
    "mamba_state_slots_reference_tokens": 65536
  }
}
```

These are the values from the validation run: a 2 GiB CPU pool and an 8 GiB disk
budget, chosen so that test documents had to go to disk. For your own deployment,
use a larger CPU pool and set `max_size_bytes` to your disk budget.

- `max_size_bytes` is **required**. It is a positive integer: the total page budget
  for everything under `root_dir`. Configuration fails without it.
- `min_free_bytes` is the free-space floor of the file system. The default is 5% of
  its capacity.
- Use one dedicated directory for each engine. A lock file allows only one engine per
  root.
- To reuse disk pages after a restart, the model, ReVolta version and page layout must
  be the same. Keep `.hash_seed.json` in the root. Restart reuse in the release
  gate: passed (a restarted engine read 667 MB of pages back from disk; output identical).
- The SSD tier is a disposable cache, not model storage.
- A prefix restored from disk is staged in the CPU pool first, so `cpu_bytes_to_use` must
  be larger than the largest prefix you expect to restore from disk (about 21 KB per
  token with MTP4 and about 50 KB with DFlash2; a 100K-token prefix needs more than
  2 GiB with MTP4 and about 5 GB with DFlash2). Otherwise the request is computed
  again. The quick-start 48 GiB pool holds about 2.4M tokens with MTP4; with DFlash2 the
  same pool holds about 1M tokens, an estimate from the same about 50 KB per token.
- The SSD tier is **not supported with INT8 KV** yet. That combination is
  rejected at configuration time.

Validation scope: the full release gate passed with the SSD tier on a network file
system (8 GiB CPU pool, 40 GiB disk budget), including a 99K-token prefix restored from
disk (96,288 cached tokens, output identical). In 0.1.4 the tier was also measured on a
local NVMe SSD with DFlash2 (7 GiB CPU pool, 64 GiB disk budget, one read and one write
thread). After an engine restart, 11 of 11 documents were read back from disk with the
same 64-token greedy output as a cold run, reading the pages from disk at about
2.2–2.4 GB/s inside the restore. That rate covers the disk read on its own: it is not the
stored bytes divided by the time to the first token, and that quotient is the lower of
the two. A 32K / 99K / 131K-token prefix came back in 1.47 / 2.93 / 3.77 s instead of
10.12 / 31.88 / 42.43 s when recomputed. A root holding 27.8 GB in 1,088 files did not
slow the restart. See [Coming back to a document](performance.md#coming-back-to-a-document).

**4. CPU + verified disk (`revolta_fs`, experimental).** A stricter variant of the
filesystem tier meant to keep conversations reusable across engine restarts. Use
`"type": "revolta_fs"` instead of `"fs"`; every `fs` key above still applies. It adds:

- **Integrity:** each page carries a CRC32C of its contents in a user extended
  attribute, bound to the page's prefix and the tier's identity. A corrupted page, or
  an intact page that belongs to another prefix or layout, is deleted and computed
  again; it is never loaded. Pages are written to a temporary file, synced and then
  renamed. The file system under `root_dir` must support user extended attributes
  (ext4 and XFS do); otherwise this tier disables itself and CPU serving continues.
- **Identity:** pages are bound to the contents of the loaded checkpoint, not its
  path. Create a content manifest once per checkpoint, with the same paths the engine
  loads, and keep it outside the model directories:

  ```bash
  python tools/kv_offload/make_content_manifest.py \
      --role target=/path/to/model --role draft=/path/to/draft \
      --out /path/to/manifests/model.json
  ```

  If a listed file changes, a file is added to a model directory, or the manifest is
  missing, the tier disables itself: nothing is read from or written to disk, and the
  CPU tier works as before. Create a new manifest after changing the checkpoint.
  The target and effective draft must resolve to local checkpoints covered by the
  manifest. Only the default `auto` and `safetensors` loaders are accepted; extra
  loader options, secondary weights, `model_weights`, LoRA, multimodal sources and
  runtime KV scale calibration disable this tier. Keep checkpoint generations
  immutable from manifest creation through model loading and serving. `stat`
  verification depends on that contract; `full` also rehashes the files.
  A dedicated full SHA-256 identity covers loader selection, target and effective
  draft settings (including quantization, dtype and RoPE), KV dtype/layout and page
  format. Each page records its complete key and namespace digest separately from
  its payload CRC32C and compares both exactly before accepting a load.
- **Recency:** prefixes used from the GPU or CPU pool also refresh their disk pages,
  with the start of a prefix kept most recent, so a shared system prompt is not the
  first page evicted.

Options of a `revolta_fs` tier entry, listed under the `ssd.` namespace (the key in the
tier entry is the part after `ssd.`):

| Option | Type | Default | Values | Scope | Effect and trade-off | Since | Status |
|---|---|---|---|---|---|---|---|
| `ssd.content_manifest_path` | string | none | path | deployment | Content manifest of the loaded checkpoint. Without it the tier is disabled. | 0.2 | experimental |
| `ssd.content_verify` | enum | `stat` | `stat`, `full` | deployment | `stat` checks size, modification time and inode of every listed file at startup (milliseconds). `full` also recomputes every content hash (about a minute per 100 GB) and catches a rewrite that restored the modification time. | 0.2 | experimental |
| `ssd.mtime_update_interval_s` | float | `60` | `>= 0` | deployment | Minimum time between refreshes unless another page in the touched chain is due; then the whole chain is stamped tail to head. The times order eviction after a restart; shorter is more precise and writes more metadata. | 0.2 | experimental |
| `ssd.store_timeout_s` | float | `30` | `>= 0`, `0` = never | deployment | A store job is reported failed when completion polling observes this deadline, including queue time, so the CPU pool is not held by a stuck disk. Safe because each page's checksum is taken from memory before the write starts. | 0.2 | experimental |
| `ssd.drain_timeout_s` | float | `30` | `>= 0` | deployment | At shutdown, queued page writes keep running for up to this time; the rest are dropped. Must fit in the engine's shutdown time. | 0.2 | experimental |
| `ssd.max_io_retries` | int | `2` | `>= 0` | deployment | Retries of a page read or write after a transient error (`EAGAIN`, `EINTR`, `EBUSY`, `ESTALE`, `ETIMEDOUT`, `EIO`), with interruptible backoff capped at one second and the remaining job budget. Checksum failures are not retried. | 0.2 | experimental |
| `ssd.max_inflight_store_blocks` | int | `0` (auto) | `>= 0` | deployment | Admission counts physical queued and active pages even after timed-out jobs release CPU pins. New jobs are skipped when they would exceed this limit. A single job may exceed the limit when the tier is empty. Auto is a quarter of the CPU pool's blocks, at least 64. | 0.2 | experimental |

When store timeouts are configured, the deadline covers the entire store job,
including time in the queue. Completion polling reports overdue jobs as failed
after any protected source snapshot finishes. Each page shares the job deadline.
An active kernel I/O call still has to return before its worker can shut down.

Counters of a `revolta_fs` tier (pages written, existing, verified, checksum
failures, pages skipped after a failure in the same chain, modification-time updates,
retries, write jobs skipped because the disk was behind, timed-out writes)
are logged when the engine shuts down.

## See also

- [Reference launch tutorial](getting-started.md)
- [Measured readback times](performance.md#coming-back-to-a-document)
- [Known limits](known-limits.md)
