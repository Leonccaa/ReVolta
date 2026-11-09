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

States at a shared-prefix junction are never demoted. Under `request_fixed` and `extend`, the prefix-hit state of any other active request is not demoted either, and `extend` never demotes the request's own resend checkpoint (the state an identical resend, capped at the prompt length minus one, resumes from; with an EAGLE/MTP shift, the one before it as well). The scheduler logs `Offload counters: demoted_states=... cpu_stored_blocks=... cpu_evicted_blocks=... rejected_boundary_offers=... rule=... cpu_draft_blocks_skipped=... cpu.draft_offload=... stored_blocks_by_kind=target:...,draft:...,state:... lookup_shrink_tokens=draft_window:...,recurrent_state:...,sliding_window:...,page_align:... hit_draft_window_needed=true:...,false:...` every 60 seconds (cumulative totals; the last three are the store and lookup counters below). `rejected_boundary_offers` counts recurrent states the CPU pool could not take when they were offered because all its state slots were in use by transfers; those states are not offloaded and are computed again on a later hit.

`cpu.draft_offload` selects which blocks of the speculative drafter's sliding-window KV cache the CPU tier stores. It is experimental, introduced in 0.2 (0.2.0rc5 and later), and set per deployment in `kv_connector_extra_config`.

- Type: string. Default: `"all"`. Allowed values: `"all"`, `"window"`, `"none"`. Any other value is rejected at startup.
- Scope: deployment. It applies only to a KV cache group recognised as a draft model's sliding window: speculative decoding is configured, the group uses a plain sliding window longer than one offloaded block, and the group is flagged as a draft group or its layers are numbered after the target model's (as with the DFlash2 drafter). Full-attention groups, recurrent-state groups and a target model's own sliding-window groups are not affected. If no group is recognised, the scheduler logs a warning and the setting has no effect.
- `all`: store every draft block, as before this option existed.
- `window`: store only the draft blocks inside the window before each boundary whose recurrent state the request stores, and before its shared-prefix junction. A prefix hit needs the recurrent state at the hit point anyway, so hits normally land on these boundaries. The other draft blocks are not stored, leaving CPU pool slots for other prefixes. With independent grouped pools, each group's budget is fixed: unused draft slots cannot become target or recurrent-state slots. If a needed window is incomplete, the hit falls back to an earlier boundary whose window is complete, and the target model computes the rest again. A window may be evicted after storage, or may never be stored completely: a late state hand-off can arrive after its GPU window was freed, the CPU pool can reject a window store, and conservative in-flight token accounting can exclude blocks still held by the core. A CPU cache reset reconsiders still-held window blocks, including a prefix loaded on re-admission after preemption; it cannot recover freed GPU blocks. Accepted block records and past boundary records are retired as the processed-token window advances. Pending steps do not advance this retirement boundary, so speculative rollback preserves records for blocks still held. These records cover the retained window and in-flight or scheduled work rather than the whole request history.
- `none`: store no draft blocks. The drafter cannot compute its cache again (it is projected from target hidden states), and a hit needs the draft window before the hit point, so a fresh CPU tier with a draft group cannot serve prefix hits. This setting controls writes; old draft keys already present in a backend can still serve hits. Use it only to measure the cost of the draft blocks.
- Counters: `vllm:kv_offload_cpu_draft_blocks_skipped_total` counts settled draft block skips because of this setting, and the `Offload counters` log line reports the same total as `cpu_draft_blocks_skipped`. For non-null scanned blocks, reconcile **accepted offers + settled skipped + still deferred = scanned**. Blocks not yet needed stay deferred until offered, freed, preempted or finished. A rejected store or retry keeps the same deferred obligation. After a CPU cache reset, previously offered or loaded blocks still held by the request can start new scan obligations and be offered again; existing deferred blocks retain their obligation, and previously settled skips are not counted again. A new admission after preemption starts a new scan. Offers include keys already resident or filtered by the store threshold, so they are not counts of new writes. The skipped counter measures neither saved resident slots nor bytes, and cannot generally be added to the all-group `cpu_stored_blocks` counter. With `all` its internal value stays at 0; the Prometheus series may be absent until a positive increment creates its labels.

`cpu.enable_dependency_eviction` removes provably unusable CPU blocks after a
capacity eviction. `cpu.residency_stats` enables residency statistics independently.
Both are experimental deployment settings in `kv_connector_extra_config`.

- Type: boolean for both keys. Default: `false`. Non-boolean values are rejected
  at startup. With both keys absent or false, no dependency index or residency
  scan runs, and store selection, lookups and eviction order are unchanged.
- Scope: dependency removal supports CPU-only LRU tiers, including unified
  shared-row pools and independent grouped pools. It requires full-attention and
  Mamba `align` groups on the same offloaded page size, with any recognised draft
  sliding-window block size dividing that size. All recurrent groups participate
  in a checkpoint. EAGLE/MTP is treated conservatively; target sliding-window
  groups and other geometries retain their existing behaviour. With **any
  secondary tier**, including a filesystem tier, dependency removal is disabled:
  a missing CPU page may still be reachable through that tier. Residency is then
  reported as `unknown`. ARC and unsupported geometries also report `unknown`.
- Removal is a closure over resident dependencies: a state needs every recurrent
  group, the final target page in every full-attention group, and a mandatory
  suffix of each draft window. Target pages with no reachable resident state and
  draft pages outside every potentially usable state's window can then be freed.
  Admission is unchanged; the remaining pages keep their relative LRU order.
  Freed slots return to their original pool, so grouped pools cannot exchange
  capacity. `cpu.draft_offload=all|window|none` still controls writes only.
- Safety limits: CPU absence alone cannot prove an earlier prefix is broken,
  because lookup can start after a GPU prefix hit, including a partial prefix
  entry. Missing window heads may similarly be supplied by a GPU prefix. These
  cases are retained. Transfer pins, active requests' known hashes (including
  prefix hits and possible later state stores), and known junction dependencies
  are retained. Demotion under `superseded_state_rule=extend` is never evidence
  of unusability. Unknown annotations retain all rows. The index owns one record
  per resident CPU block, with bounded parent/group/window references, and drops
  it on every removal, failed store and reset. This does not promise idealised
  dependency collection, a particular capacity gain, or GPU performance.
- Metric: `vllm:kv_offload_cpu_resident_blocks{kind,status}` is a gauge, enabled
  by either setting, sampled by the scheduler at most once every 60 seconds.
  `kind` uses the same KV spec classifier as completed stores: recognised draft
  windows are `draft`, all `MambaSpec` modes are `state`, and other groups are
  `target`. Non-align recurrent modes remain unsupported for deletion and
  report `unknown`. Statuses are `potentially_servable`
  (a conservative upper bound, including GPU-assisted hits), `chain_broken`
  (a state's mandatory final target page is missing), `missing_dependency`
  (a required recurrent peer or mandatory draft suffix is missing), `no_state`
  (target page participates in no possible resident state hit), `outside_window`
  (draft page lies in no potentially usable state's window), and `unknown`
  (protected, incomplete metadata, secondary tiers or unsupported geometry).
  There is no `demoted_unservable` classification: a demoted state can still hit.
  The full state of the GPU prefix cache is not available to this index, so
  `potentially_servable` is not an exact count of cold CPU hits. Protected rows
  are counted as `unknown` even when their current dependencies are incomplete.
- Reconciliation: the gauge's sum over all labels equals occupied CPU slots
  **at sampling time**, including in-flight stores and loads. It counts blocks,
  not bytes, and must not be summed over time. All zero-valued labels are sent
  on later samples, including after reset. The existing `Offload counters` line
  adds `cpu_resident_blocks` and `cpu_residency` only when enabled. Additional
  removals use `vllm:kv_offload_cpu_evicted_blocks_total{reason="dependency"}`;
  `capacity + superseded + dependency = cpu_evicted_blocks` in the log, excluding
  `reset`. Dependency removal events carry exactly the freed keys. Resets and
  failed reservations are separate from successful writes, so stored minus
  evicted alone need not equal residency. Draft skips count write selection,
  not dependency removals or saved resident bytes.
`cpu.offload_counters` enables three store and lookup counters (experimental,
0.2). Type: boolean. Default: `false`; non-boolean values are rejected at
startup. Set it in `kv_connector_extra_config`, independently of
`cpu.enable_dependency_eviction` and `cpu.residency_stats`. With this key absent
or false, no counter object, group classification log, counting work, log suffix
or metric registration/samples are added. When true, these counters are read-only
bookkeeping: no store, load, lookup or eviction depends on them. Each also appears
in the single `Offload counters` log line as a cumulative total per label,
alongside residency if enabled. A Prometheus series appears with its first
positive increment.

- `vllm:kv_offload_cpu_stored_blocks_total{kind}` counts offloaded blocks whose store to the CPU tier completed (all workers reported the job done), by KV cache group. `kind="draft"`: a group `cpu.draft_offload` recognises as a speculative drafter's sliding window (the same test, whatever the option's value). `kind="state"`: recurrent-state (Mamba, GDN) groups; each recurrent state is one block. `kind="target"`: every other group, including a target model's own sliding window and EAGLE/MTP groups that are not such a sliding window (an MTP layer with full attention is stored and looked up like the target's full attention). The classification is by group, so it is the same with one shared pool and with per-group pools; the scheduler logs it at startup (`Offload counter kinds by KV cache group`). Reconciliation: `cpu_stored_blocks` in the log line counts prepared reservations, while C1 counts completed store jobs. They can differ because of in-flight or failed stores, reset-discarded jobs, grouped reservations rolled back when another pool rejects the store, and disk-to-CPU staging. Keys already resident are not stored again and do not increment C1. Draft scans reconcile as **accepted offers + settled skips + deferred = scanned**. Completed `C1{kind="draft"}` cannot replace accepted offers: an accepted offer may include already-resident keys or keys filtered by admission `store_threshold`, even after all work drains. A rejected store, including grouped rollback, keeps the deferred obligation until retry or settlement; failed stores do not undo accepted offers. Reset can start new scan obligations for held blocks as described above. These counts measure scan obligations, not unique resident blocks or saved bytes. The byte counter `vllm:kv_offload_total_bytes_total` is recorded by the workers when a transfer finishes, so unlike this counter it can include stores whose completion a reset discarded.
- `vllm:kv_offload_lookup_shrink_tokens_total{cause}` counts tokens by which the sliding-window and recurrent-state groups lowered a CPU-tier lookup below its full-attention reach. The reach is the hit the full-attention groups alone allow: their consecutive offloaded blocks after the local GPU hit, capped at the request length minus one and kept on the Mamba page grid. The served length is the local hit plus the lookup's external hit. For each lookup, the causes add up to reach minus served. `cause="recurrent_state"`: the recurrent state is not stored at the reach, so the hit moves back to the previous stored state. `cause="draft_window"`: the drafter's window before the hit is incomplete. `cause="sliding_window"`: another sliding window (for example a target model's own). `cause="page_align"`: rounding a sliding-window group's hit down to the Mamba page grid, and with EAGLE/MTP the one-page shift of the restored state before the reach (every such lookup adds one page here). A lookup that ends without an external hit is served at the local hit; the rest of the reduction goes to the group examined last. A lookup counts once, when the core admits the request (as the external prefix cache query counters do); a lookup the backend defers, or repeated because the request could not be allocated, counts only in the attempt that is admitted. A lookup whose full-attention reach is less than one block beyond the local hit, and a model without both a full-attention group and a sliding-window or recurrent-state group, count nothing. The served length is the lookup's result: a partial prefix-cache hit resolved afterwards in favour of the GPU hit does not change it. Shrinks to no external hit at all are counted as well.
- `vllm:kv_offload_hit_draft_window_needed_total{needed}` counts admitted requests with an external (CPU-tier) hit on a model with a draft group. `needed="true"` when fewer tokens than the draft window remain to be prefilled after the hit (request length minus local hit minus external hit): the drafter's window at the first decode step then reaches back into the loaded draft blocks. With `needed="false"` that window lies inside the tokens prefilled after the hit. The counter compares lengths only; it does not observe what the drafter reads. The two labels add up to the external hits of such a model. With several draft groups the largest window applies.

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
| `ssd.write_delay_s` | float | `300` | `>= 0` | deployment | Wait before cascading CPU rows to disk. Only keys are queued; CPU rows remain evictable. At the deadline, write only rows still resident, ready and not demoted, resolving CPU slots again by key. `0` keeps immediate cascading. A longer delay can avoid superseded writes but leaves fewer rows durable before a crash. | 0.2 | experimental |
| `ssd.write_warn_bytes_24h` | int | `1000000000000` (1 TB) | `>= 0`, `0` disables warnings | deployment | Warn at most hourly when the rolling 24-hour submitted page-byte estimate exceeds this threshold. Grouped pools sharing a root share the persisted count. This is a log-only warning: writes continue. It is not a physical SSD write or endurance counter. Applies to delayed writes. | 0.2 | experimental |
| `ssd.state_latest_only` | bool | `false` | `true`, `false` | deployment | With delayed writes, discard a finished request's older recurrent-state keys from pending writes, keeping its newest state, shared junction and states other active requests resume from. CPU residency is unchanged. Leave disabled to retain the normal pending-state selection. | 0.2 | experimental |

Delayed queues are serviced on engine steps and while the engine is idle, with an
idle polling interval of up to one second. At shutdown, still-live pending rows are
submitted newest first within the tier's existing bounded drain. Rows evicted or
demoted before their deadline are skipped; no CPU slot is pinned during the delay.
These options belong in the `revolta_fs` tier entry without the `ssd.` prefix, for
example `"write_delay_s": 300`, `"write_warn_bytes_24h": 1000000000000` and
`"state_latest_only": false`.

When store timeouts are configured, the deadline covers the entire store job,
including time in the queue. Completion polling reports overdue jobs as failed
after any protected source snapshot finishes. Each page shares the job deadline.
An active kernel I/O call still has to return before its worker can shut down.

Keep `--shutdown-timeout` positive when this tier is enabled (ReVolta's default is 30;
upstream vLLM's is 0), and keep `ssd.drain_timeout_s` below it. With
`--shutdown-timeout 0` a stop that arrives while requests are in flight kills the engine
process at once: the tier never reaches its bounded drain, and the HTTP server can wait
for the unfinished requests until the container runtime kills it. This affects every
deployment, with or without a disk tier.

Counters of a `revolta_fs` tier (pages written, existing, verified, checksum
failures, pages skipped after a failure in the same chain, modification-time updates,
retries, write jobs skipped because the disk was behind, timed-out writes)
are logged when the engine shuts down. With delayed writes enabled, the manager
also logs `Delayed writes (...)` and `SSD tier stats (...)` at most once every
60 seconds when it services persistence. The latter reports cumulative
`verified_load_pages` and `verified_load_bytes`: pages successfully read back from
the SSD tier after CRC32C and identity verification, and their full page bytes.
Grouped pools report their own tier counters; these are not unique prompt tokens.
The external prefix-hit counters combine CPU and SSD hits, so they cannot provide
an exact SSD share of prompt tokens. These SSD counters are log output, not a
Prometheus endpoint or a prompt-token hit rate. The delayed queue's `written` and
`written_bytes_24h` count submitted rows and page bytes, which may still fail or be
skipped by the disk tier; they do not prove successful physical writes.

## Active Host KV development configuration

`kernel_config.long_context_kv` is an experimental, opt-in configuration for
QSA main K/V in host memory. Host placement requires the built-in
`OffloadingConnector`; state and draft prefix transfers still use that connector.
This development interface does not extend the project's supported deployment
scope.

`hot_pool_gib` and `hot_pool_ratio` are mutually exclusive ways to set a
**guaranteed minimum number of hot slots**. GiB sizing converts the total
per-worker budget, including payload, metadata and shared scratch, into whole
slots. Ratio sizing retains the conversion from a fraction of the host pool's
rank-local tokens. Neither setting imposes an upper limit on the actual pool.
The current implementation allocates a static pool and may raise it to fit the
minimum active working set. A requested minimum that cannot coexist with that
working set causes a startup configuration error; it is never silently reduced.

With neither setting supplied, or with `hot_pool_gib="auto"`, the hot pool uses
the KV budget remaining after state, demand-sized history and prefill staging.
Auto guarantees the configured verify working set, capped by each request's
maximum local history. **Omitting both settings now selects auto, replacing the
older implicit ratio sizing.** The host pool must still be sized explicitly.

The host reader reserves a fixed `R_cap=256` rows of indices scratch. Startup
checks the resolved decode capture sizes and `max_num_seqs` times the verify
width against that limit. Uniform decode uses one pass; mixed batches require
`mixed_mode != FULL`. Dynamic sharing between index pages and hot slots remains
a later development phase.

## See also

- [Reference launch tutorial](getting-started.md)
- [Measured readback times](performance.md#coming-back-to-a-document)
- [Known limits](known-limits.md)
