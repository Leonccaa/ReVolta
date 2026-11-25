# KV offload modes

Use this guide to configure prefix-cache offload to CPU memory and disk. Start with
[a working reference launch](getting-started.md) and check its [host memory budget](configuration.md#system-memory-and-storage).

See [Coming back to a document](performance.md#coming-back-to-a-document) for measured readback times.

In GPU KV mode, the active KV cache lives on the GPUs. Offload adds lower tiers
that keep evicted prefixes, so a later request with the same prefix can read them
back instead of running prefill again. Readback with scalar E4M3 matched the
off-mode reference token for token in the tests below. For active main K/V
placement in host memory, see [Host KV](#active-host-kv-development-configuration).

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
With `kernel_config.long_context_kv` enabled (the host KV tier) neither key
has an effect; the scheduler logs a warning and ignores them.

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

**3. GPU KV + CPU + SSD.** Replace the launch command's `--kv-transfer-config`
with the JSON below. First create the content manifest for the target and any
separately loaded draft as described under [Verified disk tier](#verified-disk-tier-revolta_fs-experimental).
Eligible QSA models on V100 use planar INT8 with `--kv-cache-dtype auto`.
The raw `fs` tier is rejected if any target or draft group uses planar INT8 or
per-token-head E4M3. The example uses `revolta_fs`, which supports both.

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
        "type": "revolta_fs",
        "content_manifest_path": "/path/to/manifests/model.json",
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

The example sets a 2 GiB CPU pool and an 8 GiB disk budget. Set
`cpu_bytes_to_use` for the prefixes you need to stage in RAM and
`max_size_bytes` for the space available under `root_dir`. Keep
`TieringOffloadingSpec` selected so the engine constructs the SSD tier.

- `max_size_bytes` is **required**. It is a positive integer: the total page budget
  for everything under `root_dir`. Configuration fails without it.
- `min_free_bytes` is the free-space floor of the file system. The default is 5% of
  its capacity.
- Use one dedicated directory for each engine. A lock file allows only one engine per
  root.
- To reuse disk pages after a restart, keep the same checkpoint, compatible
  page layout and `.hash_seed.json` in the root.
- The SSD tier is a disposable cache, not model storage.
- A prefix restored from disk is staged in the CPU pool first, so `cpu_bytes_to_use` must
  be larger than the largest prefix you expect to restore from disk (about 21 KB per
  token with MTP4 and about 50 KB with DFlash2; a 100K-token prefix needs more than
  2 GiB with MTP4 and about 5 GB with DFlash2). Otherwise the request is computed
  again. The quick-start 48 GiB pool holds about 2.4M tokens with MTP4; with DFlash2 the
  same pool holds about 1M tokens, an estimate from the same about 50 KB per token.

Validation scope: the full release gate passed with the SSD tier on a network file
system (8 GiB CPU pool, 40 GiB disk budget), including a 99K-token prefix restored from
disk (96,288 cached tokens, output identical). The restarted engine read 667 MB
of pages from disk and produced the same output. In 0.1.4 the tier was also measured on a
local NVMe SSD with DFlash2 (7 GiB CPU pool, 64 GiB disk budget, one read and one write
thread). After an engine restart, 11 of 11 documents were read back from disk with the
same 64-token greedy output as a cold run, reading the pages from disk at about
2.2–2.4 GB/s inside the restore. That rate covers the disk read on its own: it is not the
stored bytes divided by the time to the first token, and that quotient is the lower of
the two. A 32K / 99K / 131K-token prefix came back in 1.47 / 2.93 / 3.77 s instead of
10.12 / 31.88 / 42.43 s when recomputed. A root holding 27.8 GB in 1,088 files did not
slow the restart. See [Coming back to a document](performance.md#coming-back-to-a-document).

## Verified disk tier (`revolta_fs`, experimental)

Eligible QSA `auto` selects planar INT8 on SM70 and `fp8_per_token_head` on other
supported CUDA devices. The `revolta_fs` tier persists the planar INT8 K/V scale
planes with each page. Planar INT8, planar E4M3 and calibrated scalar E4M3 use
separate SSD namespaces. Existing files under a different codec or page layout
miss and are recomputed. The readback measurements above used scalar E4M3.

The tier adds:

- **Integrity:** each page carries a CRC32C of its contents in a user extended
  attribute, bound to the page's prefix and the tier's identity. A corrupted page, or
  an intact page that belongs to another prefix or layout, is deleted and computed
  again; it is never loaded. Pages are written to a temporary file, synced and then
  renamed. The file system under `root_dir` must support user extended attributes
  (ext4 and XFS do); otherwise this tier disables itself and CPU serving continues.
- **Identity:** pages are bound to the contents of the loaded checkpoint, not its
  path. Create a content manifest once per checkpoint, with the same paths the engine
  loads, and keep it outside the model directories. Run the tool from a ReVolta
  checkout. Omit the `draft` role when the draft is part of the target checkpoint.

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

Set a positive `--shutdown-timeout` when this tier should drain at shutdown (the
default is 0, as upstream vLLM), and keep `ssd.drain_timeout_s` below it. Give the
container or service supervisor at least 120 seconds before force-stopping the
engine, and longer if the engine's own budget requires it. See
[#51](https://github.com/Leonccaa/ReVolta/issues/51) for shared-memory cleanup
after restarts. With `--shutdown-timeout 0` a stop aborts in-flight requests and
stops the engine before the tier can drain. Any queued writes are lost.

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

For a minimal RAM-only Host KV setup, replace the launch command's kernel and
KV transfer settings with these flags. Size `host_store_gib` to the machine's
available memory, including its shared-memory budget.

```bash
--kernel-config '{"long_context_kv":{"enabled":true,"main_kv_placement":"host","host_store_gib":48}}'
--kv-transfer-config '{"kv_connector":"OffloadingConnector","kv_role":"kv_both"}'
```

For an SSD tier in this mode, set `host_kv.ssd` in
`kv_connector_extra_config`, with `enabled: true`, `content_manifest_path`,
`root_dir` and a positive `max_size_bytes`. The keys `cpu_bytes_to_use` and
`secondary_tiers` belong to the GPU KV example above.

An MTP drafter (`speculative_config.method="mtp"`) shares the target's KV cache
groups, block table and block hashes in this mode. Its main K/V, compressed
index and raw-key ring stay in GPU history tensors, charged for every history
ID, and a host row carries the draft's pages next to the target's index pages.
CPU and SSD prefix hits therefore restore the draft together with the target
and its recurrent state, through the same host rows, SSD files and shutdown
path as without a drafter. As with any EAGLE-style drafter, a hit stops one
block early: a prompt of L tokens reuses at most floor((L-1-S)/S)*S tokens for
a block of S tokens. With `kernel_config.qsa_dcp_index_shard` and DCP2 the
draft's main K/V and index are sharded like the target's (the raw-key ring
stays replicated); index sharding with MTP is not supported outside this mode.
`qsa_dcp_index_shard` splits each compressed indexer over every tensor-parallel
rank (four ranks with TP4 and DCP2), not only over the DCP pair: rank `r` holds
the groups `g` with `g % TP == r`, scores only those, and the ranks merge their
candidates into the same selection an unsplit indexer would make. With TP4/DCP2
this halves the GPU bytes of every history ID's index again, so the same budget
holds about twice as many contexts. Cached pages written with a narrower split
are not read back. `kernel_config.long_context_kv.draft_main_placement` chooses
where a matching draft's main K/V lives (default `host`); the draft index
and raw-key ring always stay in GPU history.

- `"host"` (default): the draft's main K/V is one more layer of the host
  store, reached through the hot pool like the target's. Each history ID then
  costs about 0.71 MB (about 1.13 MB with `"gpu"`), so the same GPU budget
  holds many more contexts at 1M tokens. Measured on four V100 GPUs with
  sampling and a single request, a decode round takes about 0.2 ms (under 1%)
  longer than with `"gpu"`. The setting is a different configuration of the
  same build: with `"host"` the output is not guaranteed to be bitwise equal
  to `"gpu"`, because the draft's proposals, and with them the shapes of the
  verified batches, can differ.
- `"gpu"`: the draft's main K/V is a GPU history tensor, charged for
  every history ID (about 1.13 MB per ID per GPU with DCP2 and the index split
  over two ranks). Draft forwards read only GPU memory, so decode is fastest.

The default suits deployments with many long sessions or high concurrency at
long context, where the number of history IDs limits throughput. Set `"gpu"`
explicitly when single-request speed matters more than context concurrency,
when pinned host memory is tight, or when output must stay bitwise comparable
with releases that defaulted to `"gpu"`. `"host"` needs
`main_kv_placement="host"` and matching target/draft storage formats and page
geometry. With DCP2, explicitly set `"qsa_dcp_index_shard": true` at the top
level of `--kernel-config` for Host MTP. Use the unsharded index for GPU MTP.
Without the index shard in DCP2, the draft stays on GPU. A draft with a different
format or geometry stays on the GPU and logs a warning. Changing the value
changes the host row layout, so an SSD tier written with the other value is not
read back.

`speculative_config.gdn_state_replay=true` replays the accepted target GDN
inputs for MTP, so recurrent layers keep two state blocks per request instead
of one per draft position; `gdn_state_replay_graph` (default true) captures
that replay for FULL verify shapes. The SSD tier's content manifest only needs
the target checkpoint, which also holds the MTP layers.

`hot_pool_gib` and `hot_pool_ratio` are mutually exclusive ways to set a
**guaranteed minimum number of hot slots**. GiB sizing converts the total
per-worker budget, including payload, metadata and shared scratch, into whole
slots. Ratio sizing retains the conversion from a fraction of the host pool's
rank-local tokens. Neither setting imposes an upper limit on the actual pool.
The static pool (see the shared hot region below) may be raised to fit the
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
`mixed_mode != FULL`.

**Shared hot region.** When every QSA layer's padded index page holds a whole
number of hot slots, the hot slots alias the index pages of free history IDs
that carry no cached prefix. The KV budget then goes to history IDs; the hot
region grows and shrinks with them, above a floor that keeps the admitted
requests' verify working sets (and any manual minimum) resident. Growth first
evicts cached prefix indexes beyond their share, then shrinks the hot region to
the floor, and only then preempts. Every worker uses one common plan, so pass
plans match across ranks. `VLLM_SM70_HOST_KV_STATIC_HOT_POOL=1` restores the
separate static pool.

## KV capacity in long-context Host KV mode

These RC15 sizing examples use Qwen3.8 Flash-Next AWQ, four 32 GiB V100s,
TP4/DCP2, TP-wide compressed-index sharding, `max_num_seqs=8` and
`--gpu-memory-utilization 0.956`. Startup measurements use PCIe V100s.
DFlash2 planner rows reuse its measured per-GPU budget. MTP4 rows use an
estimated budget. Apply the planner rows to SXM2 with the same cache
geometry and per-GPU budget.

The host row uses **13,248 bytes per logical token for DFlash2 with planar
INT8**. The same row uses 13,056 bytes with calibrated E4M3. A local token in
one QSA layer stores 512 bytes of INT8 K/V payload and two 4-byte FP32 scales.
Summed over the DCP shards and 12 target layers, the scale planes add 192
bytes per logical token. The row also includes the compressed target index.
DFlash2 keeps its sliding-window FP16 draft KV on the GPU. With MTP4 the
host row also carries the draft layer's K/V and compressed index. It uses
**14,352 bytes per logical token with planar INT8** (14,144 with calibrated
E4M3) for either draft placement. The additional 208 bytes cover both FP32
scale planes across all 13 QSA layers. [RC11 E4M3 table](https://github.com/Leonccaa/ReVolta/releases/tag/v0.2.0rc11#kv-capacity-in-long-context-host-kv-mode)

| Configuration | Context per request | Host pool | Full-context requests (planner bound) | Usable Host KV tokens | SSD payload estimate (350 GiB) | GPU-indexable history tokens | Capacity basis |
| --- | ---: | ---: | --- | ---: | ---: | ---: | --- |
| DFlash2 | 524,288 | 98 GiB | 8 (GPU state, max sequences) | 7,940,608 | 28,366,208 | 26,625,664 | Startup |
| DFlash2 | 524,288 | 192 GiB | 8 (GPU state, max sequences) | 15,558,400 | 28,366,208 | 26,625,664 | Planner |
| DFlash2 | 1,048,576 | 98 GiB | 7 (host pool) | 7,940,608 | 28,366,208 | 26,614,016 | Planner |
| DFlash2 | 1,048,576 | 192 GiB | 8 (GPU state, max sequences) | 15,558,400 | 28,366,208 | 26,614,016 | Planner |
| MTP4, draft K/V on GPU | 524,288 | 98 GiB | 8 (GPU state, max sequences) | 7,329,312 | 26,183,808 | 11.9 M | Planner |
| MTP4, draft K/V on GPU | 524,288 | 192 GiB | 8 (GPU state, max sequences) | 14,361,600 | 26,183,808 | 11.9 M | Planner |
| MTP4, draft K/V on GPU | 1,048,576 | 98 GiB | 6 (host pool) | 7,329,312 | 26,183,808 | 11.9 M | Planner |
| MTP4, draft K/V on GPU | 1,048,576 | 192 GiB | 8 (GPU state, hot pages, max sequences) | 14,361,600 | 26,183,808 | 11.9 M | Planner |
| MTP4, draft K/V on host | 524,288 | 98 GiB | 8 (GPU state, max sequences) | 7,329,312 | 26,183,808 | 25.6 M | Planner |
| MTP4, draft K/V on host | 524,288 | 192 GiB | 8 (GPU state, max sequences) | 14,361,600 | 26,183,808 | 25.6 M | Planner |
| MTP4, draft K/V on host | 1,048,576 | 98 GiB | 6 (host pool) | 7,329,312 | 26,183,808 | 25.6 M | Planner |
| MTP4, draft K/V on host | 1,048,576 | 192 GiB | 8 (GPU state, max sequences) | 14,361,600 | 26,183,808 | 25.6 M | Planner |

The DFlash2 startup allocated a 98 GiB host pool of 4,773 pages with 1,664
tokens per page. One reserved page leaves 7,940,608 usable tokens. Its
per-GPU planner budget was 6.34 GiB after weights, activations and CUDA
graphs. For MTP4 the engine planner uses a 1,632-token history block and
about 6.05 GiB per GPU. The estimate starts with the DFlash2 Host KV
planner's 6.34 GiB budget and subtracts the 0.29 GiB difference between the
GPU KV startup profiles at 0.956 (6.35 GiB for DFlash2 and 6.06 GiB for MTP4).
Use the budget in your startup log when sizing your deployment.

The engine's `plan_long_context_capacity` takes the minimum of GPU state,
GPU history, Host KV, GPU hot-page and configured request limits. Each bound
assumes all requests are at the listed context length. Startup measures page
allocation. Full-context concurrency and SSD payload are arithmetic estimates.
The SSD column rounds the byte budget down to whole pages before file
overhead. GPU-indexable history is an address limit for shared history IDs.

**GPU KV mode:** use the engine log's `GPU KV cache size` in tokens at the
configured `max_model_len`. The Host KV table uses the allocated host pages
and the corresponding GPU history and hot pools.

## GDN state replay

During speculative decoding, DFlash2 and MTP can maintain GDN recurrent state by replaying the accepted tokens after sampling instead of saving a state for every draft position. Each running request then needs a fixed two state blocks per GDN cache group, regardless of draft length. The startup log confirms this with `recurrent layers keep 2 state blocks per running request`.

Keep replay enabled, especially for long contexts and concurrent requests: the saved GPU memory leaves more room for KV history. MTP and DFlash2 enable replay by default on their supported ordinary routes.

In a four-GPU TP4/DCP2 measurement with MTP4 and sampling, replay added about **0.6 ms per decode round (2%)** at one concurrent request. With eight concurrent long-context requests, disabling replay used about **0.87 GiB more GPU memory per GPU**. These measurements describe that configuration, not a general overhead guarantee.

Replay and per-draft state storage use different computation paths. Outputs are not guaranteed to be bitwise identical with replay enabled and disabled; near-tied token scores can lead to different continuations.

Set these fields in `--speculative-config` for the supported SM70 route with Model Runner V2, `--mamba-cache-mode align`, synchronous scheduling and no pipeline parallelism:

- **MTP:** `"gdn_state_replay"` defaults to `null` (`None` in Python), selecting replay on the supported route. Set it to `false` to disable replay, or `true` to request it explicitly. `gdn_state_replay_graph` also defaults to `null`, resolving to `true` for MTP; set it to `true` explicitly to retain CUDA graph replay for FULL verify shapes. Mixed steps replay eagerly.
- **DFlash2:** set `"sm70_dflash2": {"spec_state_replay": true}`. The omitted replay field defaults to `null`, selecting replay on the ordinary supported route while yielding to an enabled packed verifier. Its `spec_state_replay_graph` defaults to `true`; a non-null top-level `gdn_state_replay_graph` overrides that graph setting. See [DFlash2 state replay](dflash2.md#state-replay-optional) for route details.

## See also

- [Reference launch tutorial](getting-started.md)
- [Measured readback times](performance.md#coming-back-to-a-document)
- [Known limits](known-limits.md)

### Checkpoint retention evidence

With long-context `latest_only` enabled, `enable_junction_evidence` (boolean,
default `true`, deployment scope, introduced in 0.2, experimental) keeps each
observed leaf's latest checkpoint and exact resend point. A capture records
where recurrent state is needed; it does not by itself prove a branch.

A captured point is retained when a registered continuation differs from the
request's own next page. If the request has no complete next page, an existing
continuation grants conservative `short_tail_junction` retention. The consumer
also recognizes distinct registered successors arriving later. Confirmed reasons
carry a version and exact content binding in optional manifest metadata and
survive child eviction and clean restart. Ordinary value eviction still applies.
Prompt identity is used only for equality; a second identity never confirms a
branch.

On startup, both legacy `junction` and `branch` booleans are treated as hints.
The complete validated manifest graph determines effective retention. Old flags
are preserved without bulk metadata rewrites. Only newly confirmed reasons need
normal manifest updates. Missing legacy request metadata retains the existing
predecessor fallback. A checkpoint without a complete newer witness is kept.

This policy adds no grace timer or watch queue. MAIN tails not covered by a
registered checkpoint remain unknown; an older state may be superseded. A later
request can recompute from the remaining shallower state and capture a new
checkpoint. Recompute may span much of the prompt. MAIN pages and files are not
deleted by this policy.

Set `enable_junction_evidence=false` to replay the previous latest-only producer,
consumer and metadata behavior. The switch has no effect with `latest_only=false`.
It changes neither SSD namespace nor compilation hash. Turning it off cannot
restore already deleted state. `superseded_keep_ssd=true` retains committed cold
checkpoints and releases only safe RAM copies. Existing `branch_points`,
`chain_latest_units`, `resend_points`, `superseded_pending`, `superseded_revoked`,
`superseded_freed_bytes` and `superseded_ssd_deleted_bytes` report effective policy
state; no additional statistics are emitted with the switch off.
