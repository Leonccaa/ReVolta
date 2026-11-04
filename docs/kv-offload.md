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

- `vllm:kv_offload_cpu_stored_blocks_total{kind}` counts offloaded blocks whose store to the CPU tier completed (all workers reported the job done), by KV cache group. `kind="draft"`: a group `cpu.draft_offload` recognises as a speculative drafter's sliding window (the same test, whatever the option's value). `kind="state"`: recurrent-state (Mamba, GDN) groups; each recurrent state is one block. `kind="target"`: every other group, including a target model's own sliding window and EAGLE/MTP groups that are not such a sliding window (an MTP layer with full attention is stored and looked up like the target's full attention). The classification is by group, so it is the same with one shared pool and with per-group pools; the scheduler logs it at startup (`Offload counter kinds by KV cache group`). Reconciliation: `cpu_stored_blocks` in the log line counts blocks when the store is prepared, so it equals the sum over kinds plus stores still in flight and stores dropped by a CPU cache reset (a reset discards them before they complete); with per-group pools it also includes reservations rolled back when another group's pool rejected the store, and with a disk tier it also counts blocks staged from disk into the CPU pool. Keys already resident are not stored again and are not counted. With `cpu.draft_offload` `window` or `none`, a draft block the store path passes is offered or counted in `cpu_draft_blocks_skipped`, so `kind="draft"` plus `cpu_draft_blocks_skipped` is at most the draft blocks passed (equal when none was already resident, none is in flight or deferred, and there was no reset). The byte counter `vllm:kv_offload_total_bytes_total` is recorded by the workers when a transfer finishes, so unlike this counter it can include stores whose completion a reset discarded.
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

## See also

- [Reference launch tutorial](getting-started.md)
- [Measured readback times](performance.md#coming-back-to-a-document)
- [Known limits](known-limits.md)
