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

States at a shared-prefix junction are never demoted. Under `request_fixed` and `extend`, the prefix-hit state of any other active request is not demoted either, and `extend` never demotes the request's own resend checkpoint (the state an identical resend, capped at the prompt length minus one, resumes from; with an EAGLE/MTP shift, the one before it as well). The scheduler logs `Offload counters: demoted_states=... cpu_stored_blocks=... cpu_evicted_blocks=... rejected_boundary_offers=... rule=... cpu_draft_blocks_skipped=... cpu.draft_offload=...` every 60 seconds. `rejected_boundary_offers` counts recurrent states the CPU pool could not take when they were offered because all its state slots were in use by transfers; those states are not offloaded and are computed again on a later hit.

`cpu.draft_offload` selects which blocks of the speculative drafter's sliding-window KV cache the CPU tier stores. It is experimental, introduced in 0.2 (0.2.0rc5 and later), and set per deployment in `kv_connector_extra_config`.

- Type: string. Default: `"all"`. Allowed values: `"all"`, `"window"`, `"none"`. Any other value is rejected at startup.
- Scope: deployment. It applies only to a KV cache group recognised as a draft model's sliding window: speculative decoding is configured, the group uses a plain sliding window longer than one offloaded block, and the group is flagged as a draft group or its layers are numbered after the target model's (as with the DFlash2 drafter). Full-attention groups, recurrent-state groups and a target model's own sliding-window groups are not affected. If no group is recognised, the scheduler logs a warning and the setting has no effect.
- `all`: store every draft block, as before this option existed.
- `window`: store only the draft blocks inside the window before each boundary whose recurrent state the request stores, and before its shared-prefix junction. A prefix hit needs the recurrent state at the hit point anyway, so hits normally land on these boundaries. The other draft blocks are not stored, leaving CPU pool slots for other prefixes. With independent grouped pools, each group's budget is fixed: unused draft slots cannot become target or recurrent-state slots. If a needed window is incomplete, the hit falls back to an earlier boundary whose window is complete, and the target model computes the rest again. A window may be evicted after storage, or may never be stored completely: a late state hand-off can arrive after its GPU window was freed, the CPU pool can reject a window store, and conservative in-flight token accounting can exclude blocks still held by the core. A CPU cache reset reconsiders still-held window blocks, including a prefix loaded on re-admission after preemption; it cannot recover freed GPU blocks. Accepted block records and past boundary records are retired as the processed-token window advances. Pending steps do not advance this retirement boundary, so speculative rollback preserves records for blocks still held. These records cover the retained window and in-flight or scheduled work rather than the whole request history.
- `none`: store no draft blocks. The drafter cannot compute its cache again (it is projected from target hidden states), and a hit needs the draft window before the hit point, so a fresh CPU tier with a draft group cannot serve prefix hits. This setting controls writes; old draft keys already present in a backend can still serve hits. Use it only to measure the cost of the draft blocks.
- Counters: `vllm:kv_offload_cpu_draft_blocks_skipped_total` counts settled draft block skips because of this setting, and the `Offload counters` log line reports the same total as `cpu_draft_blocks_skipped`. For non-null scanned blocks, reconcile **accepted offers + settled skipped + still deferred = scanned**. Blocks not yet needed stay deferred until offered, freed, preempted or finished. A rejected store or retry keeps the same deferred obligation. After a CPU cache reset, previously offered or loaded blocks still held by the request can start new scan obligations and be offered again; existing deferred blocks retain their obligation, and previously settled skips are not counted again. A new admission after preemption starts a new scan. Offers include keys already resident or filtered by the store threshold, so they are not counts of new writes. The skipped counter measures neither saved resident slots nor bytes, and cannot generally be added to the all-group `cpu_stored_blocks` counter. With `all` its internal value stays at 0; the Prometheus series may be absent until a positive increment creates its labels.

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
