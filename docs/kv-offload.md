# KV offload modes

Use this guide to configure prefix-cache offload to CPU memory and disk. Start with
[a working reference launch](getting-started.md) and check its [host memory budget](configuration.md#system-memory-and-storage).

See [Coming back to a document](performance.md#coming-back-to-a-document) for measured readback times.

The KV cache always lives on the GPUs first. Offload adds lower tiers that keep
evicted prefixes, so a later request with the same prefix reads them back instead of
running prefill again. All three modes were tested for correct readback: tokens
generated after readback matched the off-mode reference exactly.

**1. Off.** Remove `--kv-transfer-config`. This uses the least host memory (80 GiB was the smallest tested passing configuration). Evicted prefixes are
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

`cpu_bytes_to_use` sets the pool size (total for all workers). With 48 GiB, the pool
holds about 2.4M tokens. `mamba_state_slots_reference_tokens` sets how many
recurrent-state slots each layer group gets. A smaller reference gives more slots,
which helps when many short sessions share the pool.

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
  token; a 100K-token prefix needs more than 2 GiB). Otherwise the request is computed
  again. The quick-start 48 GiB pool holds about 2.4M tokens.
- The SSD tier is **not supported with INT8 KV** yet. That combination is
  rejected at configuration time.

Validation scope: the SSD tier was tested with its root on a network file system. The
full release gate passed with an 8 GiB CPU pool and a 40 GiB disk budget, including a
99K-token prefix restored from disk (96,288 cached tokens, output identical). SSD
throughput was not measured.

## See also

- [Reference launch tutorial](getting-started.md)
- [Measured readback times](performance.md#coming-back-to-a-document)
- [Known limits](known-limits.md)
