# FS secondary tier

Use a dedicated cache directory and set an explicit positive `max_size_bytes`
in each `secondary_tiers` entry with `"type": "fs"`. The byte budget covers all
groups and old layouts under that root. Missing or invalid budgets fail during
configuration validation. `min_free_bytes` defaults to 5% of filesystem capacity;
an explicit non-negative byte value overrides it (including zero). All group
pools sharing a root must use the same limits. Writes that cannot fit fail as
offload jobs; CPU serving and cached reads remain available.

At engine initialization the root stores `.hash_seed.json`, which supplies the
prefix cache's first-block hash seed. The first launch uses `PYTHONHASHSEED` if
set, or generates a random seed. Subsequent launches use the stored seed even
if the environment changes. Creation is serialized by `.hash_seed.lock`, then
written atomically and synced. Invalid identity files are rejected rather than
silently replaced. Multiple FS roots in one engine must have the same identity;
empty roots adopt an existing root's seed. Preserve this file to preserve hits.

The identity lock does not allow concurrent engines to write the same cache.
The cache's separate lifetime `.lock` still permits only one engine per root;
group pools within that engine share its accounting. Engines using CPU offload
without an FS tier retain the existing `PYTHONHASHSEED` hash behavior.

Restart reuse requires the same model, version and page layout. Upgrades and
layout changes create a cold namespace; old pages remain within the shared
budget and are evicted by LRU. Pages from older unseeded launches cannot be
recovered by inventing a new seed. This is a disposable cache, not model storage.
