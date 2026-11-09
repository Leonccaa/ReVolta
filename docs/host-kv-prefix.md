# Host KV RAM prefixes and direct SSD persistence (development)

This is the CPU-verifiable host prefix adapter adapter for a host-authoritative main-QSA KV
pool. It is disabled by default. host-history integration explicitly attaches the real host allocator
to the complete-prefix contract lookup and request-lifecycle hooks described below. GPU correctness,
all-rank restoration and deployment readiness require integration validation.

## Ownership and publication

`vllm.v1.kv_offload.host.pool.HostKVPool` borrows a writable, contiguous CPU
buffer shaped `[slots, ...]`. Each row stores one whole main-KV group block,
including its owned layers and rank slices. It allocates metadata only. Active
requests, DMA and SSD jobs retain `HostKVPage` leases on the same bytes. A
published row stays in place as a RAM prefix after its last lease is released.
The existing CPU manager's LRU can reclaim only unreferenced rows.

host-history integration must delegate this buffer's slot ownership to the adapter, or implement the
same lease contract in its own allocator. Two allocators cannot own the buffer.
Use a scheduler/controller-visible shared backing for multi-worker rows. The
adapter does not export worker-local tensors, pin memory, or perform CUDA fences.
All API calls belong to one controller thread. Background workers access only
leased bytes. Published pages are immutable. Writes and speculative changes
require another reservation/COW. Never keep a writable view after publication
or any view after release.

1. `reserve(key)` reserves an unreadable row. Failure returns `None` without
   reclaiming an active or in-flight row. host-history integration must queue, preempt or recompute.
2. Fill **all** bytes, including deterministic padding. Wait for every producer
   fence, and validate the accepted-token boundary before `publish(page)`.
   Publication turns the reservation into a pinned immutable read lease.
3. `submit_store(key)` pins that same row and queues its SSD write. Releasing
   the active request does not free an I/O source. A skipped or failed SSD
   write leaves the RAM page available. The caller can retry after backpressure.
4. `prefetch(key)` reserves a host row and queues a direct SSD read into it.
   `poll()` publishes it only after full identity, length and CRC32C checks.
   Failed reads abort the reservation, even if it contains partial bytes.
5. Release active/read leases when their consumers and fences have finished.
   There is no second main-KV CPU offload copy.

Keys use the existing `make_offload_key(prefix_hash, group_idx)` format. The
caller supplies each complete block's **terminal prefix hash** on this group's
global token grid, using the existing request hash chain and extra keys. Partial
blocks, speculative bytes, compressor rings and state checkpoints do not belong
in this main-KV pool. host-history integration must remove main KV from ordinary CPU offload. The
existing connector continues to handle indices, recurrent/PLE state and drafts.

## Configuration

Set this object under `kv_connector_extra_config["host_kv.ssd"]` for an
`OffloadingConnector` configuration, then explicitly construct
`HostKVPrefixCache(pool, offloading_spec, layout)` from the host-history integration integration.

```json
{
  "enabled": true,
  "root_dir": "/path/to/dedicated-kv-cache",
  "max_size_bytes": 8589934592,
  "content_manifest_path": "/path/to/content-manifest.json",
  "content_verify": "stat",
  "write_bytes_per_second": 67108864,
  "max_pending_blocks": 64,
  "max_files": 1000000,
  "drain_timeout_s": 30.0
}
```

`enabled` defaults to `false`. With the key absent or disabled, no SSD thread,
root, file mapper or hash-seed file is created by this feature. Enabled mode
requires an explicit positive integer disk budget. `min_free_bytes` defaults
to 5% of filesystem capacity. Zero explicitly disables that reserve. The
existing file-count limit and disk LRU apply across every layout/group sharing
the root. Use identical disk budgets for adapters sharing it.

The manifest and verification policy follow `revolta_fs`. Immutable
checkpoint generations, source coverage and fixed KV scales are required.
Missing or unverified content disables SSD reuse. `stat` checks the manifest's
size/mtime/inode contract. `full` also rehashes the files. This adapter does not
weaken that contract. The existing seed initialization also registers this root,
so fresh processes reuse the existing prefix hash chain without a fixed
`PYTHONHASHSEED`. Keep `.hash_seed.json` for restart reuse.

`HostKVLayout` must contain the actual encoding, layer order, calibrated K/V
scalars and a versioned storage description covering heads, packing and padding.
Model content, engine config, TP/PP/PCP/DCP/rank, pool row geometry and global
block span also participate in the persistent namespace. The caller is
responsible for ensuring that its descriptor matches the actual bytes. No
compatibility conversion or inference of the host KV pool prototype's tensor layout is
performed.

`max_pending_blocks` bounds **all** queued, running and completed-but-unpolled
reads/writes, including the first submission. Each job is one row, so at most
that many pool rows are retained for I/O. It defaults to 64. There is no
unbounded payload queue. The write rate defaults to 64 MiB/s **per adapter**,
shared by its two I/O workers, with at most one page of initial credit. Idle
time does not accumulate credit. It limits payload write admission, including
retries. Physical device bytes, filesystem metadata and read traffic require
separate accounting.
Multiple adapters require an explicit aggregate budget/rate plan in host-history integration.

The adapter reuses the verified `revolta_fs` payload/xattr format with a distinct
host namespace. Writes use a temporary file, checksum/full identity attribute,
file fsync, atomic rename and directory fsync. Buffered `readv`/`write` accept
unaligned host rows without a second application KV pool. The shared checksum
helper can allocate one temporary page per I/O worker if the native CRC library
is unavailable. Kernel page-cache memory and metadata also need a host-memory
allowance. The legacy CPU offload path retains its default O_DIRECT behavior.

Use one owning controller process per dedicated root, as required by the
existing `FileSystemCache` lock. Worker-local controllers cannot independently
open the same root. A single-rank CPU fixture does not prove the multi-rank
ownership/fence integration.

## complete-prefix contract common-prefix lookup and pinning

After constructing the actual host backing, explicitly attach one controller
to both paths. Both complete-prefix contract contracts must be enabled with
`kernel_config.qsa_cross_group_hit_check`.

```python
from vllm.v1.kv_offload.host.hit import HostKVHitManager

hits = HostKVHitManager.attach(
    cache,
    kv_cache_manager,
    offloading_connector_scheduler,
    gpu_binding_ready=host_allocator.binding_is_reader_ready,
)
```

The attachment returns `None` if either contract is missing, a controller is
already attached, or an existing main-KV rule has an incompatible kind/span.
Validation precedes registration on either path. host-history integration must gate host-mode hit
reuse on successful attachment. The current unified-pool startup does not
construct an authoritative host backing or call this attachment automatically.

The same probe is registered with
`register_probe(group_id, probe, block_size=pool.block_tokens)` on the GPU and
connector contracts. It uses the group's terminal request-chain hash and
`make_offload_key(hash, group_id)`, including when the main group is probe-only.
An existing GPU group can represent the index paired with the host main KV.
The supplied binding callback receives `(gpu_block_id, offload_key)` and must
validate the live block's identity and generation against its host binding.
Absent, false or failed attestation rejects that GPU hit. A probe-only GPU
group has no GPU block ID in complete-prefix contract and requires host-history integration's separate binding transaction.

Only a published host row is reader-ready. File existence, an unfinished
producer write and an unverified SSD read all answer false. Polling before
each lookup makes completed verified reads visible. host-history integration drives bounded prefetch
and retries through the existing `prefetch`/`poll` API. A failed/corrupt read
aborts its reservation and contributes zero tokens. The existing complete-prefix contract cap,
fixed-point search and final check intersect main KV with index, state and
draft availability. Successful publication alone never reports a global hit.

Each path pins every required main-KV row from token zero through the final
common hit before returning it. If any pin fails, acquired rows are released
and that path reports a miss. External extensions acquire their new prefix
before releasing the previous local lookup leases. The full final host row
is retained for hits ending inside a row.

`KVCacheManager.allocate_slots` retains these leases after ordinary/host-history integration joint
admission. External restoration still uses
`num_external_computed_tokens`, including delayed caching. Failed admission
releases lookup leases. At the end of each scheduler step, uncommitted leases
are released, including paths deferred before allocation. `free` releases
admitted leases. host-history integration must complete reader/retired fences before freeing host
consumers. SSD jobs own independent leases until physical I/O completes.

The connector's external lookup has no destination ID yet. host-history integration must bind every
newly allocated GPU history block and generation to the pinned host page
before publishing the complete restore. `hits.pinned_pages(request_id)` lends
the chain-order leases and their key/slot identities for this transaction.
The controller owns their release. This adapter leaves that physical
binding transaction to the split-pool allocator. It also leaves the existing
post-admission load-failure channel to that integration. That channel must
invalidate the entire prefix and recompute it if validation or loading fails.
The inherited complete-prefix contract load-plan exception still requires replacement there.

## Group-local availability and shutdown

`acquire_prefix(keys)` accepts keys in prefix-chain order **starting at token
zero** and returns `HostKVPrefix(group_idx, available_tokens, pages)`. It stops
at the first missing or pending row and pins the returned contiguous prefix.
An isolated tail file cannot produce a prefix hit. Direct API users hold these
leases until their consumers finish. The attachment above retains request
leases through admission and intersects availability using complete-prefix contract. All-rank
readiness and physical binding remain part of the host-history integration restore transaction.

`stats()` reports group-local prefix query/token totals, queue rejections,
load/store failures, shutdown cancellations, and
`ssd_verified_pages`/`ssd_verified_bytes`. Query totals
include repeated queries. Verified page totals include successful promotions
that may subsequently be evicted. Neither is a global request hit rate or an
SSD token hit rate. Global counters remain complete-prefix contract's responsibility after commit.

`flush(timeout)` drains jobs and returns whether the queue is empty. Inspect
failure counters separately. `close()` drains for `drain_timeout_s`, cancels
queued work and rate waits, joins physical I/O, then releases its leases. An
aborted read is never published. SSD job deadlines do not release live source
buffers. A kernel I/O syscall that does not return can delay shutdown. Safe
pool reuse requires that physical I/O finish. Storage failure can lose cache
coverage and require recomputation. This is a disposable prefix cache.
After close, readiness and prefix acquisition return misses.

CPU tests use real files, unaligned caller-owned pages, corruption/error
injection, queue/rate checks, and separate writer/reader processes. GPU fences,
TP/DCP restoration, QSA numerics, ordinary-mode split-pool regression,
800K-class restart timings and multi-week soak tests remain integration gates.
