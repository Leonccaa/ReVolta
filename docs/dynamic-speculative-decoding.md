# Dynamic speculative decoding

Dynamic speculative decoding (DSD) changes the number of draft tokens per request, K, with the number of running requests. A decode batch of `C` requests with `K` draft tokens each has `C × (K + 1)` verification rows. At higher concurrency a smaller K reduces verification work and, with MTP, skips draft forwards. This page is the reference for the schedule that sets K: its defaults, its format and how it interacts with scheduling and CUDA graphs. For choosing a speculative method, see [Speculative decoding](speculative-decoding.md).

DSD works with MTP and DFlash2 on Model Runner V2 (the default for Qwen3.8 Flash-Next) with no pipeline parallelism. MTP on a hybrid model such as Qwen3.8 Flash-Next also needs `--mamba-cache-mode align`.

## Default schedules

The ReVolta profile (`vllm serve` on four SM70 GPUs with TP4, FP16 activations and a Qwen3.8 Flash-Next checkpoint) adds a schedule when the speculative config has neither `num_speculative_tokens_per_batch_size` nor the deprecated `dflash_concurrency_drafts`, and pipeline parallelism is 1:

| Method | Condition | Default schedule | Draft tokens by running requests |
| --- | --- | --- | --- |
| DFlash2 | 7 draft tokens, no legacy `ngram_assist` | `[[1,3,7],[4,4,3],[5,8,4]]` | 7 for 1–3, 3 for 4, 4 for 5–8 |
| MTP | 4 draft tokens, prefix caching on (the profile default) | `[[1,3,4],[4,8,3]]` | 4 for 1–3, 3 for 4–8 |

Other widths get no default and run with a fixed K unless you set a schedule. For example, the 0.2.0 acceptance ran MTP5 with `[[1,3,5],[4,8,3]]`. To keep a fixed K under the profile, set the field to `null`.

## Schedule format

Set the schedule in `--speculative-config`:

```bash
--speculative-config '{
  "method": "mtp",
  "num_speculative_tokens": 5,
  "num_speculative_tokens_per_batch_size": [[1, 3, 5], [4, 8, 3]]
}'
```

- Each entry is `[first_batch_size, last_batch_size, K]`; both ends are inclusive.
- The first range must start at 1, and ranges must not overlap.
- Batch sizes in a gap between ranges, and after the last range, use the K of the preceding range.
- K can be 0. Each K is capped at `num_speculative_tokens`, which also sets the widest captured graphs.
- The batch size counts the requests that have tokens scheduled in the step, prefill and decode alike, up to `--max-num-seqs`.

## How K is applied each step

The scheduler looks up K for the current batch size. That K controls the proposal for the next step; the current step verifies the draft tokens each request already published, subject to the usual scheduling limits.

- **MTP** runs K draft forwards. At K=0 it publishes no draft tokens but still updates the draft KV in its first forward.
- **DFlash2** with K above 0 drafts its full 7-token block, including the drafter's context KV update, and publishes only the first K tokens. At K=0 it only updates the drafter's context KV and skips the draft query.

## CUDA graphs

With FULL decode graphs, ReVolta also captures the narrower verification widths that the schedule can reach:

- **MTP** captures them for target verification and the first draft forward, at the existing request capture sizes. Later draft forwards use the existing single-token graphs.
- **DFlash2** captures narrower target widths only at the request counts the schedule assigns to them, and keeps its drafter graphs.

Batch shapes outside the capture grid use the normal graph dispatch. The profile sizes the CUDA graph memory reserve for the widest captured width; see `VLLM_V2_CUDAGRAPH_MEM_MIB` in [Configuration](configuration.md#values-applied-by-the-revolta-profile).

## Adaptive MTP depth

`VLLM_1CAT_ADAPTIVE_MTP` (experimental, default `off`) adjusts K for a single decoding request from its recent acceptance, up to the schedule's K for one request. It acts only while exactly one request is scheduled; larger batches follow the schedule. It needs MTP, a schedule, synchronous scheduling and Model Runner V2. `on` needs at least 3 draft tokens; `auto` turns it on only on SM70 with at least 5.

## Restrictions

- The method must be `mtp` or `dflash`.
- With data parallelism above 1, the engine logs a message and uses a fixed K.
- Pipeline parallelism above 1 or Model Runner V1 is rejected at startup.
- DSD cannot be combined with the legacy DFlash2 `ngram_assist` lookup. [Native history lookup](dflash2.md#native-history-lookup) works with it.
- A DFlash2 schedule selects synchronous scheduling when `--async-scheduling` is not given. See [DFlash2 scheduling](dflash2.md#scheduling).
- `dflash_concurrency_drafts` (DFlash only) is a deprecated alias that maps to the same schedule; startup warns that it will be removed after this release. Set only one of the two fields. The old `{"4": 3, "5": 4}` equals the DFlash2 default schedule.

Based on [vllm-project/vllm#32374](https://github.com/vllm-project/vllm/pull/32374) and [vllm-project/vllm#57053](https://github.com/vllm-project/vllm/pull/57053).

## See also

- [Speculative decoding](speculative-decoding.md)
- [DFlash2 speculative decoding](dflash2.md)
- [Configuration reference](configuration.md)
