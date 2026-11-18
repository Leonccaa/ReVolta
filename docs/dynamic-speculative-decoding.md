# Dynamic Speculative Decoding

Dynamic speculative decoding (DSD) changes the number of published draft tokens
per request as the scheduled batch size changes. A fully active decode
batch with `C` requests and `K` drafts per request has `C × (K + 1)`
verification rows. A smaller K can reduce verification work and skip
MTP draft forwards at higher concurrency.

The ReVolta target configurations use MRv2 with MTP or DFlash (including
DFlash2) on four V100s with TP4, DP1 and PP1. Hybrid MTP requires
`--mamba-cache-mode align`. The thresholds for the MTP example below are
pending TP4 measurements.

## Configuration

```bash
--speculative-config '{
  "method": "mtp",
  "num_speculative_tokens": 4,
  "num_speculative_tokens_per_batch_size": [[1, 3, 4], [4, 8, 3]]
}'
```

Each entry has the form `[first_batch_size, last_batch_size, K]`. The
endpoints are inclusive. Ranges must start at 1 and may not overlap. Gaps
and sizes after the final range use the preceding K. K can be zero and
is capped by `num_speculative_tokens`. The example uses four drafts for
one to three scheduled requests and three drafts from four onward.

The scheduler counts requests with tokens scheduled in the current step.
This includes prefill and decode. The chosen K controls the proposal for
the next step. Current verification uses the drafts already published
for each request, subject to the existing scheduling limits. At K=0,
both methods publish zero drafts. MTP updates draft KV in its first forward.
DFlash still runs its full proposal, including the context KV update.

The field can be omitted or set to `null` for fixed K. With DP greater
than 1, the engine logs a message and uses fixed K. With DFlash, the
deprecated `dflash_concurrency_drafts` alias maps to the same schedule for
one release. Set only one field. DSD is incompatible with legacy
`ngram_assist`. The [DFlash2 setup](dflash2.md#settings) gives an example
schedule and the async conditions.

With FULL decode graphs enabled, ReVolta captures reachable narrower
query widths for target verification and the first MTP forward at the
existing request capture sizes. Later MTP draft forwards share the existing
single-step graphs. DFlash captures narrower target widths at the request
counts assigned to them by the schedule and keeps its existing drafter
graphs. Batch shapes outside the capture grid follow the existing graph dispatch.

Adapted from [vLLM #32374](https://github.com/vllm-project/vllm/pull/32374)
and [vLLM #57053](https://github.com/vllm-project/vllm/pull/57053).
