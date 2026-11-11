# DFlash2 speculative decoding (optional)

Use this guide to replace MTP4 with DFlash2 and choose its settings. First prepare
[a working reference launch](getting-started.md); MTP4 remains the default.

> [!TIP]
> **New in 0.1.2: DFlash2 speculative decoding on V100.** ReVolta runs the
> [incoai DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) on
> Volta (SM70) GPUs, with V100 attention kernels and CUDA graphs for the drafter and the
> verifier, in the 512K-context setup with prefix caching and CPU KV offload. On one
> request it writes English code at up to **234 tokens/s instead of 180** with MTP4 (1.30×)
> on community NVLink hardware. With 0.1.4 and its then-recommended DFlash2 settings (probabilistic
> drafting, state replay off) the GPU KV pool reaches **1.37 million tokens (2.61× at 512K)**;
> with greedy drafting it is 1.38 million (2.63×), and the 0.1.3 measurement with state replay
> on (greedy drafting) reached 1.54 million (2.94×). MTP4
> stays the default: Chinese text and eight concurrent requests are still faster with MTP4.
> [How to enable it](#enable-dflash2).

MTP4 is the default and the tested release configuration. Since 0.1.2, ReVolta can use
the [incoai DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
instead (tested revision `a22a1744`): 5 sliding-window layers, an 8-token block and a
2048-token window. It is off by default. Without a `dflash` speculative config, the KV
cache layout, block size and MoE routes are the same as in 0.1.1.

The configuration below is the current development recommendation, with state replay
explicitly enabled. Released versions 0.1.4–0.1.5 keep their replay-off default and
need this explicit `true`; the example therefore enables replay on those supported
releases too. Historical measurements retain their stated replay setting. With 0.1.2, also set
`VLLM_V2_CUDAGRAPH_MEM_MIB=1500`; its profile does not reserve enough memory for DFlash2's
graphs.

## Enable DFlash2

**1. Make a draft view.** The drafter must declare how it reads the target's
hyper-connection streams, and its position limit must cover the 512K target. The view
links the weights and adds both settings to `config.json`. Startup refuses a drafter
without `"target_feature_reduction": "hc_mean"`.

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

src = Path("/data/revolta/models/Qwen3.8-Flash-Next-DFlash2")
dst = Path("/data/revolta/views/dflash2-512k")
snapshot_download("incoai/Qwen3.8-Flash-Next-DFlash2", local_dir=src,
                  revision="a22a1744e85c17364dcc54a6a420af7135d4c7f6")

dst.mkdir(parents=True, exist_ok=True)
for item in src.iterdir():
    if item.name not in ("config.json", ".cache", ".git"):
        link = dst / item.name
        if not link.exists():
            link.symlink_to(item.resolve(), target_is_directory=item.is_dir())
config = json.loads((src / "config.json").read_text())
config["dflash_config"]["target_feature_reduction"] = "hc_mean"
config["max_position_embeddings"] = 524288   # the drafter's layers keep a 2048-token window
(dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
EOF
```

**2. Set one more environment variable and replace the MTP speculative config** in [step 4](getting-started.md#4-launch).
Keep every other argument, including `--no-async-scheduling`. With Docker, pass
`-e VLLM_SM70_DFLASH2_FUSED_QKV_PACK=1`.

```bash
export VLLM_SM70_DFLASH2_FUSED_QKV_PACK=1

  --speculative-config '{"method": "dflash",
      "model": "/data/revolta/views/dflash2-512k",
      "num_speculative_tokens": 7,
      "attention_backend": "FLASH_ATTN_V100",
      "draft_sample_method": "probabilistic",
      "dflash_concurrency_drafts": {"4": 3, "5": 4},
      "sm70_dflash2": {"context_kv_graph": true, "context_pipeline": true,
        "verify_fastpath": true, "fused_gdn_metadata": true,
        "fused_smallq_metadata": true, "grouped_smallq_metadata": true,
        "spec_state_replay": true}}' \
```

In the startup log, the AWQ MoE line should read `grouped QPN decode admitted for 2..48
rows (split=8, pack_rows=24, up to 2 row chunks)`.

From 0.1.4, `"draft_sample_method": "probabilistic"` is the recommended setting for DFlash2:
the drafter samples its proposals and the target accepts them with the standard rejection
rule. `"greedy"` still works. MTP must stay `"greedy"`; see
[Launch options](configuration.md#launch-options).

## Settings

- **Enable the six `sm70_dflash2` options above.** They are off unless set.
- **Do not enable** `fused_gdn_verify`, `fused_gdn_norm`, `fused_gdn_split` or
  `fused_gdn_combined_split`. They change the numerics and lowered sampled acceptance
  in our tests.
- **Leave `--block-size` unset.** The automatic block size fits the drafter cache into the
  main KV slots (1664 tokens with 7 drafts at DCP2). An explicit block can leave the
  drafter only small blocks; startup logs a warning then.
- **Do not set `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS`.** With DFlash, the AWQ MoE
  layers use `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_DFLASH_MAX_TOKENS` (default 48; no need to
  set it). The older variable changes only MTP and non-speculative routes.
- **`VLLM_SM70_DFLASH2_FUSED_QKV_PACK=1`** packs the verifier's Q/K/V rows with one copy
  kernel. It acts only on SM70 with DFlash2. In tests, greedy output was bit-identical and
  throughput about 1.5% higher. Like every `VLLM_*` variable, it is part of the
  torch.compile key, so a deployment that sets it builds its own compile cache.
- **CUDA graph memory is set by the profile from 0.1.3.** With DFlash2 on AWQ TP4, the ReVolta profile reserves 1200 MiB per GPU (`VLLM_V2_CUDAGRAPH_MEM_MIB`) and leaves the single-request verifier tail graphs off (`VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS=0`). The graphs take about 1.1 GiB per GPU instead of 1.4 GiB. Greedy output was identical in the single-request comparison, including a 256K-token prompt; repeated comparisons found no consistent speed penalty at one or three requests. MTP4 keeps 700 MiB. Remove a previous explicit 1500 MiB setting to use the new default: environment values override the profile. If you turn the tail graphs back on, also set `VLLM_V2_CUDAGRAPH_MEM_MIB=1500`. A reserve smaller than the graphs lets the KV pool take that memory, and concurrent long prompts can then run out of GPU memory.
- **State replay is on by default in current development for the supported ordinary DFlash2 route.** The recommendation above sets `"spec_state_replay": true` explicitly, including for released 0.1.4–0.1.5, with `spec_state_replay_graph=true` retained as the graph default. Set `"spec_state_replay": false` to keep per-draft state blocks instead. It adds GPU KV capacity at a small per-step cost; see [State replay (optional)](#state-replay-optional).
- **`dflash_concurrency_drafts`** maps a number of running requests (the threshold) to the
  number of drafts the target verifies per request (the width). The drafter still drafts
  7 tokens. With `{"4": 3, "5": 4}`, one to three running requests verify all 7 drafts,
  four verify 3, and five or more verify 4.
- The thresholds form a ladder: the entry with the largest threshold that is not above the
  number of running requests sets the width. Below the lowest threshold, all drafts are
  verified, and a higher entry can widen the width again. CUDA graphs are captured only
  for the request counts that use each width. Thresholds must be 2 or more, and widths 1
  to 7.
- Against `{"5": 4}`, this ladder gave identical output at 1 request and 9.7% more
  throughput at 4 requests (English, Chinese and code weighted equally); at 5–8 requests
  both use the same width.
- For mostly Chinese loads, `{"3": 3}` verifies 3 drafts from three requests on; at three
  requests that was about 15% faster in an earlier width sweep. For mostly code, `{"8": 4}`
  measured best at 8 requests.
- Setting `dflash_concurrency_drafts` turns off async scheduling; `--async-scheduling`
  together with it is an error. Pipeline parallelism is not supported with DFlash2.

## Native history lookup (optional)

Native lookup is **off by default**. It searches committed token history for a
continuation and fuses it into the seven-token DFlash2 proposal when the first
lookup token agrees with the neural proposal. The original first token and its
proposal distribution are preserved. Short continuations replace only the
available positions; the remaining neural suffix retains its original proposal
probabilities.

The recommended opt-in recipe is C64: longest match, suffix lengths 5 through 16,
a 65,536-token search window and the strict `first_token` gate. Merge this fragment
into the DFlash2 speculative config above, retaining its model and other settings:

```json
{
  "num_speculative_tokens": 7,
  "dflash_concurrency_drafts": {"4": 3, "5": 4},
  "dflash2_lookup": {
    "enabled": true,
    "policy": "longest",
    "nmin": 5,
    "nmax": 16,
    "search_window": 65536,
    "gate": "first_token",
    "collect_stats": true,
    "max_requests": 3
  }
}
```

Use `--no-async-scheduling` with this ladder. Omitting `dflash2_lookup` or setting
`"enabled": false` leaves lookup disabled. When enabling it, `policy`, `nmin` and
`nmax` must be explicit; no lookup recipe is selected automatically.

- A batch of one to `max_requests` eligible decode requests attempts lookup.
  `max_requests` defaults to 3 and accepts integers from 1 to 3; set it to 1 to
  restore single-request admission without rebuilding the image. Mixed prefill,
  grammar constraints, any row without a sampled token or sufficient sequence
  space, and batches whose concurrency ladder selects a width below seven
  bypass lookup for the whole batch. Eligibility is recalculated every round.
  Padding rows do not count; returning to an eligible batch resumes immediately.
- The configured proposal/resource width remains seven; it does not expand to
  fifteen or allocate per-request q16 graphs. The existing concurrency ladder
  can still reduce the actual verification width: with the example above,
  one to three requests use W=7, four use W=3, and five or more use W=4. Lookup
  bypasses batches of four or more requests, even without a ladder.
- Fixed width seven **without a concurrency ladder** permits async scheduling,
  subject to the existing executor, model and state-replay constraints. Lookup
  does not itself turn async off. A nonempty ladder and explicit async scheduling
  are incompatible and raise an error; with async unspecified, the existing
  ladder rule selects synchronous scheduling.
- This version supports only `longest` and `first_token`. `nearest` and expansion
  are rejected, including expansion enabled together with lookup disabled.
- Legacy `ngram_assist` and lookup environment settings remain compatible; no
  removal version is set. Use explicit `dflash2_lookup` settings for new setups.
  Do not combine enabled native lookup with legacy assist, prompt-lookup bounds
  or lookup environment overrides. Conflict checks and effective-mode warnings
  distinguish omitted settings from explicit null, empty or disabled inputs.

### Lookup measurements and limits

In one TP4 greedy A/X observation on four V100 GPUs, effective tokens committed
per round increased by about **2.3%** (long reasoning about **3.6%**, repeated
editing about **0.8%**). Outputs were token-identical across six long requests.
The observed mean round time decreased by about 0.5%; there are not enough
repeated measurements to establish a stable speed change. **Gains with sampling
have not been measured.** This is a reduction in verification rounds, not an
end-to-end throughput claim.

These observations used the pre-rebuild candidate with seven drafts, DCP2,
`spec_state_replay=false` and a 48 GiB host KV store. They are not measurements
of the rebuilt branch on current main, nor evidence for a larger host KV
capacity. The method remains optional; gains depend on repeated content and
proposal acceptance.

CPU exact-enumeration and single-GPU dense/sparse distribution tests cover the
proposal changes, including short continuations and full acceptance with a bonus
token. They do not establish real-model TP4 sampling gains or full release
qualification. Current-main TP4 off/on comparisons, real-model graph/KV resources,
content-level CPU KV reload, sustained concurrency and the full statistics cost
budget remain part of release acceptance.

### Lookup statistics

Collection is effective only when native lookup, `collect_stats` and the engine's
existing statistics logging are all enabled. `collect_stats` defaults to true
inside the optional lookup config; it does not enable lookup. With collection
disabled, no lookup counter/export buffers or counter graph nodes are created.
The switch is an instance-startup setting, not a hot toggle.

Metrics use the `vllm:dflash2_lookup_` prefix and existing model/engine labels,
without request IDs or token text:

| Suffix | Type | Meaning |
| --- | --- | --- |
| `eligible_rounds_total` | Counter | Eligible request-row lookup attempts, including misses |
| `candidate_rounds_total` | Counter | Match meets `nmin` and has at least one continuation token |
| `gate_pass_rounds_total` | Counter | Candidate's first token agrees with the neural first token |
| `adopted_rounds_total` | Counter | Gate passes and at least one later proposal position is replaced |
| `match_length` | Histogram | Candidate suffix-match length; exposes bucket, count and sum |
| `stats_pending` | Gauge | Observed producer watermark exceeds the exported coverage |
| `stats_producer` | Gauge | Producer submission watermark, not the eligible-round count |
| `stats_covered` | Gauge | Producer watermark covered by the latest applied snapshot |
| `stats_last_export_timestamp_seconds` | Gauge | Time of the latest applied snapshot |

Each eligible row contributes one attempt, so a three-request batch can add
three attempts while the producer submission watermark advances once. Gate
checks, histories, proposal probabilities and RNG positions remain per request.

The match histogram has boundaries 1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128 and
+Inf. Misses do not enter a zero bucket. Adoption rate is
`rate(adopted_rounds_total) / rate(eligible_rounds_total)` with the full prefix;
a zero denominator means no samples. Adoption describes proposal replacement,
not verifier acceptance or saved rounds.

Only the designated output rank publishes counters. Owner, worker generation
and sequence checks prevent duplicate application; a restarted instance may
reset counters. During activity, completed cumulative snapshots travel with
normal outputs. Idle tails wait for resumed activity or normal shutdown, with
pending/coverage/export time making the lag visible. No statistics RPC is sent
while idle. Normal shutdown records a final receipt with sequence, coverage and
completion status; the API metrics endpoint may already be closed. Abnormal
termination can lose the unexported tail. UniProc conservatively isolates an
instance whose outstanding device access cannot be proved complete and requires
process restart.

The three logical counter/snapshot/pinned arrays total 432 bytes, plus CPU
metadata, two CUDA events, one export stream and allocator/graph overhead.
Single-V100 component measurements observed query/merge thread-CPU p99 of about
1.9, 2.8 and 33.9 microseconds for idle, pending and completed slots, and about
68.3 microseconds for the full snapshot-triggering service. These are component
costs, not TP4 end-to-end overhead bounds; the TP4 statistics percentage remains
to be measured against a real-model decode-round denominator.

## State replay (optional)

GDN speculative state replay keeps two recurrent-state blocks per running request instead of nine: the verifier stores no per-draft state, and after sampling the accepted tokens are replayed into the request's block. It is enabled by default in current development for the supported ordinary DFlash2 route and set explicitly in the recommended configuration. Released 0.1.4–0.1.5 require that explicit `true`:

```bash
      "sm70_dflash2": {"context_kv_graph": true, "context_pipeline": true,
        "verify_fastpath": true, "fused_gdn_metadata": true,
        "fused_smallq_metadata": true, "grouped_smallq_metadata": true,
        "spec_state_replay": true}}' \
```

- An omitted `spec_state_replay` remains `None`: it selects replay on the supported ordinary route, but yields to an enabled `fused_gdn_verify` packed verifier on its separately qualified model contract. Explicit `true` overrides an automatic packed default; explicitly enabling both options is a configuration error. This recommendation does not change packed-verifier qualification.
- Explicit `"spec_state_replay": false` disables replay. MTP, non-speculative decoding and routes outside the existing replay requirements retain their current behavior.
- **Gain:** the GPU KV pool grew from 1,379,634 to 1,542,558 tokens (+11.8%; 2.63× to 2.94× maximum concurrency at 512K) on our 4 × V100 PCIe deployment. At eight concurrent 96K-token requests, peak KV use fell from 55% to 26%.
- **Cost (0.1.4, measured in the same session):** about +0.44 ms per decoding step with greedy drafting when the target samples (30.05 to 30.49 ms per step at one request, 30 replayed agent turns); with greedy decoding about 1–4% lower throughput at one to four requests and level at eight. Outputs were identical with replay on and off (24 of 24 greedy streams, 30 of 30 sampled turns). With probabilistic drafting, recommended from 0.1.4, the replay costs about 0.8 ms per step (six real agent sessions at one request, 250 W), with outputs identical to replay off.
- **Trade-off:** the capacity gain helps long contexts or several long requests at once. Explicitly disabling replay trades away that capacity and avoids the replay's per-step work.
- **CUDA graph:** with replay on, the replay after sampling runs as a CUDA graph (`"spec_state_replay_graph": true`, the default). Without it, the replay's launches cost about 5 ms per step with sampling, because sampling has already drained the GPU queue. One graph is captured per verify shape, the second time the shape runs; with the `{"4": 3, "5": 4}` width policy that is eight graphs (one to eight requests). Each capture takes 35–90 ms once, at the step that first meets the shape. The graphs added no measurable GPU memory (peak 31,564 MiB per GPU with them, 31,580 MiB without, in the same benchmark). With all eight captured, eight concurrent 96K-token requests completed without preemption at 27% peak KV use, and GPU memory peaked at 31,884 of 32,768 MiB per GPU.
- Replay needs Model Runner V2, `--mamba-cache-mode align`, synchronous scheduling and no pipeline parallelism; elsewhere it stays off with a log line.

## What to expect

- Greedy output is not bitwise identical to MTP4. It differs at near-ties, because a
  different number of tokens is verified and DFlash uses a different MoE route for some
  batch sizes.
- English code and reasoning gain the most. Chinese is slower than MTP4 in almost every
  measured case: the drafter was trained on English. Very long prompts also favour MTP4.
- With the recommended 0.1.3 settings (sharded selector codebooks, the new graph settings, state replay off), the measured GPU KV pool is 1,379,634 tokens, with a reported maximum concurrency of 2.63× at 512K; with state replay on it is 1,542,558 tokens (2.94×). The 0.1.2 DFlash2 configuration measured 1,268,831 tokens (2.42×). In 0.1.4 with probabilistic drafting, the recommended setting, the pool is 1,366,121 tokens (2.61×); with greedy drafting it is unchanged at 1,379,634. See [GPU KV capacity by method](performance.md#gpu-kv-capacity-by-method).
- On a replay of real agent sessions (one request at a time, sampling, long reasoning),
  DFlash2 and MTP4 were on par: 0.97× the MTP4 speed (95% interval 0.90–1.01). DFlash2
  was ahead on tool calls, answers and contexts up to 64K, and behind in reasoning, above
  128K and on Chinese. MTP4 stays the default.
- The first request after an empty compile cache compiles the DFlash verify kernels.
- Measured on the reference setup with the 0.1.2 recommended config (greedy, tokens/s, DFlash2 vs
  MTP4): one request, agent code 181 / 158, English code 201 / 163, English reasoning
  195 / 163, Chinese explanation 101 / 105, Chinese chat 85 / 100; four mixed requests
  238 / 236; eight mixed requests 331 / 357 (0.93×).

## See also

- [Reference launch tutorial](getting-started.md#4-launch)
- [Performance and state replay measurements](performance.md)
- [Known limits](known-limits.md)
