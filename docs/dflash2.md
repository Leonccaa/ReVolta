# DFlash2 speculative decoding

<a id="dflash2-speculative-decoding-optional"></a>

This guide replaces MTP with DFlash2 and explains its settings. DFlash2 uses the [incoai DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) (tested revision `a22a1744`): five sliding-window layers, an 8-token block and a 2,048-token window. It is off unless you select it; the reference launch uses MTP. To decide between the two methods, see [Choosing a method](speculative-decoding.md#choosing-a-method).<a id="what-to-expect"></a>

Prerequisites: a working [reference launch](getting-started.md) and its 512K target view from [step 2](getting-started.md#2-download-the-checkpoint-and-make-512k-views). DFlash2 works with AWQ and NVFP4 checkpoints and with GPU KV and Host KV; the 0.2.0 release tests cover it with GPU KV.

## Enable DFlash2

**1. Make a draft view.** Startup refuses a drafter that does not declare how it reads the target's hyper-connection streams, so the view adds `"target_feature_reduction": "hc_mean"` to `config.json`. It also raises the drafter's position limit to 524,288 without RoPE scaling, as in the tested setup; the drafter's layers only see a 2,048-token window. The view links all other files to the downloaded checkpoint.

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

**2. Replace the MTP speculative config** in [step 4](getting-started.md#4-launch) and keep every other argument. Use the 512K target view with `--max-model-len 524288`, or use a length that both unmodified models support. With Docker, mount the drafter and view directories at the same paths inside the container.

```bash
  --speculative-config '{"method": "dflash",
      "model": "/data/revolta/views/dflash2-512k"}' \
```

The ReVolta profile sets everything else; see [What the profile sets](#what-the-profile-sets). Explicit values in the speculative config take precedence.

**3. Check the startup log.** Look for these lines:

- `Defaulting DFlash2 num_speculative_tokens to 7 from checkpoint block_size.`
- `Applied SM70 speculative defaults: speculative_config.draft_sample_method=probabilistic, speculative_config.attention_backend=FLASH_ATTN_V100`
- `Using FP16 draft KV cache for SM70 DFlash while the target uses int8.`
- `GDN speculative state replay: recurrent layers keep 2 state blocks per running request and replay accepted tokens after sampling.`
- With AWQ weights: `SM70 Qwen3.8 AWQ grouped QPN decode admitted for 2..48 rows (split=8, pack_rows=24, up to 2 row chunks).`

The first request after an empty compile cache compiles the DFlash2 verification kernels.

## What the profile sets

These defaults apply to `vllm serve` on four SM70 GPUs with TP4, FP16 activations, a Qwen3.8 Flash-Next checkpoint and the DFlash2 drafter. Each row lists how to change it.

| Setting | Default | To change |
| --- | --- | --- |
| `num_speculative_tokens` | `7`, from the drafter's block size | Keep 7: the default schedule and native history lookup need it |
| `draft_sample_method` | `"probabilistic"` | `"greedy"` also works |
| `attention_backend` (drafter) | `"FLASH_ATTN_V100"` | |
| `num_speculative_tokens_per_batch_size` | The DFlash2 [default schedule](dynamic-speculative-decoding.md#default-schedules) | Your own schedule, or `null` for a fixed 7 |
| `dflash2_lookup` | Native history lookup for single-request batches | See [Native history lookup](#native-history-lookup) |
| `sm70_dflash2.sparse_target_rejection` | `true` | `false`; see [Sparse target rejection](#sparse-target-rejection) |
| `sm70_dflash2.spec_state_replay` | `true` | `false`; see [State replay](#state-replay) |
| `sm70_dflash2`: `context_kv_graph`, `context_pipeline`, `verify_fastpath`, `fused_gdn_metadata`, `fused_smallq_metadata`, `grouped_smallq_metadata` | `true` | `false` per key. An explicit `context_kv_graph: false` also leaves an unset `context_pipeline` off |
| `sm70_dflash2.sharded_selector_codebook` | `true`: each GPU keeps 1/4 of the selector codebooks | `false` keeps full codebooks on every GPU |
| Drafter KV cache | FP16, whatever the target KV format | |
| `qsa_dcp_index_shard` in `--kernel-config` | `true` when unset and the model selects 512 index groups (Qwen3.8 Flash-Next): target QSA indexer keys are split over all tensor-parallel ranks; it takes effect at DCP2 | `false` |
| `VLLM_SM70_DFLASH2_FUSED_QKV_PACK` | `1`: packs the verifier's Q/K/V rows with one kernel | `0` |
| `VLLM_QWEN3NEXT_ENABLE_SHARED_MOE_OVERLAP` | `1` with AWQ g32 weights, without expert parallelism or dual-batch overlap | `0` |
| `VLLM_V2_CUDAGRAPH_MEM_MIB` | Sized for DFlash2 by the profile | Leave unset; values are in [CUDA graph reserve](configuration.md#cuda-graph-reserve) |
| `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` | `0`: no single-request verifier tail graphs, which leaves their GPU memory to the KV cache | `1`; then also raise `VLLM_V2_CUDAGRAPH_MEM_MIB` above the profile value, to at least the graph size the startup log reports |
| Scheduling | Synchronous | See [Scheduling](#scheduling) |

Types and defaults of the main `sm70_dflash2` keys are in [Configuration](configuration.md#speculative_configsm70_dflash2-keys).

## Settings you can change

<a id="settings"></a>

### Sparse target rejection

Sparse target rejection verifies drafts against the target's top-20 logits instead of gathering the full-vocabulary logits. It is on by default, for AWQ and NVFP4. It applies to a verification round only when every request in the batch is decoding, samples with temperature above 0 and `top_k` 20 (the checkpoint's default), and uses no `min_p`, penalties, logit bias, bad words, log-probabilities or structured output. All other rounds, including every greedy round, use dense verification. To use dense verification everywhere, set `"sm70_dflash2": {"sparse_target_rejection": false}`.

### Drafting

DFlash2 drafts probabilistically by default: the drafter samples its proposals and the target accepts them with the standard rejection rule. `"draft_sample_method": "greedy"` also works. Clients can use any sampling parameters either way.

### Draft-width schedule

The [default schedule](dynamic-speculative-decoding.md#default-schedules) publishes fewer draft tokens as more requests run. With K above 0, DFlash2 still drafts its full block each step and publishes only the first K tokens. The best schedule depends on the language and concurrency of your workload; measure on it before changing the default. The schedule format is in [Dynamic speculative decoding](dynamic-speculative-decoding.md).

## Scheduling

DFlash2 with a draft-width schedule uses synchronous scheduling unless `--async-scheduling` is given; the reference launch also passes `--no-async-scheduling`. Synchronous scheduling is the recommended setting.

Explicit `--async-scheduling` is accepted when the executor and model support it. DFlash2 always needs pipeline parallelism 1. What async does to the GDN state depends on the deployment:

- AWQ weights with GPU KV, no Host KV mode and no KV connector, with state replay left on: the engine switches to the next-round native GDN commit (`gdn_deferred_commit_metadata`) automatically and keeps two state blocks per request.
- Every other combination, including NVFP4, Host KV mode and the CPU KV pool of the reference configuration: one state block per draft position, so each request uses more GPU memory.

Native history lookup works with async scheduling. The legacy `ngram_assist` adaptive lookup does not; startup rejects that combination.

## Native history lookup

<a id="native-history-lookup-optional"></a>

Native history lookup searches the request's committed tokens for an earlier occurrence of the current suffix and proposes the tokens that followed it. When the first lookup token agrees with the drafter's first token, the lookup tokens replace the later draft positions. The first token and its proposal probabilities stay unchanged. A short continuation replaces only the positions it covers; the rest of the neural proposal keeps its probabilities. Lookup helps when the output repeats earlier text, for example code edits or quoted documents.

### Default and opt-out

The profile turns lookup on for single-request batches: longest match, suffix lengths 5 to 16, a 65,536-token search window, the `first_token` gate, statistics on and `max_requests` 1. This default applies when the speculative config has no `dflash2_lookup` and no legacy lookup setting (`ngram_assist`, `prompt_lookup_min`, `prompt_lookup_max` or a `VLLM_DFLASH2_LOOKUP_*` variable). Outside the profile, lookup is off unless configured. To turn it off, add `"dflash2_lookup": {"enabled": false}` to the speculative config.

To allow lookup in batches of up to three requests, set the whole object:

```json
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
```

### When lookup runs

Lookup runs only in batches with at most `max_requests` requests where the schedule selects width 7, so four or more requests always skip it. Inside such a batch it is decided per row: rows in prefill, rows with structured output, padding and rows without a sampled token or usable history do not attempt lookup, while the other decode rows of the same batch still do. Eligibility is checked again every round. The proposal width stays 7; lookup never widens it.

### Fields

| Field | Default in an explicit object | Accepted values |
| --- | --- | --- |
| `enabled` | `false` | Boolean |
| `policy` | none | `longest`; required when enabled |
| `nmin`, `nmax` | none | Positive integers with `nmin <= nmax`, at most `--max-model-len`; required when enabled |
| `search_window` | `65536` | Positive number of tokens of committed history |
| `gate` | `first_token` | `first_token` only: the first lookup token must equal the first draft token |
| `collect_stats` | `true` | Boolean; also needs engine statistics logging |
| `max_requests` | `3` | Integer from 1 to 3 |
| `expand.enabled` | `false` | Must stay `false` |

Lookup needs the DFlash2 drafter with seven draft tokens. Other policies, gates and expansion are rejected at startup. The legacy `ngram_assist` lookup and its `VLLM_DFLASH2_LOOKUP_*` variables still work, with no removal version set; use `dflash2_lookup` for new setups. Enabled native lookup together with legacy lookup inputs is rejected at startup.

### Lookup statistics

<a id="lookup-measurements-and-limits"></a>

Statistics are collected when native lookup, `collect_stats` and the engine's statistics logging are all on. With collection off, no counters are created. The setting takes effect at startup only.

The metrics use the `vllm:dflash2_lookup_` prefix and the existing model and engine labels, without request IDs or token text:

| Suffix | Type | Meaning |
| --- | --- | --- |
| `eligible_rounds_total` | Counter | Eligible request-row lookup attempts, including misses |
| `candidate_rounds_total` | Counter | A match of at least `nmin` tokens with at least one continuation token |
| `gate_pass_rounds_total` | Counter | The candidate's first token agrees with the drafter's first token |
| `adopted_rounds_total` | Counter | The gate passed and at least one later draft position was replaced |
| `match_length` | Histogram | Suffix-match length of candidates; bucket, count and sum |

Each eligible row counts as one attempt, so a three-request batch can add three attempts in one round. The match-length histogram has the bucket bounds 1, 2, 3, 4, 5, 6, 8, 12, 16, 32, 64, 128 and +Inf; misses are not counted in it. The adoption rate is `rate(vllm:dflash2_lookup_adopted_rounds_total[5m]) / rate(vllm:dflash2_lookup_eligible_rounds_total[5m])`; a zero denominator means no samples. Each request row that replaces at least one later draft position adds 1, however many positions it replaces; adoption does not count accepted tokens or saved rounds.

Counters are exported with normal outputs while requests run; after the last request, the remaining counts wait until the next request or a normal shutdown. The `stats_*` gauges in the same family only track this export. A restart resets the counters.

## State replay

<a id="state-replay-optional"></a>

DFlash2 replays the accepted tokens into the GDN state instead of storing a state block for each draft position: each running request keeps 2 blocks per GDN cache group instead of 9. This is on by default where it is supported; the general behaviour and requirements are in [GDN state replay](speculative-decoding.md#gdn-state-replay).

- `"sm70_dflash2": {"spec_state_replay": false}` keeps one state block per draft position. This trades GPU KV capacity for slightly less work per step. In Host KV mode, replay is also on by default, and an explicit `false` logs a warning.
- Replay and the packed verifier (`fused_gdn_verify`, off for Flash-Next) exclude each other. With GPU KV, an unset `spec_state_replay` gives way to an explicit `fused_gdn_verify: true`. In Host KV mode an unset `spec_state_replay` becomes `true`, so an explicit `fused_gdn_verify: true` is rejected unless you also set `spec_state_replay: false`. An explicit `true` for both is always a configuration error.
- `spec_state_replay_graph` (default `true`) captures the replay as a CUDA graph, once per full verification batch shape, the second time that shape runs. Mixed batches replay without a graph. Without the graph, the replay's launches cost several milliseconds per step after sampling. A top-level `gdn_state_replay_graph` value overrides this key.

## See also

- [Speculative decoding](speculative-decoding.md)
- [Dynamic speculative decoding](dynamic-speculative-decoding.md)
- [Reference launch tutorial](getting-started.md#4-launch)
- [Configuration reference](configuration.md)
- [Performance and measurements](performance.md)
- [Known limits](known-limits.md)
