# Speculative decoding

ReVolta speeds up generation with speculative decoding. A small draft model proposes several tokens, and the target model checks them in one forward pass; each accepted draft token saves a full decoding step. ReVolta supports two draft methods: MTP, which the reference launch uses, and DFlash2. Without `--speculative-config`, speculative decoding stays off. This page helps you choose between them and is the reference for MTP settings.

Prerequisite: a working [reference launch](getting-started.md). It already uses MTP, so you need nothing else to start.

## The two methods

| | MTP (reference launch) | DFlash2 (optional) |
| --- | --- | --- |
| Draft model | A multi-token prediction (MTP) layer trained with the model, such as the `mtp-fp8/` draft in the AWQ checkpoints; see [Checkpoints](checkpoints.md) for the draft that goes with each checkpoint | A separate drafter, [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2): five sliding-window layers with a 2,048-token window |
| Draft tokens per step | 4 in the reference configuration (MTP4), fewer when more requests run; NVFP4 is also tested with 5 (MTP5) | Up to 7, taken from the drafter's 8-token block, fewer when more requests run |
| Drafting | Greedy | Probabilistic, set automatically |
| Draft KV cache | One attention layer over the whole context, in the target's KV format (INT8 by default) | The 2,048-token window only, in FP16 |
| Weights and KV modes | AWQ and NVFP4; GPU KV and Host KV | AWQ and NVFP4; GPU KV and Host KV |
| Extra setup | None | Download the drafter and make a draft view |

Both methods run on four V100 GPUs with tensor parallelism 4 (TP4) and decode context parallelism 2 (DCP2, which shards each request's KV cache across GPUs during decoding). The draft-width schedule, also called dynamic speculative decoding (DSD), lowers the number of draft tokens when more requests run at once; see [Dynamic speculative decoding](dynamic-speculative-decoding.md).

## Choosing a method

Start with MTP. Its draft comes with the AWQ checkpoints, and the reference configuration uses it.

Consider DFlash2 when:

- Most output is English code or English reasoning. The drafter was trained on English text, and these workloads gained the most in 0.1.2 measurements.
- You usually run one to four requests at a time.
- You need GPU KV capacity for long contexts. The DFlash2 draft keeps only a 2,048-token window per request, while the MTP draft layer stores KV for the whole context.

Stay with MTP when:

- Most text is Chinese. In 0.1.2 measurements DFlash2 was slower than MTP4 in almost every Chinese case.
- You usually run eight requests at once. With eight mixed requests, MTP4 was faster in 0.1.2 measurements.
- Very long prompts or long reasoning dominate. In a 0.1.2 replay of real agent sessions (one request at a time, sampling), DFlash2 was ahead on tool calls, answers and contexts up to 64K tokens, and behind in reasoning and above 128K; over all sessions the two methods were on par.
- You want the simplest deployment.

Greedy output differs between the two methods at near-ties, because they verify different numbers of tokens and use different kernels for some batch sizes. The measurements behind these statements, with their versions and conditions, are in [Performance](performance.md#dflash2-against-mtp4-on-pcie-012).

## MTP settings

The [reference launch](getting-started.md#4-launch) shows the complete MTP speculative config. The MTP draft and its views are described in [Checkpoints](checkpoints.md).

### What the profile turns on for MTP

For MTP the ReVolta profile adds the [default draft-width schedule](dynamic-speculative-decoding.md#default-schedules) for `num_speculative_tokens` 4 (also 5 on fully connected NVLink TP4), [GDN state replay](#gdn-state-replay) with synchronous scheduling, KV-only draft prefill (below) and, on PCIe switch P2P systems, the [AWQ MTP4 wide verifier](configuration.md#awq-mtp4-wide-verifier). The draft inherits the target's KV format, which is planar INT8 with `--kv-cache-dtype auto`. All profile values are in [ReVolta serving defaults](configuration.md#revolta-serving-defaults); explicit values take precedence.

### MTP settings you can change

- **`num_speculative_tokens`.** 4 is the reference width. The 0.2.0 PCIe acceptance set `"num_speculative_tokens_per_batch_size": [[1,3,5],[4,8,3]]` for MTP5. On fully connected NVLink TP4, the profile instead defaults to `[[1,8,3]]` for either maximum with prefix caching on. For widths without a default, pass a schedule or use a fixed width. With 5 or more, the [CUDA graph reserve](configuration.md#cuda-graph-reserve) is larger.
- **`kv_cache_dtype` in `--speculative-config`.** Set `"float16"` for an FP16 draft KV cache instead of the inherited format. FP16 draft KV takes more GPU memory per token, so the KV pool is smaller. It is accepted with an INT8, `fp8_per_token_head` or scalar `fp8_e4m3` target, on Model Runner V2 with DCP1 or DCP2, prefill context parallelism 1, no pipeline parallelism, the hybrid KV cache manager enabled, and no KV connector other than the built-in `OffloadingConnector` with `kv_role` `kv_both`. Other mixed formats fail at startup with the reason. In Host KV mode an FP16 draft keeps its history on the GPU; see [KV offload](kv-offload.md).
- **`draft_sample_method`.** Keep the default, `"greedy"`: the fast MTP draft paths need it, and probabilistic MTP drafting can corrupt output ([details](configuration.md#speculative-decoding)). Clients can still use any sampling parameters.
- **`"gdn_state_replay": false`** keeps one recurrent state block per draft position. See [GDN state replay](#gdn-state-replay).
- **KV-only draft prefill** (`VLLM_SM70_MTP_KV_ONLY_PREFILL`, default `1`). In an eager step where every request is still in the middle of a chunked prefill and no draft tokens wait for verification, the draft only writes its KV for those chunks, which shortens the time to first token of long prompts. In a step where a prefill completes, the draft computes outputs only for the sampled rows (outside FULL CUDA graph replay), which shortens the time to the second token. Both need data parallelism 1. Set `0` for full draft prefill.
- **Adaptive depth** for a single request (`VLLM_1CAT_ADAPTIVE_MTP`, experimental, off by default): see [Adaptive MTP depth](dynamic-speculative-decoding.md#adaptive-mtp-depth).
- **Draft experts.** `mtp_expert_quantization` selects the draft's routed-expert format; see [MTP drafts](checkpoints.md#mtp-drafts).

## GDN state replay

Qwen3.8 Flash-Next mixes attention layers with recurrent GDN (gated delta network) layers. Each verified draft position changes the recurrent state. Without replay, a running request keeps one state block per draft position, 2 + K blocks per GDN cache group for K draft tokens. With replay it keeps two: the verifier stores no per-position state, and after sampling the accepted tokens are replayed into the request's block. The saved GPU memory goes to the KV cache, which matters most for long contexts and concurrent requests.

Both methods turn replay on by default where it is supported: SM70, Model Runner V2, `--mamba-cache-mode align`, synchronous scheduling and no pipeline parallelism. Elsewhere it stays off with a log line. The startup log confirms it with `GDN speculative state replay: recurrent layers keep 2 state blocks per running request and replay accepted tokens after sampling.` With async scheduling, including the NVLink MTP default, MTP keeps one state block per draft position; for DFlash2 see [Scheduling](dflash2.md#scheduling).

The replay for full verification batches runs as a CUDA graph, captured per batch shape (`gdn_state_replay_graph` in `--speculative-config`, default on for MTP); mixed batches replay without a graph. It adds a little GPU work per step. Output can differ from per-position state storage at near-ties because the two use different computation paths. With MTP4 (0.2.0rc11, four V100, TP4 + DCP2, sampling), replay made a decode round about 0.6 ms (2%) longer at one request, and turning it off used about 0.87 GiB more GPU memory per GPU with eight concurrent long-context requests.

To turn replay off, set `"gdn_state_replay": false` for MTP or `"sm70_dflash2": {"spec_state_replay": false}` for DFlash2. The DFlash2 specifics are in [DFlash2 state replay](dflash2.md#state-replay).

## See also

- [DFlash2 speculative decoding](dflash2.md)
- [Dynamic speculative decoding](dynamic-speculative-decoding.md)
- [Reference launch tutorial](getting-started.md#4-launch)
- [Checkpoints, KV scales and MTP drafts](checkpoints.md)
- [Configuration reference](configuration.md)
- [Performance and measurements](performance.md)
