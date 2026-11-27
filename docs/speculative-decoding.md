# Speculative decoding

ReVolta speeds up generation with speculative decoding. A small draft model proposes several tokens, and the target model checks them in one forward pass; each accepted draft token saves a full decoding step. ReVolta supports two draft methods: MTP, the default, and DFlash2. This page helps you choose between them, lists what the ReVolta profile turns on for each, and is the reference for MTP settings.

Prerequisite: a working [reference launch](getting-started.md). It already uses MTP, so you need nothing else to start.

## The two methods

| | MTP (default) | DFlash2 (optional) |
| --- | --- | --- |
| Draft model | A multi-token prediction (MTP) layer trained with the model, such as the `mtp-fp8/` draft in the AWQ checkpoints; see [Checkpoints](checkpoints.md) for the draft that goes with each checkpoint | A separate drafter, [incoai/Qwen3.8-Flash-Next-DFlash2](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2): five sliding-window layers with a 2,048-token window |
| Draft tokens per step | 4 in the reference configuration (MTP4); NVFP4 is also tested with 5 (MTP5) | 7, taken from the drafter's 8-token block |
| Drafting | Greedy | Probabilistic, set automatically |
| Draft KV cache | One attention layer over the whole context, in the target's KV format (INT8 by default) | The 2,048-token window only, in FP16 |
| Default draft-width schedule | `[[1,3,4],[4,8,3]]` for MTP4; none for other widths | `[[1,3,7],[4,4,3],[5,8,4]]` |
| GDN state replay | On | On |
| Other automatic settings | KV-only draft prefill; the wide verifier for AWQ MTP4 under the conditions [below](#what-the-profile-turns-on-for-mtp) | Sparse target rejection, native history lookup for single requests, verifier fast paths |
| Weights and KV modes | AWQ and NVFP4; GPU KV and Host KV | AWQ and NVFP4; GPU KV and Host KV |
| Extra setup | None | Download the drafter and make a draft view |

Both methods run on four V100 GPUs with tensor parallelism 4 (TP4) and decode context parallelism 2 (DCP2, which shards each request's KV cache across GPUs during decoding). The draft-width schedule, also called dynamic speculative decoding (DSD), lowers the number of draft tokens when more requests run at once; see [Dynamic speculative decoding](dynamic-speculative-decoding.md).

## Choosing a method

Start with MTP. It is the default, its draft comes with the AWQ checkpoints, and the reference configuration uses it.

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

## What to expect in 0.2.0

Measured with 0.2.0 on four V100 32 GB PCIe GPUs behind one PCIe switch, TP4 + DCP2, INT8 KV, `--gpu-memory-utilization 0.956` and `--max-model-len 524288`.
<!-- TODO(0.2.0-acceptance): workload, power cap and sampling settings for this table -->

| Configuration | Output tokens/s at 1 / 4 / 8 requests | GPU KV pool, tokens |
| --- | --- | --- |
| AWQ + MTP4, GPU KV (reference configuration) | <!-- TODO(0.2.0-acceptance): C1/C4/C8 output throughput, D+E1 --> | <!-- TODO(0.2.0-acceptance): GPU KV cache size in tokens, D+E1 --> |
| AWQ + DFlash2, GPU KV (the maintainer's production form) | <!-- TODO(0.2.0-acceptance): C1/C4/C8 output throughput, C --> | <!-- TODO(0.2.0-acceptance): GPU KV cache size in tokens, C --> |
| NVFP4 + MTP5, GPU KV | <!-- TODO(0.2.0-acceptance): C1/C4/C8 output throughput, E3a --> | <!-- TODO(0.2.0-acceptance): GPU KV cache size in tokens, E3a --> |
| NVFP4 + DFlash2, GPU KV | <!-- TODO(0.2.0-acceptance): C1/C4/C8 output throughput, E3b --> | <!-- TODO(0.2.0-acceptance): GPU KV cache size in tokens, E3b --> |

Results from earlier versions are in [Performance](performance.md).

## MTP settings

The [reference launch](getting-started.md#4-launch) shows the complete MTP speculative config. The MTP draft and its views are described in [Checkpoints](checkpoints.md).

### What the profile turns on for MTP

The ReVolta profile applies these defaults to `vllm serve` on four SM70 GPUs with TP4, FP16 activations and a Qwen3.8 Flash-Next checkpoint. Explicit values take precedence.

- **Draft-width schedule.** With `num_speculative_tokens` 4, prefix caching on (the profile default) and no pipeline parallelism, the schedule `[[1,3,4],[4,8,3]]` is used unless you set one: four draft tokens for one to three running requests, three from four onward. Other widths get no schedule. See [Dynamic speculative decoding](dynamic-speculative-decoding.md#default-schedules).
- **GDN state replay** (`gdn_state_replay`) with prefix caching on. When `--async-scheduling` is not given, it also selects synchronous scheduling. See [GDN state replay](#gdn-state-replay).
- **Draft KV format.** The draft inherits the target's KV format, which is planar INT8 with `--kv-cache-dtype auto`.
- **KV-only draft prefill** (`VLLM_SM70_MTP_KV_ONLY_PREFILL`, default `1`). While a long prompt is still being prefilled in chunks, the draft only writes its KV for those chunks, which shortens the time to first token of long prompts. On the chunk that completes the prompt, the draft computes only the row that is sampled, which shortens the time to the second token. It applies with data parallelism 1, GPU or Host draft KV and no multimodal input in the batch. Set `0` for full draft prefill.
- **Wide MTP verifier.** The profile turns it on only for AWQ g32 with MTP4 when all of these hold: data parallelism 1, DCP2, synchronous scheduling with GDN state replay, GPU KV (not Host KV mode; a CPU KV pool or disk tier is fine), and an interconnect that the engine runs as PCIe-switch P2P (not NVLink and not NCCL only). The reference configuration, with its 48 GiB CPU KV pool, qualifies on a PCIe-switch P2P system. The profile then enables `sm70_gdn_input_v154` in `--kernel-config`, `VLLM_SM70_MTP_GDN_NATIVE_VERIFY=1` and `VLLM_SM70_QWEN38_NATIVE_ROUTER_TOPK=1`, and raises `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS` to 48. These kernels change the accumulation order, so output can differ at near-ties. See [AWQ MTP4 wide verifier](configuration.md#awq-mtp4-wide-verifier).

### MTP settings you can change

- **`num_speculative_tokens`.** 4 is the reference width. For another width, the profile adds no schedule, so either pass one or accept a fixed width. The 0.2.0 acceptance ran MTP5 with `"num_speculative_tokens_per_batch_size": [[1,3,5],[4,8,3]]`. With INT8 or `fp8_per_token_head` KV, the profile raises the CUDA graph reserve for schedules with 5 or more draft tokens.
- **`kv_cache_dtype` in `--speculative-config`.** Set `"float16"` for an FP16 draft KV cache instead of the inherited format. FP16 draft KV takes more GPU memory per token, so the KV pool is smaller. It is accepted with an INT8 or `fp8_per_token_head` target, DCP1 or DCP2, no pipeline parallelism, Model Runner V2, and no KV connector other than the built-in `OffloadingConnector`. Other mixed formats fail at startup with the reason. In Host KV mode an FP16 draft keeps its history on the GPU; see [KV offload](kv-offload.md).
- **`draft_sample_method`.** Keep the default, `"greedy"`. The fast MTP draft paths (local argmax and the QPN8 draft head) need greedy drafting, and probabilistic MTP drafting corrupted output at temperature above 0 with more than two concurrent requests upstream ([1CatAI/1Cat-vLLM#656](https://github.com/1CatAI/1Cat-vLLM/issues/656)). This setting only affects drafting; clients can still use any sampling parameters.
- **`"gdn_state_replay": false`** keeps one recurrent state block per draft position. See [GDN state replay](#gdn-state-replay).
- **`VLLM_1CAT_ADAPTIVE_MTP`** (experimental, default `off`). Adapts the draft width of a single decoding request to its recent acceptance, up to the schedule's width for one request. It needs a draft-width schedule, synchronous scheduling and Model Runner V2. `on` needs at least 3 draft tokens; `auto` turns it on only on SM70 with at least 5. It has no effect while more than one request is decoding.
- **Draft experts and vocabulary.** `mtp_expert_quantization` selects the draft's routed-expert format; see [Checkpoints](checkpoints.md). An optional static draft vocabulary is described in [Configuration](configuration.md#optional-mtp-draft-vocabulary-experimental).

## GDN state replay

Qwen3.8 Flash-Next mixes attention layers with recurrent GDN (gated delta network) layers. Each verified draft position changes the recurrent state. Without replay, a running request keeps one state block per draft position, 2 + K blocks per GDN cache group for K draft tokens. With replay it keeps two: the verifier stores no per-position state, and after sampling the accepted tokens are replayed into the request's block. The saved GPU memory goes to the KV cache, which matters most for long contexts and concurrent requests.

Both methods turn replay on by default where it is supported: SM70, Model Runner V2, `--mamba-cache-mode align`, synchronous scheduling and no pipeline parallelism. Elsewhere it stays off with a log line. The startup log confirms it with `GDN speculative state replay: recurrent layers keep 2 state blocks per running request and replay accepted tokens after sampling.` With explicit `--async-scheduling`, MTP keeps one state block per draft position; for DFlash2 see [Scheduling](dflash2.md#scheduling).

The replay for full verification batches runs as a CUDA graph, captured per batch shape (`gdn_state_replay_graph` in `--speculative-config`, default on for MTP); mixed batches replay without a graph. It adds a little GPU work per step. Output can differ from per-position state storage at near-ties because the two use different computation paths. With MTP4 (0.2.0rc11, four V100, TP4 + DCP2, sampling), replay made a decode round about 0.6 ms (2%) longer at one request, and turning it off used about 0.87 GiB more GPU memory per GPU with eight concurrent long-context requests. DFlash2 measurements are in [DFlash2 state replay](dflash2.md#state-replay).

To turn replay off, set `"gdn_state_replay": false` for MTP or `"sm70_dflash2": {"spec_state_replay": false}` for DFlash2. The DFlash2 specifics are in [DFlash2 state replay](dflash2.md#state-replay).

## See also

- [DFlash2 speculative decoding](dflash2.md)
- [Dynamic speculative decoding](dynamic-speculative-decoding.md)
- [Reference launch tutorial](getting-started.md#4-launch)
- [Checkpoints, KV scales and MTP drafts](checkpoints.md)
- [Configuration reference](configuration.md)
- [Performance and measurements](performance.md)
