# Hy4-flash (validation support)

This page shows how to serve [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) as a text-only model on four V100 32 GB GPUs.

**Status.** Hy4-flash has validation support: it was validated once, in 0.2.0rc13, and its model code ships unchanged in 0.2.0. The 0.2.0 release tests did not run it, and the engine and the shared quantized-MoE kernels changed after 0.2.0rc13. Later versions are not guaranteed to keep it working. All measurements on this page come from 0.2.0rc13.

Not supported: image and video input, the checkpoint's DFly2 speculative decoding layers (they are skipped at load), pipeline parallelism, and GPU counts other than four.

## Checkpoints

| Checkpoint | Format | Weights per GPU | Quality vs BF16: KL ↓ / top-1 ↑ | Status |
| --- | --- | ---: | ---: | --- |
| [leoncca/Hy4-flash-AWQ-g32](https://huggingface.co/leoncca/Hy4-flash-AWQ-g32) (revision `6a6ab55577a64d74426696f78a847491b5a127d6`) **(recommended)** | 4-bit AWQ g32 routed experts; FP8 Engram tables; other weights 16-bit | 25.5 GiB | 0.123 / 92.2% | Serving, long-context and quality checks passed |
| [atakans/Hy4-flash-NVFP4](https://huggingface.co/atakans/Hy4-flash-NVFP4) (revision `922cbf11f2abf547951d22eabc76b876f2a9d03b`) | ModelOpt NVFP4 routed experts, shared experts and first dense layer; FP8 Engram tables | 23.9 GiB | 0.244 / 88.6% | Loads and serves; lower quality than AWQ |

Quality is the exact full-vocabulary KL and top-1 agreement on 8,675 teacher-forced positions from 28 held-out English, Chinese, code and math records, against the BF16 parent. V100 has no FP4 units, so NVFP4 weights run as W4A16 (FP16 activations; the checkpoint's activation scales are not used).

The Engram tables stay in pinned host memory: about 60 GiB in total, split across the four ranks.

## Launch

The default setup is the AWQ g32 checkpoint with an FP8 attention KV cache, decode context parallelism over the four GPUs (DCP4, up to 512K tokens) and thinking on.

```bash
hf download leoncca/Hy4-flash-AWQ-g32 \
  --revision 6a6ab55577a64d74426696f78a847491b5a127d6 --local-dir /data/revolta/models/hy4-flash-awq-g32

export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SM70_QUANT_BACKEND=turbomind
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
vllm serve /data/revolta/models/hy4-flash-awq-g32 --served-model-name hy4-flash \
  --tensor-parallel-size 4 --decode-context-parallel-size 4 \
  --dtype half --kv-cache-dtype fp8_ds_mla \
  --max-model-len 524288 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --enable-chunked-prefill --no-disable-hybrid-kv-cache-manager --enable-prefix-caching \
  --gpu-memory-utilization 0.95 --enable-prompt-tokens-details \
  --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice
```

- **Sampling** follows the official recommendation (`temperature=0.9`, `top_p=1.0`). The server reads these values from the checkpoint's `generation_config.json`, so requests that do not set them use them.
- **Thinking** is on by default. Set it per request in `chat_template_kwargs` (with the OpenAI SDK, through `extra_body`):
    - `reasoning_effort`: `"high"` (default, thinking on), `"no_think"` (direct answer) or `"max"`. The chat template accepts `"max"`, but the official model card documents only the first two, and ReVolta has not tested it. The template rejects any other value.
    - `preserved_thinking`: whether earlier turns keep their reasoning in the prompt. When unset, it is `false` without tools and `true` with tools.
- **KV cache.** `fp8_ds_mla` stores the attention cache as E4M3 with one FP32 scale per 128 values; the RoPE part stays FP16. On the held-out set it matched the FP16 cache (KL 0.1232 against 0.1236). Leave the option out to use an FP16 cache.
- **DCP4** splits the attention KV cache across the four GPUs instead of keeping a full copy on each one. The pool grows to about 1.34 M tokens and short-context decode is about 5% slower. DCP also requires TP4, DCP2 or DCP4 and no speculative decoding. For the fastest single-request decode at up to 256K tokens, drop `--decode-context-parallel-size 4` and set `--max-model-len 262144`.
- **`expandable_segments`** keeps the allocator from holding fragmented warm-up buffers; on the 0.2.0rc13 release image it raised the KV pool by about 30% with unchanged speed.
- **NVFP4 checkpoint.** Also set `export VLLM_SM70_NVFP4_TURBOMIND=1`.

## Measurements

0.2.0rc13 release image, AWQ checkpoint, four V100 32 GB GPUs, TP4, 200 W per GPU, FP8 KV cache, CUDA graphs for decode, `expandable_segments` on. The 122K-token rows and the needle checks come from development builds of the same code without `expandable_segments`.

| | DCP1 (256K) | DCP4 (512K, default) |
| --- | ---: | ---: |
| GPU KV pool | 374,076 tokens | 1,343,083 tokens |
| Decode, one request | 28.8 tok/s | 27.3 tok/s |
| Decode, four requests (total) | 88.7 tok/s | 84.7 tok/s |
| Time to first token, 2K / 7.2K prompt | 2.3 / 6.9 s (about 940 / 1,050 tok/s) | 2.2 / 6.0 s (about 990 / 1,210 tok/s) |
| 122K-token prompt: first request / decode | 163 s (about 750 tok/s) / 40.7 ms per token | 140 s (about 870 tok/s) / 38.8 ms per token |

A needle-retrieval check passed at 16K, 65K, 122K and 256K tokens with DCP1, and at 16K, 122K and 307K tokens with DCP4. Prefill of very long prompts is slow: a 256K-token prompt took about 30 minutes with DCP1.

DCP4 changes the attention accumulation order. On the held-out set its KL was 0.125 against 0.123 without DCP, with the same top-1 agreement.

## See also

- [Support and known limits](known-limits.md)
- [Configuration reference](configuration.md)
