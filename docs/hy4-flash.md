# Hy4-flash (validation support)

ReVolta 0.2.0rc13 adds text-only support for [tencent/Hy4-flash](https://huggingface.co/tencent/Hy4-flash) on four V100 32 GB GPUs. This is a validation release: Hy4-flash support is frozen at 0.2.0rc13, and later versions are not guaranteed to keep it working. Qwen3.8 Flash-Next behaviour is unchanged from 0.2.0rc12.

Not supported: image and video input, DFly2 speculative decoding, pipeline parallelism, and GPU counts other than four.

## Checkpoints

| Checkpoint | Format | Weights per GPU | Quality vs BF16: KL ↓ / top-1 ↑ | Status |
| --- | --- | ---: | ---: | --- |
| [leoncca/Hy4-flash-AWQ-g32](https://huggingface.co/leoncca/Hy4-flash-AWQ-g32) (revision `6a6ab55577a64d74426696f78a847491b5a127d6`) **(recommended)** | 4-bit AWQ g32 routed experts; FP8 Engram tables; other weights 16-bit | 25.5 GiB | 0.123 / 92.2% | Serving, long-context and quality checks passed |
| [atakans/Hy4-flash-NVFP4](https://huggingface.co/atakans/Hy4-flash-NVFP4) (revision `922cbf11f2abf547951d22eabc76b876f2a9d03b`) | ModelOpt NVFP4 routed experts, shared experts and first dense layer; FP8 Engram tables | 23.9 GiB | 0.244 / 88.6% | Loads and serves; lower quality than AWQ |

Quality is the exact full-vocabulary KL and top-1 agreement on 8,675 teacher-forced positions from 28 held-out English, Chinese, code and math records, against the BF16 parent. V100 has no FP4 units, so NVFP4 weights run as W4A16 (FP16 activations; the checkpoint's activation scales are not used).

The Engram tables stay in pinned host memory: about 60 GiB in total, split across the four ranks.

## Launch

```bash
export VLLM_USE_V2_MODEL_RUNNER=0 VLLM_SM70_QUANT_BACKEND=turbomind
# For the NVFP4 checkpoint, also: export VLLM_SM70_NVFP4_TURBOMIND=1
vllm serve /data/revolta/models/hy4-flash-awq-g32 --served-model-name hy4-flash \
  --tensor-parallel-size 4 --dtype half --kv-cache-dtype fp8_ds_mla \
  --max-model-len 262144 --max-num-batched-tokens 2048 --max-num-seqs 4 \
  --enable-chunked-prefill --no-disable-hybrid-kv-cache-manager --enable-prefix-caching \
  --gpu-memory-utilization 0.95 --enable-prompt-tokens-details \
  --reasoning-parser hy_v4 --tool-call-parser hy_v4 --enable-auto-tool-choice
```

For longer contexts, add `--decode-context-parallel-size 4` and raise `--max-model-len` to `524288`. This splits the attention KV cache across the four GPUs instead of keeping a full copy on each one.

Thinking is on by default. To turn it off for a request, send `"chat_template_kwargs": {"reasoning_effort": "no_think"}`.

`--kv-cache-dtype fp8_ds_mla` stores the attention cache as E4M3 with one FP32 scale per 128 values; the RoPE part stays FP16. On the held-out set it matched the FP16 cache (KL 0.1232 vs 0.1236). Leave the option out to use an FP16 cache.

## Measured on four V100 32 GB GPUs

AWQ checkpoint, TP4, 200 W per GPU, FP8 KV cache, CUDA graphs for decode.

| | DCP1 (256K) | DCP4 (512K) |
| --- | ---: | ---: |
| GPU KV pool | 289,521 tokens | 1,026,152 tokens |
| Decode, one request | 28.8 tok/s | 27.3 tok/s |
| Decode, four requests (total) | 88.4 tok/s | 84.5 tok/s |
| Time to first token, 2K / 7.2K prompt | 2.3 / 6.7 s | 2.1 / 5.9 s |
| 122K-token prompt: first request / decode | 163 s / 40.7 ms per token | 140 s / 38.8 ms per token |

A needle-retrieval check passed at 16K, 65K, 122K and 256K tokens with DCP1, and at 16K, 122K and 307K tokens with DCP4. Prefill of very long prompts is slow: a 256K-token prompt took about 30 minutes with DCP1.

DCP4 changes the attention accumulation order. On the held-out set its KL was 0.125 against 0.123 without DCP, with the same top-1 agreement.
