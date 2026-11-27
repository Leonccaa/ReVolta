# SM70 Q8000-core long prefill

This directory holds the default SM70 long-prefill attention kernel for FP16 Q8000 to Q8192 chunks with six query heads per KV head and head dimension 256 (the Qwen3.8 Flash-Next full-attention shape). It replaces the legacy architecture implementation inside the normal `vllm.vllm_flash_attn._vllm_fa2_C` extension and does not load a private DSO. The [v37 implementation](../sm70_v37/README.md) remains available in the same extension.

## Build

SM70 builds (`torch_cuda_arch_list=7.0`) build the SM70 FlashAttention extension and include this kernel by default. To omit it, pass `-DVLLM_SM70_79T_PREFILL=OFF`, for example through `CMAKE_ARGS`. `cmake/sm70_79t.cmake` is the source of truth for tile shapes and flags.

## Runtime selection

The Q8000-core route is the default for its admitted shapes. These settings select it explicitly; set `VLLM_FLASH_V100_PREFILL_D256_GQA_V37=1` to roll back to v37:

```bash
export VLLM_FLASH_V100_PREFILL_D256_GQA_ARCH_128K_EXPERIMENTAL=1  # default
export VLLM_FLASH_V100_PREFILL_D256_GQA_V37=0                     # default
export VLLM_SM70_DEBUG=routing                                    # optional route summary
```

The admitted shape is FP16 Q8000 through Q8192, Hq6/Hkv1, D256, causal, scale 1/16, with KV at least Q and at most 262144 tokens and aligned to 32 tokens. The 32-token alignment is required by the prefix PV Tensor Core K tile; other KV lengths stay on the general path. Q8000 runs through its qualified 320-token tail specialization. Q8192 has a native 256-token tail specialization. Q8001 through Q8191 are leading-padded to Q8192, which preserves bottom-right causal alignment and adds less than 2.4% work. Other shapes keep their existing routes.

For one full chunk per request, use a per-request long-prefill threshold of 8192. The total `max_num_batched_tokens` can be larger: for example, 16384 with `long_prefill_token_threshold=8192` schedules two full Q8192 chunks in the same step. Non-contiguous paged KV is gathered into a reusable workspace for each request, including multi-request batches.

E4M3 KV storage uses the separately resolved `sm70_v37_e4m3_bridge`; selecting this kernel does not disable that storage conversion.

## Route counters

| Counter | Meaning |
| --- | --- |
| `prefill_dense_d256_gqa_79t_fp32` | This compute family ran |
| `prefill_dense_d256_gqa_79t_fp32_q8192` | The native Q8192 kernel ran |
| `prefill_dense_d256_gqa_79t_fp32_q8192_pad` | A padded mixed chunk ran |

In an E4M3 cold run, both the architecture counter and the E4M3 bridge counter must appear on every rank.

## Numerical recipe

The original 79T recipe uses zero-shift exponentials and unguarded FP16 accumulation. Real model inputs overflow it, although zero-mean random inputs pass. The qualified kernel samples score maxima at stride 8 with a fixed margin and an exponent cap, centers biased values, and scales value residuals by an exact power of two with 64x headroom. PV uses FP16 Tensor Core operands with FP32 MMA accumulation, writes each 24K prefix block in FP32, and performs the online prefix/tail merge in FP32 before the value center is restored. The 128x256 threadblock and 64x64 warp shape avoid the register spills that made an earlier FP32 path slow.

This remains an approximate attention path: scores and probabilities are stored in FP16 and score maxima are sampled. A change must pass both the sampled FP32-oracle checks and cold model requests. See `VALIDATION.md` for the contract, quality limits and operator measurements. Do not report an operator TFLOPS result as model prompt tokens per second.

The private `MmaPipelined79T` template keeps transform `set_valid()` and `finalize()` hooks. Without finalization, the tail row masses stay zero and normalization produces non-finite outputs. The template has a distinct name so the hooks cannot affect other CUTLASS translation units. The tail adapter keeps the normalized-output contract (`unnormalized=0`).

## Tests

```bash
.venv/bin/python -m pytest tests/kernels/attention/test_sm70_79t_stability.py -q
.venv/bin/python -m pytest tests/v1/attention/test_sm70_flash_v100_policy.py -q
```

## See also

- [SM70 v37 long-prefill attention](../sm70_v37/README.md)
- [Building ReVolta](../../../BUILDING.md)
