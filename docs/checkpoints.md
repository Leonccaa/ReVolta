# Checkpoints and KV formats

<a id="checkpoints-kv-scales-and-mtp-drafts"></a>

Use this guide to choose a Qwen3.8 Flash-Next checkpoint, prepare it with its MTP draft and 512K views, and choose the KV cache format. It assumes that ReVolta is [installed](getting-started.md#1-install) and that you have read the [reference launch](getting-started.md#4-launch); this page lists only what changes for each checkpoint. Speculative decoding settings are in [DFlash2](dflash2.md) and [Dynamic speculative decoding](dynamic-speculative-decoding.md).

## Choose a checkpoint

<a id="models"></a><a id="uncensored-or-base-awq"></a>

| Checkpoint | Weights | MTP draft | Test status |
| --- | --- | --- | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) (default) | 4-bit AWQ routed experts (group size 32), FP8 E4M3 PLE tables, 16-bit other weights; 137 GB | Its own `mtp-fp8/` | Covered by the 0.2.0 release tests <!-- TODO(0.2.0-acceptance): pass/fail summary, C, D+E1, E4M3, E2 --> |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | Same layout as Uncensored AWQ | Its own `mtp-fp8/`, converted and calibrated separately | Quantization quality measured; serving acceptance not run |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | ModelOpt NVFP4 routed experts; about 126 GiB | The Uncensored AWQ draft, shipped in the NVFP4 scale pack | Covered by the 0.2.0 release tests with MTP5 and DFlash2 <!-- TODO(0.2.0-acceptance): pass/fail summary, E3a, E3b -->; passed the 0.1.0 release gate |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVFP4 experts, MXFP8 layers, NVFP4 PLE table; about 95 GiB | Built-in MTP layer | Passed the 0.1.0 release gate; not in the 0.2.0 release tests |
| [NVIDIA NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD) | NVFP4 routed and shared experts, MXFP8 GDN and QSA projections, FP8 PLE table | Built-in MTP layer | Loading and serving checks passed in 0.2.0rc12; quantization quality not measured. The repository is not publicly downloadable at the time of writing |

- Start with **Uncensored AWQ**: it is the default, the most tested, and has the best measured quality of the compared formats.
- Choose **Base AWQ** for the original, not fine-tuned model in the same format.
- **Standard NVFP4** ran at similar speed to AWQ in most measured cases and needs about 0.5 GiB less weight memory per GPU (see [Standard NVFP4](#standard-nvfp4)).
- **FP4-Local** and **NVFP4-QAD** are smaller NVIDIA quantizations. FP4-Local has the lowest measured quality of the compared formats; QAD has not been measured.

V100 has no BF16 support, so serve every checkpoint with `--dtype half`.

<a id="quantization-comparison"></a>

Quality measurements for each format, against two BF16 references, are in [Quality measurements](performance.md#quality-measurements).

## Prepare the checkpoint

Preparing a checkpoint means three things: download a pinned revision, get its MTP draft, and make the 512K YaRN views. Use the script in [step 2 of Getting started](getting-started.md#2-download-the-checkpoint-and-make-512k-views) and change only these lines:

1. `repo` and `revision`: from the table below.
2. `views`: a separate directory per checkpoint, for example `/data/revolta/views/nvfp4`, so that the views of several checkpoints can coexist.
3. The draft line at the end, `make_view(ckpt / "mtp-fp8", views / "draft-512k")`: as in the last column.

| Checkpoint | `repo` | `revision` | Draft line |
| --- | --- | --- | --- |
| Uncensored AWQ g32 | `leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32` | `1fd7ab705731c9dacc3bcd5b5d15783c60706d64` | Keep it |
| Base AWQ g32 | `leoncca/Qwen3.8-Flash-Next-AWQ-g32` | `c6d84ec547717f71710f82ec3783027b61bbfc49` | Keep it |
| Standard NVFP4 | `RadixArk/Qwen3.8-Flash-Next-NVFP4` | `7b719225242aacd3dbd3f9407468c2ee9a9d2594` | Replace it with the three lines below: the draft is in the NVFP4 scale pack |
| NVIDIA FP4-Local | `nvidia/Qwen3.8-Flash-Next-FP4-Local` | `b9ce18a6887e18c089fdea8509a33e63429a3300` | Delete it: the built-in MTP layer loads from the target view |
| NVIDIA NVFP4-QAD | `nvidia/Qwen3.8-Flash-Next-NVFP4-QAD` | `eec0322cb03229cae4cd8e2fa45547542e472eac` | Delete it, as for FP4-Local. The repository is not publicly downloadable at the time of writing |

For Standard NVFP4, replace the draft line with:

```python
pack = Path("/data/revolta/models/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales")
snapshot_download("leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales", local_dir=pack, revision="v1.2-base-7b719225")
make_view(pack / "mtp-fp8", views / "draft-512k")
```

If prompt and output together stay within 262,144 tokens, you can serve the checkpoint directory itself with `--max-model-len 262144`; see [YaRN and native context](configuration.md#yarn-and-native-context).

### Launch

<a id="gpu-memory-per-card"></a><a id="--gpu-memory-utilization"></a>

Use the [reference command](getting-started.md#4-launch) and change only the paths and the draft (`<views>` is the directory you set in the script):

| Checkpoint | Model path | `--speculative-config` |
| --- | --- | --- |
| Uncensored AWQ, Base AWQ, Standard NVFP4 | `<views>/target-512k` | Set `"model"` to `<views>/draft-512k`; keep the other fields |
| NVIDIA FP4-Local, NVIDIA NVFP4-QAD | `<views>/target-512k` | Remove `"model"` and `"quantization"`: the built-in MTP layer loads from the target view |

<a id="gpu-memory"></a><a id="cuda-graph-reserve"></a>

Nothing else changes:

- The weight format comes from the checkpoint's quantization metadata, so `--quantization` is not needed. For a ModelOpt checkpoint the detected method replaces any `--quantization` value.
- The ReVolta profile selects the NVFP4 kernel settings and the [CUDA graph reserve](configuration.md#cuda-graph-reserve) by itself.
- The KV cache is INT8 for every checkpoint (`--kv-cache-dtype auto`), and the MTP draft inherits it. No KV scales are needed.

NVFP4-QAD was validated in 0.2.0rc12 with `--kv-cache-dtype fp8_per_token_head`. The 0.2.0 default gives it INT8, which has not been tested on QAD; set `fp8_per_token_head` to reproduce the validated setup.

### MTP drafts

<a id="kv-scales-and-draft-selection"></a>

- Use the draft that belongs to the target: an AWQ checkpoint's own `mtp-fp8/`, the scale pack's `mtp-fp8/` for Standard NVFP4, or the built-in layer of FP4-Local and NVFP4-QAD. The AWQ drafts also contain calibrated draft K/V scales, which only scalar E4M3 KV uses.
- The Standard NVFP4 draft is copied from Uncensored AWQ revision `1fd7ab705731c9dacc3bcd5b5d15783c60706d64`, including its license and provenance. With Standard NVFP4 it reached 62.5% acceptance in a 0.1.0 serving test; this is not an answer-quality measurement.
- On V100, the engine packs floating-point and block-FP8 MTP routed experts into Q8_0 once at load time and logs `Defaulting SM70 Qwen4Exp MTP routed experts to Q8_0`. Packed NVFP4 draft experts keep their checkpoint format. Add `"mtp_expert_quantization": "fp8"` to the speculative config to keep FP8 draft experts, or `"fp16"` for the FP16 reference. This affects only the draft's routed expert weights, not the draft KV.
- DFlash2 drafts need no K/V scales; see [DFlash2](dflash2.md).

## Weight and KV formats

<a id="kv-cache-format"></a><a id="int8-kv"></a>

V100 stores compressed weights and KV and converts them to FP16 for computation. The weight format is fixed by the checkpoint. The KV format is chosen with `--kv-cache-dtype`:

| Value | Storage | Calibration | Status in 0.2.0 |
| --- | --- | --- | --- |
| `auto` (default) | Planar INT8 on V100 for Qwen3.8 Flash-Next, with FP32 scales per token and KV head computed at runtime | None | Default; used by the reference configuration |
| `int8`, `int8_per_token_head` | The same planar INT8, selected explicitly | None | Same as `auto` on V100 |
| `fp8_per_token_head` | Planar E4M3 with FP32 scales per token and KV head computed at runtime | None | Supported |
| `fp8`, `fp8_e4m3` | Scalar E4M3 with calibrated per-layer K/V scales | Scale files | Frozen: still works, no longer optimized; the 0.2.0 release tests check function and answer quality, not speed |
| `float16` | FP16 | None | Holds about half as many KV tokens as the 8-bit formats in the same memory |

- **INT8 versus E4M3.** On the measured sets, INT8 has slightly lower KL and similar top-1 agreement to scalar E4M3 (see [Quality measurements](performance.md#quality-measurements)). This does not establish a difference in task-level answer quality.
- **Draft KV.** The MTP draft inherits the target format unless `speculative_config.kv_cache_dtype` sets another supported format. An FP16 draft (`"kv_cache_dtype": "float16"`) works with an INT8, `fp8_per_token_head` or scalar `fp8_e4m3` target. DFlash2 drafts use FP16 KV by default; see [What the profile sets](dflash2.md#what-the-profile-sets). Eligibility and page layouts are in [KV cache format](configuration.md#kv-cache-format).
- **Offload.** CPU KV offload and the `revolta_fs` SSD tier store planar INT8 together with its scale planes; the plain `fs` tier rejects planar formats. See [KV offload](kv-offload.md).
- **Capacity.** For measured GPU KV capacity by format, see [GPU KV capacity](performance.md#gpu-kv-capacity).

### Scalar E4M3 KV (frozen)

<a id="e4m3-kv"></a>

Use scalar E4M3 only to reproduce earlier measurements. It needs calibrated K/V scales that belong to a specific checkpoint revision:

| Target checkpoint | Target scales | MTP draft scales |
| --- | --- | --- |
| Uncensored AWQ g32 | Included in the checkpoint | Included in its `mtp-fp8/` |
| Base AWQ g32 | Included in the checkpoint | Included in its `mtp-fp8/` |
| Standard NVFP4 `7b719225` | [NVFP4 pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales) `v1.2-base-7b719225` (24 scales); materialize first | Included in the pack's `mtp-fp8/` |
| NVIDIA FP4-Local `b9ce18a6` | [FP4-Local pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales) `v1.1-base-b9ce18a6` (24 scales); materialize first | The built-in layer's two scales are in the same pack; the older `v1.0-base-b9ce18a6` tag has none |
| NVIDIA NVFP4-QAD | Not available | Not available |

For Standard NVFP4 and FP4-Local, materialize the pack onto the unmodified base checkpoint and make the target view from the result. In the preparation script, set `views` to a new directory (for example `/data/revolta/views/nvfp4-e4m3`) and replace the line `make_view(ckpt, views / "target-512k")` with the lines below, using the pack and tag from the table. Use a new output directory for each run.

```python
import subprocess, sys
pack_repo, pack_tag = "leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales", "v1.2-base-7b719225"
pack = Path("/data/revolta/models") / pack_repo.split("/")[-1]
snapshot_download(pack_repo, local_dir=pack, revision=pack_tag)
merged = Path(f"{ckpt}-e4m3")  # must not exist yet
subprocess.run([sys.executable, str(pack / "materialize_overlay.py"),
                "--pack-dir", str(pack), "--base-checkpoint", str(ckpt),
                "--output-dir", str(merged)], check=True)
make_view(merged, views / "target-512k")
```

The materializer refuses to run if the base index (and, for FP4-Local, the base `config.json`) or the scale files differ from its manifest. The materialized directory links to the base files, so keep the base checkpoint.

To launch, start from the reference command and:

- set `--kv-cache-dtype fp8_e4m3`; the MTP draft inherits it and loads its own scales;
- export `VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1`, so that startup fails if any target scale is missing. Without it, missing target scales fall back to 1.0 with a warning; that fallback is unvalidated and can saturate E4M3 values.

An MTP draft without calibrated scales is always rejected with E4M3 draft KV, even without the strict setting. For such a draft, or with the `v1.0` FP4-Local pack, set `"kv_cache_dtype": "float16"` in `--speculative-config`.

## Format-specific notes

### Standard NVFP4

<a id="standard-nvfp4-checkpoint-setup"></a><a id="speed-compared-with-awq"></a>

The checkpoint is about 126 GiB. The engine reads `quant_algo: NVFP4` from the checkpoint and selects `modelopt_fp4`.

<a id="automatic-configuration"></a><a id="settings-you-may-change"></a>

On V100 with TP4 and FP16, the ReVolta profile detects NVFP4 weights from the checkpoint's quantization metadata and sets the NVFP4 kernel switches (marked *NVFP4* in [Values applied by the ReVolta profile](configuration.md#values-applied-by-the-revolta-profile)) and the [CUDA graph reserve](configuration.md#cuda-graph-reserve). Explicit environment settings take precedence, and the startup profile line logs each audited value with its source. Raw E4M3 block scales, the profile default, cannot be combined with dual-batch overlap or microbatching; set `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=0` for those.

In 0.2.0rc12 development measurements, NVFP4 weights took about 0.5 GiB less GPU memory per GPU than Uncensored AWQ and ran at similar speed in most cases; see [Standard NVFP4 against AWQ](performance.md#standard-nvfp4-against-awq-020rc12).

### NVIDIA FP4-Local

<a id="nvidia-fp4-local-checkpoint-setup"></a>

The checkpoint is about 95 GiB. ReVolta runs its experts as W4A16 (activations stay FP16) and expands its MXFP8 layers to FP16. The engine detects the format (`modelopt_mixed`) from the checkpoint, and the same automatic NVFP4 profile applies.

The vision tower uses tensor parallelism by default. ReVolta expands 27 affected vision `fc2` weights to FP16 so that they can be sharded correctly; this configuration passed the large-image and video stress tests in 0.1.0 testing.

The NVFP4 PLE table uses about 26.8 GiB of pinned host memory across the four ranks.

### NVIDIA NVFP4-QAD

<a id="nvidia-nvfp4-qad-checkpoint-setup"></a>

`nvidia/Qwen3.8-Flash-Next-NVFP4-QAD` is a mixed-precision ModelOpt checkpoint: NVFP4 routed and shared experts, MXFP8 GDN and QSA projections (expanded to FP16 at load), an FP8 PLE table, and a built-in MTP layer whose experts use FP8 block scales. It ships no QSA K/V scales, so scalar E4M3 KV is not available.

In 0.2.0rc12 it was checked on four V100s with TP4/DCP2, MTP4, `fp8_per_token_head` KV and long-context Host KV: every quantized path loaded and served, and greedy, sampled, tool-call, 32K prefix-reuse and 4-way concurrent requests passed. Weights took 21.5 GiB per GPU, about 0.5 GiB less than Uncensored AWQ in the same configuration. Quantization quality against BF16 has not been measured.

## See also

- [Getting started](getting-started.md): the reference launch
- [DFlash2](dflash2.md) and [Dynamic speculative decoding](dynamic-speculative-decoding.md): speculative decoding settings
- [Configuration reference](configuration.md): KV dtype details and profile values
- [Performance](performance.md): GPU KV capacity, speed and quality measurements
