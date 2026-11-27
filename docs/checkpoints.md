# Checkpoints and KV formats

Use this guide to choose a Qwen3.8 Flash-Next checkpoint, prepare it with its MTP draft and 512K views, and choose the KV cache format. It assumes that ReVolta is [installed](getting-started.md#1-install) and that you have read the [reference launch](getting-started.md#4-launch); this page lists only what changes for each checkpoint. Speculative decoding settings are in [DFlash2](dflash2.md) and [Dynamic speculative decoding](dynamic-speculative-decoding.md).

## Choose a checkpoint

| Checkpoint | Weights | MTP draft | Test status |
| --- | --- | --- | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) (default) | 4-bit AWQ routed experts (group size 32), FP8 E4M3 PLE tables, 16-bit other weights; 137 GB | Its own `mtp-fp8/` | Covered by the 0.2.0 release tests <!-- TODO(0.2.0-acceptance): pass/fail summary, C, D+E1, E4M3, E2 --> |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | Same layout as Uncensored AWQ | Its own `mtp-fp8/`, converted and calibrated separately | Quantization quality measured; serving acceptance not run |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | ModelOpt NVFP4 routed experts; about 126 GiB | The Uncensored AWQ draft, shipped in the NVFP4 scale pack | Covered by the 0.2.0 release tests with MTP5 and DFlash2 <!-- TODO(0.2.0-acceptance): pass/fail summary, E3a, E3b -->; passed the 0.1.0 release gate |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVFP4 experts, MXFP8 layers, NVFP4 PLE table; about 95 GiB | Built-in MTP layer | Passed the 0.1.0 release gate; not in the 0.2.0 release tests |
| [NVIDIA NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD) | NVFP4 routed and shared experts, MXFP8 GDN and QSA projections, FP8 PLE table | Built-in MTP layer | Loading and serving checks passed in 0.2.0rc12; quantization quality not measured |

- Start with **Uncensored AWQ**: it is the default, the most tested, and has the best measured quality of the formats compared below.
- Choose **Base AWQ** for the original, not fine-tuned model in the same format.
- **Standard NVFP4** ran at similar speed to AWQ in most measured cases and needs about 0.5 GiB less weight memory per GPU (see [Standard NVFP4](#standard-nvfp4)).
- **FP4-Local** and **NVFP4-QAD** are smaller NVIDIA quantizations. FP4-Local has the lowest measured quality of the compared formats; QAD has not been measured.

V100 has no BF16 support, so serve every checkpoint with `--dtype half`.

### Quantization comparison

These quality measurements were taken before 0.2.0; their ReVolta version was not recorded. See [Quality measurements](performance.md#quality-measurements) for the datasets, reference and estimator. The E4M3 rows use the scalar E4M3 KV format.

| Format | KL vs BF16 ↓ | Top-1 agreement ↑ |
| --- | ---: | ---: |
| Uncensored AWQ g32 + E4M3 KV | 0.0273 | 96.63% |
| Standard NVFP4 + E4M3 KV | 0.0303 | 96.36% |
| NVIDIA FP4-Local + E4M3 KV | 0.0380 | 95.91% |
| Uncensored AWQ g32 + INT8 KV | 0.0267 | Within 0.1 percentage point of E4M3 |
| Uncensored AWQ g32 + FP16 KV | Within 0.001 of E4M3 | Same as E4M3 |

The first four rows use the same base BF16 reference; the Uncensored rows therefore include the effect of fine-tuning as well as quantization. The FP16 KV row is a separate comparison of FP16 and E4M3 KV against the Uncensored checkpoint's own BF16 parent: on short prompts KL 0.0220 against 0.0214 and top-1 97.08% for both; on the long-context set KL 0.0239 against 0.0238.

## Prepare the checkpoint

Preparing a checkpoint means three things: download a pinned revision, get its MTP draft, and make the 512K YaRN views. The target model and the MTP draft both need YaRN, but `--hf-overrides` reaches only the target, so the script makes a view directory for each. A view links every file of its source and contains an edited `config.json`.

Set `CHOICE` and run the script in the ReVolta environment. Keep the downloads: the views link to their files.

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

CHOICE = "uncensored-awq"  # uncensored-awq, base-awq, nvfp4, fp4-local or nvfp4-qad

# Pinned revisions. "draft" names the MTP draft directory; checkpoints without
# it use their built-in MTP layer and need no draft view.
CHECKPOINTS = {
    "uncensored-awq": {
        "repo": "leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32",
        "revision": "1fd7ab705731c9dacc3bcd5b5d15783c60706d64",
        "draft": "mtp-fp8",
    },
    "base-awq": {
        "repo": "leoncca/Qwen3.8-Flash-Next-AWQ-g32",
        "revision": "c6d84ec547717f71710f82ec3783027b61bbfc49",
        "draft": "mtp-fp8",
    },
    "nvfp4": {
        "repo": "RadixArk/Qwen3.8-Flash-Next-NVFP4",
        "revision": "7b719225242aacd3dbd3f9407468c2ee9a9d2594",
        # The draft is in the scale pack: a copy of the Uncensored AWQ draft.
        "draft_repo": "leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales",
        "draft_revision": "v1.2-base-7b719225",
        "draft": "mtp-fp8",
    },
    "fp4-local": {
        "repo": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
        "revision": "b9ce18a6887e18c089fdea8509a33e63429a3300",
    },
    "nvfp4-qad": {
        "repo": "nvidia/Qwen3.8-Flash-Next-NVFP4-QAD",
        "revision": "eec0322cb03229cae4cd8e2fa45547542e472eac",
    },
}

YARN = {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 262144}

def download(repo: str, revision: str) -> Path:
    path = Path("/data/revolta/models") / repo.split("/")[-1]
    snapshot_download(repo, revision=revision, local_dir=path)
    return path

def make_view(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name not in ("config.json", ".cache", ".git"):
            link = dst / item.name
            if not link.exists():
                link.symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    rope = dict(text["rope_parameters"])
    # YaRN goes only into text_config. Write the outer default explicitly;
    # otherwise the outer config inherits the YaRN values and fails to load.
    config["rope_parameters"] = dict(rope)
    rope.update(YARN)
    text["rope_parameters"] = dict(rope)
    text["rope_scaling"] = dict(rope)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

c = CHECKPOINTS[CHOICE]
views = Path("/data/revolta/views") / CHOICE
ckpt = download(c["repo"], c["revision"])
make_view(ckpt, views / "target-512k")
if "draft" in c:
    draft_src = ckpt
    if "draft_repo" in c:
        draft_src = download(c["draft_repo"], c["draft_revision"])
    make_view(draft_src / c["draft"], views / "draft-512k")
EOF
```

If your requests stay within 262,144 tokens, you can serve the checkpoint directory itself with `--max-model-len 262144`; see [YaRN and native context](configuration.md#yarn-and-native-context).

### Launch

Use the [reference command](getting-started.md#4-launch) and change only the paths and the draft:

| Checkpoint | Model path | `--speculative-config` |
| --- | --- | --- |
| Uncensored AWQ, Base AWQ, Standard NVFP4 | `/data/revolta/views/<CHOICE>/target-512k` | Set `"model"` to `/data/revolta/views/<CHOICE>/draft-512k`; keep the other fields |
| NVIDIA FP4-Local, NVIDIA NVFP4-QAD | `/data/revolta/views/<CHOICE>/target-512k` | Remove `"model"` and `"quantization"`: the built-in MTP layer loads from the target view |

Nothing else changes:

- The weight format comes from the checkpoint's quantization metadata, so `--quantization` is not needed. For a ModelOpt checkpoint the detected method replaces any `--quantization` value.
- The ReVolta profile selects the NVFP4 kernel settings and the CUDA graph reserve by itself (see [Standard NVFP4](#standard-nvfp4) and [CUDA graph reserve](#cuda-graph-reserve)).
- The KV cache is INT8 for every checkpoint (`--kv-cache-dtype auto`), and the MTP draft inherits it. No KV scales are needed.

NVFP4-QAD was validated in 0.2.0rc12 with `--kv-cache-dtype fp8_per_token_head`. The 0.2.0 default gives it INT8, which has not been tested on QAD; set `fp8_per_token_head` to reproduce the validated setup.

### MTP drafts

- Use the draft that belongs to the target: an AWQ checkpoint's own `mtp-fp8/`, the scale pack's `mtp-fp8/` for Standard NVFP4, or the built-in layer of FP4-Local and NVFP4-QAD. The AWQ drafts also contain calibrated draft K/V scales, which only scalar E4M3 KV uses.
- The Standard NVFP4 draft is copied from Uncensored AWQ revision `1fd7ab705731c9dacc3bcd5b5d15783c60706d64`, including its license and provenance. With Standard NVFP4 it reached 62.5% acceptance in a 0.1.x serving test; this is not an answer-quality measurement.
- On V100, the engine packs floating-point and block-FP8 MTP routed experts into Q8_0 once at load time and logs `Defaulting SM70 Qwen4Exp MTP routed experts to Q8_0`. Packed NVFP4 draft experts keep their checkpoint format. Add `"mtp_expert_quantization": "fp8"` to the speculative config to keep FP8 draft experts, or `"fp16"` for the FP16 reference. This affects only the draft's routed expert weights, not the draft KV.
- DFlash2 drafts need no K/V scales; see [DFlash2](dflash2.md).

## Weight and KV formats

V100 stores compressed weights and KV and converts them to FP16 for computation. The weight format is fixed by the checkpoint. The KV format is chosen with `--kv-cache-dtype`:

| Value | Storage | Calibration | Status in 0.2.0 |
| --- | --- | --- | --- |
| `auto` (default) | Planar INT8 on V100 for Qwen3.8 Flash-Next, with FP32 scales per token and KV head computed at runtime | None | Default; used by the reference configuration |
| `int8`, `int8_per_token_head` | The same planar INT8, selected explicitly | None | Same as `auto` on V100 |
| `fp8_per_token_head` | Planar E4M3 with FP32 scales per token and KV head computed at runtime | None | Supported |
| `fp8`, `fp8_e4m3` | Scalar E4M3 with calibrated per-layer K/V scales | Scale files | Frozen: still works, no longer optimized; the 0.2.0 release tests check function and answer quality, not speed |
| `float16` | FP16 | None | Holds about half as many KV tokens as the 8-bit formats in the same memory |

- **INT8 versus E4M3.** On the measured sets, INT8 has slightly lower KL and similar top-1 agreement to scalar E4M3 (see the [comparison](#quantization-comparison)). This does not establish a difference in task-level answer quality.
- **Draft KV.** The MTP draft inherits the target format unless `speculative_config.kv_cache_dtype` sets another supported format. An FP16 draft (`"kv_cache_dtype": "float16"`) works with an INT8 or `fp8_per_token_head` target. DFlash2 drafts use FP16 KV by default. Eligibility and page layouts are in [Launch options](configuration.md#launch-options).
- **Offload.** CPU KV offload and the `revolta_fs` SSD tier store planar INT8 together with its scale planes; the plain `fs` tier rejects planar formats. See [KV offload](kv-offload.md).
- **Capacity.** For measured GPU KV capacity by format, see [GPU KV capacity by method](performance.md#gpu-kv-capacity-by-method).

### Scalar E4M3 KV (frozen)

Use scalar E4M3 only to reproduce earlier measurements. It needs calibrated K/V scales that belong to a specific checkpoint revision:

| Target checkpoint | Target scales | MTP draft scales |
| --- | --- | --- |
| Uncensored AWQ g32 | Included in the checkpoint | Included in its `mtp-fp8/` |
| Base AWQ g32 | Included in the checkpoint | Included in its `mtp-fp8/` |
| Standard NVFP4 `7b719225` | [NVFP4 pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales) `v1.2-base-7b719225` (24 scales); materialize first | Included in the pack's `mtp-fp8/` |
| NVIDIA FP4-Local `b9ce18a6` | [FP4-Local pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales) `v1.1-base-b9ce18a6` (24 scales); materialize first | The built-in layer's two scales are in the same pack; the older `v1.0-base-b9ce18a6` tag has none |
| NVIDIA NVFP4-QAD | Not available | Not available |

For Standard NVFP4 and FP4-Local, materialize the pack onto the unmodified base checkpoint and make the target view from the result. In the preparation script, replace the line `make_view(ckpt, views / "target-512k")` with the lines below. Use a new output directory for each run.

```python
import subprocess, sys
PACKS = {
    "nvfp4": ("leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales", "v1.2-base-7b719225"),
    "fp4-local": ("leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales", "v1.1-base-b9ce18a6"),
}
pack = download(*PACKS[CHOICE])
merged = Path(f"{ckpt}-e4m3")  # must not exist yet
subprocess.run([sys.executable, str(pack / "materialize_overlay.py"),
                "--pack-dir", str(pack), "--base-checkpoint", str(ckpt),
                "--output-dir", str(merged)], check=True)
views = views.with_name(f"{CHOICE}-e4m3")
make_view(merged, views / "target-512k")
```

The materializer refuses to run if the base index (and, for FP4-Local, the base `config.json`) or the scale files differ from its manifest. The materialized directory links to the base files, so keep the base checkpoint.

To launch, start from the reference command and:

- set `--kv-cache-dtype fp8_e4m3`; the MTP draft inherits it and loads its own scales;
- export `VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1`, so that startup fails if any target scale is missing. Without it, missing target scales fall back to 1.0 with a warning; that fallback is unvalidated and can saturate E4M3 values.

An MTP draft without calibrated scales is always rejected with E4M3 draft KV, even without the strict setting. For such a draft, or with the `v1.0` FP4-Local pack, set `"kv_cache_dtype": "float16"` in `--speculative-config`.

## GPU memory

### CUDA graph reserve

Model Runner V2 cannot measure CUDA graph memory before it sizes the KV cache, so the ReVolta profile reserves graph memory per GPU through `VLLM_V2_CUDAGRAPH_MEM_MIB`. The value depends on the speculative method, the draft schedule and the KV format. Planar KV means `int8` (including `auto` on V100) or `fp8_per_token_head`.

| Speculative method | KV format | Reserve per GPU |
| --- | --- | ---: |
| MTP, fixed draft count | Planar | 600 MiB |
| MTP with a dynamic draft schedule, up to 4 drafts (the reference configuration) | Planar | 705 MiB |
| MTP with a dynamic draft schedule, 5 or more drafts | Planar | 800 MiB |
| MTP | Scalar E4M3 or FP16 | 700 MiB |
| None | Any | 700 MiB |
| DFlash2, AWQ | Planar | 1000 MiB (1150 MiB with a positive `--mixed-prefill-step-latency-ms`) |
| DFlash2, NVFP4 | Planar | 1024 MiB |
| DFlash2 | Scalar E4M3 or FP16 | 1200 MiB |

With DFlash2 the profile also turns off the single-request verifier tail graphs. An explicit `VLLM_V2_CUDAGRAPH_MEM_MIB` takes precedence.

If the startup log shows `CUDA graphs took … more than the reserve the KV cache was sized with`, set `VLLM_V2_CUDAGRAPH_MEM_MIB` to at least the value in the message, or lower `--gpu-memory-utilization`. Otherwise the KV cache uses memory that the graphs need, and a large mixed step can run out of GPU memory. A reserve that is too high wastes KV capacity.

## Format-specific notes

### Standard NVFP4

The checkpoint is about 126 GiB. The engine reads `quant_algo: NVFP4` from the checkpoint and selects `modelopt_fp4`.

**Automatic configuration.** On V100 with TP4 and FP16, the ReVolta profile detects NVFP4 weights from the checkpoint's quantization metadata and sets these switches when they are unset:

- raw E4M3 block scales (`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=1`), which avoid a second FP16 copy of the scales;
- raw grouped MoE decode for 2–40 rows (`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE=1`);
- QPN dynamic decode for every live width 2–16 (`VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=1`);
- TurboMind NVFP4 routes, grouped prefill and small-shape tuning;
- grouped Q8_0 MMQ for MTP draft batches above 64 rows (`VLLM_SM70_MTP_Q8_PREFILL_MMQ=1`);
- the [CUDA graph reserve](#cuda-graph-reserve) for the speculative method.

For long single-request prefills (1,024 to 16,384 rows) with TP4, each rank also scores only its share of the QSA query rows instead of all of them; this NVFP4-only path has no switch. Explicit environment settings take precedence, and the startup profile line logs each value with its source. The complete list is in [Values applied by the ReVolta profile](configuration.md#values-applied-by-the-revolta-profile).

**Memory.** In 0.2.0rc12 development measurements with TP4, NVFP4 weights took 21.48 GiB per GPU and Uncensored AWQ weights 22.01 GiB. Raw scales save about 1.69 GiB per GPU compared with prepared FP16 scales, and all layers share one MoE decode scratch buffer.

**Speed.** In paired measurements on 0.2.0rc12 development builds (same configuration and time window, 4 × V100 PCIe, 250 W per GPU, TP4/DCP2), NVFP4 and Uncensored AWQ ran at similar speed with MTP4 at one to eight requests and with DFlash2 at four and eight requests. With DFlash2 at one request, NVFP4 was slower (90.4 against 97.8 tok/s). <!-- TODO(0.2.0-acceptance): C1/C4/C8 tok/s, E3a NVFP4+MTP5 and E3b NVFP4+DFlash2 against D+E1 AWQ+MTP4 --> See [Performance](performance.md) for measured tables.

**Settings you may change.** Names are as defined in `vllm/envs.py`.

| Variable | When to change |
| --- | --- |
| `VLLM_V2_CUDAGRAPH_MEM_MIB` | Raise it when the reserve warning above appears. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE` | `0` keeps prepared FP16 scales. On 0.2.0rc12 development builds this ran a single request about 0.5 ms per round faster, but was slower at 4 and 8 requests and used 1.69 GiB more per GPU. Raw scales cannot be combined with DBO or microbatching; the loader rejects that and asks you to disable this variable. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE` | Needs raw scale storage; with `RAW_SCALE=0` the raw grouped route is not used. |
| `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` | The profile sets `0` for DFlash2. Set `1` to capture the single-request verifier tails again, and raise the reserve (see [DFlash2](dflash2.md)). |

### NVIDIA FP4-Local

The checkpoint is about 95 GiB. ReVolta runs its experts as W4A16 (activations stay FP16) and expands its MXFP8 layers to FP16. The engine detects the format (`modelopt_mixed`) from the checkpoint, and the same automatic NVFP4 profile applies.

The vision tower uses tensor parallelism by default. ReVolta expands 27 affected vision `fc2` weights to FP16 so that they can be sharded correctly; this configuration passed the large-image and video stress tests in 0.1.0 testing.

The NVFP4 PLE table uses about 26.8 GiB of pinned host memory across the four ranks.

### NVIDIA NVFP4-QAD

`nvidia/Qwen3.8-Flash-Next-NVFP4-QAD` is a mixed-precision ModelOpt checkpoint: NVFP4 routed and shared experts, MXFP8 GDN and QSA projections (expanded to FP16 at load), an FP8 PLE table, and a built-in MTP layer whose experts use FP8 block scales. It ships no QSA K/V scales, so scalar E4M3 KV is not available.

In 0.2.0rc12 it was checked on four V100s with TP4/DCP2, MTP4, `fp8_per_token_head` KV and long-context Host KV: every quantized path loaded and served, and greedy, sampled, tool-call, 32K prefix-reuse and 4-way concurrent requests passed. Weights took 21.5 GiB per GPU, about 0.5 GiB less than Uncensored AWQ in the same configuration. Quantization quality against BF16 has not been measured.

## See also

- [Getting started](getting-started.md): the reference launch
- [DFlash2](dflash2.md) and [Dynamic speculative decoding](dynamic-speculative-decoding.md): speculative decoding settings
- [Configuration reference](configuration.md): KV dtype details and profile values
- [Performance](performance.md): GPU KV capacity, speed and quality measurements
