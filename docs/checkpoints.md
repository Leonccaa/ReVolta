# Checkpoints, KV scales and MTP drafts

Use this guide to select and prepare a checkpoint, its KV scales and its MTP draft.
First [install ReVolta](getting-started.md#1-install); for a complete first launch, follow the tutorial.

- **Weight formats:** standard AWQ, NVFP4, NVIDIA FP4-Local, and NVIDIA NVFP4-QAD.
- **KV formats:** FP8 E4M3, FP16, and INT8.

## Models

| Checkpoint | Test status |
| --- | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) | Default; full serving acceptance passed |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | Quantization quality measured; full serving acceptance not run |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Full serving acceptance passed with format-specific settings |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | Full serving acceptance passed with vision-tower TP4 |
| [NVIDIA NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD) | Loading and serving checks passed on four V100s; quantization quality not yet measured |

Both AWQ checkpoints use group-size-32 4-bit routed experts, FP8 E4M3 PLE tables,
and 16-bit remaining weights. Each includes calibrated QSA E4M3 K/V scales and an
optional `mtp-fp8/` draft with separate calibrated scales. Use the draft from the
same repository as the target.

Use `--dtype half`: V100 has no BF16 support.

## KV scales and draft selection

Tensor-scale calibrated E4M3 KV is frozen: it still works, but it is no longer
optimized or tested in release acceptance. Eligible SM70 QSA models can use
`--kv-cache-dtype auto` (INT8) or `fp8_per_token_head` for E4M3 storage.

For explicit `fp8` / `fp8_e4m3`, calibrated scalar E4M3 KV scales belong to a
specific checkpoint revision. Keep
`VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1` enabled: missing target scales then
reject startup. Without it, missing target scales fall back to unit scale
with a warning. That fallback is unvalidated and can saturate E4M3 values.
An uncalibrated MTP draft with E4M3 draft KV is rejected even without strict
target-scale checking. Use the draft's own scales or set its
`"kv_cache_dtype": "float16"`.

| Target checkpoint | E4M3 target scales | MTP draft | Draft KV dtype |
| --- | --- | --- | --- |
| Uncensored AWQ g32 | Included in the same checkpoint | Its own `mtp-fp8/`, including draft scales | `fp8_e4m3` (or `float16`) |
| Base AWQ g32 | Included in the same checkpoint | Its own `mtp-fp8/`, converted and calibrated separately | `fp8_e4m3` (or `float16`) |
| Standard NVFP4 `7b719225` | [NVFP4 pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales), `v1.2-base-7b719225`; materialize first | Uncensored FP8 MTP and its scales, included in that pack | `fp8_e4m3` (or `float16`) |
| NVIDIA FP4-Local `b9ce18a6` | [FP4-Local pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales), `v1.1-base-b9ce18a6`; materialize first | The checkpoint's built-in MTP layer; its two draft scales are in the same pack | `fp8_e4m3` (or `float16`) |

DFlash2 drafts do not need these K/V scales. Their own configuration is in
[DFlash2 speculative decoding](dflash2.md).
The launch examples below explicitly select scalar `fp8_e4m3` for target and
MTP; the strict-scale instructions apply to that recipe. Omitting the target
CLI option selects `auto`. Eligible QSA models use planar INT8 on SM70 and
planar `fp8_per_token_head` on other supported CUDA devices. Both compute scales
at runtime. MTP inherits the target format when draft dtype is omitted. On
supported QSA Model Runner V2 configurations, an explicit FP16 draft can also
be used with an 8-bit target.
SM70 QSA DFlash/DFlash2 defaults to FP16 draft KV. See the
[KV dtype comparison](configuration.md#launch-options) for eligibility and ABI.

On V100, Flash-Next MTP uses Q8_0 for supported floating-point or block-FP8 routed
expert weights. The runtime packs the experts once at load time. Packed
NVFP4 draft experts keep their checkpoint format. To keep FP8 draft expert
storage, add `"mtp_expert_quantization":"fp8"` to the speculative config.
For the supported draft sources, `"mtp_expert_quantization":"fp16"` selects
the FP16 reference. These settings affect only the draft's routed expert
weights. The draft KV dtype is configured separately.

When target E4M3 scales are unavailable, FP16 KV needs no calibration but
holds roughly half as many KV tokens at the same memory budget. INT8 KV
computes per-token, per-head scales at runtime. MTP inherits an INT8 target
format by default. DFlash2 supports an INT8 target with an FP16 draft. On
Flash-V100, DFlash2 draft attention with sliding windows or attention sinks
requires FP16 draft KV for this combination.

The following steps pin every download. Retain the downloaded base and pack:
the materialized checkpoints and 512K YaRN views link to their files. Use a
fresh output directory for each run. **Materialize before making YaRN views**:
the materializer verifies the unmodified base index (and FP4-Local config).
Both target and draft need YaRN; `--hf-overrides` only reaches the target.
Views must also link directories such as `tensors/` and `calibration/`.
Run these commands in the installed ReVolta environment and apply the
[environment settings](getting-started.md#3-environment) before launching. These are checkpoint
setup and minimal launch examples; use the rest of the [quick-start launch
options](getting-started.md#4-launch) for the reference multimodal and CPU KV offload configuration.

## Uncensored or Base AWQ

Use the target and `mtp-fp8/` from the same repository. The default below is
Uncensored. To use Base AWQ, change `variant` to `"base"`; its revision is
pinned separately. No materialization is needed because both target and
draft scales are already included.

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

variant = "uncensored"  # or "base"
choices = {
    "uncensored": ("leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32",
                   "1fd7ab705731c9dacc3bcd5b5d15783c60706d64"),
    "base": ("leoncca/Qwen3.8-Flash-Next-AWQ-g32",
             "c6d84ec547717f71710f82ec3783027b61bbfc49"),
}
repo, revision = choices[variant]
ckpt = Path("/data/revolta/models") / repo.split("/")[-1]
views = Path("/data/revolta/views/awq")
snapshot_download(repo, revision=revision, local_dir=ckpt)

YARN = {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 262144}

def make_view(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        if item.name not in ("config.json", ".cache", ".git"):
            (dst / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    rope = dict(text["rope_parameters"])
    # Keep the outer default; YaRN applies only to text_config.
    config["rope_parameters"] = dict(rope)
    rope.update(YARN)
    text["rope_parameters"] = dict(rope)
    text["rope_scaling"] = dict(rope)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

make_view(ckpt, views / "target-512k")
make_view(ckpt / "mtp-fp8", views / "draft-512k")
EOF

VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1 \
vllm serve /data/revolta/views/awq/target-512k \
  --dtype half --tensor-parallel-size 4 --decode-context-parallel-size 2 \
  --gpu-memory-utilization 0.95 --max-model-len 524288 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align \
  --no-async-scheduling --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_V100 --kernel-config '{"ple_disk_cascade":false}' \
  --speculative-config '{"method":"mtp","model":"/data/revolta/views/awq/draft-512k","quantization":"modelopt_mixed","num_speculative_tokens":4,"draft_sample_method":"greedy","attention_backend":"FLASH_ATTN_V100","kv_cache_dtype":"fp8_e4m3"}'
```

## Standard NVFP4 checkpoint setup

The pack includes the 24 target scales and the optional Uncensored FP8 MTP
draft with its own two K/V scales. Download only the base model separately.
The draft is copied from Uncensored AWQ revision
`1fd7ab705731c9dacc3bcd5b5d15783c60706d64`, including its license and provenance.
ReVolta validated this pairing with **62.5% acceptance**; that number is a
serving-test result, not an answer-quality measurement.

```bash
python - <<'EOF'
import json
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

base = Path("/data/revolta/models/nvfp4-base")
pack = Path("/data/revolta/models/nvfp4-pack-v1.2")
merged = Path("/data/revolta/models/nvfp4-e4m3")
views = Path("/data/revolta/views/nvfp4")
snapshot_download("RadixArk/Qwen3.8-Flash-Next-NVFP4",
                  revision="7b719225242aacd3dbd3f9407468c2ee9a9d2594", local_dir=base)
snapshot_download("leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales",
                  revision="v1.2-base-7b719225", local_dir=pack)
subprocess.run([sys.executable, str(pack / "materialize_overlay.py"),
                "--pack-dir", str(pack), "--base-checkpoint", str(base),
                "--output-dir", str(merged)], check=True)

YARN = {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 262144}

def make_view(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        if item.name not in ("config.json", ".cache", ".git"):
            (dst / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    rope = dict(text["rope_parameters"])
    # Keep the outer default; YaRN applies only to text_config.
    config["rope_parameters"] = dict(rope)
    rope.update(YARN)
    text["rope_parameters"] = dict(rope)
    text["rope_scaling"] = dict(rope)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

make_view(merged, views / "target-512k")
make_view(pack / "mtp-fp8", views / "draft-512k")
EOF

VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1 \
vllm serve /data/revolta/views/nvfp4/target-512k \
  --dtype half \
  --tensor-parallel-size 4 --decode-context-parallel-size 2 \
  --gpu-memory-utilization 0.95 --max-model-len 524288 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align \
  --no-async-scheduling --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_V100 --kernel-config '{"ple_disk_cascade":false}' \
  --speculative-config '{"method":"mtp","model":"/data/revolta/views/nvfp4/draft-512k","quantization":"modelopt_mixed","num_speculative_tokens":4,"draft_sample_method":"greedy","attention_backend":"FLASH_ATTN_V100","kv_cache_dtype":"fp8_e4m3"}'
```

## NVIDIA FP4-Local checkpoint setup

The pack (`v1.1-base-b9ce18a6`) contains the 24 target scales and the two K/V
scales of the checkpoint's built-in MTP layer, so both target and draft can use
E4M3 KV. Omit the draft `model` field: the drafter uses the same materialized
512K view, its YaRN settings and its draft scales. In a paired test that changed
only the draft KV type, MTP acceptance was 63.1% (greedy) and 62.0% (sampled)
with E4M3 draft KV, against 64.4% and 61.9% with FP16 draft KV. The 0.1.3 release-image
smoke measured 1,474,106 GPU KV tokens (2.81× at 512K) with E4M3 target and draft KV,
against 1,321,612 (2.52×) with an FP16 draft cache.

`"kv_cache_dtype": "float16"` for the draft still works. The older
`v1.0-base-b9ce18a6` tag has no draft scales; with it, keep the draft in
`float16`, because an E4M3 draft cache without calibrated draft scales is
rejected at startup.

```bash
python - <<'EOF'
import json
import subprocess
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

base = Path("/data/revolta/models/fp4-local-base")
pack = Path("/data/revolta/models/fp4-local-pack-v1.1")
merged = Path("/data/revolta/models/fp4-local-e4m3")
views = Path("/data/revolta/views/fp4-local")
snapshot_download("nvidia/Qwen3.8-Flash-Next-FP4-Local",
                  revision="b9ce18a6887e18c089fdea8509a33e63429a3300", local_dir=base)
snapshot_download("leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales",
                  revision="v1.1-base-b9ce18a6", local_dir=pack)
subprocess.run([sys.executable, str(pack / "materialize_overlay.py"),
                "--pack-dir", str(pack), "--base-checkpoint", str(base),
                "--output-dir", str(merged)], check=True)

YARN = {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 262144}

def make_view(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        if item.name not in ("config.json", ".cache", ".git"):
            (dst / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    rope = dict(text["rope_parameters"])
    # Keep the outer default; YaRN applies only to text_config.
    config["rope_parameters"] = dict(rope)
    rope.update(YARN)
    text["rope_parameters"] = dict(rope)
    text["rope_scaling"] = dict(rope)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

# The built-in MTP uses this same view; no separate draft directory is needed.
make_view(merged, views / "target-512k")
EOF

VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1 \
vllm serve /data/revolta/views/fp4-local/target-512k \
  --quantization modelopt_mixed --dtype half \
  --tensor-parallel-size 4 --decode-context-parallel-size 2 \
  --gpu-memory-utilization 0.95 --max-model-len 524288 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align \
  --no-async-scheduling --kv-cache-dtype fp8_e4m3 \
  --attention-backend FLASH_ATTN_V100 --kernel-config '{"ple_disk_cascade":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy","attention_backend":"FLASH_ATTN_V100","kv_cache_dtype":"fp8_e4m3"}'
```

## NVIDIA NVFP4-QAD checkpoint setup

`nvidia/Qwen3.8-Flash-Next-NVFP4-QAD` is a mixed-precision ModelOpt checkpoint:
NVFP4 routed and shared experts, MXFP8 GDN and QSA projections (expanded to FP16
at load), an FP8 PLE table, and a built-in MTP layer whose experts use FP8 block
scales. It ships no QSA K/V scales, so this recipe uses `fp8_per_token_head`,
which needs no calibration. The built-in MTP layer follows the target's KV format.

Validated on four V100s (TP4/DCP2, MTP4, `fp8_per_token_head`, long-context host
KV): every quantized path loads and serves, and greedy, sampled, tool-call, 32K
prefix-reuse and 4-way concurrent requests pass. Weights take 21.5 GiB per GPU,
about 0.5 GiB less than the Uncensored AWQ checkpoint in the same configuration.
Quantization quality against BF16 has not been measured yet.

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

base = Path("/data/revolta/models/nvfp4-qad")
views = Path("/data/revolta/views/nvfp4-qad")
snapshot_download("nvidia/Qwen3.8-Flash-Next-NVFP4-QAD",
                  revision="eec0322cb03229cae4cd8e2fa45547542e472eac", local_dir=base)

YARN = {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 262144}

def make_view(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=False)
    for item in src.iterdir():
        if item.name not in ("config.json", ".cache", ".git"):
            (dst / item.name).symlink_to(item.resolve(), target_is_directory=item.is_dir())
    config = json.loads((src / "config.json").read_text())
    text = config["text_config"]
    rope = dict(text["rope_parameters"])
    # Keep the outer default; YaRN applies only to text_config.
    config["rope_parameters"] = dict(rope)
    rope.update(YARN)
    text["rope_parameters"] = dict(rope)
    text["rope_scaling"] = dict(rope)
    (dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")

# The built-in MTP uses this same view; no separate draft directory is needed.
make_view(base, views / "target-512k")
EOF

vllm serve /data/revolta/views/nvfp4-qad/target-512k \
  --dtype half \
  --tensor-parallel-size 4 --decode-context-parallel-size 2 \
  --gpu-memory-utilization 0.95 --max-model-len 524288 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --enable-chunked-prefill --enable-prefix-caching --mamba-cache-mode align \
  --no-async-scheduling --kv-cache-dtype fp8_per_token_head \
  --attention-backend FLASH_ATTN_V100 --kernel-config '{"ple_disk_cascade":false}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4,"draft_sample_method":"greedy","attention_backend":"FLASH_ATTN_V100"}'
```

## Weight and KV formats

V100 stores compressed weights and KV, then converts them to FP16 for computation.
The documented reference recipe is AWQ g32 with explicit scalar E4M3 KV.
The CLI default `auto` selects planar INT8 for eligible QSA models on V100.
Use `fp8_per_token_head` for dynamic planar E4M3.

## Quantization comparison

No ReVolta version is specified for these measurements. See
[Quality measurements](performance.md#quality-measurements) for the datasets, reference and estimator.
The E4M3 results below use the original scalar format, not planar E4M3.

| Format | KL vs BF16 (reference below) | Top-1 agreement |
| --- | ---: | ---: |
| Uncensored AWQ g32 + E4M3 KV | 0.0273 | 96.63% |
| Standard NVFP4 + E4M3 KV | 0.0303 | 96.36% |
| NVIDIA FP4-Local + E4M3 KV | 0.0380 | 95.91% |
| Uncensored AWQ g32 + INT8 KV | 0.0267 | Within 0.1 percentage point of E4M3 |
| Uncensored AWQ g32 + FP16 KV | Within 0.001 of E4M3 | Same as E4M3 |

The first four rows use the same base BF16 reference; the Uncensored rows there include the effect of fine-tuning as well as quantization. The FP16 KV row is a separate comparison of FP16 and E4M3 KV against the Uncensored checkpoint's own BF16 parent (on short prompts: KL 0.0220 against 0.0214, top-1 97.08% for both; on the long-context set: 0.0239 against 0.0238).

## E4M3 KV

Frozen; see [KV scales and draft selection](#kv-scales-and-draft-selection).

For explicit scalar `fp8` / `fp8_e4m3`, use the calibrated per-layer K/V scales
included in the checkpoint. Keep
`VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1` enabled to reject missing scales.
Without it, the loader warns and falls back to 1.0.

On the long-context quality set, E4M3 and FP16 KV measured KL 0.0238 and 0.0239
against the Uncensored checkpoint's own BF16 parent.

## INT8 KV

With MTP4, set `--kv-cache-dtype int8` to select planar INT8 explicitly.
The draft inherits INT8 unless `speculative_config.kv_cache_dtype` specifies
another supported format, such as FP16. Scales are computed per token and head
at runtime.

On the measured sets, INT8 has slightly lower KL and similar top-1 agreement to
E4M3. This has not established a difference in task-level answer quality.

The built-in offloading connector supports CPU and SSD persistence of planar
INT8, including its scale planes. See [KV offload](kv-offload.md).

## Standard NVFP4

The checkpoint is about 126 GiB. Follow the
[checkpoint setup](#standard-nvfp4-checkpoint-setup) for the pinned base,
combined scale/draft pack and 512K views. This section covers the settings that
matter once it loads.

### Automatic configuration

`--quantization` is not needed. The engine reads `quant_algo: NVFP4` from the
checkpoint and selects `modelopt_fp4`; this replaces a `--quantization` value given on
the command line. On the FP16 QSA SM70 TP4 profile, ReVolta detects NVFP4
weights from checkpoint quantization metadata and sets the validated kernel
switches for unset variables:

- raw E4M3 block scales (`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=1`), which
  avoid keeping a second FP16 copy of the scales;
- raw grouped decode for 2–40 rows
  (`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE=1`);
- QPN dynamic decode for every live width 2–16
  (`VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=1`);
- TurboMind NVFP4 routes, grouped prefill and small-shape tuning, and a CUDA
  graph reserve chosen by speculative method (next section).

Explicit environment settings take precedence. The startup profile audit logs
each value and its source (`profile`, `environment`, or `upstream_default`).
The complete list is in
[Values applied by the ReVolta profile](configuration.md#values-applied-by-the-revolta-profile).

### CUDA graph reserve

MRv2 does not profile graph memory, so the profile reserves it before sizing the
KV cache through `VLLM_V2_CUDAGRAPH_MEM_MIB`:

| Speculative method | KV cache dtype | Reserve per GPU |
| --- | --- | ---: |
| MTP | `fp8_per_token_head` or `int8` | 600 MiB |
| MTP | other (for example `fp8_e4m3`) | 700 MiB |
| DFlash2 | `fp8_per_token_head` or `int8` | 1024 MiB (1000 MiB for AWQ), single-request tail graphs off |
| DFlash2 | other | 1200 MiB, single-request tail graphs off |

If the startup log shows `CUDA graphs took … more than the reserve`, set
`VLLM_V2_CUDAGRAPH_MEM_MIB` to at least the value in the message, or lower
`--gpu-memory-utilization`. Otherwise the KV pool takes memory the graphs need
and a large mixed step can run out of GPU memory. Measured capture size per GPU
with `fp8_per_token_head`: 0.51 GiB for NVFP4 with MTP4 and 0.95 GiB for NVFP4
with DFlash2 (tail graphs off). Other capture sizes, such as additional draft
widths from [dynamic speculative decoding](dynamic-speculative-decoding.md),
add graphs: check the startup log after changing them.

### KV cache format

On SM70, `--kv-cache-dtype auto` selects planar INT8 for eligible QSA models.
It computes scales at runtime, so the NVFP4 checkpoint works without the
scale pack. The explicit planar `fp8_per_token_head` option also computes
scales at runtime. Calibrated `fp8_e4m3` uses the scales that the
[pack](#standard-nvfp4-checkpoint-setup) supplies. MTP inherits the target format
by default and supports an explicit FP16 draft on eligible configurations.
See [KV scales and draft selection](#kv-scales-and-draft-selection).

### GPU memory per card

Measured on 4×V100 with TP4, MTP4, `fp8_per_token_head` and long-context host
KV, per GPU:

| Item | Standard NVFP4 (RadixArk) |
| --- | ---: |
| Weights | 21.48 GiB |
| torch peak | 1.41 GiB |
| Non-torch | 0.15 GiB |
| Warmup | 0.71 GiB |
| CUDA graph reserve | 0.59 GiB |
| GPU KV cache | 5.81 GiB |

Uncensored AWQ weights take 22.01 GiB in the same configuration. Raw scales save
about 1.69 GiB per GPU compared with prepared scales, and the MoE decode scratch
is shared by all layers.

### `--gpu-memory-utilization`

Use `0.95`. With long-context host KV, eight concurrent decodes plus three
74K-token prompts completed at `0.95` with MTP4 and at `0.956` with DFlash2.
Earlier versions reserved too little CUDA graph memory for NVFP4 with DFlash2
and could run out of memory on a large mixed step at `0.95`; on those versions
use 0.92–0.93.

### Speed compared with AWQ

Same configuration and time window on 4×V100 PCIe, 250 W per GPU, TP4/DCP2.
Single-request figures are per speculative round.

| Measurement | Uncensored AWQ | Standard NVFP4 |
| --- | ---: | ---: |
| MTP4, 1 request, sampling | 28.36 ms | 28.67 ms |
| MTP4, 1 request, greedy | 27.68 ms | 28.00 ms |
| MTP4, 4 requests | 206.9 tok/s | 207.6 tok/s |
| MTP4, 8 requests | 316.5 tok/s | 325.4 tok/s |
| DFlash2, 1 request, sampling / greedy | 29.41 / 28.25 ms | 29.65 / 28.69 ms |
| DFlash2, 1 request | 97.8 tok/s | 90.4 tok/s |
| DFlash2, 4 requests | 243.4 tok/s | 238.3 tok/s |
| DFlash2, 8 requests | 321.2 tok/s | 331.8 tok/s |

- With MTP4, NVFP4 rounds are about 1.1% slower and single-request tok/s is
  level. Single-request short-prompt throughput also depends on the MTP draft's
  acceptance rate, which comes from the draft shipped with the checkpoint, not
  from the NVFP4 kernel speed.
- Graph all-reduces of 100–200 KiB now use the in-stream push kernel on PCIe
  switches. Measured on NVFP4 after that change: MTP4 at 4 requests 214.7 →
  224.0 tok/s and at 8 requests 316.9 → 328.7 tok/s; DFlash2 at 5 requests
  243.7 → 269.9 tok/s. The table above predates it.
- At four or more requests, three MTP drafts are faster than four (NVFP4, 4
  requests: 240.9 against 217.5 tok/s), while one request accepts more with
  four. [Dynamic speculative decoding](dynamic-speculative-decoding.md) switches
  between them by batch size.
- Prepared scales (`VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=0`) run a single request
  about 0.5 ms per round faster, but are slower at 4 and 8 requests and use
  1.69 GiB more per GPU, so the profile does not select them.

### Settings you may change

Names are as defined in `vllm/envs.py`; see
[configuration](configuration.md#values-applied-by-the-revolta-profile) for the rest.

| Variable | When to change |
| --- | --- |
| `VLLM_V2_CUDAGRAPH_MEM_MIB` | Raise it when the startup warning above appears. A reserve that is too high wastes KV capacity. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE` | `0` keeps prepared FP16 scales: faster single request, slower at 4–8 requests, 1.69 GiB more per GPU. Raw scales cannot be combined with DBO or microbatching; the loader rejects that and asks you to disable this variable. |
| `VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE` | Needs raw scale storage; with `RAW_SCALE=0` the raw grouped route is not used. |
| `VLLM_SM70_DFLASH2_TAIL_CUDAGRAPHS` | The profile sets `0` for DFlash2. Set `1` to capture single-request verifier tails again, and raise the reserve to 1500 MiB (see [DFlash2](dflash2.md)). |

## NVIDIA FP4-Local

[NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
is about 95 GiB. ReVolta runs its experts as W4A16, leaves activations in FP16,
and expands MXFP8 layers to FP16.

The vision tower uses TP4 by default. ReVolta converts 27 affected vision `fc2`
weights to FP16 so that their dimensions can be sharded correctly. This configuration
passed the large-image and video stress tests.

The PLE table uses about 26.8 GiB of pinned host memory across four ranks.
Use `--quantization modelopt_mixed` (matches the automatic detection); the same automatic NVFP4 profile applies.
Follow the [checkpoint setup](#nvidia-fp4-local-checkpoint-setup) for pinned
downloads, the scale overlay, 512K views and the MTP draft KV options.

## See also

- [Reference launch tutorial](getting-started.md)
- [DFlash2 setup](dflash2.md)
- [GPU KV capacity and quality measurements](performance.md)
