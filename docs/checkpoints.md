# Checkpoints, KV scales and MTP drafts

Use this guide to select and prepare a checkpoint, its KV scales and its MTP draft.
First [install ReVolta](getting-started.md#1-install); for a complete first launch, follow the tutorial.

- **Weight formats:** standard AWQ, NVFP4, and NVIDIA FP4-Local.
- **KV formats:** FP8 E4M3, FP16, and INT8.

## Models

| Checkpoint | Test status |
| --- | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) | Default; full serving acceptance passed |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | Quantization quality measured; full serving acceptance not run |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Full serving acceptance passed with format-specific settings |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | Full serving acceptance passed with vision-tower TP4 |

Both AWQ checkpoints use group-size-32 4-bit routed experts, FP8 E4M3 PLE tables,
and 16-bit remaining weights. Each includes calibrated QSA E4M3 K/V scales and an
optional `mtp-fp8/` draft with separate calibrated scales. Use the draft from the
same repository as the target.

Use `--dtype half`: V100 has no BF16 support.

## KV scales and draft selection

E4M3 KV scales belong to a specific checkpoint revision. Keep
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
When target E4M3 scales are unavailable, FP16 KV needs no calibration but
holds roughly half as many KV tokens at the same memory budget. INT8 KV
computes per-token, per-head scales at runtime; in the supported speculative
configuration use MTP with both target and draft set to `int8`. INT8 KV
cannot be combined with DFlash2 ([#25](https://github.com/Leonccaa/ReVolta/issues/25)).

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
with E4M3 draft KV, against 64.4% and 61.9% with FP16 draft KV.

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

## Weight and KV formats

V100 stores compressed weights and KV, then converts them to FP16 for computation.
The default is AWQ g32 with E4M3 KV.

## Quantization comparison

No ReVolta version is specified for these measurements. See
[Quality measurements](performance.md#quality-measurements) for the datasets, reference and estimator.

| Format | KL vs base BF16 | Top-1 agreement |
| --- | ---: | ---: |
| Uncensored AWQ g32 + E4M3 KV | 0.0273 | 96.63% |
| Standard NVFP4 + E4M3 KV | 0.0303 | 96.36% |
| NVIDIA FP4-Local + E4M3 KV | 0.0380 | 95.91% |
| Uncensored AWQ g32 + INT8 KV | 0.0267 | Within 0.1 percentage point of E4M3 |
| Uncensored AWQ g32 + FP16 KV | Within 0.001 of E4M3 | Same as E4M3 |

All KL and top-1 values use the same base BF16 reference; the FP16 KV row compares FP16 with E4M3 KV directly (on short prompts: KL 0.0220 against 0.0214, top-1 97.08% for both; on the long-context set: 0.0239 against 0.0238). The Uncensored rows include the effect of fine-tuning as well as quantization.

## E4M3 KV

Use the calibrated per-layer K/V scales included in the checkpoint. Keep
`VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1` enabled to reject missing scales.
Without it, the loader warns and falls back to 1.0.

On the long-context quality set, E4M3 and FP16 KV measured KL 0.0238 and 0.0239
against BF16.

## INT8 KV

With MTP4, set `--kv-cache-dtype int8` and `"kv_cache_dtype": "int8"` in the speculative
configuration. Target and draft KV types must match. Scales are computed per token
and head at runtime; no calibration is needed.

On the measured sets, INT8 has slightly lower KL and similar top-1 agreement to
E4M3. This has not established a difference in task-level answer quality.

Capacity is 1.1–1.3% lower than E4M3. Prefill speed is within ±1.3%; decoding is
about 5–8% slower. CPU offload works; the disk tier is not supported with INT8 in 0.1.2.

## Standard NVFP4

The checkpoint is about 126 GiB. Follow the
[checkpoint setup](#standard-nvfp4-checkpoint-setup) for the pinned base,
combined scale/draft pack and 512K views.

`--quantization` is not needed. The engine reads `quant_algo: NVFP4` from the
checkpoint and selects `modelopt_fp4`; this replaces a `--quantization` value given on
the command line. On the FP16 QSA SM70 TP4 profile,
ReVolta detects NVFP4 weights from checkpoint quantization metadata and sets
the validated kernel switches and a 700 MiB CUDA graph reserve automatically.
Explicit environment settings take precedence. The startup profile audit logs
each value and its source (`profile`, `environment`, or `upstream_default`).

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
