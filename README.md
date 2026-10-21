# ReVolta

**A daily AI server on 4 × V100 32 GB.**

Run Qwen3.8 Flash-Next for text, images, and video, with an OpenAI-compatible API.

- **GPU-resident context:** a KV pool of about 1.14 million tokens with 8-bit KV, room for
  two 512K or four 256K contexts.
- **Weight formats:** standard AWQ, NVFP4, and NVIDIA FP4-Local.
- **KV formats:** FP8 E4M3, FP16, and INT8.
- **Speed:** about **380 tokens/s total output at 8 concurrent requests** (512-token
  responses, MTP4 on).
- **Cache reuse:** CPU memory and disk offload; a cached 32K document answered in **1.28 s vs 10.06 s** when recomputed.

## Measurements

### Speed

Reference system: 4 × V100 32 GB PCIe, one PCIe switch, P2P enabled, no NVLink.

| Concurrent requests | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Total output throughput, tok/s | 152.5 | 167.2 | 262.2 | 383.6 |

These are totals across all requests, including the wait for the first token:
512-token responses, thinking off, MTP4 on. C2 varies between runs; see
[Performance](#performance).

| Cold prompt | 8K tokens | 131K tokens |
| --- | ---: | ---: |
| Prefill throughput, tok/s | 3,510 | 3,020 |

**NVLink (community result).** [@jttdev](https://github.com/jttdev) measured 0.1.0 on
4 × V100-SXM2 32 GB with an NV2 NVLink mesh ([#2](https://github.com/Leonccaa/ReVolta/issues/2)):

| Concurrent requests | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| Total output throughput, tok/s | 153.5 | 223.4 | 326.3 | 467.8 |

Same three prompts and 512-token responses, thinking off, MTP4, 512K views, base AWQ
checkpoint, `--gpu-memory-utilization 0.925`. The method differs slightly from ours: each
level runs 3 requests per stream, and the single-request value is the mean of the three
prompts, including the wait for the first token. On this system
`--gpu-memory-utilization 0.95` also started (1,100,286 KV tokens), and two concurrent
retrieval requests of about 490K tokens each found all three needles.

### Coming back to a document

A cached 32K-token document answered a follow-up in **1.28 seconds from memory**,
compared with **10.06 seconds when recomputed**. Both produced identical output.

| Prefix location | Request time |
| --- | ---: |
| GPU | 1.23 s |
| Recomputed, offload disabled | 10.06 s |
| CPU memory | 1.28 s |
| CPU memory + filesystem cache | 1.60 s |

The request generated 64 tokens and reused 31,008 cached prompt tokens. A small GPU
pool forced eviction. The filesystem tier was tested on a network filesystem;
local SSD performance has not yet been measured.

### Quantization quality

The default AWQ format keeps the model close to its BF16 parent on the measured
English, Chinese, and code samples.

| AWQ checkpoint | BF16 reference | KL ↓ | Top-1 agreement ↑ |
| --- | --- | ---: | ---: |
| Uncensored AWQ g32 | Its own parent | 0.0214 | 97.08% |
| Base AWQ g32 | Base model | 0.0207 | 97.06% |

KL measures changes in next-token probabilities. Top-1 agreement counts how often
the most likely next token matches. These are distribution measurements, not task
accuracy or a percentage of model ability retained. See [Quality measurements](#quality-measurements).

## Project direction

ReVolta focuses on dependable daily service on V100. Releases check output correctness,
long-context retrieval, concurrent use, multimodal input, cache recovery, and speed
on the reference machine.

Next: faster INT8 KV decoding, simpler NVFP4 setup, INT8 disk caching, and community
testing on NVLink systems. Longer-term work includes reducing host-memory use and
runtime dependencies.

ReVolta is a Volta-focused fork of [vLLM](https://github.com/vllm-project/vllm),
through [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).

---

## Hardware

The reference system uses four V100 32 GB PCIe cards behind one PCIe switch
(PIX/PXB in `nvidia-smi topo -m`), with P2P enabled and a 128 GiB container memory limit.

| Configuration | Test status |
| --- | --- |
| 4 × V100 32 GB PCIe, TP4 + DCP2, P2P | Full acceptance passed on the reference configuration |
| P2P disabled on the same machine | Correct output; about 30–55% of P2P throughput |
| 4 × V100-SXM2 32 GB, NVLink (NV2 mesh) | Community test by [@jttdev](https://github.com/jttdev) ([#2](https://github.com/Leonccaa/ReVolta/issues/2)): speed, retrieval to about 490K tokens, vision. Not part of our release gate |
| V100 16 GB, other GPU counts | Not yet tested |

The engine selects the interconnect automatically and logs `SM70 interconnect=<mode>`.
P2P uses custom all-reduce for messages up to 80 KiB inside CUDA graphs; other
collectives use NCCL. Override with `VLLM_SM70_INTERCONNECT=auto|pex|nccl|nvlink`
or disable custom all-reduce with `--disable-custom-all-reduce`.

### System memory and storage

**128 GiB RAM is recommended** for the default 48 GiB CPU KV pool. The AWQ PLE
embedding table uses about 47.7 GiB of pinned host memory even with KV offload disabled.

| Container memory limit | KV offload | Result |
| --- | --- | --- |
| 128 GiB | Off, or 48 GiB CPU pool | Full acceptance passed |
| 80 GiB | Off | Memory stress passed; full acceptance not run |
| 64 GiB | Off | Out of memory during startup |

80 GiB is the smallest tested passing memory configuration. With less than 128 GiB,
reduce or disable the CPU KV pool.

Image and video preprocessing also uses host memory, in the API server process. In one
test, 8 concurrent requests with 8 different 2048-pixel images each used about 33 GB at
peak. With a 32 GiB CPU KV pool on a 125 GiB host, the kernel then killed a worker. The
same burst passed with a 16 GiB pool. If you expect bursts of large images or videos,
use a smaller CPU KV pool or limit the image size and the number of concurrent requests.

The AWQ checkpoint is 137 GB at the pinned revision:

| Part | Format | Size | Where it runs |
| --- | --- | ---: | --- |
| Routed experts | 4-bit AWQ, group size 32 | 74.9 GB | GPU |
| PLE n-gram tables | FP8 E4M3 | 51.3 GB | Pinned host memory |
| Other weights (attention, embeddings, `lm_head`, vision tower, shared experts) | 16-bit | 10.9 GB | GPU |
| Optional MTP draft (`mtp-fp8/`) | FP8 | 5.3 GB | GPU |

Allow extra disk space for compilation caches and the optional disk KV tier.

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

## Quick start

Reference configuration: Uncensored AWQ g32, TP4 + DCP2,
MTP4 greedy draft, E4M3 KV cache, 512K context through YaRN, multimodal input, 8
concurrent requests in testing, and a 48 GiB CPU KV pool.

### 1. Install

The distribution name is `revolta` (version `0.1.0`). The Python import name and the
CLI are still **`vllm`** (`import vllm`, `vllm serve`); this is intentional. Only
compute capability 7.0 is built, with CUDA 12.8, Python 3.12 and PyTorch 2.10.

From the wheel attached to the GitHub release (Linux x86-64, Python 3.12):

```bash
python3.12 -m venv /data/revolta/venv
. /data/revolta/venv/bin/activate
pip install "https://github.com/Leonccaa/ReVolta/releases/download/v0.1.0/revolta-0.1.0-cp312-cp312-linux_x86_64.whl"
vllm --version          # 0.1.0
```

The wheel metadata pins `torch==2.10.0`, `torchaudio==2.10.0` and
`torchvision==0.25.0` to the CUDA 12.8 builds on `download.pytorch.org`, so pip
installs them from there. Install the NVIDIA driver for CUDA 12.8 on the host.

ReVolta does not send the anonymous usage statistics that upstream vLLM reports.

Or use the Docker image:

```bash
docker pull ghcr.io/leonccaa/revolta:0.1.0
```

### 2. Download the checkpoint and make 512K views

The checkpoint's native context is 262,144 tokens. ReVolta reaches 524,288 tokens with
YaRN (factor 2). The target model **and** the MTP draft must both see the YaRN
settings. `--hf-overrides` reaches only the target model, so make two "view"
directories instead. Each view links to every checkpoint file and contains an edited
`config.json`. The validated setup used views of this kind.

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

repo = "leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32"
ckpt = Path("/data/revolta/models/Qwen3.8-Flash-Next-Uncensored-AWQ-g32")
views = Path("/data/revolta/views")
snapshot_download(repo, local_dir=ckpt, revision="5d0ef929358053b15626c0237fe19ebcc303727b")

YARN = {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 262144}

def make_view(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.is_file() and item.name != "config.json":
            link = dst / item.name
            if not link.exists():
                link.symlink_to(item.resolve())
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

make_view(ckpt, views / "target-512k")
make_view(ckpt / "mtp-fp8", views / "draft-512k")
EOF
```

### 3. Environment

The engine detects SM70, TP4, a QSA model (Qwen3.8 Flash-Next) and FP16, and then sets
the tuned kernel switches by itself (the "ReVolta profile"). Only these deployment
settings remain:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_GRAPH_REGISTER=0                    # required with expandable segments
export VLLM_SM70_AWQ_MOE_INT4_ZERO=1             # pack AWQ zero points to 4 bits in GPU memory
export VLLM_QWEN4EXP_PLE_HOST_GIB=12             # pinned host memory for the PLE table, GiB per TP rank
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=1200       # seconds; PLE loading takes a while
export VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1    # refuse to start without calibrated E4M3 KV scales
```

Do not set other `VLLM_SM70_*` variables unless you are troubleshooting. Explicit
values override the profile.

### 4. Launch

```bash
vllm serve /data/revolta/views/target-512k \
  --served-model-name Qwen3.8-Flash-Next \
  --dtype half \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 524288 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --prefix-cache-retention-interval 0 \
  --mamba-cache-mode align \
  --no-async-scheduling \
  --kv-cache-dtype fp8_e4m3 \
  --no-language-model-only \
  --limit-mm-per-prompt '{"video": 1}' \
  --media-io-kwargs '{"video": {"num_frames": 16}}' \
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --seed 0 \
  --speculative-config '{"method": "mtp",
      "model": "/data/revolta/views/draft-512k",
      "quantization": "modelopt_mixed",
      "num_speculative_tokens": 4,
      "draft_sample_method": "greedy",
      "attention_backend": "FLASH_ATTN_V100",
      "kv_cache_dtype": "fp8_e4m3"}' \
  --attention-backend FLASH_ATTN_V100 \
  --prefill-pressure-token-budget 1632 \
  --prefill-pressure-threshold 1 \
  --long-prefill-token-threshold 4896 \
  --mixed-prefill-step-latency-ms 0 \
  --kernel-config '{"ple_disk_cascade": false}' \
  --kv-transfer-config '{"kv_connector": "OffloadingConnector",
      "kv_role": "kv_both",
      "kv_connector_extra_config": {
        "spec_name": "TieringOffloadingSpec",
        "cpu_bytes_to_use": 51539607552,
        "eviction_policy": "lru",
        "secondary_tiers": [],
        "demote_superseded_states": true,
        "mamba_state_slots_reference_tokens": 65536}}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --host 0.0.0.0 --port 8000
```

Notes:

- **Startup** takes roughly 10–15 minutes: weight and PLE loading, `torch.compile`
  and CUDA graph capture. Later starts reuse the compile cache. On a first start with
  an empty compile cache, the reported GPU KV capacity can be one allocation unit
  (7,182 tokens) lower.
- **`draft_sample_method` must be `greedy`.** A probabilistic draft corrupts output
  above `max_num_seqs=2` at temperature > 0 (upstream 1Cat-vLLM issue #656). This
  setting is about draft verification only; clients can still use any sampling
  parameters.
- **`--gpu-memory-utilization 0.95`** passed the full gate, including the multimodal
  stress test with 8 concurrent requests. Higher values ran out of memory in earlier
  multimodal stress tests. If you change concurrency or media limits, measure again.
- The scheduler settings (`--prefill-pressure-*`, `--long-prefill-token-threshold`,
  `--mixed-prefill-step-latency-ms`) and `ple_disk_cascade: false` are the tested
  values for this machine. They are deployment settings, not universal defaults.
- `cpu_bytes_to_use` is the total for all four workers (48 GiB = 12 GiB per worker).
- Media limits and request budgets: see
  [multimodal input limits](.github/MULTIMODAL_INPUTS.md).

### 5. Check

The API is the OpenAI-compatible server from vLLM:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-Flash-Next",
       "messages": [{"role": "user", "content": "What is 17 + 23?"}],
       "max_tokens": 64}'
```

In the startup log, look for:

- `ReVolta profile {"enabled": true, ...}`: the profile is active. The line lists
  each tuned switch with its value and source (profile, environment or upstream
  default).
- `SM70 interconnect=pex_p2p` (or `no_p2p` / `nvlink`) from each worker.

### Docker

The image entry point is `vllm serve`. Mount the model and view directories at the
same path inside the container, so that the view symlinks resolve.

```bash
docker run --rm --gpus all \
  --shm-size 128g \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v /data/revolta:/data/revolta \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e NCCL_GRAPH_REGISTER=0 \
  -e VLLM_SM70_AWQ_MOE_INT4_ZERO=1 \
  -e VLLM_QWEN4EXP_PLE_HOST_GIB=12 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1200 \
  -e VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1 \
  ghcr.io/leonccaa/revolta:0.1.0 \
  /data/revolta/views/target-512k --served-model-name Qwen3.8-Flash-Next ...
```

Replace `...` with the remaining arguments from step 4. Container notes:

- The CPU KV pool is a file in `/dev/shm`. Make `--shm-size` larger than
  `cpu_bytes_to_use` (the validated container used 128g).
- `--cap-add SYS_PTRACE` was required in the validated container: the workers share
  memory with each other through `pidfd_getfd`.
  Whether `seccomp=unconfined` is needed remains to be checked.
- Do not bake `VLLM_SM70_*` switches into derived images. Values from the environment
  override the profile.

## KV offload modes

See [Coming back to a document](#coming-back-to-a-document) for measured readback times.

The KV cache always lives on the GPUs first. Offload adds lower tiers that keep
evicted prefixes, so a later request with the same prefix reads them back instead of
running prefill again. All three modes were tested for correct readback: tokens
generated after readback matched the off-mode reference exactly.

**1. Off.** Remove `--kv-transfer-config`. This uses the least host memory (80 GiB was the smallest tested passing configuration). Evicted prefixes are
computed again.

**2. CPU.** The quick start configuration:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 51539607552,
    "eviction_policy": "lru",
    "secondary_tiers": [],
    "demote_superseded_states": true,
    "mamba_state_slots_reference_tokens": 65536
  }
}
```

`cpu_bytes_to_use` sets the pool size (total for all workers). With 48 GiB, the pool
holds about 2.4M tokens. `mamba_state_slots_reference_tokens` sets how many
recurrent-state slots each layer group gets. A smaller reference gives more slots,
which helps when many short sessions share the pool.

**3. CPU + disk.** Add a filesystem tier, typically on a local SSD:

```json
{
  "kv_connector": "OffloadingConnector",
  "kv_role": "kv_both",
  "kv_connector_extra_config": {
    "spec_name": "TieringOffloadingSpec",
    "cpu_bytes_to_use": 2147483648,
    "eviction_policy": "lru",
    "secondary_tiers": [
      {
        "type": "fs",
        "root_dir": "/path/to/nvme/revolta-kv",
        "max_size_bytes": 8589934592,
        "min_free_bytes": 17179869184,
        "max_files": 10000,
        "n_read_threads": 1,
        "n_write_threads": 1
      }
    ],
    "demote_superseded_states": true,
    "mamba_state_slots_reference_tokens": 65536
  }
}
```

These are the values from the validation run: a 2 GiB CPU pool and an 8 GiB disk
budget, chosen so that test documents had to go to disk. For your own deployment,
use a larger CPU pool and set `max_size_bytes` to your disk budget.

- `max_size_bytes` is **required**. It is a positive integer: the total page budget
  for everything under `root_dir`. Configuration fails without it.
- `min_free_bytes` is the free-space floor of the file system. The default is 5% of
  its capacity.
- Use one dedicated directory for each engine. A lock file allows only one engine per
  root.
- To reuse disk pages after a restart, the model, ReVolta version and page layout must
  be the same. Keep `.hash_seed.json` in the root. Restart reuse in the release
  gate: passed (a restarted engine read 667 MB of pages back from disk; output identical).
- The SSD tier is a disposable cache, not model storage.
- A prefix restored from disk is staged in the CPU pool first, so `cpu_bytes_to_use` must
  be larger than the largest prefix you expect to restore from disk (about 21 KB per
  token; a 100K-token prefix needs more than 2 GiB). Otherwise the request is computed
  again. The quick-start 48 GiB pool holds about 2.4M tokens.
- The SSD tier is **not supported with INT8 KV** in 0.1.0. That combination is
  rejected at configuration time.

Validation scope: the SSD tier was tested with its root on a network file system. The
full release gate passed with an 8 GiB CPU pool and a 40 GiB disk budget, including a
99K-token prefix restored from disk (96,288 cached tokens, output identical). SSD
throughput was not measured.

## Weight and KV formats

V100 stores compressed weights and KV, then converts them to FP16 for computation.
The default is AWQ g32 with E4M3 KV.

| Format | GPU KV capacity, tokens | KL vs base BF16 | Top-1 agreement |
| --- | ---: | ---: | ---: |
| Uncensored AWQ g32 + E4M3 KV | 1,140,505 | 0.0273 | 96.63% |
| Standard NVFP4 + E4M3 KV | 1,232,435 | 0.0303 | 96.36% |
| NVIDIA FP4-Local + E4M3 KV | 1,295,637 | 0.0380 | 95.91% |
| Uncensored AWQ g32 + INT8 KV | 1,127,578 | 0.0267 | Within 0.1 percentage point of E4M3 |

These are candidate-build measurements on the reference machine, with 8 sequences
and a 512K maximum request length. All rows use the same base BF16 reference.
The Uncensored rows include the effect of fine-tuning as well as quantization.

### E4M3 KV

Use the calibrated per-layer K/V scales included in the checkpoint. Keep
`VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1` enabled to reject missing scales.
Without it, the loader warns and falls back to 1.0.

On the long-context quality set, E4M3 and FP16 KV measured KL 0.0238 and 0.0239
against BF16.

### INT8 KV

Set `--kv-cache-dtype int8` and `"kv_cache_dtype": "int8"` in the speculative
configuration. Target and draft KV types must match. Scales are computed per token
and head at runtime; no calibration is needed.

On the measured sets, INT8 has slightly lower KL and similar top-1 agreement to
E4M3. This has not established a difference in task-level answer quality.

Capacity is 1.1–1.3% lower than E4M3. Prefill speed is within ±1.3%; decoding is
about 5–8% slower. CPU offload works; the disk tier is not supported with INT8 in 0.1.0.

### Standard NVFP4

Tested revision: `7b719225` of
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4),
about 126 GiB. Add the 24 calibrated scales from
[the QSA E4M3 scale pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales)
to the model index.

Use `--quantization modelopt_mixed` and:

```bash
export VLLM_SM70_NVFP4_QWEN38_MOE_RAW_SCALE=1
export VLLM_SM70_NVFP4_QWEN38_MOE_RAW_GROUPED_DECODE=1
export VLLM_SM70_NVFP4_QWEN38_MOE_QPN_DYNAMIC_DECODE=1
export VLLM_SM70_NVFP4_MOE_GROUPED_DECODE=0
export VLLM_SM70_NVFP4_TURBOMIND=1 VLLM_SM70_FP8_TURBOMIND=1 VLLM_SM70_FP4_TURBOMIND=1
export VLLM_SM70_NVFP4_MOE_GROUPED_PREFILL=1 VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=1
export VLLM_V2_CUDAGRAPH_MEM_MIB=700
```

These switches are not yet in the automatic profile. Validation used the FP8 MTP
draft from the Uncensored AWQ repository, with 62.5% acceptance.

### NVIDIA FP4-Local

[NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
is about 95 GiB. ReVolta runs its experts as W4A16, leaves activations in FP16,
and expands MXFP8 layers to FP16.

The vision tower uses TP4 by default. ReVolta converts 27 affected vision `fc2`
weights to FP16 so that their dimensions can be sharded correctly. This configuration
passed the large-image and video stress tests.

The PLE table uses about 26.8 GiB of pinned host memory across four ranks.
Use the checkpoint's own MTP layer with `"kv_cache_dtype": "float16"` and no draft
`model` field. Measured acceptance was 65.1%.

Use `--quantization modelopt_mixed` and the NVFP4 environment above. The checkpoint has
no QSA E4M3 KV scales; download the calibrated 24-scale pack
[`leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales)
(tag `v1.0-base-b9ce18a6`) and build a checkpoint view with its `materialize_overlay.py`.

## Performance

Release configuration: Uncensored AWQ g32, E4M3 KV, TP4 + DCP2, MTP4, on the reference system.

| Metric | 0.1.0 |
| --- | ---: |
| GPU KV capacity, tokens | 1,140,505 |
| Total output throughput, C1 | 152.5 |
| Total output throughput, C2 | 167.2 |
| Total output throughput, C4 | 262.2 |
| Total output throughput, C8 | 383.6 |
| Cold prefill, 8K prompt, tok/s | 3,510 |
| Cold prefill, 131K prompt, tok/s | 3,020 |
| 211K-token retrieval (5 needles) | 5/5 |
| 512K boundary: 524,224 input + 64 output | Passed; one more input token is rejected |

**Method.** Each output test requests 512 tokens, with thinking off,
`temperature=0`, streaming, and MTP4 enabled. English code, Chinese explanation,
and English arithmetic prompts rotate across clients. Each concurrency level
has one warm-up and eight measured runs; the table reports the mean.

At C2, the two requests share the first prefill step only if the second one reaches
the engine within a sub-millisecond window. With the same code, this happened in
0 to 8 of the 8 measured repetitions, varying from run to run, and C2 ranged
from 159.7 to 173.7 tok/s; the table's 167.2 came from a run with 4 of 8.
Repetitions where both requests started together reached about 174 tok/s.

Total output throughput is all generated tokens divided by wall time, including
time to first token. It is an end-to-end rate, not a decode-only measurement.
Prefill uses a cold prefix cache and divides computed prompt tokens by time to
first token.

Version-to-version speed comparisons use the same machine, configuration, and test
window. Other context lengths, media workloads, and thinking-mode outputs can give
different results.

## Quality measurements

Quality uses teacher forcing, measuring `KL(quantized ‖ BF16)` and top-1 agreement.

- Short set: 66 English, Chinese, and code records; 39,101 tokens; contexts up to 4K.
- Long set: three natural documents—English 32K, Chinese 64K, and SQLite source
  128K—with 14,336 sampled positions.
- KL uses top-20 log-probabilities plus a tail bucket. In one checked quantization,
  this estimate was 7–10% below full-vocabulary KL.

These KL values are lower than published KLD figures for other quantizations of this
model mainly because only the routed experts are 4-bit: the PLE tables are FP8 and all
other weights stay 16-bit. For comparison, llama.cpp Q8_0 of this model measures about
0.02 against BF16 on English text. Only English, Chinese and code were measured.

The AWQ quality table near the top compares each checkpoint with its own BF16 parent.
The format table uses the base BF16 model throughout. Results from other datasets
or estimators should be compared separately.

## Support and known limits

**Full acceptance on the reference setup:** default Uncensored AWQ, Standard NVFP4,
and FP4-Local with their documented settings. Base AWQ has quality measurements.
80 GiB RAM has memory-stress results. Disk caching has readback results; its full
release gate passed with an 8 GiB CPU pool and a 40 GiB disk budget.

V100 16 GB, TP2/TP8, and other models have not been tested. NVLink has a community
test ([#2](https://github.com/Leonccaa/ReVolta/issues/2)) but is not part of our release gate.
This release builds SM70 only; other GPU architectures, ROCm, CPU, XPU, TPU, and
GGUF are outside the build.

- **Output reproducibility.** A single request is reproducible bit for bit. When several
  requests are prefilled together, the result can differ noticeably from the same request
  run alone (first-token log-probability differences of up to about one nat were
  measured), so greedy outputs at C2–C8 can change between runs. This is under
  investigation. Fresh compiler autotuning can also change outputs; restarts that reuse
  the compile cache do not.
- **Long context.** Release retrieval is tested at 211K. The previous production
  build retrieved 20/20 needles at 489K. At 512K, the test covers capacity and the
  input/output boundary.
- **Memory.** The default GPU utilization of 0.95 passed multimodal stress. Retest
  memory use when changing concurrency or media limits. The AWQ PLE table always
  needs about 48 GiB of host memory.
- **FP4-Local vision.** Keep the default vision-tower TP4. Forcing data parallelism
  removes the tested memory arrangement; request-level image settings can raise limits.
- **NVFP4 scales.** Different gate/up global scales are not supported by the
  raw-scale path. Both documented checkpoints use equal scales.
- **Structured output with MTP.** GPU tests produced 16/16 valid JSON/tool outputs;
  the zero-draft step was covered on CPU but was not triggered on GPU.
- **Stop strings and thinking.** As in upstream vLLM, a `stop` string also matches inside
  the reasoning. With thinking on, a request can end before the answer with empty
  `content`. Streaming responses with a `stop` string can also lose the last characters of
  the reasoning and show `</think>` in `content`. With thinking on, leave `stop` unset or
  turn thinking off. Non-streaming responses also split the reasoning at a literal
  `</think>` written as plain text. A fix is planned for 0.1.1.
- **Allocator.** Keep `NCCL_GRAPH_REGISTER=0` with expandable allocator segments.
- **Small KV pools.** Startup rejects a pool that cannot hold one maximum-length request.
- **Prompt log-probabilities.** Long prompts are processed in chunks;
  `prompt_logprobs=20` measured about 2.3× ordinary prefill time.
- **Standard NVFP4 draft.** Use the FP8 `mtp-fp8/` draft from the Uncensored AWQ
  repository, as in validation.
- **Media.** Default per-prompt limits are 16 images, 1 video, and 16 audio items,
  with download/decode budgets. See [multimodal input limits](.github/MULTIMODAL_INPUTS.md).

There is no GPU CI. Releases are checked on the reference machine.

## Reporting issues

Use [GitHub Issues](https://github.com/Leonccaa/ReVolta/issues). Include the launch
command, relevant environment, model revision, host/container memory limits,
offload settings, `nvidia-smi topo -m`, and the `ReVolta profile` /
`SM70 interconnect=` log lines. Remove credentials.

Reports from NVLink V100 owners are especially welcome; thanks to
[@jttdev](https://github.com/jttdev) for the first one ([#2](https://github.com/Leonccaa/ReVolta/issues/2)). Report security issues
through GitHub private vulnerability reporting.

## Building from source

See [BUILDING.md](BUILDING.md). Builds use Docker BuildKit, CUDA 12.8, and SM70 only.
No GPU is needed to compile. Debug symbols are published separately.
CUDA 13 cannot compile SM70.

## License and acknowledgements

ReVolta's engine code and project changes are licensed under the
[Apache License 2.0](LICENSE), except for third-party files that state another
license. [NOTICE](NOTICE) lists component attributions and known source-history gaps.
[LICENSES/](LICENSES/README.md) contains the upstream license texts with their source
revisions. Model weights have their own licenses. See each model card.

ReVolta is derived from [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), which is
derived from [vLLM](https://github.com/vllm-project/vllm). Thanks to the vLLM and
1Cat-vLLM contributors, and to the authors of the attention, quantization,
communication and GPU kernel components included here (see NOTICE).

NVIDIA, Volta and Tesla are trademarks of NVIDIA Corporation and are used here only
to identify hardware. ReVolta is not affiliated with or endorsed by NVIDIA.
