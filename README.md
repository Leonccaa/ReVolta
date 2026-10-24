# ReVolta

**A daily AI server on 4 × V100 32 GB.**

Run Qwen3.8 Flash-Next for text, images, and video, with an OpenAI-compatible API.

> [!TIP]
> **New in 0.1.2: DFlash2 speculative decoding on V100.** ReVolta runs the
> [incoai DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) on
> Volta (SM70) GPUs, with V100 attention kernels and CUDA graphs for the drafter and the
> verifier, in the 512K-context setup with prefix caching and CPU KV offload. On one
> request it writes English code at **201 tokens/s instead of 163** with MTP4 (1.23×), and
> the GPU KV pool grows by about 10% to 1.27 million tokens. Our own server runs it. MTP4
> stays the default: Chinese text and eight concurrent requests are still faster with MTP4.
> [How to enable it](#dflash2-speculative-decoding-optional).

- **GPU-resident context:** a KV pool of up to about **1.3 million tokens** with 8-bit KV
  (1.27 million with the default AWQ checkpoint and DFlash2), room for two 512K or four
  256K contexts; see [Weight and KV formats](#weight-and-kv-formats).
- **Weight formats:** standard AWQ, NVFP4, and NVIDIA FP4-Local.
- **KV formats:** FP8 E4M3, FP16, and INT8.
- **Speculative decoding:** native MTP4 by default, or DFlash2 with the incoai drafter
  (new in 0.1.2, up to 1.23× on one request).
- **Speed:** up to **525 tokens/s total output at 8 concurrent requests** on NVLink
  ([community result](https://github.com/Leonccaa/ReVolta/issues/2), 0.1.1), and up to 505 on our PCIe reference (code prompts, DFlash2); see [Speed](#speed).
- **Cache reuse:** CPU memory and disk offload; a cached 32K document answered in **1.28 s vs 10.06 s** when recomputed.

## Measurements

### Speed

Best measured output speed in tokens/s, 512-token responses. The breakdown by speculative
method (MTP4 and DFlash2) and prompt type is in [Performance](#performance).

| System | 1 request | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| 4 × V100 32 GB PCIe, one PCIe switch, P2P, no NVLink (our reference) | **201** | **275** | **262** | **505** |
| 4 × V100-SXM2 32 GB, NV2 NVLink mesh ([community result](https://github.com/Leonccaa/ReVolta/issues/2), 0.1.1) | **191** | **231** | **338** | **525** |

| Cold prompt | 8K tokens | 131K tokens |
| --- | ---: | ---: |
| Prefill throughput, tok/s | 3,510 | 3,020 |

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

The distribution name is `revolta` (version `0.1.2`). The Python import name and the
CLI are still **`vllm`** (`import vllm`, `vllm serve`); this is intentional. Only
compute capability 7.0 is built, with CUDA 12.8, Python 3.12 and PyTorch 2.10.

From PyPI (Linux x86-64 with glibc 2.34 or newer, Python 3.12):

```bash
python3.12 -m venv /data/revolta/venv
. /data/revolta/venv/bin/activate
pip install revolta==0.1.2
vllm --version          # 0.1.2
```

The wheel pins `torch==2.10.0`, `torchaudio==2.10.0` and `torchvision==0.25.0`,
and pip installs them from PyPI. PyPI's `torch==2.10.0` is the CUDA 12.8 build.
Install the NVIDIA driver for CUDA 12.8 on the host.

The wheel attached to the GitHub release contains the same files. Only its metadata
differs: it pins the same versions to the CUDA 12.8 builds on `download.pytorch.org`
and has the generic `linux_x86_64` tag.

```bash
pip install "https://github.com/Leonccaa/ReVolta/releases/download/v0.1.2/revolta-0.1.2-cp312-cp312-linux_x86_64.whl"
```

ReVolta does not send the anonymous usage statistics that upstream vLLM reports.

Or use the Docker image:

```bash
docker pull ghcr.io/leonccaa/revolta:0.1.2
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
export TILELANG_CACHE_DIR=/data/revolta/cache/tilelang  # persistent TileLang kernel cache (needed in containers)
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
  --limit-mm-per-prompt '{"image": 16, "video": 1}' \
  --media-io-kwargs '{"video": {"num_frames": 16}}' \
  --compilation-config '{"mode": 3, "cudagraph_mode": "FULL_AND_PIECEWISE"}' \
  --structured-outputs-config '{"backend": "xgrammar"}' \
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
  --sse-keep-alive-interval 15 \
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
- **Keep `--limit-mm-per-prompt` on V100.** With FlashAttention-V100, the engine allows
  1 image and no video per request without it, and a value that leaves out `"video"`
  disables video. This V100 default is kept in 0.1.2. `{"image": 16, "video": 1}`
  keeps the reference deployment's limits; Qwen3.8 Flash-Next takes images and video,
  not audio, so no audio limit is set. To use the generic image count instead, set
  `--limit-mm-per-prompt '{"image": 8, "video": 1}'`; a lower image count leaves more
  host memory for bursts of large images. Large images are resized
  either way (see [Support and known limits](#support-and-known-limits)).
- `--structured-outputs-config '{"backend": "xgrammar"}'` pins the structured-output
  backend. With `auto`, the engine keeps the backend chosen for its first
  structured-output request.
- `--sse-keep-alive-interval 15` sends an SSE comment line (`: keep-alive`) after 15 idle
  seconds on streaming responses, so proxies with read timeouts keep the connection open
  during queueing and prefill. The default, 0, sends none.
- `TILELANG_CACHE_DIR` (step 3) keeps TileLang's compiled kernels on persistent storage.
  Without it, a new container compiles them again.
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
  -e TILELANG_CACHE_DIR=/data/revolta/cache/tilelang \
  ghcr.io/leonccaa/revolta:0.1.2 \
  /data/revolta/views/target-512k --served-model-name Qwen3.8-Flash-Next ...
```

Replace `...` with the remaining arguments from step 4. Container notes:

- The CPU KV pool is a file in `/dev/shm`. Make `--shm-size` larger than
  `cpu_bytes_to_use` (the validated container used 128g).
  After an abnormal service exit, recreate the container to release leftover
  CPU offload files in its private `/dev/shm`.
- `--cap-add SYS_PTRACE` was required in the validated container: the workers share
  memory with each other through `pidfd_getfd`.
  Whether `seccomp=unconfined` is needed remains to be checked.
- Do not bake `VLLM_SM70_*` switches into derived images. Values from the environment
  override the profile.

## DFlash2 speculative decoding (optional)

MTP4 is the default and the tested release configuration. Since 0.1.2, ReVolta can use
the [incoai DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2)
instead (tested revision `a22a1744`): 5 sliding-window layers, an 8-token block and a
2048-token window. It is off by default. Without a `dflash` speculative config, the KV
cache layout, block size and MoE routes are the same as in 0.1.1.

**1. Make a draft view.** The drafter must declare how it reads the target's
hyper-connection streams, and its position limit must cover the 512K target. The view
links the weights and adds both settings to `config.json`. Startup refuses a drafter
without `"target_feature_reduction": "hc_mean"`.

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

src = Path("/data/revolta/models/Qwen3.8-Flash-Next-DFlash2")
dst = Path("/data/revolta/views/dflash2-512k")
snapshot_download("incoai/Qwen3.8-Flash-Next-DFlash2", local_dir=src,
                  revision="a22a1744e85c17364dcc54a6a420af7135d4c7f6")

dst.mkdir(parents=True, exist_ok=True)
for item in src.iterdir():
    if item.is_file() and item.name != "config.json":
        link = dst / item.name
        if not link.exists():
            link.symlink_to(item.resolve())
config = json.loads((src / "config.json").read_text())
config["dflash_config"]["target_feature_reduction"] = "hc_mean"
config["max_position_embeddings"] = 524288   # the drafter's layers keep a 2048-token window
(dst / "config.json").write_text(json.dumps(config, indent=2) + "\n")
EOF
```

**2. Set two more environment variables and replace the MTP speculative config** in step 4.
Keep every other argument, including `--no-async-scheduling`. With Docker, pass both with
`-e VLLM_SM70_DFLASH2_FUSED_QKV_PACK=1 -e VLLM_V2_CUDAGRAPH_MEM_MIB=1500`.

```bash
export VLLM_SM70_DFLASH2_FUSED_QKV_PACK=1
export VLLM_V2_CUDAGRAPH_MEM_MIB=1500    # CUDA graph reserve for DFlash2 (the profile's 700 fits MTP4 only)

  --speculative-config '{"method": "dflash",
      "model": "/data/revolta/views/dflash2-512k",
      "num_speculative_tokens": 7,
      "attention_backend": "FLASH_ATTN_V100",
      "draft_sample_method": "greedy",
      "dflash_concurrency_drafts": {"4": 3, "5": 4},
      "sm70_dflash2": {"context_kv_graph": true, "context_pipeline": true,
        "verify_fastpath": true, "fused_gdn_metadata": true,
        "fused_smallq_metadata": true, "grouped_smallq_metadata": true}}' \
```

In the startup log, the AWQ MoE line should read `grouped QPN decode admitted for 2..48
rows (split=8, pack_rows=24, up to 2 row chunks)`.

Settings:

- **Enable the six `sm70_dflash2` options above.** They are off unless set.
- **Do not enable** `fused_gdn_verify`, `fused_gdn_norm`, `fused_gdn_split` or
  `fused_gdn_combined_split`. They change the numerics and lowered sampled acceptance
  in our tests.
- **Leave `--block-size` unset.** The automatic block size fits the drafter cache into the
  main KV slots (1664 tokens with 7 drafts at DCP2). An explicit block can leave the
  drafter only small blocks; startup logs a warning then.
- **Do not set `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_MAX_TOKENS`.** With DFlash, the AWQ MoE
  layers use `VLLM_SM70_AWQ_QWEN38_GROUPED_QPN_DFLASH_MAX_TOKENS` (default 48; no need to
  set it). The older variable changes only MTP and non-speculative routes.
- **`VLLM_SM70_DFLASH2_FUSED_QKV_PACK=1`** packs the verifier's Q/K/V rows with one copy
  kernel. It acts only on SM70 with DFlash2. In tests, greedy output was bit-identical and
  throughput about 1.5% higher. Like every `VLLM_*` variable, it is part of the
  torch.compile key, so a deployment that sets it builds its own compile cache.
- **`VLLM_V2_CUDAGRAPH_MEM_MIB=1500` is required with DFlash2.** It reserves GPU memory
  for the CUDA graphs before the KV pool is sized. The ReVolta profile sets 700 MiB, which
  fits MTP4's graphs (0.68 GiB per GPU); DFlash2's graphs take about 1.37–1.39 GiB. Without
  the setting, the KV pool takes about 0.7 GiB per GPU too much. Concurrent long prompts
  that fill a mixed prefill step (up to the 8192-token batch budget), with or without
  decoding, can then run out of GPU memory. An explicit value overrides the profile. Alternative: `--gpu-memory-utilization 0.93`.
- **`dflash_concurrency_drafts`** maps a number of running requests (the threshold) to the
  number of drafts the target verifies per request (the width). The drafter still drafts
  7 tokens. With `{"4": 3, "5": 4}`, one to three running requests verify all 7 drafts,
  four verify 3, and five or more verify 4.
- The thresholds form a ladder: the entry with the largest threshold that is not above the
  number of running requests sets the width. Below the lowest threshold, all drafts are
  verified, and a higher entry can widen the width again. CUDA graphs are captured only
  for the request counts that use each width. Thresholds must be 2 or more, and widths 1
  to 7.
- Against `{"5": 4}`, this ladder gave identical output at 1 request and 9.7% more
  throughput at 4 requests (English, Chinese and code weighted equally); at 5–8 requests
  both use the same width.
- For mostly Chinese loads, `{"3": 3}` verifies 3 drafts from three requests on; at three
  requests that was about 15% faster in an earlier width sweep. For mostly code, `{"8": 4}`
  measured best at 8 requests.
- Setting `dflash_concurrency_drafts` turns off async scheduling; `--async-scheduling`
  together with it is an error. Pipeline parallelism is not supported with DFlash2.

What to expect:

- Greedy output is not bitwise identical to MTP4. It differs at near-ties, because a
  different number of tokens is verified and DFlash uses a different MoE route for some
  batch sizes.
- English code and reasoning gain the most. Chinese is slower than MTP4 in almost every
  measured case: the drafter was trained on English. Very long prompts also favour MTP4.
- The GPU KV pool is about 10% larger than with MTP4: about 1.27 million tokens with the
  graph reserve (1,268,831 measured), against 1,153,143. Without the
  reserve it reached 1,445,845 tokens, but that setting can run out of memory.
- On a replay of real agent sessions (one request at a time, sampling, long reasoning),
  DFlash2 and MTP4 were on par: 0.97× the MTP4 speed (95% interval 0.90–1.01). DFlash2
  was ahead on tool calls, answers and contexts up to 64K, and behind in reasoning, above
  128K and on Chinese. MTP4 stays the default.
- The first request after an empty compile cache compiles the DFlash verify kernels.
- Measured on the reference setup with the recommended config (greedy, tokens/s, DFlash2 vs
  MTP4): one request, agent code 181 / 158, English code 201 / 163, English reasoning
  195 / 163, Chinese explanation 101 / 105, Chinese chat 85 / 100; four mixed requests
  238 / 236; eight mixed requests 331 / 357 (0.93×).

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
- The SSD tier is **not supported with INT8 KV** in 0.1.2. That combination is
  rejected at configuration time.

Validation scope: the SSD tier was tested with its root on a network file system. The
full release gate passed with an 8 GiB CPU pool and a 40 GiB disk budget, including a
99K-token prefix restored from disk (96,288 cached tokens, output identical). SSD
throughput was not measured.

## Weight and KV formats

V100 stores compressed weights and KV, then converts them to FP16 for computation.
The default is AWQ g32 with E4M3 KV.

| Format | GPU KV capacity with MTP4, tokens | GPU KV capacity with DFlash2, tokens | KL vs base BF16 | Top-1 agreement |
| --- | ---: | ---: | ---: | ---: |
| Uncensored AWQ g32 + E4M3 KV | 1,140,505 | 1,268,831 | 0.0273 | 96.63% |
| Standard NVFP4 + E4M3 KV | 1,232,435 | | 0.0303 | 96.36% |
| NVIDIA FP4-Local + E4M3 KV | 1,295,637 | | 0.0380 | 95.91% |
| Uncensored AWQ g32 + INT8 KV | 1,127,578 | | 0.0267 | Within 0.1 percentage point of E4M3 |
| Uncensored AWQ g32 + FP16 KV | | | Within 0.001 of E4M3 | Same as E4M3 |

Measured on the reference machine, with 8 sequences and a 512K maximum request length.
MTP4: 0.1.0 candidate builds (0.1.2 with MTP4 has 1,153,143 tokens in the first row).
DFlash2: the 0.1.2 release build with the recommended DFlash2 config, including
`VLLM_V2_CUDAGRAPH_MEM_MIB=1500`. Empty cells are not measured yet. All KL and top-1
values use the same base BF16 reference; the FP16 KV row compares FP16 with E4M3 KV
directly (on short prompts: KL 0.0220 against 0.0214, top-1 97.08% for both; on the
long-context set: 0.0239 against 0.0238). The Uncensored rows include the effect of
fine-tuning as well as quantization.

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
about 5–8% slower. CPU offload works; the disk tier is not supported with INT8 in 0.1.2.

### Standard NVFP4

Tested revision: `7b719225` of
[RadixArk/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4),
about 126 GiB. Add the 24 calibrated scales from
[the QSA E4M3 scale pack](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-NVFP4-QSA-FP8-E4M3-KV-Scales)
to the model index.

Use `--quantization modelopt_mixed`. On the FP16 QSA SM70 TP4 profile,
ReVolta detects NVFP4 weights from checkpoint quantization metadata and sets
the validated kernel switches and a 700 MiB CUDA graph reserve automatically.
Explicit environment settings take precedence. The startup profile audit logs
each value and its source (`profile`, `environment`, or `upstream_default`).
Validation used the FP8 MTP draft from the Uncensored AWQ repository, with
62.5% acceptance.

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

Use `--quantization modelopt_mixed`; the same automatic NVFP4 profile applies. The checkpoint has
no QSA E4M3 KV scales; download the calibrated 24-scale pack
[`leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales)
(tag `v1.0-base-b9ce18a6`) and build a checkpoint view with its `materialize_overlay.py`.

## Performance

### Reference system, MTP4

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

### DFlash2 against MTP4

Reference system, recommended DFlash2 config (new in 0.1.2), greedy, thinking off,
512-token responses. One request: decode speed after the first token. Four and eight
mixed requests: total output. With one prompt per category, the per-category values are
indicative.

| Requests and prompt | DFlash2, tok/s | MTP4, tok/s |
| --- | ---: | ---: |
| 1, English code | 201 | 163 |
| 1, English reasoning | 195 | 163 |
| 1, agent code | 181 | 158 |
| 1, Chinese explanation | 101 | 105 |
| 1, Chinese chat | 85 | 100 |
| 4, mixed | 238 | 236 |
| 8, mixed | 331 | 357 |

By prompt category, 8 prompts per category, aggregate output; measured with the 0.1.2
code before the final width ladder, with the same verify widths at 2 and 8 requests:

| Requests | DFlash2: code / English / Chinese | MTP4: code / English / Chinese |
| --- | ---: | ---: |
| 2 | 275 / 203 / 129 | 232 / 193 / 141 |
| 8 | 505 / 336 / 285 | 492 / 332 / 327 |

English code and reasoning gain the most; Chinese text is slower with DFlash2. On a
replay of real agent sessions (one request at a time, sampling), the two were on par
(0.97× the MTP4 speed). See [DFlash2 speculative decoding](#dflash2-speculative-decoding-optional).

### NVLink (community result)

[@jttdev](https://github.com/jttdev) measured 0.1.1 on 4 × V100-SXM2 32 GB with an NV2
NVLink mesh ([#2](https://github.com/Leonccaa/ReVolta/issues/2)): MTP4, the same three
prompts, base AWQ checkpoint, 512K views, `--gpu-memory-utilization 0.95` and a 16 GiB CPU
KV pool (`cpu_bytes_to_use`). Each level runs 3 requests per stream.

| | 1 request: code / Chinese / reasoning (decode) | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| 0.1.1 | 178 / 123 / 191 | 231.3 | 329.5 | 524.6 |
| 0.1.0, same session | 179 / 123 / 191 | 231.0 | 329.0 | 519.2 |

- The greedy outputs of both releases were byte-identical.
- At 262K context, `--gpu-memory-utilization 0.925` and a 32 GiB CPU pool, four requests
  reached 338 without the prefill-pressure flags (333 with them).
- The CPU KV pool size affects eight requests: in that 262K setup, eight requests gave
  513 tokens/s with a 16 GiB pool and 497 with 32 GiB, about 3% less; one to four
  requests did not change. An earlier 8-request gap against a hand-built 1Cat-vLLM setup
  did not reproduce in later runs.
- The GPU KV pool at 0.95 and 512K is 1,095,050 tokens with 0.1.1 (1,100,286 with 0.1.0);
  0.1.1 creates the NCCL communicators before memory profiling, which probably explains
  the 0.5% difference.
- Two concurrent retrieval requests of about 490K tokens each found all three needles
  (0.1.0); 0.1.1 found 3 of 3 at 91K and 244K tokens.

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
  requests are prefilled together, the result can differ from the same request run alone,
  so greedy outputs at C2–C8 can change between runs. The cause is the tensor-parallel
  all-reduce: the order in which it sums each value depends on where the request's
  tokens sit in the batch. The rounding-level difference can then be amplified where
  expert routing or sparse-attention block selection is close to a tie. On most text
  the prompt NLL changes by less than 0.015 nat/token; highly repetitive or table-like
  text changed by up to about 0.3. Against a BF16 reference, a request run alone and the
  same request in a batch were equally close. Fresh compiler autotuning can also change
  outputs; restarts that reuse the compile cache do not.
- **Long context.** Release retrieval is tested at 211K. The previous production
  build retrieved 20/20 needles at 489K. At 512K, the test covers capacity and the
  input/output boundary.
- **Memory.** The default GPU utilization of 0.95 passed multimodal stress. Retest
  memory use when changing concurrency or media limits. The AWQ PLE table always
  needs about 48 GiB of host memory.
- **PLE cascade.** Profiling with `ple_disk_cascade=true` used about 4 GiB more
  GPU memory per 32 GB card. This left room for about 345K context tokens.
  Cascade is unsupported in 0.1.x. Keep it disabled.
- **FP4-Local vision.** Keep the default vision-tower TP4. Forcing data parallelism
  removes the tested memory arrangement. Non-empty request-level
  `mm_processor_kwargs` and `media_io_kwargs` are rejected with HTTP 400 by
  default. Only enable `--trust-request-mm-kwargs` for trusted API clients;
  their image settings can then raise preprocessing resource usage.
- **NVFP4 scales.** Different gate/up global scales are not supported by the
  raw-scale path. Both documented checkpoints use equal scales.
- **Structured output with MTP.** GPU tests produced 16/16 valid JSON/tool outputs. A
  later GPU test also reached the zero-draft step, where the grammar leaves no draft
  tokens (64 such steps): all outputs were valid, and repeated runs gave identical output.
- **Stop strings and thinking.** Since 0.1.1, with thinking on, a `stop` string matches
  only after the reasoning ends, and the reasoning/content split uses the `</think>`
  token, not its spelling. A `stop` of `</think>` therefore no longer ends a request at
  the end of the reasoning. With thinking off, everything is content, also in
  non-streaming responses.
- **Tool calls.** In `qwen3_coder` tool calls, a string argument that contains the
  literal text `</function>` is cut at that point. String arguments are streamed as
  they are generated.
- **Image placeholder text.** Text that contains the literal image placeholder
  `<|image_pad|>` can take the place of a real image in the same request. Upstream vLLM
  and HF behave the same (vLLM #57740), and ReVolta follows them. Remove such tokens from
  untrusted text.
- **DFlash2.** Optional and off by default; see
  [DFlash2 speculative decoding](#dflash2-speculative-decoding-optional) for its settings
  and limits. Set `VLLM_V2_CUDAGRAPH_MEM_MIB=1500` with it. The profile's CUDA graph reserve
  is sized for MTP4. Without the larger reserve, concurrent long prompts that fill a mixed
  prefill step (up to the 8192-token batch budget), with or without decoding, ran out of
  GPU memory on the reference setup. Sizing the reserve per
  speculative method is planned for the next release.
- **Chat template.** The model's own chat template rejects a system message that is not
  the first message (`System message must be at the beginning.`). Agent stacks that send
  several system messages need to merge them.
- **Allocator.** Keep `NCCL_GRAPH_REGISTER=0` with expandable allocator segments.
- **Sleep level 2.** Not supported on the reference setup (128 GiB host, PLE tables in host
  memory): it needs about 87 GiB of free host memory for the model buffers while about
  63 GiB is available. Sleep level 1 works.
- **Small KV pools.** Startup rejects a pool that cannot hold one maximum-length request.
- **Prompt log-probabilities.** Long prompts are processed in chunks;
  `prompt_logprobs=20` measured about 2.3× ordinary prefill time.
- **Standard NVFP4 draft.** Use the FP8 `mtp-fp8/` draft from the Uncensored AWQ
  repository, as in validation.
- **Media.** On V100 with FlashAttention-V100, the default without `--limit-mm-per-prompt`
  is 1 image and no video, and a value that leaves out `"video"` disables video. This
  behaviour is kept. Set the option explicitly and include `"video"`: for example
  `'{"image": 8, "video": 1}'` (the generic image count) or
  `'{"image": 16, "video": 1}'` (the launch command above). The model has no audio input. On other hardware
  and backends, the defaults are 8 images, 1 video, and 1 audio item per whole request. Images in historical chat messages count
  toward the image limit. Qwen4Exp
  resizes large images to at most 1,048,576 pixels using HF's existing resize
  logic. Small text can become blurry; alignment near the area boundary can also
  change images whose source area is slightly below the limit.
  An empty `--mm-processor-kwargs '{}'`
  selects this default; restore the checkpoint image budget with
  `--mm-processor-kwargs '{"images_kwargs": {"size": {"longest_edge": 16777216, "shortest_edge": 65536}}}'`.
  Video sampling defaults to 16 frames. Download/decode ceilings remain separate.
  See [multimodal input limits](.github/MULTIMODAL_INPUTS.md).

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
