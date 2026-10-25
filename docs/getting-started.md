# Getting started

This tutorial installs and launches the reference configuration. Before starting, check the
[hardware scope](known-limits.md) and [host memory requirements](configuration.md#system-memory-and-storage).

Reference configuration: Uncensored AWQ g32, TP4 + DCP2,
MTP4 greedy draft, E4M3 KV cache, 512K context through YaRN, multimodal input, 8
concurrent requests in testing, and a 48 GiB CPU KV pool.

## 1. Install

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

## 2. Download the checkpoint and make 512K views

The checkpoint's native context is 262,144 tokens. ReVolta reaches 524,288 tokens with
YaRN (factor 2). The target model **and** the MTP draft must both see the YaRN
settings. `--hf-overrides` reaches only the target model, so make two "view"
directories instead. Each view links to every checkpoint file and contains an edited
`config.json`. The validated setup used views of this kind.

For the static YaRN trade-off and the native 262K alternative, see
[YaRN and native context](configuration.md#yarn-and-native-context).

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

repo = "leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32"
ckpt = Path("/data/revolta/models/Qwen3.8-Flash-Next-Uncensored-AWQ-g32")
views = Path("/data/revolta/views")
snapshot_download(repo, local_dir=ckpt, revision="1fd7ab705731c9dacc3bcd5b5d15783c60706d64")

YARN = {"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 262144}

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

make_view(ckpt, views / "target-512k")
make_view(ckpt / "mtp-fp8", views / "draft-512k")
EOF
```

## 3. Environment

Apply the reference deployment settings below; see
[Environment settings](configuration.md#environment-settings) for profile behaviour.

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

Continue with the launch command below.

## 4. Launch

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

See [Launch options](configuration.md#launch-options) before changing these settings.

## 5. Check

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

For request defaults and the model vendor's recommendations, see
[Sampling parameters](configuration.md#sampling-parameters).

## Docker

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

## See also

- [Prepare another checkpoint](checkpoints.md)
- [Configuration reference](configuration.md)
- [KV offload modes](kv-offload.md)
