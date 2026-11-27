# Getting started

This tutorial installs ReVolta 0.2.0, prepares the default checkpoint and launches the reference configuration, then checks that the server answers.

You need:

- Four V100 32 GB GPUs, connected by NVLink or under one PCIe switch with P2P enabled. See [Support and known limits](known-limits.md) for the hardware scope.
- 128 GiB of host memory for this configuration. See [System memory and storage](configuration.md#system-memory-and-storage).
- An NVIDIA driver for CUDA 12.8.
- For a pip install: Linux x86-64 with glibc 2.34 or newer, Python 3.12 and the CUDA 12.8 Toolkit. For Docker: Docker with the NVIDIA Container Toolkit.
- About 142 GB of disk space for the checkpoint and its MTP draft, plus the compile caches.

The reference configuration is Uncensored AWQ g32 with tensor parallelism of four and decode context parallelism of two (TP4 + DCP2), an MTP draft with up to four draft tokens per round, INT8 KV cache for the target and the draft, a 524,288-token (512K) context through YaRN, image and video input, up to 8 concurrent requests and a 48 GiB CPU KV pool.

## 1. Install

The distribution is `revolta` (version `0.2.0`). The Python import name and the CLI stay **`vllm`** (`import vllm`, `vllm serve`). The build contains only compute capability 7.0 (V100), with CUDA 12.8, Python 3.12 and PyTorch 2.10.

From PyPI:

```bash
python3.12 -m venv /data/revolta/venv
. /data/revolta/venv/bin/activate
pip install revolta==0.2.0
vllm --version          # 0.2.0
```

The wheel pins `torch==2.10.0` (the CUDA 12.8 build on PyPI) with the matching `torchaudio` and `torchvision`. The first start compiles kernels, so a pip install needs the CUDA 12.8 Toolkit on the serving host: check that `nvcc --version` reports CUDA 12.8. The wheel attached to the [GitHub release](https://github.com/Leonccaa/ReVolta/releases/tag/v0.2.0) has the same files and installs the same PyTorch versions from `download.pytorch.org`.

Or use the Docker image, which includes the compiler tools for the first start:

```bash
docker pull ghcr.io/leonccaa/revolta:0.2.0
```

ReVolta does not report to the upstream vLLM statistics server. It sends its own anonymous usage report; see [Anonymous usage statistics](usage-stats.md) for what it contains and how to turn it off.

## 2. Download the checkpoint and make 512K views

The checkpoint's native context is 262,144 tokens. ReVolta reaches 524,288 tokens with YaRN (factor 2). The script below gives the target model and the MTP draft each a "view" directory with the YaRN settings: the view links to every file of the checkpoint (or of its `mtp-fp8/` draft) and contains an edited `config.json`. The validated setup used views of this kind. (`--hf-overrides` reaches only the target; the engine also copies the target's YaRN scaling to an MTP draft that ships native RoPE with the same `rope_theta`, but the release tests used the two views.)

Run the script in the ReVolta virtual environment, which already has `huggingface_hub`. With Docker only, install `huggingface_hub` on the host first (`pip install huggingface_hub`).

```bash
python - <<'EOF'
import json
from pathlib import Path
from huggingface_hub import snapshot_download

repo = "leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32"
revision = "1fd7ab705731c9dacc3bcd5b5d15783c60706d64"
ckpt = Path("/data/revolta/models") / repo.split("/")[-1]
views = Path("/data/revolta/views")
snapshot_download(repo, local_dir=ckpt, revision=revision)

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

Keep the downloaded checkpoint: the views link to its files. YaRN in vLLM scales every request, short or long. If prompt and output together stay within 262,144 tokens, you can skip the views; see [YaRN and native context](configuration.md#yarn-and-native-context). For another checkpoint, change `repo`, `revision`, `views` and the draft line as described in [Prepare the checkpoint](checkpoints.md#prepare-the-checkpoint).

## 3. Environment

The engine detects V100 (SM70), TP4, a Qwen3.8 Flash-Next model and FP16, and then applies its tuned defaults by itself (the "ReVolta profile"). The first four exports below repeat values the profile already sets; they are written out because the tested deployment used them. The last two are deployment settings that the profile does not set.

```bash
# Also applied by the ReVolta profile when unset.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_GRAPH_REGISTER=0                     # set automatically with expandable segments
export VLLM_SM70_AWQ_MOE_INT4_ZERO=1             # pack AWQ zero points to 4 bits in GPU memory
# Deployment settings.
export VLLM_PLE_OFFLOAD_READY_TIMEOUT=1200       # seconds (default 600); PLE loading takes a while
export TILELANG_CACHE_DIR=/data/revolta/cache/tilelang  # keep compiled TileLang kernels across restarts
# Optional per-rank cap for the PLE table in pinned host memory. By default the
# whole FP8 table is pinned on the host.
# export VLLM_QWEN4EXP_PLE_HOST_GIB=12
```

Explicit values always override the profile. See [Environment settings](configuration.md#environment-settings) for what the profile changes.

## 4. Launch

Use the 512K target view and the matching draft view from step 2 with `--max-model-len 524288`. With unmodified checkpoints, use a length that both the target and the draft support. With `--gpu-memory-utilization 0.956`, keep the GPUs free of other processes that use GPU memory.

```bash
vllm serve /data/revolta/views/target-512k \
  --served-model-name Qwen3.8-Flash-Next \
  --dtype half \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --gpu-memory-utilization 0.956 \
  --max-model-len 524288 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --prefix-cache-retention-interval 0 \
  --mamba-cache-mode align \
  --no-async-scheduling \
  --kv-cache-dtype auto \
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
      "attention_backend": "FLASH_ATTN_V100"}' \
  --attention-backend FLASH_ATTN_V100 \
  --prefill-pressure-token-budget 1024 \
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
  --shutdown-timeout 30 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --sse-keep-alive-interval 15 \
  --host 0.0.0.0 --port 8000
```

Notes:

- Many of these options repeat values that the ReVolta profile or the engine already chooses, for example the scheduler limits, prefix caching, the attention backend and the compilation mode. They are written out so that the command matches the tested configuration. See [ReVolta serving defaults](configuration.md#revolta-serving-defaults) before changing them.
- **KV cache.** `--kv-cache-dtype auto` selects planar INT8 for this model on V100. INT8 computes its scales at runtime, so no KV scale pack is needed. The MTP draft has no `kv_cache_dtype` of its own and inherits the target's INT8. To keep the draft KV in FP16 instead, add `"kv_cache_dtype": "float16"` to `--speculative-config`. For other KV formats, see [Choose the KV format](checkpoints.md#weight-and-kv-formats).
- **Draft width.** The profile lowers the number of draft tokens when more requests run; see [Dynamic speculative decoding](dynamic-speculative-decoding.md). To compare MTP with DFlash2, see [Speculative decoding](speculative-decoding.md).
- **CPU KV pool.** `cpu_bytes_to_use` is the total for all four workers: 48 GiB, 12 GiB per worker. See [KV offload](kv-offload.md) for SSD tiers and other pool sizes.
- **Wide verifier.** On a system whose GPUs share one PCIe switch with P2P, the profile also turns on the [AWQ MTP4 wide verifier](configuration.md#awq-mtp4-wide-verifier) for this configuration; the CPU KV pool does not prevent it. NVLink systems keep the standard verifier.
- **Startup** takes roughly 10–15 minutes: weight and PLE loading, `torch.compile` and CUDA graph capture. <!-- TODO(0.2.0-acceptance): cold and warm startup time, D+E1 --> Later starts reuse the compile cache. The first start after an upgrade recompiles.

See [Launch options](configuration.md#launch-options) for the meaning of each option.

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

- `ReVolta profile {"enabled": true, ...}`: the profile is active. The line lists the audited settings with their values and sources (`profile`, `environment` or `upstream_default`); some settings the profile applies, such as the wide verifier switches, are not in it.
- `Defaulting QSA main KV storage to int8`: the target KV cache is INT8.
- `SM70 interconnect=pex_p2p` (or `no_p2p` / `nvlink`) from each worker.

The server also provides the Anthropic-compatible `/v1/messages` (including streaming) and `/v1/messages/count_tokens`; their limits are in [API behaviour](known-limits.md#api-behaviour).

For request defaults and the model vendor's sampling recommendations, see [Sampling parameters](configuration.md#sampling-parameters).

## Docker

The image's entry point is `vllm serve`, so the server options follow the image name. The command below passes the same environment and options as steps 3 and 4. Mount the model and view directories at the same path inside the container so that the view symlinks resolve.

```bash
docker run --rm --gpus all \
  --shm-size 128g \
  --stop-timeout 120 \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -p 8000:8000 \
  -v /data/revolta:/data/revolta \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e NCCL_GRAPH_REGISTER=0 \
  -e VLLM_SM70_AWQ_MOE_INT4_ZERO=1 \
  -e VLLM_PLE_OFFLOAD_READY_TIMEOUT=1200 \
  -e TILELANG_CACHE_DIR=/data/revolta/cache/tilelang \
  -e VLLM_CACHE_ROOT=/data/revolta/cache/vllm \
  ghcr.io/leonccaa/revolta:0.2.0 \
  /data/revolta/views/target-512k \
  --served-model-name Qwen3.8-Flash-Next \
  --dtype half \
  --tensor-parallel-size 4 \
  --decode-context-parallel-size 2 \
  --gpu-memory-utilization 0.956 \
  --max-model-len 524288 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --prefix-cache-retention-interval 0 \
  --mamba-cache-mode align \
  --no-async-scheduling \
  --kv-cache-dtype auto \
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
      "attention_backend": "FLASH_ATTN_V100"}' \
  --attention-backend FLASH_ATTN_V100 \
  --prefill-pressure-token-budget 1024 \
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
  --shutdown-timeout 30 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --reasoning-parser qwen3 \
  --enable-prompt-tokens-details \
  --sse-keep-alive-interval 15 \
  --host 0.0.0.0 --port 8000
```

Container notes:

- **Shared memory.** The CPU KV pool is a file in `/dev/shm`, and so is the host store of [Host KV](kv-offload.md#host-kv-long-context). Make `--shm-size` larger than `cpu_bytes_to_use` (or `host_store_gib` with Host KV); the validated container used `128g`. The engine removes the file names once every process has mapped them, so a normal or forced stop releases the memory. Only a crash during startup can leave `vllm_offload_*.mmap` files behind; recreating the container removes them.
- **Caches.** The compile cache lives under `VLLM_CACHE_ROOT` (default `/root/.cache/vllm` inside the container) and the TileLang cache under `TILELANG_CACHE_DIR`. Point both at the mounted volume, as above; otherwise a container started with `--rm` compiles everything again on each start.
- **Privileges.** `--cap-add SYS_PTRACE` was required in the validated container: the workers share memory with each other through `pidfd_getfd`. The validated container also used `--security-opt seccomp=unconfined`; it has not been tested without it.
- **Stopping.** Keep `--stop-timeout` above `--shutdown-timeout` plus the process-exit wait, which defaults to the same value (30 + 30 seconds here). See [Shutdown and diagnostics](configuration.md#shutdown-and-diagnostics).
- **Derived images.** Do not bake `VLLM_SM70_*` switches into derived images: values from the environment override the profile.

## See also

- [Checkpoints and KV formats](checkpoints.md): other checkpoints, MTP drafts and KV formats
- [Configuration reference](configuration.md): launch options, profile defaults and memory budgets
- [Speculative decoding](speculative-decoding.md): MTP and DFlash2
- [KV offload](kv-offload.md): CPU, SSD and Host KV modes
- [Multimodal input limits](multimodal.md)
