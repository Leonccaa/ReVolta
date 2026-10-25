# ReVolta

**Stable, fast inference on multi-GPU NVIDIA V100 servers.**

ReVolta turns a server with V100 32 GB GPUs and plenty of system memory into a dependable daily AI service. It runs Qwen3.8 Flash-Next for text, images and video behind an OpenAI-compatible API. ReVolta is built on [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a V100-focused fork of vLLM, so the `vllm` command, its options and its OpenAI-compatible API work as usual.

It is built for four or more V100 32 GB GPUs and validated on four. Every release passes the same gate on our own server before it ships: greedy output identical to the previous release token for token, 4 and 8 concurrent requests, images and video, CPU cache readback, and 512K-token prompts. That server runs ReVolta every day.

**You need:** 4 × V100 32 GB (PCIe with P2P, or NVLink), at least 64 GB of system memory (128 GB recommended), Linux x86-64 with glibc 2.34 or newer, and an NVIDIA driver for CUDA 12.8. Tested hardware and limits: [Support and known limits](docs/known-limits.md).

Current release: [0.1.3](https://github.com/Leonccaa/ReVolta/releases/tag/v0.1.3).

## At a glance

**Context and capacity** (Uncensored AWQ, E4M3 KV cache, 0.1.3)

| | MTP4 (default) | DFlash2 (recommended settings) |
| --- | ---: | ---: |
| Context per request | 512K tokens | 512K tokens |
| GPU KV pool | 1.15 million tokens | **1.38 million tokens** |
| Full 512K requests held on the GPU at once | 2.20 | **2.63** |
| Concurrent requests | 8 | 8 |

Prefixes evicted from the GPU stay in CPU memory or on disk: a cached 32K-token document is answered in **1.28 s instead of 10.06 s**. See [KV offload](docs/kv-offload.md).

**Speed** (total output tokens/s, 512-token responses; best of MTP4 and DFlash2)

| System | 1 request | 2 | 4 | 8 | Prefill, 8K / 131K prompt |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 × V100 32 GB PCIe, one PCIe switch (our server) | 201 | 275 | 370 | **509** | 3,382 / 3,016 |
| 4 × V100-SXM2 32 GB, NVLink ([community result](https://github.com/Leonccaa/ReVolta/issues/2)) | 234 | 260 | 368 | **525** | 4,680 / 4,037 |

Versions, power caps and the per-method breakdown are in [Performance](docs/performance.md).

## Models

| Model | Weights (bits per weight: routed experts / whole model) | Download | PLE table in host memory | GPU KV pool (MTP4) | Quality vs its BF16 parent: KL ↓ / top-1 ↑ | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) **(recommended)** | 4-bit AWQ g32 experts, FP8 PLE, BF16 rest (**4.63 / 5.95**) | 142 GB, MTP draft included | 48 GiB | 1.15 M (1.38 M with DFlash2) | 0.0214 / 97.1% | Uncensored fine-tune; the reference setup |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | 4-bit AWQ g32 experts, FP8 PLE, BF16 rest (**4.63 / 5.95**) | 142 GB | 48 GiB | Same layout as Uncensored | 0.0207 / 97.1% | The original model |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | NVFP4 experts, FP8 PLE, BF16 rest (**4.50 / 5.86**) | 135 GB + 5 GB draft pack | 48 GiB | 1.25 M | 0.0303 / 96.4% | Widely used community format |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVFP4 experts and PLE, MXFP8 attention and shared experts (**4.26 / 4.53**) | 102 GB, MTP layer built in | **27 GiB** | **1.47 M** | 0.0380 / 95.9% | NVIDIA's official checkpoint; least host memory |

In the AWQ and standard NVFP4 checkpoints only the routed experts are 4-bit: the PLE tables are FP8, and all other weights (attention, shared experts, embeddings, `lm_head` and the vision tower) stay BF16. NVIDIA FP4-Local is quantized further: its PLE table and parts of the vision tower and MTP layer are NVFP4, and its attention projections and shared experts are MXFP8; its `lm_head` stays BF16. V100 has no BF16 units, so ReVolta runs BF16 weights in FP16 (`--dtype half`). Bits per weight include quantization scales and zero points and exclude the MTP draft; the whole-model figure covers all 177 billion stored parameters, 51 billion of them in the PLE tables.

GPU KV pools use the recommended setup for each checkpoint (E4M3 target and draft KV, 512K, 8 sequences). Quality is measured with an E4M3 KV cache on 66 English, Chinese and code records against each checkpoint's own BF16 parent; method and other datasets are in [Quality measurements](docs/performance.md#quality-measurements). Test status, pinned revisions and preparation steps for each checkpoint are in [Checkpoints](docs/checkpoints.md).

## Quick start

Use Linux x86-64 with glibc 2.34 or newer, Python 3.12, and an NVIDIA driver for CUDA 12.8.
The package is `revolta`; the Python import and CLI remain `vllm`.

```bash
python3.12 -m venv /data/revolta/venv
. /data/revolta/venv/bin/activate
pip install revolta==0.1.3
vllm --version          # 0.1.3
```

Or pull the Docker image:

```bash
docker pull ghcr.io/leonccaa/revolta:0.1.3
```

Follow [Getting started](docs/getting-started.md) to download the pinned checkpoint,
make the target and draft views, apply the environment settings, launch and check the API.
It includes the complete reference launch and Docker commands.

For another checkpoint, use its [preparation guide](docs/checkpoints.md) before launching.

## Documentation

| I want to… | Read |
| --- | --- |
| Install and launch the reference configuration | [Getting started](docs/getting-started.md) |
| Select and prepare a checkpoint, KV scales and an MTP draft | [Checkpoints](docs/checkpoints.md) |
| Enable DFlash2 and choose its settings | [DFlash2](docs/dflash2.md) |
| Keep evicted prefixes in CPU memory or on disk | [KV offload](docs/kv-offload.md) |
| Look up deployment settings, sampling and memory budgets | [Configuration](docs/configuration.md) |
| Compare measured speed, KV capacity and quantization quality | [Performance](docs/performance.md) |
| Check support scope and limitations | [Known limits](docs/known-limits.md) |

## Known limits

- V100 16 GB, TP2/TP8, and other models have not been tested; NVLink has community results and is outside the release gate.
- This release builds SM70 only; other GPU architectures, ROCm, CPU, XPU, TPU, and GGUF are outside the build.
- Greedy outputs at C2–C8 can change between runs; fresh compiler autotuning can also change outputs.
- The AWQ PLE table needs about 48 GiB of host memory by default. 0.1.3 adds an experimental option that keeps the whole table on disk instead; it ran under 64 GiB and 32 GiB container limits, but speed was not measured. See [Support and known limits](docs/known-limits.md).
- PLE cascade (spilling part of the table to disk, `ple_disk_cascade=true`) is not supported yet and is off by default; full support is planned for 0.2.
- INT8 KV cannot be combined with DFlash2; the disk tier does not support INT8 yet.

See [Support and known limits](docs/known-limits.md) for the complete list and test boundaries.

## Project direction

ReVolta focuses on dependable daily service on V100. Releases check output correctness,
long-context retrieval, concurrent use, multimodal input, cache recovery, and speed
on the reference machine.

Next: faster INT8 KV decoding, simpler NVFP4 setup, INT8 disk caching, and community
testing on NVLink systems. Longer-term work includes reducing host-memory use and
runtime dependencies.

## Getting help

Use [GitHub Issues](https://github.com/Leonccaa/ReVolta/issues). Include the launch
command, relevant environment, model revision, host/container memory limits,
offload settings, `nvidia-smi topo -m`, and the `ReVolta profile` /
`SM70 interconnect=` log lines. Remove credentials.

Reports from NVLink V100 owners are especially welcome; thanks to
[@jttdev](https://github.com/jttdev) for sharing their results ([#2](https://github.com/Leonccaa/ReVolta/issues/2)). Report security issues
through GitHub private vulnerability reporting.

For project status and development background, read the [pinned overview (#1)](https://github.com/Leonccaa/ReVolta/issues/1).
See [CONTRIBUTING.md](CONTRIBUTING.md) before preparing a contribution.

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

## Moved sections

These headings keep existing README links working; follow each link for the complete section.

### Measurements

See [Measurements](docs/performance.md#measurements).

### Checkpoints, KV scales and MTP drafts

See [Checkpoints, KV scales and MTP drafts](docs/checkpoints.md).

### DFlash2 speculative decoding (optional)

See [DFlash2 speculative decoding (optional)](docs/dflash2.md).

### KV offload modes

See [KV offload modes](docs/kv-offload.md).

### Weight and KV formats

See [Weight and KV formats](docs/checkpoints.md#weight-and-kv-formats).

### Standard NVFP4

See [Standard NVFP4](docs/checkpoints.md#standard-nvfp4).

### Performance

See [Performance](docs/performance.md#performance).

### Quality measurements

See [Quality measurements](docs/performance.md#quality-measurements).

### Support and known limits

See [Support and known limits](docs/known-limits.md).

### Support matrix

See [Support matrix](docs/known-limits.md).

### Reporting issues

See [Reporting issues](#getting-help).
