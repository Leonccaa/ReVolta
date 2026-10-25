# ReVolta

**A daily AI server on 4 × V100 32 GB.**

Run Qwen3.8 Flash-Next for text, images, and video, with an OpenAI-compatible API.
ReVolta is a Volta-focused fork of [vLLM](https://github.com/vllm-project/vllm),
through [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM).

Current release: [0.1.3](https://github.com/Leonccaa/ReVolta/releases/tag/v0.1.3).

## Highlights

- **GPU-resident context:** **1.38 million tokens** with a reported maximum concurrency of **2.63× at 512K** (AWQ, E4M3 KV, DFlash2 recommended settings, 0.1.3), up to 1.54 million (2.94×) with state replay on; see [GPU KV capacity](docs/performance.md#gpu-kv-capacity-by-method).
- **Speculative decoding:** native MTP4 by default, or [DFlash2](docs/dflash2.md) with the incoai drafter
  (new in 0.1.2, up to 1.30× on one request).
- **Speed:** up to **525 tokens/s total output at 8 concurrent requests** on NVLink
  ([community result](https://github.com/Leonccaa/ReVolta/issues/2)), and up to 509 on our PCIe reference; see [Speed](docs/performance.md#speed).
- **Cache reuse:** [CPU memory and disk offload](docs/kv-offload.md); a cached 32K document answered in **1.28 s vs 10.06 s** when recomputed.

## Hardware and models

### Hardware

The reference system uses four V100 32 GB PCIe cards behind one PCIe switch
(PIX/PXB in `nvidia-smi topo -m`), with P2P enabled and a 128 GiB container memory limit.

**128 GiB RAM is recommended** for the default 48 GiB CPU KV pool.
See [memory and storage](docs/configuration.md#system-memory-and-storage) and the
[full hardware test matrix](docs/known-limits.md).

### Models

| Checkpoint | Test status | Preparation |
| --- | --- | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) | Default; full serving acceptance passed | [Setup](docs/checkpoints.md#uncensored-or-base-awq) |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | Quantization quality measured; full serving acceptance not run | [Setup](docs/checkpoints.md#uncensored-or-base-awq) |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | Full serving acceptance passed with format-specific settings | [Setup](docs/checkpoints.md#standard-nvfp4-checkpoint-setup) |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | Full serving acceptance passed with vision-tower TP4 | [Setup](docs/checkpoints.md#nvidia-fp4-local-checkpoint-setup) |

Use `--dtype half`: V100 has no BF16 support.

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
- The AWQ PLE table always needs about 48 GiB of host memory; retest memory use when changing concurrency or media limits.
- PLE cascade is unsupported in 0.1.x; keep it disabled.
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
