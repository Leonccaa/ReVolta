# ReVolta

**Stable, fast inference on multi-GPU NVIDIA V100 servers.**

ReVolta turns a server with V100 32 GB GPUs and plenty of system memory into a dependable daily AI service. It serves Qwen3.8 Flash-Next (text, images and video) behind an OpenAI-compatible API, with a 512K-token context and eight concurrent requests. Deployment is simple: pull the Docker image and launch. The ReVolta profile recognizes the setup and applies tuned settings by itself. ReVolta is built on [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a V100-focused fork of vLLM, so the `vllm` command, its options and its OpenAI-compatible API work as usual.

ReVolta is built for four or more V100 32 GB GPUs and validated on four. We recommend, and test, the two interconnects that give the best speed: NVLink, or all GPUs under one PCIe switch with P2P. Every release passes the same fixed gate on our own server, which runs ReVolta every day: concurrency, KV readback from CPU memory, long context, quality subsets and fixed prompts compared with the previous release; areas a release changes, such as images and video, are tested again.

**You need:**

- Four or more V100 32 GB GPUs (validated on four), on NVLink or with all GPUs under one PCIe switch with P2P.
- 128 GiB of system memory recommended; 80 GiB is the smallest tested limit (0.1.0, KV offload off). For less, see the [experimental PLE disk option](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md#whole-ple-table-on-disk-experimental).
- Linux x86-64 and an NVIDIA driver for CUDA 12.8.
- Docker with GPU support, or Python 3.12 with glibc 2.34 or newer.
- About 142 GB of disk for the default checkpoint. Tested hardware and limits: [Support and known limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md).

**Current release: [0.2.0](https://github.com/Leonccaa/ReVolta/releases/tag/v0.2.0).** INT8 KV cache by default, optional Host KV that keeps the KV cache in host memory, draft widths that follow the batch size, and serving defaults that the profile applies by itself. Upgrading from 0.1.5: read the [release notes](https://github.com/Leonccaa/ReVolta/releases/tag/v0.2.0).

## At a glance

**What 0.2.0 improves:** These gains are measured against earlier 0.2.0 builds, not against 0.1.5.

- **NVLink, public NVFP4 + MTP5:** In same-system tests during 0.2.0 development, automatic tuning cut wall-clock time per decode round by **38% / 33% / 28%** at 1 / 4 / 8 concurrent requests versus the earlier defaults without automatic tuning. The release measured **16.43 / 23.45 / 32.81 ms** per round on a separate fully connected NVLink system.
- **PCIe, production AWQ + DFlash2:** Compared with the same setup on the previous 0.2.0 candidate, cold time to first token at 32K fell from **9.90 to 8.54 s**, 128K prefill fell from **39.0 to 32.8 s**, and RevBench decode throughput rose **2.4%**.
- **PCIe, public AWQ + MTP4:** Compared with the same setup on the previous 0.2.0 candidate, cold time to first token at 32K fell **14%** and RevBench decode throughput rose **7.3%**.

MTP4 drafts up to four tokens per step with the model's own multi-token-prediction (MTP) layer (three per step at four or more requests); the reference launch uses it. DFlash2 uses a separate drafter model; our own server uses it.

**Context and capacity** (0.2.0; Uncensored AWQ g32, INT8 target KV, 512K context, 8 sequences, `--gpu-memory-utilization 0.956`, our 4 × V100 32 GB PCIe server)

| | MTP4 (reference launch; draft KV inherits INT8) | DFlash2 (our production setup; FP16 draft KV) |
| --- | ---: | ---: |
| Context per request | 512K tokens | 512K tokens |
| GPU KV pool | 1,194,816 tokens | 1,851,676 tokens |
| GPU KV pool / 512K tokens (capacity ratio) | 2.28× | 3.53× |
| Concurrent requests | 8 | 8 |

On the NVLink system below, NVFP4 + MTP5 with INT8 KV held **1,305,719 tokens** in the GPU KV pool.

With optional Host KV and INT8 KV, the 98 GiB host pool held **7,329,312 usable tokens** on both systems (AWQ + MTP5 on PCIe, NVFP4 + MTP5 on NVLink). In the separate GPU KV test with DFlash2, one request revisiting a 32K-token document reached its first token in **0.599 s** from CPU memory or **1.011 s** from SSD after restart, versus **8.56 s** with cold recomputation. See [KV offload](https://github.com/Leonccaa/ReVolta/blob/main/docs/kv-offload.md).

**Speed** (0.2.0): observed end-to-end output from the same short synthetic English story workload, sampled without the chat template and fixed at 512 output tokens per request. Each concurrency shows the higher total output rate of the tested MTP and DFlash2 configurations: AWQ + MTP4 versus AWQ + DFlash2 on PCIe, and NVFP4 + MTP5 versus AWQ + DFlash2 on NVLink. These are separate machines with different power limits, not a same-system interconnect comparison or a ranking for real prompts.

| System (power limit per GPU) | 1 request | 4 | 8 | Cold prefill (prompt tok/s at stated lengths) |
| --- | ---: | ---: | ---: | ---: |
| 4 × V100 32 GB PCIe, one PCIe switch with P2P (200 W) | 114.4 tok/s | 283.0 tok/s | 378.1 tok/s | 3,787 at 32K / 3,898 at 128K |
| 4 × V100-SXM2 32 GB, all six pairs directly linked by NVLink (300 W) | 136.1 tok/s | 370.9 tok/s | 532.4 tok/s | 7,892 at 32K / 7,953 at 64K |

Cold prefill uses one request and divides the actual prompt length by median prefill time for PCIe (AWQ + DFlash2) or median time to first token for NVLink (NVFP4 + MTP5). NVLink was not tested at 128K in this benchmark. See [Performance](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md#nvlink-tp4-acceptance) for test conditions.

## Models

| Model | Weights (bits per weight: routed experts / whole model) | Download | PLE table in host memory | GPU KV pool (0.2.0) | Quality vs its BF16 parent: KL ↓ / top-1 ↑ | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) **(recommended)** | 4-bit AWQ g32 experts, FP8 PLE, BF16 rest (**4.63 / 5.95**) | 142 GB, MTP draft included | 48 GiB | 1,194,816 tokens (MTP4) | 0.0214 / 97.1% | Uncensored fine-tune; the reference setup |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | 4-bit AWQ g32 experts, FP8 PLE, BF16 rest (**4.63 / 5.95**) | 142 GB, MTP draft included | 48 GiB | Same layout as Uncensored | 0.0207 / 97.1% | The original model |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | NVFP4 experts, FP8 PLE, BF16 rest (**4.50 / 5.86**) | 135 GB + 5 GB draft pack | 48 GiB | 1,488,853 tokens (MTP5) | 0.0303 / 96.4% | Widely used community format |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVFP4 experts and PLE, MXFP8 attention and shared experts (**4.26 / 4.53**) | 102 GB, MTP layer built in | **27 GiB** | 1,518,412 tokens (MTP5 smoke test) | 0.0380 / 95.9% | NVIDIA's official checkpoint; least host memory |

The PLE tables are the model's n-gram embedding tables; they hold 51 billion of its 177 billion stored parameters and stay in host memory by default. V100 has no BF16 units, so ReVolta runs BF16 weights in FP16, which `--dtype auto` selects on V100. Bits per weight include quantization scales and zero points and exclude the MTP draft; the whole-model figure covers all 177 billion stored parameters. The full weight layout of each checkpoint is in [Checkpoints](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md#choose-a-checkpoint).

The GPU KV pool is the startup log's capacity in 0.2.0 with INT8 target and draft KV, 512K context, 8 sequences and `--gpu-memory-utilization 0.956`; the AWQ row uses MTP with 4 draft tokens and the NVFP4 rows use MTP with 5, as tested. Quality is measured against each checkpoint's own BF16 parent with scalar E4M3 KV (published with 0.1.0); method and the second reference are in [Quality measurements](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md#quality-measurements). Test status, pinned revisions and preparation steps: [Checkpoints](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md).

Also supported: [NVIDIA NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD) (loading and serving checks passed; quality against BF16 not yet measured; the repository is not publicly downloadable at the time of writing). DFlash2 needs the separate [incoai DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2). Text-only [Hy4-flash](https://huggingface.co/tencent/Hy4-flash) has validation support, outside the 0.2.0 release tests; see [Hy4-flash](https://github.com/Leonccaa/ReVolta/blob/main/docs/hy4-flash.md).

## Quick start

1. Pull the image, or install the package (the Python import and the CLI stay `vllm`):

   ```bash
   docker pull ghcr.io/leonccaa/revolta:0.2.0
   # or, in a Python 3.12 virtual environment:
   pip install revolta==0.2.0   # the first start also needs the CUDA 12.8 Toolkit (nvcc)
   ```

2. Download the checkpoint and make its 512K views with [step 2 of Getting started](https://github.com/Leonccaa/ReVolta/blob/main/docs/getting-started.md#2-download-the-checkpoint-and-make-512k-views).
3. Launch with the reference command in [Getting started](https://github.com/Leonccaa/ReVolta/blob/main/docs/getting-started.md#4-launch) (pip) or its [Docker command](https://github.com/Leonccaa/ReVolta/blob/main/docs/getting-started.md#docker). The first start takes about 10–15 minutes (weights, PLE table, compilation).
4. Check that the server answers: `curl http://localhost:8000/v1/models` lists the model.

For another checkpoint, follow its [preparation guide](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md).

<a id="moved-sections"></a><a id="measurements"></a><a id="checkpoints-kv-scales-and-mtp-drafts"></a><a id="dflash2-speculative-decoding-optional"></a><a id="kv-offload-modes"></a><a id="weight-and-kv-formats"></a><a id="standard-nvfp4"></a><a id="performance"></a><a id="quality-measurements"></a><a id="support-and-known-limits"></a><a id="support-matrix"></a><a id="reporting-issues"></a>

## Documentation

| I want to… | Read |
| --- | --- |
| Install and launch the reference configuration | [Getting started](https://github.com/Leonccaa/ReVolta/blob/main/docs/getting-started.md) |
| Choose and prepare a checkpoint and its MTP draft, and choose the KV format | [Checkpoints and KV formats](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md) |
| Choose between MTP and DFlash2 | [Speculative decoding](https://github.com/Leonccaa/ReVolta/blob/main/docs/speculative-decoding.md) |
| Enable DFlash2 and choose its settings | [DFlash2](https://github.com/Leonccaa/ReVolta/blob/main/docs/dflash2.md) |
| Change the number of draft tokens with the batch size | [Dynamic speculative decoding](https://github.com/Leonccaa/ReVolta/blob/main/docs/dynamic-speculative-decoding.md) |
| Keep evicted prefixes in CPU memory or on disk, or the KV cache in host memory | [KV offload](https://github.com/Leonccaa/ReVolta/blob/main/docs/kv-offload.md); every key and metric: [KV offload reference](https://github.com/Leonccaa/ReVolta/blob/main/docs/kv-offload-reference.md) |
| Look up launch options, environment variables, sampling defaults and memory budgets | [Configuration](https://github.com/Leonccaa/ReVolta/blob/main/docs/configuration.md) |
| Compare measured speed, KV capacity and quantization quality | [Performance](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md) |
| Check the support scope and known limitations | [Known limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md) |
| Set image and video input limits | [Multimodal input limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/multimodal.md) |
| Watch one running service in a browser | [Dashboard](https://github.com/Leonccaa/ReVolta/blob/main/docs/dashboard.md) |
| See what the anonymous usage report contains, or turn it off | [Usage statistics](https://github.com/Leonccaa/ReVolta/blob/main/docs/usage-stats.md) |
| Run Hy4-flash (validation support) | [Hy4-flash](https://github.com/Leonccaa/ReVolta/blob/main/docs/hy4-flash.md) |

## Known limits

- **Scope.** V100 16 GB, GPU counts other than four, TP2/TP8 and other models have not been tested. Only SM70 is built: other GPU architectures, ROCm, CPU, XPU, TPU and GGUF are outside the build.
- **Reproducibility.** A single greedy request is reproducible bit for bit with a fixed configuration; with 2–8 concurrent requests, greedy outputs can change between runs.
- **Host memory.** The whole PLE table is pinned in host memory by default (about 48 GiB for AWQ and standard NVFP4, 27 GiB for FP4-Local); if it does not fit, startup stops and reports the required and available memory.
- **DFlash2 depends on the workload.** English and code gain the most, Chinese is often faster with MTP4; see [Choosing a method](https://github.com/Leonccaa/ReVolta/blob/main/docs/speculative-decoding.md#choosing-a-method).

The complete list and the test boundaries are in [Support and known limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md). There is no GPU CI; releases are checked on the reference machine.

## Project direction

ReVolta focuses on dependable daily service on V100. Releases check output correctness, long-context retrieval, concurrent use, cache recovery and speed on the reference machine.

After 0.2.0, work continues on faster decoding with several concurrent requests, smoother service when long prompts arrive during decoding, and further tuning for NVLink systems. Longer-term work includes lower host-memory use and fewer runtime dependencies. There are no release dates.

## Getting help

Use [GitHub Issues](https://github.com/Leonccaa/ReVolta/issues). Include the launch command, relevant environment, ReVolta version, model revision, host or container memory limits, offload settings, `nvidia-smi topo -m`, and the `ReVolta profile` and `SM70 interconnect=` log lines. Remove credentials. Report security issues through GitHub private vulnerability reporting.

Reports from NVLink V100 owners are especially welcome; share measured results with the [test results template](https://github.com/Leonccaa/ReVolta/issues/new?template=test_results.md). Thanks to [@jttdev](https://github.com/jttdev), [@That80-20](https://github.com/That80-20) and [@vincentyan2023](https://github.com/vincentyan2023) for sharing their results ([#2](https://github.com/Leonccaa/ReVolta/issues/2), [#38](https://github.com/Leonccaa/ReVolta/issues/38), [#37](https://github.com/Leonccaa/ReVolta/issues/37)).

For project status and development background, read the [pinned overview (#1)](https://github.com/Leonccaa/ReVolta/issues/1). See [CONTRIBUTING.md](https://github.com/Leonccaa/ReVolta/blob/main/CONTRIBUTING.md) before preparing a contribution, and the [support scope](https://github.com/Leonccaa/ReVolta/blob/main/.github/SUPPORT.md) for supported versions.

## Building from source

See [BUILDING.md](https://github.com/Leonccaa/ReVolta/blob/main/BUILDING.md). Builds use Docker BuildKit, CUDA 12.8 and SM70 only; CUDA 13 cannot compile SM70. No GPU is needed to compile. Debug symbols are published separately.

## License and acknowledgements

ReVolta's engine code and project changes are licensed under the [Apache License 2.0](https://github.com/Leonccaa/ReVolta/blob/main/LICENSE), except for third-party files that state another license. [NOTICE](https://github.com/Leonccaa/ReVolta/blob/main/NOTICE) lists component attributions and known source-history gaps. [LICENSES/](https://github.com/Leonccaa/ReVolta/blob/main/LICENSES/README.md) contains the upstream license texts with their source revisions. Model weights have their own licenses; see each model card.

ReVolta is derived from [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), which is derived from [vLLM](https://github.com/vllm-project/vllm). Thanks to the vLLM and 1Cat-vLLM contributors, and to the authors of the attention, quantization, communication and GPU kernel components included here (see NOTICE).

NVIDIA, Volta and Tesla are trademarks of NVIDIA Corporation and are used here only to identify hardware. ReVolta is not affiliated with or endorsed by NVIDIA.
