# ReVolta

**Stable, fast inference on multi-GPU NVIDIA V100 servers.**

ReVolta turns a server with V100 32 GB GPUs and plenty of system memory into a dependable daily AI service. It serves Qwen3.8 Flash-Next (text, images and video) behind an OpenAI-compatible API, with a 512K-token context and eight concurrent requests. Deployment is simple: pull the Docker image and launch. The ReVolta profile recognizes the setup and applies tuned settings, so a launch needs only the model, the GPU layout, the draft model and the memory setting. ReVolta is built on [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), a V100-focused fork of vLLM, so the `vllm` command, its options and its OpenAI-compatible API work as usual.

ReVolta is built for four or more V100 32 GB GPUs and validated on four. We recommend, and test, the two interconnects that give the best speed: NVLink, or all GPUs under one PCIe switch with P2P. Every release passes the same fixed gate on our own server before it ships: 4 and 8 concurrent requests, KV cache readback from CPU memory, long context, GSM8K and MMLU quality subsets, and fixed prompts compared with the previous release (the output is identical, or each difference has an identified cause and no quality loss). Images, video and other areas are tested again when a release changes them. That server runs ReVolta every day.

**You need:** four or more V100 32 GB GPUs (validated on four), on NVLink or with all GPUs under one PCIe switch with P2P; 128 GiB of system memory recommended, 80 GiB at least (tested in 0.1.0 with KV offload off) with the model's PLE embedding tables in host memory, which is the default (32 GiB was tested in 0.1.4 with the experimental whole-table PLE disk option, which made single-request decoding about 23–25% slower); Linux x86-64; an NVIDIA driver for CUDA 12.8; and Docker with GPU support, or Python 3.12 with glibc 2.34 or newer. Tested hardware and limits: [Support and known limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md).

**Current release: [0.2.0](https://github.com/Leonccaa/ReVolta/releases/tag/v0.2.0).** INT8 KV cache by default, optional Host KV that keeps the KV cache in host memory, draft widths that follow the batch size, and serving defaults that make a short launch command run the validated setup. Upgrading from 0.1.5: read the [release notes](https://github.com/Leonccaa/ReVolta/releases/tag/v0.2.0).

## At a glance

MTP4 drafts up to four tokens per step with the model's own multi-token-prediction (MTP) layer (three per step at four or more requests); the reference launch uses it. DFlash2 uses a separate drafter model; our own server uses it.

**Context and capacity** (0.2.0; Uncensored AWQ g32, INT8 target KV, 512K context, 8 sequences, `--gpu-memory-utilization 0.956`, our 4 × V100 32 GB PCIe server)

| | MTP4 (default; draft KV inherits INT8) | DFlash2 (our production setup; FP16 draft KV) |
| --- | ---: | ---: |
| Context per request | 512K tokens | 512K tokens |
| GPU KV pool | <!-- TODO(0.2.0-acceptance): GPU KV pool tokens from the startup log, D+E1 --> | <!-- TODO(0.2.0-acceptance): GPU KV pool tokens from the startup log, C --> |
| GPU KV pool / 512K tokens (capacity ratio) | <!-- TODO(0.2.0-acceptance): GPU KV pool / 524,288, D+E1 --> | <!-- TODO(0.2.0-acceptance): GPU KV pool / 524,288, C --> |
| Concurrent requests | 8 | 8 |

With the optional Host KV mode, the main KV cache lives in host memory instead: <!-- TODO(0.2.0-acceptance): usable Host KV tokens and host pool size, E2 -->. Prefixes evicted from the GPU stay in CPU memory or on disk: a cached 32K-token document comes back from CPU memory in **0.6 s**, and after an engine restart from a local NVMe disk tier in **1.5 s**, instead of **10.1 s** when recomputed (0.1.4, DFlash2, E4M3 KV). <!-- TODO(0.2.0-acceptance): 32K readback time from CPU memory and from disk after a restart, and recompute time, C --> See [KV offload](https://github.com/Leonccaa/ReVolta/blob/main/docs/kv-offload.md).

**Speed** (0.2.0; total output tokens/s, the faster of MTP4 and DFlash2 at each concurrency, 200 W per GPU <!-- TODO(0.2.0-acceptance): response length and prompt set, C and D+E1 -->)

| System | 1 request | 4 | 8 | Cold prefill, 32K / 128K prompt |
| --- | ---: | ---: | ---: | ---: |
| 4 × V100 32 GB PCIe, one PCIe switch with P2P (our server) | <!-- TODO(0.2.0-acceptance): C1 output tok/s, best of C and D+E1 --> | <!-- TODO(0.2.0-acceptance): C4 output tok/s, best of C and D+E1 --> | <!-- TODO(0.2.0-acceptance): C8 output tok/s, best of C and D+E1 --> | <!-- TODO(0.2.0-acceptance): cold prefill tok/s at 32K / 128K, C --> |

0.2.0 has not been measured on NVLink. Community results on 4 × V100-SXM2 32 GB with NVLink (0.1.1 and 0.1.2), the per-method breakdown and the test conditions are in [Performance](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md).

## Models

| Model | Weights (bits per weight: routed experts / whole model) | Download | PLE table in host memory | GPU KV pool (0.2.0) | Quality vs its BF16 parent: KL ↓ / top-1 ↑ | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| [Uncensored AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-Uncensored-AWQ-g32) **(recommended)** | 4-bit AWQ g32 experts, FP8 PLE, BF16 rest (**4.63 / 5.95**) | 142 GB, MTP draft included | 48 GiB | <!-- TODO(0.2.0-acceptance): GPU KV pool tokens, MTP4 INT8, D+E1 --> (MTP4) | 0.0214 / 97.1% | Uncensored fine-tune; the reference setup |
| [Base AWQ g32](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-AWQ-g32) | 4-bit AWQ g32 experts, FP8 PLE, BF16 rest (**4.63 / 5.95**) | 142 GB, MTP draft included | 48 GiB | Same layout as Uncensored | 0.0207 / 97.1% | The original model |
| [Standard NVFP4](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) | NVFP4 experts, FP8 PLE, BF16 rest (**4.50 / 5.86**) | 135 GB + 5 GB draft pack | 48 GiB | <!-- TODO(0.2.0-acceptance): GPU KV pool tokens, Standard NVFP4 MTP5 INT8, E3a --> (MTP5) | 0.0303 / 96.4% | Widely used community format |
| [NVIDIA FP4-Local](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local) | NVFP4 experts and PLE, MXFP8 attention and shared experts (**4.26 / 4.53**) | 102 GB, MTP layer built in | **27 GiB** | <!-- TODO(0.2.0-acceptance): GPU KV pool tokens, FP4-Local MTP5 INT8, E3c smoke --> (MTP5) | 0.0380 / 95.9% | NVIDIA's official checkpoint; least host memory |

The PLE tables are the model's n-gram embedding tables; they hold 51 billion of its 177 billion stored parameters and stay in host memory by default. In the AWQ and standard NVFP4 checkpoints only the routed experts are 4-bit: the PLE tables are FP8, and all other weights (attention, shared experts, embeddings, `lm_head` and the vision tower) stay BF16. NVIDIA FP4-Local is quantized further: its PLE table and parts of the vision tower and MTP layer are NVFP4, and its attention projections and shared experts are MXFP8; its `lm_head` stays BF16. V100 has no BF16 units, so ReVolta runs BF16 weights in FP16, which `--dtype auto` selects on V100. Bits per weight include quantization scales and zero points and exclude the MTP draft; the whole-model figure covers all 177 billion stored parameters.

The GPU KV pool is the startup log's capacity in 0.2.0 with INT8 target and draft KV, 512K context, 8 sequences and `--gpu-memory-utilization 0.956`; the AWQ row uses MTP with 4 draft tokens and the NVFP4 rows use MTP with 5, as tested. The quality figures were measured with scalar E4M3 KV on 66 English, Chinese and code records, each checkpoint against its own BF16 parent; INT8 KV measured a slightly lower KL on the AWQ checkpoint. Method and other datasets: [Quality measurements](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md#quality-measurements). Test status, pinned revisions and preparation steps: [Checkpoints](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md).

Also supported: [NVIDIA NVFP4-QAD](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4-QAD) (loading and serving checks passed; quality against BF16 not yet measured; the repository is not publicly downloadable at the time of writing). DFlash2 needs the separate [incoai DFlash2 drafter](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2). Text-only [Hy4-flash](https://huggingface.co/tencent/Hy4-flash) has validation support: it was validated once, in 0.2.0rc13, and its model code ships unchanged in 0.2.0, but the 0.2.0 release tests did not run it. See [Hy4-flash](https://github.com/Leonccaa/ReVolta/blob/main/docs/hy4-flash.md).

## Quick start

Pull the image (or install the package; the Python import and the CLI stay `vllm`):

```bash
docker pull ghcr.io/leonccaa/revolta:0.2.0
# or, in a Python 3.12 virtual environment:
pip install revolta==0.2.0   # the first start also needs the CUDA 12.8 Toolkit (nvcc)
```

Download the checkpoint and make its 512K views with [step 2 of Getting started](https://github.com/Leonccaa/ReVolta/blob/main/docs/getting-started.md#2-download-the-checkpoint-and-make-512k-views). If prompt and output together stay within 262,144 tokens, skip the views: serve the checkpoint directory with `--max-model-len 262144` and use its `mtp-fp8/` directory as the draft.

Then launch. The ReVolta profile sets the KV format, scheduler, draft widths and CUDA graph reserve for this setup, so the command only names the model, the GPU layout, the draft, the memory setting and the API options you use:

```bash
docker run --rm --gpus all --shm-size 128g --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined -p 8000:8000 -v /data/revolta:/data/revolta \
  ghcr.io/leonccaa/revolta:0.2.0 \
  /data/revolta/views/target-512k --served-model-name Qwen3.8-Flash-Next \
  --tensor-parallel-size 4 --decode-context-parallel-size 2 \
  --max-model-len 524288 --gpu-memory-utilization 0.956 \
  --speculative-config '{"method": "mtp", "model": "/data/revolta/views/draft-512k",
      "quantization": "modelopt_mixed", "num_speculative_tokens": 4}' \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 \
  --limit-mm-per-prompt '{"image": 16, "video": 1}'
```

With the pip package, pass the same options to `vllm serve`. The first start takes about 10–15 minutes (weights, PLE table, compilation); then `curl http://localhost:8000/v1/models` lists the model. [Getting started](https://github.com/Leonccaa/ReVolta/blob/main/docs/getting-started.md) adds the 48 GiB CPU KV pool and the other options of the reference configuration. For another checkpoint, follow its [preparation guide](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md).

## Documentation

| I want to… | Read |
| --- | --- |
| Install and launch the reference configuration | [Getting started](https://github.com/Leonccaa/ReVolta/blob/main/docs/getting-started.md) |
| Choose and prepare a checkpoint and its MTP draft, and choose the KV format | [Checkpoints and KV formats](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md) |
| Choose between MTP and DFlash2 | [Speculative decoding](https://github.com/Leonccaa/ReVolta/blob/main/docs/speculative-decoding.md) |
| Enable DFlash2 and choose its settings | [DFlash2](https://github.com/Leonccaa/ReVolta/blob/main/docs/dflash2.md) |
| Change the number of draft tokens with the batch size | [Dynamic speculative decoding](https://github.com/Leonccaa/ReVolta/blob/main/docs/dynamic-speculative-decoding.md) |
| Keep evicted prefixes in CPU memory or on disk, or the KV cache in host memory | [KV offload](https://github.com/Leonccaa/ReVolta/blob/main/docs/kv-offload.md) |
| Look up launch options, environment variables, sampling defaults and memory budgets | [Configuration](https://github.com/Leonccaa/ReVolta/blob/main/docs/configuration.md) |
| Compare measured speed, KV capacity and quantization quality | [Performance](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md) |
| Check the support scope and known limitations | [Known limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md) |
| Set image and video input limits | [Multimodal input limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/multimodal.md) |
| Watch one running service in a browser | [Dashboard](https://github.com/Leonccaa/ReVolta/blob/main/docs/dashboard.md) |
| See what the anonymous usage report contains, or turn it off | [Usage statistics](https://github.com/Leonccaa/ReVolta/blob/main/docs/usage-stats.md) |
| Run Hy4-flash (validation support, not in the 0.2.0 release tests) | [Hy4-flash](https://github.com/Leonccaa/ReVolta/blob/main/docs/hy4-flash.md) |

## Known limits

- **Scope.** V100 16 GB, GPU counts other than four, TP2/TP8 and other models have not been tested. Only SM70 is built: other GPU architectures, ROCm, CPU, XPU, TPU and GGUF are outside the build.
- **Reproducibility.** A single greedy request is reproducible bit for bit with a fixed configuration (sampled requests need a `seed`). With 2–8 concurrent requests, greedy outputs can change between runs, because the tensor-parallel all-reduce sums in an order that depends on the batch. Fresh compiler autotuning can also change outputs.
- **Host memory.** By default the whole PLE table is pinned in host memory: about 48 GiB for AWQ and standard NVFP4, 27 GiB for FP4-Local. If it does not fit, startup stops and reports the required and available memory; `VLLM_QWEN4EXP_PLE_HOST_GIB` keeps part of the table on the GPU instead. The experimental whole-table PLE disk option served under a 32 GiB memory limit in 0.1.4, with single-request decoding about 23–25% slower and a cold first token 1.6–2.3 times as long. PLE cascade (`ple_disk_cascade`) is not supported and stays off.
- **Disk KV tier.** INT8 and per-token-head E4M3 KV need the `revolta_fs` disk tier; startup rejects the raw `fs` tier for them. Give the service at least 120 s to stop, so that the disk tier can finish its delayed writes.
- **Scalar E4M3 KV is frozen.** Calibrated scalar `fp8_e4m3` still works, but it is no longer optimized and gets only a functional check. Use the default INT8, or `fp8_per_token_head` for E4M3 storage without calibration.
- **DFlash2 depends on the workload.** In earlier measurements, English and code prompts gained the most, while Chinese text was often faster with MTP4. See [Performance](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md).

See [Support and known limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md) for the complete list and test boundaries. There is no GPU CI; releases are checked on the reference machine.

## Project direction

ReVolta focuses on dependable daily service on V100. Releases check output correctness, long-context retrieval, concurrent use, cache recovery and speed on the reference machine.

After 0.2.0, work continues on faster decoding with several concurrent requests, smoother service when long prompts arrive during decoding, and tuning for NVLink systems once one is available for testing. Longer-term work includes lower host-memory use and fewer runtime dependencies. There are no release dates.

## Getting help

Use [GitHub Issues](https://github.com/Leonccaa/ReVolta/issues). Include the launch command, relevant environment, ReVolta version, model revision, host or container memory limits, offload settings, `nvidia-smi topo -m`, and the `ReVolta profile` and `SM70 interconnect=` log lines. Remove credentials. Report security issues through GitHub private vulnerability reporting.

Reports from NVLink V100 owners are especially welcome; share measured results with the [test results template](https://github.com/Leonccaa/ReVolta/issues/new?template=test_results.md) ([`.github/ISSUE_TEMPLATE/test_results.md`](https://github.com/Leonccaa/ReVolta/blob/main/.github/ISSUE_TEMPLATE/test_results.md); agents can fill it in and open the issue through the API). Thanks to [@jttdev](https://github.com/jttdev), [@That80-20](https://github.com/That80-20) and [@vincentyan2023](https://github.com/vincentyan2023) for sharing their results ([#2](https://github.com/Leonccaa/ReVolta/issues/2), [#38](https://github.com/Leonccaa/ReVolta/issues/38), [#37](https://github.com/Leonccaa/ReVolta/issues/37)).

For project status and development background, read the [pinned overview (#1)](https://github.com/Leonccaa/ReVolta/issues/1). See [CONTRIBUTING.md](https://github.com/Leonccaa/ReVolta/blob/main/CONTRIBUTING.md) before preparing a contribution, and the [support scope](https://github.com/Leonccaa/ReVolta/blob/main/.github/SUPPORT.md) for supported versions.

## Building from source

See [BUILDING.md](https://github.com/Leonccaa/ReVolta/blob/main/BUILDING.md). Builds use Docker BuildKit, CUDA 12.8 and SM70 only; CUDA 13 cannot compile SM70. No GPU is needed to compile. Debug symbols are published separately.

## License and acknowledgements

ReVolta's engine code and project changes are licensed under the [Apache License 2.0](https://github.com/Leonccaa/ReVolta/blob/main/LICENSE), except for third-party files that state another license. [NOTICE](https://github.com/Leonccaa/ReVolta/blob/main/NOTICE) lists component attributions and known source-history gaps. [LICENSES/](https://github.com/Leonccaa/ReVolta/blob/main/LICENSES/README.md) contains the upstream license texts with their source revisions. Model weights have their own licenses; see each model card.

ReVolta is derived from [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM), which is derived from [vLLM](https://github.com/vllm-project/vllm). Thanks to the vLLM and 1Cat-vLLM contributors, and to the authors of the attention, quantization, communication and GPU kernel components included here (see NOTICE).

NVIDIA, Volta and Tesla are trademarks of NVIDIA Corporation and are used here only to identify hardware. ReVolta is not affiliated with or endorsed by NVIDIA.

## Moved sections

These headings keep existing links to this page working; follow each link for the current content.

### Measurements

See [Performance and measurements](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md#results).

### Checkpoints, KV scales and MTP drafts

See [Checkpoints and KV formats](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md).

### DFlash2 speculative decoding (optional)

See [DFlash2 speculative decoding](https://github.com/Leonccaa/ReVolta/blob/main/docs/dflash2.md).

### KV offload modes

See [KV offload modes](https://github.com/Leonccaa/ReVolta/blob/main/docs/kv-offload.md).

### Weight and KV formats

See [Weight and KV formats](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md#weight-and-kv-formats).

### Standard NVFP4

See [Standard NVFP4](https://github.com/Leonccaa/ReVolta/blob/main/docs/checkpoints.md#standard-nvfp4).

### Performance

See [Performance and measurements](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md).

### Quality measurements

See [Quality measurements](https://github.com/Leonccaa/ReVolta/blob/main/docs/performance.md#quality-measurements).

### Support and known limits

See [Support and known limits](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md).

### Support matrix

See [Support matrix](https://github.com/Leonccaa/ReVolta/blob/main/docs/known-limits.md#support-matrix).

### Reporting issues

See [Getting help](#getting-help).
