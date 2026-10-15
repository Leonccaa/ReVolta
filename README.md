# ReVolta

ReVolta is Leon's privately maintained inference engine for **NVIDIA Volta /
Tesla V100 (SM70)**. It is derived from [1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM),
itself a fork of [vLLM](https://github.com/vllm-project/vllm).

The first deployment target is **Qwen3.8 Flash-Next multimodal serving on four
32 GB PCIe V100s behind a single PCIe switch, without NVLink**. ReVolta keeps
hardware and model specific changes for that deployment and selectively takes
upstream fixes. The repository is private; a public release is deferred.

## Current scope

The validated deployment baseline, as of 2026-10-15, includes:

- TP4 with DCP2 and four-token MTP drafting.
- SM70 AWQ group-size-32 routed experts, with FP16 dense layers.
- E4M3 KV storage, CPU KV offload, long-context serving and multimodal inputs.
- V100 attention and MoE kernels, CUDA Graph execution, and an OpenAI compatible
  serving API inherited from vLLM.
- A narrow SM70 TP4 custom all-reduce route when all four GPUs have peer access
  and NVML reports a shared PCIe-switch topology (PIX/PXB). This route is limited
  to messages of at most 80 KiB and execution inside CUDA Graphs.

These statements describe the accepted deployment configuration. They do not
establish support for every checkpoint, shape, sampling mode or upstream feature.
Repository trimming does not revalidate runtime performance or model quality.

## Support matrix

| Configuration | ReVolta status |
| --- | --- |
| 4 × V100 32 GB, PCIe, single switch, P2P available, TP4 | Validated for the target deployment |
| Qwen3.8 Flash-Next, multimodal, AWQ g32 + FP16 dense layers | Validated for the target checkpoint and configuration |
| TP4 + DCP2, MTP4, E4M3 KV, CUDA Graphs and CPU KV offload | Validated together in the target deployment |
| NVLink, including hierarchical TP8 communication | **Unverified; follows the upstream path** |
| TP2 or TP8 | **Unverified; follows the upstream path** |
| Other models or checkpoints | **Unverified; follows the upstream path** |
| Other PCIe topologies, cross-socket links or missing P2P | **Unverified; follows the upstream path** |
| Other NVIDIA architectures | Unverified; outside the ReVolta build target |
| ROCm, CPU, XPU or TPU serving | Not built or released by ReVolta |

"Upstream path" means the inherited implementation remains in the source. It is
not a claim that the configuration works. Models and quantization implementations
remain in shared source files to keep selective upstream fixes manageable; their
presence does not add them to this support matrix.

## Build for SM70

The established native build uses Linux x86-64, Docker BuildKit, CUDA **12.8.1**,
Python **3.12** and PyTorch **2.10**. CUDA 12.8 is intentional for Volta. Compile
only architecture **7.0**, with one nvcc thread per job. A GPU is not required
to compile the wheel; runtime acceptance needs the target V100 hardware.

From a clean checkout, build the wheel stage:

```bash
REV=$(git rev-parse --short=10 HEAD)
docker buildx build -f docker/Dockerfile --target build \
  --build-arg CUDA_VERSION=12.8.1 \
  --build-arg torch_cuda_arch_list=7.0 \
  --build-arg nvcc_threads=1 \
  --build-arg max_jobs=10 \
  --build-arg SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0+$REV" \
  --tag revolta-sm70-build:local --load .
```

Ten jobs previously used about 21 GB of host RAM; reduce `max_jobs` on smaller
builders. The wheel stage puts the engine wheel in `/workspace/dist` and the
FlashAttention-V100 wheel in `/workspace/flash-attn-v100-dist`. Export both from
the local build image, and use a matching Python/PyTorch/CUDA environment.
The SM70 recipe skips the newer-GPU DeepEP extension build.

The Python import package and CLI remain `vllm`; the distribution name remains
`1cat-vllm` for compatibility. Production uses a Python overlay from this source
with matching, already-built native wheels. Python-only changes can use that
overlay; native changes require a rebuild and acceptance testing. Do not mix
native extensions from unrelated revisions.

`setup.py` and CMake retain upstream platform branches, but standalone build
recipes and requirements for ROCm, CPU, XPU and TPU have been removed. Those
platforms are not supported packaging targets in this checkout. Automated CI
and publication pipelines are removed. A small set of test-dependent benchmark
helpers and example templates remains in the tree.

## Maintenance

ReVolta does not track upstream main through regular merges. Review 1Cat-vLLM
and relevant vLLM changes approximately every two weeks, then selectively take
fixes that affect the paths this deployment actually uses. Keep attribution and
history, and validate each candidate against the accepted configuration before
deployment.

For local checks, create a Python 3.12 environment with `uv`, install
`requirements/lint.txt`, and run `pre-commit run --all-files` or run it on staged
files. Environment registration and metadata checks remain active without a
local documentation tree. An environment reference can be generated on demand:

```bash
.venv/bin/python -m tools.generate_env_reference --output environment-reference.md
```

## License and acknowledgements

ReVolta's inherited engine code and project changes use
[Apache License 2.0](LICENSE), except where third-party files specify another
license. See [NOTICE](NOTICE) for repository evidence, component attributions
and items that still need verification. Model weights have separate licenses.

Thanks to the vLLM and 1Cat-vLLM contributors and the authors of the included
attention, quantization, communication and GPU kernel components. NVIDIA,
Volta and Tesla are referenced to identify the hardware; this project is not
affiliated with or endorsed by NVIDIA.

## 中文说明

ReVolta 是 Leon 自维护的 V100（SM70）推理引擎，基于 1Cat-vLLM / vLLM。
目前为私有项目，首个目标是单个 PCIe 交换芯片下的四张 V100 上运行
Qwen3.8 Flash-Next 多模态服务。只构建 SM70；NVLink、TP2/TP8 和其它模型均未验证，
沿用上游路径。约每两周复查上游，只挑相关修复。许可与待核实的第三方条目见 `NOTICE`。
