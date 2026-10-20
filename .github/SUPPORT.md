# ReVolta 0.1.0 support scope

The [support matrix in the README](../README.md#support-matrix) is the reference.
In short:

| Area | Scope |
| --- | --- |
| GPUs | 4 × NVIDIA V100 32 GB PCIe |
| PCIe switch with P2P | Validated |
| PCIe without P2P | Validated by disabling P2P on P2P-capable hardware; about 30–55% of P2P speed |
| NVLink | Unverified (no hardware available); the upstream topology path and `VLLM_SM70_INTERCONNECT` override exist |
| Host memory | 128 GiB recommended; 80 GiB measured minimum with KV offload off |
| KV offload | Off, CPU, or CPU + SSD tier |
| KV cache | E4M3 (default) or INT8; INT8 with the SSD tier is not supported |
| Models | Qwen3.8 Flash-Next (QSA, multimodal) |
| Checkpoints | AWQ g32, standard NVFP4 `RadixArk/Qwen3.8-Flash-Next-NVFP4`, `nvidia/Qwen3.8-Flash-Next-FP4-Local` |

Unverified configurations (NVLink, TP2/TP8, V100 16 GB, other models or
topologies) are handled on a best-effort basis. Other GPU architectures,
ROCm/CPU/XPU/TPU serving and GGUF are not built.

Include the complete launch command and environment in every report. Never
include credentials or private deployment details. Report security
vulnerabilities through GitHub private vulnerability reporting.
