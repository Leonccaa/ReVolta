# ReVolta support scope

The [support matrix](../docs/known-limits.md#support-matrix) in
[Support and known limits](../docs/known-limits.md) is the reference.
In short:

| Area | Scope |
| --- | --- |
| GPUs | Four or more NVIDIA V100 32 GB (PCIe or SXM2); validated on four |
| PCIe switch with P2P | Recommended; validated |
| PCIe without P2P | Validated by disabling P2P on P2P-capable hardware; about 30–55% of P2P speed |
| NVLink | Recommended; tested with community results on 4 × V100-SXM2 32 GB ([#2](https://github.com/Leonccaa/ReVolta/issues/2), [#38](https://github.com/Leonccaa/ReVolta/issues/38), [#37](https://github.com/Leonccaa/ReVolta/issues/37)) |
| Host memory | 128 GiB recommended; 80 GiB measured minimum for the default pinned PLE table with KV offload off; the experimental whole-table PLE disk option ran under 64 GiB and 32 GiB limits |
| KV offload | Off, CPU, or CPU + SSD tier |
| KV cache | E4M3 (default) or INT8; INT8 with the SSD tier is not supported |
| Models | Qwen3.8 Flash-Next (QSA, multimodal) |
| Checkpoints | AWQ g32, standard NVFP4 `RadixArk/Qwen3.8-Flash-Next-NVFP4`, `nvidia/Qwen3.8-Flash-Next-FP4-Local` |

Configurations outside our release gate (TP2/TP8, V100 16 GB, other models or
topologies) are handled on a best-effort basis. Other GPU architectures,
ROCm/CPU/XPU/TPU serving and GGUF are not built.

Include the complete launch command and environment in every report. Never
include credentials or private deployment details. Report security
vulnerabilities through GitHub private vulnerability reporting.
