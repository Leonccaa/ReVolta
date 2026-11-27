# ReVolta support scope

This page says what the maintainers support in ReVolta and where to ask for help. The [support matrix](../docs/known-limits.md#support-matrix) in [Support and known limits](../docs/known-limits.md) is the reference; the table below summarizes it.

| Area | Supported |
| --- | --- |
| Version | 0.2.0. 0.1.x gets no further releases; reproduce a problem on 0.2.0 before reporting it. <!-- doc020: proposed 0.1.x policy, awaiting the maintainer's decision --> |
| GPUs | Four or more NVIDIA V100 32 GB (PCIe or SXM2); validated on four |
| Interconnect | Recommended and tested: NVLink, or all GPUs under one PCIe switch with P2P. PCIe without P2P gave correct output at about 30–55% of the P2P throughput in the 0.1.0 release test, which disabled P2P on the reference machine |
| Parallelism | Tensor parallel 4 with decode context parallel 2 (TP4 + DCP2) |
| Host memory | 128 GiB recommended; 80 GiB is the smallest tested limit with the default PLE table in host memory and KV offload off; 32 GiB was tested in 0.1.4 with the experimental whole-table PLE disk option |
| Model | Qwen3.8 Flash-Next: text, images and video |
| Checkpoints | Uncensored AWQ g32 (recommended), Base AWQ g32, Standard NVFP4 (`RadixArk/Qwen3.8-Flash-Next-NVFP4`), `nvidia/Qwen3.8-Flash-Next-FP4-Local` and `nvidia/Qwen3.8-Flash-Next-NVFP4-QAD`; test status per checkpoint is in [Checkpoints](../docs/checkpoints.md) |
| Speculative decoding | MTP (the default, with dynamic draft widths) or DFlash2 |
| KV cache format | INT8 (the default, `--kv-cache-dtype auto`); FP16 and per-token-head E4M3 also work; calibrated scalar E4M3 is frozen and gets only a functional check |
| KV placement | GPU KV, optionally with CPU and disk (`revolta_fs`) prefix offload; or Host KV, which keeps the main KV cache in host memory |

Best effort: TP2/TP8, V100 16 GB, other GPU counts, other models and other topologies are outside the release gate. Not built: GPU architectures other than SM70, ROCm, CPU, XPU and TPU serving, and GGUF.

Hy4-flash (text only, exactly four V100 32 GB GPUs) has validation support that was added in 0.2.0rc13 and frozen there; 0.2.0 has not tested it again. See [Hy4-flash](../docs/hy4-flash.md).

## Asking for help

Open a [GitHub issue](https://github.com/Leonccaa/ReVolta/issues) with the bug report form. Include the ReVolta version, the complete launch command and environment, the model revision, host or container memory limits, `nvidia-smi topo -m`, and the `ReVolta profile` and `SM70 interconnect=` lines from the startup log. Never include credentials or private deployment details.

Report security vulnerabilities through GitHub private vulnerability reporting; see [SECURITY.md](../SECURITY.md).

## See also

- [Support and known limits](../docs/known-limits.md)
- [Getting started](../docs/getting-started.md)
- [README: Getting help](../README.md#getting-help)
