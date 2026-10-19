# ReVolta 0.1.0 support scope

Issue support follows the support table in roadmap W10. This public copy states
the release scope; validation status and release instructions belong in the
[README](../README.md). A support target does not imply that release acceptance
has already completed.

| Area | Scope |
| --- | --- |
| GPUs | Four NVIDIA V100 GPUs |
| PCIe with a PEX switch and P2P | Supported |
| PCIe without P2P | Supported; validation uses P2P-disabled simulation |
| NVLink | Upstream topology path and an explicit override are intended; hardware validation is unavailable, so this configuration is unverified |
| Host memory | 128 GB recommended, not required; the minimum with KV offload disabled still needs measurement |
| KV offload | Disabled, CPU, or CPU plus SSD |
| Models | QSA models; acceptance currently covers only Qwen3.8 Flash-Next |
| Checkpoints | AWQ g32; `nvidia/Qwen3.8-Flash-Next-FP4-Local`; standard NVFP4 `RadixArk/Qwen3.8-Flash-Next-NVFP4` |

Other hardware, models and inherited vLLM features are outside the 0.1.0 support
commitment. Include the complete environment even when reporting an issue
outside this scope. Never include credentials or private deployment details.
