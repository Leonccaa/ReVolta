# ReVolta support scope

This page gives the supported version and where to ask for help. What is supported and tested (hardware, interconnect, checkpoints, speculative decoding and KV modes) is listed in the [support matrix](../docs/known-limits.md#support-matrix).

## Supported version

0.2.0 is the supported release. 0.1.x gets no further releases; reproduce a problem on 0.2.0 before reporting it. Configurations outside the support matrix get best-effort help only.

## Asking for help

Open a [GitHub issue](https://github.com/Leonccaa/ReVolta/issues) with the bug report form, or the question form for usage questions. Include the ReVolta version, the complete launch command and environment, the model revision, host or container memory limits, `nvidia-smi topo -m`, and the `ReVolta profile` and `SM70 interconnect=` lines from the startup log. Never include credentials or private deployment details.

Report security vulnerabilities through GitHub private vulnerability reporting; see [SECURITY.md](../SECURITY.md).

## See also

- [Support and known limits](../docs/known-limits.md)
- [Getting started](../docs/getting-started.md)
- [README: Getting help](../README.md#getting-help)
