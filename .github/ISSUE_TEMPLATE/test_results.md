---
name: Report
about: Share test results or experience from running ReVolta on your hardware. People and coding agents can both fill this in.
title: "[Report] "
labels: ["report"]
---

<!--
Instructions (for people and for coding agents):
- Fill in every section. Write "not measured" instead of deleting a section.
- Measured results and experience from daily use are both welcome. For daily use,
  describe the workload under "How you measured" and what happened under
  "Stability and notes".
- Keep the keys in the Summary block; put units in the values.
- Remove credentials, tokens and private paths from commands and logs.
- For a problem or a crash, use the bug report form instead.
- To compare with the release tables, see the method in docs/performance.md
  ("Reference system, MTP4"). Any method is welcome; describe the one you used.
- If you open this issue through the API or `gh issue create`, the label may not be
  applied automatically; the maintainers add it.
-->

## Summary

```text
revolta_version:
install:                  # PyPI wheel, Docker image tag, or source
gpus:                     # model x count, memory per GPU
power_limit_w:
interconnect:             # PCIe switch with P2P, PCIe without P2P, NVLink, or other
host_memory:              # host RAM, container limit, /dev/shm size
checkpoint:               # owner/repository @ revision
drafter:                  # owner/repository @ revision, or none
speculative_method:       # MTP, DFlash2, or none
draft_sample_method:
kv_cache_dtype:
cpu_kv_offload_gib:
max_model_len:
gpu_memory_utilization:
```

## Topology

<!-- Output of `nvidia-smi topo -m`. -->

```text

```

## Launch command and environment

<!-- The complete command, including --speculative-config and --kv-transfer-config,
and every VLLM_*, NCCL_* and PYTORCH_* variable you set. -->

```bash

```

## Startup log lines

<!-- The `ReVolta profile {...}` line, the `SM70 interconnect=` lines and the
`GPU KV cache size` line. -->

```text

```

## How you measured

<!-- Prompts or prompt types, response length, thinking on or off, temperature,
concurrency levels, repetitions, and whether rates include the time to first token. -->

## Results

<!-- Tables are welcome. Give the unit of every number. -->

## Stability and notes

<!-- Errors, out-of-memory events, restarts, and any deviation from the README. -->
