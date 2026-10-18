# Third-party license evidence

The `.txt` files are verbatim upstream license or author texts. Their immutable
source URLs, revisions and SHA-256 digests are recorded in
[SOURCES.json](SOURCES.json). They supplement the existing component licenses
and per-file notices; they do not replace those notices.

| Component | License text | Source reference |
| --- | --- | --- |
| LMDeploy / TurboMind | `lmdeploy-dated.txt` (Apache-2.0) | `InternLM/lmdeploy@9b3b1933ee2ad1731a6e49b99867b52925b2a54b` |
| FlashAttention-V100 | `flash-attention-v100.txt` (BSD-3-Clause) | `ai-bond/flash-attention-v100@0ababf82e3836c1786d7937c6002670ca4ecda62` |
| SM70 FA2 | `flash-attention-sm70.txt`, `flash-attention-sm70-authors.txt` (BSD-3-Clause) | Build pin `zhinianqin/flash-attention-v100@c2eda5e6115b98c3ba4bfd181570668742eece22` |
| CUTLASS | `cutlass.txt` (BSD-3-Clause and embedded third-party notices) | v4.4.2, `da5e086dab31d63815acafdac9a9c5893b1c69e2` |
| CUTLASS, Machete source | `cutlass-machete.txt` (BSD-3-Clause) | `06b21349bcf6ddf6a1686a47a137ad1446579db9` from the source header |
| CUTLASS, SM70/SM75 FA2 submodule | `cutlass-fa2.txt` (BSD-3-Clause) | `62750a2b75c802660e4894434dc55e839f322277` |
| Marlin | `marlin.txt` (Apache-2.0) | `IST-DASLab/marlin@1f25790bdd49fba53106164a24666dade68d7c90` |
| FlashQLA | `flashqla-dated.txt` (MIT) | `QwenLM/FlashQLA@5be4204363d1f751ba586b0972eab58b818c852c` |
| SGLang CPU kernels | `sglang-kernel.txt` (Apache-2.0), `sglang-notices.txt` | `sgl-project/sglang@a9a8b20a90df3c8b64f7308abde7d3d1fa1a5d12`, component-level `sgl-kernel/` notices |
| Flash Linear Attention | `fla-dated.txt` (MIT) | `fla-org/flash-linear-attention@8a91ac503764fed5bb65ad6d9b157880eeff2820` |
| Mamba | `mamba.txt` (Apache-2.0) | v2.2.4, `95d8aba8a8c75aedcaa6143713b11e745e7cd0d9`; the license is identical at the separately cited layernorm revision `60dadf2e0ee730ac337035d5533de10bc26e4847` |
| causal-conv1d | `causal-conv1d.txt`, `causal-conv1d-authors.txt` (BSD-3-Clause) | v1.5.0, `fe772c75e29da8123d4478b670342111ce23cc6b` |
| vLLM FlashAttention | `vllm-flash-attention.txt`, `vllm-flash-attention-authors.txt` (BSD-3-Clause) | Build pin `bce29425653ec0fbc579d329883030e832d15ada` |
| SM75 FlashAttention | `flash-attention-sm75.txt`, `flash-attention-sm75-authors.txt` (BSD-3-Clause) | Build pin `Peuqui/flash-attention@43b9d29c9aa8e18d9351e7c643dc78ef7e7979fc` |
| FlashMLA | `flashmla.txt` (MIT) | Build pin `vllm-project/FlashMLA@a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b` |
| DeepGEMM | `deepgemm.txt` (MIT) | Build pin `deepseek-ai/DeepGEMM@891d57b4db1071624b5c8fa0d1e51cb317fa709f` |
| Triton kernels | `triton.txt` (MIT) | Build tag v3.5.1, `triton-lang/triton@0add68262ab0a2e33b84524346cb27cbb2787356` |
| QuTLASS | `qutlass.txt` (Apache-2.0) | Build pin `IST-DASLab/qutlass@830d2c4537c7396e14a02a46fbddd18b5d107c65` |

An evidence snapshot is not an assertion that every modified file was imported
from that revision. [NOTICE](../NOTICE) records the remaining source-history
gaps and the retained code outside the supported build target. In particular,
SGLang's full component notice is preserved conservatively; not every component
named in it is present in the copied CPU subset.

Keep these texts in source and binary distributions. `pyproject.toml` selects
this directory as license data, and `MANIFEST.in` includes it in source archives.
Builds using source-directory overrides or different dependency revisions must
recheck the corresponding licenses, authors and transitive notices.
