# QSA FP8 E4M3 KV scales: Qwen3.8-Flash-Next-FP4-Local

Calibrated FP32 K/V scales for
[`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`: 24 scales (2,640 bytes)
for the main QSA KV cache and 2 scales (296 bytes) for the checkpoint's built-in
MTP draft layer, plus a tool that builds a loadable checkpoint view from them.
The same payload is published on Hugging Face as
[`leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales)
(tag `v1.1-base-b9ce18a6`; `v1.0-base-b9ce18a6` has the 24 target scales only).
The model card there describes the calibration and the validation results.

| File | Purpose |
|---|---|
| `model-kvscales.safetensors` | 24 FP32 scalars: `k_scale` and `v_scale` for QSA layers 3, 7, ..., 47 |
| `kvscales-manifest.json` | Base revision and hashes, tensor names and values, calibration statistics |
| `model-kvscales-mtp.safetensors` | 2 FP32 scalars: `mtp.layers.0.self_attn.k_scale` and `v_scale` for the MTP draft layer |
| `draft-kvscales-manifest.json` | Draft scale values, calibration statistics and validation summary |
| `materialize_overlay.py` | Builds the checkpoint view; Python 3.9+ standard library only |
| `SHA256SUMS` | Checksums of the five files above |

## Usage

Check the files, then download the exact base revision:

```bash
sha256sum -c SHA256SUMS
hf download nvidia/Qwen3.8-Flash-Next-FP4-Local \
  --revision b9ce18a6887e18c089fdea8509a33e63429a3300 \
  --local-dir /path/to/base-checkpoint
```

Build the view. `--pack-dir` defaults to the directory of the script.

```bash
python materialize_overlay.py \
  --base-checkpoint /path/to/base-checkpoint \
  --output-dir /path/to/qsa-e4m3-checkpoint
```

The tool refuses to run if the base `model.safetensors.index.json` or
`config.json` hash, the scale file hash, or any tensor name or value differs
from the manifests. The view contains the merged index with 26 scale entries,
copies of both scale files under `calibration/`, a `kvscales-provenance.json`,
and symlinks to the unchanged base files. Keep the base directory in place.

Serve the view with a V100 (SM70) vLLM build with QSA E4M3 KV support, using
`--dtype half` and `--kv-cache-dtype fp8_e4m3`. Set
`VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1` so that the server refuses to start
unless all 24 target scales are loaded. The MTP draft (no `model` field in
`--speculative-config`) loads its two scales from the same view and can use an
E4M3 KV cache (`"kv_cache_dtype": "fp8_e4m3"`); an FP16 draft cache
(`"float16"`) also works.

## License

`materialize_overlay.py` is Apache-2.0. The scale file is a derived artifact of
the base checkpoint and follows its terms: the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
and the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
for the underlying Qwen3.8-Flash-Next model.
