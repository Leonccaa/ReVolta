# QSA FP8 E4M3 KV scales: Qwen3.8-Flash-Next-FP4-Local

Calibrated FP32 K/V scales (24 tensors, 2,640 bytes) for the main QSA KV cache
of [`nvidia/Qwen3.8-Flash-Next-FP4-Local`](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-FP4-Local)
at revision `b9ce18a6887e18c089fdea8509a33e63429a3300`, and a tool that builds a
loadable checkpoint view from them. The same payload is published on Hugging
Face as
[`leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales`](https://huggingface.co/leoncca/Qwen3.8-Flash-Next-FP4-Local-QSA-FP8-E4M3-KV-Scales)
(tag `v1.0-base-b9ce18a6`). The model card there describes the calibration and
the validation results.

| File | Purpose |
|---|---|
| `model-kvscales.safetensors` | 24 FP32 scalars: `k_scale` and `v_scale` for QSA layers 3, 7, ..., 47 |
| `kvscales-manifest.json` | Base revision and hashes, tensor names and values, calibration statistics |
| `materialize_overlay.py` | Builds the checkpoint view; Python 3.9+ standard library only |
| `SHA256SUMS` | Checksums of the three files above |

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
from the manifest. The view contains the merged index, a copy of the scale file
at `calibration/model-kvscales.safetensors`, a `kvscales-provenance.json`, and
symlinks to the unchanged base files. Keep the base directory in place.

Serve the view with a V100 (SM70) vLLM build with QSA E4M3 KV support, using
`--dtype half` and `--kv-cache-dtype fp8_e4m3`. Set
`VLLM_QWEN4EXP_QSA_E4M3_STRICT_SCALES=1` so that the server refuses to start
unless all 24 scales are loaded. The pack has no scales for the MTP layer, so
keep the draft KV cache in FP16 (`"kv_cache_dtype": "float16"` in
`--speculative-config`).

## License

`materialize_overlay.py` is Apache-2.0. The scale file is a derived artifact of
the base checkpoint and follows its terms: the
[NVIDIA Open Model Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-agreement/),
and the [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE)
for the underlying Qwen3.8-Flash-Next model.
