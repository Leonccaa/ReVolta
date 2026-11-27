# QSA scalar E4M3 KV calibration

This directory holds offline tools that calibrate per-layer K/V scales for the scalar E4M3 KV cache (`--kv-cache-dtype fp8` or `fp8_e4m3`) of Qwen4Exp-architecture checkpoints, such as Qwen3.8 Flash-Next, and build checkpoint overlays from them. You need them only to produce scales for a checkpoint that has no published scale pack.

Scalar E4M3 KV is a frozen format: it keeps working, but it is no longer optimized, and release acceptance gives it only a functional check. The default on V100, `--kv-cache-dtype auto` (planar INT8), and `fp8_per_token_head` compute their scales at run time and need no calibration. Published scale packs for the supported checkpoints are listed in [Scalar E4M3 KV](../../docs/checkpoints.md#scalar-e4m3-kv-frozen).

## Calibration contract

- SM70, tensor parallel 4 by default, FP16 activations and FP16 KV during collection, no speculative decoding.
- Per-layer, per-tensor E4M3 K and V scales for the main QSA KV cache: 12 QSA layers by default (`--expected-layers`), so 24 scales. The raw and compressed indexer caches stay FP16.

## `qsa_kv_calibration.py`

| Subcommand | What it does |
| --- | --- |
| `prepare` | Freezes source records (`--input-jsonl`) into exact prompt token IDs (`--output-jsonl`) for `--model`. For an audited corpus, pass both `--corpus-manifest` and `--processor-audit-jsonl`; every processor tensor hash must match before any output is written. |
| `collect` | Runs the checkpoint through the offline engine with FP16 KV and records K/V statistics in `--output-dir`. Pass `--input-jsonl` with `--corpus-shard`, or `--shard-manifest` to collect several named shards in one engine lifetime. The `COLLECTING` marker exists only while prompts run, so engine initialization, memory profiling and graph-capture forwards are excluded. Defaults: tensor parallel 4, 2,048 batched tokens, 32 sequences, GPU memory utilization 0.75; the collection manifest records the values used. |
| `summarize` | Requires all expected QSA layers and writes independent K/V max-abs, scale, percentile, saturation and per-shard stability data. `--exclude-corpus-shard` drops a shard. |
| `envelope` | Keeps the distribution statistics of `--base-report` and takes the observed maximum across the `--scale-report` files. Use it when nested runs give slightly different extrema because batching changes the floating-point order. |
| `overlay` | Creates a derived checkpoint directory in `--output-dir`: writes `model-kvscales.safetensors`, merges its 24 entries into `model.safetensors.index.json`, and links the other base files. It never modifies the base checkpoint. `--unit-scale-negative-control` writes unit scales, for use only as a negative control. |
| `compare-targets` | Compares the target-layer scales of two reports, or of a report and a published `kvscales-manifest.json`, and excludes the MTP layer (`--mtp-layer-id`, default 48). |

Do not promote a smoke-only or incomplete-corpus report into a production overlay.

## `materialize_qsa_scale_overlay.py`

Builds a revision-checked view from a published scale pack (`--pack-dir`, which holds `kvscales-manifest.json` and the target scale file) and adds the two MTP draft scales. The tool refuses to run if the base `model.safetensors.index.json` or the scale file does not match the manifest, or if the base already contains scale tensors.

- `--base-checkpoint`, `--output-dir`, `--pack-dir` (required).
- `--mtp-report`: a report that contains the MTP layer (`--mtp-layer-id`, default 48); its two scales are written to `model-bf16-kvscales-mtp.safetensors`.
- `--target-scale-source envelope --target-report <report>`: raise a published target scale where the report observed a larger maximum. The default, `published`, copies the published target scales.
- `--mtp-scale-from-layer` and `--mtp-unit-scale` produce prototype draft scales and mark them as such in `kvscales-provenance.json`.

The view contains the merged index, both scale files, `kvscales-provenance.json` and symlinks to the unchanged base files.

## See also

- [Checkpoints, KV scales and MTP drafts](../../docs/checkpoints.md)
- [FP4-Local scale pack](../kv-scales/fp4-local/README.md)
- [Configuration reference](../../docs/configuration.md#kv-cache-format): KV cache formats.
