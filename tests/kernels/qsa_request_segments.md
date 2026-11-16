# Request-segmented QSA scorer

`vllm/models/qwen4_exp/nvidia/ops/qsa_request_segments.py` scores multi-request
SM70 QSA batches. The paged scorer (`qsa.qsa_mqa_paged`) loads every 64-key
tile once per query row. Rows of one request share a page table, so this scorer
splits each contiguous request run into segments of up to four rows and loads
each tile once per segment. Four rows by four heads form one 128x16 query block.
The kernel keeps the paged kernel's 64x128 by 128x16 dot, 2 warps, pipelined
tile loop and per-row 16-lane reduction (the other rows' lanes are zeroed). On
SM70 the scores and visible lengths are bitwise equal to the paged scorer.

## Runtime use

`qsa.qsa_select_paged_tokens` uses this scorer in place of `qsa_mqa_paged` when
`VLLM_SM70_QSA_REQUEST_SEGMENTS=1` (the ReVolta profile sets it; an explicit
value wins) and `request_segments_unsupported` returns `None`:

- two or more requests (`single_request` keeps the shared-key and cuBLAS paths);
- at most 256 rows in the logits chunk;
- at least two rows per request on average (`one_row_per_request`: MTP drafting
  steps, including CUDA-graph padding, gain nothing from one-row segments);
- FP16 Q and indexer K with H4/D128, int32 page table, request map and lengths,
  int32 or int64 positions, positive strides.

The variable is not a torch.compile factor: the route is chosen inside the QSA
custom op at run time. A device builder writes a fixed `[M,3]` descriptor buffer
on every call (unused slots hold -1 and exit), so there is no host readback and
the call can be captured in a CUDA graph. Buffers are allocated per call, as in
the paged scorer.

The segment limit is four rows. Eight-row segments (two query blocks), query
blocks loaded inside the tile loop, a data-dependent branch in the loop body, or
more warps either spill the key tile at 2 warps or change the reduction order.

## CPU checks

From the checkout root, in an environment with PyTorch and Triton:

```bash
CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 \
  .venv/bin/python tests/kernels/test_qsa_request_segments_cpu.py
.venv/bin/python -m pytest tests/kernels/test_qsa_request_segments_dispatch.py
```

The first runs the builder and scorer in Triton's CPU interpreter against the
paged kernel of this checkout and a NumPy oracle (scores, visible lengths,
stable top-512 IDs, untouched padding, descriptors; remapped requests, inactive
rows, invalid pages, negative positions, ragged and repeated runs, strides, page
sizes 16/204/416). The interpreter does not establish SM70 rounding, launch
safety or CUDA-graph behavior. The second checks when the runtime uses the
scorer.

## SM70 checks

On one SM70 GPU, with the native exact top-k extension installed:

```bash
.venv/bin/python tests/kernels/qsa_request_segments_gpu.py --out kernel.json
.venv/bin/python tests/kernels/qsa_request_segments_select_gpu.py --out select.json --time
```

`qsa_request_segments_gpu.py` runs 36 kernel cases (C1/M8, C4/M16, C4/M20,
C8/M40 at 32768/131072/262144 tokens, segment limits 1/2/4) against the paged
scorer, eagerly and through CUDA graphs replayed after the queries, keys, page
tables, positions, lengths and request runs change. FP32 score words, visible
lengths and exact top-512 IDs must match; guard storage surrounds the buffers.

`qsa_request_segments_select_gpu.py` runs `qsa_select_paged_tokens` with the
variable off and on and compares the selected token indices: decode batches,
one row per request, ragged rows, a prefill chunk next to decode rows (with and
without the host request split), two logits chunks and CUDA-graph padding rows.
It also checks which batches use the scorer. `--time` adds the graph replay
time of the whole selection (scoring, top-512, expansion).

Measured on a V100-PCIE-32GB at 250 W, whole selection per layer, new / paged:
4 requests x 4 rows 0.39-0.40, 4 x 5 rows 0.47-0.50, 8 x 4 rows 0.32-0.36,
8 x 5 rows 0.49-0.55 (32K-256K tokens); one row per request 0.97-1.15, which is
why those batches keep the paged scorer.
