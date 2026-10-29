# Paged QSA scorer CPU study

This standalone NumPy study compares two scheduling changes for small
multi-request FP16 QSA verification batches. It prepares input fixtures and
comparison gates before GPU implementation. It does not change runtime dispatch
or import vLLM, PyTorch, Triton, CUDA, or model weights.

## Run the CPU checks

Use a project virtual environment. From the repository root:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python numpy==2.2.6
.venv/bin/python -m unittest tools.qwen4_exp.qsa_scorer_study.test_cpu -v
.venv/bin/python -m tools.qwen4_exp.qsa_scorer_study.run check \
  --suite quick --export qsa-results/inputs --output qsa-results/quick.json
.venv/bin/python -m tools.qwen4_exp.qsa_scorer_study.run check \
  --suite long --output qsa-results/long.json
.venv/bin/python -m tools.qwen4_exp.qsa_scorer_study.run matrix \
  --output qsa-results/gpu-matrix.json
```

`quick` covers ragged and repeated request runs, padding, negative positions,
invalid pages, strided tensors, ties, truncated widths, empty visibility, page
tails, and 16/20/35/40/64-row batches. `long` adds C4 at 16/20 rows and C8 at
40 rows, each at 32768/131072/262144 token contexts. `full` combines them.
`--case NAME` selects one case. CPU wall time is not a GPU performance metric.

## Algorithms and comparison contract

- `row_reference` gathers each query row independently and computes every head
  with explicit FP32 K-dimension accumulation. Head reduction is left-to-right.
- `tile_grouped` maps one, four, or eight existing 64-key tiles to one program.
  Each row retains its own request, visible length, and output coordinates.
  Grouping changes ownership and query reuse, not the CPU arithmetic contract.
- `request_segmented` splits contiguous runs into at most eight rows, reads keys
  once per segment, and writes scores back to their original query rows. Each
  row keeps its own causal limit. Repeated runs remain separate.

The reference has independent gather and score code. All variants must match
FP32 score words, visible lengths, one-write coverage, untouched padding, and
exact top-512 indices. Selection uses score descending/index ascending ties and
returns selected indices in ascending order, with -1 padding for short rows.
Invalid physical pages have -inf scores within the visible prefix.

The initial arithmetic scope is finite FP16 Q/K, four heads, dimension 128,
positive finite divisor, integer metadata, and score width within page-table
capacity. Arbitrary positive input strides and negative padded positions are
included. CUDA output beyond the visible prefix is unspecified; CPU poison
checks are stricter ownership tests. Zero-column device visibility is also
unspecified and is not a GPU comparison case.

These are CPU scheduling prototypes, not bitwise emulators of SM70 MMA or
Triton. In particular, the existing single-request shared-key test permits
`rtol=2e-6, atol=2e-6` for score comparison. Passing that test does not satisfy
the proposed multi-request bitwise gate. Keep K/head reduction and MMA choices
as separate numerical questions during GPU implementation.

## Launch and memory accounting

The shipped paged path uses one tile/program for M<=32 and eight for M>32.
Thus tile8 is already the paged baseline at M40; it is not a new optimization
there. Single-request M2..8 can instead use the existing shared-key operator
when enabled. Record the actual route before comparing a candidate.

For a synthetic 16-row, 131456-column envelope with visibility <=8293, tile1
launches 32864 programs, of which at least 30784 are empty. Tile4 launches 8224;
tile8 launches 4112. These counts prove neither speedup nor reduced HBM traffic.
131456 is a rounded grid envelope, not a universal deployment cache capacity.

The segmented prototype allocates a fixed `[M,3]` int32 descriptor buffer and
models a grid with M slots. Empty descriptor slots still launch programs. The
geometry report also shows the smaller *compact ideal* count. Achieving that
count requires a safe segment-capacity bound from existing CPU scheduler
metadata; reading a segment count back from the GPU is not an acceptable plan.
Include descriptor generation, transfer if any, empty slots, and extra launches
in the full scorer cost. Descriptor storage is 12*M bytes, before count/scan
workspace. Logical key-fetch counts are not measured memory transactions.

## Deferred GPU validation

The matrix contains 12 scorer-only shapes: C1/M8, C4/M16, C4/M20, C8/M40 at
32K/128K/256K. Long-context synthetic scorer inputs do not establish that an
inference engine can admit the corresponding number of requests. GPU candidate
kernels and runtime integration are not supplied by this CPU study.

1. Fix one source revision, FP16 inputs, compression, page layout, GPU power,
   clocks and capture mode. Preserve shipped dispatch as one control, and call
   the paged kernel with tile1/tile4/tile8 as explicit controls. Hold selection
   unchanged while evaluating the scorer.
2. Capture the paged baseline and each candidate on the same inputs. Require
   bitwise visible-prefix scores, visible lengths, and top-512 IDs. Check page
   isolation, invalid pages, ragged runs, ties and partial tiles first. Any
   numerical relaxation needs a separately approved error and quality contract.
3. Replay one fixed-capacity CUDA graph after changing queries, cache contents,
   page tables, request IDs, positions, lengths and active descriptor count.
   Check for new allocations, host readback, stale descriptors and out-of-bounds
   writes. Include the metadata builder in graph replay and scorer timing.
4. Measure scorer-only and scorer+selection, including workspace and all extra
   launches. Alternate baseline/candidate measurements in the same window and
   report distributions. Do not derive speed from CPU time or kernel counts.
5. Only then compare model decode on valid C4/C8 windows, offload and long
   contexts. Verify actual concurrency from per-step scheduler markers. A
   blocked or invalid C8 long-context case remains unavailable, not a pass.

Exported NPZ inputs preserve content and stride metadata; `load_inputs` restores
the views and verifies their digest. `cpu_reference_*` arrays are debugging
oracles, not expected GPU bit patterns. For two later GPU captures, write
`scores` (FP32), `visible` (integer), `top512` (integer `[M,512]`), and identical
nonempty `fixture_sha256` and `experiment_id` strings, then run:

```bash
.venv/bin/python -m tools.qwen4_exp.qsa_scorer_study.run compare-gpu \
  qsa-results/baseline.npz qsa-results/candidate.npz \
  --output qsa-results/comparison.json
```

The reader compares GPU with GPU, ignores undefined score padding, and exits
nonzero on a difference. It cannot verify capture provenance or GPU timing.
Its unit tests use explicitly synthetic captures; they do not run CUDA.
