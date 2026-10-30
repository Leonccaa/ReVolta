# Request-segmented QSA candidate

This is a research implementation. Serving dispatch does not import or call it.
GPU bit patterns, graph replay, memory safety, performance and engine behavior
still need qualification. Do not enable it based on CPU interpreter results.

`qsa_mqa_request_segments` splits each contiguous request run into segments of
at most 1, 2, 4 or 8 query rows. Repeated runs stay separate. A device builder
overwrites a fixed `[M,3]` int32 descriptor buffer on every call; unused slots
contain -1. A second kernel loads each segment's 64-key tile once and reuses
it across the original per-row dots. Each row keeps its own visibility mask.
The entry supports at most 256 rows and finite FP16 Q/K with H4/D128.

The candidate retains the paged scorer's 64x128 by 128x16 dot, 16 padded head
lanes, FP32 accumulation, ReLU, `tl.sum` over heads and score divisor. It uses
a device loop over rows to avoid duplicating the dot body eight times. This
preserves the arithmetic form, not a proof about the compiler's GPU output.
Combining queries into a wider dot changed random FP16 CPU interpreter results
in an exploratory control and is deferred. The existing native shared-key
operator is not used as the numerical reference.

Generic-stride offline compilation with Triton 3.6.0 to SM70 succeeds, but
also exposes a register-spill risk: the eight-row candidate has a 6856-byte
thread stack frame versus 8 bytes for the paged control in that compilation.
Both need 40 KiB dynamic shared memory. Real JIT alignment specializations
may change these resources. Treat this as a known performance limitation to
investigate in the GPU window, not as evidence of the native operator's speed.

Allocate `SegmentWorkspace` before capture, then pass the same buffers on each
call. The entry rebuilds descriptors, has no dynamic count readback and performs
no tensor allocation. Inputs and workspace must not overlap. The wrapper checks
shape, dtype, device and positive input strides; input values remain the caller's
responsibility. It does not extend the published serving support scope.

## CPU checks

Use an installed ReVolta environment with PyTorch and Triton, managed with `uv`.
From the checkout root:

```bash
CUDA_VISIBLE_DEVICES='' TRITON_INTERPRET=1 \
  .venv/bin/python tests/kernels/test_qsa_request_segments_cpu.py
.venv/bin/python tests/kernels/qsa_request_segments_gpu.py \
  --matrix qsa-segment-gpu-matrix.json
```

The first command runs the actual builder and scorer in Triton's CPU interpreter
and compares them with the paged kernel extracted from this checkout. The
independent NumPy oracle uses small dyadic FP16 values so addressing checks do
not depend on a particular BLAS reduction. Random FP16 uses the paged interpreter
control. Scores, visible lengths, stable top-512 IDs, untouched padding and
descriptor ownership are checked. Tests include request remapping between calls,
inactive rows, invalid pages, negative positions, ragged and repeated runs,
long-run splitting, strides, page sizes 16/204/416 and maximum builder capacity.

The [Triton interpreter](https://triton-lang.org/main/programming-guide/chapter-3/debugging.html)
executes on the CPU. Passing these checks does not establish SM70 rounding,
launch safety, physical key traffic or CUDA graph correctness. Zero-width scores
get deterministic visibility from the builder; the paged GPU zero-width metadata
is unspecified, so that case is not a GPU comparison gate.

## Deferred GPU checks

During a separately approved SM70 window, with the paged runtime matching this
checkout and the native exact top-k extension installed:

```bash
.venv/bin/python tests/kernels/qsa_request_segments_gpu.py \
  --out qsa-segment-gpu-results.json
```

The prepared matrix has 36 cases: C1/M8, C4/M16, C4/M20, C8/M40 at
32768/131072/262144 tokens, with segment limits 1/4/8. It captures the actual
paged call and candidate including the builder. Replay changes queries, keys,
page tables, positions, lengths, request runs and active descriptor counts.
Scores must match FP32 words, visibility and exact top-512 IDs. Guard storage
surrounds all three candidate buffers. The runner fails on any difference and
records no performance claim. Synthetic C8 long-context scorer inputs do not
establish engine admission. Use Compute Sanitizer and allocation/host-readback
instrumentation separately; guard values alone do not prove memory safety.

After correctness, alternate full scorer-plus-selection timings at identical
contexts, power, clocks and cache conditions. Include the builder, fixed empty
slots and every launch. Metadata storage is 16*M bytes (descriptors 12*M plus
visibility 4*M), in addition to `4*M*columns` score bytes; do not double-count
visibility already needed by the baseline. The fixed score grid launches
`M*ceil(columns/64)` programs, even for unused descriptor slots. At M40 the
shipped paged baseline already groups eight tiles and has a smaller grid.
Neither source-level load reuse nor interpreter time proves a speedup. Engine
C4/C8, offload and long-context acceptance comes after scorer qualification.
