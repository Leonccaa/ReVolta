# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Micro-benchmark: SM70 grouped small-M AWQ MoE decode vs the TurboMind route.

Times W13+W2 of one Qwen3.8 TP4 MoE layer (E=512, top-10, hidden 2560,
intermediate 160 per rank, AWQ g32 with compact metadata) for M rows, on full
512-expert prepared banks built from random AWQ checkpoint tensors through
the production AWQSM70MoEMethod (awq_sm70_prepare_compact):

* grouped: awq_moe_grouped_decode_sm70_out (plan + W13/SwiGLU + W2/reduce),
  W13 split 16, 8 and 4;
* turbomind: AWQSM70MoEMethod.apply() with the grouped route disabled, i.e.
  the production route (permute + active-grouped TurboMind GEMMs for M<=8,
  per-expert dispatch for M>8, SwiGLU, W2, unpermute);
* grouped20 (M 9..20): the same apply() with
  VLLM_SM70_AWQ_QWEN38_MOE_GROUPED_DECODE20 (grouped TurboMind launch specs).
  Run with VLLM_SM70_AWQ_TUNE_SMALL_SHAPES=1 and
  VLLM_SM70_AWQ_MOE_TUNE_MAX_TOKENS=200 to include tactic tuning at M=20.

Each variant is captured in a CUDA graph of --inner calls and replayed; the
median per-call time over --rounds is reported, with the effective weight
bandwidth (distinct experts x 729,600 B). --profile adds a torch.profiler pass
with per-kernel CUDA times (W13/W2 split). Routing is synthetic MTP-verify-like
(5 correlated rows per request) unless --routing-file gives a saved [N, 10]
topk_ids tensor (rows are taken in order).

Example (V100, one GPU):
  python benchmarks/kernels/benchmark_sm70_awq_grouped_decode.py \\
      --m 5 10 15 20 --json /tmp/grouped.json --profile
"""

import argparse
import json
import os
import random
import statistics
import time

import torch

BYTES_PER_EXPERT = 486_400 + 243_200  # W13 + W2 incl. compact metadata, per rank


def synthetic_routing(m, seed, share, zipf):
    """M rows, 5 rows per request; rows of one request share experts."""
    rng = random.Random(seed)
    experts = list(range(512))
    weights = [1.0 / (rank + 1) ** zipf for rank in range(512)]
    order = experts[:]
    rng.shuffle(order)  # popularity rank -> expert id

    def draw(exclude, count):
        chosen = []
        while len(chosen) < count:
            e = order[rng.choices(range(512), weights=weights)[0]]
            if e not in exclude and e not in chosen:
                chosen.append(e)
        return chosen

    rows, previous = [], None
    for token in range(m):
        if token % 5 == 0 or previous is None:
            chosen = draw(set(), 10)
        else:
            keep = [e for e in previous if rng.random() < share]
            chosen = keep + draw(set(keep), 10 - len(keep))
        rows.append(chosen)
        previous = chosen
    return torch.tensor(rows, dtype=torch.int32)


def build_layer():
    from vllm.model_executor.layers.quantization.sm70_awq_grouped_reference import (
        build_random_awq_layer,
    )

    return build_random_awq_layer()


def time_graph(fn, inner, rounds, warmup=3):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(inner):
            fn()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0 / inner)
    return statistics.median(samples), min(samples), graph


def profile_kernels(graph, replays=20):
    from torch.profiler import ProfilerActivity, profile

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(replays):
            graph.replay()
        torch.cuda.synchronize()
    rows = {}
    for event in prof.key_averages():
        if event.device_type is not None and "CUDA" in str(event.device_type):
            total = getattr(event, "device_time_total", None)
            if total is None:
                total = event.cuda_time_total
            rows[event.key[:120]] = {
                "calls": event.count,
                "total_us": total,
                "avg_us": total / max(1, event.count),
            }
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--m", type=int, nargs="+", default=[5, 10, 15, 20])
    parser.add_argument(
        "--inner", type=int, default=48, help="calls per graph (layers)"
    )
    parser.add_argument("--rounds", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument(
        "--share",
        type=float,
        default=0.5,
        help="probability a row keeps an expert of the previous row",
    )
    parser.add_argument(
        "--zipf", type=float, default=0.6, help="expert popularity skew (0 = uniform)"
    )
    parser.add_argument(
        "--routing-file",
        default=None,
        help="torch.save'd [N, 10] int topk_ids; first M rows used",
    )
    parser.add_argument("--splits", type=int, nargs="+", default=[16, 8, 4])
    parser.add_argument("--no-turbomind", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--json", default=None)
    args = parser.parse_args()

    assert torch.cuda.get_device_capability() == (7, 0), "requires SM70"
    from vllm import _sm70_ops as ops

    if not hasattr(torch.ops._C, "awq_moe_grouped_decode_sm70_out"):
        raise SystemExit("build lacks awq_moe_grouped_decode_sm70_out")
    os.environ.setdefault("VLLM_SM70_AWQ_QWEN38_GROUPED_QPN", "1")
    os.environ.setdefault("VLLM_SM70_AWQ_QWEN38_MOE_GROUPED_DECODE20", "1")
    os.environ.setdefault("VLLM_SM70_AWQ_QWEN38_MOE_COMPACT_GROUPED_DECODE", "1")
    os.environ.setdefault("VLLM_SM70_AWQ_MOE_COMPACT_METADATA", "1")
    os.environ.setdefault("VLLM_SM70_AWQ_TUNE_SMALL_SHAPES", "0")
    started = time.time()
    layer = build_layer()
    print(
        f"prepared banks in {time.time() - started:.1f}s "
        f"(compact={layer.sm70_awq_moe_compact_metadata})",
        flush=True,
    )
    method = layer._r2_method
    saved = None
    if args.routing_file:
        saved = torch.load(args.routing_file).to(torch.int32).reshape(-1, 10)

    results = []
    for m in args.m:
        ids_cpu = (
            saved[:m]
            if saved is not None
            else synthetic_routing(m, args.seed + m, args.share, args.zipf)
        )
        ids = ids_cpu.cuda().contiguous()
        distinct = int(torch.unique(ids[(ids >= 0) & (ids < 512)]).numel())
        torch.manual_seed(args.seed + m)
        x = (torch.randn(m, 2560, device="cuda") * 0.5).half()
        topk = torch.softmax(torch.randn(m, 10, device="cuda"), -1).float()
        weight_mb = distinct * BYTES_PER_EXPERT / 1e6
        entry = {"m": m, "distinct_experts": distinct, "weight_mb": weight_mb}

        out = torch.empty(m, 2560, dtype=torch.float16, device="cuda")
        inter = torch.empty(m * 10, 160, dtype=torch.float16, device="cuda")
        plan = torch.empty(
            ops.AWQ_GROUPED_DECODE_PLAN_INTS, dtype=torch.int32, device="cuda"
        )
        reference = None
        if not args.no_turbomind:
            layer.sm70_awq_qwen38_grouped_qpn = False
            reference = method.apply(layer, x, topk, ids, None, None).clone()

            def turbomind(x=x, topk=topk, ids=ids):
                layer.sm70_awq_qwen38_grouped_qpn = False
                layer.sm70_awq_qwen38_grouped_decode20 = False
                method.apply(layer, x, topk, ids, None, None)

            median, best, graph = time_graph(turbomind, args.inner, args.rounds)
            entry["turbomind_us"] = median
            entry["turbomind_best_us"] = best
            entry["turbomind_gbps"] = weight_mb * 1e6 / (median * 1e-6) / 1e9
            if args.profile:
                entry["turbomind_kernels"] = profile_kernels(graph)
            del graph
            if 9 <= m <= 20:
                # VLLM_SM70_AWQ_QWEN38_MOE_GROUPED_DECODE20 route.
                def grouped20(x=x, topk=topk, ids=ids):
                    layer.sm70_awq_qwen38_grouped_qpn = False
                    layer.sm70_awq_qwen38_grouped_decode20 = True
                    method.apply(layer, x, topk, ids, None, None)

                grouped20()
                torch.cuda.synchronize()
                out20 = method.apply(layer, x, topk, ids, None, None).clone()
                entry["grouped20_max_abs_vs_turbomind"] = (
                    (out20.double() - reference.double()).abs().max().item()
                )
                median, best, graph = time_graph(grouped20, args.inner, args.rounds)
                entry["grouped20_us"] = median
                entry["grouped20_best_us"] = best
                entry["grouped20_gbps"] = weight_mb * 1e6 / (median * 1e-6) / 1e9
                if args.profile:
                    entry["grouped20_kernels"] = profile_kernels(graph)
                del graph
                layer.sm70_awq_qwen38_grouped_decode20 = False
        for split in args.splits:

            def grouped(
                split=split, out=out, inter=inter, plan=plan, x=x, ids=ids, topk=topk
            ):
                ops.awq_moe_grouped_decode_sm70_out(
                    out,
                    inter,
                    plan,
                    x,
                    layer.w13_tm_weight,
                    layer.w13_tm_scales,
                    layer.w2_tm_weight,
                    layer.w2_tm_scales,
                    ids,
                    topk,
                    split,
                )

            grouped()
            torch.cuda.synchronize()
            if reference is not None:
                diff = (out.double() - reference.double()).abs()
                entry[f"split{split}_max_abs_vs_turbomind"] = diff.max().item()
                entry["ref_max_abs"] = reference.double().abs().max().item()
            median, best, graph = time_graph(grouped, args.inner, args.rounds)
            entry[f"split{split}_us"] = median
            entry[f"split{split}_best_us"] = best
            entry[f"split{split}_gbps"] = weight_mb * 1e6 / (median * 1e-6) / 1e9
            if args.profile:
                entry[f"split{split}_kernels"] = profile_kernels(graph)
            del graph
        layer.sm70_awq_qwen38_grouped_qpn = True
        results.append(entry)
        summary = {
            k: (round(v, 3) if isinstance(v, float) else v)
            for k, v in entry.items()
            if not k.endswith("_kernels")
        }
        print(json.dumps(summary), flush=True)
        if args.profile:
            for key in [k for k in entry if k.endswith("_kernels")]:
                print(f"  {key}:")
                for name, row in sorted(
                    entry[key].items(), key=lambda kv: -kv[1]["total_us"]
                )[:8]:
                    print(f"    {row['avg_us']:9.2f} us x{row['calls']:<4} {name}")
    if args.json:
        with open(args.json, "w") as handle:
            json.dump({"args": vars(args), "results": results}, handle, indent=1)


if __name__ == "__main__":
    main()
