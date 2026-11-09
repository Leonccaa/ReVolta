# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-device, eight-request host KV benchmark with bounded query batches.

Run as ``python -m tools.benchmark_host_kv --hot-ratio 0.1``. Defaults model
eight query rows per request, 1536 unique groups per request across those rows,
and two query rows per request per gather/attention batch. Token coordinates
are rank-local. This synthetic benchmark contains no DCP communication.
"""

import argparse
import json
import math
import statistics
from dataclasses import replace

import torch

from vllm.models.qwen4_exp.nvidia.host_kv import ActiveHostKV, HostKVConfig
from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan


def make_config(args: argparse.Namespace) -> HostKVConfig:
    if (
        min(args.steps, args.layers, args.batch_queries, args.queries) <= 0
        or args.queries * 8 > 4096
        or args.tokens % 32
        or args.tokens < 8192
        or args.queries % args.batch_queries
        or not 512 <= args.union_groups <= min(8 * 512, args.tokens // 4 - 9)
        or args.queries * 512 < args.union_groups
        or not 0 <= args.churn_fraction <= 1
        or args.warmup < 1
    ):
        raise ValueError("Invalid benchmark geometry or iteration count")
    config = HostKVConfig(
        8,
        args.tokens,
        8 * args.batch_queries,
        2051,
        4096,
        hot_ratio=args.hot_ratio,
        qsa_block_topk=512,
        eviction_policy=args.eviction_policy,
    )
    config.capacities()
    return config


def make_trace(args: argparse.Namespace, config: HostKVConfig) -> torch.Tensor:
    generator = torch.Generator().manual_seed(720)
    groups = config.groups_per_request - 9  # Exclude the protected tail.
    stable = args.union_groups - round(args.union_groups * args.churn_fraction)
    if args.scenario == "warm":
        stable = args.union_groups
    traces = []
    # One shared prefix and independently shuffled churn suffixes per request.
    permutations = [torch.randperm(groups, generator=generator) for _ in range(8)]
    for _ in range(args.warmup + args.steps):
        rows = []
        for req in range(8):
            base = permutations[req]
            suffix = base[stable:][torch.randperm(groups - stable, generator=generator)]
            union = torch.cat((base[:stable], suffix[: args.union_groups - stable]))
            for query in range(args.queries):
                start = query * args.union_groups // args.queries
                selected = union[(torch.arange(512) + start) % args.union_groups]
                expanded = (selected[:, None] * 4 + torch.arange(4)).flatten().int()
                rows.append(
                    torch.cat((expanded, torch.full((3,), -1, dtype=torch.int32)))
                )
        # [batch, request, query-within-batch, width], immediately consumed.
        traces.append(
            torch.stack(rows)
            .reshape(8, args.queries, 2051)
            .reshape(8, -1, args.batch_queries, 2051)
            .permute(1, 0, 2, 3)
            .reshape(-1, config.rows, 2051)
        )
    return torch.stack(traces)


def summarize(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1],
    }


def reset_hot(pool: ActiveHostKV) -> None:
    pool.group_to_slot.fill_(-1)
    pool.slot_owner.fill_(-1)
    pool.age.zero_()
    pool.clock.zero_()
    pool.eviction_hand.zero_()
    pool.wanted.zero_()
    pool.wanted_count.zero_()
    pool.miss_count.zero_()


def timing_events():
    return (
        torch.cuda.Event(enable_timing=True, external=True),
        torch.cuda.Event(enable_timing=True, external=True),
    )


def timed(records, name, fn, *arguments):
    begin, end = timing_events()
    begin.record()
    fn(*arguments)
    end.record()
    records[name].append((begin, end))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hot-ratio", type=float, required=True)
    parser.add_argument(
        "--eviction-policy", choices=("bucket", "lru"), default="bucket"
    )
    parser.add_argument("--compare-lru", action="store_true")
    parser.add_argument("--tokens", type=int, default=262144)
    parser.add_argument(
        "--queries", "--rows-per-request", dest="queries", type=int, default=8
    )
    parser.add_argument(
        "--batch-queries",
        "--rows-per-batch-request",
        dest="batch_queries",
        type=int,
        default=2,
    )
    parser.add_argument("--union-groups", type=int, default=1536)
    parser.add_argument("--churn-fraction", type=float, default=0.2)
    parser.add_argument(
        "--scenario", choices=("warm", "churn", "cold"), default="churn"
    )
    parser.add_argument("--payload", choices=("e4m3", "fp16"), default="e4m3")
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = make_config(args)
    slots, misses = config.capacities()
    element_bytes = 1 if args.payload == "e4m3" else 2
    print(
        json.dumps(
            {
                "config": vars(args),
                "requests": 8,
                "rows_per_batch": config.rows,
                "batches_per_step": args.queries // args.batch_queries,
                "slots_per_layer": slots,
                "cold_bound_per_batch": misses,
                "union_groups_per_request": args.union_groups,
                "union_groups_all_requests": 8 * args.union_groups,
                "pinned_payload_bytes": 8
                * args.tokens
                * 2
                * 256
                * element_bytes
                * args.layers,
                "hot_payload_bytes": slots * 8 * 256 * element_bytes * args.layers,
                "gather_workers_per_layer": min(misses, 256),
            }
        ),
        flush=True,
    )
    if args.dry_run:
        return
    policies = ("lru", "bucket") if args.compare_lru else (args.eviction_policy,)
    outputs = {}
    for policy in policies:
        outputs[policy] = run_benchmark(args, replace(config, eviction_policy=policy))
    report = outputs[args.eviction_policy]
    if args.compare_lru:
        report = {"policies": outputs}
        fractions = {
            policy: output["results"]["graph_phases"]["miss_fraction"]
            for policy, output in outputs.items()
        }
        report["miss_comparison"] = {
            "exact_lru": fractions["lru"],
            "bucket": fractions["bucket"],
            "bucket_minus_lru_percentage_points": 100
            * (fractions["bucket"] - fractions["lru"]),
        }
    print(json.dumps(report, indent=2), flush=True)


def run_benchmark(args, config):
    slots, misses = config.capacities()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "A CUDA device and the project's native extensions are required"
        )
    dtype = torch.uint8 if args.payload == "e4m3" else torch.float16
    device = torch.device("cuda")
    layout = torch.empty(
        (8 * args.tokens // 32, 2, 32, 1, 256), dtype=dtype, device="meta"
    )
    pools = [ActiveHostKV(config, layout, device=device) for _ in range(args.layers)]
    plans = [
        SparseAttentionPlan(
            rows=config.rows,
            query_heads=12,
            kv_heads=1,
            head_dim=256,
            selection_width=2051,
            device=device,
            e4m3=args.payload == "e4m3",
        )
        for _ in pools
    ]
    encoded = torch.full(
        (4096, 1, 256), 56 if args.payload == "e4m3" else 1, dtype=dtype, device=device
    )
    positions = torch.arange(4096, dtype=torch.int32, device=device)
    for pool in pools:
        for req in range(8):
            requests = torch.full_like(positions, req)
            for start in range(0, args.tokens, 4096):
                count = min(4096, args.tokens - start)
                pool.write_encoded(
                    encoded[:count],
                    encoded[:count],
                    requests[:count],
                    positions[:count] + start,
                )
    trace = make_trace(args, config).to(device)
    selections = torch.empty_like(trace[0])
    requests = torch.arange(8, dtype=torch.int32, device=device).repeat_interleave(
        args.batch_queries
    )
    lengths = torch.full((8,), args.tokens, dtype=torch.int32, device=device)
    qpos = torch.full((config.rows,), args.tokens - 1, dtype=torch.int64, device=device)
    query = torch.ones((config.rows, 12, 256), dtype=torch.float16, device=device)
    write_requests = torch.arange(
        8, dtype=torch.int32, device=device
    ).repeat_interleave(args.queries)
    write_positions = (
        torch.arange(args.queries, dtype=torch.int32, device=device)
        + args.tokens
        - args.queries
    ).repeat(8)
    key = torch.ones((8 * args.queries, 1, 256), dtype=torch.float16, device=device)

    def consume(pool, plan):
        hk, hv = pool.hot.unbind(1)
        plan.run(query, hk, hv, pool.indices, pool.hot_block_table, requests)

    def body():
        for pool, plan in zip(pools, plans):
            pool.write(key, key, write_requests, write_positions)
            for selection in selections:
                pool.gather(selection, requests, lengths, qpos)
                consume(pool, plan)

    def prepare(step):
        selections.copy_(trace[step])
        if args.scenario == "cold":
            for pool in pools:
                reset_hot(pool)

    # Compile every normal entry point on a side stream before graph capture.
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        prepare(0)
        body()
    torch.cuda.current_stream().wait_stream(warm_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    results = {}
    for route in ("eager_phases", "graph", "graph_phases"):
        for pool in pools:
            reset_hot(pool)
        for step in range(args.warmup):
            prepare(step)
            graph.replay()
        records: dict[str, list] = {
            name: []
            for name in (
                "write",
                "metadata",
                "mark_collect",
                "sort",
                "victim_selection",
                "mapping_update",
                "gather",
                "remap",
                "attention",
                "total",
            )
        }
        counters = torch.empty(
            (args.steps, args.layers, selections.shape[0], 2),
            dtype=torch.int32,
            device=device,
        )

        def measured_steps(route, records, counters):
            for index in range(args.steps):
                prepare(args.warmup + index)
                if route == "graph":
                    timed(records, "total", graph.replay)
                    continue
                for layer, (pool, plan) in enumerate(zip(pools, plans)):
                    timed(
                        records,
                        "write",
                        pool.write,
                        key,
                        key,
                        write_requests,
                        write_positions,
                    )
                    for batch, selection in enumerate(selections):
                        begin, end = timing_events()
                        begin.record()
                        timed(
                            records,
                            "mark_collect",
                            pool._mark_and_collect,
                            selection,
                            requests,
                            lengths,
                            qpos,
                        )
                        if pool.config.eviction_policy == "lru":
                            timed(records, "sort", pool._sort_victims)
                        else:
                            timed(records, "victim_selection", pool._select_victims)
                        timed(records, "mapping_update", pool._update_mapping)
                        end.record()
                        records["metadata"].append((begin, end))
                        timed(records, "gather", pool._gather_payload)
                        timed(
                            records,
                            "remap",
                            pool._remap,
                            selection,
                            requests,
                            lengths,
                            qpos,
                        )
                        timed(records, "attention", consume, pool, plan)
                        counters[index, layer, batch, 0].copy_(pool.wanted_count)
                        counters[index, layer, batch, 1].copy_(pool.miss_count)

        if route == "graph_phases":
            phase_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(phase_graph):
                measured_steps(route, records, counters)
            phase_graph.replay()
        else:
            measured_steps(route, records, counters)
        # One fence per complete route, with no per-iteration event wait.
        torch.accelerator.synchronize()
        results[route] = {}
        for name, pairs in records.items():
            if not pairs:
                if route != "graph" and name in ("sort", "victim_selection"):
                    results[route][name + "_ms_per_step"] = summarize(
                        [0.0] * args.steps
                    )
                    results[route][name + "_ms_per_layer_step"] = summarize(
                        [0.0] * args.steps
                    )
                continue
            elapsed = (
                torch.tensor([a.elapsed_time(b) for a, b in pairs])
                .reshape(args.steps, -1)
                .sum(1)
                .tolist()
            )
            results[route][name + "_ms_per_step"] = summarize(elapsed)
            results[route][name + "_ms_per_layer_step"] = summarize(
                [value / args.layers for value in elapsed]
            )
        if route != "graph":
            counts = counters.cpu().reshape(-1, 2)
            results[route]["wanted_groups_per_batch"] = summarize(counts[:, 0].tolist())
            results[route]["miss_groups_per_batch"] = summarize(counts[:, 1].tolist())
            results[route]["miss_fraction"] = float(
                counts[:, 1].sum() / counts[:, 0].sum()
            )
    return {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "timing_contract": (
            "Eager event spans include submission gaps. Graph total excludes "
            "input copies and cold resets. Graph phases use captured timing "
            "events. Values sum all batches and layers per step."
        ),
        "results": results,
    }


if __name__ == "__main__":
    main()
