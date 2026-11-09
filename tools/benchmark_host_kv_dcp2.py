# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""host KV A/B/C host-KV experiment, run with torchrun --nproc-per-node 4.

NCCL uses contiguous DCP pairs [0,1], [2,3]. CPU correctness uses gloo and
TRITON_INTERPRET=1. All JSON records identify the rank and fixed profile.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import timedelta
from functools import partial
from typing import Any

import torch
import torch.distributed as dist

from tools.host_kv_dcp2 import (
    ExperimentGroup,
    LayerExperiment,
    Profile,
    enable_cpu_interpreter,
    error_metrics,
    reset_hot,
)
from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops


def measure(fn, cuda: bool):
    if not cuda:
        begin = time.perf_counter_ns()
        result = fn()
        return result, (time.perf_counter_ns() - begin) / 1e6
    begin, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    begin.record()
    result = fn()
    end.record()
    end.synchronize()
    return result, begin.elapsed_time(end)


def check_results(reference, actual, *, allow_drift: bool = False):
    names = ("partial", "partial_lse", "output", "lse")
    metrics = {}
    for name, a, b in zip(names, reference, actual):
        valid_a = torch.isfinite(a) | (torch.isneginf(a) if "lse" in name else False)
        valid_b = torch.isfinite(b) | (torch.isneginf(b) if "lse" in name else False)
        if not bool(valid_a.all() and valid_b.all()):
            raise AssertionError(f"{name} contains invalid nonfinite values")
        metrics[name] = error_metrics(a, b)
    if not allow_drift and any(not value["bitwise"] for value in metrics.values()):
        raise AssertionError(f"partial differs or later divergence: {metrics}")
    if allow_drift:
        # Frozen before execution, same thresholds as the existing grouped
        # regression. B/C never uses tolerances or this opt-in.
        for name, a, b in zip(names, reference, actual):
            atol, rtol = (1e-3, 0.0) if "lse" in name else (3e-3, 1e-2)
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return metrics


def run_experiment(
    group: ExperimentGroup,
    profile: Profile,
    *,
    device: torch.device,
    steps: int = 2,
    warmup: int = 1,
    scenario: str = "churn",
    route: str = "auto",
    combine: str = "ag_rs",
    graph: bool = False,
    allow_plan_drift: bool = False,
    edge: bool = False,
    fail_rank: int = -1,
    fail_phase: str = "metadata",
) -> dict[str, Any]:
    cuda = device.type == "cuda"

    def inject(phase):
        if group.rank == fail_rank and phase == fail_phase:
            raise RuntimeError(f"injected failure at {phase}")

    def setup():
        inject("setup")
        return LayerExperiment(profile, group, device, route, combine)

    case = group.local("setup", setup, cuda=cuda)
    group.agree("setup-complete")
    stable_tensors = (
        case.selection,
        case.local,
        case.pool.indices,
        case.pool.hot,
        case.gathered,
        case.partial,
        case.lse,
        case.query,
        case.gate,
        case.output,
    )
    addresses = tuple(t.data_ptr() for t in stable_tensors)
    captured = None
    if graph:

        def graph_buffers():
            return (
                torch.empty_like(case.partial),
                torch.empty_like(case.lse),
                torch.empty(case.query.shape[:2], dtype=torch.float32, device=device),
            )

        graph_partial, graph_lse, graph_final_lse = group.local(
            "graph-buffers",
            graph_buffers,
            cuda=True,
        )

        def capture():
            case.prepare(0, 0, scenario, edge=edge)

            def body():
                case.gathered.copy_(group.all_gather(case.query, dim=1))
                case.localize()
                case.gather()
                case.attention("C")
                graph_partial.copy_(case.partial)
                graph_lse.copy_(case.lse)
                _, final_lse = case.finish()
                graph_final_lse.copy_(final_lse)

            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(2):
                    body()
            torch.cuda.current_stream().wait_stream(stream)
            torch.accelerator.synchronize()
            result = torch.cuda.CUDAGraph()
            with torch.cuda.graph(result):
                body()
            return result

        captured = group.local("capture", capture, cuda=True)
    group.local("reset", lambda: reset_hot(case.pool), cuda=cuda)
    records = []
    phases: dict[str, float] = {}

    def timed(name, fn):
        result, elapsed = measure(fn, cuda)
        phases[name] = phases.get(name, 0.0) + elapsed
        return result

    def snapshot(arm):
        case.attention(arm)
        return case.partial.clone(), case.lse.clone()

    def finish():
        out, lse = case.finish()
        return out.clone(), lse.clone()

    def finish_c():
        out, lse = timed("dcp_combine_gate_ms", case.finish)
        return out.clone(), lse.clone()

    def validate(a, b, c):
        inject("validate")
        return {
            "BC": check_results(b, c),
            "AB": check_results(a, b, allow_drift=allow_plan_drift),
        }

    def read_counters():
        inject("counters")
        count = int(case.pool.miss_count.cpu())
        wants = int(case.pool.wanted_count.cpu())
        gids = case.pool.miss_groups[:count].cpu().long()
        starts = gids % case.config.groups_per_request * 4
        rows_read = (case.config.max_tokens - starts).clamp(0, 4).sum().item()
        transferred = int(rows_read) * 2 * profile.dim * case.pool.host.element_size()
        return count, wants, transferred

    for step in range(warmup + steps):
        phases = {}
        byte_count = wanted = misses = 0
        transfer_spans = []
        comparisons = []
        operands = []
        b_results = []
        for batch in range(profile.queries // profile.batch_queries):

            def prepare(step=step, batch=batch):
                case.prepare(step, batch, scenario, edge=edge)
                if scenario == "cold" and batch == 0:
                    reset_hot(case.pool)
                if tuple(t.data_ptr() for t in stable_tensors) != addresses:
                    raise AssertionError("replay buffer address changed")
                if group.rank == fail_rank and fail_phase == "trace":
                    case.blocks[0, 0].bitwise_xor_(1)
                inject("before-query")

            group.local("prepare", prepare, cuda=cuda)
            digest = group.local("trace-digest", case.trace_digest)
            group.agree("trace-contract", contract=digest)
            # All local failures agree on the CPU control plane before entering
            # each subsequent device collective. Each collective has a timeout.
            if captured is None:
                group.local(
                    "query-gather",
                    lambda: timed(
                        "query_gather_ms",
                        lambda: case.gathered.copy_(
                            group.all_gather(case.query, dim=1)
                        ),
                    ),
                    cuda=cuda,
                )

                def metadata():
                    inject("metadata")
                    timed("selection_localize_ms", case.localize)
                    timed(
                        "metadata_ms",
                        lambda: case.pool._gather_metadata(
                            case.narrow,
                            case.requests,
                            case.local_lengths,
                            case.local_positions,
                        ),
                    )

                group.local("metadata", metadata, cuda=cuda)
                # World fence, outside event spans, aligns both DCP groups on
                # the same host uplink. perf_counter_ns shares the host clock.
                group.agree("simultaneous-gather")

                def payload(spans=transfer_spans):
                    start = time.perf_counter_ns()
                    timed("gather_ms", case.pool._gather_payload)
                    spans.append((start, time.perf_counter_ns()))
                    timed(
                        "remap_ms",
                        lambda: case.pool._remap(
                            case.narrow,
                            case.requests,
                            case.local_lengths,
                            case.local_positions,
                        ),
                    )
                    case.hot_full.fill_(-1)
                    case.hot_full[:, : case.config.selection_width].copy_(
                        case.pool.indices
                    )

                group.local("gather", payload, cuda=cuda)

                def attention():
                    inject("attention")
                    timed("attention_ms", lambda: case.attention("C"))
                    return case.partial.clone(), case.lse.clone()

                c_partial = group.local("attention-C", attention, cuda=cuda)
            else:

                def replay():
                    inject("graph")
                    timed("graph_total_ms", captured.replay)
                    return graph_partial.clone(), graph_lse.clone()

                c_partial = group.local("graph-replay", replay, cuda=True)
            if captured is None:
                group.local("pre-combine-C", lambda: inject("pre-combine"))
                c_final = group.local("combine-C", finish_c, cuda=cuda)
            else:
                c_final = group.local(
                    "graph-final-snapshot",
                    lambda: (case.output.clone(), graph_final_lse.clone()),
                    cuda=True,
                )
            # Independent source bytes and logical identities are checked
            # before either reference arm is allowed into a DCP collective.
            group.local("payload-gate", case.mirror_and_check, cuda=cuda)
            b_partial = group.local("attention-B", lambda: snapshot("B"), cuda=cuda)
            b_final = group.local("combine-B", finish, cuda=cuda)
            a_partial = group.local("attention-A", lambda: snapshot("A"), cuda=cuda)
            a_final = group.local("combine-A", finish, cuda=cuda)

            comparisons.append(
                group.local(
                    "numerical-gate",
                    partial(
                        validate,
                        (*a_partial, *a_final),
                        (*b_partial, *b_final),
                        (*c_partial, *c_final),
                    ),
                    cuda=cuda,
                )
            )

            def save_operands():
                return case.query.clone(), case.gate.clone(), case.local.clone()

            operands.append(
                group.local("save-original-plan-inputs", save_operands, cuda=cuda)
            )
            b_results.append((*b_partial, *b_final))
            count, wants, transferred = group.local("step-counters", read_counters)
            misses += count
            wanted += wants
            byte_count += transferred
        full = group.local(
            "original-full-row-plan", partial(case.full_reference, operands), cuda=cuda
        )
        group.local(
            "original-query-gather",
            lambda full=full: full.gathered.copy_(group.all_gather(full.query, dim=1)),
            cuda=cuda,
        )

        def full_partial(full=full):
            full.attention("A")
            return full.partial.clone(), full.lse.clone()

        original_partial = group.local("original-attention", full_partial, cuda=cuda)

        def full_finish(full=full):
            out, lse = full.finish()
            return out.clone(), lse.clone()

        original_final = group.local("original-combine", full_finish, cuda=cuda)

        def check_original(
            b_results=b_results,
            original_partial=original_partial,
            original_final=original_final,
        ):
            joined = tuple(
                case.join_batches([result[i] for result in b_results]) for i in range(4)
            )
            return check_results(
                (*original_partial, *original_final),
                joined,
                allow_drift=allow_plan_drift,
            )

        full_comparison = group.local(
            "original-plan-numerical-gate", check_original, cuda=cuda
        )
        if step >= warmup:
            records.append(
                dict(
                    step=step - warmup,
                    phases=phases.copy(),
                    wanted=wanted,
                    misses=misses,
                    transfer_bytes=byte_count,
                    transfer_spans=transfer_spans,
                    comparisons=comparisons,
                    original_full_rows=8 * profile.queries,
                    original_full_grouped_rows=full.last_grouped,
                    original_plan_comparison=full_comparison,
                    grouped_rows=case.last_grouped,
                    instrumented_ms=sum(phases.values()),
                )
            )
    summary = dict(
        rank=group.rank,
        profile=profile.__dict__,
        scenario=scenario,
        route=route,
        combine=combine,
        graph=graph,
        records=records,
        graph_scope=(
            "query gather, selection/localize, metadata, gather, remap, "
            "attention, DCP combine, FP16 rounding, gate"
        ),
        timing_contract=(
            "CUDA events include submission gaps; controls, input copies, A/B checks "
            "and cold resets excluded; sum batches, one layer; host clock spans "
            "include launch skew"
        ),
        plan_contract=dict(
            original_rows=8 * profile.queries,
            batch_rows=case.config.rows,
            global_width=profile.topk + 3,
            local_width=case.config.selection_width,
            gathered_heads=case.gathered.shape[1],
            source_page=profile.page,
            hot_page=4,
            source_stride=list(case.source.stride()),
            hot_stride=list(case.pool.hot.stride()),
            scales=case.scales.__dict__,
            partial_dtype="float32",
            lse_base=2,
            ab_drift_opt_in=allow_plan_drift,
            grouped_plan_change=(
                "batching changes the launch profile/split count; grouped planner "
                "sorts category and logical_token//4; "
                "C hot-slot token IDs change this order relative to A; "
                "B uses the identical C order and GPU source bytes"
            ),
        ),
        payload_contract=(
            "synthetic encoded operands, layer scales fixed, original expansion/"
            "localize/attention/combine/gate; fixed compressed selections, "
            "no index scorer"
        ),
    )
    group.agree("complete")
    return summary


def aggregate_bandwidth(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for index in range(len(summaries[0]["records"])):
        rows = [s["records"][index] for s in summaries]
        if not rows[0]["transfer_spans"]:
            continue
        # A single synchronized batch at a time. Add each batch's world-wide
        # wall envelope, so gaps for checking/reference attention are excluded.
        span_ns = sum(
            max(r["transfer_spans"][b][1] for r in rows)
            - min(r["transfer_spans"][b][0] for r in rows)
            for b in range(len(rows[0]["transfer_spans"]))
        )
        total_bytes = sum(r["transfer_bytes"] for r in rows)
        max_ms = max(r["phases"]["gather_ms"] for r in rows)
        results.append(
            dict(
                step=index,
                ranks=len(rows),
                bytes=total_bytes,
                per_rank_gather_ms=[r["phases"]["gather_ms"] for r in rows],
                per_rank_misses=[r["misses"] for r in rows],
                overlap_all_ranks_ms=[
                    max(
                        0,
                        min(r["transfer_spans"][b][1] for r in rows)
                        - max(r["transfer_spans"][b][0] for r in rows),
                    )
                    / 1e6
                    for b in range(len(rows[0]["transfer_spans"]))
                ],
                world_payload_wall_ms=span_ns / 1e6,
                aggregate_wall_gib_s=total_bytes / max(span_ns / 1e9, 1e-12) / 2**30,
                aggregate_max_event_gib_s=total_bytes
                / max(max_ms / 1000, 1e-12)
                / 2**30,
                shared_uplink_gib_s=12.3,
            )
        )
    return results


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument("--tokens", type=int, default=524287)
    parser.add_argument("--queries", type=int, choices=(1, 8), default=8)
    parser.add_argument("--batch-queries", type=int, default=2)
    parser.add_argument("--hot-ratio", type=float, choices=(0.1, 0.2), default=0.1)
    parser.add_argument("--union-groups", type=int, default=1536)
    parser.add_argument(
        "--scenario", choices=("warm", "churn", "cold"), default="churn"
    )
    parser.add_argument("--payload", choices=("e4m3", "fp16"), default="e4m3")
    parser.add_argument(
        "--route", choices=("auto", "triton", "grouped"), default="auto"
    )
    parser.add_argument("--combine", choices=("ag_rs", "a2a"), default="ag_rs")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--edge", action="store_true")
    parser.add_argument("--allow-plan-drift", action="store_true")
    parser.add_argument("--fail-rank", type=int, default=-1)
    parser.add_argument(
        "--fail-phase",
        choices=(
            "setup",
            "trace",
            "before-query",
            "metadata",
            "attention",
            "pre-combine",
            "validate",
            "counters",
            "graph",
        ),
        default="metadata",
    )
    args = parser.parse_args()
    if min(args.steps, args.timeout) <= 0 or args.warmup < 0:
        parser.error("steps and timeout must be positive, warmup nonnegative")
    if args.backend == "gloo" and (args.graph or args.route == "grouped"):
        parser.error("graph and native grouped require CUDA/NCCL")
    if (
        args.graph
        and args.fail_rank >= 0
        and args.fail_phase
        in (
            "metadata",
            "attention",
            "pre-combine",
        )
    ):
        parser.error("use --fail-phase graph for captured local-stage failures")
    if args.fail_phase == "graph" and args.fail_rank >= 0 and not args.graph:
        parser.error("graph failure injection requires --graph")
    return args


def make_profile(args) -> Profile:
    profile = Profile(
        queries=args.queries,
        batch_queries=args.batch_queries,
        tokens=args.tokens,
        union_groups=args.union_groups,
        hot_ratio=args.hot_ratio,
        e4m3=args.payload == "e4m3",
    )
    if args.queries == 1:
        profile = Profile(
            **(profile.__dict__ | dict(batch_queries=1, union_groups=512))
        )
    if args.backend == "gloo" or args.probe:
        profile = Profile(
            tokens=4097,
            topk=4 if args.backend == "gloo" else 16,
            union_groups=(
                min(2, args.queries)
                if args.backend == "gloo"
                else min(8, 4 * args.queries)
            ),
            queries=args.queries,
            batch_queries=1,
            heads=2 if args.backend == "gloo" else 6,
            dim=16 if args.backend == "gloo" else 256,
            page=16 if args.backend == "gloo" else 832,
            hot_ratio=args.hot_ratio,
            e4m3=args.payload == "e4m3",
        )
    return profile


def main() -> None:
    args = parse_args()
    cpu = args.backend == "gloo"
    if cpu:
        enable_cpu_interpreter()
        device = torch.device("cpu")
    else:
        torch.accelerator.set_device_index(int(os.environ["LOCAL_RANK"]))
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    if args.route == "grouped":
        # Explicit isolated diagnostic of the native route below its speed
        # gate. Nothing in the runtime package or production config is edited.
        qsa_ops._SM70_QSA_DCP_GROUPED_PAGE4_MIN_ROWS = 8
    profile = make_profile(args)
    dist.init_process_group(args.backend, timeout=timedelta(seconds=args.timeout))
    try:
        group = ExperimentGroup(args.backend, args.timeout)

        def preflight():
            if dist.get_world_size() == 1 and not args.probe:
                raise ValueError("One rank is a probe only; pass --probe")
            if not -1 <= args.fail_rank < dist.get_world_size():
                raise ValueError("fail-rank is outside the world")
            return profile.config(group.world_size)

        group.local("preflight", preflight)
        summary = run_experiment(
            group,
            profile,
            device=device,
            steps=args.steps,
            warmup=args.warmup,
            scenario=args.scenario,
            route=args.route,
            combine=args.combine,
            graph=args.graph,
            allow_plan_drift=args.allow_plan_drift,
            edge=args.edge,
            fail_rank=args.fail_rank,
            fail_phase=args.fail_phase,
        )
        all_summaries: list[Any] = [None] * dist.get_world_size()
        dist.all_gather_object(all_summaries, summary, group=group.control)
        if group.rank == 0:
            print(
                json.dumps(
                    dict(
                        status="PASS",
                        ranks=all_summaries,
                        bandwidth=aggregate_bandwidth(all_summaries),
                    ),
                    allow_nan=False,
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
