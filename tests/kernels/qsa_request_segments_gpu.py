# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Deferred SM70 scorer qualification. --matrix writes the plan without a GPU.

Run only in an approved GPU window. This does not run an inference engine or
measure performance. Failure is fatal: no numerical tolerance is applied.
"""

import argparse
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any

import torch
from test_qsa_request_segments_cpu import ROOT, inputs, load_candidate


def matrix():
    return [
        {
            "requests": c,
            "rows_per_request": r,
            "context": ctx,
            "page_size": 416,
            "table_width": 316,
            "segment_rows": segment,
        }
        for c, r in ((1, 8), (4, 4), (4, 5), (8, 5))
        for ctx in (32768, 131072, 262144)
        for segment in (1, 2, 4)
    ]


def guarded_workspace(mod, q, columns):
    # Guard all caller-owned buffers; keep the slices themselves contiguous.
    stores = []
    views = []
    for shape, dtype in (
        ((len(q), 3), torch.int32),
        ((len(q),), torch.int32),
        ((len(q), columns), torch.float32),
    ):
        n = 1
        for dim in shape:
            n *= dim
        store = torch.full((n + 128,), 991, dtype=dtype, device=q.device)
        stores.append(store)
        views.append(store[64:-64].view(shape))
    return mod.SegmentWorkspace(*views), stores


def run_case(mod, qsa, shape):
    c, r, ctx = (shape[k] for k in ("requests", "rows_per_request", "context"))
    columns = shape["page_size"] * shape["table_width"]
    x = [
        t.cuda()
        for t in inputs((r,) * c, page=shape["page_size"], columns=columns, normal=True)
    ]
    q, cache, table, requests, positions, lengths = x
    positions.copy_(torch.arange(ctx - r, ctx, device=q.device).repeat(c))
    lengths.fill_(ctx)
    ws, guards = guarded_workspace(mod, q, columns)

    def candidate():
        return mod.qsa_mqa_request_segments(*x, 4, ws, max_rows=shape["segment_rows"])

    def baseline():
        return qsa.qsa_mqa_paged(*x, 4, shared_key_scoring=False)

    for _ in range(3):
        candidate()
        baseline()
    torch.accelerator.synchronize()
    graphs = []
    outputs = []
    for fn in (baseline, candidate):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(fn())
        graphs.append(graph)
    checks = []
    for change in range(4):
        if change:
            q.neg_()
            cache.neg_()
            table.copy_(table.flip(1))
            positions.sub_(37)
            lengths.sub_(37)
        if change == 1:
            requests.zero_()  # Change active request and descriptor counts.
        elif change == 2:
            requests.copy_(torch.arange(len(q), device=q.device).remainder(c))
            requests[-1] = -1
            positions[-1] = -5
            table[0, 1] = -1  # Invalid visible page, not merely an unused tail.
        elif change == 3:
            requests.fill_(c - 1)
            lengths[-1] = 79
            positions.fill_(78)
        ws.scores.fill_(float("nan"))
        ws.descriptors.fill_(991)
        for graph in graphs:
            graph.replay()
        torch.accelerator.synchronize()
        fresh = baseline()
        for name, (score, visible) in zip(("paged_graph", "segment_graph"), outputs):
            assert torch.equal(visible, fresh[1]), (shape, change, name, "visible")
            mask = torch.arange(columns, device=q.device)[None, :] < fresh[1][:, None]
            bits_equal = torch.equal(
                score.view(torch.int32)[mask], fresh[0].view(torch.int32)[mask]
            )
            assert bits_equal, (shape, change, name, "FP32 words")
            selected = []
            for scores, vis in ((score, visible), fresh):
                ids = torch.empty((len(q), 512), dtype=torch.int32, device=q.device)
                qsa._sm70_qsa_lexicographic_topk_op()(scores, vis, ids, 512)
                selected.append(ids)
            assert torch.equal(*selected), (shape, change, name, "top512")
        for store in guards:
            assert bool((store[:64] == 991).all())
            assert bool((store[-64:] == 991).all())
        checks.append(
            {
                "change": change,
                "scores_bitwise": True,
                "visible_equal": True,
                "top512_equal": True,
                "buffer_guards": True,
            }
        )
    return {"shape": shape, "checks": checks}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matrix", type=Path)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    if args.matrix:
        args.matrix.write_text(json.dumps(matrix(), indent=2) + "\n")
        return
    if args.out is None:
        ap.error("--out is required for GPU qualification")
    assert torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)
    import vllm._custom_ops  # noqa: F401

    qsa = importlib.import_module("vllm.models.qwen4_exp.nvidia.ops.qsa")
    assert qsa.__file__ is not None
    installed = hashlib.sha256(Path(qsa.__file__).read_bytes()).hexdigest()
    source = hashlib.sha256(
        (ROOT / "vllm/models/qwen4_exp/nvidia/ops/qsa.py").read_bytes()
    ).hexdigest()
    assert installed == source, "installed paged baseline must match the checkout"
    mod = load_candidate()
    result: dict[str, Any] = {
        "baseline_sha256": source,
        "candidate_sha256": hashlib.sha256(
            (
                ROOT / "vllm/models/qwen4_exp/nvidia/ops/qsa_request_segments.py"
            ).read_bytes()
        ).hexdigest(),
        "engine_validation": "not_run",
        "performance": "not_measured",
        "cases": [],
    }
    for shape in matrix():
        result["cases"].append(run_case(mod, qsa, shape))
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result["cases"][-1]), flush=True)


if __name__ == "__main__":
    main()
