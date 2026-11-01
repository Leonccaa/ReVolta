# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM70 check of the runtime path: qsa_select_paged_tokens with and without
VLLM_SM70_QSA_REQUEST_SEGMENTS must return bitwise the same token indices.

Run on one SM70 GPU (no engine, no weights):

    python tests/kernels/qsa_request_segments_select_gpu.py --out result.json [--time]

Shapes: uniform decode batches (4 and 8 requests, 4 or 5 rows each), one row
per request (MTP drafting steps), ragged rows, a prefill chunk next to decode
rows (with and without the host request split), more rows than one logits
chunk holds, and CUDA-graph padding rows (position -1, request 0 as the QSA
metadata builder writes, the last request or out of range). One-row-per-request
batches must keep the paged scorer. Each shape is compared eagerly and through
one captured graph per mode, replayed after the inputs change in place. --time
adds the replay time of both graphs. Exit 1 on any difference.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

import vllm._custom_ops  # noqa: F401  (registers the native top-k)
import vllm.envs as envs
from vllm.models.qwen4_exp.nvidia.ops import qsa
from vllm.models.qwen4_exp.nvidia.ops.qsa_request_segments import (
    request_segments_unsupported,
)

PAGE, WIDTH, CR, TOKEN_TOPK = 416, 316, 4, 2048


def shape(name, widths, contexts, pad=0, pad_kind=None, split=False, expect=None):
    return dict(
        name=name,
        widths=widths,
        contexts=contexts,
        pad=pad,
        pad_kind=pad_kind,
        split=split,
        expect=expect,
    )


SHAPES = [
    *[
        shape(f"c{c}x{r}_{ctx // 1024}k", [r] * c, [ctx] * c)
        for c, r in ((4, 4), (4, 5), (8, 4), (8, 5))
        for ctx in (32768, 131072, 262144)
    ],
    *[
        shape(
            f"c{c}x1_{ctx // 1024}k", [1] * c, [ctx] * c, expect="one_row_per_request"
        )
        for c in (2, 4, 8)
        for ctx in (32768, 131072, 262144)
    ],
    shape("ragged", [1, 4, 5, 8, 3], [32768, 131072, 262144, 65536, 8192]),
    shape("prefill_mix", [200, 4, 4, 4], [40000, 131072, 131072, 131072]),
    shape(
        "prefill_mix_split", [200, 4, 4, 4], [40000, 131072, 131072, 131072], split=True
    ),
    shape("two_chunks", [300, 4, 4], [40000, 131072, 131072]),
    shape("pad_last", [4, 4, 4, 4], [131072] * 4, 8, "last"),
    shape("pad_oob", [4, 4, 4, 4], [131072] * 4, 8, "oob"),
    shape("pad_first", [4, 4, 4, 4], [131072] * 4, 8, "first"),
    # Drafting steps padded to the next CUDA-graph size keep the paged scorer.
    shape("pad_x1_c3", [1] * 3, [131072] * 3, 1, "first", expect="one_row_per_request"),
    shape("pad_x1_c5", [1] * 5, [131072] * 5, 3, "first", expect="one_row_per_request"),
]


def build(s, gen):
    dev = "cuda"
    widths, contexts = s["widths"], s["contexts"]
    nreq = len(widths)
    num_pages = nreq * WIDTH + 3
    k = (torch.randn((num_pages, PAGE, 1, 128), generator=gen, device=dev) * 0.5).half()
    perm = torch.randperm(num_pages, generator=gen, device=dev).int()
    table = torch.full((nreq, WIDTH), -1, dtype=torch.int32, device=dev)
    for r, ctx in enumerate(contexts):
        need = math.ceil(ctx / CR / PAGE)
        table[r, :need] = perm[r * WIDTH : r * WIDTH + need]
    rows = sum(widths) + s["pad"]
    q = (torch.randn((rows, 4, 128), generator=gen, device=dev) * 0.5).half()
    t2r = [torch.full((w,), r, dtype=torch.int32) for r, w in enumerate(widths)]
    pos = [torch.arange(c - w, c, dtype=torch.int64) for w, c in zip(widths, contexts)]
    if s["pad"]:
        pad_req = {"first": 0, "last": nreq - 1, "oob": nreq}[s["pad_kind"]]
        t2r.append(torch.full((s["pad"],), pad_req, dtype=torch.int32))
        pos.append(torch.full((s["pad"],), -1, dtype=torch.int64))
    lens = torch.tensor(contexts, dtype=torch.int32)
    qsl = torch.tensor([0, *torch.tensor(widths).cumsum(0).tolist()], dtype=torch.int32)
    return dict(
        q=q,
        k=k,
        table=table,
        t2r=torch.cat(t2r).to(dev),
        pos=torch.cat(pos).to(dev),
        lens=lens.to(dev),
        qsl=qsl if s["split"] else None,
        max_len=max(contexts),
        num_pages=num_pages,
    )


def select(x, on):
    envs.VLLM_SM70_QSA_REQUEST_SEGMENTS = on
    return qsa.qsa_select_paged_tokens(
        x["q"],
        x["k"],
        x["table"],
        x["t2r"],
        x["pos"],
        x["lens"],
        TOKEN_TOPK,
        CR,
        query_start_loc_cpu=x["qsl"],
        shared_key_scoring=True,
        max_seq_len=x["max_len"] if x["qsl"] is not None else None,
    )


def mutate(x, gen):
    x["q"].copy_((torch.randn(x["q"].shape, generator=gen, device="cuda") * 0.5).half())
    x["k"].copy_((torch.randn(x["k"].shape, generator=gen, device="cuda") * 0.5).half())
    valid = x["table"] >= 0
    x["table"].copy_(torch.where(valid, (x["table"] + 1) % x["num_pages"], x["table"]))
    x["pos"].copy_(torch.where(x["pos"] >= 0, x["pos"] - 37, x["pos"]))
    x["lens"].sub_(37)


def replay_ms(graph, reps=5, iters=50):
    times = []
    for _ in range(reps):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(iters):
            graph.replay()
        end.record()
        torch.accelerator.synchronize()
        times.append(start.elapsed_time(end) / iters)
    return sorted(times)[reps // 2]


def run(s, gen, timing):
    x = build(s, gen)
    # The dispatch sees one logits chunk at a time.
    chunk = max(1, qsa._LOGITS_WORKSPACE_BYTES // (WIDTH * PAGE * 4))
    reasons = [
        request_segments_unsupported(
            x["q"][i : i + chunk],
            x["k"],
            x["table"],
            x["t2r"][i : i + chunk],
            x["pos"][i : i + chunk],
            x["lens"],
        )
        for i in range(0, x["q"].shape[0], chunk)
    ]
    eager = {on: select(x, on) for on in (False, True)}
    torch.accelerator.synchronize()
    eager_equal = bool(torch.equal(eager[False], eager[True]))
    graphs = {}
    for on in (False, True):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(2):
                select(x, on)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = select(x, on)
        graphs[on] = (graph, out)
    rec = dict(
        shape=s["name"],
        widths=s["widths"],
        contexts=s["contexts"],
        padding=s["pad"],
        padding_request=s["pad_kind"],
        host_split=s["split"],
        rows=int(x["q"].shape[0]),
        unsupported_reasons=reasons,
        eager_equal=eager_equal,
    )
    if timing:
        rec["paged_ms"] = round(replay_ms(graphs[False][0]), 4)
        rec["segments_ms"] = round(replay_ms(graphs[True][0]), 4)
        rec["ratio"] = round(rec["segments_ms"] / rec["paged_ms"], 3)
    mutate(x, gen)
    for on in (False, True):
        graphs[on][0].replay()
    torch.accelerator.synchronize()
    fresh = select(x, False)
    torch.accelerator.synchronize()
    rec["graph_after_change_equal"] = bool(
        torch.equal(graphs[False][1], graphs[True][1])
        and torch.equal(graphs[False][1], fresh)
    )
    rec["changed_output"] = not torch.equal(fresh, eager[False])
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--only", default="", help="comma-separated shape names")
    a = ap.parse_args()
    assert torch.cuda.get_device_capability() == (7, 0), "needs SM70"
    gen = torch.Generator(device="cuda")
    gen.manual_seed(20261029)
    wanted = set(filter(None, a.only.split(",")))
    rows, bad = [], []
    for s in SHAPES:
        if wanted and s["name"] not in wanted:
            continue
        rec = run(s, gen, a.time)
        rec["dispatch_as_expected"] = all(
            r == s["expect"] for r in rec["unsupported_reasons"]
        )
        rows.append(rec)
        print(json.dumps(rec), flush=True)
        if not (
            rec["eager_equal"]
            and rec["graph_after_change_equal"]
            and rec["dispatch_as_expected"]
            and rec["changed_output"]
        ):
            bad.append(s["name"])
        torch.accelerator.empty_cache()
    result = dict(
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        qsa_file=qsa.__file__,
        rows=rows,
        mismatch=bad,
        finished=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )
    a.out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps({"summary": "MISMATCH" if bad else "ok", "mismatch": bad}),
        flush=True,
    )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
