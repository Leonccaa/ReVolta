# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU interpreter checks; run directly with TRITON_INTERPRET=1.

The source paged kernel is the arithmetic control. An independent NumPy
integer-valued oracle checks addressing without emulating SM70 MMA rounding.
"""

import ast
import importlib.util
import math
import os
import sys
import unittest
from pathlib import Path
from types import ModuleType
from typing import Any, ClassVar

import numpy as np
import torch

from vllm.triton_utils import tl, triton

ROOT = Path(__file__).resolve().parents[2]
OPS = ROOT / "vllm/models/qwen4_exp/nvidia/ops"


def load_candidate():
    spec = importlib.util.spec_from_file_location(
        "qsa_request_segments_candidate", OPS / "qsa_request_segments.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def paged_kernel():
    path = OPS / "qsa.py"
    tree = ast.parse(path.read_text())
    fn = next(
        x
        for x in tree.body
        if isinstance(x, ast.FunctionDef) and x.name == "_qsa_mqa_paged_kernel"
    )
    namespace = {"tl": tl, "triton": triton}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[fn.name]


def inputs(widths=(5, 3), page=16, columns=67, *, strided=False, normal=False):
    rng = np.random.default_rng(1929)
    m = sum(widths)
    width = math.ceil(columns / page)
    pages = width * len(widths) + 2

    def values(shape):
        if normal:
            return torch.from_numpy(rng.standard_normal(shape).astype(np.float16))
        return torch.from_numpy((rng.integers(-8, 9, shape) / 8).astype(np.float16))

    q = values((m, 4, 256 if strided else 128))
    k = values((pages, page * (2 if strided else 1), 1, 256 if strided else 128))
    table = torch.from_numpy(rng.permutation(pages)[: width * len(widths)].copy()).int()
    table = table.reshape(len(widths), width)
    if strided:
        q, k = q[..., ::2], k[:, ::2, :, ::2]
        temp = torch.full((len(widths), width * 2), -1, dtype=torch.int32)
        temp[:, ::2] = table
        table = temp[:, ::2]
    req = torch.repeat_interleave(
        torch.arange(len(widths), dtype=torch.int32), torch.tensor(widths)
    )
    lens = torch.arange(len(widths), dtype=torch.int32) * 7 + columns * 4 - 3
    pos = torch.cat(
        [
            torch.arange(int(lens[r]) - n, int(lens[r]), dtype=torch.int64)
            for r, n in enumerate(widths)
        ]
    )
    return [q, k, table, req, pos, lens]


def oracle(x, columns):
    q, cache, table, req, pos, lens = [t.numpy() for t in x]
    scores = np.full((len(q), columns), np.nan, np.float32)
    visible = []
    for row, request in enumerate(req):
        length = int(lens[request]) if 0 <= request < len(lens) else 0
        div = lambda n: n // 4 if n >= 0 else -((-n) // 4)
        n = min(div(int(pos[row]) + 1), div(length))
        visible.append(n)
        for col in range(max(0, min(n, columns))):
            physical = int(table[request, col // cache.shape[1]])
            if not 0 <= physical < len(cache):
                scores[row, col] = -np.inf
                continue
            key = cache[physical, col % cache.shape[1], 0].astype(np.float32)
            # Exact dyadic products in the addressing fixtures avoid rounding ambiguity.
            heads = (q[row].astype(np.float32) * key).sum(1).clip(min=0)
            scores[row, col] = ((heads[0] + heads[1]) + heads[2]) + heads[3]
            scores[row, col] /= np.float32(math.sqrt(128))
    return torch.from_numpy(scores), torch.tensor(visible, dtype=torch.int32)


def top512(scores, visible):
    out = torch.full((len(scores), 512), -1, dtype=torch.int32)
    for row, n in enumerate(visible.tolist()):
        n = max(0, min(n, scores.shape[1]))
        ids = torch.argsort(scores[row, :n], descending=True, stable=True)[:512]
        out[row, : len(ids)] = ids.sort().values.int()
    return out


def launch_baseline(kernel, x, columns):
    q, k, table, req, pos, lens = x
    # The paged kernel has no row-metadata strides; normalize only its control.
    req, pos, lens = (t.contiguous() for t in (req, pos, lens))
    out = torch.full((len(q), columns), float("nan"), device=q.device)
    visible = torch.empty(len(q), dtype=torch.int32, device=q.device)
    if columns and len(q):
        kernel[(len(q), triton.cdiv(columns, 64))](
            q,
            k,
            table,
            req,
            pos,
            lens,
            visible,
            out,
            *q.stride(),
            k.stride(0),
            k.stride(1),
            k.stride(3),
            *table.stride(),
            out.stride(0),
            len(q),
            columns,
            len(k),
            len(table),
            math.sqrt(128),
            PAGE_SIZE=k.shape[1],
            PAGE_TABLE_WIDTH=table.shape[1],
            NUM_HEADS=4,
            HEAD_DIM=128,
            BLOCK_N=64,
            BLOCK_D=128,
            TILES_PER_PROG=1,
            STAGES=2,
            MAX_N=16,
            COMPRESS_RATIO=4,
            num_warps=2,
        )
    return out, visible


@unittest.skipUnless(
    os.environ.get("TRITON_INTERPRET") == "1", "CPU interpreter required"
)
class RequestSegmentTests(unittest.TestCase):
    mod: ClassVar[ModuleType]
    baseline: ClassVar[Any]

    @classmethod
    def setUpClass(cls):
        cls.mod = load_candidate()
        cls.baseline = staticmethod(paged_kernel())

    def check_case(self, x, columns, *, max_rows=4, check_oracle=True):
        ws = self.mod.SegmentWorkspace.allocate(x[0], columns)
        ws.scores.fill_(float("nan"))
        ws.descriptors.fill_(991)
        actual, visible = self.mod.qsa_mqa_request_segments(
            *x, 4, ws, max_rows=max_rows
        )
        if columns:
            ref, refvis = launch_baseline(self.baseline, x, columns)
            self.assertTrue(torch.equal(visible, refvis))
            mask = torch.arange(columns)[None, :] < visible[:, None]
            self.assertTrue(
                torch.equal(actual.view(torch.int32)[mask], ref.view(torch.int32)[mask])
            )
            self.assertTrue(torch.isnan(actual[~mask]).all())
            self.assertTrue(torch.equal(top512(actual, visible), top512(ref, refvis)))
        if check_oracle:
            ref, refvis = oracle(x, columns)
            self.assertTrue(torch.equal(visible, refvis))
            mask = torch.arange(columns)[None, :] < visible[:, None]
            self.assertTrue(
                torch.equal(actual.view(torch.int32)[mask], ref.view(torch.int32)[mask])
            )
        # An independent CPU run splitter verifies one owner for every query row.
        owners = torch.zeros(len(x[0]), dtype=torch.int32)
        expected = torch.full_like(ws.descriptors, -1)
        begin = 0
        requests = x[3].tolist()
        while begin < len(requests):
            end = begin + 1
            while (
                end < len(requests)
                and requests[end] == requests[begin]
                and end - begin < max_rows
            ):
                end += 1
            expected[begin] = torch.tensor([requests[begin], begin, end])
            owners[begin:end] += 1
            begin = end
        self.assertTrue(torch.equal(ws.descriptors, expected))
        self.assertTrue(torch.equal(owners, torch.ones_like(owners)))
        return ws

    def test_ragged_split_and_layouts(self):
        for widths in ((8,), (4, 4, 4, 4), (5, 5, 5, 5), (17, 3), (1, 1, 1)):
            for max_rows in (1, 2, 4):
                with self.subTest(widths=widths, max_rows=max_rows):
                    self.check_case(inputs(widths), 67, max_rows=max_rows)

    def test_page_tails_truncated_width_and_strides(self):
        for page in (16, 204, 416):
            for columns in (1, 63, 65, page + 3):
                with self.subTest(page=page, columns=columns):
                    self.check_case(
                        inputs(page=page, columns=columns, strided=True), columns
                    )

    def test_invalid_pages_requests_and_negative_positions(self):
        x = inputs((9, 7, 3), columns=521)
        x[2][:, ::3] = -1
        x[2][:, 1::7] = len(x[1])
        x[3][::5] = -1
        x[3][2::7] = len(x[2])
        x[4][::3] = torch.tensor([-2, -3, -4, -5, -6, -1, -7])[: len(x[4][::3])]
        self.check_case(x, 521)

    def test_strided_row_metadata(self):
        x = inputs((9, 3), columns=67)
        for i in (3, 4, 5):
            storage = torch.empty(x[i].numel() * 2, dtype=x[i].dtype)
            storage[::2] = x[i]
            x[i] = storage[::2]
        self.check_case(x, 67)

    def test_repeated_interleaved_request_runs(self):
        x = inputs((5, 5, 5), columns=529)
        x[3].copy_(torch.tensor([0, 1, 0, 0, 2, 2, 1, 0, 2, 1, 1, 1, 0, 2, 0]))
        self.check_case(x, 529)

    def test_ties_and_zero_visibility(self):
        x = inputs(columns=529)
        x[0].zero_()
        self.check_case(x, 529)
        x[5].zero_()
        self.check_case(x, 529)

    def test_random_fp16_paged_control(self):
        for dtype in (torch.int32, torch.int64):
            x = inputs(columns=529, normal=True)
            x[4] = x[4].to(dtype)
            self.check_case(x, 529, check_oracle=False)

    def test_zero_columns_and_empty_queries(self):
        self.check_case(inputs(columns=1), 0)
        x = inputs(columns=1)
        x[0], x[3], x[4] = x[0][:0], x[3][:0], x[4][:0]
        self.check_case(x, 1)

    def test_fixed_capacity_builder(self):
        for widths in ((40,), (5,) * 8, (256,), (64, 33, 7, 2)):
            with self.subTest(widths=widths):
                self.check_case(inputs(widths, columns=1), 0)

    def test_workspace_reuse_with_changed_request_runs(self):
        x = inputs((5, 5), columns=529)
        ws = self.mod.SegmentWorkspace.allocate(x[0], 529)
        pointers = [t.data_ptr() for t in (ws.scores, ws.visible, ws.descriptors)]
        for requests in ([0] * 10, [0, 1] * 5, [1] * 4 + [-1] * 3 + [0] * 3):
            x[3].copy_(torch.tensor(requests))
            x[0].neg_()
            x[1].neg_()
            x[2].copy_(x[2].flip(1))
            x[4].sub_(4)
            x[5].sub_(4)
            ws.scores.fill_(float("nan"))
            out, vis = self.mod.qsa_mqa_request_segments(*x, 4, ws)
            ref, rv = launch_baseline(self.baseline, x, 529)
            self.assertTrue(torch.equal(vis, rv))
            mask = torch.arange(529)[None, :] < vis[:, None]
            self.assertTrue(
                torch.equal(out.view(torch.int32)[mask], ref.view(torch.int32)[mask])
            )
            self.assertEqual(
                pointers,
                [t.data_ptr() for t in (ws.scores, ws.visible, ws.descriptors)],
            )
            expected = self.check_case(x, 529)
            self.assertTrue(torch.equal(ws.descriptors, expected.descriptors))

    def test_reject_invalid_contract(self):
        x = inputs()
        ws = self.mod.SegmentWorkspace.allocate(x[0], 67)
        for kwargs in (
            {"max_rows": 3},
            {"max_rows": 8},  # two query blocks spill on SM70; not supported
            {"score_divisor": 0},
            {"score_divisor": float("nan")},
        ):
            with self.assertRaises(ValueError):
                self.mod.qsa_mqa_request_segments(*x, 4, ws, **kwargs)
        bad = [*x]
        bad[0] = bad[0].float()
        with self.assertRaises(ValueError):
            self.mod.qsa_mqa_request_segments(*bad, 4, ws)
        ws.descriptors = ws.descriptors.long()
        with self.assertRaises(ValueError):
            self.mod.qsa_mqa_request_segments(*x, 4, ws)


if __name__ == "__main__":
    unittest.main(verbosity=2)
