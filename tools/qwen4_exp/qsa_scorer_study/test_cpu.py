# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU contract and adversarial checks; synthetic captures do not test CUDA."""

import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .fixtures import Case, make_inputs, quick_cases
from .prototype import (
    Inputs,
    Result,
    assert_equivalent,
    launch_geometry,
    make_segments,
    request_segmented,
    row_reference,
    segment_descriptors,
    select_topk,
    tile_grouped,
    visible_lengths,
)
from .run import (
    compare_gpu_captures,
    export_inputs,
    input_digest,
    load_inputs,
    prepare_matrix,
)


class ScorerTest(unittest.TestCase):
    def compare_all(self, x):
        expected = row_reference(x)
        for group in (1, 4, 8):
            assert_equivalent(expected, tile_grouped(x, group))
        assert_equivalent(expected, request_segmented(x))
        return expected

    def test_hand_computed_scores(self):
        q = np.zeros((2, 4, 128), np.float16)
        q[:, 0] = 1
        q[:, 1] = -1
        cache = np.ones((2, 16, 1, 128), np.float16)
        cache[1] = 2
        x = Inputs(
            q,
            cache,
            np.array([[0, 1]], np.int32),
            np.zeros(2, np.int32),
            np.array([127, 79], np.int64),
            np.array([128], np.int32),
            32,
            divisor=1,
        )
        result = self.compare_all(x)
        np.testing.assert_array_equal(result.scores[0, :16], 128)
        np.testing.assert_array_equal(result.scores[0, 16:], 256)
        np.testing.assert_array_equal(result.scores[1, :16], 128)
        np.testing.assert_array_equal(result.scores[1, 16:20], 256)
        self.assertEqual(result.writes[1, 20:].sum(), 0)

    def test_awkward_layouts_and_request_runs(self):
        names = {"strided", "interleaved", "padding_negative", "split_long_run"}
        for case in quick_cases():
            if case.name in names:
                with self.subTest(case=case.name):
                    x = make_inputs(case)
                    if case.strided:
                        self.assertFalse(x.q.flags.c_contiguous)
                        self.assertFalse(x.cache.flags.c_contiguous)
                        self.assertFalse(x.table.flags.c_contiguous)
                    self.compare_all(x)

    def test_signed_division_and_invalid_requests(self):
        x = make_inputs(Case("negative", (5,), (10,), 16))
        x = replace(x, positions=np.array([-2, -3, -4, -5, -6], np.int64))
        np.testing.assert_array_equal(visible_lengths(x), [0, 0, 0, -1, -1])
        self.compare_all(x)
        x = replace(x, requests=np.array([-1, 1, 0, 0, 0], np.int32))
        np.testing.assert_array_equal(visible_lengths(x), [0, 0, 0, -1, -1])
        self.compare_all(x)

    def test_head_reduction_tree_is_part_of_the_contract(self):
        x = make_inputs(Case("rounding", (2,), (4,), 1))
        x.q.fill(0)
        x.cache.fill(0)
        x.q[:, 0, 0] = 4096
        x.q[:, 1:, 0] = np.float16(2**-12)
        x.cache[:, :, 0, 0] = 4096
        result = self.compare_all(
            replace(x, divisor=1, positions=np.full(2, 3, np.int64))
        )
        self.assertEqual(result.scores[0, 0], np.float32(2**24))
        balanced = np.float32(np.float32(2**24 + 1) + np.float32(1 + 1))
        self.assertNotEqual(result.scores[0, 0], balanced)

    def test_empty_query_batch(self):
        x = Inputs(
            np.empty((0, 4, 128), np.float16),
            np.empty((0, 16, 1, 128), np.float16),
            np.empty((0, 0), np.int32),
            np.empty(0, np.int32),
            np.empty(0, np.int64),
            np.empty(0, np.int32),
            0,
        )
        self.compare_all(x)

    def test_invalid_pages_and_top512_ties(self):
        x = make_inputs(
            Case("ties", (5, 5), (3100, 3100), 1025, invalid_pages=True, ties=True)
        )
        result = self.compare_all(x)
        # Independent scalar selection includes -inf invalid-page scores, then
        # uses the original logical column as the tie breaker.
        for row, visible in enumerate(result.visible):
            ids = sorted(
                range(int(visible)), key=lambda c: (-float(result.scores[row, c]), c)
            )[:512]
            np.testing.assert_array_equal(select_topk(result)[row], sorted(ids))

    def test_each_visible_column_written_once(self):
        # Two rows straddle every 64-key and grouped-program tail. A page size
        # of 16 also puts page boundaries inside a scoring tile.
        for columns in (1, 63, 64, 65, 255, 256, 257, 511, 512, 513):
            x = make_inputs(
                Case("tail", (2,), (4 * (columns + 3),), columns, page_size=16)
            )
            expected = self.compare_all(x)
            np.testing.assert_array_equal(expected.writes, 1)

    def test_segments_partition_valid_rows(self):
        rng = np.random.default_rng(41)
        for rows in (0, 1, 5, 8, 9, 17, 20, 35, 40, 64):
            for _ in range(30):
                requests = rng.integers(-1, 5, rows, dtype=np.int32)
                counts = np.zeros(rows, np.int32)
                segments = make_segments(requests, 4)
                self.assertLessEqual(len(segments), rows)
                for segment in segments:
                    self.assertTrue(1 <= segment.end - segment.begin <= 8)
                    np.testing.assert_array_equal(
                        requests[segment.begin : segment.end], segment.request
                    )
                    counts[segment.begin : segment.end] += 1
                np.testing.assert_array_equal(
                    counts, ((requests >= 0) & (requests < 4)).astype(np.int32)
                )
        # This long run cannot be represented as one native M>8 call.
        self.assertEqual(
            [(s.begin, s.end) for s in make_segments(np.zeros(20, np.int32), 1)],
            [(0, 8), (8, 16), (16, 20)],
        )

    def test_fixed_capacity_descriptors_and_metadata_changes(self):
        requests = np.repeat(np.arange(4, dtype=np.int32), 5)
        descriptors = segment_descriptors(requests, 4)
        self.assertEqual(descriptors.shape, (20, 3))
        self.assertEqual(np.count_nonzero(descriptors[:, 0] >= 0), 4)
        requests[::2] = -1
        changed = segment_descriptors(requests, 4)
        self.assertEqual(changed.shape, descriptors.shape)
        self.assertGreater(np.count_nonzero(changed[:, 0] >= 0), 4)
        with self.assertRaises(ValueError):
            segment_descriptors(requests, 4, capacity=4)

    def test_request_isolation_and_wrong_table_negative_control(self):
        x = make_inputs(Case("isolation", (5, 5), (3100, 3100), 1025))
        expected = row_reference(x)
        wrong = request_segmented(replace(x, table=x.table[::-1]))
        with self.assertRaises(AssertionError):
            assert_equivalent(expected, wrong)
        cache = x.cache.copy()
        pages = x.table[1][x.table[1] >= 0]
        cache[pages] *= np.float16(2)
        changed = request_segmented(replace(x, cache=cache))
        np.testing.assert_array_equal(
            changed.scores[:5].view(np.uint32), expected.scores[:5].view(np.uint32)
        )
        self.assertTrue(
            np.any(
                changed.scores[5:].view(np.uint32)
                != expected.scores[5:].view(np.uint32)
            )
        )

    def test_comparison_detects_corruption_and_padding_writes(self):
        x = make_inputs(Case("corrupt", (2,), (100,), 65))
        expected = row_reference(x)
        for column in (0, 64):
            wrong = Result(
                expected.scores.copy(), expected.visible.copy(), expected.writes.copy()
            )
            wrong.scores.view(np.uint32)[0, column] ^= np.uint32(1)
            with self.assertRaises(AssertionError):
                assert_equivalent(expected, wrong)
        wrong = Result(
            expected.scores.copy(), expected.visible.copy(), expected.writes.copy()
        )
        wrong.writes[0, 0] = 2
        with self.assertRaises(AssertionError):
            assert_equivalent(expected, wrong)

    def test_bad_input_contracts(self):
        x = make_inputs(Case("bad", (2,), (100,), 65))
        for changes in (
            {"compress": 0},
            {"divisor": 0},
            {"divisor": float("nan")},
            {"columns": -1},
            {"columns": 999999},
            {"q": x.q.astype(np.float32)},
            {"requests": x.requests.astype(float)},
        ):
            with self.subTest(changes=list(changes)), self.assertRaises(ValueError):
                row_reference(replace(x, **changes))
        for rows in (0, 9):
            with self.assertRaises(ValueError):
                make_segments(x.requests, 1, rows)

    def test_geometry_and_observed_grid_envelope(self):
        # Use a closed-form independent count, including zero visible rows.
        for columns in (0, 1, 63, 64, 65, 8192, 131456):
            for visible in (-6, 0, 1, 63, 64, 65, 8293, columns, columns + 17):
                plans = launch_geometry(
                    np.zeros(16, np.int32), np.full(16, visible), columns, 1
                )
                end = min(columns, max(0, visible))
                for group in (1, 4, 8):
                    width = 64 * group
                    p = plans[f"tile{group}"]
                    self.assertEqual(
                        p["launched"], 16 * ((columns + width - 1) // width)
                    )
                    self.assertEqual(p["active"], 16 * ((end + width - 1) // width))
                    self.assertEqual(p["empty"], p["launched"] - p["active"])
        example = launch_geometry(
            np.repeat(np.arange(4), 4), np.full(16, 8293), 131456, 4
        )
        self.assertEqual(
            example["tile1"], {"launched": 32864, "active": 2080, "empty": 30784}
        )
        self.assertEqual(example["tile4"]["launched"], 8224)
        self.assertEqual(example["tile8"]["launched"], 4112)

    def test_export_restores_strides_and_detects_changed_input(self):
        x = make_inputs(Case("export", (4, 5), (2500, 3200), 1031, strided=True))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.npz"
            export_inputs(path, x, row_reference(x))
            got = load_inputs(path)
            self.assertEqual(input_digest(x), input_digest(got))
            for name in ("q", "cache", "table"):
                self.assertEqual(getattr(x, name).strides, getattr(got, name).strides)
            got.q[0, 0, 0] += np.float16(1)
            self.assertNotEqual(input_digest(x), input_digest(got))

    def test_capture_comparator_with_synthetic_data(self):
        x = make_inputs(Case("capture", (2,), (100,), 65))
        expected = row_reference(x)
        fields: dict[str, Any] = {
            "fixture_sha256": "a" * 64,
            "experiment_id": "synthetic-reader-test",
            "scores": expected.scores.copy(),
            "visible": expected.visible,
            "top512": select_topk(expected),
        }
        with tempfile.TemporaryDirectory() as directory:
            baseline, candidate = (
                Path(directory) / "base.npz",
                Path(directory) / "got.npz",
            )
            np.savez(baseline, **fields)
            fields["scores"][0, 64] = (
                123  # Undefined padding is ignored for GPU captures.
            )
            np.savez(candidate, **fields)
            self.assertTrue(compare_gpu_captures(baseline, candidate)["qualified"])
            fields["scores"].view(np.uint32)[0, 0] ^= np.uint32(1)
            np.savez(candidate, **fields)
            self.assertFalse(compare_gpu_captures(baseline, candidate)["qualified"])
            fields["fixture_sha256"] = "b" * 64
            np.savez(candidate, **fields)
            with self.assertRaises(ValueError):
                compare_gpu_captures(baseline, candidate)

    def test_matrix_is_deferred_and_requires_no_runtime(self):
        matrix = prepare_matrix()
        self.assertEqual(len(matrix), 12)
        self.assertTrue(all(c["gpu_validation"] == "not_run" for c in matrix))
        for name in ("torch", "vllm", "triton"):
            self.assertNotIn(name, sys.modules)


if __name__ == "__main__":
    unittest.main()
