# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run CPU comparisons, export inputs, and compare deferred GPU captures."""

import argparse
import hashlib
import json
from dataclasses import asdict
from functools import partial
from pathlib import Path

import numpy as np

from .fixtures import geometry_cases, long_cases, make_inputs, quick_cases
from .prototype import (
    Array,
    Inputs,
    Result,
    assert_equivalent,
    launch_geometry,
    request_segmented,
    row_reference,
    select_topk,
    tile_grouped,
)

ARRAY_NAMES = ("q", "cache", "table", "requests", "positions", "lengths")


def input_digest(x: Inputs) -> str:
    h = hashlib.sha256()
    for name in ARRAY_NAMES:
        a = getattr(x, name)
        h.update(json.dumps([name, a.dtype.str, a.shape, a.strides]).encode())
        h.update(a.tobytes())
    h.update(json.dumps([x.columns, x.compress, x.divisor]).encode())
    return h.hexdigest()


def export_inputs(path: Path, x: Inputs, reference: Result) -> None:
    layout = {name: list(getattr(x, name).strides) for name in ARRAY_NAMES}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        **{name: getattr(x, name) for name in ARRAY_NAMES},
        columns=x.columns,
        compress=x.compress,
        divisor=x.divisor,
        layout=json.dumps(layout),
        fixture_sha256=input_digest(x),
        cpu_reference_scores=reference.scores,
        cpu_reference_visible=reference.visible,
        cpu_reference_top512=select_topk(reference),
    )


def load_inputs(path: Path) -> Inputs:
    """Restore positive-stride layouts; np.savez alone loses those strides."""
    with np.load(path, allow_pickle=False) as data:
        layout = json.loads(str(data["layout"]))
        arrays = []
        for name in ARRAY_NAMES:
            values = data[name]
            strides = tuple(layout[name])
            if len(strides) != values.ndim or any(s < 0 for s in strides):
                raise ValueError("unsupported fixture strides")
            span = values.dtype.itemsize + sum(
                max(0, n - 1) * s for n, s in zip(values.shape, strides)
            )
            array: Array = np.ndarray(
                values.shape, values.dtype, buffer=bytearray(span), strides=strides
            )
            array[...] = values
            arrays.append(array)
        x = Inputs(
            q=arrays[0],
            cache=arrays[1],
            table=arrays[2],
            requests=arrays[3],
            positions=arrays[4],
            lengths=arrays[5],
            columns=int(data["columns"]),
            compress=int(data["compress"]),
            divisor=float(data["divisor"]),
        )
        x.validate()
        if input_digest(x) != str(data["fixture_sha256"]):
            raise ValueError("fixture digest mismatch")
    return x


def compare_gpu_captures(baseline: Path, candidate: Path) -> dict:
    """Compare GPU with GPU; the CPU score oracle is never a GPU bitwise gate.

    Capture producers must provide the same fixture digest and experiment id.
    This offline reader validates arrays, not the truth of capture provenance.
    """
    with (
        np.load(baseline, allow_pickle=False) as base,
        np.load(candidate, allow_pickle=False) as got,
    ):
        for key in ("fixture_sha256", "experiment_id"):
            if not str(base[key]) or str(base[key]) != str(got[key]):
                raise ValueError(f"capture {key} mismatch")
        scores, actual = base["scores"], got["scores"]
        visible, av = base["visible"], got["visible"]
        selected, ids = base["top512"], got["top512"]
        if scores.dtype != np.float32 or actual.dtype != np.float32:
            raise ValueError("captures must contain FP32 scores")
        if scores.shape != actual.shape or scores.ndim != 2:
            raise ValueError("capture score shape mismatch")
        if visible.shape != (len(scores),) or av.shape != visible.shape:
            raise ValueError("capture visible shape mismatch")
        if visible.dtype.kind not in "iu" or av.dtype.kind not in "iu":
            raise ValueError("capture visible lengths must be integers")
        if selected.shape != (len(scores), 512) or ids.shape != selected.shape:
            raise ValueError("capture selection shape mismatch")
        if selected.dtype.kind not in "iu" or ids.dtype.kind not in "iu":
            raise ValueError("capture selections must be integers")
        mask = np.arange(scores.shape[1])[None, :] < visible[:, None]
        compared = int(mask.sum())
        bit_differences = int(
            np.count_nonzero(
                scores.view(np.uint32)[mask] != actual.view(np.uint32)[mask]
            )
        )
        visible_equal = bool(np.array_equal(visible, av))
        selected_equal = bool(np.array_equal(selected, ids))
        return {
            "fixture_sha256": str(base["fixture_sha256"]),
            "experiment_id": str(base["experiment_id"]),
            "compared_score_words": compared,
            "different_score_words": bit_differences,
            "visible_equal": visible_equal,
            "top512_equal": selected_equal,
            "qualified": bit_differences == 0 and visible_equal and selected_equal,
            "timing_evaluated": False,
        }


def prepare_matrix() -> list[dict]:
    out = []
    for case in geometry_cases():
        plans = launch_geometry(
            np.asarray(case["requests"]),
            np.asarray(case["visible"]),
            case["columns"],
            len(case["contexts"]),
        )
        out.append({**case, "plans": plans})
    return out


def check(args: argparse.Namespace) -> dict:
    cases = []
    if args.suite in ("quick", "full"):
        cases += quick_cases()
    if args.suite in ("long", "full"):
        cases += long_cases()
    if args.case:
        cases = [c for c in cases if c.name == args.case]
        if not cases:
            raise ValueError("unknown case in selected suite")
    results = []
    totals = {"score_words": 0, "visible_score_words": 0, "top512_rows": 0}
    for index, case in enumerate(cases):
        x = make_inputs(case, seed=1029 + index)
        expected = row_reference(x)
        variants = {}
        for name, implementation in [
            ("tile1", partial(tile_grouped, x, 1)),
            ("tile4", partial(tile_grouped, x, 4)),
            ("tile8", partial(tile_grouped, x, 8)),
            ("segmented", partial(request_segmented, x)),
        ]:
            actual = implementation()
            counts = assert_equivalent(expected, actual)
            for key in totals:
                totals[key] += counts[key]
            variants[name] = {
                **counts,
                "logical_key_fetches": actual.logical_key_fetches,
                "launched_programs": actual.launched_programs,
                "active_programs": actual.active_programs,
            }
            print(f"PASS {case.name} {name}", flush=True)
        if args.export:
            export_inputs(Path(args.export) / f"{case.name}.npz", x, expected)
            roundtrip = load_inputs(Path(args.export) / f"{case.name}.npz")
            assert input_digest(roundtrip) == input_digest(x)
        results.append(
            {
                "case": asdict(case),
                "input_sha256": input_digest(x),
                "reference_logical_key_fetches": expected.logical_key_fetches,
                "variants": variants,
            }
        )
    return {
        "backend": "numpy_cpu",
        "numeric_cases": len(results),
        "candidate_comparisons": len(results) * 4,
        "all_equal": True,
        "totals": totals,
        "cases": results,
        "gpu_matrix": prepare_matrix(),
        "gpu_numerics_validated": False,
        "gpu_performance_validated": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("check", help="compare CPU algorithms")
    run.add_argument("--suite", choices=("quick", "long", "full"), default="quick")
    run.add_argument("--case")
    run.add_argument("--export", help="export NPZ input fixtures with layout metadata")
    matrix = sub.add_parser("matrix", help="prepare deferred GPU shapes without Q/K")
    compare = sub.add_parser(
        "compare-gpu", help="compare baseline/candidate NPZ captures"
    )
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    for cmd in (run, matrix, compare):
        cmd.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.command == "check":
        result = check(args)
    elif args.command == "matrix":
        result = {"gpu_matrix": prepare_matrix(), "gpu_validation": "not_run"}
    else:
        result = compare_gpu_captures(args.baseline, args.candidate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote {args.output}", flush=True)
    if result.get("qualified") is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
