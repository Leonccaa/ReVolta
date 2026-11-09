# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the runnable long-context benchmark profiles and traces."""

import argparse
import json

import pytest
import torch

from tools.benchmark_host_kv import main, make_config, make_trace


def _args(**kwargs):
    values = dict(
        tokens=262144,
        queries=8,
        batch_queries=2,
        union_groups=1536,
        churn_fraction=0.2,
        steps=3,
        warmup=1,
        layers=1,
        hot_ratio=0.1,
        scenario="churn",
        eviction_policy="bucket",
    )
    values.update(kwargs)
    return argparse.Namespace(**values)


@pytest.mark.parametrize("ratio", [0.1, 0.2])
@pytest.mark.parametrize("queries,union", [(6, 1100), (8, 1536), (8, 1600)])
def test_eight_request_benchmark_preserves_rows_and_exact_step_union(
    ratio, queries, union
):
    args = _args(hot_ratio=ratio, queries=queries, union_groups=union)
    config = make_config(args)
    trace = make_trace(args, config)
    assert trace.shape == (4, queries // 2, 16, 2051)
    assert config.capacities()[0] >= config.required_groups
    for step in trace:
        rows = (
            step.reshape(queries // 2, 8, 2, 2051)
            .permute(1, 0, 2, 3)
            .reshape(8, queries, 2051)
        )
        assert (rows[:, :, -3:] == -1).all()
        for req in range(8):
            selected = rows[req, :, :2048]
            assert selected.min() >= 0
            assert selected.max() < config.max_tokens - 32
            assert (selected // 4).unique().numel() == union
            expanded = selected.reshape(queries, 512, 4)
            assert torch.equal(expanded % 4, torch.arange(4).expand_as(expanded))


def test_warm_trace_reuses_identical_groups_and_rejects_invalid_batches():
    args = _args(scenario="warm")
    trace = make_trace(args, make_config(args))
    assert torch.equal(trace[0], trace[-1])
    for changes in (
        {"batch_queries": 0},
        {"queries": 7},
        {"union_groups": 4096, "queries": 6},
    ):
        with pytest.raises(ValueError):
            make_config(_args(**changes))


@pytest.mark.parametrize("ratio", [0.1, 0.2])
def test_dry_run_never_initializes_cuda(monkeypatch, capsys, ratio):
    def fail_cuda():
        raise AssertionError("dry-run accessed CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", fail_cuda)
    monkeypatch.setattr(
        "sys.argv", ["benchmark", "--hot-ratio", str(ratio), "--dry-run"]
    )
    main()
    summary = json.loads(capsys.readouterr().out)
    assert summary["cold_bound_per_batch"] == 8280
    assert summary["union_groups_all_requests"] == 12288
    assert summary["gather_workers_per_layer"] == 256


@pytest.mark.parametrize("rows", [1, 8])
def test_twelve_layer_profile_and_parameterized_ratio(rows):
    args = _args(
        queries=rows,
        batch_queries=rows,
        layers=12,
        hot_ratio=0.075,
        union_groups=512 if rows == 1 else 1536,
    )
    config = make_config(args)
    assert config.qsa_block_topk == 512
    assert config.rows == rows * 8
    assert config.required_groups == rows * 8 * 513 + 8 * 9
    assert config.capacities()[0] == 39322
    assert make_trace(args, config).shape == (4, 1, rows * 8, 2051)


def test_compare_lru_reports_both_miss_fractions(monkeypatch, capsys):
    calls = []

    def run(args, config):
        calls.append(config)
        fraction = 0.1 if config.eviction_policy == "lru" else 0.12
        return {"results": {"graph_phases": {"miss_fraction": fraction}}}

    monkeypatch.setattr("tools.benchmark_host_kv.run_benchmark", run)
    monkeypatch.setattr(
        "sys.argv", ["benchmark", "--hot-ratio", "0.1", "--compare-lru"]
    )
    main()
    summary_line, report_text = capsys.readouterr().out.split("\n", 1)
    assert json.loads(summary_line)["cold_bound_per_batch"] == 8280
    report = json.loads(report_text)
    assert [config.eviction_policy for config in calls] == ["lru", "bucket"]
    assert report["miss_comparison"]["exact_lru"] == 0.1
    assert report["miss_comparison"]["bucket"] == 0.12
    assert report["miss_comparison"][
        "bucket_minus_lru_percentage_points"
    ] == pytest.approx(2.0)
