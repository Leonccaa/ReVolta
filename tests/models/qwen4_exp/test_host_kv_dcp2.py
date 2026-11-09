# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Four real gloo processes, original kernels interpreted on CPU, no GPU."""

import json
import os
import time
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tools.benchmark_host_kv_dcp2 import (
    aggregate_bandwidth,
    check_results,
    make_profile,
    parse_args,
    run_experiment,
)
from tools.host_kv_dcp2 import (
    ExperimentGroup,
    Profile,
    RankFailure,
    enable_cpu_interpreter,
    error_metrics,
    local_length,
)
from vllm.v1.attention.ops.dcp_alltoall import _lse_weighted_combine

pytestmark = [
    pytest.mark.skip_global_cleanup,
    pytest.mark.skipif(
        os.environ.get("TRITON_INTERPRET") != "1", reason="CPU interpreter required"
    ),
]


def _worker(rank, root, options):
    torch.set_num_threads(1)
    enable_cpu_interpreter()
    dist.init_process_group(
        "gloo",
        init_method=f"file://{root}/rendezvous",
        rank=rank,
        world_size=4,
        timeout=timedelta(seconds=30),
    )
    try:
        group = ExperimentGroup("gloo", timeout=30)
        profile = Profile(
            tokens=4097,
            queries=options["queries"],
            batch_queries=1,
            topk=4,
            union_groups=2 if options["queries"] == 8 else 1,
            dim=16,
            heads=2,
            page=16,
            hot_ratio=options["ratio"],
            e4m3=options["e4m3"],
        )
        try:
            result = run_experiment(
                group,
                profile,
                device=torch.device("cpu"),
                steps=2,
                warmup=1,
                scenario=options["scenario"],
                edge=True,
                fail_rank=options.get("fail_rank", -1),
                fail_phase=options.get("fail_phase", "metadata"),
            )
        except RankFailure as exc:
            if "fail_rank" not in options:
                raise
            result = {"failure": str(exc)}
        if "fail_rank" not in options:
            _independent_collective_oracle(group)
        Path(root, f"rank-{rank}.json").write_text(json.dumps(result))
    finally:
        dist.destroy_process_group()


def _independent_collective_oracle(group):
    # Each query rank owns distinct heads. Verify head order independently of
    # attention and verify the production base-2 combine against torch math.
    query = torch.full((3, 2, 16), float(group.rank_in_group))
    gathered = group.all_gather(query, dim=1)
    assert torch.equal(gathered[:, :2], torch.zeros_like(query))
    assert torch.equal(gathered[:, 2:], torch.ones_like(query))
    outputs = torch.arange(2 * 3 * 4 * 16).reshape(2, 3, 4, 16).float() / 100
    lses = torch.arange(24).reshape(2, 3, 4).float() / 4
    outputs[:, 2] = 0
    lses[:, 2] = -torch.inf
    lses[1, 0] = -torch.inf  # rank 1 owns no visible KV in row zero
    expected, expected_lse = _lse_weighted_combine(
        outputs,
        lses,
        return_lse=True,
        is_lse_base_on_e=False,
    )
    actual, actual_lse = group.combine(
        outputs[group.rank_in_group].clone(),
        lses[group.rank_in_group].clone(),
        "ag_rs",
    )
    head = slice(group.rank_in_group * 2, (group.rank_in_group + 1) * 2)
    torch.testing.assert_close(actual, expected[:, head], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(actual_lse, expected_lse[:, head], atol=2e-6, rtol=2e-6)
    assert torch.isneginf(actual_lse[2]).all()
    assert not actual[2].any()


def _launch(tmp_path, **options):
    # Fork inherits the CPU platform fixture from the dedicated runner. No
    # accelerator context exists. Each child still creates a real gloo rank.
    processes = mp.start_processes(
        _worker,
        args=(str(tmp_path), options),
        nprocs=4,
        join=False,
        start_method="fork",
    )
    deadline = time.monotonic() + 150
    try:
        while not processes.join(timeout=1):
            if time.monotonic() > deadline:
                raise AssertionError("four-rank experiment exceeded 150 seconds")
    finally:
        for process in processes.processes:
            if process.is_alive():
                process.terminate()
            process.join(timeout=5)
    return [
        json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(4)
    ]


@pytest.mark.parametrize("queries", [1, 8])
@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize("ratio", [0.1, 0.2])
def test_four_rank_abc_payload_lse_gate_and_changed_inputs(
    tmp_path, queries, e4m3, ratio
):
    # Each run covers warm-up, changed query/gate, eviction pressure and two
    # recorded steps. The separate scenario tests check reset semantics.
    results = _launch(
        tmp_path, queries=queries, e4m3=e4m3, ratio=ratio, scenario="churn"
    )
    for rank, result in enumerate(results):
        assert result["rank"] == rank
        assert result["profile"]["tokens"] % 2 == 1
        assert len(result["records"]) == 2
        for record in result["records"]:
            assert record["wanted"] > 0
            assert record["original_full_rows"] == 8 * queries
            assert all(
                value["bitwise"]
                for value in record["original_plan_comparison"].values()
            )
            assert "dcp_combine_gate_ms" in record["phases"]
            for batch in record["comparisons"]:
                assert all(
                    value["bitwise"] for arm in batch.values() for value in arm.values()
                )
    bandwidth = aggregate_bandwidth(results)
    assert len(bandwidth) == 2
    assert all(
        step["ranks"] == 4 and step["world_payload_wall_ms"] > 0 for step in bandwidth
    )


@pytest.mark.parametrize("scenario", ["warm", "cold"])
def test_four_rank_warm_and_cold_actual_miss_counters(tmp_path, scenario):
    results = _launch(tmp_path, queries=1, e4m3=True, ratio=0.1, scenario=scenario)
    for result in results:
        misses = [r["misses"] for r in result["records"]]
        if scenario == "warm":
            assert misses == [0, 0]
        else:
            assert all(r["misses"] == r["wanted"] > 0 for r in result["records"])


@pytest.mark.parametrize(
    "phase",
    [
        "setup",
        "before-query",
        "metadata",
        "attention",
        "pre-combine",
        "validate",
        "counters",
    ],
)
def test_one_rank_failure_is_identical_on_all_four_ranks(tmp_path, phase):
    results = _launch(
        tmp_path,
        queries=1,
        e4m3=True,
        ratio=0.1,
        scenario="cold",
        fail_rank=3,
        fail_phase=phase,
    )
    failures = [r["failure"] for r in results]
    assert len(set(failures)) == 1
    assert f"injected failure at {phase}" in failures[0]


def test_local_lengths_match_independent_interleaved_owner_counts():
    lengths = torch.arange(40, dtype=torch.int32)
    for rank in range(2):
        expected = torch.tensor(
            [sum(t % 2 == rank for t in range(n)) for n in lengths.tolist()]
        )
        assert torch.equal(local_length(lengths, 2, rank), expected)


def test_bit_gate_includes_signed_zero_and_nonfinite_values():
    expected = torch.tensor([0.0, -torch.inf, 1.0])
    actual = torch.tensor([-0.0, -torch.inf, 1.0])
    metrics = error_metrics(expected, actual)
    assert not metrics["bitwise"] and metrics["mismatches"] == 1
    assert metrics["max_abs"] == 0
    assert metrics["first"] == [0]
    assert metrics["nonfinite_mismatches"] == 0
    clean = torch.tensor([0.0, 1.0, 1.0])
    signed = torch.tensor([-0.0, 1.0, 1.0])
    with pytest.raises(AssertionError, match="partial differs"):
        check_results(
            (clean, expected, clean, expected), (signed, expected, signed, expected)
        )
    with pytest.raises(AssertionError):
        check_results(
            (expected,) * 4, (torch.full_like(actual, torch.nan),) * 4, allow_drift=True
        )


@pytest.mark.parametrize("ratio", [0.1, 0.2])
@pytest.mark.parametrize("queries", [1, 8])
def test_gpu_profile_admission_and_local_width(ratio, queries):
    profile = Profile(
        queries=queries,
        batch_queries=1 if queries == 1 else 2,
        union_groups=512 if queries == 1 else 1536,
        hot_ratio=ratio,
    )
    config = profile.config(2)
    assert config.selection_width == 1026
    slots, misses = config.capacities()
    assert slots >= misses == config.required_groups
    assert config.max_tokens == 262144


def test_bandwidth_uses_world_wall_envelope_not_sum_of_rank_rates():
    results = []
    for rank in range(4):
        record = dict(
            transfer_spans=[(rank * 1000000, (rank + 1) * 1000000)],
            transfer_bytes=2**20,
            phases={"gather_ms": 1.0},
            misses=512,
        )
        results.append({"records": [record]})
    result = aggregate_bandwidth(results)[0]
    assert result["aggregate_wall_gib_s"] == 0.9765625
    assert result["aggregate_max_event_gib_s"] == 3.90625


@pytest.mark.parametrize("grouped_rows", [0, 8, 16])
def test_gpu_dispatch_preserves_full_grouped_width_and_narrow_triton_remainder(
    monkeypatch, grouped_rows
):
    from types import SimpleNamespace

    from tools.host_kv_dcp2 import LayerExperiment
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    # Dispatch-only fixture. Native execution remains a separate GPU gate.
    case = object.__new__(LayerExperiment)
    case.device = torch.device("cpu")
    case.cpu_plan = None
    case.config = SimpleNamespace(rows=16, selection_width=1026)
    case.p = Profile()
    case.scales = SimpleNamespace(key=0.375, value=1.75)
    case.gathered = torch.empty((16, 12, 256), dtype=torch.float16)
    case.partial = torch.empty_like(case.gathered, dtype=torch.float32)
    case.lse = torch.empty((16, 12), dtype=torch.float32)
    case.requests = torch.zeros(16, dtype=torch.int32)
    case.local = torch.empty((16, 2051), dtype=torch.int32)
    case.hot_full = torch.empty_like(case.local)
    case.source = torch.empty((1, 2, 832, 1, 256), dtype=torch.uint8)
    case.b_hot = torch.empty((1, 2, 4, 1, 256), dtype=torch.uint8)
    case.pool = SimpleNamespace(
        hot=case.b_hot,
        block_table=torch.zeros((8, 1), dtype=torch.int32),
        hot_block_table=torch.zeros((8, 1), dtype=torch.int32),
    )
    case.route = "auto"
    seen = []

    def grouped(*args):
        assert args[3].shape == (16, 2051)
        assert args[6].dtype == args[7].dtype == torch.float32
        return grouped_rows

    def sparse(query, key, value, indices, table, requests, **kwargs):
        seen.append(query.shape[0])
        assert indices.shape == (16 - grouped_rows, 1026)
        assert kwargs["out"].dtype == torch.float32
        assert kwargs["lse"].dtype == torch.float32
        assert kwargs["v_scale"] == 1.75

    monkeypatch.setattr(
        qsa_ops, "qsa_dcp_sparse_paged_attention_sm70_grouped_page4", grouped
    )
    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", sparse)
    for arm in ("A", "B", "C"):
        case.attention(arm)
        assert case.last_grouped == grouped_rows
    assert seen == ([16 - grouped_rows] * 3 if grouped_rows < 16 else [])
    if grouped_rows == 0:
        case.route = "grouped"
        with pytest.raises(RuntimeError, match="unavailable"):
            case.attention("C")


def test_final_gate_rounds_merged_fp32_to_fp16_before_sigmoid():
    from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_output_gate

    enable_cpu_interpreter()
    merged = torch.linspace(-4.003, 3.991, 32).reshape(1, 2, 16)
    gate = torch.linspace(-1, 1, 32).reshape_as(merged).half()
    output = merged.half()
    _qsa_output_gate(output, gate)
    expected = (merged.half().float() * gate.float().sigmoid()).half()
    assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("cpu", [False, True])
@pytest.mark.parametrize("queries", [1, 8])
def test_cli_small_profiles_are_admitted_for_probe_and_cpu(monkeypatch, cpu, queries):
    arguments = ["benchmark", "--queries", str(queries)]
    arguments += ["--backend", "gloo"] if cpu else ["--probe"]
    monkeypatch.setattr("sys.argv", arguments)
    profile = make_profile(parse_args())
    for world in (1, 2):
        config = profile.config(world)
        assert config.capacities()[0] >= config.required_groups


def test_divergent_rank_selection_is_rejected_before_device_collectives(tmp_path):
    results = _launch(
        tmp_path,
        queries=1,
        e4m3=True,
        ratio=0.1,
        scenario="cold",
        fail_rank=3,
        fail_phase="trace",
    )
    failures = [result["failure"] for result in results]
    assert len(set(failures)) == 1
    assert "trace-contract" in failures[0]
