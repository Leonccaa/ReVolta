# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the long-context KV A/B tool's planning and comparison."""

import copy
import json
import math
from typing import Any

import pytest

from tools import long_context_kv_ab as ab


def _request(name="r", tokens=(5, 6, 7), lp="-0x1.0p-1", cached=0):
    return {
        "name": name,
        "request_id": "0",
        "prompt_len": 10,
        "expected_prompt_len": 10,
        "num_cached_tokens": cached,
        "token_ids": list(tokens),
        "max_tokens": len(tokens),
        "finish_reason": "length",
        "logprobs": [[[t, lp, 1], [t + 1, "-0x1.0p+1", 2]] for t in tokens],
        "prompt_logprobs": None,
    }


def _run(label, requests, parity="hard", checks=None, status="ok"):
    return {
        "label": label,
        "status": status,
        "checks": checks or [],
        "scenarios": [
            {
                "name": "s",
                "parity": parity,
                "phases": [{"name": "p", "requests": requests}],
            }
        ],
    }


def test_prompts_are_deterministic_and_shared_prefixes_match():
    assert ab.make_tokens("k", 50, 0) == ab.make_tokens("k", 50, 0)
    assert ab.make_tokens("k", 50, 0) != ab.make_tokens("k", 50, 1)
    phases = ab.SCENARIOS["prefix"].phases
    a = ab.prompt_tokens(phases[0].requests[0], 0)
    b = ab.prompt_tokens(phases[1].requests[0], 0)
    assert a[:3500] == b[:3500] and a[3500:] != b[3500:3620]
    assert all(ab.TOKEN_LOW <= t < ab.TOKEN_HIGH for t in a)


@pytest.mark.parametrize("name", sorted(ab.CONFIGS))
def test_every_config_fits_max_model_len(name):
    ab.validate_config(ab.CONFIGS[name])


def test_probe_prefill_spans_two_chunks():
    config = ab.CONFIGS["probe"]
    longest = max(r.prompt_len for r in ab.SCENARIOS["probe"].phases[0].requests)
    assert (
        config.max_num_batched_tokens < longest <= 2 * (config.max_num_batched_tokens)
    )
    chunked = ab.SCENARIOS["chunked"].phases[0].requests[0]
    assert chunked.prompt_len > 2 * ab.CONFIGS["main"].max_num_batched_tokens


# The 1.9B stand-in's recorded geometry (first host-placement probe).
GEOMETRY = {
    "block_tokens": 1568,
    "num_workers": 1,
    "main_kv_bytes_per_block": 3_211_264,
    "index_bytes_per_block": 100_352,
    "state_bytes_per_block": 3_211_264,
    "unified_bytes_per_block": 3_311_616,
    "state_blocks_per_request": 8,
    "ring_blocks_per_request": 1,
    "max_model_len": 8192,
    "hot_group_tokens": 4,
    "hot_row_groups": 513,
    "decode_rows_per_request": 1,
    "group_domains": [[1, 2], [1], [0], [0], [0], [0]],
    "dcp_world_size": 1,
    "hot_layers": 1,
    "hot_max_write_rows": 2048,
    "hot_query_heads": 24,
    "hot_head_dim": 256,
    "hot_kv_heads": 2,
}


def test_hot_pool_arithmetic_matches_the_probe():
    # 1 GiB, two requests: the probe log said 131072 groups, 255 rows/pass,
    # and 4,708 GPU metadata bytes per history block (8,719,980 - 8,329,216
    # over 83 blocks).
    assert ab.hot_groups_for_bytes(GEOMETRY, ab.GIB) < 131072
    assert ab.rows_per_pass(GEOMETRY, 131072, 2) == 255
    assert ab.host_metadata_bytes_per_block(GEOMETRY) * 83 == 8_719_980 - 8_329_216
    for rows, seqs in ((1024, 2), (2048, 4), (255, 4)):
        gib = ab.hot_pool_gib_for_rows(GEOMETRY, rows, seqs)
        groups = ab.hot_groups_for_bytes(GEOMETRY, int(gib * ab.GIB), seqs)
        assert ab.rows_per_pass(GEOMETRY, groups, seqs) >= rows
        less = ab.hot_groups_for_bytes(GEOMETRY, int((gib - 1 / 1024) * ab.GIB), seqs)
        assert ab.rows_per_pass(GEOMETRY, less, seqs) < rows
    assert ab.hot_pool_gib_for_rows(GEOMETRY, 2048, 4) > 8.02


def _geometry_object(geometry):
    from vllm.v1.core.long_context_kv import LongContextGeometry

    fields = dict(geometry)
    fields.setdefault("hot_read_rows", None)
    fields["group_domains"] = tuple(tuple(d) for d in fields["group_domains"])
    return LongContextGeometry(**fields)


def test_tool_arithmetic_matches_the_engine_plan():
    geo = _geometry_object(GEOMETRY)
    assert ab.host_metadata_bytes_per_block(GEOMETRY) == (
        geo.host_metadata_bytes_per_block - 4  # Legacy receipt predates page_of.
    )
    current = {**GEOMETRY, "logical_index_bytes_per_block": 100_352}
    assert (
        ab.host_metadata_bytes_per_block(current) == geo.host_metadata_bytes_per_block
    )
    for num_bytes in (ab.GIB, 3 * ab.GIB + 12345):
        groups = ab.hot_groups_for_bytes(GEOMETRY, num_bytes)
        assert groups == geo.hot_groups_for_bytes(num_bytes)
        assert ab.hot_bytes_for_groups(GEOMETRY, groups) == (
            geo.hot_bytes_for_groups(groups)
        )


@pytest.mark.parametrize("legacy", [False, True])
def test_sizing_reads_serialized_plan_geometry_with_derived_fields(legacy):
    from vllm.config.kernel import LongContextKVConfig
    from vllm.v1.core.long_context_kv import plan_long_context_capacity

    geometry = _geometry_object(GEOMETRY)
    plan = plan_long_context_capacity(
        LongContextKVConfig(enabled=True, host_pool_tokens=32 * 1568, hot_pool_gib=1),
        geometry,
        3 * ab.GIB,
        4,
    )
    receipt = ab._plan_dict(plan)["geometry"]
    assert "host_metadata_bytes_per_block" in receipt
    assert "host_bytes_per_block" in receipt
    if legacy:
        receipt["main_kv_layers"] = receipt.pop("hot_layers")
    assert ab.hot_groups_for_bytes(receipt, ab.GIB, 4) == plan.hot_pool_groups
    assert (
        ab.hot_bytes_for_groups(receipt, plan.hot_pool_groups, 4) == plan.hot_pool_bytes
    )


@pytest.mark.parametrize("hot_gib", [1.0, 0.25])
def test_tight_settings_give_both_placements_the_same_pools(hot_gib):
    from vllm.config.kernel import LongContextKVConfig
    from vllm.v1.core.long_context_kv import plan_long_context_capacity

    geo = _geometry_object(GEOMETRY)
    history = math.ceil(8192 / 1568)
    plans = {}
    for placement in ("gpu", "host"):
        derived = ab.tight_settings(GEOMETRY, 8192, placement, hot_gib)
        assert derived["state_blocks"] == 12
        assert derived["target_history_blocks"] == history + 4
        assert derived["history_blocks"] >= derived["target_history_blocks"]
        settings = LongContextKVConfig(
            enabled=True,
            host_pool_tokens=derived["host_pool_tokens"],
            hot_pool_gib=hot_gib,
            state_blocks=derived["state_blocks"],
            main_kv_placement=placement,
        )
        budget = (
            derived["num_gpu_blocks_override"] * GEOMETRY["unified_bytes_per_block"]
        )
        plan = plan_long_context_capacity(settings, geo, budget, 4)
        assert not plan.startup_problems(), plan.startup_problems()
        assert plan.main_kv_on_gpu == (placement == "gpu")
        assert plan.state_blocks == derived["state_blocks"]
        assert plan.gpu_history_blocks == derived["history_blocks"]
        plans[placement] = plan
    # Same state and history IDs: the two placements schedule alike.
    assert plans["gpu"].segment_num_blocks() == plans["host"].segment_num_blocks()
    # The host arm's history IDs are capped by its host pages, not its budget.
    host = plans["host"]
    assert host.gpu_history_budget_blocks >= host.gpu_history_blocks
    assert host.host_page_blocks == host.gpu_history_blocks
    off = ab.tight_settings(GEOMETRY, 8192, "off")
    assert off["num_gpu_blocks_override"] == 8 + history + 1 + 4


def test_compare_identical_and_bitwise_logprob_difference():
    a = _run("a", [_request()])
    assert ab.compare_runs(a, copy.deepcopy(a))["hard_identical"]
    b = _run("b", [_request(lp="-0x1.0000000000001p-1")])
    result = ab.compare_runs(a, b)
    assert not result["hard_identical"]
    first = result["scenarios"][0]["first_difference"]
    assert first["field"] == "logprobs" and first["position"] == 0


def test_compare_reports_first_token_and_cache_difference():
    a = _run("a", [_request(tokens=(1, 2, 3, 4))])
    b = _run("b", [_request(tokens=(1, 2, 9, 4))])
    first = ab.compare_runs(a, b)["scenarios"][0]["first_difference"]
    assert first["field"] == "token_ids" and first["position"] == 2
    c = _run("c", [_request(tokens=(1, 2, 3, 4), cached=1616)])
    first = ab.compare_runs(a, c)["scenarios"][0]["first_difference"]
    assert first["field"] == "num_cached_tokens"


def test_verdicts():
    a = _run("off", [_request()])
    same = copy.deepcopy(a)
    assert ab.verdict(a, same, copy.deepcopy(a))["verdict"] == ab.PASS
    diff = _run("on", [_request(tokens=(5, 6, 8))])
    assert ab.verdict(a, diff)["verdict"] == ab.FAIL
    # The reference arm does not reproduce itself: inconclusive.
    assert ab.verdict(a, same, diff)["verdict"] == ab.INCONCLUSIVE
    # Soft scenarios are reported, not judged; a run with no hard scenario
    # cannot pass.
    soft_a = _run("off", [_request()], parity="soft")
    soft_b = _run("on", [_request(tokens=(5, 6, 8))], parity="soft")
    result = ab.verdict(soft_a, soft_b)
    assert result["a_vs_b"]["soft_differences"] == ["s"]
    assert result["verdict"] == ab.FAIL
    failed = _run("on", [_request()], checks=[{"name": "x", "ok": False}])
    assert ab.verdict(a, failed)["verdict"] == ab.FAIL
    broken = _run("on", [], status="init_failed")
    assert ab.verdict(a, broken)["verdict"] == ab.FAIL
    skipped = _run("on", [], status="skipped")
    assert ab.verdict(a, skipped)["verdict"] == ab.INCONCLUSIVE
    assert ab.verdict(broken, skipped)["verdict"] == ab.FAIL


def test_output_problems():
    assert ab.output_problems(_request()) == []
    short = _request()
    short["max_tokens"] = 5
    assert ab.output_problems(short)
    nan = _request(lp=float("nan").hex())
    assert any("non-finite" in p for p in ab.output_problems(nan))


def _structure(placement="gpu"):
    """A recorded structure for one arm: off, gpu or host placement."""
    on = placement != "off"
    state_n, history_n = 13, 11
    index, main = 100, 400
    geometry = dict(
        GEOMETRY,
        block_tokens=16,
        index_bytes_per_block=index,
        main_kv_bytes_per_block=main,
        hot_group_tokens=4,
        hot_row_groups=3,
    )
    meta = ab.host_metadata_bytes_per_block(geometry)
    history_block = main + index if placement == "gpu" else index + meta
    plan = {
        "geometry": geometry,
        "max_num_seqs": 2,
        "segment_num_blocks": [state_n, history_n],
        "gpu_state_bytes": state_n * 300,
        "gpu_history_bytes": history_n * history_block,
        "gpu_budget_bytes": 1_000_000,
        "main_kv_on_gpu": placement == "gpu",
        "host_history_blocks": history_n + 2,
        "host_pool_bytes": (history_n + 2) * main,
        "hot_pool_groups": 40,
        "hot_pool_bytes": 40 * main * 4 // 16,
        "gpu_hot_pool_bytes": 0 if placement == "gpu" else 40 * main * 4 // 16,
        "staging_bytes": 0,
    }
    history_size = history_n * (history_block if placement == "gpu" else index)
    tensors = [
        {"size": state_n * 100, "num_blocks": state_n, "segment": "state"},
        {"size": state_n * 200, "num_blocks": state_n, "segment": "state"},
        {"size": history_size, "num_blocks": history_n, "segment": "history"},
    ]
    if not on:
        for t in tensors:
            t["segment"] = "unified"
    for i, t in enumerate(tensors):
        t["layers"] = [f"layer{i}"]
        t["shared_by"] = t["layers"]
        t["members"] = 1
    storages = [
        {
            "ptr": 1000 * (i + 1),
            "nbytes": t["size"],
            "layers": t["layers"],
            "segments": [t["segment"]],
        }
        for i, t in enumerate(tensors)
    ]
    structure: dict[str, Any] = {
        "num_blocks": history_n,
        "plan": plan if on else None,
        "block_segments": {
            "names": ["state", "history"],
            "num_blocks": [state_n, history_n],
        }
        if on
        else None,
        "kv_cache_tensors": tensors,
        "worker_storages": storages,
        "zeroer": {"segments": 2, "segment_spaces": ["history", "history"]},
        "host_kv_layers": [],
        "host_store": None,
    }
    if placement == "host":
        structure["host_kv_layers"] = ["attn"]
        storages.insert(
            0, {"ptr": 0, "nbytes": 0, "layers": ["attn"], "segments": ["history"]}
        )
        structure["host_store"] = {
            "num_blocks": history_n,
            "page_bytes": main,
            "host_bytes": history_n * main,
            "metadata_bytes": history_n * meta,
            "hot_slots": 40,
            "hot_bytes": plan["gpu_hot_pool_bytes"],
            "rows_per_pass": ab.rows_per_pass(geometry, 40, 2),
            "layers": ["attn"],
            "bound_layers": ["attn"],
        }
    return structure


def _failed(checks):
    return {c["name"] for c in checks if c["ok"] is False}


@pytest.mark.parametrize(
    "arm,placement",
    [("off", "off"), ("on", "gpu"), ("on-gpu", "gpu"), ("on-host", "host")],
)
def test_segment_and_zeroing_checks_pass(arm, placement):
    structure = _structure(placement)
    assert ab.structure_placement(structure, arm) == placement
    checks = ab.segment_tensor_checks(structure, arm)
    assert not _failed(checks), checks
    assert not _failed(ab.zeroing_checks(structure, arm))


def test_gpu_placement_check_failures():
    bad = _structure("gpu")
    bad["kv_cache_tensors"][0]["size"] += 13
    bad["worker_storages"][0]["nbytes"] += 13
    assert _failed(ab.segment_tensor_checks(bad, "on-gpu")) == {
        "state segment bytes equal the plan"
    }
    bad["zeroer"]["segment_spaces"] = ["history", "state"]
    assert _failed(ab.zeroing_checks(bad, "on-gpu"))
    # The arm asked for the host placement but the plan kept main K/V on GPU.
    assert "main K/V placement is the requested one" in _failed(
        ab.segment_tensor_checks(_structure("gpu"), "on-host")
    )


def test_host_placement_explains_every_storage():
    structure = _structure("host")
    explained = ab.explain_storages(structure)
    purposes = [e["purpose"] for e in explained["storages"]]
    assert purposes[0].startswith("host-placed main K/V")
    assert all(p.startswith("configured KV tensor") for p in purposes[1:])
    assert not explained["unexplained"] and not explained["tensors_without_storage"]
    # The GPU-placement accounting (tensors == plan history bytes) would fail
    # here: the history tensors hold the index pages only.
    history = sum(
        t["size"] for t in structure["kv_cache_tensors"] if t["segment"] == "history"
    )
    assert history < structure["plan"]["gpu_history_bytes"]
    # An empty storage of a layer that is not host-placed is unexplained.
    stray = _structure("host")
    stray["host_kv_layers"] = []
    stray["host_store"]["layers"] = stray["host_store"]["bound_layers"] = []
    failed = _failed(ab.segment_tensor_checks(stray, "on-host"))
    assert "every worker KV storage is explained and backs one KV tensor" in failed
    # A configured tensor without a worker storage.
    missing = _structure("host")
    del missing["worker_storages"][1]
    failed = _failed(ab.segment_tensor_checks(missing, "on-host"))
    assert "every worker KV storage is explained and backs one KV tensor" in failed


@pytest.mark.parametrize(
    "field,delta,name",
    [
        ("metadata_bytes", 4, "host store metadata bytes match the plan"),
        ("num_blocks", 1, "host store pages match the plan"),
        ("page_bytes", 8, "host store pages match the plan"),
        ("hot_slots", -1, "hot pool matches the plan"),
        ("rows_per_pass", 1, "hot pool matches the plan"),
    ],
)
def test_host_store_check_failures(field, delta, name):
    structure = _structure("host")
    structure["host_store"][field] += delta
    assert name in _failed(ab.segment_tensor_checks(structure, "on-host"))


def test_value_shadow_bytes_count_toward_the_history_segment():
    # value arbitration bills one shadow byte per history group in the history segment.
    structure = _structure("host")
    structure["plan"]["gpu_history_bytes"] += 7 * 4
    name = "history segment bytes equal the plan (tensors + store metadata)"
    assert name in _failed(ab.segment_tensor_checks(structure, "on-host"))
    structure["host_store"]["shadow_gpu_bytes"] = 7 * 4
    assert not _failed(ab.segment_tensor_checks(structure, "on-host"))


def test_host_history_segment_and_binding_failures():
    structure = _structure("host")
    structure["kv_cache_tensors"][2]["size"] += 11
    structure["worker_storages"][3]["nbytes"] += 11
    failed = _failed(ab.segment_tensor_checks(structure, "on-host"))
    assert failed == {
        "history segment tensors hold the index and ring pages only",
        "history segment bytes equal the plan (tensors + store metadata)",
    }
    unbound = _structure("host")
    unbound["host_store"]["bound_layers"] = []
    assert _failed(ab.segment_tensor_checks(unbound, "on-host")) == {
        "host store is bound to every host-placed layer"
    }
    no_store = _structure("host")
    no_store["host_store"] = None
    assert "host history store present" in _failed(
        ab.segment_tensor_checks(no_store, "on-host")
    )
    # Main K/V on GPU must not carry a host store.
    gpu = _structure("gpu")
    gpu["host_kv_layers"] = ["attn"]
    assert "no host history store with the main K/V on GPU" in _failed(
        ab.segment_tensor_checks(gpu, "on-gpu")
    )


def _phase(allocated, free=(10, 10), preemptions=()):
    return {
        "name": "p",
        "expect_cached": False,
        "requests": [_request()],
        "allocated": allocated,
        "pools_after": [{"usable": 10, "free": f} for f in free],
        "preemptions": list(preemptions),
    }


def test_reuse_preemption_and_pool_checks():
    phases = [_phase([[1, 2], [1]]), _phase([[3], [2]]), _phase([[1], [1, 3]])]
    assert ab.reuse_overlap(phases) == [1, 1]
    scenario = {
        "name": "reuse",
        "phases": phases,
        "expect_preemption": False,
        "expect_reuse": True,
    }
    assert all(c["ok"] for c in ab.scenario_checks([scenario], "on"))
    phases[2]["allocated"] = [[4], [1, 3]]
    assert not all(c["ok"] for c in ab.scenario_checks([scenario], "on"))
    pre = {
        "name": "preempt",
        "phases": [_phase([[1], [1]])],
        "expect_preemption": True,
        "expect_reuse": False,
    }
    result = {c["name"]: c["ok"] for c in ab.scenario_checks([pre], "on")}
    assert result["preempt preempts"] is False
    result = {c["name"]: c["ok"] for c in ab.scenario_checks([pre], "off")}
    assert result["preempt preempts"] is None
    leak = {"name": "x", "phases": [_phase([[1], [1]], free=(9, 10))]}
    assert ab.pool_checks([leak])[0]["ok"] is False


def test_log_patterns_match_engine_lines():
    lines = [
        "INFO [kv_cache_manager.py:223] long-context KV plan (main K/V on GPU "
        "in the history segment, hot pool budgets admission only): GPU budget",
        "INFO [gpu_worker.py:707] Available KV cache memory: 3.3 GiB",
        "INFO [model_runner.py:1037] Graph capturing finished in 33 secs, took "
        "0.13 GiB",
        "INFO [interface.py:727] Setting attention block size to 1568 tokens to",
        "INFO [engine.py:110] Long-context main K/V in host memory: 83 history "
        "blocks x 1 layers (0.25 GiB pinned on this worker), hot pool 131072 "
        "groups per layer, 255 query rows per gather pass.",
    ]
    found = ab.parse_log_lines(lines)
    assert all(
        found[k]
        for k in ("plan", "available_kv", "graph_capture", "block_size", "host_store")
    )


def test_compare_and_summary_commands(tmp_path):
    a = _run("off", [_request()])
    b = copy.deepcopy(a)
    b["label"] = "on"
    for name, run in (("a", a), ("b", b)):
        (tmp_path / f"{name}.json").write_text(json.dumps(run))
    out = tmp_path / "v.json"
    code = ab.main(
        [
            "compare",
            str(tmp_path / "a.json"),
            str(tmp_path / "b.json"),
            "--label",
            "main",
            "--out",
            str(out),
        ]
    )
    assert code == 0 and json.loads(out.read_text())["verdict"] == ab.PASS
    summary = tmp_path / "s.json"
    assert ab.main(["summary", str(out), "--out", str(summary)]) == 0
    assert json.loads(summary.read_text())["verdict"] == ab.PASS


def test_difference_stats_attribute_a_logprob_difference():
    a = _run("gpu", [_request(name="x"), _request(name="y")])
    b = copy.deepcopy(a)
    # One value of one position differs; tokens, top-k sets and ranks agree.
    b["scenarios"][0]["phases"][0]["requests"][1]["logprobs"][0][1][1] = (
        "-0x1.0400000000000p+1"
    )
    stats = ab.compare_runs(a, b)["difference_stats"]
    assert stats["requests"] == 2 and stats["requests_different"] == 1
    assert stats["token_ids_identical"] and stats["topk_tokens_and_ranks_identical"]
    assert stats["values_different"] == 1 and stats["values_compared"] == 12
    assert stats["max_abs_logprob_diff"] == pytest.approx(2 * (2**-6))
    assert stats["positions"] == [
        {"scenario": "s", "request": "y", "field": "logprobs", "position": 0}
    ]
    # A rank change is a top-k difference.
    b["scenarios"][0]["phases"][0]["requests"][1]["logprobs"][1][1][2] = 3
    assert not ab.compare_runs(a, b)["difference_stats"][
        "topk_tokens_and_ranks_identical"
    ]
    c = _run("host", [_request(name="x", tokens=(5, 6, 9)), _request(name="y")])
    assert not ab.difference_stats(a, c)["token_ids_identical"]


def test_verdict_reports_the_new_arm_against_the_repeat():
    a = _run("gpu", [_request()])
    b = _run("host", [_request(tokens=(5, 6, 8))])
    result = ab.verdict(a, b, copy.deepcopy(a))
    assert result["verdict"] == ab.FAIL
    assert not result["b_vs_control"]["hard_identical"]


class _FakeTensor:
    """Row-sliceable stand-in for a tensor (shape and slicing only)."""

    def __init__(self, rows, start=0):
        self.rows, self.start = rows, start
        self.shape = (rows, 4)

    def __getitem__(self, index):
        start, stop, _ = index.indices(self.rows)
        return _FakeTensor(stop - start, self.start + start)

    def new_empty(self, shape):
        return _FakeTensor(shape[0])


def _fake_attention(calls):
    def qsa_sparse_paged_attention(
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        out=None,
        output_gate=None,
        query_positions=None,
        sequence_lengths=None,
        kv_cache_dtype="auto",
        k_scale=1.0,
        v_scale=1.0,
        lse=None,
    ):
        calls.append(
            {
                "rows": q.rows,
                "start": q.start,
                "indices": (logical_indices.start, logical_indices.rows),
                "out": (out.start, out.rows),
                "gate": None if output_gate is None else output_gate.start,
                "positions": query_positions is not None,
                "lengths": sequence_lengths is not None,
            }
        )
        return out

    return qsa_sparse_paged_attention


def test_host_like_route_drops_positions_and_splits_rows():
    calls: list[dict[str, Any]] = []
    probe = ab.QSARouteProbe(host_like=True, pass_rows=3)
    attention = probe.attention(_fake_attention(calls))
    t = _FakeTensor
    out = t(7)
    result = attention(
        t(7), "k", "v", t(7), "table", t(7), out,
        output_gate=t(7), query_positions=t(7), sequence_lengths="lens",
    )  # fmt: skip
    assert result is out
    assert [(c["start"], c["rows"]) for c in calls] == [(0, 3), (3, 3), (6, 1)]
    assert [c["indices"] for c in calls] == [(0, 3), (3, 3), (6, 1)]
    assert [c["out"] for c in calls] == [(0, 3), (3, 3), (6, 1)]
    assert [c["gate"] for c in calls] == [0, 3, 6]
    assert not any(c["positions"] or c["lengths"] for c in calls)
    assert probe.stats["split_calls"] == 1 and probe.stats["row_passes"] == 3
    # A call that fits one pass keeps its rows; LSE (DCP) calls pass through.
    calls.clear()
    attention(t(2), "k", "v", t(2), "table", t(2), t(2), query_positions=t(2))
    attention(t(9), "k", "v", t(9), "table", t(9), t(9), lse="lse",
              query_positions=t(9))  # fmt: skip
    assert [(c["rows"], c["positions"]) for c in calls] == [(2, False), (9, True)]
    assert probe.stats["passthrough_calls"] == 1
    assert probe.stats["attention_max_rows"] == 9
    # The default route only counts.
    calls.clear()
    plain = ab.QSARouteProbe()
    plain.attention(_fake_attention(calls))(
        t(7), "k", "v", t(7), "table", t(7), t(7), query_positions=t(7)
    )
    assert [(c["rows"], c["positions"]) for c in calls] == [(7, True)]
    assert plain.stats["attention_calls"] == 1 and plain.stats["host_like_calls"] == 0


def test_host_pass_counter():
    seen = []
    probe = ab.QSARouteProbe()
    passes = probe.passes(lambda layer, selection, *rest: seen.append(selection.rows))
    layer = type("Layer", (), {"rows_per_pass": 255})()
    passes(layer, _FakeTensor(1024), "req", "table", "lens", "attend")
    passes(layer, _FakeTensor(2), "req", "table", "lens", "attend")
    assert seen == [1024, 2]
    stats = probe.stats
    assert stats["host_steps"] == 2 and stats["host_multi_pass_steps"] == 1
    assert stats["host_max_passes"] == 5 and stats["host_rows_per_pass"] == 255


def test_route_hooks_reach_both_qsa_paths(monkeypatch):
    """The callers import the hooked names inside the function, at call
    time; a refactor to a module-level import would bypass the hooks."""
    import inspect

    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.models.qwen4_exp.nvidia.host_kv import history
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    impl = qsa.Qwen4ExpQSAFlashAttentionImpl
    assert "from .ops.qsa import qsa_sparse_paged_attention" in inspect.getsource(
        impl.forward_qsa
    )
    host_source = inspect.getsource(impl.forward_qsa_host)
    assert "from .host_kv.history import for_each_pass" in host_source
    assert "from .ops.qsa import qsa_sparse_paged_attention" in host_source
    # The remap passes pass neither positions nor sequence lengths; the
    # union path (selection-union gather) makes the GPU arm's call, positions and
    # lengths
    # included, behind its own hook.
    attend = host_source[host_source.index("def attend(") :]
    attend = attend[: attend.index("if union:")]
    assert "query_positions" not in attend and "sequence_lengths" not in attend
    assert "from .host_kv.history import for_each_pass, run_union_windows" in (
        host_source
    )
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(qsa_ops, "qsa_sparse_paged_attention", _fake_attention(calls))
    monkeypatch.setattr(history, "for_each_pass", lambda *a, **k: None)
    probe = ab.QSARouteProbe(host_like=True, pass_rows=4)
    probe.install()
    t = _FakeTensor
    qsa_ops.qsa_sparse_paged_attention(t(5), "k", "v", t(5), "b", t(5), t(5))
    history.for_each_pass(type("L", (), {"rows_per_pass": 4})(), t(9))
    assert [c["rows"] for c in calls] == [4, 1]
    assert probe.stats["host_steps"] == 1 and probe.stats["host_max_passes"] == 3


def _args(*extra):
    argv = ["run", "--model", "m", "--out", "o.json", *extra]
    return ab.build_parser().parse_args(argv)


def test_engine_args_by_arm(tmp_path):
    kwargs, derived = ab.resolve_engine_args(
        _args("--arm", "off", "--config", "main"), ab.CONFIGS["main"]
    )
    assert "long_context_kv" not in kwargs["kernel_config"]
    assert derived["placement"] == "off"
    for arm, placement in (("on-gpu", "gpu"), ("on-host", "host")):
        kwargs, derived = ab.resolve_engine_args(
            _args("--arm", arm, "--config", "main"), ab.CONFIGS["main"]
        )
        lc = kwargs["kernel_config"]["long_context_kv"]
        assert lc["main_kv_placement"] == placement == derived["placement"]
        assert lc["host_pool_tokens"] == ab.CONFIGS["main"].host_pool_tokens
        assert lc["hot_pool_gib"] == 1.0
    # "on" leaves the placement to the engine default.
    kwargs, derived = ab.resolve_engine_args(
        _args("--arm", "on", "--config", "probe", "--host-pool-gib", "2"),
        ab.CONFIGS["probe"],
    )
    lc = kwargs["kernel_config"]["long_context_kv"]
    assert "main_kv_placement" not in lc and lc["host_pool_gib"] == 2.0
    assert "host_pool_tokens" not in lc
    from vllm.config.kernel import LongContextKVConfig

    assert derived["placement"] == LongContextKVConfig().main_kv_placement


def test_engine_args_size_pools_from_a_geometry(tmp_path):
    source = tmp_path / "gpu.json"
    source.write_text(json.dumps({"structure": {"plan": {"geometry": GEOMETRY}}}))
    kwargs, derived = ab.resolve_engine_args(
        _args(
            "--arm", "on-host", "--config", "main", "--single-pass",
            "--geometry-from", str(source),
        ),
        ab.CONFIGS["main"],
    )  # fmt: skip
    lc = kwargs["kernel_config"]["long_context_kv"]
    assert derived["hot_pool_rows"] == 2048
    assert lc["hot_pool_gib"] == ab.hot_pool_gib_for_rows(GEOMETRY, 2048, 4)
    kwargs, derived = ab.resolve_engine_args(
        _args(
            "--arm", "on-host", "--config", "tight",
            "--geometry-from", str(source),
        ),
        ab.CONFIGS["tight"],
    )  # fmt: skip
    lc = kwargs["kernel_config"]["long_context_kv"]
    expected = ab.tight_settings(GEOMETRY, 8192, "host", 1.0)
    assert lc["host_pool_tokens"] == expected["host_pool_tokens"]
    assert lc["state_blocks"] == expected["state_blocks"]
    assert kwargs["num_gpu_blocks_override"] == expected["num_gpu_blocks_override"]
    kwargs, _ = ab.resolve_engine_args(
        _args("--arm", "off", "--config", "tight", "--geometry-from", str(source)),
        ab.CONFIGS["tight"],
    )
    off = ab.tight_settings(GEOMETRY, 8192, "off")
    assert kwargs["num_gpu_blocks_override"] == off["num_gpu_blocks_override"]


@pytest.mark.parametrize("arm", ["on", "on-host", "on-gpu"])
def test_engine_args_accept_auto_hot_pool(arm):
    kwargs, _ = ab.resolve_engine_args(
        _args("--arm", arm, "--config", "main", "--hot-pool-gib", "auto"),
        ab.CONFIGS["main"],
    )
    settings = kwargs["kernel_config"]["long_context_kv"]
    assert settings["hot_pool_gib"] == "auto"
    assert settings["host_pool_tokens"] == ab.CONFIGS["main"].host_pool_tokens


@pytest.mark.parametrize("placement", ["host", "gpu"])
def test_tight_auto_budget_fits_full_verify_working_sets(placement):
    from vllm.config.kernel import LongContextKVConfig
    from vllm.v1.core.long_context_kv import (
        plan_long_context_capacity,
        validate_long_context_plan,
    )

    derived = ab.tight_settings(GEOMETRY, 8192, placement, "auto")
    geo = _geometry_object(GEOMETRY)
    plan = plan_long_context_capacity(
        LongContextKVConfig(
            enabled=True,
            hot_pool_gib="auto",
            main_kv_placement=placement,
            host_pool_tokens=derived["host_pool_tokens"],
            state_blocks=derived["state_blocks"],
        ),
        geo,
        derived["num_gpu_blocks_override"] * geo.unified_bytes_per_block,
        4,
    )
    validate_long_context_plan(plan)
    assert plan.hot_pool_source == "auto"
    assert plan.hot_pool_groups >= 4 * (513 + 9)
    assert plan.gpu_history_blocks == derived["history_blocks"]


def test_sizing_refreshes_write_rows_from_the_requested_scenario(tmp_path):
    source = tmp_path / "main.json"
    source.write_text(json.dumps({"structure": {"plan": {"geometry": GEOMETRY}}}))
    kwargs, _ = ab.resolve_engine_args(
        _args(
            "--arm", "on-host", "--config", "probe", "--single-pass",
            "--geometry-from", str(source),
        ),
        ab.CONFIGS["probe"],
    )  # fmt: skip
    expected = dict(GEOMETRY, max_model_len=4096, hot_max_write_rows=1024)
    assert kwargs["kernel_config"]["long_context_kv"]["hot_pool_gib"] == (
        ab.hot_pool_gib_for_rows(expected, 1024, 2)
    )


def test_single_pass_budget_guard_skips_before_allocation_and_restores_validator():
    from vllm.config.kernel import LongContextKVConfig
    from vllm.v1.core import kv_cache_utils
    from vllm.v1.core.long_context_kv import plan_long_context_capacity

    geo = _geometry_object(GEOMETRY)
    settings = LongContextKVConfig(
        enabled=True,
        main_kv_placement="host",
        host_pool_tokens=32 * 1568,
        hot_pool_gib=ab.hot_pool_gib_for_rows(GEOMETRY, 2048, 4),
    )
    validate = kv_cache_utils.validate_long_context_plan
    too_small = plan_long_context_capacity(settings, geo, ab.GIB, 4)
    with (
        ab.single_pass_budget_guard(True),
        pytest.raises(ab.SinglePassUnavailable, match="无法单趟"),
    ):
        kv_cache_utils.validate_long_context_plan(too_small)
    assert kv_cache_utils.validate_long_context_plan is validate
    fits = plan_long_context_capacity(settings, geo, 10 * ab.GIB, 4)
    with ab.single_pass_budget_guard(True):
        kv_cache_utils.validate_long_context_plan(fits)
    with ab.single_pass_budget_guard(False):
        assert kv_cache_utils.validate_long_context_plan is validate


def test_run_records_budget_skip_as_skipped(tmp_path, monkeypatch):
    import vllm

    def unavailable(**kwargs):
        raise ab.SinglePassUnavailable("无法单趟：test GPU budget; skipping")

    monkeypatch.setattr(vllm, "LLM", unavailable)
    monkeypatch.setattr(ab.QSARouteProbe, "install", lambda self: None)
    out = tmp_path / "skip.json"
    args = _args(
        "--arm", "on-host", "--config", "main", "--single-pass", "--hot-pool-gib", "12",
        "--out", str(out),
    )  # fmt: skip
    assert ab.cmd_run(args) == 0
    result = json.loads(out.read_text())
    assert result["status"] == "skipped"
    assert "无法单趟" in result["skip_reason"]
    assert "error" not in result


@pytest.mark.parametrize(
    "extra",
    [
        ("--arm", "on-host", "--qsa-route", "host-like"),
        ("--arm", "off", "--qsa-route", "host-like"),
        ("--arm", "on-gpu", "--qsa-pass-rows", "255"),
        ("--arm", "on-host", "--single-pass"),
        ("--arm", "on-host", "--hot-pool-rows", "64"),
        ("--arm", "on-gpu", "--hot-pool-gib", "1", "--hot-pool-rows", "64"),
        ("--arm", "on-host", "--hot-pool-gib", "auto", "--hot-pool-rows", "64"),
        ("--arm", "on-gpu", "--host-pool-gib", "1", "--host-pool-tokens", "64"),
    ],
)
def test_engine_args_refuse_inconsistent_options(extra):
    with pytest.raises(ValueError):
        ab.resolve_engine_args(_args(*extra, "--config", "main"), ab.CONFIGS["main"])


def test_route_checks():
    config = ab.CONFIGS["main"]
    stats = ab.QSARouteProbe().stats
    host_like = _args("--arm", "on-gpu", "--config", "main", "--qsa-route", "host-like")
    checks = ab.route_checks(dict(stats), "gpu", host_like, config, {})
    assert _failed(checks) == {"host-like QSA route used for every attention call"}
    used = dict(stats, host_like_calls=4)
    assert not _failed(ab.route_checks(used, "gpu", host_like, config, {}))
    single = _args("--arm", "on-host", "--config", "main", "--single-pass")
    store = {"host_store": {"rows_per_pass": 2049}}
    ok = dict(stats, host_steps=10)
    assert not _failed(ab.route_checks(ok, "host", single, config, store))
    multi = dict(ok, host_multi_pass_steps=1)
    assert _failed(ab.route_checks(multi, "host", single, config, store))
    plan = {
        "plan": {"geometry": GEOMETRY, "hot_pool_groups": 131072, "max_num_seqs": 4}
    }
    assert _failed(ab.route_checks(ok, "gpu", single, config, plan))


def test_v2_receipts_enforce_read_cap_and_short_context_bounds():
    geo = dict(GEOMETRY, hot_read_rows=256)
    with pytest.raises(ValueError, match="R_cap=256"):
        ab.hot_pool_gib_for_rows(geo, 512, 4)
    assert ab.rows_per_pass(geo, 10_000_000, 8) == 256
    short = dict(geo, max_model_len=128, dcp_world_size=2, dcp_interleave=1)
    assert ab.rows_per_pass(short, 26, 1) == 256
    assert ab.rows_per_pass(short, 79, 8) == 1


def test_route_probe_counts_actual_planner_callbacks():
    probe = ab.QSARouteProbe()
    layer = type("Layer", (), {"rows_per_pass": 16})()
    selection = type("Selection", (), {"shape": (32, 1)})()
    consumed = []

    def original(layer, selection, token_to_req, table, lengths, attend, **kwargs):
        attend(0, 32, None)  # Fixed single-pass decode, despite a 16-row bound.

    probe.passes(original)(
        layer,
        selection,
        None,
        None,
        None,
        lambda start, end, ids: consumed.append((start, end)),
    )
    assert consumed == [(0, 32)]
    assert probe.stats["host_max_passes"] == 1
    assert probe.stats["host_rows_per_pass"] == 32
    assert probe.stats["host_multi_pass_steps"] == 0
