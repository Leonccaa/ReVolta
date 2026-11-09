# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks of the long-context KV A/B tool's planning and comparison."""

import copy
import json
import math

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


def test_tight_settings_respect_startup_minimums():
    geometry = {
        "block_tokens": 1616,
        "state_blocks_per_request": 8,
        "ring_blocks_per_request": 1,
        "unified_bytes_per_block": 20_000_000,
        "state_bytes_per_block": 9_000_000,
        "index_bytes_per_block": 1_000_000,
        "main_kv_bytes_per_block": 10_000_000,
    }
    on = ab.tight_settings(geometry, 8192, "on")
    history = math.ceil(8192 / 1616)
    assert on["state_blocks"] == 12 and on["target_history_blocks"] == history + 4
    budget = on["num_gpu_blocks_override"] * geometry["unified_bytes_per_block"]
    left = budget - (on["state_blocks"] + 1) * geometry["state_bytes_per_block"]
    assert left // 11_000_000 - 1 >= on["target_history_blocks"]
    off = ab.tight_settings(geometry, 8192, "off")
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


def test_output_problems():
    assert ab.output_problems(_request()) == []
    short = _request()
    short["max_tokens"] = 5
    assert ab.output_problems(short)
    nan = _request(lp=float("nan").hex())
    assert any("non-finite" in p for p in ab.output_problems(nan))


def _structure(on=True):
    state_n, history_n = 13, 11
    plan = {
        "segment_num_blocks": [state_n, history_n],
        "gpu_state_bytes": state_n * 300,
        "gpu_history_bytes": history_n * 500,
        "gpu_budget_bytes": 10_000,
    }
    tensors = [
        {"size": state_n * 100, "num_blocks": state_n, "segment": "state"},
        {"size": state_n * 200, "num_blocks": state_n, "segment": "state"},
        {"size": history_n * 500, "num_blocks": history_n, "segment": "history"},
    ]
    storages = [
        {
            "ptr": 1000 * i,
            "nbytes": t["size"],
            "layers": [str(i)],
            "segments": [t["segment"]],
        }
        for i, t in enumerate(tensors)
    ]
    return {
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
    }


def test_segment_and_zeroing_checks():
    structure = _structure()
    checks = ab.segment_tensor_checks(structure, "on")
    assert all(c["ok"] for c in checks), checks
    assert all(c["ok"] for c in ab.zeroing_checks(structure, "on"))
    bad = _structure()
    bad["kv_cache_tensors"][0]["size"] += 13
    bad["worker_storages"][0]["nbytes"] += 13
    names = {c["name"] for c in ab.segment_tensor_checks(bad, "on") if not c["ok"]}
    assert names == {"state segment bytes equal the plan"}
    bad["zeroer"]["segment_spaces"] = ["history", "state"]
    assert not all(c["ok"] for c in ab.zeroing_checks(bad, "on"))
    off = _structure(on=False)
    assert all(c["ok"] for c in ab.segment_tensor_checks(off, "off"))


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
    ]
    found = ab.parse_log_lines(lines)
    assert all(
        found[k] for k in ("plan", "available_kv", "graph_capture", "block_size")
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
