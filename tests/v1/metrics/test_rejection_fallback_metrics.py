# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The count of empty-residual rejection fallbacks reaches /metrics.

The worker keeps a running total on the device and reports it with each step
output; the scheduler turns it into per-step increments in SpecDecodingStats
and the Prometheus logger adds them to a counter. These checks run on CPU.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import vllm
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.spec_decode.metrics import SpecDecodingProm, SpecDecodingStats

ROOT = Path(vllm.__file__).parent


def _scheduler(log_stats: bool = True):
    return SimpleNamespace(
        _rejection_fallbacks_total=0, log_stats=log_stats, num_spec_tokens=4
    )


def _observe(sched, stats, total):
    return Scheduler._observe_rejection_fallbacks(sched, stats, total)


def test_scheduler_counts_increments_of_running_total():
    sched = _scheduler()
    assert _observe(sched, None, None) is None
    assert _observe(sched, None, 0) is None

    stats = _observe(sched, None, 5)
    assert stats is not None and stats.num_rejection_fallbacks == 5

    # Same total again: nothing new; an existing stats object is kept.
    existing = SpecDecodingStats.new(4)
    assert _observe(sched, existing, 5) is existing
    assert existing.num_rejection_fallbacks == 0

    stats = _observe(sched, existing, 8)
    assert stats is existing and stats.num_rejection_fallbacks == 3
    assert sched._rejection_fallbacks_total == 8


def test_scheduler_tracks_total_without_stats_logging():
    sched = _scheduler(log_stats=False)
    assert _observe(sched, None, 7) is None
    assert sched._rejection_fallbacks_total == 7
    # Later increments are measured from the last total, not from zero.
    sched.log_stats = True
    stats = _observe(sched, None, 9)
    assert stats is not None and stats.num_rejection_fallbacks == 2


class _FakeCounter:
    created: list["_FakeCounter"] = []

    def __init__(self, name, documentation, labelnames):
        self.name = name
        self.value = 0
        _FakeCounter.created.append(self)

    def labels(self, *values):
        return self

    def inc(self, amount=1):
        self.value += amount


def test_prometheus_counter_increments(monkeypatch):
    _FakeCounter.created.clear()
    monkeypatch.setattr(SpecDecodingProm, "_counter_cls", _FakeCounter)
    prom = SpecDecodingProm(
        SimpleNamespace(num_speculative_tokens=2), ["model_name"], {0: ["m"]}
    )
    stats = SpecDecodingStats.new(2)
    stats.observe_draft(num_draft_tokens=2, num_accepted_tokens=1)
    stats.num_rejection_fallbacks = 3
    prom.observe(stats, 0)
    prom.observe(SpecDecodingStats.new(2), 0)
    by_name = {c.name: c for c in _FakeCounter.created}
    assert by_name["vllm:spec_decode_num_rejection_fallbacks"].value == 3
    assert by_name["vllm:spec_decode_num_drafts"].value == 1


def test_model_runner_output_defaults_to_no_report():
    output = ModelRunnerOutput(req_ids=[], req_id_to_index={})
    assert output.rejection_fallbacks_total is None


def _source(rel: str) -> str:
    return (ROOT / rel).read_text()


def _keyword_values(src: str, func: str, keyword: str) -> list[str]:
    values = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Call) and ast.unparse(node.func).endswith(func):
            for kw in node.keywords:
                if kw.arg == keyword:
                    values.append(ast.unparse(kw.value))
    return values


def test_every_dense_rejection_call_passes_the_counter():
    for rel, expected in (
        ("v1/worker/gpu/spec_decode/rejection_sampler.py", "self.fallback_count"),
        (
            "v1/worker/gpu/spec_decode/dflash2/sparse_rejection.py",
            "rejection_sampler.fallback_count",
        ),
    ):
        src = _source(rel)
        assert _keyword_values(src, "rejection_sample", "fallback_count") == [
            expected
        ], rel


def test_model_runner_copies_the_counter_with_the_step_output():
    src = _source("v1/worker/gpu/model_runner.py")
    values = _keyword_values(src, "AsyncOutput", "rejection_fallback_count")
    assert len(values) == 1
    assert "self.rejection_sampler.fallback_count" in values[0]


def test_counter_has_one_entry_per_fallback_tier():
    # The insert kernel adds to fallback_count_ptr (second tier) and
    # fallback_count_ptr + 1 (target distribution).
    src = _source("v1/worker/gpu/spec_decode/rejection_sampler.py")
    calls = [
        node
        for node in ast.walk(ast.parse(src))
        if isinstance(node, ast.Assign)
        and ast.unparse(node.targets[0]) == "self.fallback_count"
    ]
    assert len(calls) == 1
    assert ast.unparse(calls[0].value).startswith("torch.zeros(2,")
    kernel = _source("v1/worker/gpu/spec_decode/rejection_sampler_utils.py")
    assert "tl.atomic_add(fallback_count_ptr, 1)" in kernel
    assert "tl.atomic_add(fallback_count_ptr + 1, 1)" in kernel
