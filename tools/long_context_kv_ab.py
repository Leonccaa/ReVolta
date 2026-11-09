# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B end-to-end check of the long-context KV mode on one GPU.

The same token prompts and sampling settings run through an in-process engine
with ``kernel_config.long_context_kv`` off (one block pool) and on (separate
state and history block spaces; the main K/V stays on the GPU). The mode only
renumbers blocks and moves tensor bases, so token IDs and logprobs are expected
to match bit for bit.

Subcommands::

    run      one engine (one arm, one config) -> JSON with outputs and checks
    compare  two run JSONs (plus an optional repeat of the reference arm)
             -> verdict JSON: PASS, FAIL or INCONCLUSIVE, first difference
    summary  verdict JSONs -> one overall verdict

Configs (each arm of a config starts its own process):

    probe  load + two short requests (one prefill split into two chunks)
    main   decode, chunked prefill, four concurrent requests, prefix hits
    tight  small pools: block reuse across waves, then preemption

The engine runs in-process (``VLLM_ENABLE_V1_MULTIPROCESSING=0``) so a run can
inspect the scheduler's block pools and the worker's KV tensors: per-segment
tensor bytes against the capacity plan, the block-zeroing segments, pool
usage after every phase, preemptions and reused block IDs.

Example::

    python -m tools.long_context_kv_ab run --arm off --config main \\
        --model /path/to/model --out off.json
    python -m tools.long_context_kv_ab run --arm on --config main \\
        --model /path/to/model --out on.json
    python -m tools.long_context_kv_ab compare --label main off.json on.json \\
        --out verdict.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import regex as re

GIB = 1 << 30
TOP_LOGPROBS = 5
# Token IDs drawn for synthetic prompts; below any special/added tokens.
TOKEN_LOW = 1000
TOKEN_HIGH = 100_000

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"


# --------------------------------------------------------------------------
# Scenarios
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Req:
    """One request: prompt segments (key, length) and sampling settings."""

    name: str
    segments: tuple[tuple[str, int], ...]
    max_tokens: int
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int | None = None
    prompt_logprobs: int | None = None

    @property
    def prompt_len(self) -> int:
        return sum(length for _, length in self.segments)


@dataclass(frozen=True)
class Phase:
    """Requests submitted together in one ``generate`` call."""

    name: str
    requests: tuple[Req, ...]
    expect_cached: bool = False
    """Every request must report a prefix-cache hit."""


@dataclass(frozen=True)
class Scenario:
    name: str
    phases: tuple[Phase, ...]
    parity: str = "hard"
    """``hard``: off and on must match bit for bit. ``soft``: compared and
    reported, but scheduling differs between arms (preemption), so a
    difference does not fail the run."""
    expect_preemption: bool = False
    """The long-context arm must preempt at least once."""
    expect_reuse: bool = False
    """Later phases must reuse block IDs freed by earlier ones, per pool."""


def _one(name: str, key: str, length: int, max_tokens: int, **kw: Any) -> Req:
    return Req(name, ((key, length),), max_tokens, **kw)


SCENARIOS: dict[str, Scenario] = {
    "probe": Scenario(
        "probe",
        (
            Phase(
                "p0",
                (
                    _one("probe-short", "probe-a", 200, 16, prompt_logprobs=1),
                    # Two prefill chunks at max_num_batched_tokens=1024.
                    _one("probe-chunked", "probe-b", 1400, 24),
                ),
            ),
        ),
    ),
    "short": Scenario(
        "short",
        (Phase("s0", (_one("short", "short", 320, 32, prompt_logprobs=1),)),),
    ),
    "chunked": Scenario(
        # Three prefill chunks at max_num_batched_tokens=2048, then decode.
        "chunked",
        (Phase("c0", (_one("chunked", "chunked", 5000, 96),)),),
    ),
    "concurrent": Scenario(
        "concurrent",
        (
            Phase(
                "x0",
                (
                    _one("x-700", "x-a", 700, 160),
                    _one(
                        "x-1900-seeded",
                        "x-b",
                        1900,
                        96,
                        temperature=0.8,
                        top_p=0.95,
                        seed=1234,
                    ),
                    _one("x-3300", "x-c", 3300, 128),
                    _one("x-4300", "x-d", 4300, 64),
                ),
            ),
        ),
    ),
    "prefix": Scenario(
        "prefix",
        (
            Phase(
                "a-cold",
                (Req("pfx-a", (("pfx-X", 3500), ("tail-a", 120)), 48),),
            ),
            Phase(
                "b-hit",
                (Req("pfx-b", (("pfx-X", 3500), ("tail-b", 200)), 48),),
                expect_cached=True,
            ),
            Phase(
                "c-hit-pair",
                (
                    Req("pfx-c", (("pfx-X", 3500), ("tail-c", 90)), 48),
                    Req("pfx-d", (("pfx-X", 3500), ("tail-d", 300)), 64),
                ),
                expect_cached=True,
            ),
        ),
    ),
    "reuse": Scenario(
        # One request per wave: never preempted, so both arms schedule the
        # same; the small pools make later waves take freed block IDs (state
        # blocks are not zeroed in the long-context arm, history blocks are).
        "reuse",
        (
            Phase("integration requirement", (_one("r1", "r-1", 2600, 64),)),
            Phase("integration requirement", (_one("r2", "r-2", 3000, 64),)),
            Phase("integration requirement", (_one("r3", "r-3", 4000, 48),)),
            Phase("integration requirement", (_one("r4", "r-4", 1800, 64),)),
            Phase("integration requirement", (_one("r5", "r-5", 3500, 64),)),
        ),
        expect_reuse=True,
    ),
    "preempt": Scenario(
        "preempt",
        (
            Phase(
                "g0",
                (
                    _one("g-1200", "g-a", 1200, 1000),
                    _one("g-1700", "g-b", 1700, 1000),
                    _one("g-2300", "g-c", 2300, 1000),
                    _one("g-2900", "g-d", 2900, 1000),
                ),
            ),
        ),
        parity="soft",
        expect_preemption=True,
    ),
}


@dataclass(frozen=True)
class EngineConfig:
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    scenarios: tuple[str, ...]


CONFIGS: dict[str, EngineConfig] = {
    "probe": EngineConfig(4096, 2, 1024, ("probe",)),
    "main": EngineConfig(8192, 4, 2048, ("short", "chunked", "concurrent", "prefix")),
    "tight": EngineConfig(8192, 4, 2048, ("reuse", "preempt")),
}


def make_tokens(key: str, length: int, seed: int) -> list[int]:
    """Deterministic synthetic token IDs for one prompt segment."""
    rng = random.Random(f"{seed}:{key}")
    return [rng.randrange(TOKEN_LOW, TOKEN_HIGH) for _ in range(length)]


def prompt_tokens(req: Req, seed: int) -> list[int]:
    return [t for key, length in req.segments for t in make_tokens(key, length, seed)]


def validate_config(config: EngineConfig) -> None:
    """Every request fits max_model_len and every scenario exists."""
    for name in config.scenarios:
        for phase in SCENARIOS[name].phases:
            for req in phase.requests:
                if req.prompt_len + req.max_tokens > config.max_model_len:
                    raise ValueError(
                        f"{req.name}: {req.prompt_len}+{req.max_tokens} tokens "
                        f"exceed max_model_len={config.max_model_len}"
                    )
            if len(phase.requests) > config.max_num_seqs * 4:
                raise ValueError(f"{phase.name}: too many requests")


# --------------------------------------------------------------------------
# Tight-pool sizing from a recorded geometry
# --------------------------------------------------------------------------


def tight_settings(
    geometry: dict[str, Any], max_model_len: int, arm: str
) -> dict[str, int]:
    """Pool sizes that make the tight config reuse blocks and preempt.

    The long-context arm keeps one max_model_len request's worth of state
    blocks plus half again, and history for one such request plus three
    blocks; ``num_gpu_blocks_override`` fixes the budget the plan splits.
    The reference arm gets a unified pool of about the same per-request
    slack. Every value still passes the startup checks.
    """
    block_tokens = int(geometry["block_tokens"])
    s_req = int(geometry["state_blocks_per_request"])
    ring = int(geometry["ring_blocks_per_request"])
    history = math.ceil(max_model_len / block_tokens)
    unified = int(geometry["unified_bytes_per_block"])
    if arm == "off":
        return {"num_gpu_blocks_override": s_req + history + ring + max(3, s_req // 2)}
    state_blocks = s_req + max(1, s_req // 2)
    history_blocks = history + ring + 3
    state_bytes = (state_blocks + 1) * int(geometry["state_bytes_per_block"])
    history_bytes = (history_blocks + 1) * (
        int(geometry["index_bytes_per_block"])
        + int(geometry["main_kv_bytes_per_block"])
    )
    return {
        "state_blocks": state_blocks,
        "num_gpu_blocks_override": -(-(state_bytes + history_bytes) // unified),
        "target_history_blocks": history_blocks,
    }


# --------------------------------------------------------------------------
# Serialization of outputs
# --------------------------------------------------------------------------


def _fhex(value: float) -> str:
    return float(value).hex()


def _position(entry: Any) -> list[list[Any]] | None:
    """One logprob position as sorted [token, logprob hex, rank] rows."""
    if entry is None:
        return None
    rows = []
    for token, lp in entry.items():
        rows.append([int(token), _fhex(lp.logprob), lp.rank])
    rows.sort(key=lambda r: r[0])
    return rows


def _positions(entries: Iterable[Any] | None) -> list[Any] | None:
    if entries is None:
        return None
    return [_position(entry) for entry in entries]


def serialize_output(name: str, req: Req, out: Any) -> dict[str, Any]:
    completion = out.outputs[0]
    return {
        "name": name,
        "request_id": out.request_id,
        "prompt_len": len(out.prompt_token_ids or []),
        "expected_prompt_len": req.prompt_len,
        "num_cached_tokens": out.num_cached_tokens,
        "token_ids": [int(t) for t in completion.token_ids],
        "max_tokens": req.max_tokens,
        "finish_reason": completion.finish_reason,
        "logprobs": _positions(completion.logprobs),
        "prompt_logprobs": _positions(out.prompt_logprobs),
    }


def _finite_hex(value: str) -> bool:
    return math.isfinite(float.fromhex(value))


def output_problems(rec: dict[str, Any]) -> list[str]:
    """Hard per-request problems: length, finish, non-finite logprobs."""
    problems = []
    if rec["prompt_len"] != rec["expected_prompt_len"]:
        problems.append("prompt length mismatch")
    if len(rec["token_ids"]) != rec["max_tokens"]:
        problems.append(
            f"generated {len(rec['token_ids'])} of {rec['max_tokens']} tokens"
        )
    lps = rec["logprobs"] or []
    if len(lps) != len(rec["token_ids"]):
        problems.append("logprob positions != generated tokens")
    for pos, (token, rows) in enumerate(zip(rec["token_ids"], lps)):
        if not rows:
            problems.append(f"no logprobs at position {pos}")
            break
        if token not in {row[0] for row in rows}:
            problems.append(f"sampled token missing from logprobs at {pos}")
            break
        if not all(_finite_hex(row[1]) for row in rows):
            problems.append(f"non-finite logprob at position {pos}")
            break
    for pos, rows in enumerate(rec["prompt_logprobs"] or []):
        if rows and not all(_finite_hex(row[1]) for row in rows):
            problems.append(f"non-finite prompt logprob at position {pos}")
            break
    return problems


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


def _first_list_diff(a: Sequence[Any], b: Sequence[Any]) -> int | None:
    for index, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return index
    return None if len(a) == len(b) else min(len(a), len(b))


def compare_request(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any] | None:
    """First difference between two serialized requests, or None."""
    for key in ("prompt_len", "num_cached_tokens", "finish_reason"):
        if a[key] != b[key]:
            return {"field": key, "a": a[key], "b": b[key]}
    index = _first_list_diff(a["token_ids"], b["token_ids"])
    if index is not None:
        return {
            "field": "token_ids",
            "position": index,
            "a": a["token_ids"][index : index + 4],
            "b": b["token_ids"][index : index + 4],
        }
    for key in ("logprobs", "prompt_logprobs"):
        la, lb = a[key] or [], b[key] or []
        index = _first_list_diff(la, lb)
        if index is not None:
            return {
                "field": key,
                "position": index,
                "a": la[index] if index < len(la) else None,
                "b": lb[index] if index < len(lb) else None,
            }
    return None


def compare_runs(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """Scenario-by-scenario comparison of two run JSONs."""
    scenarios = []
    for sa in a.get("scenarios", []):
        sb = next((s for s in b.get("scenarios", []) if s["name"] == sa["name"]), None)
        entry: dict[str, Any] = {
            "name": sa["name"],
            "parity": sa["parity"],
            "requests": 0,
            "tokens": 0,
            "first_difference": None,
        }
        if sb is None:
            entry["first_difference"] = {"field": "missing scenario"}
            scenarios.append(entry)
            continue
        reqs_a = [r for p in sa["phases"] for r in p["requests"]]
        reqs_b = {r["name"]: r for p in sb["phases"] for r in p["requests"]}
        diffs = 0
        for ra in reqs_a:
            rb = reqs_b.get(ra["name"])
            entry["requests"] += 1
            entry["tokens"] += len(ra["token_ids"])
            diff = (
                {"field": "missing request"} if rb is None else compare_request(ra, rb)
            )
            if diff is not None:
                diffs += 1
                if entry["first_difference"] is None:
                    entry["first_difference"] = {"request": ra["name"], **diff}
        entry["identical"] = entry["first_difference"] is None
        entry["requests_different"] = diffs
        scenarios.append(entry)
    hard = [s for s in scenarios if s["parity"] == "hard"]
    return {
        "scenarios": scenarios,
        "hard_identical": all(s["identical"] for s in hard) and bool(hard),
        "soft_differences": [
            s["name"] for s in scenarios if s["parity"] == "soft" and not s["identical"]
        ],
    }


def failed_checks(run: dict[str, Any]) -> list[str]:
    if run.get("status") != "ok":
        return [f"run status {run.get('status')}: {run.get('error', '')[:300]}"]
    return [
        f"{c['name']}: {c.get('detail', '')}"
        for c in run.get("checks", [])
        if c.get("ok") is False
    ]


def verdict(
    a: dict[str, Any], b: dict[str, Any], control: dict[str, Any] | None = None
) -> dict[str, Any]:
    """PASS: checks pass and hard scenarios match bit for bit (and the repeat
    of the reference arm matches itself). INCONCLUSIVE: the reference arm
    does not reproduce itself. FAIL otherwise."""
    reasons = [f"A {x}" for x in failed_checks(a)] + [
        f"B {x}" for x in failed_checks(b)
    ]
    result: dict[str, Any] = {"a": a.get("label"), "b": b.get("label")}
    ab = compare_runs(a, b) if not reasons else None
    result["a_vs_b"] = ab
    if control is not None:
        reasons += [f"control {x}" for x in failed_checks(control)]
        result["control"] = control.get("label")
        result["a_vs_control"] = compare_runs(a, control) if not reasons else None
    if reasons:
        result.update(verdict=FAIL, reasons=reasons)
        return result
    assert ab is not None
    control_ok = control is None or result["a_vs_control"]["hard_identical"]
    if ab["hard_identical"] and control_ok:
        result.update(verdict=PASS, reasons=[])
    elif not control_ok:
        result.update(
            verdict=INCONCLUSIVE,
            reasons=["the reference arm does not reproduce itself"],
        )
    else:
        first = next(s for s in ab["scenarios"] if not s["identical"])
        result.update(
            verdict=FAIL,
            reasons=[f"{first['name']}: {json.dumps(first['first_difference'])}"],
        )
    return result


# --------------------------------------------------------------------------
# Log capture
# --------------------------------------------------------------------------

LOG_PATTERNS = {
    "plan": re.compile(r"long-context KV plan"),
    "available_kv": re.compile(r"Available KV cache memory: ([0-9.]+) GiB"),
    "graph_capture": re.compile(r"Graph capturing finished"),
    "kv_cache_size": re.compile(r"GPU KV cache size: ([0-9,]+) tokens"),
    "block_size": re.compile(r"Setting attention block size to (\d+) tokens"),
}


class LogCapture(logging.Handler):
    """Keeps matching vLLM log lines and every ERROR record."""

    def __init__(self) -> None:
        super().__init__(level=logging.INFO)
        self.matches: dict[str, list[str]] = {k: [] for k in LOG_PATTERNS}
        self.errors: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - malformed record
            return
        for key, pattern in LOG_PATTERNS.items():
            if pattern.search(message):
                self.matches[key].append(message[:2000])
        if record.levelno >= logging.ERROR:
            self.errors.append(message[:2000])


def parse_log_lines(lines: Iterable[str]) -> dict[str, list[str]]:
    """The same matching over plain text lines (for saved logs and tests)."""
    found: dict[str, list[str]] = {k: [] for k in LOG_PATTERNS}
    for line in lines:
        for key, pattern in LOG_PATTERNS.items():
            if pattern.search(line):
                found[key].append(line.rstrip("\n"))
    return found


# --------------------------------------------------------------------------
# Structural checks (pure functions over recorded dicts)
# --------------------------------------------------------------------------


def check(name: str, ok: bool | None, detail: Any = "") -> dict[str, Any]:
    return {"name": name, "ok": ok, "detail": detail}


def segment_tensor_checks(structure: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """Tensor bytes and block counts against the plan; storages per segment."""
    checks = []
    plan = structure.get("plan")
    tensors = structure["kv_cache_tensors"]
    storages = structure.get("worker_storages", [])
    total = sum(t["size"] for t in tensors)
    worker_total = sum(s["nbytes"] for s in storages)
    checks.append(
        check(
            "worker allocated the configured KV tensors",
            worker_total == total and len(storages) == len(tensors),
            {
                "config_bytes": total,
                "worker_bytes": worker_total,
                "config_tensors": len(tensors),
                "worker_storages": len(storages),
                "layers_missing": structure.get(
                    "layers_missing_from_forward_context", []
                )[:5],
            },
        )
    )
    mixed = [s for s in storages if len(s["segments"]) != 1]
    checks.append(
        check(
            "each KV storage serves one block space",
            not mixed,
            [s["layers"][:3] for s in mixed][:5],
        )
    )
    if arm == "off":
        checks.append(check("reference arm has no block segments", plan is None, ""))
        return checks
    if plan is None:
        checks.append(check("long-context plan present", False, "no plan"))
        return checks
    segments = structure.get("block_segments")
    state_n, history_n = plan["segment_num_blocks"]
    checks.append(
        check(
            "block segments match the plan",
            segments is not None
            and list(segments["num_blocks"]) == [state_n, history_n]
            and list(segments["names"]) == ["state", "history"],
            {"segments": segments, "plan": [state_n, history_n]},
        )
    )
    by_segment = {"state": 0, "history": 0}
    bad_blocks = []
    for t in tensors:
        n = t["num_blocks"]
        if n not in (state_n, history_n) or t["size"] % n:
            bad_blocks.append(t)
        seg = t.get("segment")
        if seg in by_segment:
            by_segment[seg] += t["size"]
    checks.append(
        check(
            "tensor sizes are whole blocks of their segment",
            not bad_blocks,
            bad_blocks[:3],
        )
    )
    checks.append(
        check(
            "state segment bytes equal the plan",
            by_segment["state"] == plan["gpu_state_bytes"],
            {"tensors": by_segment["state"], "plan": plan["gpu_state_bytes"]},
        )
    )
    checks.append(
        check(
            "history segment bytes equal the plan",
            by_segment["history"] == plan["gpu_history_bytes"],
            {"tensors": by_segment["history"], "plan": plan["gpu_history_bytes"]},
        )
    )
    checks.append(
        check(
            "KV tensors fit the planned budget",
            total <= plan["gpu_budget_bytes"],
            {
                "tensors": total,
                "budget": plan["gpu_budget_bytes"],
                "unused": plan["gpu_budget_bytes"] - total,
            },
        )
    )
    return checks


def zeroing_checks(structure: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    zero = structure.get("zeroer")
    if zero is None:
        return [check("block zeroer present", None, "no zeroer")]
    segs = zero["segment_spaces"]
    unknown = [s for s in segs if s is None]
    checks = [
        check("every zeroing segment lies in a KV storage", not unknown, len(unknown))
    ]
    if arm == "on":
        spaces = {s for s in segs if s is not None}
        checks.append(
            check(
                "zeroing covers only the history segment",
                bool(segs) and spaces == {"history"},
                {"segments": len(segs), "spaces": sorted(spaces)},
            )
        )
    else:
        checks.append(check("zeroing segments (reference)", None, len(segs)))
    return checks


def pool_checks(scenarios: list[dict[str, Any]]) -> list[dict[str, Any]]:
    leaks = [
        {"scenario": s["name"], "phase": p["name"], "pools": p["pools_after"]}
        for s in scenarios
        for p in s["phases"]
        if any(pool["free"] != pool["usable"] for pool in p["pools_after"])
    ]
    return [check("every pool is fully free after each phase", not leaks, leaks[:3])]


def reuse_overlap(phases: list[dict[str, Any]]) -> list[int]:
    """Per pool: block IDs allocated in a phase that an earlier phase used."""
    if not phases:
        return []
    num_pools = len(phases[0]["allocated"])
    overlap = [0] * num_pools
    seen: list[set[int]] = [set() for _ in range(num_pools)]
    for phase in phases:
        for index, ids in enumerate(phase["allocated"]):
            overlap[index] += len(seen[index] & set(ids))
            seen[index] |= set(ids)
    return overlap


def scenario_checks(scenarios: list[dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    checks = []
    problems = [
        {"request": r["name"], "problems": output_problems(r)}
        for s in scenarios
        for p in s["phases"]
        for r in p["requests"]
        if output_problems(r)
    ]
    checks.append(
        check("requests complete with finite logprobs", not problems, problems[:5])
    )
    for s in scenarios:
        for p in s["phases"]:
            if p["expect_cached"]:
                cached = [r["num_cached_tokens"] for r in p["requests"]]
                checks.append(
                    check(
                        f"{s['name']}/{p['name']} hits the prefix cache",
                        all((c or 0) > 0 for c in cached),
                        cached,
                    )
                )
        if s["expect_preemption"]:
            count = sum(len(p["preemptions"]) for p in s["phases"])
            checks.append(
                check(
                    f"{s['name']} preempts",
                    count > 0 if arm == "on" else None,
                    count,
                )
            )
        if s["expect_reuse"]:
            overlap = reuse_overlap(s["phases"])
            checks.append(
                check(
                    f"{s['name']} reuses freed block IDs in every pool",
                    bool(overlap) and all(n > 0 for n in overlap),
                    overlap,
                )
            )
    return checks


# --------------------------------------------------------------------------
# Engine side (imports vLLM and torch lazily)
# --------------------------------------------------------------------------


def _tensor_leaves(value: Any) -> Iterable[Any]:
    import torch

    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _tensor_leaves(item)


def _plan_dict(plan: Any) -> dict[str, Any] | None:
    if plan is None:
        return None
    data = dataclasses.asdict(plan)
    data.update(
        gpu_state_bytes=plan.gpu_state_bytes,
        gpu_history_bytes=plan.gpu_history_bytes,
        history_bytes_per_block=plan.history_bytes_per_block,
        segment_num_blocks=list(plan.segment_num_blocks()),
        max_fitting_len=plan.max_fitting_len(),
        describe=plan.describe(),
    )
    return data


def _group_spaces(config: Any) -> list[str]:
    segments = getattr(config, "block_segments", None)
    if segments is None:
        return ["unified"] * len(config.kv_cache_groups)
    return [segments.names[s] for s in segments.group_segments]


def inspect_structure(sched: Any, runner: Any) -> dict[str, Any]:
    """Scheduler-side config and plan; worker-side storages and zeroing."""
    config = runner.kv_cache_config
    spaces = _group_spaces(config)
    layer_space = {
        name: spaces[gid]
        for gid, group in enumerate(config.kv_cache_groups)
        for name in group.layer_names
    }
    groups = []
    for gid, group in enumerate(config.kv_cache_groups):
        spec = group.kv_cache_spec
        members = getattr(spec, "kv_cache_specs", None)
        kinds = sorted(
            {type(m).__name__ for m in members.values()}
            if members
            else {type(spec).__name__}
        )
        groups.append(
            {
                "id": gid,
                "space": spaces[gid],
                "kinds": kinds,
                "block_size": spec.block_size,
                "page_size_bytes": spec.page_size_bytes,
                "layers": len(group.layer_names),
                "first_layers": list(group.layer_names[:2]),
            }
        )
    tensors = []
    for t in config.kv_cache_tensors:
        tspaces = {layer_space.get(name) for name in t.shared_by}
        tensors.append(
            {
                "size": t.size,
                "num_blocks": config.tensor_num_blocks(t),
                "shared_by": list(t.shared_by)[:4],
                "members": len(t.shared_by),
                "segment": tspaces.pop()
                if len(tspaces) == 1
                else sorted(str(s) for s in tspaces),
            }
        )
    context = runner.compilation_config.static_forward_context
    storages: dict[int, dict[str, Any]] = {}
    missing_layers: list[str] = []
    for name, space in layer_space.items():
        layer = context.get(name)
        if layer is None:
            missing_layers.append(name)
            continue
        for tensor in _tensor_leaves(getattr(layer, "kv_cache", None)):
            storage = tensor.untyped_storage()
            entry = storages.setdefault(
                storage.data_ptr(),
                {"nbytes": storage.nbytes(), "layers": [], "segments": set()},
            )
            if name not in entry["layers"]:
                entry["layers"].append(name)
            entry["segments"].add(space)
    storage_list = [
        {
            "ptr": ptr,
            "nbytes": s["nbytes"],
            "layers": s["layers"],
            "segments": sorted(s["segments"]),
        }
        for ptr, s in sorted(storages.items())
    ]
    zero: dict[str, Any] | None = None
    zeroer = getattr(runner, "_kv_block_zeroer", None)
    meta = getattr(zeroer, "_meta", None)
    if zeroer is not None:
        addrs: list[int] = []
        if meta is not None:
            import torch

            addrs = meta[0].view(torch.int64).cpu().tolist()
        spaces_of: list[str | None] = []
        for addr in addrs:
            hit = next(
                (s for s in storage_list if s["ptr"] <= addr < s["ptr"] + s["nbytes"]),
                None,
            )
            if hit is None or len(hit["segments"]) != 1:
                spaces_of.append(None)
            else:
                spaces_of.append(hit["segments"][0])
        zero = {"segments": len(addrs), "segment_spaces": spaces_of}
    segments = getattr(config, "block_segments", None)
    plan = getattr(sched.kv_cache_config, "long_context_plan", None)
    return {
        "num_blocks": config.num_blocks,
        "groups": groups,
        "kv_cache_tensors": tensors,
        "block_segments": None
        if segments is None
        else {
            "names": list(segments.names),
            "num_blocks": list(segments.num_blocks),
            "group_segments": list(segments.group_segments),
            "primary": segments.primary,
        },
        "plan": _plan_dict(plan),
        "worker_storages": storage_list,
        "layers_missing_from_forward_context": missing_layers,
        "zeroer": zero,
        "scheduler_pools": [p.num_gpu_blocks for p in _pools(sched)],
    }


def _pools(sched: Any) -> tuple[Any, ...]:
    coordinator = sched.kv_cache_manager.coordinator
    pools = getattr(coordinator, "block_pools", ())
    return tuple(pools) if pools else (sched.kv_cache_manager.block_pool,)


def _pool_state(sched: Any) -> list[dict[str, int]]:
    return [
        {
            "blocks": p.num_gpu_blocks,
            "usable": p.num_gpu_blocks - 1,
            "free": p.get_num_free_blocks(),
        }
        for p in _pools(sched)
    ]


class Recorder:
    """Instance-level hooks on the scheduler, its pools and the zeroer."""

    def __init__(self, sched: Any, runner: Any) -> None:
        self.sched = sched
        self.allocated: list[list[int]] = []
        self.preemptions: list[dict[str, Any]] = []
        self.zero_calls = 0
        self.zero_blocks = 0
        self.zero_max_id = -1
        for index, pool in enumerate(_pools(sched)):
            self._wrap_pool(index, pool)
        self._wrap_preempt()
        zeroer = getattr(runner, "_kv_block_zeroer", None)
        if zeroer is not None:
            self._wrap_zeroer(zeroer)
        self.reset_phase()

    def reset_phase(self) -> None:
        self.allocated = [[] for _ in _pools(self.sched)]
        self.preemptions = []

    def _wrap_pool(self, index: int, pool: Any) -> None:
        original = pool.get_new_blocks

        def get_new_blocks(*args: Any, **kwargs: Any) -> Any:
            blocks = original(*args, **kwargs)
            self.allocated[index].extend(b.block_id for b in blocks)
            return blocks

        pool.get_new_blocks = get_new_blocks

    def _wrap_preempt(self) -> None:
        sched = self.sched
        original = sched._preempt_request

        def preempt(request: Any, *args: Any, **kwargs: Any) -> Any:
            self.preemptions.append(
                {
                    "request_id": request.request_id,
                    "num_computed_tokens": request.num_computed_tokens,
                    "pools_free": [p.get_num_free_blocks() for p in _pools(sched)],
                }
            )
            return original(request, *args, **kwargs)

        sched._preempt_request = preempt

    def _wrap_zeroer(self, zeroer: Any) -> None:
        original = zeroer.zero_block_ids

        def zero_block_ids(block_ids: list[int]) -> None:
            if block_ids:
                self.zero_calls += 1
                self.zero_blocks += len(block_ids)
                self.zero_max_id = max(self.zero_max_id, max(block_ids))
            original(block_ids)

        zeroer.zero_block_ids = zero_block_ids


def _memory() -> dict[str, Any]:
    import torch

    free, total = torch.cuda.mem_get_info()
    acc = getattr(torch, "accelerator", None)
    out: dict[str, Any] = {"device_used": total - free, "device_total": total}
    for key in ("memory_allocated", "max_memory_allocated", "memory_reserved"):
        fn = getattr(acc, key, None)
        if fn is not None:
            try:
                out[key] = int(fn())
            except Exception as exc:  # pragma: no cover - API differences
                out[key] = f"unavailable: {exc}"
    try:
        smi = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        out["nvidia_smi_mib"] = smi.stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        out["nvidia_smi_mib"] = f"unavailable: {exc}"
    return out


def _sampling(req: Req) -> Any:
    from vllm import SamplingParams

    return SamplingParams(
        n=1,
        temperature=req.temperature,
        top_p=req.top_p,
        seed=req.seed,
        max_tokens=req.max_tokens,
        ignore_eos=True,
        logprobs=TOP_LOGPROBS,
        prompt_logprobs=req.prompt_logprobs,
        detokenize=False,
    )


def _preempted_names(
    preemptions: list[dict[str, Any]], records: list[dict[str, Any]]
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in preemptions:
        rid = str(event["request_id"])
        for rec in records:
            ext = str(rec["request_id"])
            if rid == ext or rid.startswith(ext + "-"):
                counts[rec["name"]] = counts.get(rec["name"], 0) + 1
    return counts


def run_scenarios(
    llm: Any, sched: Any, recorder: Recorder, names: Sequence[str], seed: int
) -> list[dict[str, Any]]:
    results = []
    for name in names:
        scenario = SCENARIOS[name]
        reset_ok = llm.reset_prefix_cache()
        phases = []
        t0 = time.time()
        for phase in scenario.phases:
            recorder.reset_phase()
            prompts = [
                {"prompt_token_ids": prompt_tokens(req, seed)} for req in phase.requests
            ]
            params = [_sampling(req) for req in phase.requests]
            start = time.time()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            records = [
                serialize_output(req.name, req, out)
                for req, out in zip(phase.requests, outputs)
            ]
            ledger = getattr(sched.kv_cache_manager, "admission_ledger", None)
            phases.append(
                {
                    "name": phase.name,
                    "seconds": round(time.time() - start, 3),
                    "expect_cached": phase.expect_cached,
                    "requests": records,
                    "pools_after": _pool_state(sched),
                    "allocated": [sorted(set(ids)) for ids in recorder.allocated],
                    "allocated_count": [len(ids) for ids in recorder.allocated],
                    "preemptions": list(recorder.preemptions),
                    "preempted_requests": _preempted_names(
                        recorder.preemptions, records
                    ),
                    "ledger": None if ledger is None else ledger.snapshot(),
                }
            )
        results.append(
            {
                "name": name,
                "parity": scenario.parity,
                "expect_preemption": scenario.expect_preemption,
                "expect_reuse": scenario.expect_reuse,
                "prefix_reset_before": reset_ok,
                "seconds": round(time.time() - t0, 3),
                "phases": phases,
            }
        )
    return results


def _long_context_settings(args: argparse.Namespace) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "enabled": True,
        "host_pool_tokens": args.host_pool_tokens,
        "hot_pool_gib": args.hot_pool_gib,
    }
    if args.state_blocks is not None:
        settings["state_blocks"] = args.state_blocks
    return settings


def resolve_engine_args(
    args: argparse.Namespace, config: EngineConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """LLM keyword arguments and the derived settings recorded in the JSON."""
    derived: dict[str, Any] = {}
    if args.config == "tight" and args.geometry_from:
        with open(args.geometry_from) as f:
            geometry = json.load(f)["structure"]["plan"]["geometry"]
        derived = tight_settings(geometry, config.max_model_len, args.arm)
        if args.arm == "on" and args.state_blocks is None:
            args.state_blocks = derived["state_blocks"]
        if args.num_gpu_blocks_override is None:
            args.num_gpu_blocks_override = derived["num_gpu_blocks_override"]
    kernel_config: dict[str, Any] = json.loads(args.kernel_config_extra)
    if args.arm == "on":
        kernel_config["long_context_kv"] = _long_context_settings(args)
    kwargs: dict[str, Any] = dict(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        seed=args.seed,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        enable_prefix_caching=True,
        mamba_cache_mode="align",
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=not args.graphs,
        kernel_config=kernel_config,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    if args.attention_backend:
        kwargs["attention_backend"] = args.attention_backend
    if args.num_gpu_blocks_override is not None:
        kwargs["num_gpu_blocks_override"] = args.num_gpu_blocks_override
    if args.async_scheduling != "default":
        kwargs["async_scheduling"] = args.async_scheduling == "on"
    return kwargs, derived


def cmd_run(args: argparse.Namespace) -> int:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    config = CONFIGS[args.config]
    validate_config(config)
    # Importing vllm runs dictConfig on the "vllm" logger, which drops any
    # handler added before it; import first so the capture survives.
    import vllm.logger  # noqa: F401

    capture = LogCapture()
    logging.getLogger("vllm").addHandler(capture)
    result: dict[str, Any] = {
        "label": args.label or f"{args.config}-{args.arm}",
        "arm": args.arm,
        "config": args.config,
        "graphs": args.graphs,
        "engine_config": dataclasses.asdict(config),
        "env": {
            k: v
            for k, v in os.environ.items()
            if k.startswith(("VLLM_", "CUDA_", "TORCH", "PYTORCH"))
        },
        "status": "init",
    }
    t0 = time.time()
    exit_code = 0
    try:
        kwargs, derived = resolve_engine_args(args, config)
        result["engine_kwargs"] = {k: v for k, v in kwargs.items() if k != "model"}
        result["model"] = args.model
        result["derived_tight_settings"] = derived
        from vllm import LLM

        llm = LLM(**kwargs)
        result["init_seconds"] = round(time.time() - t0, 1)
        result["memory_after_init"] = _memory()
        engine = llm.llm_engine
        sched = engine.engine_core.engine_core.scheduler
        runner = engine.model_executor.driver_worker.worker.model_runner
        result["runner"] = type(runner).__module__ + "." + type(runner).__name__
        structure = inspect_structure(sched, runner)
        result["structure"] = structure
        recorder = Recorder(sched, runner)
        result["status"] = "running"
        scenarios = run_scenarios(llm, sched, recorder, config.scenarios, args.seed)
        result["scenarios"] = scenarios
        result["memory_end"] = _memory()
        result["zeroing_calls"] = {
            "calls": recorder.zero_calls,
            "blocks": recorder.zero_blocks,
            "max_block_id": recorder.zero_max_id,
        }
        checks = segment_tensor_checks(structure, args.arm)
        checks += zeroing_checks(structure, args.arm)
        primary_blocks = structure["num_blocks"]
        checks.append(
            check(
                "zeroed block IDs stay in the primary block space",
                recorder.zero_max_id < primary_blocks,
                {"max_id": recorder.zero_max_id, "primary_blocks": primary_blocks},
            )
        )
        checks += pool_checks(scenarios)
        checks += scenario_checks(scenarios, args.arm)
        checks.append(
            check(
                "log capture is live",
                bool(capture.matches["available_kv"]),
                capture.matches["available_kv"][:1],
            )
        )
        plan_lines = capture.matches["plan"]
        checks.append(
            check(
                "capacity plan logged" if args.arm == "on" else "no plan logged",
                bool(plan_lines) if args.arm == "on" else not plan_lines,
                plan_lines[:1],
            )
        )
        if args.graphs:
            checks.append(
                check(
                    "CUDA graphs captured",
                    bool(capture.matches["graph_capture"]),
                    capture.matches["graph_capture"][:1],
                )
            )
        checks.append(
            check("no ERROR log records", not capture.errors, capture.errors[:5])
        )
        result["checks"] = checks
        result["status"] = "ok"
        if any(c["ok"] is False for c in checks):
            exit_code = 3
    except Exception as exc:
        result["status"] = "error" if result["status"] == "running" else "init_failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        exit_code = 2
    result["log_matches"] = capture.matches
    result["log_errors"] = capture.errors[:20]
    result["total_seconds"] = round(time.time() - t0, 1)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1, default=str)
    failed = [c["name"] for c in result.get("checks", []) if c["ok"] is False]
    print(
        f"[long_context_kv_ab] {result['label']}: status={result['status']} "
        f"failed_checks={failed} seconds={result['total_seconds']}",
        flush=True,
    )
    return exit_code


def _load(path: str) -> dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def cmd_compare(args: argparse.Namespace) -> int:
    a, b = _load(args.a), _load(args.b)
    control = _load(args.control) if args.control else None
    result = verdict(a, b, control)
    result["label"] = args.label
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(
        f"[long_context_kv_ab] {args.label}: {result['verdict']} "
        f"{'; '.join(result['reasons'])}",
        flush=True,
    )
    return 0 if result["verdict"] == PASS else 1


def cmd_summary(args: argparse.Namespace) -> int:
    verdicts = [_load(path) for path in args.verdicts]
    states = [v["verdict"] for v in verdicts]
    overall = (
        FAIL
        if FAIL in states
        else INCONCLUSIVE
        if INCONCLUSIVE in states or not states
        else PASS
    )
    summary = {
        "verdict": overall,
        "comparisons": [
            {
                "label": v.get("label"),
                "verdict": v["verdict"],
                "reasons": v["reasons"],
                "scenarios": [
                    {
                        k: s[k]
                        for k in (
                            "name",
                            "parity",
                            "identical",
                            "requests",
                            "tokens",
                            "first_difference",
                        )
                    }
                    for s in (v.get("a_vs_b") or {}).get("scenarios", [])
                ],
            }
            for v in verdicts
        ],
    }
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"[long_context_kv_ab] overall: {overall}", flush=True)
    return 0 if overall == PASS else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one arm of one config")
    run.add_argument("--arm", choices=("off", "on"), required=True)
    run.add_argument("--config", choices=tuple(CONFIGS), required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--label", default=None)
    run.add_argument("--graphs", action="store_true", help="compile + CUDA graphs")
    run.add_argument("--dtype", default="float16")
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    run.add_argument("--attention-backend", default=None)
    run.add_argument(
        "--async-scheduling", choices=("default", "on", "off"), default="default"
    )
    run.add_argument(
        "--kernel-config-extra",
        default="{}",
        help="JSON merged into --kernel-config for both arms",
    )
    run.add_argument("--host-pool-tokens", type=int, default=131072)
    run.add_argument("--hot-pool-gib", type=float, default=1.0)
    run.add_argument("--state-blocks", type=int, default=None)
    run.add_argument("--num-gpu-blocks-override", type=int, default=None)
    run.add_argument(
        "--geometry-from",
        default=None,
        help="run JSON of an 'on' arm; sizes the tight config's pools",
    )
    run.set_defaults(func=cmd_run)

    cmp = sub.add_parser("compare", help="compare two runs")
    cmp.add_argument("a")
    cmp.add_argument("b")
    cmp.add_argument("--control", default=None, help="repeat of arm A")
    cmp.add_argument("--label", required=True)
    cmp.add_argument("--out", required=True)
    cmp.set_defaults(func=cmd_compare)

    summ = sub.add_parser("summary", help="combine verdicts")
    summ.add_argument("verdicts", nargs="+")
    summ.add_argument("--out", required=True)
    summ.set_defaults(func=cmd_summary)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
