# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A/B end-to-end check of the long-context KV mode on one GPU.

The same token prompts and sampling settings run through an in-process engine
in one of four arms:

    off      ``kernel_config.long_context_kv`` off (one block pool)
    on       long-context mode, main K/V placement left at the engine default
    on-gpu   long-context mode, ``main_kv_placement="gpu"``: separate state and
             history block spaces, main K/V pages in the GPU history segment
    on-host  long-context mode, ``main_kv_placement="host"``: the GPU history
             segment keeps the index and ring pages; the main K/V lives in a
             pinned host store, read through a GPU hot-page pool in passes

Any two arms can be compared. Splitting the pools only renumbers blocks and
moves tensor bases, so off and on-gpu are expected to match bit for bit. The
host placement runs the same sparse attention kernel but drops the query
positions (no position-gated route) and, when a step has more query rows than
one gather pass holds, launches it per pass: the split-K plan of those rows can
change. ``--qsa-route host-like`` (test only) gives a GPU-placement arm the
same route and passes, and ``--hot-pool-rows``/``--single-pass`` size the hot
pool so that every step is one pass; either makes on-gpu and on-host
comparable bit for bit, and ``compare`` reports how far apart the default
routes are (token IDs, top-k sets and ranks, largest logprob difference).

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
inspect the scheduler's block pools, the worker's KV tensors and host history
store: per-segment tensor bytes against the capacity plan (by placement), the
purpose of every worker KV storage, the block-zeroing segments, pool usage
after every phase, preemptions and reused block IDs.

Example::

    python -m tools.long_context_kv_ab run --arm on-gpu --config main \\
        --model /path/to/model --out gpu.json
    python -m tools.long_context_kv_ab run --arm on-host --config main \\
        --model /path/to/model --out host.json
    python -m tools.long_context_kv_ab compare --label main gpu.json host.json \\
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
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Literal

import regex as re

GIB = 1 << 30
TOP_LOGPROBS = 5
# Token IDs drawn for synthetic prompts; below any special/added tokens.
TOKEN_LOW = 1000
TOKEN_HIGH = 100_000

PASS = "PASS"
FAIL = "FAIL"
INCONCLUSIVE = "INCONCLUSIVE"

# Arm -> main K/V placement requested from the engine ("default": leave
# ``main_kv_placement`` unset; None: long-context mode off).
ARM_PLACEMENT: dict[str, str | None] = {
    "off": None,
    "on": "default",
    "on-gpu": "gpu",
    "on-host": "host",
}
ARMS = tuple(ARM_PLACEMENT)
# Host store geometry (``host_kv.history``): newest-token groups every request
# keeps resident in the hot pool.
HOT_TAIL_GROUPS = 9


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
    # Determinism study: variants of "concurrent" that remove one factor.
    "conc-greedy": Scenario(
        "conc-greedy",
        (
            Phase(
                "x0",
                (
                    _one("x-700", "x-a", 700, 160),
                    _one("x-1900", "x-b", 1900, 96),
                    _one("x-3300", "x-c", 3300, 128),
                    _one("x-4300", "x-d", 4300, 64),
                ),
            ),
        ),
    ),
    "conc-samelen": Scenario(
        "conc-samelen",
        (
            Phase(
                "x0",
                (
                    _one("x-700", "x-a", 700, 96),
                    _one("x-1900", "x-b", 1900, 96),
                    _one("x-3300", "x-c", 3300, 96),
                    _one("x-4300", "x-d", 4300, 96),
                ),
            ),
        ),
    ),
    "conc-short4": Scenario(
        # Every prompt fits one step: one prefill step, then pure decode
        # whose row count steps down 4 -> 3 -> 2 -> 1.
        "conc-short4",
        (
            Phase(
                "y0",
                (
                    _one("y-150", "y-a", 150, 160),
                    _one("y-300", "y-b", 300, 96),
                    _one("y-450", "y-c", 450, 128),
                    _one("y-600", "y-d", 600, 64),
                ),
            ),
        ),
    ),
    "conc-short4-samelen": Scenario(
        "conc-short4-samelen",
        (
            Phase(
                "y0",
                (
                    _one("y-150", "y-a", 150, 96),
                    _one("y-300", "y-b", 300, 96),
                    _one("y-450", "y-c", 450, 96),
                    _one("y-600", "y-d", 600, 96),
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
    host_pool_tokens: int = 262_144
    """Default host history pool of the long-context arms. Host placement
    backs every history block ID with one page, so this caps the history
    IDs; 256Ki tokens hold every request of a config several times over."""
    hot_pool_gib: float = 1.0
    """Default GPU hot-page pool (host placement; admission only on GPU)."""


CONFIGS: dict[str, EngineConfig] = {
    "probe": EngineConfig(4096, 2, 1024, ("probe",)),
    "main": EngineConfig(8192, 4, 2048, ("short", "chunked", "concurrent", "prefix")),
    # The tight arms derive their host pool from the recorded geometry.
    "tight": EngineConfig(8192, 4, 2048, ("reuse", "preempt")),
    # Determinism study (same engine shape as main).
    "det": EngineConfig(
        8192,
        4,
        2048,
        (
            "concurrent",
            "conc-greedy",
            "conc-samelen",
            "conc-short4",
            "conc-short4-samelen",
            "short",
            "chunked",
        ),
    ),
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


def _local_block_tokens(geometry: dict[str, Any]) -> int:
    return int(geometry["block_tokens"]) // int(geometry.get("dcp_world_size") or 1)


def _capacity_geometry(geometry: dict[str, Any]):
    # Engine-only sizing is imported lazily so compare/summary remain usable
    # without importing vLLM or loading a model. Older receipts name layers
    # main_kv_layers, before the full device-memory accounting was added.
    from vllm.v1.core.long_context_kv import LongContextGeometry

    fields = dict(geometry)
    legacy_layers = fields.pop("main_kv_layers", None)
    fields.setdefault("hot_layers", legacy_layers or 1)
    fields.setdefault("hot_read_rows", None)  # Older receipts used slot-sized scratch.
    constructor_fields = {
        field.name for field in dataclasses.fields(LongContextGeometry)
    }
    return LongContextGeometry(
        **{name: value for name, value in fields.items() if name in constructor_fields}
    )


def hot_groups_for_bytes(
    geometry: dict[str, Any], num_bytes: int, max_num_seqs: int = 1
) -> int:
    """Hot-pool groups per layer that ``num_bytes`` per worker hold (the
    plan's ``LongContextGeometry.hot_groups_for_bytes``)."""
    return _capacity_geometry(geometry).hot_groups_for_bytes(num_bytes, max_num_seqs)


def hot_bytes_for_groups(
    geometry: dict[str, Any], groups: int, max_num_seqs: int = 1
) -> int:
    return _capacity_geometry(geometry).hot_bytes_for_groups(groups, max_num_seqs)


def rows_per_pass(geometry: dict[str, Any], hot_groups: int, max_num_seqs: int) -> int:
    """Query rows of one host gather pass (``HostHistoryShape.rows_per_pass``)."""
    bound = (hot_groups - max_num_seqs * HOT_TAIL_GROUPS) // int(
        geometry["hot_row_groups"]
    )
    cap = geometry.get("hot_read_rows")
    if cap is not None:
        length = int(geometry["max_model_len"])
        dcp = int(geometry.get("dcp_world_size", 1))
        il = int(geometry.get("dcp_interleave", 1))
        local = length // (dcp * il) * il + min(length % (dcp * il), il)
        groups = (local + 3) // 4
        if groups <= int(geometry["hot_row_groups"]) + HOT_TAIL_GROUPS:
            tail = min(groups, HOT_TAIL_GROUPS)
            if hot_groups >= groups * max_num_seqs:
                return int(cap)
            bound = (hot_groups - max_num_seqs * tail) // max(groups - tail, 1)
    return bound if cap is None else min(int(cap), bound)


def host_metadata_bytes_per_block(geometry: dict[str, Any]) -> int:
    """GPU bytes the host store keeps per history block ID (hot-group map of
    every layer, shared wanted map, readiness bytes, generation)."""
    reported = geometry.get("host_metadata_bytes_per_block")
    if reported is not None:
        return int(reported)
    current = _capacity_geometry(geometry).host_metadata_bytes_per_block
    # page ownership receipts carry the logical host index field. Earlier raw geometries
    # predate the device page_of table and retain their original byte bill.
    return current if "logical_index_bytes_per_block" in geometry else current - 4


def hot_pool_gib_for_rows(
    geometry: dict[str, Any], rows: int, max_num_seqs: int
) -> float:
    """Smallest ``hot_pool_gib`` (1/1024 GiB steps) whose gather pass holds
    ``rows`` query rows: with ``rows >= max_num_batched_tokens`` every step
    runs in one pass, i.e. with the GPU placement's split-K plan."""
    cap = geometry.get("hot_read_rows")
    if cap is not None and rows > cap:
        raise ValueError(f"Requested {rows} rows exceed host-history R_cap={cap}")
    need = rows * int(geometry["hot_row_groups"]) + max_num_seqs * HOT_TAIL_GROUPS
    gib = (
        math.ceil(hot_bytes_for_groups(geometry, need, max_num_seqs) * 1024 / GIB)
        / 1024
    )
    while True:
        groups = hot_groups_for_bytes(geometry, int(gib * GIB), max_num_seqs)
        if rows_per_pass(geometry, groups, max_num_seqs) >= rows:
            return gib
        gib += 1 / 1024


def tight_settings(
    geometry: dict[str, Any],
    max_model_len: int,
    placement: str,
    hot_pool_gib: float | Literal["auto"] = 1.0,
) -> dict[str, int]:
    """Pool sizes that make the tight config reuse blocks and preempt.

    The long-context arms keep one max_model_len request's worth of state
    blocks plus half again, and history IDs for one such request plus three
    blocks; ``num_gpu_blocks_override`` fixes the budget the plan splits.
    With manual sizing both placements get the same state and history ID
    counts. The GPU arm's count follows from the rounded-up budget. The host
    arm's host pool backs exactly that many IDs (plus the null page). Its
    budget also carries the hot pool and the per-block store metadata. The reference arm
    gets a unified pool of about the same per-request slack. Every value
    still passes the startup checks.
    """
    block_tokens = int(geometry["block_tokens"])
    s_req = int(geometry["state_blocks_per_request"])
    ring = int(geometry["ring_blocks_per_request"])
    history = math.ceil(max_model_len / block_tokens)
    unified = int(geometry["unified_bytes_per_block"])
    if placement == "off":
        return {"num_gpu_blocks_override": s_req + history + ring + max(3, s_req // 2)}
    if placement not in ("gpu", "host"):
        raise ValueError(f"unknown placement {placement!r}")
    index = int(geometry["index_bytes_per_block"])
    state_blocks = s_req + max(1, s_req // 2)
    target = history + ring + 3
    state_bytes = (state_blocks + 1) * int(geometry["state_bytes_per_block"])
    gpu_block = index + int(geometry["main_kv_bytes_per_block"])
    if hot_pool_gib == "auto":
        # Leave room for four full verify working sets. State remains tight
        # enough to exercise preemption; host history is demand-sized by auto.
        target = 4 * (history + ring)
        groups = 4 * (
            int(geometry["decode_rows_per_request"]) * int(geometry["hot_row_groups"])
            + 9
        )
        hot = hot_bytes_for_groups(geometry, groups, 4)
        block = (
            gpu_block
            if placement == "gpu"
            else index + host_metadata_bytes_per_block(geometry)
        )
        override = -(-(state_bytes + hot + (target + 1) * block) // unified)
        return {
            "state_blocks": state_blocks,
            "num_gpu_blocks_override": override,
            "target_history_blocks": target,
            "history_blocks": (
                (override * unified - state_bytes) // block - 1
                if placement == "gpu"
                else target
            ),
            "host_pool_tokens": (target + 1) * block_tokens,
        }
    gpu_override = -(-(state_bytes + (target + 1) * gpu_block) // unified)
    history_blocks = (gpu_override * unified - state_bytes) // gpu_block - 1
    override = gpu_override
    if placement == "host":
        hot = hot_bytes_for_groups(
            geometry, hot_groups_for_bytes(geometry, int(hot_pool_gib * GIB), 4), 4
        )
        host_block = index + host_metadata_bytes_per_block(geometry)
        override = -(
            -(state_bytes + hot + (history_blocks + 1) * host_block) // unified
        )
    return {
        "state_blocks": state_blocks,
        "num_gpu_blocks_override": override,
        "target_history_blocks": target,
        "history_blocks": history_blocks,
        "host_pool_tokens": (history_blocks + 1) * block_tokens,
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


def _logprob_map(rows: list[list[Any]] | None) -> dict[int, tuple[str, Any]]:
    return {int(row[0]): (row[1], row[2]) for row in rows or []}


def difference_stats(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """How far apart two runs are, to attribute a difference.

    Counts requests that differ, whether every generated token and cache hit
    matches, whether every compared position has the same top-k tokens with
    the same ranks, how many logprob values differ and the largest absolute
    difference, plus the first differing positions. After a token diverges
    the later positions see different contexts; ``token_ids_identical``
    tells whether that happened."""
    stats: dict[str, Any] = {
        "requests": 0,
        "requests_different": 0,
        "token_ids_identical": True,
        "cached_tokens_identical": True,
        "topk_tokens_and_ranks_identical": True,
        "values_compared": 0,
        "values_different": 0,
        "max_abs_logprob_diff": 0.0,
        "positions": [],
    }
    reqs_b = {
        r["name"]: r
        for s in b.get("scenarios", [])
        for p in s["phases"]
        for r in p["requests"]
    }
    for scenario in a.get("scenarios", []):
        for phase in scenario["phases"]:
            for ra in phase["requests"]:
                rb = reqs_b.get(ra["name"])
                if rb is None:
                    continue
                stats["requests"] += 1
                differs = False
                if ra["token_ids"] != rb["token_ids"]:
                    stats["token_ids_identical"] = False
                    differs = True
                if ra["num_cached_tokens"] != rb["num_cached_tokens"]:
                    stats["cached_tokens_identical"] = False
                    differs = True
                for field in ("logprobs", "prompt_logprobs"):
                    la, lb = ra[field] or [], rb[field] or []
                    if len(la) != len(lb):
                        differs = True
                    for pos, (x, y) in enumerate(zip(la, lb)):
                        if x == y:
                            stats["values_compared"] += len(x or [])
                            continue
                        differs = True
                        mx, my = _logprob_map(x), _logprob_map(y)
                        if set(mx) != set(my) or any(
                            mx[t][1] != my[t][1] for t in mx.keys() & my.keys()
                        ):
                            stats["topk_tokens_and_ranks_identical"] = False
                        for token in mx.keys() & my.keys():
                            stats["values_compared"] += 1
                            if mx[token][0] != my[token][0]:
                                stats["values_different"] += 1
                                delta = abs(
                                    float.fromhex(mx[token][0])
                                    - float.fromhex(my[token][0])
                                )
                                stats["max_abs_logprob_diff"] = max(
                                    stats["max_abs_logprob_diff"], delta
                                )
                        if len(stats["positions"]) < 20:
                            stats["positions"].append(
                                {
                                    "scenario": scenario["name"],
                                    "request": ra["name"],
                                    "field": field,
                                    "position": pos,
                                }
                            )
                if differs:
                    stats["requests_different"] += 1
    return stats


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
        "difference_stats": difference_stats(a, b),
        "hard_identical": all(s["identical"] for s in hard) and bool(hard),
        "soft_differences": [
            s["name"] for s in scenarios if s["parity"] == "soft" and not s["identical"]
        ],
    }


def failed_checks(run: dict[str, Any]) -> list[str]:
    if run.get("status") != "ok":
        reason = run.get("skip_reason", run.get("error", ""))
        return [f"run status {run.get('status')}: {reason[:300]}"]
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
    does not reproduce itself or an arm was skipped for lack of GPU budget.
    FAIL otherwise."""
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
        # Diagnostic only: when A differs from both B and its own repeat, the
        # first run of A (cold caches) is the odd one out.
        result["b_vs_control"] = compare_runs(b, control) if not reasons else None
    if reasons:
        runs = [a, b] + ([control] if control is not None else [])
        skipped = [run for run in runs if run.get("status") == "skipped"]
        only_skips = skipped and all(
            not failed_checks(run) for run in runs if run.get("status") != "skipped"
        )
        result.update(verdict=INCONCLUSIVE if only_skips else FAIL, reasons=reasons)
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
    "host_store": re.compile(r"Long-context (?:main K/V in host memory|host memory)"),
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


def structure_placement(structure: dict[str, Any], arm: str) -> str:
    """``off``, ``gpu`` or ``host`` from the recorded plan (``unknown``
    without one): the arm ``on`` follows the engine default."""
    if ARM_PLACEMENT.get(arm) is None:
        return "off"
    plan = structure.get("plan")
    if plan is None:
        return "unknown"
    return "gpu" if plan.get("main_kv_on_gpu", True) else "host"


def explain_storages(structure: dict[str, Any]) -> dict[str, Any]:
    """The purpose of every worker KV storage, and config tensors no storage
    backs. A storage is a configured KV tensor (same layers, same bytes) or,
    with the main K/V in the host store, the empty placeholder those layers
    keep (no GPU bytes); anything else is unexplained."""
    tensors = structure["kv_cache_tensors"]
    host_layers = set(structure.get("host_kv_layers") or [])
    used: set[int] = set()
    explained = []
    for storage in structure.get("worker_storages", []):
        layers = set(storage["layers"])
        entry: dict[str, Any] = {
            "nbytes": storage["nbytes"],
            "layers": storage["layers"][:4],
            "num_layers": len(layers),
            "segments": storage["segments"],
            "tensor": None,
        }
        if storage["nbytes"] == 0 and layers and layers <= host_layers:
            entry["purpose"] = (
                "host-placed main K/V: empty GPU placeholder, pages in the host store"
            )
            explained.append(entry)
            continue
        match = None
        for index, tensor in enumerate(tensors):
            if index in used or tensor["size"] != storage["nbytes"]:
                continue
            members = set(tensor.get("layers") or tensor["shared_by"])
            if members == layers or ("layers" not in tensor and members <= layers):
                match = index
                break
        if match is None:
            entry["purpose"] = "unexplained"
        else:
            used.add(match)
            tensor = tensors[match]
            entry["tensor"] = match
            entry["purpose"] = (
                f"configured KV tensor of the {tensor.get('segment')} block space "
                f"({tensor.get('members', len(layers))} layers)"
            )
        explained.append(entry)
    return {
        "storages": explained,
        "unexplained": [e for e in explained if e["purpose"] == "unexplained"],
        "tensors_without_storage": [i for i in range(len(tensors)) if i not in used],
    }


def _host_store_checks(
    structure: dict[str, Any], plan: dict[str, Any], history_tensor_bytes: int
) -> list[dict[str, Any]]:
    """Host placement: the GPU history segment holds the index and ring pages
    plus the store's per-block metadata; the store's pages, metadata and hot
    pool match the plan."""
    geometry = plan["geometry"]
    _, history_n = plan["segment_num_blocks"]
    store = structure.get("host_store")
    host_layers = sorted(structure.get("host_kv_layers") or [])
    checks = [
        check(
            "history segment tensors hold the index and ring pages only",
            history_tensor_bytes == history_n * int(geometry["index_bytes_per_block"]),
            {
                "tensors": history_tensor_bytes,
                "index_and_ring": history_n * int(geometry["index_bytes_per_block"]),
                "blocks": history_n,
            },
        )
    ]
    if store is None:
        checks.append(check("host history store present", False, "no store"))
        return checks
    meta_plan = history_n * host_metadata_bytes_per_block(geometry)
    checks.append(
        check(
            "history segment bytes equal the plan (tensors + store metadata)",
            history_tensor_bytes
            + store["metadata_bytes"]
            + store.get("per_slot_gpu_bytes", 0)
            == plan["gpu_history_bytes"],
            {
                "tensors": history_tensor_bytes,
                "store_metadata": store["metadata_bytes"],
                "per_slot": store.get("per_slot_gpu_bytes", 0),
                "plan": plan["gpu_history_bytes"],
            },
        )
    )
    checks.append(
        check(
            "host store metadata bytes match the plan",
            store["metadata_bytes"] == meta_plan,
            {"store": store["metadata_bytes"], "plan": meta_plan},
        )
    )
    checks.append(
        check(
            "host store is bound to every host-placed layer",
            bool(host_layers)
            and sorted(store["layers"]) == host_layers
            and sorted(store["bound_layers"]) == host_layers,
            {
                "host_kv_layers": host_layers,
                "store": store["layers"],
                "bound": store["bound_layers"],
            },
        )
    )
    workers = int(geometry.get("num_workers") or 1)
    physical_pages = store.get("physical_host_pages", store["num_blocks"])
    shared_host_bytes = store.get("shared_host_bytes", store["host_bytes"] * workers)
    # The shared hot region shared region decouples history IDs from host pages.
    aliased = bool(store.get("hot_aliases_index"))
    pages_ok = (
        store["num_blocks"] == history_n
        and (aliased or store["num_blocks"] <= plan["host_history_blocks"])
        and store["page_bytes"] == int(geometry["main_kv_bytes_per_block"])
        and physical_pages <= plan["host_history_blocks"]
        and store["host_bytes"] == physical_pages * store["page_bytes"]
        and shared_host_bytes <= plan["host_pool_bytes"]
    )
    checks.append(
        check(
            "host store pages match the plan",
            pages_ok,
            {
                "pages": store["num_blocks"],
                "history_blocks": history_n,
                "plan_host_blocks": plan["host_history_blocks"],
                "page_bytes": store["page_bytes"],
                "plan_page_bytes": int(geometry["main_kv_bytes_per_block"]),
                "host_bytes": store["host_bytes"],
                "plan_host_pool_bytes": plan["host_pool_bytes"],
            },
        )
    )
    expected_rows = rows_per_pass(
        geometry, int(plan["hot_pool_groups"]), int(plan["max_num_seqs"])
    )
    hot_plan = plan.get("gpu_hot_pool_bytes", plan["hot_pool_bytes"])
    # Shared region: the null ID page aliases k reserved slots the plan's
    # U = (A - 1) k leaves out.
    null_slots = int(store.get("slots_per_block", 0)) if aliased else 0
    checks.append(
        check(
            "hot pool matches the plan",
            store["hot_slots"] == plan["hot_pool_groups"] + null_slots
            and store["hot_bytes"] + store.get("scratch_gpu_bytes", 0) == hot_plan
            and store["rows_per_pass"] == expected_rows,
            {
                "hot_slots": store["hot_slots"],
                "plan_groups": plan["hot_pool_groups"],
                "hot_bytes": store["hot_bytes"],
                "plan_bytes": hot_plan,
                "rows_per_pass": store["rows_per_pass"],
                "expected_rows_per_pass": expected_rows,
            },
        )
    )
    return checks


def segment_tensor_checks(structure: dict[str, Any], arm: str) -> list[dict[str, Any]]:
    """Tensor bytes and block counts against the plan, by main K/V placement;
    the purpose of every worker storage."""
    checks = []
    plan = structure.get("plan")
    tensors = structure["kv_cache_tensors"]
    storages = structure.get("worker_storages", [])
    total = sum(t["size"] for t in tensors)
    worker_total = sum(s["nbytes"] for s in storages)
    explained = explain_storages(structure)
    checks.append(
        check(
            "every worker KV storage is explained and backs one KV tensor",
            not explained["unexplained"] and not explained["tensors_without_storage"],
            {
                **explained,
                "layers_missing": structure.get(
                    "layers_missing_from_forward_context", []
                )[:5],
            },
        )
    )
    checks.append(
        check(
            "worker allocated the configured KV tensor bytes",
            worker_total == total,
            {"config_bytes": total, "worker_bytes": worker_total},
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
    placement = structure_placement(structure, arm)
    if placement == "off":
        checks.append(check("reference arm has no block segments", plan is None, ""))
        return checks
    if plan is None:
        checks.append(check("long-context plan present", False, "no plan"))
        return checks
    wanted = ARM_PLACEMENT.get(arm)
    checks.append(
        check(
            "main K/V placement is the requested one",
            wanted == "default" or wanted == placement,
            {"arm": arm, "plan": placement},
        )
    )
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
    used = total
    if placement == "host":
        checks += _host_store_checks(structure, plan, by_segment["history"])
        store = structure.get("host_store") or {}
        used += (
            store.get("metadata_bytes", 0)
            + store.get("per_slot_gpu_bytes", 0)
            + store.get("hot_bytes", 0)
            + store.get("scratch_gpu_bytes", 0)
            + int(plan.get("staging_bytes", 0))
        )
    else:
        checks.append(
            check(
                "history segment bytes equal the plan",
                by_segment["history"] == plan["gpu_history_bytes"],
                {"tensors": by_segment["history"], "plan": plan["gpu_history_bytes"]},
            )
        )
        checks.append(
            check(
                "no host history store with the main K/V on GPU",
                not structure.get("host_kv_layers") and not structure.get("host_store"),
                structure.get("host_kv_layers"),
            )
        )
    checks.append(
        check(
            "GPU KV memory fits the planned budget",
            used <= plan["gpu_budget_bytes"],
            {
                "used": used,
                "kv_tensors": total,
                "budget": plan["gpu_budget_bytes"],
                "unused": plan["gpu_budget_bytes"] - used,
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
    if ARM_PLACEMENT.get(arm) is not None:
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
                    count > 0 if ARM_PLACEMENT.get(arm) is not None else None,
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
    for name in ("gpu_hot_pool_bytes", "host_page_blocks", "gpu_history_budget_blocks"):
        if hasattr(plan, name):
            data[name] = getattr(plan, name)
    geometry = plan.geometry
    for name in ("host_metadata_bytes_per_block", "host_bytes_per_block"):
        if hasattr(geometry, name):
            data["geometry"][name] = getattr(geometry, name)
    return data


# Host store tensors the capacity plan charges per history block.
_HOST_STORE_PLANNED = ("group_to_slot", "wanted", "ready", "generation", "page_of")


def _host_store_dict(
    store: Any, host_layers: list[str], context: dict[str, Any]
) -> dict[str, Any] | None:
    """Pages, metadata and hot pool the worker's host history store holds."""
    if store is None:
        return None
    import torch

    def nbytes(tensor: torch.Tensor) -> int:
        return tensor.numel() * tensor.element_size()

    shape = store.shape
    # shared hot region shared region: one hot view per layer aliases that layer's index
    # storage, which the history segment already counts; no separate bytes.
    aliased = isinstance(store.hot, (list, tuple))
    hot_bytes = 0 if aliased else nbytes(store.hot)
    planned = sum(
        nbytes(getattr(store, name))
        for name in _HOST_STORE_PLANNED
        if hasattr(store, name)
    )
    skip = {"host", "host_device", "hot", *_HOST_STORE_PLANNED}
    other = {
        name: nbytes(value)
        for name, value in vars(store).items()
        if isinstance(value, torch.Tensor)
        and name not in skip
        and value.device.type != "cpu"
    }
    per_slot: dict[str, int] = {}
    if aliased:
        from vllm.models.qwen4_exp.nvidia.host_kv.budget import SHARED_PER_SLOT

        # Billed per history ID in the history segment, not as hot scratch.
        per_slot = {n: other.pop(n) for n in sorted(SHARED_PER_SLOT) if n in other}
    bound = [
        name
        for name in host_layers
        if getattr(getattr(context.get(name), "host_history", None), "store", None)
        is store
    ]
    try:
        pinned: bool | str = bool(store.host.is_pinned())
    except Exception as exc:  # pragma: no cover - backend differences
        pinned = f"unavailable: {exc}"
    return {
        "num_blocks": shape.num_blocks,
        "kernel_blocks_per_block": shape.kernel_blocks_per_block,
        "kernel_block_tokens": shape.kernel_block_tokens,
        "num_layers": shape.num_layers,
        "kv_heads": shape.kv_heads,
        "head_dim": shape.head_dim,
        "dtype": str(shape.dtype),
        "page_bytes": shape.page_bytes,
        "host_bytes": nbytes(store.host),
        "physical_host_pages": store.host.shape[0],
        **(
            {"shared_host_bytes": store.region.total_size_bytes}
            if getattr(store, "region", None) is not None
            else {}
        ),
        "host_pinned": pinned,
        "metadata_bytes": planned,
        "hot_slots": shape.hot_slots,
        "hot_bytes": hot_bytes,
        "hot_aliases_index": aliased,
        "slots_per_block": getattr(shape, "slots_per_block", 0),
        "rows_per_pass": shape.rows_per_pass,
        "max_requests": shape.max_requests,
        "selection_width": shape.selection_width,
        "max_write_rows": shape.max_write_rows,
        "scratch_gpu_bytes": sum(other.values()),
        "scratch_gpu_tensors": other,
        "per_slot_gpu_bytes": sum(per_slot.values()),
        "per_slot_gpu_tensors": per_slot,
        "layers": list(store.layer_index),
        "bound_layers": bound,
    }


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
                "layers": list(t.shared_by),
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
    host_layers = list(getattr(config, "host_kv_layers", None) or [])
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
        "host_kv_layers": host_layers,
        "host_store": _host_store_dict(
            getattr(runner, "host_history", None), host_layers, context
        ),
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
            # The elastic pool's free count reserves the hot floor (GPU
            # admission); an idle pool is fully free physically.
            "free": p.num_physically_free_blocks()
            if getattr(p, "elastic", False)
            else p.get_num_free_blocks(),
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
        self.steps: list[list[list[Any]]] = []
        self.budgets = []
        self._wrap_schedule()
        self.reset_phase()

    def reset_phase(self) -> None:
        self.allocated = [[] for _ in _pools(self.sched)]
        self.preemptions = []
        self.steps = []
        self.budgets: list[int] = []

    def _wrap_schedule(self) -> None:
        sched = self.sched
        original = sched.schedule

        def schedule(*args: Any, **kwargs: Any) -> Any:
            out = original(*args, **kwargs)
            counts = getattr(out, "num_scheduled_tokens", None) or {}
            if counts:
                self.steps.append([[str(r), int(n)] for r, n in counts.items()])
                self.budgets.append(int(getattr(out, "mixed_prefill_budget", -1)))
            return out

        sched.schedule = schedule

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


def _named_steps(
    steps: list[list[list[Any]]], records: list[dict[str, Any]]
) -> list[list[list[Any]]]:
    """Scheduled tokens per engine step, request IDs mapped to names."""

    def name_of(rid: str) -> str:
        for rec in records:
            ext = str(rec["request_id"])
            if rid == ext or rid.startswith(ext + "-"):
                return str(rec["name"])
        return rid

    return [[[name_of(rid), n] for rid, n in step] for step in steps]


def run_scenarios(
    llm: Any,
    sched: Any,
    recorder: Recorder,
    names: Sequence[str],
    seed: int,
    repeat: int = 1,
) -> list[dict[str, Any]]:
    results = []
    for rep_index, name in enumerate([n for _ in range(repeat) for n in names]):
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
                    "schedule": _named_steps(recorder.steps, records),
                    "mixed_prefill_budget": list(recorder.budgets),
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
                "repeat": rep_index // max(len(names), 1),
                "seconds": round(time.time() - t0, 3),
                "phases": phases,
            }
        )
    return results


class QSARouteProbe:
    """Test-only hooks on the QSA sparse attention; nothing in vLLM changes.

    Every arm counts the attention calls and, with the main K/V in the host
    store, the gather passes of each step (``host_kv.history.for_each_pass``).

    ``host_like`` (GPU placement only) runs the GPU-placement attention the
    way the host placement calls it: without query positions and sequence
    lengths, which only gate the position-dependent SM70 XQA page4 route (the
    split-K Triton kernel never reads them, and the host path never passes
    them), and, with ``pass_rows``, in contiguous passes of that many rows
    like ``for_each_pass``. A pass changes the split-K plan of its rows
    (``_qsa_sparse_launch_profile`` picks the splits from the row count), so
    the two placements then share the route and the plan. Calls with an LSE
    output (DCP partials) or an INT8 cache pass through and are counted.
    Counts are of Python calls: CUDA graph replays do not call Python."""

    def __init__(self, host_like: bool = False, pass_rows: int | None = None) -> None:
        if pass_rows is not None and pass_rows < 1:
            raise ValueError("pass rows must be positive")
        self.host_like = host_like
        self.pass_rows = pass_rows
        self.stats: dict[str, int] = {
            "attention_calls": 0,
            "attention_max_rows": 0,
            "host_like_calls": 0,
            "split_calls": 0,
            "row_passes": 0,
            "passthrough_calls": 0,
            "host_steps": 0,
            "host_multi_pass_steps": 0,
            "host_max_passes": 0,
            "host_max_rows": 0,
            "host_rows_per_pass": 0,
        }

    def attention(self, original: Any) -> Any:
        """``original`` (``qsa_sparse_paged_attention``) wrapped."""
        import inspect

        signature = inspect.signature(original)
        stats = self.stats

        def attention(*args: Any, **kwargs: Any) -> Any:
            arguments = signature.bind(*args, **kwargs).arguments
            q = arguments["q"]
            rows = int(q.shape[0])
            stats["attention_calls"] += 1
            stats["attention_max_rows"] = max(stats["attention_max_rows"], rows)
            if not self.host_like:
                return original(*args, **kwargs)
            if (
                arguments.get("lse") is not None
                or arguments.get("kv_cache_dtype") == "int8"
            ):
                stats["passthrough_calls"] += 1
                return original(*args, **kwargs)
            stats["host_like_calls"] += 1
            arguments.pop("query_positions", None)
            arguments.pop("sequence_lengths", None)
            step = self.pass_rows or rows
            if rows <= step:
                return original(**arguments)
            out = arguments.get("out")
            if out is None:
                out = arguments["out"] = q.new_empty(q.shape)
            gate = arguments.get("output_gate")
            stats["split_calls"] += 1
            for start in range(0, rows, step):
                end = min(rows, start + step)
                part = dict(
                    arguments,
                    q=q[start:end],
                    logical_indices=arguments["logical_indices"][start:end],
                    token_to_req=arguments["token_to_req"][start:end],
                    out=out[start:end],
                )
                if gate is not None:
                    part["output_gate"] = gate[start:end]
                original(**part)
                stats["row_passes"] += 1
            return out

        return attention

    def passes(self, original: Any) -> Any:
        """``original`` (``for_each_pass``) wrapped to count passes."""
        stats = self.stats

        def for_each_pass(layer: Any, selection: Any, *args: Any, **kwargs: Any) -> Any:
            rows = int(selection.shape[0])
            step = int(layer.rows_per_pass)
            passes = -(-rows // step) if rows else 0
            # New planners and fixed single-pass decode can differ from the
            # conservative diagnostic rows_per_pass. Count actual callbacks.
            attend = args[3] if len(args) >= 4 else kwargs.get("attend")
            if callable(attend):
                passes, step = 0, 0

                def counted(start: int, end: int, hot_ids: Any) -> Any:
                    nonlocal passes, step
                    passes += 1
                    step = max(step, end - start)
                    return attend(start, end, hot_ids)

                if len(args) >= 4:
                    args = (*args[:3], counted, *args[4:])
                else:
                    kwargs["attend"] = counted
            result = original(layer, selection, *args, **kwargs)
            stats["host_steps"] += 1
            stats["host_multi_pass_steps"] += int(passes > 1)
            stats["host_max_passes"] = max(stats["host_max_passes"], passes)
            stats["host_max_rows"] = max(stats["host_max_rows"], rows)
            stats["host_rows_per_pass"] = step
            return result

        return for_each_pass

    def install(self) -> None:
        # Both callers import these names inside the function at call time,
        # so replacing the module attributes reaches them.
        from vllm.models.qwen4_exp.nvidia.host_kv import history
        from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

        qsa_ops.qsa_sparse_paged_attention = self.attention(
            qsa_ops.qsa_sparse_paged_attention
        )
        history.for_each_pass = self.passes(history.for_each_pass)


def resolve_placement(arm: str) -> str:
    """``off``, ``gpu`` or ``host``; ``on`` resolves the engine default."""
    placement = ARM_PLACEMENT[arm]
    if placement is None:
        return "off"
    if placement == "default":
        from vllm.config.kernel import LongContextKVConfig

        return str(LongContextKVConfig().main_kv_placement)
    return placement


def _load_geometry(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    with open(path) as f:
        return json.load(f)["structure"]["plan"]["geometry"]


def _rows_from(path: str) -> int:
    """Gather-pass rows of a host-placement run JSON."""
    with open(path) as f:
        run = json.load(f)
    store = (run.get("structure") or {}).get("host_store")
    if not store:
        raise ValueError(f"{path} has no host history store (not an on-host run)")
    return int(store["rows_per_pass"])


def resolve_long_context(
    args: argparse.Namespace,
    config: EngineConfig,
    placement: str,
    geometry: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """``long_context_kv`` settings of a long-context arm, and the derived
    pool sizes recorded in the JSON (tight config, hot pool for rows)."""
    derived: dict[str, Any] = {}
    if geometry is not None:
        # A geometry receipt can come from another scenario. Staging must
        # match this run's scheduler, not the recorded run's write rows.
        geometry = dict(
            geometry,
            max_model_len=config.max_model_len,
            hot_max_write_rows=config.max_num_batched_tokens,
        )
    if args.hot_pool_gib is not None:
        hot_gib = args.hot_pool_gib
    elif args.hot_pool_rows is not None or args.single_pass:
        if geometry is None:
            raise ValueError(
                "--hot-pool-rows/--single-pass size the hot pool from a "
                "recorded geometry: pass --geometry-from or --hot-pool-gib"
            )
        rows = args.hot_pool_rows or config.max_num_batched_tokens
        hot_gib = hot_pool_gib_for_rows(geometry, rows, config.max_num_seqs)
        derived["hot_pool_rows"] = rows
        derived["hot_pool_gib"] = hot_gib
    else:
        hot_gib = config.hot_pool_gib
    settings: dict[str, Any] = {"enabled": True, "hot_pool_gib": hot_gib}
    if ARM_PLACEMENT[args.arm] in ("gpu", "host"):
        settings["main_kv_placement"] = ARM_PLACEMENT[args.arm]
    host_tokens: int | None = config.host_pool_tokens
    if args.config == "tight" and geometry is not None:
        derived.update(
            tight_settings(geometry, config.max_model_len, placement, hot_gib)
        )
        host_tokens = derived["host_pool_tokens"]
        if args.state_blocks is None:
            args.state_blocks = derived["state_blocks"]
        if args.num_gpu_blocks_override is None:
            args.num_gpu_blocks_override = derived["num_gpu_blocks_override"]
    if args.host_pool_gib is not None:
        settings["host_pool_gib"] = args.host_pool_gib
    else:
        settings["host_pool_tokens"] = args.host_pool_tokens or host_tokens
    if args.state_blocks is not None:
        settings["state_blocks"] = args.state_blocks
    return settings, derived


def resolve_engine_args(
    args: argparse.Namespace, config: EngineConfig
) -> tuple[dict[str, Any], dict[str, Any]]:
    """LLM keyword arguments and the derived settings recorded in the JSON."""
    placement = resolve_placement(args.arm)
    if args.qsa_route == "host-like" and placement != "gpu":
        raise ValueError("--qsa-route host-like applies to the GPU placement only")
    if args.qsa_route != "host-like" and (
        args.qsa_pass_rows or args.qsa_pass_rows_from
    ):
        raise ValueError("--qsa-pass-rows needs --qsa-route host-like")
    if args.host_pool_gib is not None and args.host_pool_tokens is not None:
        raise ValueError("pass one of --host-pool-gib and --host-pool-tokens")
    if args.hot_pool_gib is not None and args.hot_pool_rows is not None:
        raise ValueError("pass one of --hot-pool-gib and --hot-pool-rows")
    geometry = _load_geometry(args.geometry_from)
    derived: dict[str, Any] = {"placement": placement}
    kernel_config: dict[str, Any] = json.loads(args.kernel_config_extra)
    if placement == "off":
        if args.config == "tight" and geometry is not None:
            derived.update(tight_settings(geometry, config.max_model_len, "off"))
            if args.num_gpu_blocks_override is None:
                args.num_gpu_blocks_override = derived["num_gpu_blocks_override"]
    else:
        settings, extra = resolve_long_context(args, config, placement, geometry)
        derived.update(extra)
        kernel_config["long_context_kv"] = settings
    kwargs: dict[str, Any] = dict(
        model=args.model,
        dtype=args.dtype,
        tensor_parallel_size=1,
        seed=args.seed,
        max_model_len=config.max_model_len,
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        enable_prefix_caching=not args.no_prefix_caching,
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
    if args.mixed_prefill_latency_ms is not None:
        kwargs["mixed_prefill_step_latency_ms"] = args.mixed_prefill_latency_ms
    if args.prefill_pressure_budget is not None:
        kwargs["prefill_pressure_token_budget"] = args.prefill_pressure_budget
    if placement == "host":
        # Host main K/V moves state/draft prefixes through the built-in
        # connector; the engine refuses host placement without it.
        kwargs["kv_transfer_config"] = {
            "kv_connector": "OffloadingConnector",
            "kv_role": "kv_both",
            "kv_connector_extra_config": {
                "cpu_bytes_to_use": int(args.offload_cpu_mib) << 20
            },
        }
    return kwargs, derived


def route_checks(
    stats: dict[str, int],
    placement: str,
    args: argparse.Namespace,
    config: EngineConfig,
    structure: dict[str, Any],
) -> list[dict[str, Any]]:
    """The requested attention route and pass structure were in effect."""
    checks = []
    if args.qsa_route == "host-like":
        checks.append(
            check(
                "host-like QSA route used for every attention call",
                stats["host_like_calls"] > 0 and stats["passthrough_calls"] == 0,
                stats,
            )
        )
    if placement == "host":
        checks.append(check("host gather passes (recorded)", None, stats))
    if args.single_pass:
        rows = config.max_num_batched_tokens
        plan = structure.get("plan") or {}
        if placement == "host":
            store = structure.get("host_store") or {}
            ok = (
                store.get("rows_per_pass", 0) >= rows
                and stats["host_steps"] > 0
                and stats["host_multi_pass_steps"] == 0
            )
            detail: Any = {"rows_per_pass": store.get("rows_per_pass"), **stats}
        else:
            geometry = plan.get("geometry")
            got = (
                rows_per_pass(geometry, plan["hot_pool_groups"], plan["max_num_seqs"])
                if geometry
                else 0
            )
            ok = got >= rows
            detail = {"plan_rows_per_pass": got}
        checks.append(
            check(
                f"every step fits one gather pass ({rows} rows)",
                ok,
                detail,
            )
        )
    return checks


class SinglePassUnavailable(ValueError):
    """The profiled GPU KV budget cannot hold the requested gather pass."""


@contextmanager
def single_pass_budget_guard(enabled: bool):
    """Inspect the live plan before KV allocation; keep budget skips distinct
    from engine failures. The A/B engine runs in this process."""
    if not enabled:
        yield
        return
    from vllm.v1.core import kv_cache_utils

    validate = kv_cache_utils.validate_long_context_plan

    def guarded(plan):
        if not plan.main_kv_on_gpu:
            geo = plan.geometry
            history = (
                geo.history_blocks(geo.max_model_len) + geo.ring_blocks_per_request + 1
            )
            need = (
                plan.gpu_state_bytes
                + plan.gpu_hot_pool_bytes
                + plan.staging_bytes
                + history * plan.history_bytes_per_block
            )
            if need > plan.gpu_budget_bytes:
                raise SinglePassUnavailable(
                    f"无法单趟：requested hot pool and minimum state/history need "
                    f"{need} GPU bytes, effective KV budget is {plan.gpu_budget_bytes} "
                    f"(hot pool {plan.gpu_hot_pool_bytes} bytes); skipping"
                )
        validate(plan)

    kv_cache_utils.validate_long_context_plan = guarded
    try:
        yield
    finally:
        kv_cache_utils.validate_long_context_plan = validate


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
        placement = derived["placement"]
        result["placement"] = placement
        result["engine_kwargs"] = {k: v for k, v in kwargs.items() if k != "model"}
        result["model"] = args.model
        result["derived_settings"] = derived
        pass_rows = args.qsa_pass_rows
        if args.qsa_pass_rows_from:
            pass_rows = _rows_from(args.qsa_pass_rows_from)
        route = QSARouteProbe(args.qsa_route == "host-like", pass_rows)
        result["qsa_route"] = {
            "route": args.qsa_route,
            "pass_rows": pass_rows,
            "stats": route.stats,
        }
        route.install()
        from vllm import LLM

        with single_pass_budget_guard(
            args.single_pass or args.hot_pool_rows is not None
        ):
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
        scenarios = run_scenarios(
            llm, sched, recorder, config.scenarios, args.seed, args.repeat
        )
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
        checks += route_checks(route.stats, placement, args, config, structure)
        if placement != "off":
            host_lines = capture.matches["host_store"]
            checks.append(
                check(
                    "host store logged"
                    if placement == "host"
                    else "no host store logged",
                    bool(host_lines) == (placement == "host"),
                    host_lines[:1],
                )
            )
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
                "capacity plan logged" if placement != "off" else "no plan logged",
                bool(plan_lines) == (placement != "off"),
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
        if args.explicit_shutdown:
            region = getattr(sched, "host_region", None)
            region_path = getattr(region, "mmap_path", None)
            core = engine.engine_core.engine_core
            recorder = runner = structure = None
            engine.engine_core.shutdown()
            drained = bool(getattr(core, "_host_kv_drained", False))
            unlinked = region_path is None or not os.path.exists(region_path)
            result["shutdown"] = {
                "region_path": region_path,
                "host_kv_drained": drained,
                "region_unlinked": unlinked,
            }
            if placement == "host":
                checks.append(
                    check(
                        "host KV drained and region unlinked at shutdown",
                        drained and unlinked and region_path is not None,
                        result["shutdown"],
                    )
                )
        result["checks"] = checks
        result["status"] = "ok"
        if any(c["ok"] is False for c in checks):
            exit_code = 3
    except SinglePassUnavailable as exc:
        result["status"] = "skipped"
        result["skip_reason"] = str(exc)
        print(str(exc), flush=True)
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
                "difference_stats": (v.get("a_vs_b") or {}).get("difference_stats"),
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
    run.add_argument(
        "--arm",
        choices=ARMS,
        required=True,
        help="off: one pool; on: long-context, engine-default main K/V "
        "placement; on-gpu / on-host: main_kv_placement gpu / host",
    )
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
    run.add_argument(
        "--host-pool-tokens",
        type=int,
        default=None,
        help="host history pool (default: the config's, or derived for tight)",
    )
    run.add_argument("--host-pool-gib", type=float, default=None)
    run.add_argument(
        "--hot-pool-gib",
        type=lambda value: "auto" if value == "auto" else float(value),
        default=None,
        help="GPU hot-page pool in GiB or auto for remaining KV memory "
        "(default: the config's manual 1.0 GiB)",
    )
    run.add_argument(
        "--hot-pool-rows",
        type=int,
        default=None,
        help="size the hot pool so one gather pass holds this many query "
        "rows (needs --geometry-from)",
    )
    run.add_argument(
        "--single-pass",
        action="store_true",
        help="require every step to fit one gather pass; without "
        "--hot-pool-gib/--hot-pool-rows the hot pool is sized for "
        "max_num_batched_tokens rows from --geometry-from",
    )
    run.add_argument("--state-blocks", type=int, default=None)
    run.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="run the config's scenario list this many times in one process",
    )
    run.add_argument("--no-prefix-caching", action="store_true")
    run.add_argument(
        "--mixed-prefill-latency-ms",
        type=float,
        default=None,
        help="scheduler mixed_prefill_step_latency_ms (0 disables the "
        "timing-fed prefill budget; default: the engine's)",
    )
    run.add_argument(
        "--prefill-pressure-budget",
        type=int,
        default=None,
        help="scheduler prefill_pressure_token_budget: a fixed cap on prefill "
        "tokens per step while decode is runnable",
    )
    run.add_argument("--num-gpu-blocks-override", type=int, default=None)
    run.add_argument(
        "--offload-cpu-mib",
        type=int,
        default=512,
        help="OffloadingConnector CPU backing of the on-host arm, MiB",
    )
    run.add_argument(
        "--explicit-shutdown",
        action="store_true",
        help="shut the engine core down before writing the JSON and record "
        "the host KV drain and region unlink",
    )
    run.add_argument(
        "--geometry-from",
        default=None,
        help="run JSON of a long-context arm; its plan geometry sizes the "
        "tight config's pools and --hot-pool-rows/--single-pass",
    )
    run.add_argument(
        "--qsa-route",
        choices=("default", "host-like"),
        default="default",
        help="test only, GPU placement: host-like drops the query positions "
        "(no XQA page4 route) like the host path",
    )
    run.add_argument(
        "--qsa-pass-rows",
        type=int,
        default=None,
        help="with --qsa-route host-like: launch the attention in passes of "
        "this many rows, like the host store's gather passes",
    )
    run.add_argument(
        "--qsa-pass-rows-from",
        default=None,
        help="on-host run JSON whose rows per gather pass --qsa-pass-rows takes",
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
