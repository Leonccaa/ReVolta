# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seeded random-schedule fuzzing of the long-context (Active Host KV) mode.

One run drives a real ``Scheduler`` or ``AsyncScheduler`` built on the
deployed CSA/GDN/PLE(/DFlash) geometry (TP4/DCP2, 1,664-token blocks, see
``long_context_kv_utils.production_specs``) with a seeded random workload and
seeded fault injection. A scheduler-side simulated offloading connector
(``SimOffloadConnector``) stands in for ``OffloadingConnector``: it keeps a
page-granular external tier keyed by prefix chain hashes, offers asynchronous
external hits, loads and stores with random latency, and injects read-back
failures, transfer submission failures, keys lost between lookup and plan,
store failures and stuck drains (a transfer fence that does not retire for a
bounded number of steps).

Every step checks the invariants in ``Driver.check``; a failure raises
``FuzzFailure`` with the seed, the step, a one-line reproduction and the tail
of the event log. ``shrink`` re-runs a failing seed with each event class
switched off to name the classes the failure needs.

Extension points for the Active Host KV v2 phases (page ownership-host SSD persistence)
are the
``EXTENSIONS`` hooks: an extension contributes random events
(``on_events``), worker-side completions (``on_worker_step``), per-step
invariants (``check``) and end-of-run invariants (``check_drained``).
"""

from __future__ import annotations

import dataclasses
import hashlib
import random
from collections import Counter, OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.outputs import DraftTokenIds, KVConnectorOutput, ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from .long_context_kv_utils import SPAN, host_step_receipts

MAX_MODEL_LEN = 10 * SPAN
HASH_BLOCK = 16
PAGE_HASHES = SPAN // HASH_BLOCK  # chain hashes per external page
VOCAB = 50_000


class FuzzFailure(AssertionError):
    pass


KNOWN_ISSUES = {
    "K2": (
        "A streaming session that holds KV while not running (idle, or "
        "resumed with new input and re-queued behind other requests) keeps "
        "HOST pages or state/history segment blocks; when the queue head "
        "needs them, admission stops at the head and the idle-session "
        "reclamation only runs for a HOT refusal, so nothing runs again. "
        "When K2 is avoided, a stall of this shape is broken by recomputing "
        "those holders through eviction recovery when the step cannot run."
    ),
    "K4": (
        "A completed async load is promoted only when the waiting-queue scan "
        "reaches it. A request ahead of it that needs the KV or hot quota the "
        "parked load holds stops the scan: under PRIORITY a higher-priority "
        "newcomer, under FCFS any request skipped earlier (for example a "
        "lookup the connector could not answer yet) that sits ahead in "
        "skipped_waiting. Parked loads are neither preempted nor reclaimed by "
        "eviction recovery, so nothing runs again. Avoided by FCFS or, for a stall "
        "of this shape, promoting the completed loads and recomputing them."
    ),
    "K5": (
        "AsyncScheduler with two frames in flight + speculative decoding: a "
        "state segment of exactly one request (state_blocks == "
        "state_blocks_per_request, accepted at startup) cannot hold one long "
        "prefill; it preempts itself at its third chunk and restarts forever. "
        "Avoided by giving such profiles a two-request state segment."
    ),
}

# ---------------------------------------------------------------------------
# Profile: everything a seed decides before the first step.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Profile:
    seed: int
    async_scheduling: bool
    pipeline_depth: int
    speculative_tokens: int
    full_isl: bool
    placement: str
    priority_policy: bool
    max_num_seqs: int
    host_pages: int
    hot_requests: float
    state_requests: int | None
    tier_pages: int
    delay_free_stores: bool
    # Workload.
    p_arrive: float = 0.35
    p_shared_prefix: float = 0.5
    p_stream: float = 0.25
    p_max_tokens_one: float = 0.3
    # Client and fault events (per step unless noted).
    p_abort: float = 0.02
    p_append_running: float = 0.02
    p_reset: float = 0.01
    p_reset_connector: float = 0.5  # given a reset
    p_reset_running: float = 0.005
    p_tier_evict: float = 0.03
    p_lookup_none: float = 0.02  # per lookup
    p_evict_after_lookup: float = 0.04  # per committed external hit
    p_submit_fail: float = 0.04  # per transfer
    p_load_fail: float = 0.06  # per load
    p_store_fail: float = 0.04  # per store
    p_stuck: float = 0.03  # per transfer: drain does not retire
    max_latency: int = 3
    max_stuck: int = 24
    max_client_delay: int = 12
    # Known findings the run steps around so that it keeps exploring; see
    # KNOWN_ISSUES. An empty tuple reproduces them.
    avoid: tuple[str, ...] = ("K2", "K4", "K5")

    FAULTS = (
        "p_abort",
        "p_append_running",
        "p_reset",
        "p_reset_running",
        "p_tier_evict",
        "p_lookup_none",
        "p_evict_after_lookup",
        "p_submit_fail",
        "p_load_fail",
        "p_store_fail",
        "p_stuck",
        "p_stream",
        "p_shared_prefix",
        "p_max_tokens_one",
    )

    def describe(self) -> str:
        return (
            f"seed={self.seed} {'async' if self.async_scheduling else 'sync'}"
            f"(depth={self.pipeline_depth}) spec={self.speculative_tokens} "
            f"full_isl={self.full_isl} placement={self.placement} "
            f"policy={'priority' if self.priority_policy else 'fcfs'} "
            f"max_seqs={self.max_num_seqs} host_pages={self.host_pages} "
            f"hot_requests={self.hot_requests} state_requests="
            f"{self.state_requests} tier_pages={self.tier_pages} "
            f"delay_free_stores={self.delay_free_stores}"
        )


def make_profile(seed: int, **overrides: Any) -> Profile:
    rng = random.Random(f"profile-{seed}")
    async_scheduling = rng.random() < 0.5
    profile = Profile(
        seed=seed,
        async_scheduling=async_scheduling,
        pipeline_depth=rng.choice((1, 2)) if async_scheduling else 1,
        speculative_tokens=rng.choice((0, 7)),
        full_isl=rng.random() < 0.3,
        placement=rng.choice(("gpu", "host")),
        priority_policy=rng.random() < 0.2,
        max_num_seqs=rng.choice((2, 4, 8)),
        host_pages=rng.choice((10, 11, 13, 16, 24, 40)),
        hot_requests=rng.choice((1.0, 1.5, 2.0, 3.0, 8.0)),
        state_requests=rng.choice((None, None, 1, 2, 3)),
        tier_pages=rng.choice((4, 16, 64)),
        delay_free_stores=rng.random() < 0.25,
    )
    profile = dataclasses.replace(profile, **overrides)
    if "K4" in profile.avoid and profile.priority_policy:
        profile = dataclasses.replace(profile, priority_policy=False)
    if (
        "K5" in profile.avoid
        and profile.pipeline_depth > 1
        and profile.speculative_tokens
        and profile.state_requests == 1
    ):
        profile = dataclasses.replace(profile, state_requests=2)
    return profile


# ---------------------------------------------------------------------------
# Scheduler construction.
# ---------------------------------------------------------------------------

_GEOMETRY: dict[tuple[str, int], Any] = {}


def _geometry(placement: str, spec: int):
    from .test_long_context_admission import build, settings

    key = (placement, spec)
    if key not in _GEOMETRY:
        probe = build(
            settings(
                host_pool_tokens=40 * SPAN,
                hot_pool_gib=2.0,
                main_kv_placement=placement,
            ),
            speculative_tokens=spec,
        )
        _GEOMETRY[key] = probe.kv_cache_manager.admission_ledger.plan.geometry
    return _GEOMETRY[key]


def build_scheduler(profile: Profile):
    from vllm.config import KVTransferConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
    from vllm.v1.core.kv_cache_hit_contract import load_failure_recompute_enabled
    from vllm.v1.core.long_context_kv import GiB
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    from .test_long_context_admission import build, settings

    geo = _geometry(profile.placement, profile.speculative_tokens)
    per_request = geo.hot_reserve_groups(MAX_MODEL_LEN)
    groups = int(per_request * profile.hot_requests)
    if profile.placement == "host":
        groups = max(
            groups, geo.hot_row_groups + profile.max_num_seqs * geo.hot_tail_groups
        )
    hot_gib = geo.hot_bytes_for_groups(groups, profile.max_num_seqs) / GiB + 1e-9
    host_tokens = profile.host_pages * SPAN
    if profile.placement == "host":
        host_tokens += 2 * SPAN  # ring and null IDs own host pages too
    kwargs: dict[str, Any] = dict(
        host_pool_tokens=host_tokens,
        hot_pool_gib=hot_gib,
        main_kv_placement=profile.placement,
    )
    if profile.state_requests:
        kwargs["state_blocks"] = geo.state_blocks_per_request * profile.state_requests
    base = build(
        settings(**kwargs),
        max_num_seqs=profile.max_num_seqs,
        reserve_full_isl=profile.full_isl,
        speculative_tokens=profile.speculative_tokens,
    )
    config = base.vllm_config
    config.scheduler_config.policy = "priority" if profile.priority_policy else "fcfs"
    config.scheduler_config.async_scheduling = profile.async_scheduling
    # Construct without a connector (the factory only accepts the built-in
    # OffloadingConnector with split block spaces), then attach the simulated
    # one with the long-context failure policy the real scheduler would use.
    config.kv_transfer_config = None
    cls = AsyncScheduler if profile.async_scheduling else Scheduler
    scheduler = cls(
        vllm_config=config,
        kv_cache_config=base.kv_cache_config,
        structured_output_manager=base.structured_output_manager,
        block_size=16,
        hash_block_size=HASH_BLOCK,
    )
    config.kv_transfer_config = KVTransferConfig(
        kv_connector="MockKVConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"matched_tokens": 0, "is_async": True},
    )
    scheduler.recompute_kv_load_failures = True
    scheduler.recompute_failed_loads = load_failure_recompute_enabled(config)
    ledger = scheduler.kv_cache_manager.admission_ledger
    assert ledger is not None
    assert not ledger.plan.startup_problems(), ledger.plan.startup_problems()
    assert scheduler.recompute_failed_loads
    scheduler.connector = SimOffloadConnector(
        config, KVConnectorRole.SCHEDULER, scheduler.kv_cache_config
    )
    return scheduler


# ---------------------------------------------------------------------------
# Simulated offloading connector (scheduler side plus aggregated workers).
# ---------------------------------------------------------------------------


@dataclass
class Job:
    job_id: int
    kind: str  # "load" | "store"
    req_id: str
    keys: list[Any]
    created_step: int
    submitted_step: int | None = None
    due_step: int = 0
    fail: bool = False
    stuck_until: int | None = None
    reason: str = ""


def _base_connector():
    from tests.v1.core.test_async_kv_admission import HybridMockConnector

    return HybridMockConnector


class SimOffloadConnector(_base_connector()):  # type: ignore[misc]
    """Contract-level stand-in for the scheduler side of OffloadingConnector.

    Matches the parts of the real connector the scheduler observes: lookup
    returns ``None`` while the request still has transfer jobs, an external
    hit ends on a page boundary below ``num_tokens`` (one page earlier with
    speculative decoding), a voided load is reported through
    ``take_failed_loads`` and the next lookup of that request offers no
    external hit, a failed transfer still reports ``finished_recving`` once,
    and stores of a preempted request are flushed. Worker results reach the
    scheduler aggregated over ranks.
    """

    driver: Driver

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tier: OrderedDict[Any, None] = OrderedDict()
        self.pins: Counter = Counter()
        self.jobs: dict[int, Job] = {}
        self.req_jobs: dict[str, set[int]] = {}
        self.new_jobs: list[Job] = []
        self.failed: set[str] = set()
        self.skip_once: set[str] = set()
        self.lookups: dict[str, tuple[int, int]] = {}
        self.stored_upto: dict[str, int] = {}
        self.delayed: set[str] = set()
        self.next_job = 0
        self.shutdown_called = False
        self.stats: Counter = Counter()

    # -- helpers --------------------------------------------------------------
    @staticmethod
    def page_key(request: Request, page: int):
        return request.block_hashes[(page + 1) * PAGE_HASHES - 1]

    def _jobs_of(self, req_id: str, kind: str | None = None) -> list[Job]:
        return [
            self.jobs[j]
            for j in sorted(self.req_jobs.get(req_id, ()))
            if kind is None or self.jobs[j].kind == kind
        ]

    def _add_job(self, job: Job) -> None:
        self.jobs[job.job_id] = job
        self.req_jobs.setdefault(job.req_id, set()).add(job.job_id)
        self.new_jobs.append(job)

    def _new_id(self) -> int:
        self.next_job += 1
        return self.next_job

    def _insert_tier(self, keys) -> None:
        cap = self.driver.profile.tier_pages
        for key in keys:
            self.tier[key] = None
            self.tier.move_to_end(key)
        for key in list(self.tier):
            if len(self.tier) <= cap:
                break
            if not self.pins[key]:
                del self.tier[key]
                self.stats["tier_evictions"] += 1

    def evict_random(self, rng: random.Random) -> int:
        victims = [k for k in self.tier if not self.pins[k]]
        rng.shuffle(victims)
        victims = victims[: rng.randint(0, len(victims))]
        for key in victims:
            del self.tier[key]
        return len(victims)

    # -- scheduler-side API ---------------------------------------------------
    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        d = self.driver
        rid = request.request_id
        self.stored_upto[rid] = 0
        if self._jobs_of(rid):
            # Real connector: a preempted request's stores are still pending.
            self.stats["lookup_none_pending"] += 1
            return None, False
        if d.rng.random() < d.profile.p_lookup_none:
            self.stats["lookup_none_injected"] += 1
            d.log(f"lookup {rid}: None (injected)")
            return None, False
        if rid in self.skip_once or request.skip_reading_prefix_cache:
            return 0, False
        limit = ((request.num_tokens - 1) // SPAN) * SPAN
        if d.profile.speculative_tokens:
            limit = max(limit - SPAN, 0)
        pages = 0
        while (pages + 1) * SPAN <= limit and self.page_key(
            request, pages
        ) in self.tier:
            pages += 1
        hit = pages * SPAN
        if hit - num_computed_tokens < SPAN:
            return 0, False
        for page in range(pages):
            self.tier.move_to_end(self.page_key(request, page))
        self.lookups[rid] = (num_computed_tokens, pages)
        self.stats["external_hits"] += 1
        external = hit - num_computed_tokens
        d.log(f"lookup {rid}: local={num_computed_tokens} external={external}")
        return external, True

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        d = self.driver
        rid = request.request_id
        if num_external_tokens == 0:
            self.skip_once.discard(rid)
            return
        local, pages = self.lookups.pop(rid)
        assert local + num_external_tokens == pages * SPAN, (local, pages)
        keys = [self.page_key(request, p) for p in range(local // SPAN, pages)]
        if d.rng.random() < d.profile.p_evict_after_lookup:
            for key in keys:
                if not self.pins[key]:
                    self.tier.pop(key, None)
        if not all(key in self.tier for key in keys):
            # The committed hit is no longer readable: void it.
            self.failed.add(rid)
            self.skip_once.add(rid)
            self.stats["abandoned_at_plan"] += 1
            d.log(f"plan {rid}: keys gone, load abandoned")
            return
        for key in keys:
            self.pins[key] += 1
        job = Job(self._new_id(), "load", rid, keys, d.step)
        self._add_job(job)
        self.stats["loads"] += 1

    def take_failed_loads(self) -> set[str]:
        failed, self.failed = self.failed, set()
        return failed

    def build_connector_meta(self, scheduler_output):
        d = self.driver
        rng = d.rng
        p = d.profile
        step = d.step
        # Preempted requests flush their pending stores (jobs_to_flush).
        for rid in sorted(scheduler_output.preempted_req_ids or ()):
            for job in self._jobs_of(rid, "store"):
                job.due_step = min(job.due_step, step)
        # Stores for pages computed by earlier steps.
        reqs = {r.req_id for r in scheduler_output.scheduled_new_reqs}
        reqs.update(scheduler_output.scheduled_cached_reqs.req_ids)
        for rid in sorted(reqs):
            request = d.scheduler.requests[rid]
            full = min(request.num_computed_tokens, len(request.block_hashes) * 16)
            pages = full // SPAN
            start = self.stored_upto.get(rid, 0)
            if pages <= start:
                continue
            self.stored_upto[rid] = pages
            pending = {
                k for j in self.jobs.values() if j.kind == "store" for k in j.keys
            }
            keys = [
                key
                for key in (self.page_key(request, i) for i in range(start, pages))
                if key not in self.tier and key not in pending
            ]
            if keys:
                self._add_job(Job(self._new_id(), "store", rid, keys, step))
                self.stats["stores"] += 1
        for job in self.new_jobs:
            job.submitted_step = step
            if rng.random() < p.p_submit_fail:
                job.fail = True
                job.reason = "submit"
                job.due_step = step
            else:
                job.due_step = step + rng.randint(0, p.max_latency)
                fail_p = p.p_load_fail if job.kind == "load" else p.p_store_fail
                if rng.random() < fail_p:
                    job.fail = True
                    job.reason = "readback" if job.kind == "load" else "store"
            if rng.random() < p.p_stuck:
                job.stuck_until = step + rng.randint(1, p.max_stuck)
                self.stats["stuck"] += 1
            d.log(
                f"submit {job.kind}#{job.job_id} {job.req_id} pages={len(job.keys)}"
                f" due={job.due_step} fail={job.reason or '-'}"
                f" stuck_until={job.stuck_until}"
            )
        self.new_jobs = []
        return super().build_connector_meta(scheduler_output)

    def request_finished_all_groups(self, request, block_ids):
        rid = request.request_id
        self.skip_once.discard(rid)
        self.failed.discard(rid)
        self.lookups.pop(rid, None)
        self.stored_upto.pop(rid, None)
        if self.driver.profile.delay_free_stores and self._jobs_of(rid, "store"):
            self.delayed.add(rid)
            return True, None
        return False, None

    def reset_cache(self) -> bool:
        dropped = [k for k in self.tier if not self.pins[k]]
        for key in dropped:
            del self.tier[key]
        self.driver.log(f"connector reset: dropped {len(dropped)} pages")
        return True

    def shutdown(self):
        self.shutdown_called = True

    # -- aggregated worker side -----------------------------------------------
    def worker_step(self, executed_step: int) -> KVConnectorOutput:
        """Results the workers report with the output of ``executed_step``."""
        d = self.driver
        recv: set[str] = set()
        sent: set[str] = set()
        for job_id in sorted(self.jobs):
            job = self.jobs[job_id]
            if job.submitted_step is None or job.submitted_step > executed_step:
                continue
            if job.due_step > d.step:
                continue
            if job.stuck_until is not None and job.stuck_until > d.step:
                continue
            del self.jobs[job_id]
            owned = self.req_jobs[job.req_id]
            owned.discard(job_id)
            if not owned:
                del self.req_jobs[job.req_id]
            request = d.scheduler.requests.get(job.req_id)
            if job.kind == "load":
                for key in job.keys:
                    self.pins[key] -= 1
                    if not self.pins[key]:
                        del self.pins[key]
                if job.fail:
                    self.stats["failed_loads"] += 1
                    if request is not None and not request.is_finished():
                        self.failed.add(job.req_id)
                        self.skip_once.add(job.req_id)
                recv.add(job.req_id)
            else:
                if job.fail:
                    self.stats["failed_stores"] += 1
                else:
                    self._insert_tier(job.keys)
                if job.req_id in self.delayed and not self._jobs_of(
                    job.req_id, "store"
                ):
                    self.delayed.discard(job.req_id)
                    sent.add(job.req_id)
            d.log(
                f"done {job.kind}#{job.job_id} {job.req_id}"
                f" {'FAILED(' + job.reason + ')' if job.fail else 'ok'}"
            )
        return KVConnectorOutput(
            finished_recving=recv or None, finished_sending=sent or None
        )

    def recover_all(self) -> None:
        for job in self.jobs.values():
            job.stuck_until = None


# ---------------------------------------------------------------------------
# Extension hooks for the v2 phases.
# ---------------------------------------------------------------------------


class Extension:
    """A v2 component joins the same driver through these hooks."""

    name = "extension"

    def attach(self, driver: Driver) -> None:  # build-time wiring
        pass

    def on_events(self, driver: Driver) -> None:  # random events, per step
        pass

    def on_worker_step(self, driver: Driver, output: KVConnectorOutput) -> None:
        pass

    def check(self, driver: Driver) -> None:  # per-step invariants
        pass

    def check_drained(self, driver: Driver) -> None:  # end-of-run invariants
        pass


EXTENSIONS: list[type[Extension]] = []


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------


@dataclass
class ClientSession:
    resumable: bool
    action_at: int | None = None
    queued_tokens: int = 0
    ended: bool = False


@dataclass
class Progress:
    high: int = 0  # computed tokens high-water mark plus output tokens
    last_step: int = 0


@dataclass
class RunResult:
    seed: int
    profile: str
    steps: int
    drain_steps: int
    finished: int
    aborted: int
    preemptions: int
    stats: dict[str, int]
    ledger_blocked: list[int]
    max_idle_wait: int
    k2_breaks: int
    k4_promotions: int
    digest: str
    events: list[str] = field(default_factory=list)


class Driver:
    def __init__(
        self,
        profile: Profile,
        steps: int,
        *,
        extensions: list[Extension] | None = None,
        watchdog: int = 64,
        drain_limit: int = 2000,
    ):
        init_none_hash(sha256)
        self.profile = profile
        self.steps = steps
        self.rng = random.Random(f"run-{profile.seed}")
        self.events: list[str] = []
        self.step = 0
        self.scheduler = build_scheduler(profile)
        self.conn: SimOffloadConnector = self.scheduler.connector
        self.conn.driver = self
        self.ledger = self.scheduler.kv_cache_manager.admission_ledger
        self.pipeline: deque = deque()
        self.clients: dict[str, ClientSession] = {}
        self.progress: dict[str, Progress] = {}
        self.last_global_progress = 0
        self.next_id = 0
        self.docs = [
            [self.rng.randrange(1, VOCAB) for _ in range(n * SPAN + extra)]
            for n, extra in ((1, 0), (2, 37), (4, 0), (6, 900))
        ]
        self.finished = 0
        self.aborted = 0
        self.arrivals_open = True
        self.max_idle_wait = 0
        self.k2_breaks = 0
        self.k4_promotions = 0
        self.aborted_ids: set[str] = set()
        self.p_stream = profile.p_stream
        self.digest = hashlib.sha256()
        self.watchdog = (
            watchdog
            + profile.max_client_delay
            + profile.max_stuck
            + profile.max_latency
        )
        self.drain_limit = drain_limit
        self.extensions = extensions or [cls() for cls in EXTENSIONS]
        for ext in self.extensions:
            ext.attach(self)

    # -- logging --------------------------------------------------------------
    def log(self, message: str) -> None:
        self.events.append(f"[{self.step}] {message}")

    def fail(self, message: str) -> None:
        tail = "\n".join(self.events[-60:])
        raise FuzzFailure(
            f"{message}\n  profile: {self.profile.describe()}\n"
            f"  reproduce: run_seed({self.profile.seed}, steps={self.steps}) "
            f"failed at step {self.step}\n  last events:\n{tail}"
        )

    # -- workload -------------------------------------------------------------
    def _request(self, rid, tokens, max_tokens, *, resumable=False, priority=0):
        params = SamplingParams(max_tokens=max_tokens, ignore_eos=True)
        params.update_from_generation_config({}, 0)
        request = Request(
            request_id=rid,
            prompt_token_ids=tokens,
            sampling_params=params,
            pooling_params=None,
            priority=priority,
            block_hasher=get_request_block_hasher(HASH_BLOCK, sha256),
        )
        request.resumable = resumable
        return request

    def _max_tokens(self) -> int:
        if self.rng.random() < self.profile.p_max_tokens_one:
            return 1
        return self.rng.choice((2, 3, 8, 16, 40))

    def _prompt(self, budget: int) -> list[int]:
        rng = self.rng
        shape = rng.random()
        if shape < 0.3:
            length = rng.randint(16, SPAN)
        elif shape < 0.55:
            length = rng.randint(SPAN, 4 * SPAN)
        elif shape < 0.8:
            length = rng.randint(4 * SPAN, budget)
        else:  # just below a page boundary: decode crosses into a new page
            length = rng.randint(1, max(1, budget // SPAN)) * SPAN - rng.randint(1, 8)
        length = max(1, min(length, budget))
        tokens: list[int] = []
        if rng.random() < self.profile.p_shared_prefix:
            doc = rng.choice(self.docs)
            tokens = list(doc[: min(len(doc), length)])
        tokens += [rng.randrange(1, VOCAB) for _ in range(length - len(tokens))]
        return tokens

    def arrive(self) -> None:
        live = sum(1 for r in self.scheduler.requests.values() if not r.is_finished())
        if live >= 3 * self.profile.max_num_seqs:
            return
        max_tokens = self._max_tokens()
        resumable = self.rng.random() < self.p_stream
        budget = MAX_MODEL_LEN - max_tokens - 16
        if resumable:
            budget = min(budget, MAX_MODEL_LEN // 2)
        tokens = self._prompt(budget)
        rid = f"r{self.next_id}"
        self.next_id += 1
        priority = self.rng.randint(0, 3) if self.profile.priority_policy else 0
        request = self._request(
            rid, tokens, max_tokens, resumable=resumable, priority=priority
        )
        self.clients[rid] = ClientSession(resumable=resumable)
        self.progress[rid] = Progress(last_step=self.step)
        self.scheduler.add_request(request)
        self.log(
            f"arrive {rid} prompt={len(tokens)} max_tokens={max_tokens}"
            f" stream={resumable} prio={priority}"
        )

    def _session_room(self, request: Request) -> int:
        client = self.clients[request.request_id]
        used = request.num_tokens + client.queued_tokens + 64
        return MAX_MODEL_LEN - used

    def _append(self, request: Request, *, queued: bool) -> bool:
        room = self._session_room(request)
        max_tokens = self._max_tokens()
        if room - max_tokens < 16:
            return False
        n = self.rng.randint(1, min(room - max_tokens, 3 * SPAN))
        tokens = [self.rng.randrange(1, VOCAB) for _ in range(n)]
        update = self._request(request.request_id, tokens, max_tokens, resumable=True)
        if queued:
            self.clients[request.request_id].queued_tokens += n + max_tokens
        self.scheduler.add_request(update)
        self.log(
            f"append {request.request_id} +{n} max_tokens={max_tokens} queued={queued}"
        )
        return True

    def _end(self, request: Request) -> None:
        sentinel = self._request(request.request_id, [1], 1, resumable=False)
        self.clients[request.request_id].ended = True
        if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
            self.aborted_ids.add(request.request_id)
        self.scheduler.add_request(sentinel)
        self.log(f"end session {request.request_id}")

    def abort(self, rid: str, why: str) -> None:
        done = self.scheduler.finish_requests(rid, RequestStatus.FINISHED_ABORTED)
        if done:
            self.aborted += 1
            self.aborted_ids.add(rid)
            self.log(f"abort {rid} ({why})")

    def client_events(self) -> None:
        rng = self.rng
        p = self.profile
        for rid, request in list(self.scheduler.requests.items()):
            if request.is_finished():
                continue
            client = self.clients[rid]
            if request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                client.queued_tokens = 0
                if client.action_at is None:
                    client.action_at = self.step + rng.randint(0, p.max_client_delay)
                if self.step >= client.action_at or not self.arrivals_open:
                    client.action_at = None
                    roll = rng.random()
                    if (
                        self.arrivals_open
                        and roll < 0.6
                        and self._append(request, queued=False)
                    ):
                        continue
                    if roll < 0.9 or not p.p_abort:
                        self._end(request)
                    else:
                        self.abort(rid, "client left an idle session")
                continue
            if (
                client.resumable
                and not client.ended
                and self.arrivals_open
                and request.status == RequestStatus.RUNNING
                and rng.random() < p.p_append_running
            ):
                self._append(request, queued=True)
        if p.p_abort and rng.random() < p.p_abort:
            live = sorted(
                rid for rid, r in self.scheduler.requests.items() if not r.is_finished()
            )
            if live:
                self.abort(rng.choice(live), "random")

    def system_events(self) -> None:
        rng = self.rng
        p = self.profile
        if p.p_tier_evict and rng.random() < p.p_tier_evict:
            n = self.conn.evict_random(rng)
            self.log(f"tier pressure: evicted {n} pages")
        if p.p_reset and rng.random() < p.p_reset:
            connector = rng.random() < p.p_reset_connector
            ok = self.scheduler.reset_prefix_cache(reset_connector=connector)
            self.log(f"reset_prefix_cache(connector={connector}) -> {ok}")
        if p.p_reset_running and rng.random() < p.p_reset_running:
            self.reset_running()

    def reset_running(self) -> None:
        """pause_generation(keep) + reset: drain the pipeline, then preempt."""
        s = self.scheduler
        while self.pipeline:
            self.complete_oldest()
        manager = s.kv_cache_manager
        for rid, request in s.requests.items():
            if request.status != RequestStatus.RUNNING and any(
                manager.coordinator.single_type_managers[g].req_to_blocks.get(rid)
                for g in range(len(manager.coordinator.single_type_managers))
            ):
                self.log(f"reset_running skipped: {rid} holds KV while not running")
                return
        connector = self.rng.random() < 0.5
        s.reset_prefix_cache(reset_running_requests=True, reset_connector=connector)
        self.log(f"reset_prefix_cache(running=True, connector={connector})")

    # -- engine ---------------------------------------------------------------
    def engine_step(self) -> None:
        s = self.scheduler
        output = s.schedule()
        snapshot = {}
        for rid in output.num_scheduled_tokens:
            request = s.requests[rid]
            spec = len(output.scheduled_spec_decode_tokens.get(rid, ()))
            snapshot[rid] = (not request.is_prefill_chunk, spec)
        self.digest.update(
            repr(
                (
                    self.step,
                    sorted(output.num_scheduled_tokens.items()),
                    sorted(output.preempted_req_ids or ()),
                )
            ).encode()
        )
        if output.num_scheduled_tokens or output.preempted_req_ids:
            self.log(
                f"schedule {dict(output.num_scheduled_tokens)}"
                + (
                    f" preempted={sorted(output.preempted_req_ids)}"
                    if output.preempted_req_ids
                    else ""
                )
            )
        self.pipeline.append((self.step, output, snapshot))
        while len(self.pipeline) >= self.profile.pipeline_depth:
            self.complete_oldest()

    def complete_oldest(self) -> None:
        s = self.scheduler
        executed, output, snapshot = self.pipeline.popleft()
        ids = list(output.num_scheduled_tokens)
        sampled: list[list[int]] = []
        for rid in ids:
            samples, spec = snapshot[rid]
            if not samples:
                sampled.append([])
                continue
            accepted = self.rng.randint(0, spec) if spec else 0
            sampled.append([1] * (1 + accepted))
        kv_output = self.conn.worker_step(executed)
        if getattr(s, "host_history_pool", None) is not None:
            # Every rank executed the step and retired its host KV seals.
            kv_output.kv_connector_worker_meta = (
                host_step_receipts(s, output) or OffloadingWorkerMetadata()
            )
        for ext in self.extensions:
            ext.on_worker_step(self, kv_output)
        before = {rid: r.status for rid, r in s.requests.items() if not r.is_finished()}
        s.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=ids,
                req_id_to_index={rid: i for i, rid in enumerate(ids)},
                sampled_token_ids=sampled,
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
                kv_connector_output=kv_output,
            ),
        )
        for rid in before:
            request = s.requests.get(rid)
            if request is None or request.is_finished():
                self.finished += 1
                self.log(f"finished {rid}")
        if not self.profile.async_scheduling and self.profile.speculative_tokens:
            drafts = [
                rid
                for rid, (samples, _) in snapshot.items()
                if samples
                and rid in s.requests
                and s.requests[rid].status == RequestStatus.RUNNING
            ]
            if drafts:
                s.update_draft_token_ids(
                    DraftTokenIds(
                        req_ids=drafts,
                        draft_token_ids=[[1] * self.profile.speculative_tokens]
                        * len(drafts),
                    )
                )

    # -- invariants -------------------------------------------------------------
    def _held_blocks(self) -> dict[str, int]:
        held: Counter = Counter()
        for manager in self.scheduler.kv_cache_manager.coordinator.single_type_managers:
            for rid, blocks in manager.req_to_blocks.items():
                held[rid] += sum(1 for b in blocks if not b.is_null)
        return held

    def check(self) -> None:
        from vllm.v1.core.long_context_kv import HOST

        from .test_long_context_admission import recount

        s = self.scheduler
        ledger = self.ledger
        manager = s.kv_cache_manager
        coordinator = manager.coordinator
        # 1. Domains in range and the ledger matches the managers.
        for domain in ledger.ledger_domains:
            if not 0 <= ledger.in_use[domain] <= ledger.capacity[domain]:
                self.fail(
                    f"domain {domain} in_use={ledger.in_use} cap={ledger.capacity}"
                )
        if not 0 <= ledger.in_use[HOST] <= ledger.capacity[HOST]:
            self.fail(f"HOST in_use={ledger.in_use[HOST]} cap={ledger.capacity[HOST]}")
        counted = recount(s)
        if ledger.in_use != counted:
            self.fail(f"ledger in_use {ledger.in_use} != recount {counted}")
        if ledger.untracked_allocations:
            self.fail(f"untracked allocations {ledger.untracked_allocations}")
        if ledger.hot_used != sum(ledger._hot.values()):
            self.fail(f"hot_used {ledger.hot_used} != sum {sum(ledger._hot.values())}")
        if not 0 <= ledger.hot_used <= ledger.plan.hot_pool_groups:
            self.fail(f"hot_used {ledger.hot_used} > {ledger.plan.hot_pool_groups}")
        held = self._held_blocks()
        for rid in ledger._hot:
            if rid not in s.requests:
                self.fail(f"hot reservation of freed request {rid}")
            if not held.get(rid):
                self.fail(f"hot reservation of {rid} without any held KV")
        # 2. Every referenced block is owned; pools account exactly.
        refs: Counter = Counter()
        for gid, single in enumerate(coordinator.single_type_managers):
            for rid, blocks in single.req_to_blocks.items():
                if blocks and rid not in s.requests:
                    self.fail(f"group {gid} keeps blocks of freed request {rid}")
                for block in blocks:
                    if not block.is_null:
                        refs[id(block)] += 1
        for index, pool in enumerate(coordinator.block_pools):
            used = 0
            for block in pool.blocks:
                if block.is_null:
                    continue
                if block.ref_cnt != refs.get(id(block), 0):
                    self.fail(
                        f"pool {index} block {block.block_id} ref_cnt={block.ref_cnt}"
                        f" but {refs.get(id(block), 0)} owners"
                    )
                used += block.ref_cnt > 0
            free, expected = pool.get_num_free_blocks(), pool.num_gpu_blocks - 1 - used
            if (
                pool is getattr(s, "host_history_pool", None)
                and self.pipeline
                and free <= expected
            ):
                # Host pages a step still in flight may read are withheld until
                # its receipts retire them; equality returns once it completes.
                continue
            if free != expected:
                self.fail(f"pool {index} free count drifted")
        # 3. Scheduler structure.
        queued: Counter = Counter()
        where: dict[str, list[str]] = {}
        for name, queue in (
            ("waiting", s.waiting),
            ("skipped_waiting", s.skipped_waiting),
            ("running", s.running),
        ):
            for request in queue:
                queued[request.request_id] += 1
                where.setdefault(request.request_id, []).append(name)
        for request in s.running:
            if request.status != RequestStatus.RUNNING:
                self.fail(f"{request.request_id} in running with {request.status}")
        if len(s.running) > self.profile.max_num_seqs:
            self.fail(f"{len(s.running)} running > max_num_seqs")
        streaming = 0
        for rid, request in s.requests.items():
            if request.is_finished():
                if queued[rid]:
                    self.fail(f"finished {rid} still queued")
                if not self._finished_held_reason(rid):
                    self.fail(f"finished {rid} kept without a pending transfer")
                continue
            if queued[rid] != 1:
                self.fail(
                    f"{rid} ({request.status.name}) queued {queued[rid]} times:"
                    f" {where.get(rid)}"
                )
            streaming += request.status == RequestStatus.WAITING_FOR_STREAMING_REQ
            if request.num_in_flight_tokens < 0 or request.num_output_placeholders < 0:
                self.fail(f"{rid} negative in-flight/placeholder counters")
            if not self.pipeline and request.num_in_flight_tokens:
                self.fail(f"{rid} in-flight tokens with an empty pipeline")
        for rid in queued:
            if rid not in s.requests:
                self.fail(f"queued {rid} is not tracked")
        if streaming != s.num_waiting_for_streaming_input:
            self.fail(
                f"{streaming} idle sessions but counter "
                f"{s.num_waiting_for_streaming_input}"
            )
        for request in s._inflight_prefills:
            if s.requests.get(request.request_id) is not request:
                self.fail(f"in-flight prefill {request.request_id} not tracked")
        for rid in s.finished_recving_kv_req_ids | s.failed_recving_kv_req_ids:
            if rid not in s.requests:
                self.fail(f"receive marker of untracked {rid}")
        # 4. Simulated connector: pins and jobs belong to live transfers.
        conn = self.conn
        expect: Counter = Counter()
        for job in conn.jobs.values():
            request = s.requests.get(job.req_id)
            if job.kind == "load":
                expect.update(job.keys)
                if request is None:
                    self.fail(f"load job {job.job_id} of freed {job.req_id}")
                if (
                    not request.is_finished()
                    and request.status != RequestStatus.WAITING_FOR_REMOTE_KVS
                ):
                    self.fail(f"load job of {job.req_id} in {request.status.name}")
        if +expect != +conn.pins:
            self.fail("connector pins do not match live load jobs")
        for ext in self.extensions:
            ext.check(self)
        # 5. Progress.
        self.check_progress()

    def _finished_held_reason(self, rid: str) -> bool:
        s = self.scheduler
        loads = self.conn._jobs_of(rid, "load")
        if loads and rid not in s.finished_recving_kv_req_ids:
            return True  # aborted during a load: freed on its receive
        return rid in self.conn.delayed

    def check_progress(self) -> None:
        s = self.scheduler
        moved = False
        for rid, request in s.requests.items():
            if request.is_finished():
                continue
            tracker = self.progress.setdefault(rid, Progress(last_step=self.step))
            mark = request.num_computed_tokens + 1_000_000 * request.num_output_tokens
            mark += 10_000_000 * request.num_prompt_tokens
            if mark > tracker.high:
                tracker.high = mark
                tracker.last_step = self.step
                moved = True
            elif request.status == RequestStatus.WAITING_FOR_STREAMING_REQ:
                tracker.last_step = self.step  # the client owns this wait
            else:
                self.max_idle_wait = max(
                    self.max_idle_wait, self.step - tracker.last_step
                )
        for rid in list(self.progress):
            if rid not in s.requests or s.requests[rid].is_finished():
                del self.progress[rid]
                # A client abort or session end is not scheduler progress.
                moved |= rid not in self.aborted_ids
        if moved:
            self.last_global_progress = self.step
        stalled = [
            rid
            for rid, r in s.requests.items()
            if not r.is_finished()
            and r.status != RequestStatus.WAITING_FOR_STREAMING_REQ
        ]
        idle_for = self.step - self.last_global_progress
        if (
            stalled
            and {"K2", "K4"} & set(self.profile.avoid)
            and idle_for > self.watchdog // 2
            and self.reclaim_holders()
        ):
            self.last_global_progress = self.step
            return
        if stalled and idle_for > self.watchdog:
            self.fail(
                f"no progress for {self.watchdog} steps: {self.wait_reasons(stalled)}"
            )

    def reclaim_holders(self) -> bool:
        """K2 avoidance: recompute non-running KV holders, as load regression does for
        HOT."""
        s = self.scheduler
        if s.running:
            return False
        if "K4" in self.profile.avoid:
            for request in list(s.requests.values()):
                if (
                    request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                    and request.request_id in s.finished_recving_kv_req_ids
                ):
                    s._try_promote_blocked_waiting_request(request)
                    self.k4_promotions += 1
                    self.log(f"K4 stall: promoted completed load {request.request_id}")
        held = self._held_blocks()
        holders = [
            r
            for rid, r in s.requests.items()
            if held.get(rid)
            and r.status
            in (
                RequestStatus.WAITING,
                RequestStatus.PREEMPTED,
                RequestStatus.WAITING_FOR_STREAMING_REQ,
            )
            and r.num_computed_tokens > 0
            and r.num_in_flight_tokens == 0
        ]
        if not holders:
            return False
        for request in holders:
            s._preempt_request(request, 0.0, enqueue=False)
        self.k2_breaks += 1
        self.log(f"stall broken: recomputing {[r.request_id for r in holders]}")
        return True

    def wait_reasons(self, rids) -> str:
        s = self.scheduler
        held = self._held_blocks()
        ledger = self.ledger
        parts = []
        for rid in rids:
            r = s.requests[rid]
            jobs = [
                f"{j.kind}#{j.job_id}{'(stuck)' if j.stuck_until else ''}"
                for j in self.conn._jobs_of(rid)
            ]
            parts.append(
                f"{rid}:{r.status.name} computed={r.num_computed_tokens}/"
                f"{r.num_tokens} blocks={held.get(rid, 0)} jobs={jobs}"
            )
        idle = [
            f"{rid}(blocks={held.get(rid, 0)})"
            for rid, r in s.requests.items()
            if r.status == RequestStatus.WAITING_FOR_STREAMING_REQ
        ]
        return (
            "; ".join(parts)
            + f" | idle sessions: {idle} | ledger {ledger.snapshot()}"
            + f" blocked={ledger.blocked} | pools free="
            + str(
                [
                    p.get_num_free_blocks()
                    for p in s.kv_cache_manager.coordinator.block_pools
                ]
            )
        )

    def check_drained(self) -> None:
        s = self.scheduler
        ledger = self.ledger
        if s.requests or s.waiting or s.skipped_waiting or s.running:
            self.fail(f"requests left after drain: {sorted(s.requests)}")
        for index, pool in enumerate(s.kv_cache_manager.coordinator.block_pools):
            if pool.get_num_free_blocks() != pool.num_gpu_blocks - 1:
                self.fail(f"pool {index} not empty after drain")
        if ledger.in_use != [0, 0, 0] or ledger.hot_used or ledger._hot:
            self.fail(f"ledger not zero after drain: {ledger.snapshot()}")
        if s._inflight_prefills or s.finished_recving_kv_req_ids:
            self.fail("scheduler receive state left after drain")
        if s.failed_recving_kv_req_ids or s.num_waiting_for_streaming_input:
            self.fail("scheduler failure/streaming markers left after drain")
        conn = self.conn
        if conn.jobs or +conn.pins or conn.failed or conn.skip_once or conn.delayed:
            self.fail("connector state left after drain")
        if not conn.shutdown_called:
            self.fail("connector shutdown not called")
        for ext in self.extensions:
            ext.check_drained(self)

    # -- run ------------------------------------------------------------------
    def tick(self) -> None:
        self.step += 1
        for ext in self.extensions:
            ext.on_events(self)
        self.engine_step()
        self.check()

    def run(self) -> RunResult:
        try:
            for _ in range(self.steps):
                if self.arrivals_open and self.rng.random() < self.profile.p_arrive:
                    self.arrive()
                self.client_events()
                self.system_events()
                self.tick()
            self.drain()
        except FuzzFailure:
            raise
        except Exception as error:  # an assert or crash inside vLLM
            import traceback

            self.fail(
                f"exception {type(error).__name__}: {error}\n"
                + "".join(traceback.format_exception(error)[-6:])
            )
        return RunResult(
            seed=self.profile.seed,
            profile=self.profile.describe(),
            steps=self.steps,
            drain_steps=self.step - self.steps,
            finished=self.finished,
            aborted=self.aborted,
            preemptions=sum(1 for e in self.events if "preempted=" in e),
            stats=dict(self.conn.stats),
            ledger_blocked=list(self.ledger.blocked),
            max_idle_wait=self.max_idle_wait,
            k2_breaks=self.k2_breaks,
            k4_promotions=self.k4_promotions,
            digest=self.digest.hexdigest()[:16],
        )

    def drain(self) -> None:
        """Shutdown: close arrivals, end sessions, let faults recover."""
        self.arrivals_open = False
        self.log("drain: arrivals closed")
        abort_all = self.rng.random() < 0.3
        if abort_all:
            for rid in sorted(self.scheduler.requests):
                self.abort(rid, "shutdown")
        start = self.step
        recovered = False
        while self.scheduler.requests or self.pipeline or self.conn.jobs:
            if self.step - start > self.drain_limit:
                stuck = [
                    rid
                    for rid, r in self.scheduler.requests.items()
                    if not r.is_finished()
                ]
                self.fail(f"drain did not finish: {self.wait_reasons(stuck)}")
            if not recovered and self.step - start > self.profile.max_stuck:
                self.conn.recover_all()
                recovered = True
            self.client_events()
            if self.scheduler.requests or self.conn.jobs:
                self.tick()
            elif self.pipeline:
                self.complete_oldest()
        self.scheduler.shutdown()
        self.check_drained()


def run_seed(seed: int, steps: int = 300, **overrides: Any) -> RunResult:
    return Driver(make_profile(seed, **overrides), steps).run()


def shrink(seed: int, steps: int, **overrides: Any) -> dict[str, Any]:
    """Name the event classes a failing seed needs and its first failing step."""

    def fails(**extra) -> bool:
        try:
            run_seed(seed, steps, **{**overrides, **extra})
        except FuzzFailure:
            return True
        return False

    assert fails(), "seed does not fail"
    needed: dict[str, float] = {}
    for name in Profile.FAULTS:
        off = {**needed, name: 0.0}
        if fails(**off):
            needed[name] = 0.0
    lo, hi = 1, steps
    while lo < hi:
        mid = (lo + hi) // 2
        try:
            run_seed(seed, mid, **{**overrides, **needed})
            lo = mid + 1
        except FuzzFailure:
            hi = mid
    return {
        "seed": seed,
        "steps": lo,
        "switched_off": sorted(needed),
        "still_required": sorted(set(Profile.FAULTS) - set(needed)),
    }
