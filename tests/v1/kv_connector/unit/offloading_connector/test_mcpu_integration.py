# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent gates and joint scheduler statistics must compose."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from tests.v1.kv_connector.unit.offloading_connector.test_cpu_draft_offload import (
    _cpu_manager,
    _hashes,
    _kv_config,
    _output,
    _request,
    _standard_run,
    _vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.test_offload_counters import (
    _admit,
    _complete,
    _sibling,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import scheduler as sm
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    CPU_RESIDENT_KEY,
    LABELLED_COUNTERS,
    OffloadingConnectorStats,
    OffloadPromMetrics,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
    OffloadingConnector,
)

pytestmark = pytest.mark.cpu_test
ABSENT = "absent"


def make(counters=ABSENT, *, dependency=False, stats=False, grouped=False):
    config = _vllm_config("window")
    extra = config.kv_transfer_config.kv_connector_extra_config
    if counters != ABSENT:
        extra["cpu.offload_counters"] = counters
    if dependency:
        extra["cpu.enable_dependency_eviction"] = True
    if stats:
        extra["cpu.residency_stats"] = True
    spec = MockOffloadingSpec(config, _kv_config())
    spec.manager = _cpu_manager(grouped)
    return config, sm.OffloadingConnectorScheduler(spec)


def connector(scheduler):
    result = object.__new__(OffloadingConnector)
    result.connector_worker = None
    result.connector_scheduler = scheduler
    return result


def prometheus(config):
    registry = CollectorRegistry()

    def bind(cls):
        return lambda *args, **kwargs: cls(*args, registry=registry, **kwargs)

    prom = OffloadPromMetrics(
        config,
        {Gauge: bind(Gauge), Counter: bind(Counter), Histogram: bind(Histogram)},
        ["engine"],
        {0: ["0"]},
    )
    return prom, registry


@pytest.mark.parametrize("value", [0, 1, "true", "false", None, [], {}])
def test_counter_bool_validation(value):
    config = _vllm_config("all")
    config.kv_transfer_config.kv_connector_extra_config["cpu.offload_counters"] = value
    with pytest.raises(ValueError, match=r"cpu\.offload_counters must be a bool"):
        MockOffloadingSpec(config, _kv_config())


@pytest.mark.parametrize("value", [ABSENT, False])
@pytest.mark.parametrize(
    "dependency,stats", [(False, False), (True, False), (False, True)]
)
def test_disabled_counters_do_no_counting_or_registration(value, dependency, stats):
    with (
        patch.object(
            sm,
            "build_offload_counters",
            side_effect=AssertionError("counter construction"),
        ),
        patch.object(sm.logger, "info") as log,
    ):
        config, scheduler = make(value, dependency=dependency, stats=stats)
        _standard_run(scheduler, complete=True).finish()
        req = _sibling(scheduler, 230)
        hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
        assert hit == 192
        with patch.object(
            scheduler,
            "_count_admitted_lookup",
            side_effect=AssertionError("admission counting"),
        ):
            _admit(scheduler, req, hit)
        assert scheduler._offload_counters is None
        assert scheduler.offload_counter_deltas() == {}
        assert scheduler._req_status[req.request_id].lookup_shrink is None
        scheduler._counters_logged_at = -1e9
        scheduler._maybe_log_counters()
    messages = [call.args[0] % call.args[1:] for call in log.call_args_list]
    assert not any("Offload counter kinds" in m for m in messages)
    [line] = [m for m in messages if m.startswith("Offload counters:")]
    assert "stored_blocks_by_kind=" not in line
    assert "lookup_shrink_tokens=" not in line
    assert "hit_draft_window_needed=" not in line
    prom, registry = prometheus(config)
    sample = connector(scheduler).get_kv_connector_stats()
    if sample is not None:
        assert not (sample.data.keys() & LABELLED_COUNTERS.keys())
        prom.observe(sample.data)
    # Even misrouted labelled data cannot register or sample a disabled family.
    injected = OffloadingConnectorStats()
    for key in LABELLED_COUNTERS:
        injected.record_labelled_counts(key, {"target": 1})
    prom.observe(injected.data)
    names = {metric.name for metric in registry.collect()}
    assert not (names & {name for name, _, _ in LABELLED_COUNTERS.values()})
    assert not prom.labelled_counters


@pytest.mark.parametrize(
    "dependency,stats", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("grouped", [False, True])
def test_joint_stats_latest_gauge_and_additive_counters(dependency, stats, grouped):
    config, scheduler = make(True, dependency=dependency, stats=stats, grouped=grouped)
    _standard_run(scheduler, complete=True).finish()
    req = _sibling(scheduler, 230)
    hit, _ = scheduler.get_num_new_matched_tokens(req, 0)
    _admit(scheduler, req, hit)
    load = scheduler.build_connector_meta(_output(req, 0))
    _complete(scheduler, load.load_jobs)
    other = _request("r3", 230, hashes=_hashes(8) + _hashes(6, salt="other"))
    scheduler.on_new_request(other)
    assert scheduler.get_num_new_matched_tokens(other, 0)[0] == 0
    _admit(scheduler, other, 0)
    conn = connector(scheduler)
    prom, registry = prometheus(config)
    with patch("time.monotonic", return_value=100):
        with patch.object(sm.logger, "info") as log:
            scheduler._counters_logged_at = 0
            scheduler._maybe_log_counters()
        [call] = [
            c for c in log.call_args_list if c.args[0].startswith("Offload counters:")
        ]
        line = call.args[0] % call.args[1:]
        assert line.count("stored_blocks_by_kind=") == 1
        assert ("cpu_resident_blocks=" in line) == (dependency or stats)
        first = conn.get_kv_connector_stats()
        assert first is not None
        assert "cpu_stored_blocks" in first.data
        assert "hit_draft_window_needed" in first.data
        assert "lookup_shrink_tokens" in first.data
        assert (CPU_RESIDENT_KEY in first.data) == (dependency or stats)
        assert conn.get_kv_connector_stats() is None
    scheduler.reset_cache()
    with patch("time.monotonic", return_value=160):
        second = conn.get_kv_connector_stats()
    assert second is not None  # reset eviction delta, plus the fresh zero gauge
    extra = OffloadingConnectorStats()
    extra.record_labelled_counts("cpu_stored_blocks", {"target": 2})
    merged = (
        OffloadingConnectorStats().aggregate(first).aggregate(second).aggregate(extra)
    )
    reduced = merged.reduce()
    assert reduced["cpu_stored_blocks_target"] == 8
    assert reduced["cpu_stored_blocks_state"] == 2
    assert reduced["cpu_stored_blocks_draft"] == 5
    assert reduced["hit_draft_window_needed_true"] == 1
    assert reduced["lookup_shrink_tokens_draft_window"] == 128
    if dependency or stats:
        assert len(merged.data[CPU_RESIDENT_KEY]) == 1
        assert sum(v for k, v in reduced.items() if k.startswith(CPU_RESIDENT_KEY)) == 0
    prom.observe(merged.data)
    assert (
        registry.get_sample_value(
            "vllm:kv_offload_cpu_stored_blocks_total", {"engine": "0", "kind": "draft"}
        )
        == 5
    )
    with patch("time.monotonic", return_value=160):
        assert conn.get_kv_connector_stats() is None


def test_no_transfer_config_registers_no_counter_families():
    prom, registry = prometheus(SimpleNamespace(kv_transfer_config=None))
    assert not prom._labelled_counters
    assert not (
        {m.name for m in registry.collect()}
        & {name for name, _, _ in LABELLED_COUNTERS.values()}
    )
