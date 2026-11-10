# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-local SSD lineage and verified physical reads through normal IO."""

from types import SimpleNamespace

import pytest

from tests.v1.kv_offload.test_unified_ssd_transactions import UnitEnv
from tests.v1.metrics.test_store_metrics import request
from vllm.v1.core.kv_cache_hit_contract import (
    GroupHitRule,
    HitNeed,
    KVHitContract,
    chain_hash_from_request,
)
from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host.page_table import HostPageTable, Result
from vllm.v1.metrics.store import StoreMetrics


@pytest.mark.parametrize(
    "group,need,index,expected",
    [
        (0, HitNeed.PREFIX, 1, 16),
        (1, HitNeed.STATE, 3, 64),
        (2, HitNeed.WINDOW, 3, 16),
        (1, HitNeed.STATE, 1, 0),  # unused recurrent checkpoint
    ],
)
@pytest.mark.parametrize("external", [0, 32])
@pytest.mark.parametrize(
    "admission_retries,recycle", [(0, False), (2, False), (2, True)]
)
def test_prefetch_lineage_survives_local_admission_and_receive(
    group, need, index, expected, external, admission_retries, recycle
):
    from vllm.v1.kv_offload.host.receive import ReceiveEntry, ReceiveSource, ReceiveTxn

    m, r = StoreMetrics(), request()
    r.request_id = "reader"
    r.block_hashes = [bytes([i]) * 32 for i in range(1, 7)]
    rule = GroupHitRule(group, need, 16, window_tokens=32)
    contract = KVHitContract([rule], "connector", 16)
    table = HostPageTable(10, num_workers=1)
    h = table.reserve()
    table.producer_result(h, compute=Result.OK, seal=Result.OK)
    chain = chain_hash_from_request(r.block_hashes, 16)
    table.publish(h, make_offload_key(chain(rule, index), group))
    holds = {r.request_id: [h]}
    ssd = SimpleNamespace(table=table, holds=lambda req_id: holds.get(req_id, []))
    m.begin_lookup(r)
    m.prefetched(r, ssd, contract, 64)
    holds.clear()  # lookup releases holds before allocation/admission
    if recycle:
        key = table.get(h).key
        table._recycle(table.get(h))
        replacement = table.reserve()
        table.producer_result(replacement, compute=Result.OK, seal=Result.OK)
        table.publish(replacement, key)
        expected = 0
    for _ in range(admission_retries):
        m.lookup(r, "local", 100, 64 - external)
        m.retry(r, "capacity", 10.5)
        m.begin_lookup(r)
        m.prefetched(r, ssd, contract, 64)
    m.admit(r, 64 - external, external, True, 11.0)
    if external:
        txn = ReceiveTxn(
            1,
            r.request_id,
            1,
            tuple(
                ReceiveEntry(i, ReceiveSource.REUSE_LOCAL, 0, i + 1, 1, None)
                for i in range(4)
            ),
        )
        m.received(r, txn, 16, 12.0)
    m.first_token(r, 13.0)
    assert m.counters["prompt_tokens_ssd"] == expected
    assert m.counters["prompt_tokens_host_reuse"] == 64 - expected
    assert m.counters["prompt_tokens_compute"] == 36
    # Another request benefits from warm RAM, with no SSD lineage.
    other = request()
    other.request_id, other.block_hashes = "other", r.block_hashes
    m.prefetched(other, ssd, contract, 64)
    m.admit(other, 64, 0, True, 14.0)
    m.first_token(other, 15.0)
    assert m.counters["prompt_tokens_ssd"] == expected
    assert m.counters["prompt_tokens_host_reuse"] == 128 - expected
    # A later attempt cannot inherit a consumed round's lineage.
    m.begin_lookup(r)
    assert not m.state(r).ssd_intervals


@pytest.mark.parametrize("restart,keep_pages", [(False, 0), (False, 3), (True, 0)])
def test_unit_verified_reads_export_once_and_preserve_payload(
    tmp_path, restart, keep_pages
):
    env = UnitEnv(tmp_path)
    try:
        unit = env.variant()
        env.persist(unit)
        expected = {
            k: bytes(
                env.backing[
                    h.page_id * env.ssd.page_bytes : (h.page_id + 1)
                    * env.ssd.page_bytes
                ]
            )
            for k in unit.keys
            if (h := env.table.lookup(k)) is not None
        }
        if restart:
            uid = unit.uid
            env.ssd.close(io_grace_s=5)
            env = UnitEnv(tmp_path)
            unit = env.manager.units.units[uid]
        else:
            for k in unit.keys[keep_pages:]:
                env.table._recycle(env.table.get(env.table.lookup(k)))
        m = StoreMetrics()
        before = m.export(1.0, ssd=env.ssd, manager=env.manager)
        assert not before["counters"]["host_ssd_verified_load_bytes"]
        assert env.ssd.prefetch("reader", [unit.keys[0]])
        assert env.settle(lambda: not env.ssd.waiting("reader"))
        pages = len(unit.keys) - keep_pages
        after = m.export(61.0, ssd=env.ssd, manager=env.manager)
        for prefix in ("host_ssd", "unified"):
            assert (
                after["counters"][f"{prefix}_verified_load_bytes"]
                == pages * env.ssd.page_bytes
            )
            assert after["counters"].get(f"{prefix}_verified_load_pages", 0) == pages
        for k, value in expected.items():
            h = env.table.lookup(k)
            assert (
                bytes(
                    env.backing[
                        h.page_id * env.ssd.page_bytes : (h.page_id + 1)
                        * env.ssd.page_bytes
                    ]
                )
                == value
            )
        again = m.export(121.0, ssd=env.ssd, manager=env.manager)
        assert not again["counters"]["host_ssd_verified_load_bytes"]
        assert not again["counters"]["unified_verified_load_bytes"]
        env.ssd.cancel("reader")
    finally:
        env.ssd.close(io_grace_s=5)


def test_connector_captures_lineage_before_lookup_cancels_round():
    from tests.v1.core.long_context_kv_utils import SPAN
    from tests.v1.kv_connector.unit.offloading_connector.test_host_receive_txn import (
        lookup_allocate,
        setup_receive,
    )

    scheduler, connector, r, bindings = setup_receive(page_state="ready")
    table = scheduler.host_history_pool.host_pages
    h = next(iter(bindings.values()))[1]
    held = [h]
    ssd = SimpleNamespace(
        table=table,
        enabled=False,
        holds=lambda req_id: held.copy() if req_id == r.request_id else [],
        cancel=lambda req_id: held.clear(),
    )
    connector.host_page_ssd = ssd
    m = connector.store_metrics = StoreMetrics()
    m.begin_lookup(r)
    _, txn = lookup_allocate(scheduler, connector, r)
    assert not held
    m.admit(r, 0, 4 * SPAN, True, 11.0)
    m.received(r, txn, SPAN, 12.0)
    m.first_token(r, 13.0)
    assert m.counters["prompt_tokens_ssd"] == SPAN
    assert m.counters["prompt_tokens_host_reuse"] == 3 * SPAN


def test_unit_waiters_only_inherit_reads_pending_at_their_join(tmp_path):
    from tests.v1.kv_offload.test_host_page_ssd import Clock

    clock = Clock()
    env = UnitEnv(
        tmp_path, clock=clock, prefetch_max_per_request=1, prefetch_max_inflight=1
    )
    try:
        unit = env.variant()
        env.persist(unit)
        env.evict_cpu(unit)
        assert env.ssd.prefetch("first", [unit.keys[0]])
        txn = env.store.prefetches[unit.uid]
        assert env.settle(
            lambda: any(env.table.is_ready(h) for h in txn.handles.values())
        )
        # Join a still-running transaction after at least one read completed.
        ready_before = {h for h in txn.handles.values() if env.table.is_ready(h)}
        assert ready_before and not txn.complete
        assert env.ssd.prefetch("second", [unit.keys[0]])
        assert env.settle(lambda: txn.complete)
        assert set(env.ssd.holds("first")) == set(txn.handles.values())
        assert set(env.ssd.holds("second")) == set(txn.handles.values()) - ready_before
        # Joining a completed transaction is warm reuse, including an ID
        # which already cancelled its earlier lookup episode.
        env.ssd.cancel("second")
        assert env.ssd.prefetch("second", [unit.keys[0]])
        assert not env.ssd.holds("second")
        assert env.ssd.prefetch("third", [unit.keys[0]])
        assert not env.ssd.holds("third")
        env.ssd.cancel("first")
        env.ssd.cancel("second")
        env.ssd.cancel("third")
    finally:
        env.ssd.close(io_grace_s=5)
