# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""page ownership physical ownership, late receipts and retirement without CUDA."""

from dataclasses import replace

import pytest

from vllm.v1.kv_offload.base import make_offload_key
from vllm.v1.kv_offload.host.page_table import HostPageTable, PageState, Result


def key(n):
    return make_offload_key(bytes([n]) * 32, 0)


def private(table, hid=1, hgen=1):
    handle = table.reserve()
    assert handle is not None
    table.bind(handle, hid, hgen)
    return handle


def ready(table, handle, hid=1, hgen=1, step=1, k=None):
    ticket = table.open_seal(handle, hid, hgen, key(1) if k is None else k, step)
    assert table.seal_result(ticket, table.all_ranks, 0)
    return ticket


def test_rank_dedup_and_gap_free_watermark():
    table = HostPageTable(2, 2)
    handle = private(table)
    ticket = table.open_seal(handle, 1, 1, key(1), 1)
    assert not table.seal_result(ticket, 1, 0)
    assert not table.seal_result(ticket, 1, 0)
    assert table.lookup(key(1)) is None
    table.complete_step(3, 3)
    table.complete_step(1, 1)
    table.complete_step(1, 1)
    assert table.watermark == 0
    assert table.seal_result(ticket, 2, 0)
    table.complete_step(1, 2)
    assert table.watermark == 1
    table.complete_step(2, 3)
    assert table.watermark == 3
    assert table.lookup(key(1)) == handle


@pytest.mark.parametrize(
    "state", [PageState.PRIVATE, PageState.READY, PageState.SHADOW]
)
def test_s1_write_s3_read_cancel_waits_for_reader_retirement(state):
    table = HostPageTable(2)
    handle = private(table)
    if state == PageState.READY:
        ready(table, handle)
    else:
        table.get(handle).state = state
    table.complete_step(1, 1)
    table.complete_step(2, 1)
    table.unbind(handle, 1, 1, last_use=3)
    assert table.available() == 0
    assert table.reserve() is None
    assert table.reserve(loading_key=key(2)) is None
    assert table.get(handle).retirements[0].retire_after == 3
    table.complete_step(3, 1)
    replacement = table.reserve(loading_key=key(2))
    assert replacement.page_id == handle.page_id
    assert replacement.epoch == handle.epoch + 1


def test_hid_reuse_preserves_old_page_retirement_and_epoch_rejects_late_seal():
    table = HostPageTable(3)
    old = private(table, 7, 1)
    ticket = ready(table, old, 7, 1)
    table.complete_step(1, 1)
    table.unbind(old, 7, 1, last_use=3)
    new = private(table, 7, 2)
    table.unbind(new, 7, 2, last_use=1)
    assert table.get(old).retirements[0].hgen == 1
    assert table.get(old).retirements[0].retire_after == 3
    assert not table.seal_result(replace(ticket, handle=new), 1, 0)
    table.complete_step(2, 1)
    table.complete_step(3, 1)
    table.reserve()  # Eviction can now recycle old.
    assert not table.seal_result(ticket, 1, 0)


@pytest.mark.parametrize("canonical_loading", [False, True])
def test_shadow_is_never_promoted(canonical_loading):
    table = HostPageTable(3)
    if canonical_loading:
        canonical = table.reserve(loading_key=key(1))
        table.start_io(canonical, 99)
    else:
        canonical = private(table)
        ready(table, canonical)
    shadow = private(table, 2)
    ready(table, shadow, hid=2, k=key(1))
    assert table.get(shadow).state == PageState.SHADOW
    assert table.lookup(key(1)) == canonical
    if canonical_loading:
        table.invalidate(canonical)
        table.finish_io(canonical, 99)
    else:
        table.unbind(canonical, 1, 1, 1)
        table.complete_step(1, 1)
        table.reserve()
    assert table.lookup(key(1)) is None
    assert table.get(shadow).state == PageState.SHADOW


@pytest.mark.parametrize("compute_failed", [0, 2])
def test_compute_and_seal_results_are_independent(compute_failed):
    table = HostPageTable(2, 2)
    handle = private(table)
    ticket = table.open_seal(handle, 1, 1, key(1), 1)
    assert not table.seal_result(ticket, 2, 2, compute_failed)
    assert table.seal_result(ticket, 1, 0)
    page = table.get(handle)
    assert page.state == PageState.PRIVATE
    assert page.compute_result == (Result.FAILED if compute_failed else Result.OK)
    assert page.seal_result == Result.FAILED
    assert table.lookup(key(1)) is None


def test_reset_removes_visibility_before_owners_and_rejects_old_publication():
    table = HostPageTable(2, 2)
    handle = private(table)
    ticket = table.open_seal(handle, 1, 1, key(1), 1)
    table.start_io(handle, 99)
    table.reset()
    assert table.ssd_epoch == 1
    assert table.lookup(key(1)) is None
    assert not table.get(handle).binds
    assert table.reserve() is None
    assert table.seal_result(ticket, 3, 0)
    table.complete_step(1, 3)
    assert table.reserve() is None
    table.finish_io(handle, 99)
    new = table.reserve()
    assert new.epoch == handle.epoch + 1
    assert not table.seal_result(ticket, 3, 0)


def test_lease_io_and_binding_owners_survive_publication():
    table = HostPageTable(2)
    handle = private(table)
    table.pin(handle, 11)
    table.start_io(handle, 12)
    ready(table, handle)
    page = table.get(handle)
    assert page.binds == {(1, 1): "inuse"}
    assert page.writer == (1, 1, 1)
    assert page.leases == {11} and page.io == {12}
    table.unbind(handle, 1, 1, 1)
    table.complete_step(1, 1)
    table.release(handle, 11)
    assert table.reserve() is None
    table.finish_io(handle, 12)
    assert table.reserve() is not None


def test_cancelled_loading_waits_for_io_and_step():
    table = HostPageTable(2)
    handle = table.reserve(loading_key=key(1))
    table.start_io(handle, 99)
    table.touch(handle, 3)
    table.reset()
    table.finish_io(handle, 99)
    assert table.reserve(loading_key=key(2)) is None
    for step in (3, 1, 2):
        table.complete_step(step, 1)
    assert table.reserve(loading_key=key(2)) is not None


def test_live_loading_cannot_be_recycled_before_owner_is_installed():
    table = HostPageTable(2)
    handle = table.reserve(loading_key=key(1))
    assert not table.reclaimable(table.get(handle))
    assert table.reserve() is None
    table.start_io(handle, 9)
    table.reset()
    assert table.reserve() is None
    table.finish_io(handle, 9)
    replacement = table.reserve()
    assert replacement.page_id == handle.page_id
    assert replacement.epoch == handle.epoch + 1


def test_seal_only_retry_does_not_inherit_previous_ticket_masks():
    table = HostPageTable(2, 2)
    handle = private(table)
    old = table.open_seal(handle, 1, 1, key(1), 1)
    assert table.seal_result(old, 3, 2)
    retry = table.open_seal(handle, 1, 1, key(1), 2)
    assert not table.seal_result(old, 3, 0)
    assert not table.seal_result(retry, 1, 0)
    assert table.lookup(key(1)) is None
    assert table.get(handle).failed_ranks == 0
    assert table.seal_result(retry, 2, 0)
    assert table.is_ready(handle)


def test_zero_page_is_permanent_and_bad_receipts_are_rejected():
    table = HostPageTable(1)
    assert table.available() == 0 and table.reserve() is None
    with pytest.raises(ValueError, match="Unknown rank"):
        table.complete_step(1, 2)
    table = HostPageTable(2)
    ticket = table.open_seal(private(table), 1, 1, key(1), 1)
    with pytest.raises(ValueError, match="Failure must"):
        table.seal_result(ticket, 0, 1)


def test_pool_adapter_shares_the_existing_physical_owner():
    from vllm.v1.kv_offload.host.pool import HostKVPool

    table = HostPageTable(2)
    view = memoryview(bytearray(2 * 16)).cast("B", shape=(2, 16))
    pool = HostKVPool(view, group_idx=0, block_tokens=16, table=table)
    writer = pool.reserve(key(1))
    assert writer is not None and writer.handle.page_id == 1
    table.bind(writer.handle, 7, 1)
    pool.publish(writer)
    pool.release(writer)
    assert table.reserve() is None
    assert table.get(writer.handle).binds == {(7, 1): "inuse"}
    with pytest.raises(ValueError, match="differ in size"):
        HostKVPool(view, group_idx=0, block_tokens=16, table=HostPageTable(3))


def test_publication_requires_both_producer_results():
    table = HostPageTable(2)
    handle = private(table)
    with pytest.raises(ValueError, match="successful compute and seal"):
        table.publish(handle, key(1))
    table.producer_result(handle, compute=Result.OK, seal=Result.PENDING)
    with pytest.raises(ValueError, match="successful compute and seal"):
        table.publish(handle, key(1))
    assert table.lookup(key(1)) is None
