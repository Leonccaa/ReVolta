# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Successful verified SSD loads are observable separately from external hits."""

from unittest.mock import MagicMock, patch

from vllm.v1.kv_offload.tiering.manager import TieringOffloadingManager


def test_verified_reads_report_exact_page_bytes_without_changing_counters():
    tier = MagicMock()
    tier.tier_type = "revolta_fs"
    tier._block_size = 4096
    stats = {"pages_verified": 3, "checksum_failures": 2}
    tier.get_revolta_stats.return_value = stats
    queue = MagicMock()
    queue.take_due.return_value = []
    queue.summary.return_value = {"depth": 4}
    manager = object.__new__(TieringOffloadingManager)
    manager._delayed = {tier: queue}
    manager._delayed_logged_at = 0
    with (
        patch("vllm.v1.kv_offload.tiering.manager.time.monotonic", return_value=61),
        patch("vllm.v1.kv_offload.tiering.manager.logger.info") as log,
    ):
        manager.flush_delayed_writes()
    assert log.call_args.args[1:] == ("revolta_fs", 3, 12288, stats)
    assert stats == {"pages_verified": 3, "checksum_failures": 2}
    tier.submit_store.assert_not_called()
    assert manager._delayed_logged_at == 61
