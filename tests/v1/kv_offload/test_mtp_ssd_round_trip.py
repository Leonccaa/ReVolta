# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cold restore of the planned MTP host row across two real processes."""

import subprocess
import sys
from pathlib import Path

import pytest
import torch

from tests.v1.core.long_context_kv_utils import mtp_specs
from tests.v1.core.test_long_context_mtp import DRAFT, build, settings
from tests.v1.kv_offload import test_unified_ssd_transactions as transactions
from vllm.v1.attention.kv_cache_view import pack_page_payload, scatter_page_payload
from vllm.v1.kv_offload.base import ReqContext
from vllm.v1.kv_offload.host.page_table import PageKind
from vllm.v1.kv_offload.host.region import host_region_layout
from vllm.v1.kv_offload.host.ssd import HostKVLayout

pytestmark = pytest.mark.cpu_test


def _cold_round_trip(root: Path, mode: str):
    torch.set_num_threads(1)
    specs = mtp_specs()
    _, config = build(settings(), specs)
    region = host_region_layout(config, "round-trip")
    group = config.kv_cache_groups[config.host_kv_group_id]
    members = [n for n in group.layer_names if n not in config.host_kv_layers]
    assert DRAFT in members and f"{DRAFT}.compressed" in members
    assert region.index_bytes == tuple(specs[n].page_size_bytes for n in members)
    sizes = (region.main_bytes, *region.index_bytes)
    layout = HostKVLayout(
        version="mtp-host-row-test-v1",
        storage_layout=f"row=[main|{tuple(zip(members, region.index_bytes))}]",
        encoding="raw-bytes",
        layer_names=tuple(config.host_kv_layers) + tuple(members),
        kv_scales=((1.0, 1.0),) * (len(config.host_kv_layers) + len(members)),
    )
    # Keep UnitEnv's real file queue and unit cold-rebuild path; replace only
    # its small row ABI with the planned target + MTP row for one worker.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transactions, "ROW", region.worker_bytes)
        patch.setattr(transactions, "LAYOUT", layout)
        patch.setattr(
            transactions, "payload", lambda n: bytes([n]) * region.worker_bytes
        )
        env = transactions.UnitEnv(root / "ssd", pages=16, max_size_bytes=512 << 20)
        try:
            if mode == "write":
                unit = env.variant(boundary=1, ahead=2)
                expected = {}
                for row_index, key in enumerate(unit.keys):
                    if env.table.kind_for_key(key) != PageKind.MAIN:
                        continue
                    source = []
                    packed = []
                    for member_index, size in enumerate(sizes):
                        # Two logical subpages with padding between them,
                        # like the kernel-page views used by seal/receive.
                        view = torch.empty_strided(
                            (2, size // 2), (size, 1), dtype=torch.uint8
                        )
                        values = torch.arange(size).to(torch.uint8)
                        view.copy_(
                            (values + row_index + member_index).reshape(view.shape)
                        )
                        source.append(view.clone())
                        packed.append(pack_page_payload(view, 0, subdivisions=2))
                    expected[bytes(key)] = source
                    row = torch.cat(packed)
                    handle = env.table.lookup(key)
                    start = handle.page_id * region.worker_bytes
                    env.backing[start : start + region.worker_bytes] = (
                        row.numpy().tobytes()
                    )
                torch.save(expected, root / "source.pt")
                env.persist(unit)
            else:
                assert len(env.store.manifests) == 1
                unit = next(u for u in env.manager.units.units.values() if u.parent)
                expected = torch.load(root / "source.pt", weights_only=True)
                assert all(env.table.lookup(k) is None for k in unit.keys)
                assert env.manager.units.cold_recoverable(unit.uid)
                assert env.ssd.prefetch("cold", unit.keys)
                assert env.settle(lambda: not env.ssd.waiting("cold"))
                assert all(
                    env.manager.lookup(k, ReqContext("cold")) is True for k in unit.keys
                )
                for key, source in expected.items():
                    handle = env.table.lookup(key)
                    start = handle.page_id * region.worker_bytes
                    # Compare the actual final host row and scatter its bytes
                    # into fresh kernel-page views, never a generated payload.
                    final_host = torch.frombuffer(env.backing, dtype=torch.uint8)[
                        start : start + region.worker_bytes
                    ]
                    offset = 0
                    for original, size in zip(source, sizes):
                        restored = torch.empty_strided(
                            original.shape, (size, 1), dtype=torch.uint8
                        )
                        scatter_page_payload(
                            final_host[offset : offset + size],
                            restored,
                            0,
                            subdivisions=2,
                        )
                        assert torch.equal(restored, original)
                        changed = original.clone()
                        changed[0, 0] ^= 1
                        with pytest.raises(AssertionError):
                            assert torch.equal(restored, changed)
                        offset += size
                    assert offset == region.worker_bytes
                env.ssd.cancel("cold")
                env.evict_cpu(unit)
                missing = next(iter(expected))
                Path(env.store.path(missing)).unlink()
                assert env.ssd.prefetch("missing", unit.keys) == []
                assert env.manager.lookup(missing, ReqContext("missing")) is False
                assert not env.manager.units.cold_recoverable(unit.uid)
        finally:
            env.ssd.close(io_grace_s=5)


def test_mtp_host_row_cold_restore_in_two_processes(tmp_path):
    script = """
import sys
from pathlib import Path
import vllm.platforms
from vllm.platforms.cpu import CpuPlatform
vllm.platforms._current_platform = CpuPlatform()
from tests.v1.kv_offload.test_mtp_ssd_round_trip import _cold_round_trip
_cold_round_trip(Path(sys.argv[1]), sys.argv[2])
"""
    for mode in ("write", "read"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout + result.stderr
