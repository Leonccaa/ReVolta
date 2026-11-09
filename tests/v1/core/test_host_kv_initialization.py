# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The parent owns the backing name before workers can create it or warm up."""

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest


@pytest.mark.parametrize("workers_stop", [False, True])
def test_warmup_failure_retains_owner_until_workers_stop(monkeypatch, workers_stop):
    from vllm.v1.engine import core
    from vllm.v1.kv_offload.host.region import HostRegionLayout

    layout = HostRegionLayout(uuid4().hex, 4, 1, 128, (32,))
    config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16),
        cache_config=SimpleNamespace(),
        instance_id=layout.instance_id,
        validate_block_size=lambda: None,
    )
    kv = SimpleNamespace(
        num_blocks=4,
        kv_cache_groups=[],
        host_kv_layers=["l"],
        long_context_plan=object(),
    )
    monkeypatch.setattr(core, "get_kv_cache_configs", lambda *_: [kv])
    monkeypatch.setattr(core, "generate_scheduler_kv_cache_config", lambda _: kv)
    monkeypatch.setattr(
        "vllm.v1.kv_offload.host.region.host_region_layout", lambda *_: layout
    )
    engine = core.EngineCore.__new__(core.EngineCore)
    engine.available_gpu_memory_for_kv_cache = -1
    events = []
    worker_regions = []

    def initialize(_):
        worker_regions.append(layout.open(0))
        raise RuntimeError("injected warmup failure")

    def shutdown():
        events.append("worker-shutdown")
        if workers_stop:
            for region in worker_regions:
                region.cleanup()

    engine.model_executor = SimpleNamespace(
        get_kv_cache_specs=lambda: [{"l": object()}],
        determine_available_memory=lambda: [1024],
        initialize_from_config=initialize,
        shutdown=shutdown,
        workers=[
            SimpleNamespace(proc=SimpleNamespace(is_alive=lambda: not workers_stop))
        ],
    )
    try:
        with pytest.raises(RuntimeError, match="injected warmup failure"):
            engine._initialize_kv_caches(config)
        assert events == ["worker-shutdown"]
        owner = engine._host_region_owner
        assert owner is not None
        assert Path(owner.mmap_path).exists() == (not workers_stop)
    finally:
        for region in worker_regions:
            region.cleanup()
        # Only the random, tiny file created by this test is removed here.
        Path(f"/dev/shm/vllm_offload_host_v2_{layout.instance_id}.mmap").unlink(
            missing_ok=True
        )
