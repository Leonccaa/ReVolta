# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Independent review oracle for varied groups, windows and all token offsets."""

import random
from dataclasses import replace

import pytest

from tests.v1.kv_connector.unit.offloading_connector.test_cpu_draft_offload import (
    _hashes,
    _kv_config,
    _request,
    _vllm_config,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import MockOffloadingSpec
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
    RequestOffloadState,
)
from vllm.v1.kv_cache_interface import KVCacheGroupSpec
from vllm.v1.kv_offload.base import ReqContext, RequestOffloadingContext
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupedCPUOffloadingManager,
)

pytestmark = pytest.mark.cpu_test


def enumerate_hits(s, paths):
    results, useful = [], set()
    for hashes in paths:
        for end in range(33, 130):
            req = _request("oracle", end, hashes)
            rs = RequestOffloadState(
                s.config, req, ReqContext("oracle"), RequestOffloadingContext()
            )
            rs.update_offload_keys()
            for local in range(end):
                rs.num_locally_computed_tokens = local
                hit = s._lookup(rs)
                results.append(hit)
                if hit:
                    for g in s.config.kv_group_configs:
                        keys = rs.group_states[g.group_idx].offload_keys[
                            local // g.offloaded_block_size : (
                                local + hit + g.offloaded_block_size - 1
                            )
                            // g.offloaded_block_size
                        ]
                        if g.sliding_window_size_in_blocks is not None:
                            keys = keys[-g.sliding_window_size_in_blocks :]
                        useful.update(keys)
    return results, useful


@pytest.mark.parametrize("seed", range(24))
def test_multigroup_all_offsets_oracle(seed):
    rng = random.Random(93001 + seed)
    config = _vllm_config(("all", "window", "none")[seed % 3])
    config.kv_transfer_config.kv_connector_extra_config[
        "cpu.enable_dependency_eviction"
    ] = True
    config.speculative_config.use_eagle_kv_cache = lambda: bool(seed % 2)
    kv = _kv_config()
    # Two target groups, two state peers, two heterogeneous draft windows.
    kv.kv_cache_groups.extend(
        [
            KVCacheGroupSpec(
                ["model.layers.2.self_attn.attn"], kv.kv_cache_groups[0].kv_cache_spec
            ),
            KVCacheGroupSpec(
                ["model.layers.3.linear_attn"], kv.kv_cache_groups[1].kv_cache_spec
            ),
            KVCacheGroupSpec(
                ["model.layers.5.self_attn.attn"],
                replace(
                    kv.kv_cache_groups[2].kv_cache_spec,
                    block_size=(8, 16, 32)[seed % 3],
                    sliding_window=(17, 65, 95)[seed % 3],
                ),
            ),
        ]
    )
    spec = MockOffloadingSpec(config, kv)
    pools = [
        CPUOffloadingManager(512, enable_events=True)
        for _ in range(6 if seed % 2 else 1)
    ]
    spec.manager = (
        GroupedCPUOffloadingManager(dict(enumerate(pools)))
        if len(pools) > 1
        else pools[0]
    )
    s = OffloadingConnectorScheduler(spec)
    assert s._dependency_index.supported
    paths = []
    for branch in range(2):
        # Common chained prefix, disjoint suffix; enough hashes for smallest page.
        hashes = _hashes(8, "common") + _hashes(24, str(branch))
        paths.append(hashes)
        req = _request(str(branch), 129, hashes)
        s.on_new_request(req)
        rs = s._req_status[str(branch)]
        rs.update_offload_keys()
        offered = [
            k
            for g in s.config.kv_group_configs
            for k in rs.group_states[g.group_idx].offload_keys
            if not (seed % 3 == 2 and g.draft_window_tokens is not None)
        ]
        out = s._prepare_store(offered, rs)
        s.manager.complete_store(out.keys_to_store, rs.req_context)
        req.is_finished.return_value = True
        s.request_finished(req)
    idx = s._dependency_index
    for k in list(idx.records):
        if rng.random() < 0.22:
            idx.owners[k].remove_dependency_blocks([k])
    protected = set()
    for k in idx.records:
        if rng.random() < 0.08:
            protected.add(k)
        if rng.random() < 0.04:
            idx.owners[k].prepare_load([k], ReqContext("pin"))
    before, useful = enumerate_hits(s, paths)
    order = [p.resident_keys() for p in pools]
    removed = set(idx.evict(protected))
    assert removed.isdisjoint(useful)
    assert removed.isdisjoint(protected)
    assert enumerate_hits(s, paths)[0] == before
    assert [p.resident_keys() for p in pools] == [
        [k for k in row if k not in removed] for row in order
    ]
    assert (
        idx.records.keys()
        == idx.owners.keys()
        == {k for p in pools for k in p.resident_keys()}
    )
    assert sum(idx.totals(protected).values()) == len(idx.records)
    print(
        "ORACLE",
        seed,
        "queries",
        len(before),
        "removed",
        len(removed),
        "useful",
        len(useful),
    )
