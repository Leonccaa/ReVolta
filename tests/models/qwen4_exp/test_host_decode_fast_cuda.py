# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GPU gates for the candidate, run by the GPU window owner."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def test_full_graph_fast_refill_matches_original_bits(monkeypatch):
    from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan
    from vllm.models.qwen4_exp.nvidia.host_kv.pool import LayerScales

    from .test_host_kv_history_cuda import DIM, QHEADS, SCALES, WIDTH, Arms

    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST", "0")
    baseline = Arms(True, rows_per_pass=2)
    monkeypatch.setenv("REVOLTA_RESEARCH_HOST_DECODE_FAST", "1")
    candidate = Arms(True, rows_per_pass=2)
    lengths = [4100, 4000]
    for req, length in enumerate(lengths):
        key = torch.randn((length, 1, DIM), device="cuda").half()
        value = torch.randn_like(key)
        for arm in (baseline, candidate):
            arm.write(key, value, arm.slots(req, list(range(length))))

    key = torch.zeros((2, 1, DIM), device="cuda", dtype=torch.float16)
    value = torch.zeros_like(key)
    slots = baseline.slots(0, [lengths[0]]).repeat(2)
    selection = torch.empty((2, WIDTH), dtype=torch.int32, device="cuda")
    requests = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
    lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    query = torch.randn((2, QHEADS, DIM), device="cuda").half()
    plans = [
        SparseAttentionPlan(
            rows=2,
            query_heads=QHEADS,
            kv_heads=1,
            head_dim=DIM,
            selection_width=WIDTH,
            device="cuda",
            e4m3=True,
            scales=LayerScales(*SCALES),
        )
        for _ in range(2)
    ]

    def load(step):
        key.normal_()
        value.normal_()
        lens.copy_(torch.tensor([n + step + 1 for n in lengths]))
        slots.copy_(
            torch.cat([baseline.slots(r, [n + step]) for r, n in enumerate(lengths)])
        )
        selection.copy_(
            torch.tensor(
                [baseline.selection(n + step) for n in lengths], dtype=torch.int32
            )
        )

    def body(arm, plan):
        arm.layer.write(
            key, value, slots, arm.kv, arm.k_scale, arm.v_scale, decode=True
        )
        ids = arm.layer.gather(selection, requests, arm.table, lens, decode=True)
        plan.run(
            query,
            arm.layer.key_cache,
            arm.layer.value_cache,
            ids,
            arm.store.hot_block_table,
            requests,
        )

    load(0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body(baseline, plans[0])
        body(candidate, plans[1])
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body(candidate, plans[1])
    for step in range(1, 6):
        load(step)
        body(baseline, plans[0])
        graph.replay()
        torch.accelerator.synchronize()
        assert torch.equal(
            plans[0].output.view(torch.uint8), plans[1].output.view(torch.uint8)
        )
        assert torch.equal(
            plans[0].lse.view(torch.uint8), plans[1].lse.view(torch.uint8)
        )
        assert torch.equal(baseline.store.ready, candidate.store.ready)
        assert torch.equal(baseline.layer.host_view, candidate.layer.host_view)


@pytest.mark.parametrize("rows", [2, 5, 8])
def test_sharded_shared_key_fused_scores_match_global(monkeypatch, rows):
    from .test_qsa_index_dcp_shard import test_shared_key_scores_sharded_match_global

    monkeypatch.setenv("REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED", "1")
    test_shared_key_scores_sharded_match_global(world=2, rows=rows)
