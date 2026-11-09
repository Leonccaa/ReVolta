# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hardware gates: native writer, UVA, original attention, changed-input graphs.

These tests deliberately do not run through Triton's interpreter. A passing
CPU suite cannot qualify these gates or production TP4/DCP2 behavior.
"""

import os

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv import (
    ActiveHostKV,
    HostKVConfig,
    LayerScales,
)
from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or os.environ.get("TRITON_INTERPRET") == "1",
    reason="Requires real CUDA/UVA and compiled Triton kernels",
)


def _setup(e4m3, *, rows=4, heads=12, padded=True, width=2051, kv_heads=1):
    from vllm._custom_ops import reshape_and_cache_flash

    r, t, h, d, p = 2, 4096, kv_heads, 256, 32
    config = HostKVConfig(
        r, t, rows, width, t, hot_groups=min(r * (t // 4), rows * width + r * 9)
    )
    dtype = torch.uint8 if e4m3 else torch.float16
    shape = (r * t // p, 2, p, h, d)
    strides = (p * h * d * 4, p * h * d, d, p * d, 1)
    layout = (
        torch.empty_strided(shape, strides, dtype=dtype, device="meta")
        if padded
        else torch.empty(shape, dtype=dtype, device="meta")
    )
    table = torch.arange(shape[0] - 1, -1, -1, dtype=torch.int32).reshape(r, -1)
    scales = LayerScales(0.375, 1.75)
    pool = ActiveHostKV(config, layout, device="cuda", scales=scales, block_table=table)
    source = torch.empty_strided(
        layout.shape, layout.stride(), dtype=dtype, device=pool.device
    )
    generator = torch.Generator().manual_seed(120451)
    positions = torch.arange(t, dtype=torch.int32, device=pool.device)
    source_k, source_v = source.unbind(1)
    for req in range(r):
        key = torch.randn((t, h, d), generator=generator).half().to(pool.device)
        value = torch.randn(key.shape, generator=generator).half().to(pool.device)
        requests = torch.full((t,), req, dtype=torch.int32, device=pool.device)
        pool.write(key, value, requests, positions)
        slots = (
            pool.block_table[req, positions.long() // p].long() * p
            + positions.long() % p
        )
        reshape_and_cache_flash(
            key,
            value,
            source_k,
            source_v,
            slots,
            "fp8_e4m3" if e4m3 else "auto",
            pool.k_scale,
            pool.v_scale,
        )
    selection = torch.full((rows, width), -1, dtype=torch.int32)
    for row in range(rows):
        # QSA's expanded four-token groups plus a three-column causal tail.
        n = min(width // 4, 512)
        groups = torch.randperm(t // 4, generator=generator)[:n]
        expanded = (groups[:, None] * 4 + torch.arange(4)).flatten().int()
        selection[row, : expanded.numel()] = expanded
        if width >= 3:
            selection[row, -3:] = torch.tensor([1, 1, -1])  # Duplicate and padding.
    if rows > 1:
        selection[-1].fill_(-1)
    selection = selection.to(pool.device)
    requests = torch.arange(rows, dtype=torch.int32, device=pool.device) % r
    lengths = torch.full((r,), t, dtype=torch.int32, device=pool.device)
    qpos = torch.full((rows,), t - 1, dtype=torch.int64, device=pool.device)
    query = torch.randn((rows, heads, d), generator=generator).half().to(pool.device)
    plan = SparseAttentionPlan(
        rows=rows,
        query_heads=heads,
        kv_heads=h,
        head_dim=d,
        selection_width=width,
        device=pool.device,
        e4m3=e4m3,
        scales=scales,
    )
    return pool, source, query, selection, requests, lengths, qpos, plan


def _assert_original(pool, source, query, selection, requests, plan, gate=None):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    key, value = source.unbind(1)
    original = torch.empty_like(query)
    original_lse = torch.empty(
        query.shape[:2], dtype=torch.float32, device=query.device
    )
    kwargs = dict(
        kv_cache_dtype="fp8_e4m3" if pool.is_e4m3 else "auto",
        k_scale=pool.scales.key,
        v_scale=pool.scales.value,
    )
    # No query-position/XQA dispatch: the selected IDs are already causal.
    qsa_sparse_paged_attention(
        query,
        key,
        value,
        selection,
        pool.block_table,
        requests,
        out=original,
        output_gate=gate,
        **kwargs,
    )
    # Independently check LSE without gating, as required by the public ABI.
    qsa_sparse_paged_attention(
        query,
        key,
        value,
        selection,
        pool.block_table,
        requests,
        lse=original_lse,
        **kwargs,
    )
    adapter = plan.run(
        query, key, value, selection, pool.block_table, requests, output_gate=gate
    ).clone()
    assert torch.equal(original.view(torch.uint8), adapter.view(torch.uint8))
    assert torch.equal(original_lse.view(torch.uint8), plan.lse.view(torch.uint8))
    return original, original_lse


@pytest.mark.parametrize("e4m3", [False, True])
@pytest.mark.parametrize(
    "rows,heads,kv_heads", [(1, 6, 1), (4, 12, 1), (8, 12, 1), (4, 12, 2)]
)
@pytest.mark.parametrize("padded", [False, True])
def test_native_writer_uva_and_original_attention_bitwise(
    e4m3, rows, heads, kv_heads, padded
):
    pool, source, query, selection, requests, lengths, qpos, plan = _setup(
        e4m3, rows=rows, heads=heads, padded=padded, kv_heads=kv_heads
    )
    assert pool.host.is_pinned()
    assert pool.host_device.stride() == pool.host.stride()
    assert pool.host_device.is_cuda
    torch.accelerator.synchronize()
    assert torch.equal(source.cpu().view(torch.uint8), pool.host.view(torch.uint8))
    gate = (
        torch.linspace(-2, 2, query.numel(), device=query.device)
        .reshape_as(query)
        .half()
    )
    expected, expected_lse = _assert_original(
        pool, source, query, selection, requests, plan, gate
    )
    mapped = pool.gather(selection, requests, lengths, qpos)
    hk, hv = pool.hot.unbind(1)
    actual = plan.run(
        query, hk, hv, mapped, pool.hot_block_table, requests, output_gate=gate
    )
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(expected_lse.view(torch.uint8), plan.lse.view(torch.uint8))
    pool.warm(selection, requests, lengths, qpos)
    assert int(pool.miss_count) == 0


@pytest.mark.parametrize("e4m3", [False, True])
def test_fp32_dcp_local_output_and_lse_are_bitwise_equal(e4m3):
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    pool, source, query, selection, requests, lengths, qpos, _ = _setup(e4m3)
    plan = SparseAttentionPlan(
        rows=4,
        query_heads=12,
        kv_heads=1,
        head_dim=256,
        selection_width=2051,
        device=pool.device,
        e4m3=e4m3,
        scales=pool.scales,
        output_dtype=torch.float32,
    )
    expected = torch.empty_like(plan.output)
    expected_lse = torch.empty_like(plan.lse)
    key, value = source.unbind(1)
    qsa_sparse_paged_attention(
        query,
        key,
        value,
        selection,
        pool.block_table,
        requests,
        out=expected,
        lse=expected_lse,
        kv_cache_dtype="fp8_e4m3" if e4m3 else "auto",
        k_scale=pool.scales.key,
        v_scale=pool.scales.value,
    )
    mapped = pool.gather(selection, requests, lengths, qpos)
    key, value = pool.hot.unbind(1)
    actual = plan.run(query, key, value, mapped, pool.hot_block_table, requests)
    assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
    assert torch.equal(expected_lse.view(torch.uint8), plan.lse.view(torch.uint8))


@pytest.mark.parametrize("e4m3", [False, True])
def test_changed_input_graph_replay_includes_append_write_gather_and_attention(e4m3):
    from vllm._custom_ops import reshape_and_cache_flash

    pool, source, query, selection, requests, lengths, qpos, plan = _setup(
        e4m3, rows=1, heads=6, width=17
    )
    # A small hot cache forces eviction across replays. Truncate a suffix so
    # each replay writes new prefill rows, including an already resident group.
    lengths[0] = 4093
    pool.truncate(lengths)
    key = torch.ones((3, 1, 256), dtype=torch.float16, device=pool.device)
    value = -key
    write_requests = torch.zeros(3, dtype=torch.int32, device=pool.device)
    positions = torch.arange(4093, 4096, dtype=torch.int32, device=pool.device)
    hk, hv = pool.hot.unbind(1)

    def replay_body():
        pool.write(key, value, write_requests, positions)
        pool.gather(selection, requests, lengths, qpos)
        plan.run(query, hk, hv, pool.indices, pool.hot_block_table, requests)

    # Warm all native/Triton kernels before capture; the graph body contains
    # no host scalar read, shape-dependent allocation, or data-dependent grid.
    lengths.fill_(4096)
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        replay_body()
    torch.cuda.current_stream().wait_stream(warm_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_body()
    output_ptr, index_ptr = plan.output.data_ptr(), pool.indices.data_ptr()
    generator = torch.Generator().manual_seed(1107)
    source_k, source_v = source.unbind(1)
    slots = (
        pool.block_table[0, positions.long() // 32].long() * 32 + positions.long() % 32
    )
    first = None
    for step in range(5):
        key.fill_(0.125 * (step + 1))
        value.fill_(-0.25 * (step + 1))
        query.copy_(
            torch.randn(query.shape, generator=generator).half().to(pool.device)
        )
        selected = (torch.arange(17, dtype=torch.int32) * 4 + 80 * step) % 3000
        selected[-3:] = torch.arange(4093, 4096, dtype=torch.int32)
        selection.copy_(selected[None].to(pool.device))
        graph.replay()
        actual, actual_lse = plan.output.clone(), plan.lse.clone()
        reshape_and_cache_flash(
            key,
            value,
            source_k,
            source_v,
            slots,
            "fp8_e4m3" if e4m3 else "auto",
            pool.k_scale,
            pool.v_scale,
        )
        expected, expected_lse = _assert_original(
            pool, source, query, selection, requests, plan
        )
        assert torch.equal(expected.view(torch.uint8), actual.view(torch.uint8))
        assert torch.equal(expected_lse.view(torch.uint8), actual_lse.view(torch.uint8))
        assert (
            plan.output.data_ptr() == output_ptr
            and pool.indices.data_ptr() == index_ptr
        )
        if first is None:
            first = actual
        else:
            assert not torch.equal(first, actual)
    torch.accelerator.synchronize()
    assert torch.equal(source.cpu().view(torch.uint8), pool.host.view(torch.uint8))


@pytest.mark.parametrize("e4m3", [False, True])
def test_changed_length_graph_replay_reads_prewritten_rows_and_reentered_suffix(e4m3):
    pool, source, query, selection, requests, lengths, qpos, plan = _setup(
        e4m3, rows=1, heads=6, width=17
    )
    selection.fill_(-1)
    selection[0, 0] = 60
    lengths.fill_(61)
    qpos.fill_(60)
    hk, hv = pool.hot.unbind(1)

    def body():
        pool.gather(selection, requests, lengths, qpos)
        plan.run(query, hk, hv, pool.indices, pool.hot_block_table, requests)

    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        body()
    torch.cuda.current_stream().wait_stream(warm_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    pointers = (pool.indices.data_ptr(), pool.hot.data_ptr(), plan.output.data_ptr())
    for reenter in (False, True):
        if reenter:
            lengths.fill_(61)
            pool.truncate(lengths)
            # Restore committed source bytes ahead of the visible frontier.
            for req in range(pool.config.requests):
                positions = torch.arange(
                    61, 4096, dtype=torch.int32, device=pool.device
                )
                pages = pool.block_table[req, positions.long() // 32].long()
                key = source[pages, 0, positions.long() % 32].contiguous()
                value = source[pages, 1, positions.long() % 32].contiguous()
                pool.write_encoded(
                    key, value, torch.full_like(positions, req), positions
                )
        for length in (61, 62, 63, 64, 65, 61, 64, 129):
            lengths.fill_(length)
            qpos.fill_(length - 1)
            selection[0, 0] = length - 1
            graph.replay()
            actual, actual_lse = plan.output.clone(), plan.lse.clone()
            slot = int(pool.indices[0, 0]) // 4
            page = int(pool.block_table[0, (length - 1) // 32])
            torch.accelerator.synchronize()
            assert torch.equal(
                pool.hot[slot, :, (length - 1) % 4].cpu(),
                pool.host[page, :, (length - 1) % 32],
            )
            expected, expected_lse = _assert_original(
                pool, source, query, selection, requests, plan
            )
            assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
            assert torch.equal(
                actual_lse.view(torch.uint8), expected_lse.view(torch.uint8)
            )
            assert pointers == (
                pool.indices.data_ptr(),
                pool.hot.data_ptr(),
                plan.output.data_ptr(),
            )


@pytest.mark.parametrize("ratio", [0.1, 0.2])
@pytest.mark.parametrize("policy", ["bucket", "lru"])
def test_large_parallel_lru_metadata_is_graph_capturable(ratio, policy):
    config = HostKVConfig(
        8,
        262144,
        16,
        2051,
        1,
        hot_ratio=ratio,
        qsa_block_topk=512,
        eviction_policy=policy,
    )
    layout = torch.empty((65536, 2, 32, 1, 256), dtype=torch.uint8, device="meta")
    pool = ActiveHostKV(config, layout, device="cuda")
    # This gate exercises large sort/assignment with synthetic ready payload.
    # Native write and byte-preserving UVA are checked by the other gates.
    pool.host.fill_(56)
    pool.ready.fill_(1)
    selection = torch.full((16, 2051), -1, dtype=torch.int32, device=pool.device)
    expanded = (
        torch.arange(512, dtype=torch.int32, device=pool.device)[:, None] * 4
        + torch.arange(4, dtype=torch.int32, device=pool.device)
    ).flatten()
    selection[:, :2048] = expanded
    requests = torch.arange(16, dtype=torch.int32, device=pool.device) % 8
    lengths = torch.full((8,), 262144, dtype=torch.int32, device=pool.device)
    qpos = torch.full((16,), 262143, dtype=torch.int64, device=pool.device)
    args = (selection, requests, lengths, qpos)
    warm_stream = torch.cuda.Stream()
    warm_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warm_stream):
        pool.gather(*args)
    torch.cuda.current_stream().wait_stream(warm_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        pool.gather(*args)
    for step in range(6):
        selection[:, :2048] = expanded + step * 4096 * 4
        graph.replay()
        assert (pool.indices[:, :2048] >= 0).all()
        owners = pool.slot_owner[pool.slot_owner >= 0]
        assert owners.unique().numel() == owners.numel()
        slots = pool.group_to_slot.flatten()[owners.long()]
        assert torch.equal(pool.slot_owner[slots.long()], owners)
        selected_slots = pool.indices[:, :2048] // 4
        selected_owners = pool.slot_owner[selected_slots.long()]
        assert torch.equal(
            selected_owners // config.groups_per_request,
            requests[:, None].expand_as(selected_owners),
        )
        assert torch.equal(
            selected_owners % config.groups_per_request,
            selection[:, :2048] // 4,
        )
        assert (pool.hot[selected_slots.long()] == 56).all()
        assert int(pool.wanted_count) == 8 * (512 + 8)
        graph.replay()
        assert int(pool.miss_count) == 0


@pytest.mark.parametrize("e4m3", [False, True])
def test_qsa_513_groups_per_row_at_exact_capacity_in_graph(e4m3):
    config = HostKVConfig(1, 20000, 8, 2051, 1, hot_groups=4113, qsa_block_topk=512)
    dtype = torch.uint8 if e4m3 else torch.float16
    layout = torch.empty((625, 2, 32, 1, 256), dtype=dtype, device="meta")
    pool = ActiveHostKV(config, layout, device="cuda")
    pool.host.fill_(56)
    pool.ready.fill_(1)
    starts = torch.arange(8, device=pool.device) * 513 * 4
    base = (starts[:, None] + torch.arange(2051, device=pool.device)).int()
    selection = base.clone()
    requests = torch.zeros(8, dtype=torch.int32, device=pool.device)
    lengths = torch.tensor([19999], dtype=torch.int32, device=pool.device)
    qpos = (starts + 2050).long()

    def body():
        pool.gather(selection, requests, lengths, qpos)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        body()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    pointers = (pool.hot.data_ptr(), pool.indices.data_ptr())
    for step in range(4):
        selection.copy_(base + step * 4)
        qpos.copy_(starts + 2050 + step * 4)
        graph.replay()
        assert int(pool.wanted_count) == pool.slots == 4113
        assert (pool.indices >= 0).all()
        assert pool.slot_owner.unique().numel() == pool.slots
        owners = pool.slot_owner[(pool.indices // 4).long()]
        assert torch.equal(owners, selection // 4)
        assert (pool.hot == 56).all()
        assert pointers == (pool.hot.data_ptr(), pool.indices.data_ptr())
        graph.replay()
        assert int(pool.miss_count) == 0
