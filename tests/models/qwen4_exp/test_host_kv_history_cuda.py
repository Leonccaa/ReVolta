# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CUDA gates for the host history store (host KV host-history integration). Skipped
without CUDA.

Single card:
    CUDA_VISIBLE_DEVICES=0 python -m pytest -q \
        tests/models/qwen4_exp/test_host_kv_history_cuda.py

1. Writes: the store's bytes equal ``reshape_and_cache_flash`` into a paged
   GPU cache at the same slots (E4M3 with calibrated scales, and FP16).
2. Reads: the public Triton sparse route on the paged cache with logical IDs
   and on the hot pool with remapped IDs gives identical output bits at the
   production selection width (top-k 2048, 2051 columns) in one pass.
3. CUDA graph: write + gather passes + attention captured once and replayed
   with new keys, selections and lengths; equal to the eager GPU arm and
   with stable buffer addresses.
"""

import random

import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
    for_each_pass,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="host history CUDA gates need a GPU"
)

PAGE, DIM, QHEADS, TOPK = 64, 256, 6, 2048
WIDTH = TOPK + 3
SCALES = (0.375, 1.75)


def _ops():
    from vllm._custom_ops import reshape_and_cache_flash
    from vllm.models.qwen4_exp.nvidia.ops.qsa import qsa_sparse_paged_attention

    return reshape_and_cache_flash, qsa_sparse_paged_attention


class Arms:
    def __init__(self, e4m3: bool, *, rows_per_pass: int, requests: int = 2):
        self.dtype = torch.uint8 if e4m3 else torch.float16
        self.kv = "fp8_e4m3" if e4m3 else "auto"
        self.requests = requests
        self.blocks = 160
        shape = HostHistoryShape(
            num_blocks=self.blocks,
            kernel_blocks_per_block=1,
            kernel_block_tokens=PAGE,
            num_layers=2,
            kv_heads=1,
            head_dim=DIM,
            dtype=self.dtype,
            hot_slots=rows_per_pass * 513 + requests * 9,
            max_requests=requests,
            selection_width=WIDTH,
            max_write_rows=8192,
        )
        self.store = HostHistoryStore(shape, ["l0", "l1"], device="cuda")
        self.layer = self.store.layer("l1")
        self.paged = torch.zeros(
            (self.blocks, 2, PAGE, 1, DIM), dtype=self.dtype, device="cuda"
        )
        self.k_scale = torch.tensor(SCALES[0], device="cuda")
        self.v_scale = torch.tensor(SCALES[1], device="cuda")
        # Request r owns blocks 1 + r, 1 + r + requests, ...: interleaved.
        cols = (self.blocks - 1) // requests
        self.table = (
            (
                torch.arange(cols * requests, dtype=torch.int32)
                .reshape(cols, requests)
                .T
                + 1
            )
            .contiguous()
            .cuda()
        )
        self.gen = torch.Generator(device="cuda").manual_seed(13)
        self.rng = random.Random(13)

    def slots(self, req: int, positions: list[int]) -> torch.Tensor:
        table = self.table.cpu()
        return torch.tensor(
            [int(table[req, p // PAGE]) * PAGE + p % PAGE for p in positions],
            dtype=torch.int64,
            device="cuda",
        )

    def write(self, key, value, slots):
        cache, _ = _ops()
        self.layer.write(key, value, slots, self.kv, self.k_scale, self.v_scale)
        k, v = self.paged.unbind(1)
        cache(key, value, k, v, slots, self.kv, self.k_scale, self.v_scale)

    def selection(self, q: int) -> list[int]:
        tail = (q + 1) // 4 * 4
        groups = self.rng.sample(range(tail // 4), min(tail // 4, TOPK // 4))
        row = [g * 4 + i for g in sorted(groups) for i in range(4)]
        row += list(range(tail, q + 1))
        return row + [-1] * (WIDTH - len(row))

    def attend_both(self, query, selection, token_to_req, seq_lens, out_host):
        _, sparse = _ops()
        k, v = self.paged.unbind(1)
        expected = sparse(
            query, k, v, selection, self.table, token_to_req,
            kv_cache_dtype=self.kv, k_scale=SCALES[0], v_scale=SCALES[1],
        )  # fmt: skip

        def attend(start, end, hot_ids):
            sparse(
                query[start:end], self.layer.key_cache, self.layer.value_cache,
                hot_ids, self.store.hot_block_table, token_to_req[start:end],
                out_host[start:end], kv_cache_dtype=self.kv,
                k_scale=SCALES[0], v_scale=SCALES[1],
            )  # fmt: skip

        for_each_pass(self.layer, selection, token_to_req, self.table, seq_lens, attend)
        return expected


@pytest.mark.parametrize("e4m3", [True, False])
def test_writes_equal_the_paged_cache_writer(e4m3):
    arms = Arms(e4m3, rows_per_pass=4)
    n = 3000
    key = torch.randn((n, 1, DIM), device="cuda", generator=arms.gen).half()
    value = torch.randn_like(key)
    slots = arms.slots(0, list(range(n)))
    slots[::7] = -1  # tokens another DCP rank owns
    arms.write(key, value, slots)
    torch.accelerator.synchronize()
    ready = arms.store.ready.view(arms.blocks, PAGE).bool().cpu()
    host = arms.layer.host_view.permute(0, 2, 1, 3, 4)[ready]
    paged = arms.paged.cpu().permute(0, 2, 1, 3, 4)[ready]
    assert int(ready.sum()) == int((slots >= 0).sum()) + PAGE  # + null block
    assert torch.equal(host.view(torch.uint8), paged.view(torch.uint8))


@pytest.mark.parametrize("e4m3", [True, False])
def test_one_pass_triton_route_is_bitwise_equal(e4m3):
    arms = Arms(e4m3, rows_per_pass=8)
    lengths = [4100, 3333]
    for req, length in enumerate(lengths):
        key = torch.randn((length, 1, DIM), device="cuda", generator=arms.gen).half()
        arms.write(key, torch.randn_like(key), arms.slots(req, list(range(length))))
    rows = [(0, 4099)] * 3 + [(1, 3332)] * 3 + [(0, 2000), (1, 77)]
    selection = torch.tensor(
        [arms.selection(q) for _, q in rows], dtype=torch.int32, device="cuda"
    )
    token_to_req = torch.tensor([r for r, _ in rows], dtype=torch.int32, device="cuda")
    seq_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    query = torch.randn((len(rows), QHEADS, DIM), device="cuda").half() * 0.25
    out = torch.empty_like(query)
    expected = arms.attend_both(query, selection, token_to_req, seq_lens, out)
    assert torch.equal(expected.view(torch.uint8), out.view(torch.uint8))


def test_graph_replay_with_changed_inputs():
    arms = Arms(True, rows_per_pass=4)
    lengths = [3000, 2000]
    for req, length in enumerate(lengths):
        key = torch.randn((length, 1, DIM), device="cuda", generator=arms.gen).half()
        arms.write(key, torch.randn_like(key), arms.slots(req, list(range(length))))
    rows, step_tokens = 8, 2
    s_key = torch.zeros((step_tokens, 1, DIM), dtype=torch.float16, device="cuda")
    s_value = torch.zeros_like(s_key)
    s_slots = torch.full((step_tokens,), -1, dtype=torch.int64, device="cuda")
    s_sel = torch.full((rows, WIDTH), -1, dtype=torch.int32, device="cuda")
    s_req = torch.zeros(rows, dtype=torch.int32, device="cuda")
    s_lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")
    s_query = torch.zeros((rows, QHEADS, DIM), dtype=torch.float16, device="cuda")
    s_out = torch.zeros_like(s_query)
    _, sparse = _ops()

    def body():
        arms.layer.write(s_key, s_value, s_slots, arms.kv, arms.k_scale, arms.v_scale)

        def attend(start, end, hot_ids):
            sparse(
                s_query[start:end], arms.layer.key_cache, arms.layer.value_cache,
                hot_ids, arms.store.hot_block_table, s_req[start:end],
                s_out[start:end], kv_cache_dtype=arms.kv,
                k_scale=SCALES[0], v_scale=SCALES[1],
            )  # fmt: skip

        for_each_pass(arms.layer, s_sel, s_req, arms.table, s_lens, attend)

    def load(step):
        # Each step appends one token to each request, then attends.
        positions = [lengths[0] + step, lengths[1] + step]
        s_key.copy_(torch.randn(s_key.shape, device="cuda", generator=arms.gen))
        s_value.copy_(torch.randn(s_key.shape, device="cuda", generator=arms.gen))
        s_slots.copy_(torch.cat([arms.slots(r, [p]) for r, p in enumerate(positions)]))
        s_lens.copy_(torch.tensor([p + 1 for p in positions]))
        qs = [(r, positions[r] - (i // 2) * 37) for i, r in enumerate([0, 1] * 4)]
        s_req.copy_(torch.tensor([r for r, _ in qs]))
        s_sel.copy_(torch.tensor([arms.selection(q) for _, q in qs]))
        s_query.copy_(torch.randn(s_query.shape, device="cuda", generator=arms.gen))

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        load(0)
        body()  # compile and warm outside capture
        # The GPU arm also gets the warm-up step's tokens.
        cache, _ = _ops()
        k, v = arms.paged.unbind(1)
        cache(s_key, s_value, k, v, s_slots, arms.kv, arms.k_scale, arms.v_scale)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        body()
    pointers = (s_out.data_ptr(), arms.store.indices.data_ptr())
    for step in range(1, 6):
        load(step)
        graph.replay()
        # Eager GPU arm on the same inputs: write the paged cache, attend.
        cache, _ = _ops()
        k, v = arms.paged.unbind(1)
        cache(s_key, s_value, k, v, s_slots, arms.kv, arms.k_scale, arms.v_scale)
        expected = sparse(
            s_query, k, v, s_sel, arms.table, s_req, kv_cache_dtype=arms.kv,
            k_scale=SCALES[0], v_scale=SCALES[1],
        )  # fmt: skip
        torch.accelerator.synchronize()
        # Two passes of four rows: same 2051-column tiles, per-pass splits.
        torch.testing.assert_close(s_out, expected, atol=2e-3, rtol=1e-2)
        assert (s_out.data_ptr(), arms.store.indices.data_ptr()) == pointers
