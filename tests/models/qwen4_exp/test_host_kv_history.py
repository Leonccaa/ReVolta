# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Host history store addressed by history block ID (host KV host-history integration),
CPU interpreter.

Two arms share one engine trace (block allocation, slot mappings, selections):

* GPU arm: a paged cache ``[kernel blocks, 2, kbs, H, D]`` written at the slot
  mapping and read by the unchanged QSA Triton split/merge kernels with the
  logical selection and block table, i.e. "main K/V on GPU".
* Host arm: the store writes the same encoded bytes to its host pages, gathers
  the selected groups into the hot pool and runs the same kernels on hot IDs.

With the same rows per call the computation plan is identical, so outputs and
LSE must match bit for bit; the host bytes must equal the paged cache bytes.
"""

import os
import random

import numpy as np
import pytest
import torch

from vllm.models.qwen4_exp.nvidia.host_kv import LayerScales
from vllm.models.qwen4_exp.nvidia.host_kv.attention import SparseAttentionPlan
from vllm.models.qwen4_exp.nvidia.host_kv.history import (
    HostHistoryShape,
    HostHistoryStore,
    for_each_pass,
)
from vllm.models.qwen4_exp.nvidia.host_kv.pool import UnsupportedHostKVConfig

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="The host history kernels run on the CPU interpreter here",
)

KBS, HEADS, DIM, QHEADS, WIDTH = 16, 1, 16, 2, 12
SCALES = LayerScales(0.375, 1.75)


@pytest.fixture(autouse=True)
def _interpreter_boolean_comparison(monkeypatch):
    # Same Triton 3.6 interpreter repair as test_host_kv_attention.py.
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    from vllm.triton_utils import tl

    def greater(_builder, lhs, rhs):
        return TensorHandle(np.greater(lhs.data, rhs.data), tl.int1)

    monkeypatch.setattr(InterpreterBuilder, "create_fcmpOGT", greater)


def _shape(e4m3=True, *, blocks=33, rows_per_pass=3, requests=2, **kwargs):
    # Each column may select its own group (claims per column), so a row
    # touches at most WIDTH groups; the hot pool holds rows_per_pass rows.
    fields = dict(
        num_blocks=blocks,
        kernel_blocks_per_block=1,
        kernel_block_tokens=KBS,
        num_layers=2,
        kv_heads=HEADS,
        head_dim=DIM,
        dtype=torch.uint8 if e4m3 else torch.float16,
        hot_slots=rows_per_pass * WIDTH + requests * 9,
        max_requests=requests,
        selection_width=WIDTH,
        max_write_rows=128,
        row_groups=WIDTH,
    )
    fields.update(kwargs)
    return HostHistoryShape(**fields)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(torch.uint8)


class Engine:
    """Scheduler-like trace driving both arms for one layer."""

    def __init__(self, shape, *, dcp=(1, 0), seed=7):
        self.shape = shape
        self.store = HostHistoryStore(shape, ["l0", "l1"], device="cpu")
        self.layer = self.store.layer("l1")
        self.world, self.rank = dcp
        self.paged = torch.zeros(
            (shape.num_kernel_blocks, 2, KBS, HEADS, DIM), dtype=shape.dtype
        )
        self.free = list(range(1, shape.num_blocks))
        self.blocks: dict[str, list[int]] = {}
        self.length: dict[str, int] = {}
        self.rng = random.Random(seed)
        self.gen = torch.Generator().manual_seed(seed)
        self.e4m3 = shape.dtype == torch.uint8
        self.calls = 0
        self.misses = 0

    # -- scheduler side ------------------------------------------------
    def ensure_blocks(self, req: str, tokens: int) -> None:
        span = KBS * self.world
        need = -(-tokens // span)
        new = []
        while len(self.blocks.setdefault(req, [])) < need:
            block = self.free.pop(0)
            self.blocks[req].append(block)
            new.append(block)
        if new:
            # The worker's zero hook: GPU arm zeroes, host arm begins blocks.
            for block in new:
                self.paged[block].zero_()
            self.store.begin_blocks(new)

    def finish(self, req: str) -> list[int]:
        blocks = self.blocks.pop(req)
        self.length.pop(req)
        self.free = blocks + self.free  # LIFO reuse exercises stale pages
        return blocks

    # -- worker side ---------------------------------------------------
    def owned(self, position: int) -> bool:
        return position % self.world == self.rank

    def local(self, position: int) -> int:
        return position // self.world

    def slot(self, req: str, position: int) -> int:
        local = self.local(position)
        return self.blocks[req][local // KBS] * KBS + local % KBS

    def write(self, req: str, positions: list[int]) -> None:
        self.ensure_blocks(req, positions[-1] + 1)
        key = torch.randn((len(positions), HEADS, DIM), generator=self.gen).half()
        value = torch.randn(key.shape, generator=self.gen).half()
        slots = torch.tensor(
            [self.slot(req, p) if self.owned(p) else -1 for p in positions],
            dtype=torch.int64,
        )
        self.layer.write(
            key,
            value,
            slots,
            "fp8_e4m3" if self.e4m3 else "auto",
            torch.tensor(SCALES.key),
            torch.tensor(SCALES.value),
        )
        # GPU arm: the same encoder, scattered into the paged cache.
        k_enc, v_enc = self.store.staging[: len(positions), :, 0].unbind(1)
        for row, slot in enumerate(slots.tolist()):
            if slot >= 0:
                self.paged[slot // KBS, 0, slot % KBS] = k_enc[row]
                self.paged[slot // KBS, 1, slot % KBS] = v_enc[row]
        self.length[req] = max(self.length.get(req, 0), positions[-1] + 1)

    def table(self, reqs: list[str]) -> torch.Tensor:
        width = max(len(self.blocks[r]) for r in reqs)
        table = torch.zeros((len(reqs), width + 1), dtype=torch.int32)
        for i, r in enumerate(reqs):
            table[i, : len(self.blocks[r])] = torch.tensor(self.blocks[r])
        return table

    def selection(self, req: str, query_position: int) -> list[int]:
        visible = [p for p in range(query_position + 1) if self.owned(p)]
        picks = [self.local(p) for p in self.rng.choices(visible, k=WIDTH - 2)]
        picks += [-1, picks[0]]  # padding and a repeated column stay put
        self.rng.shuffle(picks)
        return picks

    def step(self, rows: list[tuple[str, int]]) -> None:
        """Attend ``rows`` of (request, query position) in both arms."""
        self.calls += 1
        reqs = sorted({r for r, _ in rows})
        table = self.table(reqs)
        seq_lens = torch.tensor([self.length[r] for r in reqs], dtype=torch.int32)
        token_to_req = torch.tensor([reqs.index(r) for r, _ in rows], dtype=torch.int32)
        selection = torch.tensor(
            [self.selection(r, q) for r, q in rows], dtype=torch.int32
        )
        query = (torch.randn((len(rows), QHEADS, DIM), generator=self.gen) * 0.5).half()
        gpu_k, gpu_v = self.paged.unbind(1)
        seen = []

        def attend(start, end, hot_ids):
            plan = SparseAttentionPlan(
                rows=end - start,
                query_heads=QHEADS,
                kv_heads=HEADS,
                head_dim=DIM,
                selection_width=WIDTH,
                device="cpu",
                e4m3=self.e4m3,
                scales=SCALES,
            )
            expected = plan.run(
                query[start:end],
                gpu_k,
                gpu_v,
                selection[start:end].contiguous(),
                table,
                token_to_req[start:end].contiguous(),
            ).clone()
            expected_lse = plan.lse.clone()
            actual = plan.run(
                query[start:end],
                self.layer.key_cache,
                self.layer.value_cache,
                hot_ids,
                self.store.hot_block_table,
                token_to_req[start:end].contiguous(),
            )
            assert torch.equal(_bits(expected), _bits(actual))
            assert torch.equal(_bits(expected_lse), _bits(plan.lse))
            # Every column maps back to its logical token (order, repeats,
            # padding), through the hot slot owner and the block table.
            for row, ids in zip(range(start, end), hot_ids.tolist()):
                for col, hot in enumerate(ids):
                    token = int(selection[row, col])
                    if token < 0:
                        assert hot == -1
                        continue
                    gid = int(self.layer.slot_owner[hot // 4])
                    kb = int(table[int(token_to_req[row]), token // KBS])
                    assert gid == kb * (KBS // 4) + token % KBS // 4
                    assert hot % 4 == token % 4
            seen.append((start, end))
            self.misses += int(self.store.miss_count)

        for_each_pass(self.layer, selection, token_to_req, table, seq_lens, attend)
        assert seen and seen[0][0] == 0 and seen[-1][1] == len(rows)
        self.check_bytes()

    def check_bytes(self) -> None:
        """Host pages equal the paged cache for every committed slot."""
        ready = self.store.ready.view(-1, KBS).bool()
        host = self.layer.host_view.permute(0, 2, 1, 3, 4)[ready]
        paged = self.paged.permute(0, 2, 1, 3, 4)[ready]
        assert ready.sum() > 0
        assert torch.equal(_bits(host), _bits(paged))


def _prefill(engine, req, start, end):
    engine.write(req, list(range(start, end)))
    engine.step([(req, q) for q in range(start, end)])


@pytest.mark.parametrize("e4m3", [True, False])
def test_append_prefill_decode_growth_and_passes_match_gpu_arm(e4m3):
    engine = Engine(_shape(e4m3))
    # Prefill A in two chunks (append), then B; 7 rows -> passes of 3.
    _prefill(engine, "a", 0, 24)
    _prefill(engine, "a", 24, 31)
    _prefill(engine, "b", 0, 40)
    for _ in range(20):  # decode growth across block boundaries
        for req in ("a", "b"):
            engine.write(req, [engine.length[req]])
        engine.step([("a", engine.length["a"] - 1), ("b", engine.length["b"] - 1)])
    assert engine.length == {"a": 51, "b": 60}
    # Both layers share the ready map; only layer l1 was written here, so
    # the other layer's pages stay zero and never become hot.
    assert int((engine.store.slot_owner[0] >= 0).sum()) == 0


def test_small_hot_pool_evicts_and_still_matches_gpu_arm():
    # One row per pass and one request tail: 21 hot groups for 25+ groups of
    # history, so passes must evict groups an earlier pass made resident.
    engine = Engine(_shape(rows_per_pass=1, requests=1, blocks=12), seed=3)
    _prefill(engine, "a", 0, 100)
    for _ in range(6):
        engine.write("a", [engine.length["a"]])
        engine.step([("a", engine.length["a"] - 1)])
    assert engine.misses > engine.shape.hot_slots
    assert int((engine.layer.slot_owner >= 0).sum()) == engine.shape.hot_slots


def test_reused_block_ids_never_read_old_data():
    engine = Engine(_shape())
    _prefill(engine, "a", 0, 32)  # two full blocks, now hot
    old = engine.finish("a")
    hot_before = int((engine.layer.slot_owner >= 0).sum())
    assert hot_before > 0
    generation = engine.store.generation[old].clone()
    # C reuses A's IDs: begin_blocks retires every hot group of those blocks,
    # forgets readiness and bumps the generation before anything is written.
    engine.ensure_blocks("c", 32)
    assert engine.blocks["c"][:2] == old
    assert torch.equal(engine.store.generation[old], generation + 1)
    for layer in engine.store.layers:
        gids = torch.cat([torch.arange(b * 4, b * 4 + 4) for b in old])
        assert (layer.group_to_slot[gids] == -1).all()
    owners = engine.layer.slot_owner
    assert not any(int(g) // 4 in old for g in owners[owners >= 0])
    assert not engine.store.ready.view(-1, KBS)[old].any()
    # C writes 5 tokens of its first block and attends: equal to the GPU arm.
    engine.length["c"] = 0
    engine.write("c", list(range(5)))
    engine.step([("c", 4)])
    # A selection of an unwritten slot of the reused block is refused,
    # although A had committed it before.
    engine.length["c"] = 9
    with pytest.raises(ValueError, match="not committed"):
        engine.layer.gather(
            torch.tensor([[7] + [-1] * (WIDTH - 1)], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            engine.table(["c"]),
            torch.tensor([9], dtype=torch.int32),
        )


def test_preemption_recompute_matches_gpu_arm():
    engine = Engine(_shape(), seed=11)
    _prefill(engine, "a", 0, 20)
    _prefill(engine, "b", 0, 20)
    for _ in range(5):
        engine.write("b", [engine.length["b"]])
        engine.step([("b", engine.length["b"] - 1)])
    # Preempt B: its blocks go back to the pool; A grows into them.
    engine.finish("b")
    _prefill(engine, "a", 20, 50)
    # B is recomputed from scratch on newly allocated (reused) IDs.
    _prefill(engine, "b", 0, 25)
    engine.step([("a", 49), ("b", 24), ("a", 30)])


@pytest.mark.parametrize("rank", [0, 1])
def test_dcp2_rank_local_store_matches_gpu_arm(rank):
    from vllm.models.qwen4_exp.nvidia.ops.qsa_dcp import _localize_indices
    from vllm.triton_utils import triton

    shape = _shape(dcp_world_size=2, dcp_rank=rank)
    engine = Engine(shape, dcp=(2, rank))
    # Engine writes global positions; slots for other-rank tokens are -1.
    engine.write("a", list(range(70)))
    engine.write("b", list(range(33)))
    # The DCP2 localization kernel produces this rank's compacted IDs from a
    # global selection; the engine helper above already yields rank-local IDs
    # for owned positions, which the kernel must reproduce.
    glob = torch.tensor([[0, 1, 2, 3, 64, 65, 66, 69, -1, 5, 8, 9]], dtype=torch.int32)
    local = torch.empty_like(glob)
    _localize_indices[(1,)](
        glob, local, glob.stride(0), local.stride(0), WIDTH=WIDTH, WORLD=2,
        RANK=rank, INTERLEAVE=1, BLOCK=triton.next_power_of_2(WIDTH),
    )  # fmt: skip
    owned = [p // 2 for p in glob[0].tolist() if p >= 0 and p % 2 == rank]
    assert local[0].tolist() == owned + [-1] * (WIDTH - len(owned))
    engine.step([("a", 69), ("b", 32), ("a", 40), ("b", 10)])


def test_shape_rejects_a_hot_pool_below_one_row():
    with pytest.raises(UnsupportedHostKVConfig, match="one query row"):
        _shape(rows_per_pass=0).validate()
    with pytest.raises(UnsupportedHostKVConfig):
        _shape(kernel_block_tokens=6).validate()
    with pytest.raises(UnsupportedHostKVConfig):
        _shape(dtype=torch.bfloat16).validate()


def test_host_page_is_one_contiguous_block_for_all_layers():
    engine = Engine(_shape())
    engine.write("a", list(range(20)))
    block = engine.blocks["a"][1]
    page = engine.store.host_page(block)
    assert page.is_contiguous()
    assert page.shape == (1, 2, 2, KBS, HEADS, DIM)
    assert page.data_ptr() == engine.store.host[block].data_ptr()
    # Layer l1's page bytes are its kernel-block view of the same block.
    assert torch.equal(_bits(page[0, 1]), _bits(engine.layer.host_view[block]))
    with pytest.raises(IndexError):
        engine.store.host_page(engine.shape.num_blocks)


def test_pages_restored_outside_the_forward_are_published_by_mark_ready():
    # An host prefix adapter tier restores block contents into newly allocated IDs:
    # begin,
    # copy the bytes into host_page(), mark_ready(), then attend.
    engine = Engine(_shape(), seed=21)
    engine.ensure_blocks("a", 40)
    blocks = engine.blocks["a"]
    gen = torch.Generator().manual_seed(4)
    for block in blocks:
        page = engine.store.host_page(block)
        payload = torch.randint(0, 120, page.shape, generator=gen, dtype=torch.uint8)
        page.copy_(payload)
        # GPU arm: the same layer-1 bytes in the paged cache.
        engine.paged[block] = payload[0, 1]
    with pytest.raises(ValueError, match="not committed"):
        engine.length["a"] = 40
        engine.step([("a", 39)])
    generation = engine.store.generation[blocks].clone()
    engine.store.mark_ready(blocks)
    assert torch.equal(engine.store.generation[blocks], generation)
    engine.step([("a", 39), ("a", 17), ("a", 3)])


def test_m6b_leased_pages_restore_into_history_ids_without_transferring_ownership():
    from vllm.v1.kv_offload.base import make_offload_key
    from vllm.v1.kv_offload.host.pool import HostKVPool

    engine = Engine(_shape(), seed=23)
    engine.write("a", list(range(40)))
    engine.step([("a", 39)])  # Populate hot groups before recycling the IDs.
    blocks = engine.blocks["a"]
    page_bytes = engine.shape.page_bytes
    backing = bytearray(len(blocks) * page_bytes)
    pool = HostKVPool(
        memoryview(backing).cast("B", shape=[len(blocks), page_bytes]),
        group_idx=0,
        block_tokens=KBS,
    )
    keys = [make_offload_key(bytes([i + 1]), 0) for i in range(len(blocks))]
    for key, block in zip(keys, blocks):
        page = pool.reserve(key)
        page.view[:] = engine.store.host_page(block).numpy().tobytes()
        pool.publish(page)
        pool.release(page)
    leases = pool.acquire_prefix(keys)
    assert len(leases.pages) == len(blocks)
    engine.store.begin_blocks(blocks)
    generation = engine.store.generation[blocks].clone()
    for page, block in zip(leases.pages, blocks):
        target = memoryview(engine.store.host_page(block).numpy()).cast("B")
        target[:] = page.view
    assert pool.reserve(make_offload_key(b"other", 0)) is None
    with pytest.raises(ValueError, match="not committed"):
        engine.step([("a", 39)])
    engine.store.mark_ready(blocks)
    assert torch.equal(engine.store.generation[blocks], generation)
    engine.step([("a", 39), ("a", 17), ("a", 3)])
    assert all(not page._closed for page in leases.pages)
    leases.release()
    assert not pool._leases
