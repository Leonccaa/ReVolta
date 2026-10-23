# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU workspace contracts and MTP ordering model.

Real loader, QSA buffer registration, MTP hooks and compaction; model forwards
are simulated. These tests establish ordering, not GPU kernel/graph safety.
"""

import random
import weakref
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.common.qsa_cache import _build_qsa_metadata_torch
from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMultiTokenPredictor
from vllm.models.qwen4_exp.nvidia.qsa import Qwen4ExpQSAAttention
from vllm.utils import torch_utils
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as spec_mod
from vllm.v1.worker.gpu.spec_decode.eagle import utils

# Workspace size for an 8192-token batch and a 2048-token indexer budget.
PROD_MAX_TOKENS = 8192
PROD_WIDTH = (
    2048 + 4 - 1
)  # indexer_budget + indexer_compress_ratio - 1 (indexer_qsa.py:175)
# Simulation uses a smaller row count so the trace stays cheap; widths are irrelevant
# to ordering, row ranges are what matter.
SIM_MAX_TOKENS = 512
SIM_WIDTH = 8
NUM_TARGET_QSA = 12  # layer_types: full_attention every 4th of 48 layers
NUM_SPEC = 4  # num_speculative_tokens
MAX_REQS = 8


class Tracer:
    def __init__(self):
        self.events = []
        self.last_writer = {}  # data_ptr -> list[str|None] per row
        self.hazards = []

    def _rows(self, buf):
        key = buf.data_ptr()
        if key not in self.last_writer:
            self.last_writer[key] = [None] * buf.shape[0]
        return self.last_writer[key]

    def write(self, phase, actor, buf, rows):
        lw = self._rows(buf)
        for r in rows:
            lw[r] = actor
        self.events.append((phase, actor, "W", buf.data_ptr(), len(rows)))

    def read(self, phase, actor, buf, rows):
        lw = self._rows(buf)
        bad = [r for r in rows if lw[r] not in (actor,)]
        if bad:
            self.hazards.append((phase, actor, len(bad), lw[bad[0]]))
        self.events.append((phase, actor, "R", buf.data_ptr(), len(rows)))


def attention(buffer=None):
    # Exercise the real registration/alias contract without loading weights.
    layer = Qwen4ExpQSAAttention.__new__(Qwen4ExpQSAAttention)
    nn.Module.__init__(layer)
    layer.indexer = SimpleNamespace(skip_topk=False, output_width=SIM_WIDTH)
    layer._set_topk_indices_buffer(
        max_tokens=SIM_MAX_TOKENS, topk_indices_buffer=buffer
    )
    if buffer is None:
        layer.topk_indices_buffer.fill_(-7)
    return layer


def build_models():
    target_inner = nn.Module()
    target_inner.topk_indices_buffer = torch.full(
        (SIM_MAX_TOKENS, SIM_WIDTH), -9, dtype=torch.int32
    )
    target_inner.qsa = nn.ModuleList(
        attention(target_inner.topk_indices_buffer) for _ in range(NUM_TARGET_QSA)
    )
    target_inner.embed_tokens = nn.Embedding(8, 4)
    target = nn.Module()
    target.model = target_inner

    draft_inner = Qwen4ExpMultiTokenPredictor.__new__(Qwen4ExpMultiTokenPredictor)
    nn.Module.__init__(draft_inner)
    layer = nn.Module()
    layer.self_attn = attention()
    draft_inner.layers = nn.ModuleList([layer])
    draft_inner.embed_tokens = nn.Embedding(8, 4)
    draft = nn.Module()
    draft.model = draft_inner
    return target, draft


def load(monkeypatch, target, draft, *, method="mtp", async_scheduling=False):
    monkeypatch.setattr(utils, "get_model", lambda **_k: draft)
    monkeypatch.setattr(utils, "get_pp_group", lambda: SimpleNamespace(world_size=1))
    monkeypatch.setattr(
        "vllm.compilation.backends.set_model_tag", lambda _t: nullcontext()
    )
    cfg = SimpleNamespace(
        speculative_config=SimpleNamespace(draft_model_config=object(), method=method),
        scheduler_config=SimpleNamespace(async_scheduling=async_scheduling),
    )
    return utils.load_eagle_model(target, cfg)


def tag(actor_id, epoch, layer, rows):
    # unique, checkable payload per (writer, epoch, layer, row)
    base = actor_id * 10_000_000 + epoch * 1000 + layer * 50
    return torch.tensor(
        [[base + r] * SIM_WIDTH for r in rows], dtype=torch.int32
    ) + torch.arange(SIM_WIDTH, dtype=torch.int32)


class Sim:
    def __init__(self, target, draft, overlap=False):
        self.target, self.draft = target, draft
        self.tr = Tracer()
        self.epoch = 0
        self.reads = []  # payload digests of draft decode reads
        self.mismatch = 0
        self.overlap = overlap
        sp = spec_mod.EagleSpeculator.__new__(spec_mod.EagleSpeculator)
        sp.share_mtp_topk_indices = True
        sp.num_speculative_steps = NUM_SPEC
        sp.last_token_indices = torch.zeros(MAX_REQS, dtype=torch.int64)
        sp.model = draft
        self.sp = sp

    @property
    def dattn(self):
        return self.draft.model.layers[0].self_attn

    def target_forward(self, phase, n):
        self.epoch += 1
        rows = list(range(n))
        for li, attn in enumerate(self.target.model.qsa):
            buf = attn.topk_indices_buffer
            buf[:n].copy_(tag(1, self.epoch, li, rows))
            self.tr.write(phase, "target", buf, rows)
            self.tr.read(phase, "target", buf, rows)
            if not torch.equal(buf[:n], tag(1, self.epoch, li, rows)):
                self.mismatch += 1

    def draft_forward(self, phase, n):
        attn = self.dattn
        buf = attn.topk_indices_buffer
        rows = list(range(n))
        if not attn.indexer.skip_topk:
            self.epoch += 1
            buf[:n].copy_(tag(2, self.epoch, 0, rows))
            self.tr.write(phase, "draft", buf, rows)
            self.step0 = buf[:n].clone()
        self.tr.read(phase, "draft", buf, rows)
        return buf[:n].clone()

    def propose(self, phase, n_tokens, num_reqs, last_idx, between=None):
        sp = self.sp
        sp.last_token_indices[:num_reqs] = torch.tensor(last_idx)
        sp._mtp_prefill_begin()
        self.draft_forward(phase + ":draft_prefill", n_tokens)
        expect = self.step0[torch.tensor(last_idx)]
        buf = self.dattn.topk_indices_buffer
        self.tr.read(phase + ":compact", "draft", buf, list(last_idx))
        sp._mtp_prefill_end(num_reqs)  # real compact_topk_indices
        self.tr.write(phase + ":compact", "draft", buf, list(range(num_reqs)))
        if NUM_SPEC == 1:
            return
        if between is not None:
            between()
        sp._mtp_decode_begin()
        try:
            for step in range(1, NUM_SPEC):
                got = self.draft_forward(phase + f":draft_decode{step}", num_reqs)
                if not torch.equal(got, expect):
                    self.mismatch += 1
                self.reads.append(got.sum().item())
        finally:
            sp._mtp_decode_end()

    def run(self, seed=0, steps=300):
        rng = random.Random(seed)
        # 1) profile run: target max tokens + dummy propose (model_runner.py:800-855)
        self.target_forward("profile", SIM_MAX_TOKENS)
        self.propose(
            "profile",
            SIM_MAX_TOKENS,
            MAX_REQS,
            [SIM_MAX_TOKENS - 1 - i for i in range(MAX_REQS)],
        )
        # 2) MTP MoE kernel warmup dummy_run(16) (speculator.py:225-288)
        self.target_forward("warmup16", 16)
        self.propose("warmup16", 16, 2, [7, 15])
        # 3) target graph capture sizes
        for n in (1, 2, 4, 8, 16, 24, 32, 40):
            self.target_forward(f"capture_target{n}", n)
        # 4) speculator.capture (speculator.py:565-611)
        self.sp.last_token_indices.zero_()
        self.sp._mtp_prefill_begin()
        for n in (5, 10, 20, 40):
            self.draft_forward(f"capture_draft_prefill{n}", n)
        self.sp._mtp_prefill_end(MAX_REQS)
        self.tr.write(
            "capture:compact",
            "draft",
            self.dattn.topk_indices_buffer,
            list(range(MAX_REQS)),
        )
        self.sp._mtp_decode_begin()
        try:
            for n in range(1, MAX_REQS + 1):
                # decode graph captured with skip_topk=True: reads rows only
                self.tr.read(
                    f"capture_draft_decode{n}",
                    "draft",
                    self.dattn.topk_indices_buffer,
                    list(range(n)),
                )
        finally:
            self.sp._mtp_decode_end()
        # 5) serving steps
        for s in range(steps):
            num_reqs = rng.randint(1, MAX_REQS)
            per_req = []
            for _ in range(num_reqs):
                if rng.random() < 0.25:
                    per_req.append(
                        rng.randint(1, SIM_MAX_TOKENS // MAX_REQS)
                    )  # prefill chunk
                else:
                    per_req.append(NUM_SPEC + 1)  # verify tokens
            n_tokens = sum(per_req)
            starts = [sum(per_req[:i]) for i in range(num_reqs)]
            last_idx = [st + rng.randint(0, q - 1) for st, q in zip(starts, per_req)]
            self.target_forward(f"step{s}:target", n_tokens)
            between = None
            if self.overlap:
                between = lambda n=n_tokens, s=s: self.target_forward(
                    f"step{s}:overlap", n
                )
            self.propose(f"step{s}", n_tokens, num_reqs, last_idx, between)
        return self


def _summary(sim):
    ptrs: dict[str, set[int]] = {}
    for ph, actor, op, ptr, n in sim.tr.events:
        ptrs.setdefault(actor, set()).add(ptr)
    return {
        "events": len(sim.tr.events),
        "hazards": len(sim.tr.hazards),
        "first_hazards": sim.tr.hazards[:3],
        "value_mismatches": sim.mismatch,
        "distinct_buffers_target": len(ptrs.get("target", ())),
        "distinct_buffers_draft": len(ptrs.get("draft", ())),
        "shared": ptrs.get("target", set()) == ptrs.get("draft", set()),
    }


def test_loader_shares_the_qsa_layer_buffer(monkeypatch):
    target, draft = build_models()
    old_buffer = weakref.ref(draft.model.layers[0].self_attn.topk_indices_buffer)
    load(monkeypatch, target, draft)
    shared = target.model.topk_indices_buffer
    assert draft.model.topk_indices_buffer is shared
    assert draft.model.layers[0].self_attn.topk_indices_buffer is shared
    assert old_buffer() is None


@pytest.mark.parametrize("method", ["eagle", "eagle3", "dflash", "draft_model", None])
def test_other_methods_keep_private_qsa_buffer(monkeypatch, method):
    target, draft = build_models()
    own = draft.model.layers[0].self_attn.topk_indices_buffer
    load(monkeypatch, target, draft, method=method)
    assert draft.model.layers[0].self_attn.topk_indices_buffer is own


@pytest.mark.parametrize("async_scheduling", [True, None])
def test_async_or_unresolved_scheduling_keeps_private_buffer(
    monkeypatch, async_scheduling
):
    target, draft = build_models()
    own = draft.model.layers[0].self_attn.topk_indices_buffer
    load(monkeypatch, target, draft, async_scheduling=async_scheduling)
    assert draft.model.layers[0].self_attn.topk_indices_buffer is own


@pytest.mark.parametrize("count", [0, 2])
def test_multiple_draft_layers_keep_independent_step_zero_indices(monkeypatch, count):
    target, draft = build_models()
    draft.model.layers = nn.ModuleList()
    for _ in range(count):
        layer = nn.Module()
        layer.self_attn = attention()
        draft.model.layers.append(layer)
    original = [a.topk_indices_buffer for a in draft.model._iter_qsa_attentions()]
    load(monkeypatch, target, draft)
    current = [a.topk_indices_buffer for a in draft.model._iter_qsa_attentions()]
    assert all(a is b for a, b in zip(original, current))
    if count == 2:
        for i, a in enumerate(draft.model._iter_qsa_attentions()):
            a.topk_indices_buffer.fill_(i + 1)
        draft.model.compact_topk_indices(torch.tensor([3, 1]))
        for i, a in enumerate(draft.model._iter_qsa_attentions()):
            assert (a.topk_indices_buffer[:2] == i + 1).all()


@pytest.mark.parametrize(
    "shape,dtype,device",
    [
        ((SIM_MAX_TOKENS - 1, SIM_WIDTH), torch.int32, "cpu"),
        ((SIM_MAX_TOKENS, SIM_WIDTH + 1), torch.int32, "cpu"),
        ((SIM_MAX_TOKENS,), torch.int32, "cpu"),
        ((SIM_MAX_TOKENS, SIM_WIDTH), torch.int64, "cpu"),
        ((SIM_MAX_TOKENS, SIM_WIDTH), torch.int32, "meta"),
    ],
)
def test_incompatible_target_buffer_keeps_private_buffer(
    monkeypatch, shape, dtype, device
):
    target, draft = build_models()
    target.model.topk_indices_buffer = torch.empty(shape, dtype=dtype, device=device)
    own = draft.model.layers[0].self_attn.topk_indices_buffer
    load(monkeypatch, target, draft)
    assert draft.model.layers[0].self_attn.topk_indices_buffer is own


def test_larger_target_buffer_is_usable(monkeypatch):
    target, draft = build_models()
    target.model.topk_indices_buffer = torch.empty(
        SIM_MAX_TOKENS + 1, SIM_WIDTH, dtype=torch.int32
    )
    load(monkeypatch, target, draft)
    assert (
        draft.model.layers[0].self_attn.topk_indices_buffer
        is target.model.topk_indices_buffer
    )


@pytest.mark.parametrize("async_scheduling", [False, True])
def test_workspace_selection_is_logged(monkeypatch, async_scheduling):
    messages = []
    monkeypatch.setattr(
        utils.logger, "info", lambda message, *args: messages.append(message % args)
    )
    target, draft = build_models()
    load(monkeypatch, target, draft, async_scheduling=async_scheduling)
    expected = "Keeping private" if async_scheduling else "Sharing target"
    assert any(expected in message for message in messages)


def test_alias_remains_a_nonpersistent_buffer_and_apply_can_break_it(monkeypatch):
    target, draft = build_models()
    load(monkeypatch, target, draft)
    name = "layers.0.self_attn.topk_indices_buffer"
    shared = target.model.topk_indices_buffer
    assert dict(draft.model.named_buffers())[name] is shared
    assert name not in draft.model.state_dict()
    # Sharing is installed after loader device/dtype transforms. A later
    # allocating _apply (including device moves) requires rebinding before
    # capturing graphs; the target's plain tensor attribute is not transformed.
    draft.model._apply(lambda t: t.clone())
    assert dict(draft.model.named_buffers())[name] is not shared
    assert target.model.topk_indices_buffer is shared


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_order_shared_vs_separate(monkeypatch, seed):
    t1, d1 = build_models()
    load(monkeypatch, t1, d1)
    # force sharing (what the proposal does) regardless of loader version
    d1.model.layers[0].self_attn.topk_indices_buffer = t1.model.topk_indices_buffer
    t2, d2 = build_models()  # separate buffers (0.1.1 behaviour)
    d2.model.layers[0].self_attn.topk_indices_buffer = torch.full(
        (SIM_MAX_TOKENS, SIM_WIDTH), -7, dtype=torch.int32
    )
    shared = Sim(t1, d1).run(seed)
    sep = Sim(t2, d2).run(seed)
    s1, s2 = _summary(shared), _summary(sep)
    assert s1["shared"] and not s2["shared"]
    assert s1["hazards"] == 0 and s2["hazards"] == 0
    assert s1["value_mismatches"] == 0 and s2["value_mismatches"] == 0
    assert shared.reads == sep.reads


@pytest.mark.parametrize("num_reqs,padded", [(1, 4), (3, 4), (5, 8), (7, 8), (8, 8)])
def test_compact_and_decode_metadata_keep_padding_outside_real_rows(
    monkeypatch, num_reqs, padded
):
    # This checks compaction and real CPU side metadata, not GPU masking.
    monkeypatch.setattr(torch_utils, "PIN_MEMORY", False)
    real_indices = torch.arange(num_reqs * SIM_WIDTH, dtype=torch.int32).view(
        num_reqs, SIM_WIDTH
    )
    for shared in (False, True):
        target, draft = build_models()
        buf = draft.model.layers[0].self_attn.topk_indices_buffer
        if shared:
            buf = target.model.topk_indices_buffer
            draft.model.layers[0].self_attn.topk_indices_buffer = buf
        # Different stale padding is permitted; compaction must touch only
        # actual requests, and metadata must mark the other rows inert.
        buf.fill_(123456 if shared else -7)
        buf[MAX_REQS : MAX_REQS + num_reqs].copy_(real_indices)
        tail = buf[num_reqs:padded].clone()
        sp = Sim(target, draft).sp
        sp.last_token_indices[:num_reqs] = torch.arange(MAX_REQS, MAX_REQS + num_reqs)
        sp._mtp_prefill_end(num_reqs)
        assert torch.equal(buf[:num_reqs], real_indices)
        assert torch.equal(buf[num_reqs:padded], tail)

        sp.draft_attn_layer_names = {"draft"}
        sp.arange = torch.arange(MAX_REQS + 1, dtype=torch.int32)
        sp.max_model_len = 64
        slots = torch.tensor(list(range(num_reqs)) + [-1] * (padded - num_reqs))
        starts = sp.arange[: padded + 1].clamp(max=num_reqs)
        tables = torch.ones(padded, 2, dtype=torch.int32)
        sp.block_tables = SimpleNamespace(
            input_block_tables=[tables], slot_mappings=slots.unsqueeze(0)
        )
        sp.input_buffers = SimpleNamespace(
            query_start_loc=starts,
            seq_lens=torch.tensor([9] * num_reqs + [0] * (padded - num_reqs)),
        )
        sp.attn_groups = []
        sp.kv_cache_config = object()

        def build_metadata(_slots=slots, _tables=tables, **kwargs):
            return CommonAttentionMetadata(
                num_actual_tokens=kwargs["num_tokens"],
                num_reqs=kwargs["num_reqs"],
                max_query_len=kwargs["max_query_len"],
                max_seq_len=kwargs["max_seq_len"],
                query_start_loc=kwargs["query_start_loc_gpu"],
                query_start_loc_cpu=kwargs["query_start_loc_cpu"],
                seq_lens=kwargs["seq_lens"],
                slot_mapping=_slots,
                block_table_tensor=_tables,
            )

        monkeypatch.setattr(spec_mod, "build_attn_metadata", build_metadata)
        common = sp._build_draft_attn_metadata(num_reqs, padded, padded)
        _, positions, mapped_slots = _build_qsa_metadata_torch(
            common,
            torch.empty(padded, dtype=torch.int32),
            torch.empty(padded, dtype=torch.int64),
            torch.empty(padded, dtype=torch.int64),
            storage_block_size=4,
            compress_ratio=1,
        )
        assert positions[:num_reqs].tolist() == [8] * num_reqs
        assert positions[num_reqs:].tolist() == [-1] * (padded - num_reqs)
        assert mapped_slots[num_reqs:].tolist() == [-1] * (padded - num_reqs)


def test_negative_control_target_between_prefill_and_decode(monkeypatch):
    t1, d1 = build_models()
    load(monkeypatch, t1, d1)
    d1.model.layers[0].self_attn.topk_indices_buffer = t1.model.topk_indices_buffer
    t2, d2 = build_models()
    sh = Sim(t1, d1, overlap=True).run(0, steps=50)
    sp = Sim(t2, d2, overlap=True).run(0, steps=50)
    # A target forward between compaction and decode corrupts only the shared case.
    assert _summary(sh)["hazards"] == 150 and sh.mismatch == 150
    assert _summary(sp)["hazards"] == 0 and sp.mismatch == 0


def test_workspace_byte_size():
    nbytes = (
        PROD_MAX_TOKENS
        * PROD_WIDTH
        * torch.tensor([], dtype=torch.int32).element_size()
    )
    assert nbytes == 67_207_168
