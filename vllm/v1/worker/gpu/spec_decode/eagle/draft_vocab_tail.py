# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Global, fixed-slot MTP vocabulary tail for the V2 worker.

LRU semantics follow update_dynamic_draft_vocab_lru_state in
vllm.v1.spec_decode.static_draft_vocab (inherited from 1Cat).
"""

from typing import Any

import torch

from vllm.distributed import tensor_model_parallel_all_gather
from vllm.triton_utils import tl, triton
from vllm.v1.spec_decode.static_draft_vocab import StaticDraftVocabRuntime
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample


@triton.jit
def global_tail_update(
    slot_token,
    slot_stamp,
    clock,
    changed,
    base_mask,
    sampled,
    num_sampled,
    draft_logits,
    num_reqs,
    sampled_stride: tl.constexpr,
    full_vocab: tl.constexpr,
    tokenizer_vocab: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
    RK: tl.constexpr,
):
    slots = tl.arange(0, BLOCK)
    tokens = tl.load(slot_token + slots, slots < T, other=-1)
    stamps = tl.load(slot_stamp + slots, slots < T, other=0x7FFFFFFFFFFFFFFF)
    changes = tl.full((BLOCK,), False, tl.int1)
    tick = tl.load(clock)
    for r in range(num_reqs):
        count = tl.load(num_sampled + r)
        for j in range(count):
            token = tl.load(sampled + r * sampled_stride + j)
            if (token >= 0) & (token < tokenizer_vocab):
                is_base = tl.load(base_mask + token)
                if not is_base:
                    tick += 1
                    hit = (slots < T) & (tokens == token)
                    if tl.sum(hit.to(tl.int32), 0) > 0:
                        stamps = tl.where(hit, tick, stamps)
                    else:
                        oldest = tl.min(stamps, 0)
                        victim = tl.min(
                            tl.where(
                                (slots < T) & (stamps == oldest), slots, 0x7FFFFFFF
                            ),
                            0,
                        )
                        if RK > 0:
                            old_token = tl.sum(tl.where(slots == victim, tokens, 0), 0)
                            rows = tl.arange(0, triton.next_power_of_2(RK))
                            tl.store(
                                draft_logits + rows * full_vocab + old_token,
                                float("-inf"),
                                rows < RK,
                            )
                        replace = slots == victim
                        tokens = tl.where(replace, token, tokens)
                        stamps = tl.where(replace, tick, stamps)
                        changes = changes | replace
    tl.store(slot_token + slots, tokens, slots < T)
    tl.store(slot_stamp + slots, stamps, slots < T)
    tl.store(changed + slots, changes, slots < T)
    tl.store(clock, tick)


@triton.jit
def _refresh_tail(
    slot_token,
    changed,
    source,
    weight,
    valid,
    ext_map,
    B: tl.constexpr,
    T: tl.constexpr,
    H: tl.constexpr,
    SOURCE_STRIDE: tl.constexpr,
    SHARD: tl.constexpr,
    RANK: tl.constexpr,
    TP: tl.constexpr,
    SINK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    slot = tl.program_id(0)
    if tl.load(changed + slot):
        token = tl.load(slot_token + slot)
        local = (token >= RANK * SHARD) & (token < (RANK + 1) * SHARD)
        cols = tl.arange(0, BLOCK)
        row = tl.load(
            source + (token - RANK * SHARD) * SOURCE_STRIDE + cols,
            local & (cols < H),
            other=0,
        )
        tl.store(weight + slot * H + cols, row, local & (cols < H))
        tl.store(valid + slot, local)
        for rank in tl.static_range(TP):
            owned = (token >= rank * SHARD) & (token < (rank + 1) * SHARD)
            tl.store(ext_map + B + rank * T + slot, tl.where(owned, token, SINK))
        tl.store(changed + slot, False)


@triton.jit
def _merge_local(
    base_values,
    base_keys,
    tail_logits,
    tail_tokens,
    tail_valid,
    pairs,
    B: tl.constexpr,
    T: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    slots = tl.arange(0, BLOCK)
    valid = tl.load(tail_valid + slots, slots < T, other=False)
    vals = tl.load(tail_logits + row * T + slots, slots < T, other=float("-inf")).to(
        tl.float32
    )
    keys = B + tl.load(tail_tokens + slots, slots < T, other=0)
    vals = tl.where(valid & (tl.abs(vals) < float("inf")), vals, float("-inf"))
    keys = tl.where(valid & (slots < T), keys, 0x7FFFFFFF)
    best = tl.max(vals, 0)
    key = tl.min(tl.where(vals == best, keys, 0x7FFFFFFF), 0)
    base_val = tl.load(base_values + row).to(tl.float32)
    base_key = tl.load(base_keys + row)
    base_val = tl.where(tl.abs(base_val) < float("inf"), base_val, float("-inf"))
    use_base = (base_val > best) | ((base_val == best) & (base_key < key))
    tl.store(pairs + row * 2, tl.where(use_base, base_val, best))
    tl.store(pairs + row * 2 + 1, tl.where(use_base, base_key, key).to(tl.float32))


@triton.jit
def _merge_global(
    pairs, token_map, output, B: tl.constexpr, TP: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    ranks = tl.arange(0, BLOCK)
    vals = tl.load(pairs + row * TP * 2 + ranks * 2, ranks < TP, other=float("-inf"))
    keys = tl.load(
        pairs + row * TP * 2 + ranks * 2 + 1, ranks < TP, other=0x7FFFFFFF
    ).to(tl.int64)
    best = tl.max(vals, 0)
    key = tl.min(tl.where((ranks < TP) & (vals == best), keys, 0x7FFFFFFF), 0)
    base_token = tl.load(token_map + key, key < B, other=0)
    tl.store(output + row, tl.where(key < B, base_token, key - B))


def update_reference(
    slot_token: torch.Tensor,
    slot_stamp: torch.Tensor,
    clock: torch.Tensor,
    sampled: torch.Tensor,
    num_sampled: torch.Tensor,
    base_mask: torch.Tensor,
    tokenizer_vocab_size: int,
) -> None:
    """Pure torch CPU oracle, using the same fixed physical slots."""
    for r in range(sampled.shape[0]):
        for token in sampled[r, : int(num_sampled[r])]:
            token_id = int(token)
            if token_id < 0 or token_id >= tokenizer_vocab_size or base_mask[token_id]:
                continue
            clock += 1
            hit = slot_token == token_id
            if hit.any():
                slot_stamp[hit] = clock
            else:
                victim = slot_stamp.argmin()
                slot_token[victim] = token_id
                slot_stamp[victim] = clock


def merge_top1_reference(values: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    """Pure torch total-order reduction: value descending, key ascending."""
    values = values.masked_fill(~torch.isfinite(values), -torch.inf)
    best = values.max(-1, keepdim=True).values
    return keys.masked_fill(values != best, torch.iinfo(torch.int64).max).min(-1).values


class GlobalDraftVocabTail:
    def __init__(
        self,
        runtime: StaticDraftVocabRuntime,
        source_head: Any,
        ranking_path: str,
        tail_size: int,
        max_num_reqs: int,
        num_steps: int,
        draft_logits: torch.Tensor | None,
        tp_size: int,
        tp_rank: int,
    ) -> None:
        payload = torch.load(ranking_path, map_location="cpu", weights_only=True)
        ranking = payload["global_ranking"]
        self.base_size = runtime.shortlist_size
        self.tail_size = tail_size
        self.full_vocab_size = runtime.full_vocab_size
        self.tokenizer_vocab_size = payload["tokenizer_vocab_size"]
        if ranking.numel() < self.base_size + tail_size:
            raise ValueError(
                "Global MTP draft tail requires ranking length >= base + tail."
            )
        if source_head.weight.dtype != torch.float16:
            raise ValueError("Global MTP draft tail requires an fp16 LM-head.")
        processor = runtime.logits_processor
        if processor.soft_cap is not None or processor.scale != 1.0:
            raise ValueError("Global MTP draft tail requires no soft cap and scale=1.")
        base_ids = set(runtime.token_id_map.tolist())
        self.sink = next(
            (
                i
                for i in range(self.tokenizer_vocab_size, self.full_vocab_size)
                if i not in base_ids
            ),
            -1,
        )
        if self.sink < 0:
            raise ValueError(
                "Global MTP draft tail requires a padding token outside "
                "the base vocabulary for SINK."
            )
        seed = [
            token_id
            for token_id in ranking[self.base_size :].tolist()
            if token_id < self.tokenizer_vocab_size and token_id not in base_ids
        ][:tail_size]
        if len(seed) < tail_size:
            raise ValueError(
                "Global MTP draft tail requires at least tail_size non-padding "
                "seeds outside the base vocabulary after ranking position base."
            )
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.runtime = runtime
        self.source_weight = source_head.weight
        self.shard_width = source_head.num_embeddings_per_partition
        self.draft_logits = draft_logits
        device = self.source_weight.device
        T, B = tail_size, self.base_size
        self.slot_token = torch.tensor(seed, dtype=torch.int64, device=device)
        self.slot_stamp = torch.arange(1, T + 1, dtype=torch.int64, device=device)
        self.clock = torch.tensor([T], dtype=torch.int64, device=device)
        self.changed = torch.ones(T, dtype=torch.bool, device=device)
        self.base_mask = torch.zeros(
            self.full_vocab_size, dtype=torch.bool, device=device
        )
        self.base_mask[runtime.token_id_map] = True
        self.tail_weight = torch.empty(
            (T, self.source_weight.shape[1]), dtype=torch.float16, device=device
        )
        self.tail_valid = torch.empty(T, dtype=torch.bool, device=device)
        self.ext_token_map = torch.empty(
            B + tp_size * T, dtype=torch.int64, device=device
        )
        self.ext_token_map[:B].copy_(runtime.token_id_map)
        self.tail_logits = torch.empty(
            (max_num_reqs, T), dtype=torch.float16, device=device
        )
        self.local_pair = torch.empty(
            (max_num_reqs, 2), dtype=torch.float32, device=device
        )
        self.out_tokens = torch.empty(max_num_reqs, dtype=torch.int64, device=device)
        self.ext_logits = torch.empty(
            (max_num_reqs, B + tp_size * T), dtype=torch.float16, device=device
        )
        self.reduced_draft_logits = (
            torch.full(
                (max_num_reqs, num_steps, B + tp_size * T),
                -torch.inf,
                dtype=torch.float32,
                device=device,
            )
            if draft_logits is not None
            else None
        )
        self._refresh()

    def _refresh(self) -> None:
        _refresh_tail[(self.tail_size,)](
            self.slot_token,
            self.changed,
            self.source_weight,
            self.tail_weight,
            self.tail_valid,
            self.ext_token_map,
            self.base_size,
            self.tail_size,
            self.source_weight.shape[1],
            self.source_weight.stride(0),
            self.shard_width,
            self.tp_rank,
            self.tp_size,
            self.sink,
            triton.next_power_of_2(self.source_weight.shape[1]),
        )

    def update(
        self, sampled: torch.Tensor, num_sampled: torch.Tensor, num_reqs: int
    ) -> None:
        global_tail_update[(1,)](
            self.slot_token,
            self.slot_stamp,
            self.clock,
            self.changed,
            self.base_mask,
            sampled,
            num_sampled,
            self.draft_logits,
            num_reqs,
            sampled.stride(0),
            self.full_vocab_size,
            self.tokenizer_vocab_size,
            self.tail_size,
            triton.next_power_of_2(self.tail_size),
            self.draft_logits.shape[0] * self.draft_logits.shape[1]
            if self.draft_logits is not None
            else 0,
        )
        self._refresh()

    def _tail_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        logits = self.tail_logits[: hidden.shape[0]]
        torch.mm(hidden, self.tail_weight.t(), out=logits)
        return logits

    def sample_greedy(self, hidden: torch.Tensor) -> torch.Tensor:
        values, keys = self.runtime.logits_processor.get_local_top1(
            self.runtime.lm_head, hidden
        )
        tail_logits = self._tail_logits(hidden)
        rows = hidden.shape[0]
        pair = self.local_pair[:rows]
        _merge_local[(rows,)](
            values,
            keys,
            tail_logits,
            self.slot_token,
            self.tail_valid,
            pair,
            self.base_size,
            self.tail_size,
            triton.next_power_of_2(self.tail_size),
        )
        gathered = (
            tensor_model_parallel_all_gather(pair, dim=-1) if self.tp_size > 1 else pair
        )
        output = self.out_tokens[:rows]
        _merge_global[(rows,)](
            gathered,
            self.runtime.token_id_map,
            output,
            self.base_size,
            self.tp_size,
            triton.next_power_of_2(self.tp_size),
        )
        return output

    def _combined_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        base = self.runtime.logits_processor(self.runtime.lm_head, hidden)
        tail = self._tail_logits(hidden)
        tail.masked_fill_(~self.tail_valid, -torch.inf)
        gathered = (
            tensor_model_parallel_all_gather(tail, dim=-1) if self.tp_size > 1 else tail
        )
        logits = self.ext_logits[: hidden.shape[0]]
        logits[:, : self.base_size].copy_(base)
        logits[:, self.base_size :].copy_(gathered)
        return logits

    def sample_probabilistic(
        self,
        hidden: torch.Tensor,
        idx_mapping: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        pos: torch.Tensor,
        draft_step: torch.Tensor,
        draft_logits: torch.Tensor | None,
        use_fp64: bool,
    ) -> torch.Tensor:
        logits = self._combined_logits(hidden)
        if draft_logits is None:
            sampled = logits.argmax(-1)
        else:
            sampled = gumbel_sample(
                logits,
                idx_mapping,
                temperature,
                seeds,
                pos + 1,
                apply_temperature=True,
                is_drafting=True,
                output_processed_logits=self.reduced_draft_logits,
                output_processed_logits_col=draft_step,
                use_fp64=use_fp64,
            )
            draft_logits.scatter_(
                2,
                self.ext_token_map.view(1, 1, -1).expand_as(self.reduced_draft_logits),
                self.reduced_draft_logits,
            )
        return self.ext_token_map[sampled]
