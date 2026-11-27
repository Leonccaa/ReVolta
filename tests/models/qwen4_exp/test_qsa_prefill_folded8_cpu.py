# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU checks for the 1Cat TP4 QSA prefill row partition."""

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia import indexer_qsa
from vllm.models.qwen4_exp.nvidia.ops import qsa

pytestmark = pytest.mark.skip_global_cleanup


def test_model_quantization_and_parallel_gate(monkeypatch):
    import vllm.distributed.parallel_state as parallel_state

    group = NS(world_size=4, rank_in_group=0)
    get_group = Mock(return_value=group)
    monkeypatch.setattr(parallel_state, "get_tp_group", get_group)
    monkeypatch.setattr(indexer_qsa, "ReplicatedLinear", lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(indexer_qsa, "GemmaRMSNorm", lambda *a, **kw: nn.Identity())
    monkeypatch.setattr(indexer_qsa, "QSAKeyStateCache", lambda **kw: NS())
    monkeypatch.setattr(
        indexer_qsa, "QSACompressedKeyCache", lambda **kw: NS(prefix="compressed")
    )
    monkeypatch.setattr(indexer_qsa, "qsa_index_dcp_sharded", lambda *a: False)
    parallel = NS(
        tensor_parallel_size=4,
        pipeline_parallel_size=1,
        decode_context_parallel_size=2,
        prefill_context_parallel_size=1,
    )
    cfg = NS(
        cache_config=NS(),
        model_config=NS(dtype=torch.float16, uses_mrope=False),
        kernel_config=NS(sm70_qsa_shared_key=False, sm70_qsa_fused_topk=False),
        parallel_config=parallel,
    )
    model = NS(
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=2048,
        indexer_compress_ratio=4,
        hidden_size=2560,
    )
    rotary = NS(rotary_dim=64, is_neox_style=True)
    for name, layers, expected in [
        ("awq", {}, False),
        ("modelopt_fp4", {}, True),
        (
            "modelopt_mixed",
            {"model.layers.0.mlp.experts": {"quant_algo": "NVFP4"}},
            True,
        ),
        (
            "modelopt_mixed",
            {"model.layers.0.mlp.experts": {"quant_algo": "FP8"}},
            False,
        ),
        ("modelopt_mixed", {"visual.proj": {"quant_algo": "NVFP4"}}, False),
    ]:
        quant = NS(get_name=lambda name=name: name, quantized_layers=layers)
        for wrapped in (False, True):
            obj = indexer_qsa.QSAIndexer(
                vllm_config=cfg,
                config=model,
                layer_id=0,
                rotary_emb=rotary,
                quant_config=NS(fallback=quant) if wrapped else quant,
            )
            assert (obj.prefill_tp_group is group) == expected
    parallel.prefill_context_parallel_size = 2
    obj = indexer_qsa.QSAIndexer(
        vllm_config=cfg,
        config=model,
        layer_id=0,
        rotary_emb=rotary,
        quant_config=NS(get_name=lambda: "modelopt_fp4"),
    )
    assert obj.prefill_tp_group is None


def test_indexer_forward_passes_tp_group(monkeypatch):
    group = NS(world_size=4, rank_in_group=0)
    rows = 4
    raw = NS(
        num_actual_tokens=rows,
        slot_mapping=None,
        block_table=None,
        query_start_loc=None,
        logical_positions=None,
    )
    compressed = NS(
        slot_mapping=None,
        k_work_metadata=None,
        block_table=None,
        token_to_req=None,
        logical_positions=None,
        seq_lens=None,
        query_start_loc_cpu=None,
        max_seq_len=None,
        seq_lens_cpu_upper_bound=None,
        index_dcp_world_size=1,
        index_dcp_rank=0,
        prefill_ranges=(),
    )
    indexer = NS(
        _metadata=lambda: (raw, compressed),
        index_qk_proj=NS(_sm70_qsa_input_packed=True, output_size=640),
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        use_fused_pre_indexer=True,
        skip_topk=False,
        compress_ratio=4,
        token_topk=2048,
        rotary_emb=NS(cos_sin_cache=None),
        q_layernorm=NS(weight=None, variance_epsilon=1e-6),
        k_layernorm=NS(weight=None),
        raw_key_cache=NS(kv_cache=None, rope_position_cache=None),
        compressed_key_cache=NS(kv_cache=None),
        shared_key_scoring=False,
        fused_topk=False,
        prefill_tp_group=group,
    )
    select = Mock(return_value=torch.empty(0))
    monkeypatch.setattr(indexer_qsa, "qsa_pre_indexer", lambda *a, **kw: None)
    monkeypatch.setattr(qsa, "qsa_select_paged_tokens", select)
    monkeypatch.setattr(
        indexer_qsa, "get_forward_context", lambda: NS(is_uniform_decode=False)
    )
    indexer_qsa.QSAIndexer.forward(indexer, torch.empty(rows, 640), torch.arange(rows))
    assert select.call_args.kwargs["tp_group"] is group


class Scored(Exception):
    def __init__(self, visible):
        self.visible = visible


@pytest.mark.parametrize(
    "rows,split,folded",
    [
        (1023, False, False),
        (1024, True, False),
        (2047, True, False),
        (2048, True, True),
        (2049, True, False),
        (2056, True, True),
        (16384, True, True),
        (16385, False, False),
    ],
)
def test_real_select_route_boundaries(monkeypatch, rows, split, folded):
    # Use the supported legacy 256-row tuning to exercise the 1024 boundary.
    _route_setup(monkeypatch, min_rows=256)
    visible = torch.arange(rows, dtype=torch.int32)
    monkeypatch.setattr(qsa, "_qsa_visible_blocks", lambda *a, **kw: visible)
    monkeypatch.setattr(
        qsa, "_qsa_mqa_cublas", lambda q, k, valid, v: (_ for _ in ()).throw(Scored(v))
    )
    rank = 1
    group = NS(world_size=4, rank_in_group=rank)
    with pytest.raises(Scored) as exc:
        _select(rows, group=group)
    if not split:
        chunk_rows = qsa._LOGITS_WORKSPACE_BYTES // (8192 * 4)
        expected = visible[:chunk_rows]
    elif folded:
        h = rows // 8
        expected = torch.cat(
            (
                visible[rank * h : (rank + 1) * h],
                visible[(7 - rank) * h : (8 - rank) * h],
            )
        )
    else:
        step = (rows + 3) // 4
        expected = visible[rank * step : (rank + 1) * step]
    assert torch.equal(exc.value.visible, expected)


def _route_setup(monkeypatch, min_rows=512):
    monkeypatch.setattr(qsa, "_SM70_INDEXER_CUBLAS", True)
    monkeypatch.setattr(qsa, "_SM70_INDEXER_CUBLAS_MIN_ROWS", min_rows)
    monkeypatch.setattr(qsa.current_platform, "is_device_capability", lambda *_: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        qsa,
        "_qsa_gather_single_request_keys",
        lambda *a: (torch.empty(0), torch.empty(0)),
    )


def _select(rows, group=None, world=1, requests=1, columns=8192):
    query = torch.zeros(rows, 4, 128, dtype=torch.float16)
    cache = torch.empty(1, 64, 1, 128, dtype=torch.float16)
    table = torch.zeros(requests, columns // 64, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int32)
    return qsa.qsa_select_paged_tokens(
        query,
        cache,
        table,
        torch.zeros_like(positions),
        positions,
        torch.full((requests,), columns * 4, dtype=torch.int32),
        2048,
        4,
        max_seq_len=columns * 4,
        index_dcp_world_size=world,
        tp_group=group,
    )


def test_default_per_rank_and_work_thresholds(monkeypatch):
    _route_setup(monkeypatch)
    monkeypatch.setattr(qsa, "_qsa_visible_blocks", lambda t, *a, **kw: t)
    monkeypatch.setattr(
        qsa, "_qsa_mqa_cublas", lambda q, k, valid, v: (_ for _ in ()).throw(Scored(v))
    )
    group = NS(world_size=4, rank_in_group=0)
    for rows, columns, expected in [
        (1024, 8192, 1024),
        (2048, 8192, 512),
        (2048, 1984, 2048),
        (2048, 2048, 512),
    ]:
        with pytest.raises(Scored) as exc:
            _select(rows, group, columns=columns)
        assert exc.value.visible.numel() == expected


def test_dcp_column_shards_do_not_row_split(monkeypatch):
    _route_setup(monkeypatch)
    monkeypatch.setattr(qsa, "_qsa_visible_blocks", lambda t, *a, **kw: t)
    monkeypatch.setattr(
        qsa, "_qsa_mqa_cublas", lambda q, k, valid, v: (_ for _ in ()).throw(Scored(v))
    )
    with pytest.raises(Scored) as exc:
        _select(2048, NS(world_size=4, rank_in_group=0), world=4)
    assert exc.value.visible.numel() == 2048


def test_multi_request_never_uses_row_split(monkeypatch):
    _route_setup(monkeypatch)

    class Batched(Exception):
        pass

    monkeypatch.setattr(
        qsa, "qsa_mqa_paged", lambda *a, **kw: (_ for _ in ()).throw(Batched())
    )
    monkeypatch.setattr(qsa, "_use_sm70_qsa_request_segments", lambda *a: False)
    with pytest.raises(Batched):
        _select(2048, NS(world_size=4, rank_in_group=0), requests=2)


def test_mixed_batch_request_split_keeps_replicated_scoring(monkeypatch):
    _route_setup(monkeypatch)
    monkeypatch.setattr(qsa, "_qsa_visible_blocks", lambda t, *a, **kw: t)
    monkeypatch.setattr(
        qsa, "_qsa_mqa_cublas", lambda q, k, valid, v: (_ for _ in ()).throw(Scored(v))
    )
    real_select = qsa.qsa_select_paged_tokens

    def select(*args, **kwargs):
        if args[0].shape[0] < 512:
            return args[8]
        return real_select(*args, **kwargs)

    monkeypatch.setattr(qsa, "qsa_select_paged_tokens", select)
    rows = 2050
    with pytest.raises(Scored) as exc:
        real_select(
            torch.zeros(rows, 4, 128, dtype=torch.float16),
            torch.empty(1, 64, 1, 128, dtype=torch.float16),
            torch.zeros(2, 128, dtype=torch.int32),
            torch.zeros(rows, dtype=torch.int32),
            torch.arange(rows, dtype=torch.int32),
            torch.full((2,), 32768, dtype=torch.int32),
            2048,
            4,
            query_start_loc_cpu=torch.tensor([0, 2, rows]),
            max_seq_len=32768,
            tp_group=NS(world_size=4, rank_in_group=0),
        )
    assert exc.value.visible.numel() == 2048


@pytest.mark.parametrize("rows,folded", [(2048, True), (2049, False)])
def test_real_selector_gathers_all_chunks(monkeypatch, rows, folded):
    _route_setup(monkeypatch, min_rows=256)
    monkeypatch.setattr(qsa, "_LOGITS_WORKSPACE_BYTES", 170 * 8192 * 4)
    monkeypatch.setattr(
        qsa,
        "_qsa_visible_blocks",
        lambda t, *a, **kw: torch.arange(rows, dtype=torch.int32),
    )
    monkeypatch.setattr(qsa, "_use_sm70_qsa_lexicographic_topk", lambda k: True)
    monkeypatch.setattr(qsa, "_qsa_mqa_cublas", lambda q, k, valid, v: (v[:, None], v))

    def topk(logits, visible, out, k):
        out.fill_(-1)
        out[:, 0] = logits[:, 0]

    monkeypatch.setattr(qsa, "_sm70_qsa_lexicographic_topk_op", lambda: topk)
    step = (rows + 3) // 4
    expected = []
    for rank in range(4):
        if folded:
            h = rows // 8
            indices = torch.cat(
                (
                    torch.arange(rank * h, (rank + 1) * h),
                    torch.arange((7 - rank) * h, (8 - rank) * h),
                )
            )
        else:
            indices = torch.arange(rank * step, min((rank + 1) * step, rows))
        blocks = torch.full((step, 512), -1, dtype=torch.int32)
        blocks[: len(indices), 0] = indices.int()
        expected.append(blocks)

    def expand(blocks, positions, lengths, requests, ratio, topk, out, *, folded8):
        assert folded8 == folded
        assert torch.equal(blocks, torch.cat(expected)[:rows])
        return out

    monkeypatch.setattr(qsa, "expand_qsa_block_indices_cuda", expand)
    for rank in range(4):

        def gather(blocks, dim, rank=rank):
            assert dim == 0
            assert torch.equal(blocks, expected[rank])
            return torch.cat(expected)

        _select(rows, NS(world_size=4, rank_in_group=rank, all_gather=gather))


def test_folded8_expansion_and_small_geometry_topk(monkeypatch):
    # This Triton interpreter version incorrectly retains the integer operand
    # dtype for comparisons. Restore boolean comparison results for CPU only.
    import numpy as np
    from triton.runtime.interpreter import InterpreterBuilder, TensorHandle

    from vllm.triton_utils import tl

    for name, op in {
        "SLT": np.less,
        "SLE": np.less_equal,
        "SGT": np.greater,
        "SGE": np.greater_equal,
        "ULT": np.less,
        "ULE": np.less_equal,
        "UGT": np.greater,
        "UGE": np.greater_equal,
        "EQ": np.equal,
        "NE": np.not_equal,
    }.items():
        monkeypatch.setattr(
            InterpreterBuilder,
            f"create_icmp{name}",
            lambda self, lhs, rhs, op=op: TensorHandle(op(lhs.data, rhs.data), tl.int1),
        )
    # Exact tied scores deliberately exercise lower-column tie breaking.
    torch.manual_seed(4)
    rows, heads, dim, columns, k = 16, 2, 4, 12, 4
    queries = torch.randint(-2, 3, (rows, heads, dim), dtype=torch.float32)
    queries[0].zero_()
    keys = torch.randint(-2, 3, (columns, dim), dtype=torch.float32)
    keys[1] = keys[0]

    def select(q):
        scores = (q @ keys.T).clamp_min(0).sum(dim=1)
        return (
            torch.argsort(scores, dim=1, descending=True, stable=True)[:, :k]
            .sort(dim=1)
            .values.to(torch.int32)
        )

    whole = select(queries)
    assert whole[0].tolist() == list(range(k))
    assert bool(((whole == 0).any(dim=1) & (whole == 1).any(dim=1)).any())
    contiguous = torch.cat([select(chunk) for chunk in queries.chunk(4)])
    assert torch.equal(contiguous, whole)
    h = rows // 8
    shards = [
        select(
            torch.cat(
                (queries[r * h : (r + 1) * h], queries[(7 - r) * h : (8 - r) * h])
            )
        )
        for r in range(4)
    ]
    gathered = torch.cat(shards)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    positions = torch.full((rows,), 4 * k - 1, dtype=torch.int64)
    lengths = torch.full((1,), 4 * columns, dtype=torch.int32)
    req = torch.zeros(rows, dtype=torch.int32)
    baseline = qsa.expand_qsa_block_indices_cuda(
        whole, positions, lengths, req, 4, 4 * k
    )
    restored = qsa.expand_qsa_block_indices_cuda(
        gathered, positions, lengths, req, 4, 4 * k, folded8=True
    )
    assert torch.equal(restored, baseline)
    assert torch.equal(restored[:, : 4 * k : 4] // 4, whole)
