# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch._subclasses.fake_tensor import FakeTensorMode

from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
from vllm.config import KernelConfig
from vllm.models.qwen4_exp.nvidia import indexer_qsa, qsa
from vllm.models.qwen4_exp.nvidia import sm70_fp16_gemv as gemv


@pytest.fixture
def should_do_global_cleanup_after_test():
    return False


def test_target_pair_packing_and_dispatch(monkeypatch):
    assert not KernelConfig().sm70_qsa_input_batch
    qkv = torch.arange(3584 * 2560, dtype=torch.int32).reshape(3584, 2560)
    qkv = (qkv % 511).half()
    index = qkv[:640].clone()
    packs = tuple(gemv._pack_qsa_input_weight(t) for t in (qkv, index))
    assert [p.shape for p in packs] == [(112, 160, 2, 32, 8), (20, 160, 2, 32, 8)]
    for original, packed in zip((qkv, index), packs, strict=True):
        assert packed.is_contiguous()
        assert packed.nbytes == original.nbytes
        assert torch.equal(
            packed.permute(0, 3, 1, 2, 4).reshape(original.shape), original
        )

    native = Mock()
    monkeypatch.setattr(
        torch.ops._C, "qwen38_qsa_input_mtp_batch_sm70_out", native, raising=False
    )
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda _: True))
    fallback = Mock(
        side_effect=lambda x, w, role: x.new_zeros((x.shape[0], w.shape[0]))
    )
    monkeypatch.setattr(gemv, "_qwen38_sm70_fp16_gemv", fallback)
    for m in (1, 3, 4, 12, 48, 64, 65):
        native.reset_mock()
        fallback.reset_mock()
        x = torch.empty(m, 2560, dtype=torch.float16)
        a, b = gemv._qwen38_sm70_fp16_qsa_input(x, qkv, index, *packs)
        assert a.shape == (m, 3584) and b.shape == (m, 640)
        if 2 <= m <= 64:
            native.assert_called_once_with(a, b, x, *packs)
            fallback.assert_not_called()
        else:
            native.assert_not_called()
            assert [c.args[2] for c in fallback.call_args_list] == [
                gemv._QSA_QKV_SUFFIX,
                gemv._QSA_INDEX_SUFFIX,
            ]
    with FakeTensorMode(allow_non_fake_inputs=True):
        x = torch.empty(48, 2560, dtype=torch.float16)
        out = torch.ops.vllm.qwen38_sm70_fp16_qsa_input(x, qkv, index, *packs)
        assert [p.shape for p in out] == [(48, 3584), (48, 640)]


def test_target_pair_preparation_scope(monkeypatch):
    class Linear(nn.Module):
        def __init__(self, prefix, rows):
            super().__init__()
            self.prefix = prefix
            self.weight = nn.Parameter(torch.empty(rows, 2560, device="meta"))
            self.bias = None
            self.quant_method = gemv.UnquantizedLinearMethod()

    monkeypatch.setattr(gemv, "LinearBase", Linear)
    monkeypatch.setattr(gemv, "_exact_runtime_contract", lambda _: True)
    monkeypatch.setattr(gemv, "_batch_runtime_contract", lambda _: True)
    monkeypatch.setattr(
        gemv, "current_platform", NS(is_device_capability=lambda _: True)
    )
    monkeypatch.setattr(
        gemv.torch.backends.cuda,
        "matmul",
        NS(
            allow_fp16_accumulation=False,
            allow_fp16_reduced_precision_reduction=False,
        ),
    )
    for flag in ("FP16_GEMV", "BATCH_FASTPATH", "DUAL_COMPILE"):
        monkeypatch.setenv(f"VLLM_SM70_QWEN38_{flag}", "1")
    config = NS(
        kernel_config=KernelConfig(),
        parallel_config=NS(tensor_parallel_size=4),
        model_config=NS(quantization="modelopt_fp4"),
        speculative_config=NS(use_qwen4_exp_mtp=lambda: True),
    )
    for enabled, target in ((False, True), (True, False), (True, True)):
        # Only QSA's two projections participate in model-load preparation.
        owner = qsa.Qwen4ExpQSAAttention.__new__(qsa.Qwen4ExpQSAAttention)
        nn.Module.__init__(owner)
        owner.qkv_proj = Linear("model.layers.3.self_attn.qkv_proj", 3584)
        owner.indexer = nn.Module()
        owner.indexer.index_qk_proj = Linear(
            "model.layers.3.self_attn.indexer.index_qk_proj", 640
        )
        config.kernel_config.sm70_qsa_input_batch = enabled
        gemv.enable_qwen38_sm70_fp16_gemv(
            owner, torch.float16, config, target_qsa=target
        )
        for linear in (owner.qkv_proj, owner.indexer.index_qk_proj):
            assert isinstance(linear.quant_method, gemv.Qwen38SM70FP16LinearMethod)
            assert hasattr(linear, "_sm70_prepare_qsa_input_batch") == (
                enabled and target
            )


def test_target_forward_passes_projected_indices_only_during_decode(monkeypatch):
    x, packed = torch.empty(4, 2560), torch.empty(0)
    qkv, idx = torch.empty(4, 3584), torch.arange(2560).reshape(4, 640).half()
    q_proj = Mock(
        return_value=(qkv, None), weight=packed, _sm70_qsa_input_packed=packed
    )
    index_proj = Mock(weight=packed, _sm70_qsa_input_packed=packed, output_size=640)
    pair = Mock(return_value=(qkv, idx))
    monkeypatch.setattr(qsa, "current_platform", NS(opaque_attention_op=lambda: False))
    monkeypatch.setattr(torch.ops.vllm, "qwen38_sm70_fp16_qsa_input", pair)
    attention_call = Mock()
    monkeypatch.setattr(qsa, "qwen4_exp_qsa_with_output", attention_call)
    owner = NS(
        qkv_proj=q_proj,
        indexer=NS(index_qk_proj=index_proj),
        num_heads=6,
        num_kv_heads=1,
        head_dim=256,
        layer_name="model.layers.3.self_attn",
        _project_qkv_gate=lambda *a: (
            torch.empty(4, 1536),
            torch.empty(4, 256),
            torch.empty(4, 256),
            torch.empty(4, 1536),
        ),
        o_proj=lambda y: (y, None),
    )
    with sm70_decode_graph_compilation(False):
        qsa.Qwen4ExpQSAAttention.forward(owner, torch.arange(4), None, x)
    assert attention_call.call_args.args[0] is x
    pair.assert_not_called()
    with sm70_decode_graph_compilation():
        qsa.Qwen4ExpQSAAttention.forward(owner, torch.arange(4), None, x)
    assert attention_call.call_args.args[0] is idx
    pair.assert_called_once()
    q_proj.assert_called_once()

    raw = NS(
        num_actual_tokens=3,
        slot_mapping=None,
        block_table=None,
        query_start_loc=None,
        logical_positions=None,
    )
    side_input = Mock()
    monkeypatch.setattr(indexer_qsa, "qsa_pre_indexer", side_input)
    indexer = NS(
        _metadata=lambda: (raw, NS(slot_mapping=None, k_work_metadata=None)),
        index_qk_proj=index_proj,
        index_n_heads=4,
        index_kv_heads=1,
        index_head_dim=128,
        use_fused_pre_indexer=True,
        skip_topk=True,
        compress_ratio=4,
        rotary_emb=NS(cos_sin_cache=None),
        q_layernorm=NS(weight=None, variance_epsilon=1e-6),
        k_layernorm=NS(weight=None),
        raw_key_cache=NS(kv_cache=None, rope_position_cache=None),
        compressed_key_cache=NS(kv_cache=None),
    )
    out = torch.empty(3, 2051, dtype=torch.int32)
    assert indexer_qsa.QSAIndexer.forward(indexer, idx, torch.arange(4), out) is out
    index_proj.assert_not_called()
    torch.testing.assert_close(side_input.call_args.args[0], idx[:3, :512])
    torch.testing.assert_close(side_input.call_args.args[1], idx[:3, 512:])
