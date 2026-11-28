# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from vllm import envs
from vllm.compilation.sm70_decode_graph import sm70_decode_graph_compilation
from vllm.distributed import communication_op
from vllm.distributed.device_communicators import sm70_hc_ll as hc_ll


def assert_bits(actual, expected):
    assert torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@torch.inference_mode()
@pytest.mark.parametrize("rows", [1, 64, 65])
def test_mtp_fc_peer_runtime_rows_and_fallback(rows, monkeypatch):
    pytest.importorskip("torchvision")
    from vllm.models.qwen4_exp.nvidia import mtp

    monkeypatch.setattr(envs, "VLLM_SM70_QWEN38_DUAL_COMPILE", True)

    def gather(shard, dim=-1):
        return torch.cat(tuple(shard + rank for rank in range(4)), dim=dim)

    monkeypatch.setattr(
        communication_op, "get_tp_group", lambda: SimpleNamespace(all_gather=gather)
    )
    peer = Mock(side_effect=lambda e, h: gather(e.unsqueeze(-2) + h).flatten(-2))
    monkeypatch.setattr(
        hc_ll, "get_tp_hc_ll", lambda: SimpleNamespace(add_all_gather_fc=peer)
    )
    e = torch.randn(rows, 640, dtype=torch.float16)
    h = torch.randn(rows, 4, 640, dtype=torch.float16)
    expected = gather(e.unsqueeze(-2) + h).flatten(-2)
    predictor = SimpleNamespace(
        hc_count=4,
        hidden_size=2560,
        _sm70_nvlink_mtp_fc_peer=True,
        pre_fc_norm_embedding=nn.Identity(),
        pre_fc_norm_hidden=nn.Identity(),
        fc_embedding=lambda x: x[..., :640].contiguous(),
        fc_hidden=lambda x: x[..., :640].contiguous(),
    )
    project = mtp.Qwen4ExpMultiTokenPredictor._project_inputs
    embeds = F.pad(e, (0, 1920))
    hidden = F.pad(h, (0, 1920)).flatten(-2)
    assert_bits(project(predictor, embeds, hidden), expected)
    assert not peer.called
    with sm70_decode_graph_compilation():
        result = project(predictor, embeds, hidden)
    assert peer.call_count == int(rows <= 64)
    assert_bits(result, expected)
