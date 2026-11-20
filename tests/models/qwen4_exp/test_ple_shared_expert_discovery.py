# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

import pytest
import torch
from torch import nn


@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_shared_expert_stream_requires_accelerator_storage(monkeypatch, device):
    from vllm.model_executor.layers.fused_moe.fused_moe_method_base import (
        FusedMoEMethodBase,
    )
    from vllm.model_executor.layers.fused_moe.runner import shared_experts as module

    layer = nn.Linear(8, 16, device="cpu" if device == "cuda" else device)
    if device == "cuda":
        # Exercise the admission check without allocating GPU storage.
        monkeypatch.setattr(
            layer, "parameters", lambda: iter([Mock(device=torch.device(device))])
        )
    make_stream = Mock()
    monkeypatch.setenv("VLLM_DISABLE_SHARED_EXPERTS_STREAM", "0")
    monkeypatch.setattr(module, "aux_stream", make_stream)

    shared = module.SharedExperts(
        layer, Mock(), Mock(spec=FusedMoEMethodBase), enable_dbo=False
    )

    if device == "cuda":
        make_stream.assert_called_once_with()
        assert shared._stream is make_stream.return_value
    else:
        make_stream.assert_not_called()
        assert shared._stream is None
