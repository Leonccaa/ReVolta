# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU coverage of the V2 static MTP vocabulary and proposal distribution."""

import os
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from vllm.v1.spec_decode.static_draft_vocab import StaticDraftVocabRuntime
from vllm.v1.worker.gpu.sample.gumbel import gumbel_sample
from vllm.v1.worker.gpu.spec_decode.eagle import speculator as eagle


def _speculator(monkeypatch, enabled, local_argmax=False):
    spec = object.__new__(eagle.EagleSpeculator)
    spec.method = "mtp"
    spec.device = torch.device("cpu")
    spec.max_num_reqs = 3
    spec.num_speculative_steps = 2
    spec.use_local_argmax_reduction = local_argmax
    spec.use_fp64_gumbel = False
    spec._static_draft_vocab = None
    spec._reduced_draft_logits = None
    spec.temperature = torch.tensor([0.7, 1.3, 0.0])
    spec.seeds = torch.tensor([123, 456, 789])
    spec.draft_logits = torch.zeros(3, 2, 8)
    spec.vllm_config = SimpleNamespace()
    spec.draft_model_config = SimpleNamespace(hf_config=SimpleNamespace())
    spec.speculative_config = SimpleNamespace(draft_sample_method="greedy")
    weight = torch.tensor(
        [
            [0.0, 1.0],
            [9.0, 0.0],
            [1.0, 2.0],
            [0.0, 8.0],
            [2.0, 3.0],
            [4.0, 1.0],
            [1.0, 4.0],
            [3.0, 1.0],
        ]
    )
    model = nn.Module()
    model.lm_head = nn.Linear(2, 8, bias=False)
    model.lm_head.weight.data.copy_(weight)
    model.model = nn.Module()
    model.compute_logits = Mock(side_effect=lambda h: h @ weight.T)
    model.get_top_tokens = Mock(side_effect=lambda h: (h @ weight.T).argmax(-1))
    token_map = torch.tensor([3, 0, 6, 4])
    head = nn.Linear(2, 4, bias=False)
    head.weight.data.copy_(weight[token_map])
    processor = Mock(side_effect=lambda head, h: h @ head.weight.T)
    processor.get_top_tokens = Mock(
        side_effect=lambda head, h: (h @ head.weight.T).argmax(-1)
    )
    runtime = StaticDraftVocabRuntime(head, processor, token_map, 8, 4, None)
    initialize = Mock(return_value=runtime)
    monkeypatch.setattr(eagle, "initialize_static_draft_vocab", initialize)
    monkeypatch.setattr(eagle, "load_eagle_model", Mock(return_value=model))
    monkeypatch.setattr(eagle, "get_layers_from_vllm_config", lambda *_args: {})
    monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_SIZE", "4" if enabled else "0")
    if enabled:
        monkeypatch.setenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING", "ranking.pt")
    else:
        monkeypatch.delenv("VLLM_SM70_MTP_STATIC_DRAFT_VOCAB_RANKING", raising=False)
    spec.load_model(nn.Module())
    return spec, runtime, initialize


@pytest.mark.parametrize("enabled", [False, True])
def test_v2_static_vocab_load(monkeypatch, enabled):
    spec, runtime, initialize = _speculator(monkeypatch, enabled)
    if enabled:
        initialize.assert_called_once_with(
            spec.model.lm_head, "ranking.pt", 4, torch.device("cpu")
        )
        assert spec.model.sm70_static_draft_lm_head is runtime.lm_head
        assert spec._static_draft_vocab is runtime
        assert spec._reduced_draft_logits.shape == (3, 2, 4)
        assert torch.isneginf(spec.draft_logits).all()
    else:
        initialize.assert_not_called()
        assert spec._static_draft_vocab is None
        assert spec._reduced_draft_logits is None
        assert torch.count_nonzero(spec.draft_logits) == 0


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("local_argmax", [False, True])
def test_v2_static_vocab_greedy_mapping(monkeypatch, enabled, local_argmax):
    spec, runtime, _ = _speculator(monkeypatch, enabled, local_argmax)
    hidden = torch.eye(2)
    actual = spec._sample_draft(
        hidden, torch.tensor([1, 0]), torch.tensor([9, 17]), torch.tensor(0), None
    )
    expected = torch.tensor([4, 3]) if enabled else torch.tensor([1, 3])
    assert actual.numpy().tobytes() == expected.numpy().tobytes()
    if enabled:
        spec.model.compute_logits.assert_not_called()
        spec.model.get_top_tokens.assert_not_called()
        if local_argmax:
            runtime.logits_processor.get_top_tokens.assert_called_once_with(
                runtime.lm_head, hidden
            )
            runtime.logits_processor.assert_not_called()


@pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET") != "1",
    reason="CPU Gumbel kernels require TRITON_INTERPRET=1",
)
@pytest.mark.parametrize("enabled", [False, True])
def test_v2_static_vocab_probabilistic_mapping(monkeypatch, enabled):
    spec, runtime, _ = _speculator(monkeypatch, enabled)
    # The interpreter does not implement the FP32 route's libdevice.log1p.
    spec.use_fp64_gumbel = True
    mapping = torch.tensor([1, 0, -1], dtype=torch.int32)
    positions = torch.tensor([9, 17, 0])
    previous = None
    for step, hidden in enumerate((torch.eye(3, 2), torch.ones(3, 2))):
        logits = hidden @ spec.model.lm_head.weight.T
        if enabled:
            logits = logits[:, runtime.token_id_map]
        reference = torch.zeros(3, 2, logits.shape[-1])
        sampled = gumbel_sample(
            logits,
            mapping,
            spec.temperature,
            spec.seeds,
            positions + 1,
            apply_temperature=True,
            is_drafting=True,
            output_processed_logits=reference,
            output_processed_logits_col=torch.tensor(step),
            use_fp64=True,
        )
        actual = spec._sample_draft(
            hidden, mapping, positions, torch.tensor(step), spec.draft_logits
        )
        # The third row is graph padding, ignored by the sampler's output.
        expected_ids = runtime.token_id_map[sampled] if enabled else sampled
        torch.testing.assert_close(actual[:2], expected_ids[:2], rtol=0, atol=0)
        expected_logits = reference[:2, step]
        if enabled:
            full = torch.full((2, 8), -torch.inf)
            full[:, runtime.token_id_map] = expected_logits
            expected_logits = full
        torch.testing.assert_close(
            spec.draft_logits[:2, step], expected_logits, rtol=0, atol=0
        )
        # Verify the distribution handed to rejection sampling, independently
        # of the Gumbel reference, including request remapping and temperature.
        direct = logits[:2] / spec.temperature[mapping[:2], None]
        actual_probs = spec.draft_logits[:2, step].softmax(-1)
        if enabled:
            direct_probs = torch.zeros(2, 8)
            direct_probs[:, runtime.token_id_map] = direct.softmax(-1)
        else:
            direct_probs = direct.softmax(-1)
        torch.testing.assert_close(actual_probs[mapping[:2]], direct_probs)
        if previous is not None:
            torch.testing.assert_close(spec.draft_logits[:2, 0], previous)
        previous = spec.draft_logits[:2, 0].clone()
    if enabled:
        spec.model.compute_logits.assert_not_called()
