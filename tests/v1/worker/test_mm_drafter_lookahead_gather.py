# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Drafter multimodal gather when the lookahead reaches an unencoded item.

The scheduler may compute the last token before an item that it could not
encode yet (see tests/v1/core/test_scheduler_mm_spec_lookahead.py). The
EAGLE/MTP drafter reads one position ahead, so its gather reaches that item.
It must keep the placeholder embedding there instead of failing; target
gathers and drafter gathers that need the item for scheduled positions keep
the encoder cache miss assertion.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm.multimodal.inputs import PlaceholderRange
from vllm.v1.worker.gpu.mm.encoder_runner import EncoderRunner
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from ..core.utils import create_requests

pytestmark = pytest.mark.cpu_test

HIDDEN = 4


def mm_features():
    (request,) = create_requests(
        num_requests=1,
        num_tokens=32,
        mm_hashes_list=[["imgX", "imgY"]],
        mm_positions=[[PlaceholderRange(8, 8), PlaceholderRange(16, 8)]],
        req_ids=["r"],
    )
    return request.mm_features


def encoder_outputs(names):
    return {name: torch.full((8, HIDDEN), float(i + 1)) for i, name in enumerate(names)}


def gather_v1(num_computed, num_scheduled, cached, shift):
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(req_ids=["r"]),
        requests={
            "r": SimpleNamespace(
                num_computed_tokens=num_computed, mm_features=mm_features()
            )
        },
        encoder_cache=encoder_outputs(cached),
        is_multimodal_pruning_enabled=False,
        uses_mrope=False,
    )
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=num_scheduled,
        num_scheduled_tokens={"r": num_scheduled},
    )
    return GPUModelRunner._gather_mm_embeddings(
        runner, scheduler_output, shift_computed_tokens=shift
    )


def gather_v2(num_computed, num_scheduled, cached, lookahead):
    runner = SimpleNamespace(
        encoder_cache=SimpleNamespace(
            mm_features={"r": mm_features()},
            encoder_outputs=encoder_outputs(cached),
        ),
        device="cpu",
    )
    return EncoderRunner.gather_mm_embeddings(
        runner,
        ["r"],
        num_scheduled,
        np.array([num_scheduled]),
        np.array([0, num_scheduled]),
        np.array([32]),
        np.array([num_computed + lookahead]),
        num_lookahead_tokens=lookahead,
    )


@pytest.mark.parametrize("gather", [gather_v1, gather_v2])
def test_lookahead_only_item_missing_keeps_placeholder(gather):
    # Last token of X, X still cached, Y not encoded yet.
    mm_embeds, is_mm_embed = gather(15, 1, ["imgX"], 1)
    assert mm_embeds == []
    assert not is_mm_embed.any()


@pytest.mark.parametrize("gather", [gather_v1, gather_v2])
def test_lookahead_item_present_is_still_gathered(gather):
    mm_embeds, is_mm_embed = gather(15, 1, ["imgX", "imgY"], 1)
    assert len(mm_embeds) == 1
    assert torch.equal(mm_embeds[0], torch.full((1, HIDDEN), 2.0))
    assert is_mm_embed.tolist() == [True]


@pytest.mark.parametrize("gather", [gather_v1, gather_v2])
def test_missing_item_for_scheduled_positions_still_asserts(gather):
    # Target gather (no shift) over Y.
    with pytest.raises(AssertionError, match="Encoder cache miss"):
        gather(16, 4, ["imgX"], 0)
    # Drafter gather where scheduled positions (not only the lookahead) need Y.
    with pytest.raises(AssertionError, match="Encoder cache miss"):
        gather(15, 2, ["imgX"], 1)
