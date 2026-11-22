# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from math import erfc, sqrt

import numpy as np
import pytest
from transformers import OPTConfig
from xgrammar import (
    GrammarCompiler,
    GrammarMatcher,
    TokenizerInfo,
    allocate_token_bitmask,
)

from tests.v1.core.utils import create_scheduler
from vllm.config import SpeculativeConfig
from vllm.sampling_params import SamplingParams, StructuredOutputsParams
from vllm.v1.outputs import DraftTokenIds, ModelRunnerOutput
from vllm.v1.request import Request
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar

VOCAB = ["[", "0,", '"A"', '"B"', '"C"', "]", "[eos]"]
OPEN, PREFIX, A, B, INVALID, CLOSE, EOS = range(len(VOCAB))
SCHEMA = {
    "type": "array",
    "prefixItems": [{"const": 0}, {"enum": ["A", "B"]}],
    "minItems": 2,
    "maxItems": 2,
}
SAMPLE_COUNT = 12_000
ALPHA = 0.001


def _verification_rows(tmp_path, width, async_scheduling, greedy, draft):
    OPTConfig(architectures=["OPTForCausalLM"]).save_pretrained(tmp_path)
    scheduler = create_scheduler(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        async_scheduling=async_scheduling,
        num_speculative_tokens=None if async_scheduling else width,
    )
    scheduler.use_v2_model_runner = True
    # Supply the real scheduler and mask manager with a local stand-in for the
    # drafter config. The draft ids and distributions stand in for inference.
    scheduler.vllm_config.speculative_config = SpeculativeConfig(
        model="ngram",
        num_speculative_tokens=width,
        draft_sample_method="probabilistic" if width == 7 else "greedy",
    )
    if async_scheduling:
        scheduler.num_spec_tokens = width
        scheduler._spec_token_placeholders = (-1,) * width

    info = TokenizerInfo(VOCAB, stop_token_ids=[EOS])
    compiled = GrammarCompiler(info).compile_json_schema(json.dumps(SCHEMA))
    grammar = XgrammarGrammar(len(VOCAB), GrammarMatcher(compiled), compiled)
    params = SamplingParams(
        temperature=0 if greedy else 1,
        structured_outputs=StructuredOutputsParams(json=SCHEMA),
    )
    params.update_from_generation_config({}, EOS)
    request = Request("enum", [PREFIX], params, None)
    structured = request.structured_output_request
    assert structured is not None
    structured.grammar = grammar
    scheduler.structured_output_manager._grammar_bitmask = allocate_token_bitmask(
        width + 1, len(VOCAB)
    )
    scheduler.add_request(request)
    scheduler.update_from_output(
        scheduler.schedule(),
        ModelRunnerOutput(
            req_ids=["enum"], req_id_to_index={"enum": 0}, sampled_token_ids=[[OPEN]]
        ),
    )

    proposal = [PREFIX, draft, *([INVALID] * (width - 2))]
    drafts = DraftTokenIds(["enum"], [proposal])
    if async_scheduling:
        output = scheduler.schedule()
        scheduler.update_draft_token_ids_in_output(drafts, output)
    else:
        scheduler.update_draft_token_ids(drafts)
        output = scheduler.schedule()
    mask = scheduler.get_grammar_bitmask(output)
    assert mask is not None
    assert grammar.num_processed_tokens == 1
    assert not isinstance(structured.grammar, Exception)

    # V2 takes the width from the scheduler but uses the original device IDs.
    scheduled = output.scheduled_spec_decode_tokens.get("enum", [])
    ids = proposal[: len(scheduled)]
    allowed = (mask.grammar_bitmask[:, :1] & (1 << np.arange(len(VOCAB)))) != 0
    assert len(allowed) == len(ids) + 1
    return ids, allowed


@pytest.mark.parametrize(
    ("width", "async_scheduling", "mode"),
    [
        pytest.param(7, False, "probabilistic", id="dflash2-sync"),
        pytest.param(7, True, "probabilistic", id="dflash2-async"),
        pytest.param(4, False, "one-hot-invalid", id="mtp4-sync"),
        pytest.param(4, True, "one-hot-invalid", id="mtp4-async"),
        pytest.param(4, False, "one-hot-valid", id="mtp4-sync-valid"),
        pytest.param(4, True, "one-hot-valid", id="mtp4-async-valid"),
        pytest.param(7, False, "greedy", id="dflash2-target-greedy"),
        pytest.param(4, True, "greedy", id="mtp4-target-greedy"),
    ],
)
def test_structured_speculation_preserves_distribution(
    tmp_path, width, async_scheduling, mode
):
    rng = np.random.default_rng(61557)
    target = np.zeros((width + 1, len(VOCAB)))
    target[0, PREFIX] = 1
    target[1, [A, B, INVALID]] = [0.3, 0.2, 0.5]
    target[2:, CLOSE] = 1

    p = np.array([0.6, 0.4])
    probabilistic = mode == "probabilistic"
    greedy = mode == "greedy"
    candidates = [A, B, INVALID]
    q = (
        [0.55, 0.05, 0.40]
        if probabilistic
        else [1, 0, 0]
        if mode == "one-hot-valid"
        else [0, 0, 1]
    )
    choices = rng.choice(candidates, size=SAMPLE_COUNT, p=q)
    expected = (
        np.full(SAMPLE_COUNT, A) if greedy else rng.choice([A, B], SAMPLE_COUNT, p=p)
    )
    observed = np.empty(SAMPLE_COUNT, dtype=int)
    accepted = np.zeros(SAMPLE_COUNT, dtype=int)

    for choice in np.unique(choices):
        ids, allowed = _verification_rows(
            tmp_path, width, async_scheduling, greedy, int(choice)
        )
        probs = target[:2] * allowed[:2]
        probs /= probs.sum(axis=1, keepdims=True)
        assert probs[0, PREFIX] == 1
        np.testing.assert_allclose(probs[1, [A, B]], p)
        assert probs[1, INVALID] == 0
        if choice != INVALID:
            # A valid sampled value may be accepted. The following draft never is.
            assert allowed[2, CLOSE] and not allowed[2, INVALID]
        rows = np.flatnonzero(choices == choice)
        accepted[rows] = 1  # The fixed prefix is always accepted.

        if greedy:
            if not async_scheduling:
                assert ids == [PREFIX]
            observed[rows] = probs[1].argmax()
        elif len(ids) == 1:
            observed[rows] = rng.choice(len(VOCAB), len(rows), p=probs[1])
        else:
            assert ids[1] == choice
            proposal = np.eye(len(VOCAB))[choice]
            if probabilistic:
                proposal[candidates] = q
            pass_test = rng.random(len(rows)) < probs[1, choice] / proposal[choice]
            observed[rows[pass_test]] = choice
            accepted[rows[pass_test]] += 1
            residual = np.maximum(probs[1] - proposal, 0)
            rejected = rows[~pass_test]
            if len(rejected):
                observed[rejected] = rng.choice(
                    len(VOCAB), len(rejected), p=residual / residual.sum()
                )

    assert np.isin(observed, [A, B]).all()
    observed_a = int(np.count_nonzero(observed == A))
    expected_a = int(np.count_nonzero(expected == A))
    if not greedy:
        pooled = (observed_a + expected_a) / (2 * SAMPLE_COUNT)
        z = (observed_a - expected_a) / sqrt(2 * SAMPLE_COUNT * pooled * (1 - pooled))
        p_value = erfc(abs(z) / sqrt(2))
    else:
        p_value = float(np.array_equal(observed, expected))
    print(
        f"A={observed_a}/{SAMPLE_COUNT} baseline={expected_a}/{SAMPLE_COUNT} "
        f"p={p_value:.6g} schema=100% mean_accepted={accepted.mean():.5f}"
    )
    assert p_value >= ALPHA
