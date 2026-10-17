# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU regressions for MRv2 per-request buffer limits."""

from types import SimpleNamespace as NS

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import SamplingParams


class Tokenizer:
    max_token_id = 248319

    def __len__(self):
        return 248320

    def encode(self, text, add_special_tokens=False):
        # Both prefix variants have the same length and different first IDs.
        return [2 if text.startswith(" ") else 1] + [3] * (len(text.lstrip()) - 1)


def verify(p):
    p.verify(
        NS(max_logprobs=20, get_vocab_size=lambda: 248320, logits_processors=None),
        NS(),
        None,
        Tokenizer(),
    )


@pytest.mark.parametrize("count", [1024, 1025])
def test_allowed_limit(count):
    p = SamplingParams(allowed_token_ids=list(range(count)))
    if count > 1024:
        with pytest.raises(VLLMValidationError):
            verify(p)
    else:
        verify(p)


@pytest.mark.parametrize("count", [64, 65])
def test_bad_words_count(count):
    p = SamplingParams(bad_words=["abc"] * count)
    if count > 64:
        with pytest.raises(VLLMValidationError):
            p.update_from_tokenizer(Tokenizer())
    else:
        p.update_from_tokenizer(Tokenizer())
        assert len(p.bad_words_token_ids) == 128


def test_bad_words_total_tokens():
    p = SamplingParams(bad_words=["a" * 600])
    with pytest.raises(VLLMValidationError):
        p.update_from_tokenizer(Tokenizer())


@pytest.mark.parametrize(
    "count,minimum,should_reject", [(126, 1, False), (127, 1, True), (500, 0, False)]
)
def test_stop_limit_after_eos(count, minimum, should_reject):
    p = SamplingParams(min_tokens=minimum, stop_token_ids=list(range(count)))
    p.update_from_generation_config({"eos_token_id": [2000, 2001]}, 2000)
    verify(p)
    # A missing method on baseline is an explicit test failure.
    if should_reject:
        with pytest.raises(VLLMValidationError):
            p.validate_stop_token_ids_limit()
    else:
        p.validate_stop_token_ids_limit()


def test_stop_strings_are_not_limited_to_four():
    params = SamplingParams(stop=[f"stop-{i}" for i in range(20)])
    verify(params)
    assert len(params.stop) == 20


@pytest.mark.parametrize("ignore_eos", [True, False])
def test_stop_limit_includes_generation_eos(ignore_eos):
    params = SamplingParams(
        min_tokens=1, stop_token_ids=list(range(127)), ignore_eos=ignore_eos
    )
    params.update_from_generation_config({"eos_token_id": [2000, 2001]}, 2000)
    with pytest.raises(VLLMValidationError):
        params.validate_stop_token_ids_limit()


@pytest.mark.parametrize(
    "parameter", ["bad_words", "allowed_token_ids", "stop_token_ids"]
)
def test_oversized_input_is_rejected_before_engine_submission(parameter):
    from unittest.mock import Mock

    from vllm.v1.engine.input_processor import InputProcessor

    processor = object.__new__(InputProcessor)
    processor.model_config = NS(
        max_logprobs=20, get_vocab_size=lambda: 248320, logits_processors=None
    )
    processor.speculative_config = None
    processor.structured_outputs_config = None
    processor.use_v2_model_runner = True
    processor.vllm_config = NS(
        parallel_config=NS(
            data_parallel_size=1, data_parallel_size_local=1, local_engines_only=False
        )
    )
    processor.renderer = NS(tokenizer=Tokenizer(), get_eos_token_id=lambda: 2000)
    processor.generation_config_fields = {"eos_token_id": [2000, 2001]}
    processor._validate_lora = Mock()
    processor._validate_model_inputs = Mock()
    arguments = {
        "bad_words": {"bad_words": ["abc"] * 65},
        "allowed_token_ids": {"allowed_token_ids": list(range(1025))},
        # Only exceeds the limit AFTER the generation-config EOS merge.
        "stop_token_ids": {"min_tokens": 1, "stop_token_ids": list(range(127))},
    }
    with pytest.raises(VLLMValidationError) as error:
        processor.process_inputs(
            "oversized",
            {"type": "token", "prompt_token_ids": [1, 2]},
            SamplingParams(**arguments[parameter]),
            ("generate",),
        )
    assert error.value.parameter == parameter
