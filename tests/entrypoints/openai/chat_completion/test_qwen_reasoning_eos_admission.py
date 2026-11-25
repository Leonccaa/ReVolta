# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from types import SimpleNamespace

import msgspec
import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
    _encode,
    _generate,
    _request,
    _serving,
)
from vllm import envs
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.utils import check_stop
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import Request, RequestStatus

EOS_TAGS = ("<|im_end|>", "<|endoftext|>")


@pytest.mark.parametrize(
    "switch, hidden, disabled, phase, ignore, stream, mm, mrv2",
    [
        (True, False, False, "open", False, True, False, True),
        (True, False, False, "open", False, False, True, True),
        (False, False, False, "open", False, True, False, True),
        (True, True, False, "open", False, False, False, True),
        (True, False, True, "open", False, False, False, True),
        (True, False, False, "closed", False, False, False, True),
        (True, False, False, "absent", False, False, False, True),
        (True, False, False, "history_only", False, False, False, True),
        (True, False, False, "open", True, False, False, True),
        (True, False, False, "open", False, False, False, False),
    ],
)
def test_chat_admission(
    monkeypatch, switch, hidden, disabled, phase, ignore, stream, mm, mrv2
):
    monkeypatch.setattr(envs, "VLLM_QWEN_REASONING_EOS_MASK", switch)
    tokenizer = _byte_tokenizer()
    tokenizer.add_special_tokens({"additional_special_tokens": list(EOS_TAGS)})
    primary, extra = (tokenizer.get_vocab()[tag] for tag in EOS_TAGS)
    suffix = {
        "open": "<think>",
        "closed": "<think></think>",
        "absent": "",
        "history_only": "<think>previous" + EOS_TAGS[0] + "question",
    }
    prompt = _encode(tokenizer, "question" + suffix[phase])
    engine_input = {"type": "multimodal" if mm else "token", "prompt_token_ids": prompt}
    serving = _serving()
    serving.renderer = SimpleNamespace(tokenizer=tokenizer)
    serving.reasoning_parser_cls = Qwen3ReasoningParser
    serving.chat_template = None
    serving.chat_template_content_format = "auto"
    serving.default_chat_template_kwargs = {}
    serving.models = SimpleNamespace(model_name=lambda *args: "test")
    serving.model_config = SimpleNamespace(max_model_len=128, is_encoder_decoder=False)
    serving.default_sampling_params = {}
    serving.override_max_tokens = None
    serving.request_logger = None
    captured = []

    async def render(request):
        return [], [engine_input]

    def generate(inputs, params, *args, **kwargs):
        assert inputs is engine_input
        # Same merge order as the engine input processor.
        params = params.clone()
        params.update_from_generation_config({"eos_token_id": [extra]}, primary)
        captured.append(
            msgspec.msgpack.decode(msgspec.msgpack.encode(params), type=SamplingParams)
        )
        return _generate([])

    async def respond(*args, **kwargs):
        return "done"

    serving.engine_client = SimpleNamespace(
        generate=generate, vllm_config=SimpleNamespace(use_v2_model_runner=mrv2)
    )
    monkeypatch.setattr(serving, "render_chat_request", render)
    monkeypatch.setattr(serving, "_maybe_get_adapters", lambda *args, **kwargs: None)
    monkeypatch.setattr(serving, "chat_completion_stream_generator", respond)
    monkeypatch.setattr(serving, "chat_completion_full_generator", respond)
    request = _request(
        include_reasoning=not hidden,
        reasoning_effort="none" if disabled else "medium",
        ignore_eos=ignore,
        stream=stream,
        stop_token_ids=[17],
        max_tokens=16,
    )

    async def run():
        result = await serving._create_chat_completion(request)
        return await result if stream else result

    assert asyncio.run(run()) == "done"
    (params,) = captured
    if switch and mrv2 and phase == "open" and not (hidden or disabled or ignore):
        config = params._qwen_reasoning_eos
        assert config is not None and config[0] == (primary, extra)
    else:
        assert params._qwen_reasoning_eos is None
    assert 17 in params.stop_token_ids
    assert (extra in params.stop_token_ids) == (not ignore)
    assert params.eos_token_id == (None if ignore else primary)


def test_only_configured_model_eos_are_masked_and_length_stop_is_unchanged():
    params = SamplingParams(
        max_tokens=2,
        stop_token_ids=[6, 17],
        _qwen_reasoning_eos=((4, 6), 7, 8),
    )
    params.update_from_generation_config({}, eos_token_id=4)
    assert params._qwen_reasoning_eos == ((4,), 7, 8)
    req = Request("test", [1, 9], params, None)
    req.append_output_token_ids(10)
    assert not check_stop(req, 128)
    req.append_output_token_ids(10)
    assert check_stop(req, 128)
    assert req.status == RequestStatus.FINISHED_LENGTH_CAPPED

    req = Request("test", [1, 9], params, None)
    req.append_output_token_ids(17)
    assert check_stop(req, 128)
    assert req.status == RequestStatus.FINISHED_STOPPED and req.stop_reason == 17


def test_token_spelling_does_not_confirm_a_reasoning_boundary():
    tokenizer = _byte_tokenizer()
    parser = Qwen3ReasoningParser(tokenizer)
    params = SamplingParams(
        _qwen_reasoning_eos=((260,), parser.end_token_id, parser._tool_call_token_id)
    )
    req = Request("test", _encode(tokenizer, "question<think>"), params, None)
    literal_ids = [tokenizer.get_vocab()[char] for char in parser.end_token]
    assert tokenizer.decode(literal_ids) == parser.end_token
    req.append_output_token_ids(literal_ids)
    assert not req.qwen_reasoning_eos_ended
    req.append_output_token_ids(parser.end_token_id)
    assert req.qwen_reasoning_eos_ended


@pytest.mark.parametrize("config", [None, ((4, 6), 7, 8)])
def test_mrv2_thinking_budget_validation_is_unchanged(monkeypatch, config):
    monkeypatch.setattr(SamplingParams, "verify", lambda *args: None)
    processor = SimpleNamespace(
        model_config=None,
        speculative_config=None,
        structured_outputs_config=None,
        tokenizer=None,
        vllm_config=SimpleNamespace(reasoning_config=SimpleNamespace(enabled=True)),
        use_v2_model_runner=True,
    )
    params = SamplingParams(thinking_token_budget=10, _qwen_reasoning_eos=config)
    with pytest.raises(ValueError, match="not yet supported by the V2 model runner"):
        InputProcessor._validate_params(processor, params, ("generate",))
