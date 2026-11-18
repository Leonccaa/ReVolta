# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hy4-flash parsers against the checkpoint's own chat template.

Model outputs are produced by rendering an assistant turn with the real
``chat_template.jinja`` (what the model is trained to emit after the
generation prompt) and then parsed with the reasoning and tool parsers, both
non-streaming and streamed token by token with the real tokenizer.
"""

import json
from typing import Any

import pytest

from tests.tool_parsers.hy_v4_utils import (
    THINK_BEGIN,
    THINK_END,
    assistant_turn,
    encode,
    load_tokenizer,
    make_request,
    model_output_via_template,
    new_parser,
    parse_nonstreaming,
    parse_streaming,
    render,
    tool_calls_json,
)
from vllm.renderers.hf import resolve_chat_template_kwargs, safe_apply_chat_template

pytestmark = pytest.mark.skip_global_cleanup

USER = {"role": "user", "content": "What is the weather in Paris, and what time is it?"}

# Placeholder tool arguments, as the OpenAI API gives them to the template.
WEATHER = ("get_weather", {"city": "Paris", "days": 3})
SEARCH = (
    "search",
    {
        "query": '细粒度 <b>stuff</b>, "quoted"\nsecond line',
        "filters": {"year": 2024, "tags": ["a", "b"]},
        "strict": True,
        "limit": 2.5,
    },
)
TIME: tuple[str, dict[str, Any]] = ("get_time", {})


@pytest.fixture(scope="module")
def env():
    return load_tokenizer()


# effort passed to the template; None means the request does not mention it
EFFORTS = [None, "high", "no_think"]


def _kwargs(effort):
    return {} if effort is None else {"reasoning_effort": effort}


def _thinking(effort) -> bool:
    return effort != "no_think"


def _expected_reasoning(effort, text):
    return text if _thinking(effort) else None


def _parse_all(tokenizer, request, prompt, output, effort):
    """Yield (label, Parsed) for non-streaming and several streaming splits."""
    kw = _kwargs(effort)
    yield "nonstream", parse_nonstreaming(tokenizer, request, output, kw)
    for chunk in (1, 3):
        yield (
            f"stream{chunk}",
            parse_streaming(tokenizer, request, prompt, output, kw, chunk),
        )


@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize("with_tools", [False, True])
def test_plain_answer(env, effort, with_tools):
    tokenizer, template = env
    reasoning = "The user greets me." if _thinking(effort) else None
    reply = "Hello! 你好，how can I help?\n\nSecond paragraph with a < b."
    prompt, output = model_output_via_template(
        tokenizer,
        template,
        [{"role": "user", "content": "hi"}],
        assistant_turn(reasoning=reasoning, content=reply),
        tools=with_tools,
        **_kwargs(effort),
    )
    # Only a thinking turn carries the closing token in its output.
    assert (THINK_END in output) == _thinking(effort)
    request = make_request([USER], tools=with_tools)
    for label, parsed in _parse_all(tokenizer, request, prompt, output, effort):
        assert parsed.reasoning == _expected_reasoning(effort, reasoning), label
        assert parsed.content == reply, label
        assert parsed.tool_calls == {}, label


@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize(
    "calls",
    [[WEATHER], [WEATHER, TIME], [SEARCH, WEATHER, TIME]],
    ids=["one", "two", "three"],
)
@pytest.mark.parametrize("lead_in", ["", "Let me check both.\n"])
def test_tool_calls(env, effort, calls, lead_in):
    tokenizer, template = env
    reasoning = "I should call the tools." if _thinking(effort) else None
    prompt, output = model_output_via_template(
        tokenizer,
        template,
        [USER],
        assistant_turn(reasoning=reasoning, content=lead_in, tool_calls=calls),
        **_kwargs(effort),
    )
    request = make_request([USER])
    for label, parsed in _parse_all(tokenizer, request, prompt, output, effort):
        assert parsed.reasoning == _expected_reasoning(effort, reasoning), label
        assert (parsed.content or "") == lead_in, label
        assert tool_calls_json(parsed) == [(n, a) for n, a in calls], label


def test_streaming_ids_are_unique_and_arguments_arrive_in_pieces(env):
    tokenizer, template = env
    prompt, output = model_output_via_template(
        tokenizer,
        template,
        [USER],
        assistant_turn(reasoning="r", tool_calls=[SEARCH, WEATHER]),
    )
    request = make_request([USER])
    parsed = parse_streaming(tokenizer, request, prompt, output, {}, chunk=1)
    assert len(parsed.ids_minted) == 2 == len(set(parsed.ids_minted))
    assert [c["name"] for _, c in sorted(parsed.tool_calls.items())] == [
        "search",
        "get_weather",
    ]


def test_template_parse_template_roundtrip(env):
    """Parsed calls fed back into the history reproduce the model output."""
    tokenizer, template = env
    calls = [SEARCH, WEATHER, TIME]
    history = [USER]
    prompt, output = model_output_via_template(
        tokenizer, template, history, assistant_turn(content="ok ", tool_calls=calls)
    )
    parsed = parse_nonstreaming(tokenizer, make_request([USER]), output, {})
    _, again = model_output_via_template(
        tokenizer,
        template,
        history,
        assistant_turn(
            reasoning=parsed.reasoning,
            content=parsed.content or "",
            tool_calls=[(n, a) for n, a in tool_calls_json(parsed)],
        ),
    )
    # The reasoning is empty here, so only the rendering of the call block
    # and content is compared.
    assert again.split(THINK_END)[-1] == output.split(THINK_END)[-1]


def _tool_turn_history(effort):
    calls = [WEATHER, TIME]
    return [
        USER,
        assistant_turn(
            reasoning="Need the weather and the time.",
            content="Checking.",
            tool_calls=calls,
            ids=["call_w", "call_t"],
        ),
        # Results arrive out of order; the template re-orders them by id.
        {"role": "tool", "tool_call_id": "call_t", "content": "10:42"},
        {"role": "tool", "tool_call_id": "call_w", "content": "sunny, 21C"},
    ]


@pytest.mark.parametrize("effort", EFFORTS)
def test_tool_result_continuation_prompt(env, effort):
    tokenizer, template = env
    history = _tool_turn_history(effort)
    prompt = render(tokenizer, template, history, **_kwargs(effort))
    ids = encode(tokenizer, prompt)

    # Responses follow the call order of the assistant turn, not arrival order.
    assert prompt.index("sunny, 21C") < prompt.index("10:42")
    # The generation prompt is the thinking switch the parsers rely on.
    if _thinking(effort):
        assert prompt.endswith(THINK_BEGIN)
        assert "Need the weather and the time." in prompt  # preserved thinking
    else:
        assert prompt.endswith(THINK_BEGIN + THINK_END)
        assert "Need the weather and the time." not in prompt
    parser = new_parser(tokenizer, make_request([USER]), _kwargs(effort))
    assert parser.is_reasoning_end(ids) == (not _thinking(effort))


@pytest.mark.parametrize("effort", EFFORTS)
@pytest.mark.parametrize("next_turn", ["answer", "more_tools"])
def test_tool_result_continuation_output(env, effort, next_turn):
    tokenizer, template = env
    history = _tool_turn_history(effort)
    reasoning = "Now I can answer." if _thinking(effort) else None
    if next_turn == "answer":
        assistant = assistant_turn(
            reasoning=reasoning, content="It is sunny and 10:42 in Paris."
        )
        calls: list[tuple[str, dict[str, Any]]] = []
    else:
        calls = [("search", {"query": "restaurants in Paris", "limit": 3})]
        assistant = assistant_turn(reasoning=reasoning, tool_calls=calls)
    prompt, output = model_output_via_template(
        tokenizer, template, history, assistant, **_kwargs(effort)
    )
    request = make_request([USER])
    for label, parsed in _parse_all(tokenizer, request, prompt, output, effort):
        assert parsed.reasoning == _expected_reasoning(effort, reasoning), label
        assert tool_calls_json(parsed) == calls, label
        if next_turn == "answer":
            assert parsed.content == "It is sunny and 10:42 in Paris.", label
        else:
            assert not parsed.content, label


def test_truncated_thinking_has_no_content(env):
    tokenizer, template = env
    prompt = render(tokenizer, template, [USER])
    output = "still thinking about the weather"
    request = make_request([USER])
    for chunk in (1, 4):
        parsed = parse_streaming(tokenizer, request, prompt, output, {}, chunk)
        assert parsed.reasoning == output
        assert parsed.content is None
        assert parsed.tool_calls == {}


# --- reasoning_effort handling on the request path ---------------------------


def _template_kwargs(request, tokenizer, template, **defaults):
    params = request.build_chat_params(template, "string").with_defaults(defaults)
    return resolve_chat_template_kwargs(
        tokenizer, template, params.chat_template_kwargs
    )


def _apply(request, tokenizer, template):
    kwargs = _template_kwargs(request, tokenizer, template)
    return safe_apply_chat_template(
        None,  # type: ignore[arg-type]  # unused with an explicit template
        tokenizer,
        [{"role": "user", "content": "hi"}],
        chat_template=template,
        tokenize=False,
        **kwargs,
    )


def test_default_request_does_not_pass_reasoning_effort(env):
    tokenizer, template = env
    request = make_request([USER], tools=False)
    assert request.reasoning_effort is None
    # No key at all (in particular no None), so the template default "high"
    # applies.
    assert "reasoning_effort" not in _template_kwargs(request, tokenizer, template)
    prompt = _apply(request, tokenizer, template)
    assert "reasoning_effort:high" in prompt
    assert prompt.endswith(THINK_BEGIN)


@pytest.mark.parametrize("effort", ["high", "max", "no_think"])
def test_reasoning_effort_in_chat_template_kwargs(env, effort):
    tokenizer, template = env
    request = make_request(
        [USER], tools=False, chat_template_kwargs={"reasoning_effort": effort}
    )
    kwargs = _template_kwargs(request, tokenizer, template)
    assert kwargs["reasoning_effort"] == effort
    prompt = _apply(request, tokenizer, template)
    assert f"reasoning_effort:{effort}" in prompt
    assert prompt.endswith(
        THINK_BEGIN + THINK_END if effort == "no_think" else THINK_BEGIN
    )


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "xhigh"])
def test_top_level_reasoning_effort_outside_template_set_is_rejected(env, effort):
    """The generic API values the template does not know give a 400 (ValueError)."""
    tokenizer, template = env
    request = make_request([USER], tools=False, reasoning_effort=effort)
    with pytest.raises(ValueError, match=f"reasoning_effort error : {effort}"):
        _apply(request, tokenizer, template)


def test_top_level_max_is_accepted(env):
    tokenizer, template = env
    request = make_request([USER], tools=False, reasoning_effort="max")
    assert "reasoning_effort:max" in _apply(request, tokenizer, template)


@pytest.mark.parametrize("bad", ["off", "No_Think", "", "disabled"])
def test_invalid_chat_template_kwargs_value_is_rejected(env, bad):
    tokenizer, template = env
    request = make_request(
        [USER], tools=False, chat_template_kwargs={"reasoning_effort": bad}
    )
    with pytest.raises(ValueError, match="reasoning_effort error"):
        _apply(request, tokenizer, template)


def test_explicit_null_in_chat_template_kwargs_is_rejected(env):
    tokenizer, template = env
    request = make_request(
        [USER], tools=False, chat_template_kwargs={"reasoning_effort": None}
    )
    with pytest.raises(ValueError, match="reasoning_effort error : None"):
        _apply(request, tokenizer, template)


def test_enable_thinking_alone_does_not_switch_thinking_off(env):
    """Only reasoning_effort is the switch; enable_thinking is ignored."""
    tokenizer, template = env
    request = make_request(
        [USER], tools=False, chat_template_kwargs={"enable_thinking": False}
    )
    assert _apply(request, tokenizer, template).endswith(THINK_BEGIN)


def test_top_level_effort_overrides_chat_template_kwargs_consistently(env):
    """The renderer and the parser see the same merged value."""
    tokenizer, template = env
    request = make_request(
        [USER],
        tools=False,
        reasoning_effort="high",
        chat_template_kwargs={"reasoning_effort": "no_think"},
    )
    merged = request.build_chat_params(template, "string").chat_template_kwargs
    assert merged["reasoning_effort"] == "high"
    prompt = _apply(request, tokenizer, template)
    assert prompt.endswith(THINK_BEGIN)
    parser = new_parser(tokenizer, request, merged)
    assert not parser.is_reasoning_end(encode(tokenizer, prompt))


def test_json_arguments_survive_the_openai_message_format(env):
    """Assistant history with JSON-string arguments (as clients send them)."""
    tokenizer, template = env
    request = make_request(
        [
            USER,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "c1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps(WEATHER[1]),
                        },
                    }
                ],
            },
        ]
    )
    # The messages validate, which is what the renderer starts from.
    assert request.messages[1]["tool_calls"][0]["function"]["name"] == "get_weather"  # type: ignore[index]
