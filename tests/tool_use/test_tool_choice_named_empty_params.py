# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""A forced named tool without parameters must emit exactly an empty object."""

import pytest
from openai.types.responses import FunctionTool, ToolChoiceFunction
from xgrammar import Grammar
from xgrammar.testing import _is_grammar_accept_string

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionNamedToolChoiceParam,
    ChatCompletionRequest,
    ChatCompletionToolsParam,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.tool_parsers.abstract_tool_parser import ToolParser
from vllm.tool_parsers.utils import get_json_schema_from_tools

pytestmark = pytest.mark.cpu_test

EMPTY_OBJECT = {"type": "object", "properties": {}, "additionalProperties": False}
PARAMS = {"type": "object", "properties": {"x": {"type": "string"}}}


def _chat_tool(params):
    function = {"name": "ping"}
    if params is not ...:
        function["parameters"] = params
    return ChatCompletionToolsParam.model_validate(
        {"type": "function", "function": function}
    )


@pytest.mark.parametrize(
    "params",
    [
        ...,
        None,
        {},
        {"type": "object", "properties": {}},
        {"title": "Ping", **EMPTY_OBJECT},
    ],
)
def test_chat_named_empty_params_constrain_an_object(params):
    choice = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "ping"}}
    )
    assert get_json_schema_from_tools(choice, [_chat_tool(params)]) == EMPTY_OBJECT


@pytest.mark.parametrize("params", [None, {}])
def test_responses_named_empty_params_constrain_an_object(params):
    tool = FunctionTool(type="function", name="ping", parameters=params)
    choice = ToolChoiceFunction(type="function", name="ping")
    assert get_json_schema_from_tools(choice, [tool]) == EMPTY_OBJECT


@pytest.mark.parametrize(
    "params",
    [
        PARAMS,
        {"type": "object"},
        {"type": "object", "properties": {}, "additionalProperties": True},
        {"type": "object", "properties": {}, "patternProperties": {".*": {}}},
    ],
)
def test_named_params_are_unchanged(params):
    choice = ChatCompletionNamedToolChoiceParam.model_validate(
        {"type": "function", "function": {"name": "ping"}}
    )
    assert get_json_schema_from_tools(choice, [_chat_tool(params)]) == params
    tool = FunctionTool(type="function", name="ping", parameters=params)
    choice = ToolChoiceFunction(type="function", name="ping")
    assert get_json_schema_from_tools(choice, [tool]) == params


def test_required_schema_is_unchanged():
    schema = get_json_schema_from_tools("required", [_chat_tool(None)])
    assert schema["items"]["anyOf"][0]["properties"]["parameters"] == {
        "type": "object",
        "properties": {},
    }


@pytest.mark.parametrize("api", ["chat", "responses"])
def test_named_request_without_params_gets_exact_empty_grammar(api):
    parser = ToolParser.__new__(ToolParser)
    if api == "chat":
        chat_request = ChatCompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "ping"}],
            tools=[{"type": "function", "function": {"name": "ping"}}],
            tool_choice={"type": "function", "function": {"name": "ping"}},
            response_format={"type": "json_object"},
        )
        parser.adjust_request(chat_request)
        params = chat_request.to_sampling_params(
            max_tokens=32, default_sampling_params={}
        )
    else:
        responses_request = ResponsesRequest(
            model="test",
            input="ping",
            tools=[{"type": "function", "name": "ping", "parameters": None}],
            tool_choice={"type": "function", "name": "ping"},
            text={"format": {"type": "text"}, "verbosity": "low"},
        )
        parser.adjust_request(responses_request)
        params = responses_request.to_sampling_params(default_max_tokens=32)
        assert responses_request.text is not None
        assert responses_request.text.verbosity == "low"

    assert params.structured_outputs is not None
    assert params.structured_outputs.regex is not None
    grammar = Grammar.from_regex(params.structured_outputs.regex)
    assert _is_grammar_accept_string(grammar, "{}")
    for invalid in ("{ }", ' {"x": 1}', '{"x": 1}', " {}"):
        assert not _is_grammar_accept_string(grammar, invalid)
