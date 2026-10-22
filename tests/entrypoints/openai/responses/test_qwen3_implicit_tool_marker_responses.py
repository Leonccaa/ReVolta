# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Responses API: an implicit <tool_call> reasoning end under a JSON grammar.

The Responses API keeps the tool_choice="required"/named grammar on
``request.text`` (json_schema) instead of ``request.structured_outputs``.
"""

import pytest
from openai.types.responses import FunctionTool

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
)
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.parser.abstract_parser import _WrappedParser
from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

TOOL = FunctionTool(
    type="function",
    name="f",
    parameters={"type": "object", "properties": {"x": {"type": "string"}}},
)
REASONING = "Call f.\n"
XML_CALL = (
    "<tool_call>\n<function=f>\n<parameter=x>\nv\n</parameter>\n</function>\n"
    "</tool_call>"
)


@pytest.fixture
def tok():
    return _byte_tokenizer()


@pytest.fixture(autouse=True)
def wrapped_parser(monkeypatch):
    monkeypatch.setattr(_WrappedParser, "tool_parser_cls", Qwen3CoderToolParser)
    monkeypatch.setattr(_WrappedParser, "reasoning_parser_cls", Qwen3ReasoningParser)


def _request(tok, tool_choice):
    request = ResponsesRequest(
        model="test", input="call f", tools=[TOOL], tool_choice=tool_choice
    )
    return Qwen3CoderToolParser(tok, request.tools).adjust_request(request)


def _outputs(tok, request, text):
    parser = _WrappedParser(
        tok, request.tools, chat_template_kwargs={"enable_thinking": True}
    )
    ids = tok.encode(text, add_special_tokens=False)
    return parser.extract_response_outputs(
        model_output=text,
        model_output_token_ids=ids,
        request=request,
        enable_auto_tools=True,
    )


@pytest.mark.parametrize(
    "tool_choice, grammar_output",
    [
        ("required", '[{"name": "f", "parameters": {"x": "v"}}]'),
        ({"type": "function", "name": "f"}, '{"x": "v"}'),
    ],
)
def test_responses_json_grammar_after_implicit_marker(tok, tool_choice, grammar_output):
    request = _request(tok, tool_choice)
    items = _outputs(tok, request, REASONING + "<tool_call>" + grammar_output)
    calls = [item for item in items if item.type == "function_call"]
    assert [(c.name, c.arguments) for c in calls] == [("f", '{"x": "v"}')]
    (reasoning,) = [item for item in items if item.type == "reasoning"]
    assert reasoning.content[0].text == REASONING


def test_responses_auto_keeps_the_marker_for_the_xml_parser(tok):
    request = _request(tok, "auto")
    items = _outputs(tok, request, REASONING + XML_CALL)
    calls = [item for item in items if item.type == "function_call"]
    assert [(c.name, c.arguments) for c in calls] == [("f", '{"x": "v"}')]
