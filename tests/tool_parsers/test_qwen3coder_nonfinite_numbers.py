# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""qwen3_coder: a non-finite number argument must not drop the tool call.

``int(float("inf"))`` raised OverflowError while converting a "number"
parameter. Non-streaming parsing then returned every call of the response as
raw content; streaming aborted. Non-finite values are kept as strings
(vllm-project/vllm#43984).
"""

import json

import pytest

from tests.entrypoints.openai.chat_completion.test_output_correctness import (
    _byte_tokenizer,
)
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "scale",
            "parameters": {
                "type": "object",
                "properties": {
                    "factor": {"type": "number"},
                    "name": {"type": "string"},
                },
            },
        },
    }
]


def _call(factor):
    return (
        "<tool_call>\n<function=scale>\n"
        f"<parameter=factor>\n{factor}\n</parameter>\n"
        "<parameter=name>\nx\n</parameter>\n"
        "</function>\n</tool_call>"
    )


@pytest.mark.parametrize(
    "factor, expected",
    [("inf", "inf"), ("Infinity", "Infinity"), ("1e999", "1e999"), ("2.5", 2.5)],
)
def test_non_streaming_keeps_the_call(factor, expected):
    tok = _byte_tokenizer()
    request = ChatCompletionRequest(model="test", messages=[], tools=TOOLS)
    info = Qwen3CoderToolParser(tok, request.tools).extract_tool_calls(
        _call(factor), request
    )
    assert info.tools_called
    (call,) = info.tool_calls
    arguments = json.loads(call.function.arguments)
    assert arguments == {"factor": expected, "name": "x"}
