# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared request shaping for parsers using the Qwen XML tool format.

Extracted from Qwen3XMLToolParser (1CatAI/1Cat-vLLM#653).
"""

import json
from logging import Logger
from typing import Any

from xgrammar import Grammar
from xgrammar.structural_tag import (
    AnyTextFormat,
    ConstStringFormat,
    JSONSchemaFormat,
    OptionalFormat,
    OrFormat,
    RegexFormat,
    SequenceFormat,
    StructuralTag,
    TagFormat,
)

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
from vllm.sampling_params import (
    StructuredOutputsParams,
)
from vllm.tool_parsers.structural_tag_registry import (
    get_enable_structured_outputs_in_reasoning,
    get_model_structural_tag,
)

_JSON_OBJECT_NO_TOOL_HINT = (
    "If no tool call is required to answer the user, reply directly with a raw "
    "JSON object without markdown code fences."
)
_JSON_SCHEMA_NO_TOOL_HINT = (
    "If no tool call is required to answer the user, reply directly with a raw "
    "JSON value matching the requested schema without markdown code fences."
)


def qwen_xml_json_answer(
    request: ChatCompletionRequest,
) -> tuple[dict[str, Any], str] | None:
    # An omitted tool choice is normalized to "auto" when tools are present.
    if not request.tools or request.tool_choice != "auto":
        return None
    # Explicit structured outputs keep their existing precedence.
    if (
        request.structured_outputs is not None
        and not request.structured_outputs.all_constraints_none()
    ):
        return None
    response_format = request.response_format
    if response_format is None:
        return None
    if response_format.type == "json_object":
        return {"type": "object"}, _JSON_OBJECT_NO_TOOL_HINT
    if response_format.type == "json_schema":
        json_schema = response_format.json_schema
        if json_schema is not None and json_schema.json_schema is not None:
            return json_schema.json_schema, _JSON_SCHEMA_NO_TOOL_HINT
    return None


def inject_qwen_xml_response_format_hint(messages: list[Any], hint: str) -> list[Any]:
    """Tell the model how to answer directly in the Qwen XML tool/JSON union."""
    if not messages:
        return messages

    first = messages[0]
    if isinstance(first, dict) and first.get("role") in ("system", "developer"):
        content = first.get("content")
        if isinstance(content, str):
            if hint in content:
                return messages
            updated_first = {
                **first,
                "content": f"{content}\n\n{hint}" if content else hint,
            }
            return [updated_first, *messages[1:]]
        if isinstance(content, list):
            updated_first = {
                **first,
                "content": [*content, {"type": "text", "text": f"\n\n{hint}"}],
            }
            return [updated_first, *messages[1:]]
    return [{"role": "system", "content": hint}, *messages]


def adjust_qwen_xml_tool_request(
    request: ChatCompletionRequest | ResponsesRequest,
    logger: Logger,
) -> str | None:
    """Install a tool/answer union and return its raw JSON rendering hint.

    A None result leaves the request untouched for the caller's base path.
    """
    if not isinstance(request, ChatCompletionRequest):
        return None
    answer = qwen_xml_json_answer(request)
    if answer is None:
        return None
    assert request.tools is not None
    schema, hint = answer
    answer_format = JSONSchemaFormat(json_schema=schema)
    reasoning = get_enable_structured_outputs_in_reasoning()

    def build_tag_json(tools):
        # Reuse the registry for the tool-call branch so the wire format
        # stays in sync with vLLM. tool_choice="required"
        # (TagsWithSeparatorFormat) not "auto" (TriggeredTagsFormat): on the
        # production tokenizer (vocab 248077) the "auto" shape leaves ~248075
        # tokens allowed at position 0, making the union degenerate and the
        # answer branch unreachable; "required" keeps the mask tight.
        tools_tag = get_model_structural_tag(
            model="qwen_3_5",
            tools=tools,
            tool_choice="required",
            reasoning=False,
        )
        assert tools_tag is not None
        # qwen_3_5 hardcodes separator="" in the registry, but the chat
        # template emits "\n<tool_call>" between parallel calls, so a real
        # </tool_call>\n<tool_call> sequence would otherwise be rejected. Fix
        # the separator at this use site (do NOT edit the shared registry).
        tools_format = tools_tag.format.model_copy(update={"separator": "\n"})
        # Allow the model's natural leading whitespace: with
        # enable_in_reasoning=False the grammar binds right after "</think>"
        # while the template emits "\n</think>\n\n", so the first real token
        # is "\n\n". Without this the tool-vs-answer decision would be made on
        # a masked token. Whitespace alone cannot satisfy the OrFormat, so it
        # cannot terminate the match.
        union = SequenceFormat(
            elements=[
                OptionalFormat(content=RegexFormat(pattern="[ \\n\\t]{1,8}")),
                OrFormat(elements=[tools_format, answer_format]),
            ]
        )
        if reasoning:
            prefix = SequenceFormat(
                elements=[
                    TagFormat(begin="", content=AnyTextFormat(), end="</think>"),
                    ConstStringFormat(value="\n\n"),
                ]
            )
            union = SequenceFormat(elements=[prefix, union])
        return json.dumps(StructuralTag(format=union).model_dump())

    def compiles(tag_json):
        try:
            Grammar.from_structural_tag(tag_json)
        except Exception as exc:
            return exc
        return True

    # Self-validate: the union compiles every tool's parameters through the
    # qwen_xml converter, which today's "auto" path never does. A schema
    # feature xgrammar cannot express (e.g. regex lookahead) would raise and,
    # under backend="auto", fall back to the guidance backend which crashes
    # on v2 structural tags. Degrade gracefully instead of erroring.
    tag_json = build_tag_json(request.tools)
    outcome = compiles(tag_json)
    if outcome is not True:
        logger.warning(
            "Qwen3XMLToolParser: tool schemas are not xgrammar-compatible "
            "(%s); retrying with unconstrained tool arguments.",
            outcome,
        )
        # Relax every tool's parameters to unconstrained. Not lossy vs. the
        # status quo: today's "auto" path does not constrain tool arguments
        # at all, and the XML call structure stays constrained.
        relaxed_tools = [tool.model_copy(deep=True) for tool in request.tools]
        for tool in relaxed_tools:
            tool.function.parameters = None
        tag_json = build_tag_json(relaxed_tools)
        outcome = compiles(tag_json)
        if outcome is not True:
            logger.warning(
                "Qwen3XMLToolParser: structural tag still uncompilable (%s); "
                "falling back to default tool calling.",
                outcome,
            )
            return None

    # Match AbstractToolParser.adjust_request's structural-tag style.
    if request.structured_outputs is None:
        request.structured_outputs = StructuredOutputsParams(
            structural_tag=tag_json,
        )
    else:
        request.structured_outputs.structural_tag = tag_json
    # Mandatory: otherwise to_sampling_params derives a second constraint
    # from response_format and the single-constraint validation raises.
    request.response_format = None
    return hint
