# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Helpers for the Hy4-flash parser tests.

They use the checkpoint's real tokenizer and ``chat_template.jinja``. Point
``HY4_FLASH_DIR`` at a directory holding ``tokenizer.json``,
``tokenizer_config.json`` and ``chat_template.jinja`` of ``tencent/Hy4-flash``
(revision ``d88c9dfd96ba``); otherwise the files are fetched from the Hub.
"""

import json
import os
from dataclasses import dataclass, field
from typing import Any

import pytest
import tokenizers

from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatCompletionToolsParam,
    FunctionDefinition,
)
from vllm.tokenizers import TokenizerLike, get_tokenizer

MODEL = "tencent/Hy4-flash"
REVISION = "d88c9dfd96ba4f2e6b92adba0b8e8f1c8696f959"
SUFFIX = ":opensource"

# Structural tokens, spelled as the chat template does.
THINK_BEGIN = f"<think{SUFFIX}>"
THINK_END = f"</think{SUFFIX}>"
TOOL_CALLS_BEGIN = f"<tool_calls{SUFFIX}>"
TOOL_CALLS_END = f"</tool_calls{SUFFIX}>"
TOOL_CALL_BEGIN = f"<tool_call{SUFFIX}>"
TOOL_CALL_END = f"</tool_call{SUFFIX}>"
ARG_KEY_BEGIN = f"<arg_key{SUFFIX}>"
ARG_KEY_END = f"</arg_key{SUFFIX}>"
ARG_VALUE_BEGIN = f"<arg_value{SUFFIX}>"
ARG_VALUE_END = f"</arg_value{SUFFIX}>"
HY_END = f"<｜hy_end{SUFFIX}｜>"


def _model_dir() -> str:
    path = os.environ.get("HY4_FLASH_DIR")
    if path:
        return path
    try:
        from huggingface_hub import snapshot_download

        return snapshot_download(
            MODEL,
            revision=REVISION,
            allow_patterns=[
                "tokenizer.json",
                "tokenizer_config.json",
                "chat_template.jinja",
            ],
        )
    except Exception as e:  # offline or no access
        pytest.skip(f"Hy4-flash tokenizer files unavailable: {e}")
        raise


def load_tokenizer() -> tuple[TokenizerLike, str]:
    path = _model_dir()
    with open(os.path.join(path, "chat_template.jinja"), encoding="utf-8") as f:
        template = f.read()
    return get_tokenizer(path), template


def tool(name: str, parameters: dict[str, Any] | None = None):
    return ChatCompletionToolsParam(
        function=FunctionDefinition(name=name, parameters=parameters or {})
    )


TOOLS = [
    tool(
        "get_weather",
        {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "unit": {"type": "string", "enum": ["c", "f"]},
                "days": {"type": "integer"},
            },
        },
    ),
    tool(
        "search",
        {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "filters": {"type": "object"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "strict": {"type": "boolean"},
                "limit": {"type": "number"},
                "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
    ),
    tool("get_time"),
]


def make_request(
    messages: list[dict[str, Any]],
    *,
    tools: bool = True,
    tool_choice: Any = "auto",
    **kwargs: Any,
) -> ChatCompletionRequest:
    extra: dict[str, Any] = {}
    if tools:
        extra["tools"] = tool_dicts()
        extra["tool_choice"] = tool_choice
    return ChatCompletionRequest(model=MODEL, messages=messages, **extra, **kwargs)


def tool_dicts() -> list[dict[str, Any]]:
    return [t.model_dump(exclude_none=True) for t in TOOLS]


def assistant_turn(
    *,
    reasoning: str | None = None,
    content: str = "",
    tool_calls: list[tuple[str, dict[str, Any]]] | None = None,
    ids: list[str] | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning:
        msg["reasoning_content"] = reasoning
    if tool_calls:
        msg["tool_calls"] = [
            {
                "id": ids[i] if ids else f"call_{i}",
                "type": "function",
                "function": {"name": n, "arguments": a},
            }
            for i, (n, a) in enumerate(tool_calls)
        ]
    return msg


def render(
    tokenizer: TokenizerLike,
    template: str,
    messages: list[dict[str, Any]],
    *,
    tools: bool = True,
    generation_prompt: bool = True,
    **kwargs: Any,
) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tools=tool_dicts() if tools else None,
        chat_template=template,
        tokenize=False,
        add_generation_prompt=generation_prompt,
        **kwargs,
    )


def model_output_via_template(
    tokenizer: TokenizerLike,
    template: str,
    history: list[dict[str, Any]],
    assistant: dict[str, Any],
    *,
    tools: bool = True,
    **kwargs: Any,
) -> tuple[str, str]:
    """Return ``(prompt, model_output)`` for the next assistant turn.

    The output is what the template writes for ``assistant`` after the
    generation prompt, i.e. what the model is trained to produce.
    """
    prompt = render(tokenizer, template, history, tools=tools, **kwargs)
    full = render(
        tokenizer,
        template,
        [*history, assistant],
        tools=tools,
        generation_prompt=False,
        **kwargs,
    )
    assert full.startswith(prompt)
    assert full.endswith(HY_END)
    return prompt, full[len(prompt) : -len(HY_END)]


def encode(tokenizer: TokenizerLike, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def stream_deltas(
    tokenizer: TokenizerLike,
    prompt_ids: list[int],
    output_ids: list[int],
    chunk: int = 1,
) -> list[tuple[str, list[int]]]:
    """Detokenize like the engine: ``chunk`` new tokens per streamed delta."""
    stream = tokenizers.decoders.DecodeStream(ids=prompt_ids, skip_special_tokens=True)
    raw = tokenizer._tokenizer  # type: ignore[attr-defined]
    deltas: list[tuple[str, list[int]]] = []
    text = ""
    ids: list[int] = []
    for i, token_id in enumerate(output_ids):
        piece = stream.step(raw, token_id)
        text += piece or ""
        ids.append(token_id)
        if len(ids) == chunk or i == len(output_ids) - 1:
            deltas.append((text, ids))
            text, ids = "", []
    return deltas


@dataclass
class Parsed:
    reasoning: str | None = None
    content: str | None = None
    # index -> {"id", "name", "arguments"}
    tool_calls: dict[int, dict[str, Any]] = field(default_factory=dict)
    ids_minted: list[str] = field(default_factory=list)


def make_parser_cls():
    from vllm.parser.abstract_parser import _WrappedParser
    from vllm.reasoning.hy_v4_reasoning_parser import HYV4ReasoningParser
    from vllm.tool_parsers.hy_v4_tool_parser import HYV4ToolParser

    cls = type(
        "HYV4TestParser",
        (_WrappedParser,),
        {
            "reasoning_parser_cls": HYV4ReasoningParser,
            "tool_parser_cls": HYV4ToolParser,
        },
    )
    return cls


def new_parser(tokenizer, request: ChatCompletionRequest, chat_template_kwargs):
    return make_parser_cls()(
        tokenizer,
        request.tools,
        chat_template_kwargs=chat_template_kwargs,
    )


def parse_streaming(
    tokenizer: TokenizerLike,
    request: ChatCompletionRequest,
    prompt: str,
    output: str,
    chat_template_kwargs: dict[str, Any] | None = None,
    chunk: int = 1,
) -> Parsed:
    parser = new_parser(tokenizer, request, chat_template_kwargs or {})
    prompt_ids = encode(tokenizer, prompt)
    output_ids = encode(tokenizer, output)
    deltas = stream_deltas(tokenizer, prompt_ids, output_ids, chunk)
    assert "".join(t for t, _ in deltas) == output

    out = Parsed()
    for i, (text, ids) in enumerate(deltas):
        msg = parser.parse_delta(
            delta_text=text,
            delta_token_ids=ids,
            request=request,
            prompt_token_ids=prompt_ids,
            finished=i == len(deltas) - 1,
        )
        if msg is None:
            continue
        if msg.reasoning:
            out.reasoning = (out.reasoning or "") + msg.reasoning
        if msg.content:
            out.content = (out.content or "") + msg.content
        for tc in msg.tool_calls:
            entry = out.tool_calls.setdefault(
                tc.index, {"id": None, "name": None, "arguments": ""}
            )
            if tc.id is not None:
                assert entry["id"] is None, "tool call id minted twice"
                entry["id"] = tc.id
                out.ids_minted.append(tc.id)
            if tc.function is not None:
                if tc.function.name is not None:
                    assert entry["name"] is None, "tool name sent twice"
                    entry["name"] = tc.function.name
                if tc.function.arguments is not None:
                    entry["arguments"] += tc.function.arguments
    return out


def parse_nonstreaming(
    tokenizer: TokenizerLike,
    request: ChatCompletionRequest,
    output: str,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> Parsed:
    """Same order as the chat serving layer: reasoning first, then tools."""
    parser = new_parser(tokenizer, request, chat_template_kwargs or {})
    reasoning_parser = parser.reasoning_parser
    tool_parser = parser.tool_parser
    assert reasoning_parser is not None and tool_parser is not None
    reasoning, content = reasoning_parser.extract_reasoning_with_token_ids(
        output, encode(tokenizer, output), request
    )
    out = Parsed(reasoning=reasoning, content=content)
    info = tool_parser.extract_tool_calls(content or "", request)
    if info.tools_called:
        out.content = info.content
        for i, tc in enumerate(info.tool_calls):
            out.tool_calls[i] = {
                "id": tc.id,
                "name": tc.function.name,
                "arguments": tc.function.arguments,
            }
    return out


def tool_calls_json(parsed: Parsed) -> list[tuple[str, Any]]:
    return [
        (c["name"], json.loads(c["arguments"]))
        for _, c in sorted(parsed.tool_calls.items())
    ]
