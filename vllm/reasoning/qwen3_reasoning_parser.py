# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.reasoning.qwen3_boundary import Qwen3ReasoningBoundary

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest
    from vllm.tokenizers import TokenizerLike


class Qwen3ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for the Qwen3/Qwen3.5 model family.

    The Qwen3 model family uses <think>...</think> tokens to denote reasoning
    text. Starting with Qwen3.5, the chat template places <think> in the
    prompt so only </think> appears in the generated output. The model
    provides a strict switch to disable reasoning output via the
    'enable_thinking=False' parameter.

    When thinking is disabled, the template places <think>\\n\\n</think>\\n\\n
    in the prompt. The serving layer detects this via prompt_is_reasoning_end
    and routes deltas as content without calling the streaming parser.

    NOTE: Models up to the 2507 release (e.g., Qwen/Qwen3-235B-A22B-Instruct-2507)
    use an older chat template where the model generates <think> itself.
    This parser handles both styles: if <think> appears in the generated output
    it is stripped before extraction (non-streaming) or skipped (streaming).

    NOTE: Qwen3.5 models may emit <tool_call> inside the thinking block
    without closing </think> first. <tool_call> is treated as an implicit
    end of reasoning, matching the approach in KimiK2ReasoningParser.
    """

    def __init__(self, tokenizer: "TokenizerLike", *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)

        chat_kwargs = kwargs.get("chat_template_kwargs", {}) or {}
        # Qwen3 defaults to thinking enabled; only treat output as
        # pure content when the user explicitly disables it.
        self.thinking_enabled = chat_kwargs.get("enable_thinking", True)

        self._tool_call_tag = "<tool_call>"
        self._tool_call_token_id = self.vocab.get(self._tool_call_tag)
        self._tool_call_end_tag = "</tool_call>"
        self._tool_call_end_token_id = self.vocab.get(self._tool_call_end_tag)
        self._stream_boundary = self._new_boundary()
        self._boundary_tokens_seen = 0

    @property
    def start_token(self) -> str:
        """The token that starts reasoning content."""
        return "<think>"

    @property
    def end_token(self) -> str:
        """The token that ends reasoning content."""
        return "</think>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        tool_call_token_id = self._tool_call_token_id
        tool_call_end_token_id = self._tool_call_end_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            token_id = input_ids[i]
            if token_id == start_token_id:
                # Found <think> before </think> or <tool_call>
                return False
            if token_id == end_token_id:
                return True
            if tool_call_token_id is not None and token_id == tool_call_token_id:
                # Only treat as implicit reasoning end if this <tool_call>
                # is NOT followed by </tool_call>.  Paired occurrences are
                # template examples in the prompt, not model output.
                if tool_call_end_token_id is not None and any(
                    input_ids[j] == tool_call_end_token_id
                    for j in range(i + 1, len(input_ids))
                ):
                    continue
                return True
        return False

    def is_reasoning_end_streaming(
        self, input_ids: Sequence[int], delta_ids: Iterable[int]
    ) -> bool:
        if super().is_reasoning_end_streaming(input_ids, delta_ids):
            return True
        if self._tool_call_token_id is not None:
            return self._tool_call_token_id in delta_ids
        return False

    def extract_content_ids(self, input_ids: list[int]) -> list[int]:
        """
        Extract content token ids from the input_ids.
        """
        result = super().extract_content_ids(input_ids)
        if result:
            return result
        # Fall back: content starts at <tool_call> (implicit reasoning end).
        if (
            self._tool_call_token_id is not None
            and self._tool_call_token_id in input_ids
        ):
            tool_call_index = (
                len(input_ids) - 1 - input_ids[::-1].index(self._tool_call_token_id)
            )
            return input_ids[tool_call_index:]
        return []

    def extract_reasoning(
        self, model_output: str, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> tuple[str | None, str | None]:
        """
        Extract reasoning content from the model output.

        The <think> token is placed in the prompt by the chat template,
        so typically only </think> appears in the generated output.
        If <think> is present (e.g. from a different template), it is
        stripped before extraction.

        When thinking is explicitly disabled and no </think> appears,
        returns (None, model_output) — all output is content.
        Otherwise (thinking enabled, default), a missing </think> means
        the output was truncated and everything is reasoning:
        returns (model_output, None).

        Returns:
            tuple[Optional[str], Optional[str]]: reasoning content and content
        """

        # Strip <think> if present in the generated output.
        model_output_parts = model_output.partition(self.start_token)
        model_output = (
            model_output_parts[2] if model_output_parts[1] else model_output_parts[0]
        )

        if self.end_token in model_output:
            reasoning, _, content = model_output.partition(self.end_token)
            return reasoning, content or None

        if not self.thinking_enabled:
            # Thinking explicitly disabled — treat everything as content.
            return None, model_output

        # No </think> — check for implicit reasoning end via <tool_call>.
        tool_call_index = model_output.find(self._tool_call_tag)
        if tool_call_index != -1:
            reasoning = model_output[:tool_call_index]
            content = model_output[tool_call_index:]
            return reasoning or None, content or None
        # Thinking enabled but no </think>: output was truncated.
        # Everything generated so far is reasoning.
        return model_output, None

    def _new_boundary(self) -> Qwen3ReasoningBoundary:
        return Qwen3ReasoningBoundary(
            self.start_token_id, self.end_token_id, self._tool_call_token_id
        )

    def _observe_boundary(
        self,
        boundary: Qwen3ReasoningBoundary,
        ids: Sequence[int],
        start: int,
        text: str,
    ) -> None:
        # Decode only marker prefixes, once per boundary, rather than searching
        # for their spelling. Ordinary BPE tokens can spell the same literal.
        for i in range(start, len(ids)):
            if boundary.ended:
                break
            token_id = ids[i]
            if token_id not in (
                self.start_token_id,
                self.end_token_id,
                self._tool_call_token_id,
            ):
                continue
            before = self.model_tokenizer.decode(
                list(ids[:i]), skip_special_tokens=False
            )
            after = self.model_tokenizer.decode(
                list(ids[: i + 1]), skip_special_tokens=False
            )
            # Some tokenizers classify the markers as special and remove them.
            # Choose the decode mode matching the actual detokenized prefix.
            if not after.startswith(text) and not text.startswith(after):
                before = self.model_tokenizer.decode(
                    list(ids[:i]), skip_special_tokens=True
                )
                after = self.model_tokenizer.decode(
                    list(ids[: i + 1]), skip_special_tokens=True
                )
            boundary.observe(token_id, len(before), len(after))

    def _boundary_matches_text(
        self, boundary: Qwen3ReasoningBoundary, text: str
    ) -> bool:
        """Check that token-derived offsets bracket the markers in text."""
        start = boundary.reasoning_start
        if start > len(text) or (start and not text[:start].endswith(self.start_token)):
            return False
        if boundary.content_start is None:
            return True
        end, content_start = boundary.reasoning_end, boundary.content_start
        if end is None or not start <= end <= content_start <= len(text):
            return False
        if boundary.implicit_end:
            return end == content_start and text.startswith(
                self._tool_call_tag, content_start
            )
        return text[end:content_start] == self.end_token

    def extract_reasoning_with_token_ids(
        self,
        model_output: str,
        token_ids: Sequence[int],
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        if not self.thinking_enabled:
            # The chat template already closed reasoning in the prompt, and
            # streaming routes every delta to content. A literal </think> in
            # the answer must not split it here either.
            return None, model_output
        boundary = self._new_boundary()
        self._observe_boundary(boundary, token_ids, 0, model_output)
        if not self._boundary_matches_text(boundary, model_output):
            # Offsets come from decoding token prefixes. If that decode does
            # not reproduce model_output around the markers, they would move
            # the split silently; use the text-based split instead.
            return self.extract_reasoning(model_output, request)
        reasoning = model_output[boundary.reasoning_start : boundary.reasoning_end]
        content = (
            model_output[boundary.content_start :]
            if boundary.content_start is not None
            else None
        )
        # Preserve the legacy empty-string result for an explicit end (or
        # truncated empty output); only implicit tool entry uses None.
        return (
            reasoning or None
        ) if boundary.implicit_end else reasoning, content or None

    def is_reasoning_output_end(
        self, input_ids: Sequence[int], delta_ids: Sequence[int], text: str
    ) -> bool:
        boundary = self._stream_boundary
        return (
            boundary.content_start is not None and len(text) >= boundary.content_start
        )

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        if not previous_token_ids:
            self._stream_boundary = self._new_boundary()
            self._boundary_tokens_seen = 0
        boundary = self._stream_boundary
        self._observe_boundary(
            boundary, current_token_ids, self._boundary_tokens_seen, current_text
        )
        self._boundary_tokens_seen = len(current_token_ids)
        # Token ids may be ahead of text because stop lookback holds a suffix.
        # Keep assigning delayed text to reasoning until its true offset, and
        # suppress only the actual marker span (possibly across many deltas).
        lo, hi = len(previous_text), len(current_text)
        reasoning_end = boundary.reasoning_end
        reasoning = current_text[
            max(lo, boundary.reasoning_start) : min(hi, reasoning_end)
            if reasoning_end is not None
            else hi
        ]
        content = (
            current_text[max(lo, boundary.content_start) : hi]
            if boundary.content_start is not None
            else ""
        )
        if not reasoning and not content:
            return None
        # Preserve unset fields on ordinary deltas. Serving serializes unset
        # and explicitly-null fields differently, even if their values agree.
        if boundary.content_start is None or (
            boundary.reasoning_end is not None and hi <= boundary.reasoning_end
        ):
            return DeltaMessage(reasoning=reasoning)
        if lo >= boundary.content_start:
            return DeltaMessage(content=content)
        return DeltaMessage(reasoning=reasoning or None, content=content or None)
