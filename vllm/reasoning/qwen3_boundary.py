# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass


@dataclass
class Qwen3ReasoningBoundary:
    """First output boundary, identified by token origin, with text offsets.

    Both the frontend detokenizer and the reasoning parser use this state.
    Offsets describe decoded output, before stop lookback or API splitting.
    A tool opener retains the existing implicit-end compatibility.
    """

    start_token_id: int
    end_token_id: int
    tool_token_id: int | None
    reasoning_start: int = 0
    reasoning_end: int | None = None
    content_start: int | None = None
    implicit_end: bool = False

    @property
    def ended(self) -> bool:
        return self.content_start is not None

    def observe(self, token_id: int, before: int, after: int) -> None:
        if self.ended:
            return
        if token_id == self.start_token_id:
            self.reasoning_start = after
        elif token_id == self.end_token_id:
            self.reasoning_end = before
            self.content_start = after
        elif self.tool_token_id is not None and token_id == self.tool_token_id:
            self.reasoning_end = before
            self.content_start = before
            self.implicit_end = True
