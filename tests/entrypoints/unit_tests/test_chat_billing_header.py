# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Claude Code's attribution header in chat completions system messages."""

from copy import deepcopy
from unittest.mock import MagicMock

import pytest

from vllm.entrypoints.chat_utils import _parse_chat_message_content

HEADER = (
    "x-anthropic-billing-header: cc_version=2.1.220.300; "
    "cc_entrypoint=claude-desktop; cch=7adab;"
)


def _parse(message, content_format="openai"):
    tracker = MagicMock()
    tracker.create_parser.return_value.mm_placeholder_storage.return_value = {}
    return _parse_chat_message_content(
        message, tracker, content_format, interleave_strings=False
    )


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "header",
    [
        HEADER,
        "x-anthropic-billing-header: cc_version=2.1.286.99b; cc_entrypoint=sdk-cli;",
    ],
)
def test_system_header_part_is_dropped(header):
    (msg,) = _parse(
        {
            "role": "system",
            "content": [
                {"type": "text", "text": header},
                {"type": "text", "text": "You are a Claude agent."},
            ],
        }
    )
    assert msg["content"] == [{"type": "text", "text": "You are a Claude agent."}]


@pytest.mark.skip_global_cleanup
def test_system_header_line_is_stripped_from_string():
    (msg,) = _parse(
        {"role": "system", "content": HEADER + "\nYou are a Claude agent."},
        content_format="string",
    )
    assert msg["content"] == "You are a Claude agent."


@pytest.mark.skip_global_cleanup
def test_requests_differing_only_in_cch_parse_identically():
    def system(cch):
        return {
            "role": "system",
            "content": HEADER.replace("7adab", cch) + "\nYou are a Claude agent.",
        }

    assert _parse(system("7adab")) == _parse(system("986dd"))


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize(
    "message",
    [
        {"role": "system", "content": "You are a Claude agent."},
        {"role": "system", "content": "Quote: " + HEADER},
        {"role": "user", "content": HEADER + "Hello"},
    ],
)
def test_other_content_is_unchanged(message):
    (msg,) = _parse(message, content_format="string")
    assert msg["content"] == message["content"]


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("content_format", ["string", "openai"])
def test_only_first_header_line_is_removed_without_mutating_request(content_format):
    remainder = "\n  Preserve spaces 中文\r\n" + HEADER + "\nlast line\n"
    message = {
        "role": "system",
        "content": [{"type": "text", "text": HEADER + "\n" + remainder}],
    }
    original = deepcopy(message)
    (msg,) = _parse(message, content_format=content_format)
    expected = (
        remainder
        if content_format == "string"
        else [{"type": "text", "text": remainder}]
    )
    assert msg["content"] == expected
    assert message == original


@pytest.mark.skip_global_cleanup
@pytest.mark.parametrize("role", ["user", "assistant", "developer"])
def test_non_system_roles_keep_header_and_all_following_bytes(role):
    message = {"role": role, "content": HEADER + "\n\n  中文\n"}
    (msg,) = _parse(message, content_format="string")
    assert msg["content"] == message["content"]
