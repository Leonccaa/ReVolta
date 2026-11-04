# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""tools/gen_env_docs.py renders a current, deterministic reference."""

from tools import gen_env_docs
from tools.env_docs_notes import DFLASH2_NOTES, NOTES


def test_checked_in_reference_is_current():
    assert gen_env_docs.main(["--check"]) == 0


def test_render_is_deterministic_and_complete():
    first, second = gen_env_docs.render(), gen_env_docs.render()
    assert first == second
    for name in NOTES:
        assert f"| `{name}` |" in first
    for key in DFLASH2_NOTES:
        assert f"| `{key}` |" in first


def test_updated_replaces_only_the_block():
    block = gen_env_docs.render()
    text = "# Title\n\nintro\n\n## See also\n\n- x\n"
    once = gen_env_docs.updated(text, block)
    assert once.startswith("# Title\n\nintro\n\n" + gen_env_docs.BEGIN)
    assert once.endswith("\n## See also\n\n- x\n")
    assert gen_env_docs.updated(once, block) == once
