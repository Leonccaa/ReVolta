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


def test_missing_type_annotation_is_an_error(monkeypatch):
    import pytest

    probe = "VLLM_SM70_UNTYPED_PROBE"
    notes = dict(gen_env_docs.NOTES)
    notes[probe] = dict(
        section="profile", since="0.2.0", range="0, 1", tradeoff="x", metrics=""
    )
    original = gen_env_docs.read_metadata

    def with_probe(source):
        metadata, errors = original(source)
        metadata = dict(metadata)
        metadata[probe] = {"declared_default": "False", "category": "debug"}
        return metadata, errors

    monkeypatch.setattr(gen_env_docs, "NOTES", notes)
    monkeypatch.setattr(gen_env_docs, "read_metadata", with_probe)
    with pytest.raises(SystemExit, match="TYPE_CHECKING"):
        gen_env_docs.render()
