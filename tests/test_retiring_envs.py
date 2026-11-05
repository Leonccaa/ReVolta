# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""0.2.0rc5: DDTree and glm5next switches are deprecated (removed in rc6);
the ngram_assist lookup knobs are documented as experimental."""

import logging
import warnings
from pathlib import Path

import pytest

from tools.pre_commit.check_env_metadata import read_metadata
from vllm import envs

ROOT = Path(__file__).resolve().parents[1]
META, _ = read_metadata((ROOT / "vllm/envs.py").read_text())


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for name in envs.RETIRING_ENVS:
        monkeypatch.delenv(name, raising=False)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_retiring_set_is_ddtree_plus_glm5next():
    ddtree = {
        n for n in envs.environment_variables if n.startswith("VLLM_DFLASH_DDTREE_")
    }
    assert len(ddtree) == 39
    assert set(envs.RETIRING_ENVS) == ddtree | {
        "VLLM_DFLASH_DEBUG_COORD_TRACE",
        "VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT",
    }
    for name in envs.RETIRING_ENVS:
        assert META[name]["category"] == "deprecated"


def test_startup_warns_once_per_set_switch(monkeypatch, caplog):
    monkeypatch.setenv("VLLM_DFLASH_DDTREE_DEBUG", "1")
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_COORD_TRACE", "1")
    with caplog.at_level(logging.WARNING):
        envs.validate_environ(hard_fail=False)
    text = caplog.text
    assert text.count("VLLM_DFLASH_DDTREE_DEBUG is deprecated") == 1
    assert text.count("VLLM_DFLASH_DEBUG_COORD_TRACE is deprecated") == 1
    assert "0.2.0rc6" in text


def test_startup_silent_when_unset(caplog):
    with caplog.at_level(logging.WARNING):
        envs.validate_environ(hard_fail=False)
    assert "is deprecated" not in caplog.text


def test_reading_keeps_the_value_and_warns(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH_DDTREE_DEBUG", "1")
    monkeypatch.setenv("VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT", "5")
    with pytest.warns(FutureWarning):
        assert envs.VLLM_DFLASH_DDTREE_DEBUG is True
    with pytest.warns(FutureWarning):
        assert envs.VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT == "5"


def test_unset_reads_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert envs.VLLM_DFLASH_DDTREE_DEBUG is False
        assert envs.VLLM_DFLASH_DEBUG_COORD_TRACE is None


def test_lookup_knobs_are_experimental():
    lookup = [n for n in META if n.startswith("VLLM_DFLASH2_LOOKUP_")]
    assert len(lookup) == 9
    for name in lookup:
        assert META[name]["category"] == "experimental"
        assert "ngram_assist" in META[name]["description"]


def test_glm5next_sites_read_through_envs():
    for path in (
        "vllm/model_executor/layers/sparse_attn_indexer_kpool.py",
        "vllm/models/glm5next/sm70/sparse.py",
        "vllm/models/glm5next/nvidia/model.py",
    ):
        text = (ROOT / path).read_text()
        for name in (
            "VLLM_DFLASH_DEBUG_COORD_TRACE",
            "VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT",
        ):
            assert f'os.getenv("{name}"' not in text, (path, name)


def _old_coord_flag(raw):
    # Original glm5next sparse.py: bool(int(os.getenv(name, "0")))
    return bool(int("0" if raw is None else raw))


def _old_dump_limit(raw):
    # Original glm5next model.py: max(0, int(os.getenv(name, "2")))
    return max(0, int("2" if raw is None else raw))


def _run(expr_source: str, raw):
    from types import SimpleNamespace

    ns = {
        "envs": SimpleNamespace(
            VLLM_DFLASH_DEBUG_COORD_TRACE=raw, VLLM_DFLASH_DEBUG_PP_AUX_DUMP_LIMIT=raw
        )
    }
    exec(expr_source, ns)  # noqa: S102 - test-only, source from this repo
    return ns


def _outcome(fn):
    try:
        return ("value", fn())
    except ValueError as exc:
        return ("ValueError", type(exc).__name__)


RAWS = [None, "", "0", "1", "3", "-3"]


@pytest.mark.parametrize("raw", RAWS)
def test_glm5next_sparse_flag_parses_like_before(raw):
    src = (ROOT / "vllm/models/glm5next/sm70/sparse.py").read_text()
    start = src.index("_COORD_TRACE = envs.")
    end = src.index("\n", src.index("_DEBUG_DFLASH_SPARSE_INDICES = ", start))
    code = src[start:end]
    new = _outcome(lambda: _run(code, raw)["_DEBUG_DFLASH_SPARSE_INDICES"])
    assert new == _outcome(lambda: _old_coord_flag(raw))


@pytest.mark.parametrize("raw", RAWS)
def test_glm5next_dump_limit_parses_like_before(raw):
    import ast

    tree = ast.parse((ROOT / "vllm/models/glm5next/nvidia/model.py").read_text())
    stmts = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = ast.unparse(node.targets[0])
            if target in ("dump_limit", "self._debug_pp_aux_dump_limit"):
                stmts[target] = ast.unparse(node.value)
    code = (
        f"dump_limit = {stmts['dump_limit']}\n"
        f"result = {stmts['self._debug_pp_aux_dump_limit']}"
    )
    new = _outcome(lambda: _run(code, raw)["result"])
    assert new == _outcome(lambda: _old_dump_limit(raw))


@pytest.mark.parametrize("raw", RAWS)
def test_kpool_coord_trace_compares_like_before(raw):
    src = (ROOT / "vllm/model_executor/layers/sparse_attn_indexer_kpool.py").read_text()
    assert 'envs.VLLM_DFLASH_DEBUG_COORD_TRACE == "1"' in src
    # Original: os.getenv(name, "0") == "1"
    old = ("0" if raw is None else raw) == "1"
    assert (raw == "1") == old
