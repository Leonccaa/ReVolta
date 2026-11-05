# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""0.2.0rc5 promotion of the REVOLTA_* research switches to VLLM_SM70_* names."""

import warnings

import pytest

from vllm import envs

FLAGS = {
    "REVOLTA_GDN_CPU_SPEC_QSL": "VLLM_SM70_GDN_CPU_SPEC_QSL",
    "REVOLTA_QSA_GROUPED_LPT": "VLLM_SM70_QSA_GROUPED_LPT",
    "REVOLTA_SKINNY_MPLAN": "VLLM_SM70_GDN_SKINNY_MPLAN",
}
ALL = {
    **FLAGS,
    "REVOLTA_GDN_CPU_SPEC_QSL_CHECK_FILE": "VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE",
}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for old, new in ALL.items():
        monkeypatch.delenv(old, raising=False)
        monkeypatch.delenv(new, raising=False)
    envs.disable_envs_cache()
    yield
    envs.disable_envs_cache()


def test_mapping_matches_registry():
    assert envs.PROMOTED_RESEARCH_ENVS == ALL
    for old, new in ALL.items():
        assert old in envs.environment_variables
        assert new in envs.environment_variables


@pytest.mark.parametrize("old,new", FLAGS.items())
def test_default_off(old, new):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getattr(envs, new) is False


@pytest.mark.parametrize("old,new", FLAGS.items())
def test_new_name(monkeypatch, old, new):
    monkeypatch.setenv(new, "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getattr(envs, new) is True


@pytest.mark.parametrize("old,new", FLAGS.items())
def test_old_name_is_a_deprecated_alias(monkeypatch, old, new):
    monkeypatch.setenv(old, "1")
    with pytest.warns(FutureWarning, match=f"{old} is deprecated.*Use {new}"):
        assert getattr(envs, new) is True


@pytest.mark.parametrize("old,new", FLAGS.items())
def test_new_name_wins(monkeypatch, old, new):
    monkeypatch.setenv(old, "1")
    monkeypatch.setenv(new, "0")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getattr(envs, new) is False


def test_check_file_alias(monkeypatch, tmp_path):
    path = str(tmp_path / "check")
    monkeypatch.setenv("REVOLTA_GDN_CPU_SPEC_QSL_CHECK_FILE", path)
    with pytest.warns(FutureWarning):
        assert path == envs.VLLM_SM70_GDN_CPU_SPEC_QSL_CHECK_FILE


def test_switches_do_not_change_the_compile_cache_key(monkeypatch):
    base = envs.compile_factors()
    for old, new in ALL.items():
        monkeypatch.setenv(new, "1")
        monkeypatch.setenv(old, "1")
    envs.disable_envs_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert envs.compile_factors() == base
