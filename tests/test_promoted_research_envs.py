# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Promotion of the REVOLTA_* research switches to VLLM_SM70_* names.

0.2.0rc5 promotions kept the promoted default (off); 0.2.0rc12 promotions
turned their switches on by default. In both cases the old name remains a
deprecated alias for one release and the new name wins when both are set.
"""

import warnings

import pytest

from vllm import envs

# old name -> (new name, default when neither is set)
FLAGS = {
    "REVOLTA_QSA_GROUPED_LPT": ("VLLM_SM70_QSA_GROUPED_LPT", False),
    "REVOLTA_SKINNY_MPLAN": ("VLLM_SM70_GDN_SKINNY_MPLAN", False),
    "REVOLTA_RESEARCH_HOST_DECODE_FAST": ("VLLM_SM70_HOST_DECODE_FAST", True),
    "REVOLTA_RESEARCH_QSA_LOCAL_INPUTS_FUSED": (
        "VLLM_SM70_QSA_LOCAL_INPUTS_FUSED",
        True,
    ),
    "REVOLTA_RESEARCH_CHECKPOINT_CACHE_DROP": (
        "VLLM_SM70_CHECKPOINT_CACHE_DROP",
        True,
    ),
}
ALL = {k: v[0] for k, v in FLAGS.items()}


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


@pytest.mark.parametrize("old,new_default", FLAGS.items())
def test_default_follows_the_promoted_value(old, new_default):
    new, default = new_default
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getattr(envs, new) is default


@pytest.mark.parametrize("old,new_default", FLAGS.items())
@pytest.mark.parametrize("raw", ["1", "0"])
def test_new_name_explicit_values_win(monkeypatch, old, new_default, raw):
    new, _ = new_default
    monkeypatch.setenv(new, raw)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getattr(envs, new) is (raw == "1")


@pytest.mark.parametrize("old,new_default", FLAGS.items())
def test_old_name_is_a_deprecated_alias(monkeypatch, old, new_default):
    new, _ = new_default
    monkeypatch.setenv(old, "1")
    with pytest.warns(FutureWarning, match=f"{old} is deprecated.*Use {new}"):
        assert getattr(envs, new) is True


@pytest.mark.parametrize("old,new_default", FLAGS.items())
def test_old_name_can_still_disable(monkeypatch, old, new_default):
    new, _ = new_default
    monkeypatch.setenv(old, "0")
    with pytest.warns(FutureWarning):
        assert getattr(envs, new) is False


@pytest.mark.parametrize("old,new_default", FLAGS.items())
def test_new_name_wins(monkeypatch, old, new_default):
    new, default = new_default
    monkeypatch.setenv(old, "1" if not default else "0")
    monkeypatch.setenv(new, "0" if default else "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert getattr(envs, new) is not default


def test_switches_do_not_change_the_compile_cache_key(monkeypatch):
    base = envs.compile_factors()
    for old, new in ALL.items():
        monkeypatch.setenv(new, "1")
        monkeypatch.setenv(old, "1")
    envs.disable_envs_cache()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert envs.compile_factors() == base
