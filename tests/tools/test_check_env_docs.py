# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""check_env_docs: documented, well-named ReVolta variables; shrinking baseline."""

from pathlib import Path

import pytest

from tools.pre_commit import check_env_docs

ROOT = Path(__file__).resolve().parents[2]

ENVS = """
environment_variables = {
    "VLLM_SM70_QSA_GROUPED_LPT": lambda: False,
    "VLLM_DFLASH2_OLD_SWITCH": lambda: False,
    "VLLM_HOST_IP": lambda: "",
}
"""


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "vllm").mkdir()
    (tmp_path / "docs").mkdir()
    (tmp_path / "vllm/envs.py").write_text(ENVS)
    return tmp_path


def run(repo, docs="", baseline=""):
    (repo / "docs/configuration.md").write_text(docs)
    path = repo / "baseline.txt"
    path.write_text(baseline)
    return check_env_docs.check(repo, path)


def test_undocumented_and_legacy_names_fail_without_baseline(repo):
    errors = run(repo)
    assert any(
        "VLLM_SM70_QSA_GROUPED_LPT: registered but not documented" in e for e in errors
    )
    assert any("VLLM_DFLASH2_OLD_SWITCH: name must be" in e for e in errors)
    # Inherited non-ReVolta prefixes are out of scope.
    assert not any("VLLM_HOST_IP" in e for e in errors)


def test_documented_and_baselined_pass(repo):
    errors = run(
        repo,
        docs="| `VLLM_SM70_QSA_GROUPED_LPT` | bool | off |\n",
        baseline="VLLM_DFLASH2_OLD_SWITCH undocumented\n"
        "VLLM_DFLASH2_OLD_SWITCH legacy-name\n",
    )
    assert errors == []


def test_baseline_only_shrinks(repo):
    errors = run(
        repo,
        docs="VLLM_SM70_QSA_GROUPED_LPT VLLM_DFLASH2_OLD_SWITCH",
        baseline="VLLM_SM70_QSA_GROUPED_LPT undocumented\n"
        "VLLM_DFLASH2_OLD_SWITCH legacy-name\n"
        "VLLM_SM70_QSA_GROUPED_LPT legacy-name\n"
        "VLLM_SM70_GONE undocumented\n"
        "VLLM_SM70_QSA_GROUPED_LPT typo\n",
    )
    assert any("is documented now; remove it" in e for e in errors)
    assert any("follows the naming rule; remove it" in e for e in errors)
    assert any("VLLM_SM70_GONE is no longer registered" in e for e in errors)
    assert any("unknown reason 'typo'" in e for e in errors)


def test_name_must_not_match_as_substring(repo):
    errors = run(repo, docs="VLLM_SM70_QSA_GROUPED_LPT_EXTRA")
    assert any(
        "VLLM_SM70_QSA_GROUPED_LPT: registered but not documented" in e for e in errors
    )


@pytest.mark.parametrize(
    "name, ok",
    [
        ("VLLM_SM70_QSA_GROUPED_LPT", True),
        ("REVOLTA_RESEARCH_GDN_CPU_SPEC_QSL", True),
        ("REVOLTA_GDN_CPU_SPEC_QSL", False),
        ("VLLM_SM70_LPT", False),
        ("VLLM_DFLASH2_FOO", False),
    ],
)
def test_name_rules(name, ok):
    assert any(r.match(name) for r in check_env_docs.NAME_RULES) is ok


def test_repository_passes_with_checked_in_baseline():
    assert check_env_docs.check(ROOT) == []
