# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def inventory():
    if importlib.util.find_spec("tools.config_inventory") is None:
        pytest.skip("baseline has no inventory entry point")
    from tools import config_inventory

    return config_inventory


def test_profile_inventory_is_complete(inventory):
    data, errors = inventory.collect(inventory.ROOT)
    assert not errors
    assert len(data["profile_defaults"]) == 3


@pytest.mark.parametrize(
    "broken", ["registration", "condition", "format", "interconnect"]
)
def test_missing_metadata_is_rejected(inventory, tmp_path, broken):
    source = Path(inventory.ROOT / "vllm/envs.py").read_text()
    if broken == "registration":
        source = source.replace(
            '"VLLM_SM70_GREEDY_VERIFY": env_var(', '"VLLM_MISSING": env_var('
        )
    elif broken == "condition":
        source = source.replace(
            "vllm/revolta_profile.py:apply_revolta_profile", "missing_policy"
        )
    elif broken == "format":
        source = source.replace("Only AWQ checkpoints", "missing_format")
    else:
        source = source.replace("no_p2p", "missing_interconnect")
    (tmp_path / "vllm").mkdir()
    (tmp_path / "vllm/envs.py").write_text(source)
    (tmp_path / "vllm/revolta_profile.py").write_text(
        (inventory.ROOT / "vllm/revolta_profile.py").read_text()
    )
    (tmp_path / "vllm/config").mkdir()
    (tmp_path / "vllm/config/kernel.py").write_text(
        (inventory.ROOT / "vllm/config/kernel.py").read_text()
    )
    _, errors = inventory.collect(tmp_path)
    assert errors
