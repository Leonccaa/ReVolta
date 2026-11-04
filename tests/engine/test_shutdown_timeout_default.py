# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReVolta defaults --shutdown-timeout to 30 (upstream vLLM: 0)."""

import dataclasses

import pytest

from vllm.config import VllmConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser


def _parse(argv):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    return EngineArgs.from_cli_args(parser.parse_args(argv))


def test_cli_default_is_30():
    assert _parse([]).shutdown_timeout == 30


@pytest.mark.parametrize("value", [0, 1, 30, 120])
def test_cli_explicit_value_is_kept(value):
    assert _parse(["--shutdown-timeout", str(value)]).shutdown_timeout == value


def test_dataclass_and_config_defaults_are_30():
    assert EngineArgs().shutdown_timeout == 30
    field = {f.name: f for f in dataclasses.fields(VllmConfig)}["shutdown_timeout"]
    default = getattr(field.default, "default", field.default)  # pydantic FieldInfo
    assert default == 30
