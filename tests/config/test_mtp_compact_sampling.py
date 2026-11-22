# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compact MTP policy selection needs no model or accelerator."""

import json

import pytest

from vllm.config import KernelConfig
from vllm.engine.arg_utils import EngineArgs
from vllm.utils.argparse_utils import FlexibleArgumentParser

pytestmark = pytest.mark.skip_global_cleanup


@pytest.mark.parametrize("mode", [False, True, "device_top64"])
def test_compact_sampling_cli_and_hash(mode):
    parser = EngineArgs.add_cli_args(FlexibleArgumentParser())
    parsed = parser.parse_args(
        ["--kernel-config", json.dumps({"sm70_sparse_target_sampling": mode})]
    )
    config = EngineArgs.from_cli_args(parsed).kernel_config
    assert isinstance(config, KernelConfig)
    assert config.sm70_sparse_target_sampling == mode
    assert type(config.sm70_sparse_target_sampling) is type(mode)
    default = KernelConfig()
    assert default.sm70_sparse_target_sampling is False
    assert config.compute_hash() == default.compute_hash()
