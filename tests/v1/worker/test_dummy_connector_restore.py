# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ordinary worker dummy execution restores the connector after exceptions."""

from types import SimpleNamespace

import pytest

from vllm.v1.worker.gpu.model_runner import GPUModelRunner


def test_failed_dummy_execution_restores_connector():
    disabled: list[bool] = []

    def fail(*args, **kwargs):
        raise RuntimeError("injected model failure")

    runner = SimpleNamespace(
        max_num_reqs=8,
        is_first_pp_rank=True,
        execute_model=fail,
        kv_connector=SimpleNamespace(set_disabled=disabled.append),
    )
    with pytest.raises(RuntimeError, match="injected model failure"):
        GPUModelRunner._dummy_run(runner, 8)
    assert disabled == [True, False]
