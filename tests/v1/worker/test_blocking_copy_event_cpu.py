# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The MRv2 output and draft-token copy events wait by sleeping, not by
spinning on the CUDA driver lock (upstream vLLM #47081).

Waiting on these events covers the remaining GPU time of every step; a
spinning wait keeps one CPU core per worker busy and contends for the driver
lock with other threads of the same process. CPU check of the construction
only; the latency effect needs a GPU.
"""

import ast
import inspect

import pytest

from vllm.v1.worker.gpu import async_utils
from vllm.v1.worker.gpu.spec_decode import utils as spec_utils


def _copy_event_calls(module):
    tree = ast.parse(inspect.getsource(module))
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "copy_event":
                calls.append(node.value)
    return calls


@pytest.mark.parametrize(
    "module, expected", [(async_utils, 2), (spec_utils, 1)], ids=["async", "draft"]
)
def test_copy_events_are_blocking(module, expected):
    calls = _copy_event_calls(module)
    assert len(calls) == expected
    for call in calls:
        assert isinstance(call, ast.Call)
        assert ast.unparse(call.func) == "torch.cuda.Event"
        kwargs = {k.arg: k.value for k in call.keywords}
        assert "blocking" in kwargs
        assert isinstance(kwargs["blocking"], ast.Constant)
        assert kwargs["blocking"].value is True
