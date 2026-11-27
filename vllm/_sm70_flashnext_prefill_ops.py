# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Loader for the SM70 native GDN prefill extension."""

import importlib
from functools import cache


@cache
def has_sm70_flashnext_prefill_ops() -> bool:
    try:
        importlib.import_module("vllm._sm70_flashnext_prefill_C")
    except ImportError:
        return False

    return True
