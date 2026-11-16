# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bound the control wait without pretending to cancel a physical access."""

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future
from typing import TypeVar

T = TypeVar("T")


def remaining(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


def wait_until(callback: Callable[[], T], deadline: float, phase: str) -> T:
    """Wait at most until an absolute monotonic deadline.

    Executor Future wrappers and in-process CUDA fences may ignore timeout.
    The daemon keeps the callback and its resources alive if the control wait
    expires. The caller must quarantine them and must never retry that RPC,
    release its buffers or interpret timeout as a completion receipt.
    """
    if not remaining(deadline):
        raise TimeoutError(f"Host KV shutdown deadline reached before {phase}")
    result: Future[T] = Future()

    def run():
        try:
            result.set_result(callback())
        except BaseException as error:
            result.set_exception(error)

    threading.Thread(target=run, name=f"host-kv-{phase}", daemon=True).start()
    try:
        return result.result(timeout=remaining(deadline))
    except TimeoutError as error:
        raise TimeoutError(
            f"Host KV shutdown deadline reached during {phase}; retaining resources"
        ) from error
