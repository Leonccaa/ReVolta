# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared teardown budget and observable, exception-isolated cleanup stages."""

import os
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from vllm.logger import init_logger

logger = init_logger(__name__)

# Leave room before the executor's first 4 s wait expires. The hard guard is
# restricted to terminal worker processes; native calls cannot be cancelled.
SHUTDOWN_TIMEOUT_S = 3.0
PROCESS_SHUTDOWN_TIMEOUT_S = 3.5
_deadline: ContextVar[float | None] = ContextVar("shutdown_deadline", default=None)


class ShutdownError(RuntimeError):
    """Cleanup failed; successful stages must not hide failed releases."""


class ShutdownIncompleteError(ShutdownError):
    """Quiescence or release completion is unknown. Stop orderly finalization."""


@contextmanager
def shutdown_deadline() -> Iterator[None]:
    if _deadline.get() is not None:
        yield
        return
    token = _deadline.set(time.monotonic() + SHUTDOWN_TIMEOUT_S)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining_shutdown_time() -> float:
    deadline = _deadline.get()
    return (
        SHUTDOWN_TIMEOUT_S
        if deadline is None
        else max(0.0, deadline - time.monotonic())
    )


@contextmanager
def process_shutdown_guard() -> Iterator[None]:
    """Isolate a terminal worker stuck in native cleanup, with failure status.

    Do not run Python/native destructors against an unretired CUDA context.
    The parent still owns process auditing and IPC recovery. This fallback is
    a failed shutdown, never evidence of a successful resource release.
    """

    def expire() -> None:
        # Inherited stderr may share its open file description with the parent.
        # Neither a blocking write nor changing its flags is safe here. The
        # parent observes the failure exit and owns shutdown diagnostics.
        os._exit(1)

    guard = threading.Timer(PROCESS_SHUTDOWN_TIMEOUT_S, expire)
    guard.daemon = True
    guard.start()
    try:
        yield
    finally:
        guard.cancel()


class Cleanup:
    def __init__(self) -> None:
        # Store text, not exception tracebacks that can keep graph owners alive.
        self.errors: list[str] = []

    def run(self, stage: str, action: Callable[[], object]) -> bool:
        if remaining_shutdown_time() <= 0:
            raise ShutdownIncompleteError(f"Shutdown deadline exhausted before {stage}")
        logger.info(
            "Shutdown stage=%s state=begin monotonic=%.6f", stage, time.monotonic()
        )
        try:
            action()
        except ShutdownIncompleteError:
            logger.exception("Shutdown stage=%s state=incomplete", stage)
            raise
        except Exception as error:
            self.errors.append(f"{stage}: {type(error).__name__}: {error}")
            logger.exception("Shutdown stage=%s state=failed", stage)
            return False
        if remaining_shutdown_time() <= 0:
            raise ShutdownIncompleteError(f"Shutdown deadline exhausted after {stage}")
        logger.info(
            "Shutdown stage=%s state=done monotonic=%.6f", stage, time.monotonic()
        )
        return True

    def finish(self) -> None:
        if self.errors:
            raise ShutdownError("; ".join(self.errors))
