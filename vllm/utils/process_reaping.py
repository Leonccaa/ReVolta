# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-way terminal retirement receipts, distinct from process exit.

A receipt only grants bounded time for finalizers and kernel thread-group exit.
Only is_alive()/join() establish terminal status. EOF is never a receipt.
"""

import os
import struct
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from multiprocessing.connection import Connection
from multiprocessing.process import BaseProcess

_RECEIPT = struct.Struct("!Bd")


def notify_process_retired(
    writer: Connection | None, *, isolated: bool = False
) -> None:
    """Send one atomic, nonblocking receipt; safe even in the hard guard.

    This dedicated pipe carries only small terminal receipts before exit.
    A lost/broken pipe falls back to the parent's ordinary escalation policy.
    Do not log or format exceptions on this terminal path.
    """
    if writer is None:
        return
    try:
        os.set_blocking(writer.fileno(), False)
        os.write(writer.fileno(), _RECEIPT.pack(2 if isolated else 1, time.monotonic()))
    except (OSError, ValueError):
        pass


@dataclass
class ProcessRetirement:
    reader: Connection
    retired_at: float | None = None
    isolated: bool = False
    closed: bool = False

    def observe(self) -> bool:
        if self.retired_at is not None:
            return True
        if self.closed:
            return False
        try:
            if not self.reader.poll():
                return False
            data = os.read(self.reader.fileno(), _RECEIPT.size)
            if len(data) == _RECEIPT.size:
                status, stamp = _RECEIPT.unpack(data)
                if status in (1, 2) and 0 <= stamp <= time.monotonic():
                    self.retired_at = stamp
                    self.isolated = status == 2
        except (OSError, ValueError):
            pass
        self.close()
        return self.retired_at is not None

    def close(self) -> None:
        if not self.closed:
            self.reader.close()
            self.closed = True


def reap_processes(
    procs: list[BaseProcess],
    retirements: Mapping[BaseProcess, ProcessRetirement] | None = None,
    *,
    reap_timeout: float = 0,
    graceful_timeout: float = 4,
    started_at: float | None = None,
    terminate_timeout: float | None = 4,
    defer_retired: bool = False,
    kill: Callable[[BaseProcess], None] | None = None,
) -> list[BaseProcess]:
    """Wait for exit, protecting receipts until their absolute reap deadline.

    Unreported processes retain the shared graceful + TERM budgets. A retired
    process gets at most reap_timeout from its receipt, then the same TERM
    grace. Neither another rank nor a second call resets a receipt's deadline.
    Terminal callers may defer successful retirees until persistence finishes.
    Return processes still alive after escalation; a signal is not a terminal
    observation and cannot authorize further resource cleanup.
    """
    retirements = retirements or {}
    started = time.monotonic() if started_at is None else started_at
    terminated: dict[int, float] = {}
    pending = list(procs)
    unreaped: list[BaseProcess] = []
    while pending:
        now = time.monotonic()
        for proc in pending[:]:
            receipt = retirements.get(proc) if retirements else None
            if not proc.is_alive():
                if receipt is not None:
                    receipt.close()
                pending.remove(proc)
                continue
            retired = receipt is not None and receipt.observe()
            # Failure isolation grants exit time, not IO quiescence. Only a
            # completed shutdown can let persistence proceed before reaping.
            if retired and defer_retired:
                assert receipt is not None
                if not receipt.isolated:
                    pending.remove(proc)
                    continue
            deadline = started + graceful_timeout
            if retired:
                assert receipt is not None and receipt.retired_at is not None
                deadline = max(deadline, receipt.retired_at + reap_timeout)
            if now < deadline:
                continue
            if terminate_timeout is None:
                assert kill is not None
                kill(proc)
                if proc.is_alive():
                    unreaped.append(proc)
                if receipt is not None:
                    receipt.close()
                pending.remove(proc)
            elif id(proc) not in terminated:
                proc.terminate()
                terminated[id(proc)] = now
            elif now >= terminated[id(proc)] + terminate_timeout:
                if kill is None:
                    proc.kill()
                else:
                    kill(proc)
                if proc.is_alive():
                    unreaped.append(proc)
                if receipt is not None:
                    receipt.close()
                pending.remove(proc)
        if pending:
            time.sleep(0.05)
    return unreaped


def process_reap_timeout(config: object) -> float:
    limit = float(getattr(config, "shutdown_timeout", 0))
    requested = getattr(config, "shutdown_reap_timeout", None)
    return max(0.0, min(limit, limit if requested is None else float(requested)))
