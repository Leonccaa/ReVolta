# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Retain an in-process CUDA owner whose physical quiescence is unproven."""

from typing import Any

_retained: dict[int, Any] = {}
_closing: dict[int, Any] = {}
_reason: str | None = None


def quarantine(reason: str, *objects: Any) -> None:
    global _reason
    _reason = reason
    for value in objects:
        if value is not None:
            _retained[id(value)] = value


def isolated(value: Any) -> bool:
    return id(value) in _retained or id(value) in _closing


def require_available_context() -> None:
    if _reason is not None:
        raise RuntimeError(
            f"Lookup statistics quarantine requires process restart: {_reason}"
        )


def require_reusable_context() -> None:
    require_available_context()
    if _closing:
        raise RuntimeError("Lookup statistics shutdown is in progress")


def context_isolated() -> bool:
    return _reason is not None or bool(_closing)


def _hold_closing(*objects: Any) -> None:
    for value in objects:
        if value is not None:
            _closing[id(value)] = value


def _release_closing(*objects: Any) -> None:
    # Only the shutdown controller calls this after a positive CPU quiescence
    # proof. Quarantined owners remain in _retained for the process lifetime.
    for value in objects:
        _closing.pop(id(value), None)
