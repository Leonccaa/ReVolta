# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-driven dynamic verification width contraction for DFlash2.

Adapts the zero-sync geometric MLE + hysteresis estimator from 1Cat-vLLM #1570
to DFlash2 (draft count k in [3, 7], i.e., verification width q in [4, 8]),
with concurrency-aware contraction into the SM70 AWQ grouped kernel 40/48-row
sweet spot at higher batch sizes (e.g. C8 q=5/6 -> 40/48 rows vs q=8 -> 64 rows).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from statistics import fmean

MIN_ADAPTIVE_DFLASH2_DRAFTS = 3  # q = 4 (1 anchor + 3 drafts)
MAX_ADAPTIVE_DFLASH2_DRAFTS = 7  # q = 8 (1 anchor + 7 drafts)
MAX_ADAPTIVE_DFLASH2_BATCH = 8
AWQ_GROUPED_SWEET_SPOT_ROWS = 48

# 4 boundaries map p_hat into 5 draft counts: {3, 4, 5, 6, 7} (q in {4..8}).
_DFLASH2_THRESHOLDS = (0.58, 0.70, 0.79, 0.87)
_HYSTERESIS = 0.025
_EMA_ALPHA = 0.28
_HIGH_ACCEPT_OVERRIDE = 0.85


@dataclass
class AdaptiveDFlash2Depth:
    """Per-request geometric MLE acceptance tracker for DFlash2."""

    depth: int
    successes: float = 5.0
    trials: float = 7.0

    @property
    def accept_rate(self) -> float:
        return self.successes / self.trials if self.trials > 0.0 else 0.0

    def observe(self, drafted: int, accepted: int, cap: int) -> None:
        """Update EMA acceptance rate from one verification round.

        A round observes the accepted prefix and, when rejected, one failure.
        Positions after the first rejection supply no evidence.
        """
        if drafted <= 0:
            return
        trials = min(accepted + 1, drafted)
        self.successes = (1.0 - _EMA_ALPHA) * self.successes + _EMA_ALPHA * accepted
        self.trials = (1.0 - _EMA_ALPHA) * self.trials + _EMA_ALPHA * trials
        effective_cap = min(cap, MAX_ADAPTIVE_DFLASH2_DRAFTS)
        if accepted >= drafted:
            self.depth = min(effective_cap, self.depth + 1)
            return

        p = self.accept_rate
        down = MIN_ADAPTIVE_DFLASH2_DRAFTS + bisect_right(
            _DFLASH2_THRESHOLDS, p + _HYSTERESIS
        )
        up = MIN_ADAPTIVE_DFLASH2_DRAFTS + bisect_right(
            _DFLASH2_THRESHOLDS, p - _HYSTERESIS
        )
        if self.depth > down:
            self.depth = max(MIN_ADAPTIVE_DFLASH2_DRAFTS, self.depth - 1)
        elif self.depth < min(effective_cap, up):
            self.depth = min(effective_cap, self.depth + 1)


def choose_batch_dflash2_drafts(
    batch_size: int,
    base_cap: int,
    tracked_depths: Iterable[AdaptiveDFlash2Depth],
    *,
    max_grouped_rows: int = AWQ_GROUPED_SWEET_SPOT_ROWS,
    high_accept_override: float = _HIGH_ACCEPT_OVERRIDE,
) -> int:
    """Choose the verification draft count k for a DFlash2 batch.

    - For C1 (batch_size == 1): uses the request's tracked depth directly.
    - For C2+ (batch_size >= 2): uses the batch's rounded mean depth and, when
      batch_size * (width) would exceed ``max_grouped_rows`` (e.g. 64 > 48 at
      C8 q=8), contracts into the single-launch grouped sweet spot unless the
      batch mean acceptance rate reaches ``high_accept_override``, in which
      case the batch retains its full allowed width.
    """
    cap = min(base_cap, MAX_ADAPTIVE_DFLASH2_DRAFTS)
    if cap <= MIN_ADAPTIVE_DFLASH2_DRAFTS or not (
        1 <= batch_size <= MAX_ADAPTIVE_DFLASH2_BATCH
    ):
        return base_cap
    depths = list(tracked_depths)
    if len(depths) != batch_size:
        return cap
    if batch_size == 1:
        return min(cap, depths[0].depth)
    if fmean(d.accept_rate for d in depths) >= high_accept_override:
        return cap

    if max_grouped_rows > 0:
        grouped_cap = max_grouped_rows // batch_size - 1
        if grouped_cap < MIN_ADAPTIVE_DFLASH2_DRAFTS:
            return cap
        cap = min(cap, grouped_cap)

    avg_depth = round(sum(d.depth for d in depths) / batch_size)
    return max(MIN_ADAPTIVE_DFLASH2_DRAFTS, min(cap, avg_depth))


def limit_dflash2_drafts(planned: int, selected: int | None) -> int:
    """Apply contraction to the trained block after lookup chooses a width."""
    if selected is None:
        return planned
    if selected > MAX_ADAPTIVE_DFLASH2_DRAFTS:
        return selected
    return min(planned, selected)
