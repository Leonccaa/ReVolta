# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Acceptance-driven MTP depth, adapted from 1Cat-vLLM #1570."""

from bisect import bisect_right
from dataclasses import dataclass

MIN_ADAPTIVE_MTP_DEPTH = 2
_DEPTH_THRESHOLDS = (0.62, 0.74, 0.82, 0.89)
_HYSTERESIS = 0.025
_EMA_ALPHA = 0.28


@dataclass
class AdaptiveMTPDepth:
    depth: int
    successes: float = 4.0
    trials: float = 5.0

    def observe(self, drafted: int, accepted: int, cap: int) -> None:
        # A round observes the accepted prefix and, when present, one rejection.
        # Positions after the first rejection supply no evidence.
        trials = min(accepted + 1, drafted)
        self.successes = (1 - _EMA_ALPHA) * self.successes + _EMA_ALPHA * accepted
        self.trials = (1 - _EMA_ALPHA) * self.trials + _EMA_ALPHA * trials
        if accepted == drafted:
            self.depth = min(cap, self.depth + 1)
            return

        p = self.successes / self.trials
        down = MIN_ADAPTIVE_MTP_DEPTH + bisect_right(_DEPTH_THRESHOLDS, p + _HYSTERESIS)
        up = MIN_ADAPTIVE_MTP_DEPTH + bisect_right(_DEPTH_THRESHOLDS, p - _HYSTERESIS)
        if self.depth > down:
            self.depth -= 1
        elif self.depth < min(cap, up):
            self.depth += 1
