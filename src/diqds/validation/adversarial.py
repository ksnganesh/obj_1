"""Adversarial, non-i.i.d. devices with memory.  Each wraps an honest device and knows the round index.

Deterministic rounds use a hidden string lambda shared by the two devices and known to Eve: both output lambda_i whatever
their inputs.  Then a xor b = 0, so the CHSH game is won iff x (y - 2) = 0: with probability 3/4 (classical), and the
key-round bits are lambda_i, known to Eve, with no extra errors.

HonestThenDeterministic   honest for the first fraction s of the rounds, deterministic afterwards (the stress test of the
                          acceptance test named in the plan): the observed winning frequency drops to ~ s w + (1-s) 3/4.
WindowDevice              deterministic only in a window of W consecutive rounds starting at round t0: long enough to fix
                          one signature's key segment, short enough to hide in the statistical tolerance.  This is the
                          counter-example to signing without privacy amplification (Sec. VI).
FractionDevice            deterministic in each round independently with probability f (entropy concentration on a random
                          subset); used to compare the measured acceptance probability with the binomial prediction.
"""
from __future__ import annotations

from typing import Optional

import numpy as np


class _Deterministic:
    def __init__(self, base, seed: int = 12345):
        self.base = base
        self.P = getattr(base, "P", None)
        self.hidden_rng = np.random.default_rng(seed)
        self.mask: Optional[np.ndarray] = None          # rounds played deterministically (Eve's side information)
        self.hidden: Optional[np.ndarray] = None        # the bits output in those rounds

    def _mask(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def sample(self, x, y, rng, mode: str = "multinomial"):
        a, b = self.base.sample(x, y, rng, mode)
        m = self._mask(x.size, rng)
        lam = self.hidden_rng.integers(0, 2, x.size, dtype=np.uint8)
        a = np.where(m, lam, a).astype(np.uint8)
        b = np.where(m, lam, b).astype(np.uint8)
        self.mask, self.hidden = m, lam
        return a, b


class HonestThenDeterministic(_Deterministic):
    def __init__(self, base, honest_fraction: float = 0.5, seed: int = 12345):
        super().__init__(base, seed)
        self.s = honest_fraction

    def _mask(self, n, rng):
        m = np.zeros(n, bool)
        m[int(self.s * n):] = True
        return m


class WindowDevice(_Deterministic):
    def __init__(self, base, start: int, length: int, seed: int = 12345):
        super().__init__(base, seed)
        self.start, self.length = int(start), int(length)

    def _mask(self, n, rng):
        m = np.zeros(n, bool)
        m[self.start:min(n, self.start + self.length)] = True
        return m


class FractionDevice(_Deterministic):
    def __init__(self, base, fraction: float, seed: int = 12345):
        super().__init__(base, seed)
        self.f = fraction

    def _mask(self, n, rng):
        return self.hidden_rng.random(n) < self.f


def deterministic_win_probability() -> float:
    return 0.75
