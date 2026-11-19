"""Shared constants, the QDS security budget (Table I of Fan et al.) and honest-device statistics.

Fan, Bian, Li, Wen, Yin, Chen, "Device-independent quantum digital signatures under realistic conditions",
Phys. Rev. A 114, 032617 (2026): the DI-QDS protocol and security budget analysed here.

    Eq. (4)   r_sig = ell / (3 n_s)                       one signature costs 3 n_s key bits per link
    Eq. (5)   eps_for = 2^(1 - n_s) n_m + eps_sec         -> n_s = 46 for eps = 1e-10, n_m = 1000, eps_sec = 5e-11
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

SQRT2 = math.sqrt(2.0)
LN2 = math.log(2.0)
W_CLASSICAL = 0.75                       # classical CHSH winning probability
W_MAX = (2.0 + SQRT2) / 4.0              # Tsirelson, 0.853553...
W_MIN = (2.0 - SQRT2) / 4.0              # smallest quantum winning probability


def h2(x):
    """Binary entropy in bits; vectorised, h2(0) = h2(1) = 0."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -x * np.log2(x) - (1.0 - x) * np.log2(1.0 - x)
    out = np.where((x <= 0.0) | (x >= 1.0), 0.0, out)
    return float(out) if out.ndim == 0 else out


@dataclass(frozen=True)
class QDSBudget:
    """Table I of Fan et al., plus the derived signature length (Eq. 5)."""
    eps: float = 1.0e-10                 # total QDS security parameter
    eps_snd: float = 5.0e-11             # soundness of each DI key-distribution link (enters Eq. 5 as eps_sec)
    n_m: int = 1000                      # document length in bits
    eps_corr: float = 2.0 ** -61         # error-verification hash
    eps_prime: float = 1.0e-11           # classical-message error probability (authenticated channels)

    @property
    def n_s(self) -> int:
        """Smallest n_s with 2^(1-n_s) n_m + eps_snd <= eps (Eq. 5)."""
        slack = self.eps - self.eps_snd
        ns = max(1, math.ceil(1.0 + math.log2(self.n_m / slack)))
        while ns > 1 and 2.0 ** (1 - (ns - 1)) * self.n_m + self.eps_snd <= self.eps:
            ns -= 1
        return ns

    @property
    def key_bits_per_signature(self) -> int:
        return 3 * self.n_s

    def eps_forgery(self, n_s: int | None = None) -> float:
        ns = self.n_s if n_s is None else n_s
        return 2.0 ** (1 - ns) * self.n_m + self.eps_snd


TABLE_I = QDSBudget()
N_S = TABLE_I.n_s                                    # 46
KEY_BITS_PER_SIGNATURE = TABLE_I.key_bits_per_signature   # 138


def chsh_to_w(S: float) -> float:
    return (4.0 + S) / 8.0


def w_to_chsh(w: float) -> float:
    return 8.0 * w - 4.0


@dataclass(frozen=True)
class Honest:
    """Statistics of the honest devices: CHSH winning probability w and key-round bit error rate Q.

    Qx is the error rate of a key generated in Alice's second CHSH setting (needed only by protocols that key on both
    settings).  None means the state-noise inference ``Q_x_state``; an explicit value overrides it (e.g. Qx = Q when
    the reported QBER was not measured in one of Alice's CHSH settings, or for the isotropic sensitivity case)."""
    w: float
    Q: float
    label: str = ""
    Qx: Optional[float] = None

    @property
    def S(self) -> float:
        return w_to_chsh(self.w)

    @property
    def Q_x_state(self) -> float:
        """With Alice's settings Z, X and Bob's CHSH settings (Z +- X)/sqrt2, S = sqrt2 (E_ZZ + E_XX) for ANY state, so a
        state with ideal measurement directions has Q_x = 1 - S/(2 sqrt2) - Q.  This is the largest X-basis error
        consistent with (S, Q) (misaligned CHSH settings only lower S); it equals Q for Werner states and exceeds Q for
        dephasing-dominated memories, as in the QuTiP/Aer device models (randomness/qutip_states.py)."""
        return float(min(max(1.0 - self.S / (2.0 * SQRT2) - self.Q, 0.0), 0.5))

    @property
    def Q_x(self) -> float:
        return self.Q_x_state if self.Qx is None else float(self.Qx)

    def isotropic(self) -> "Honest":
        """Same statistics with equal error rates in both key bases (Q_x = Q): the sensitivity case."""
        return replace(self, Qx=self.Q, label=(self.label + " [Q_x = Q]").strip())

    @classmethod
    def werner(cls, V: float) -> "Honest":
        """Werner state of visibility V with ideal CHSH settings: S = 2 sqrt2 V, Q = (1 - V)/2 (Fan et al., Fig. 3)."""
        return cls(w=(2.0 + SQRT2 * V) / 4.0, Q=(1.0 - V) / 2.0, label=f"Werner V={V:g}")

    @classmethod
    def from_chsh(cls, S: float, Q: float, label: str = "") -> "Honest":
        return cls(w=chsh_to_w(S), Q=Q, label=label)
