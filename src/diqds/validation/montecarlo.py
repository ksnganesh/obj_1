"""Monte Carlo statistics of the emulated protocol against the analytic completeness and correctness claims.

Acceptance test.  For i.i.d. devices the counts (lost, won, generation) are multinomial(n; gamma(1-w), gamma w, 1-gamma).
Given t = n - K_perp test rounds, the wins are Bin(t, w) and the test passes iff
    K_perp >= k_lo_perp  and  wins >= max(k_lo_win, t - k_up_loss),
so the exact acceptance probability is a one-dimensional sum over t; the union bound sum_i tail_i is what the bound
promises.  A device that plays deterministically in a fraction f of the rounds has w -> (1-f) w + f 3/4 (and fewer
key-round errors), so its acceptance probability follows from the same formula.

Correctness.  After Cascade the 64-bit Toeplitz tag (2^-64 collision probability for a random seed) must catch every
residual error; the emulation records residual errors before the tag and key mismatches after it.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
from scipy.stats import binom

from .pipeline import LinkPlan, _inputs, run_qds


def acceptance_probability(plan: LinkPlan, w: float) -> float:
    """Exact P(accept) of the acceptance test for i.i.d. devices with winning probability w."""
    acc, n, g = plan.acceptance, plan.n, plan.gamma
    mu, sd = g * n, math.sqrt(n * g * (1 - g))
    t = np.arange(max(0, int(mu - 12 * sd)), min(n, int(mu + 12 * sd)) + 1)
    p_t = binom.pmf(t, n, g)                                    # number of test rounds
    perp_ok = (n - t) >= acc.k_lo[2]
    need = np.maximum(acc.k_lo[1], t - acc.k_up[0])
    p_ok = binom.sf(need - 1, t, w)                              # P(wins >= need)
    return float(np.sum(p_t * perp_ok * p_ok))


def acceptance_mc(plan: LinkPlan, device, trials: int, rng: np.random.Generator, sampling: str = "multinomial") -> float:
    """Fraction of runs whose acceptance test passes (EC assumed successful: Bob uses Alice's true test outputs)."""
    ok = 0
    for _ in range(trials):
        X, Y, T, key = _inputs(plan.protocol, plan.n, plan.gamma, rng)
        a, b = device.sample(X, Y, rng, sampling)
        yp = (Y.astype(np.int16) - 2).clip(0, 1).astype(np.uint8)
        wins = ((a[T] ^ b[T]) == (X[T].astype(np.uint8) & yp[T]))
        k = (int(T.sum() - wins.sum()), int(wins.sum()), int((~T).sum()))
        acc = plan.acceptance
        ok += int(all(acc.k_lo[i] <= k[i] <= acc.k_up[i] for i in range(3)))
    return ok / trials


@dataclass
class MCSummary:
    trials: int
    accepted: int = 0
    reasons: Counter = field(default_factory=Counter)
    signatures: List[int] = field(default_factory=list)
    honest_ok: int = 0
    honest_total: int = 0
    forged_accepted: int = 0
    forged_tried: int = 0
    targeted_forged: int = 0
    targeted_total: int = 0
    residual_error_runs: int = 0                    # links where Cascade left errors (must be caught by the tag)
    key_mismatch: int = 0                           # accepted links whose final keys differ (correctness failure)
    leak_fraction: List[float] = field(default_factory=list)
    win_freq: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict:
        return {"trials": self.trials, "accepted": self.accepted, "abort_rate": 1 - self.accepted / max(self.trials, 1),
                "reasons": dict(self.reasons), "mean_signatures": float(np.mean(self.signatures)) if self.signatures else 0.0,
                "honest_ok": self.honest_ok, "honest_total": self.honest_total, "forged_accepted": self.forged_accepted,
                "forged_tried": self.forged_tried, "targeted_forged": self.targeted_forged,
                "targeted_total": self.targeted_total, "residual_error_runs": self.residual_error_runs,
                "key_mismatch": self.key_mismatch,
                "win_freq_mean": float(np.mean(self.win_freq)) if self.win_freq else None,
                "leak_over_budget_max": max(self.leak_fraction) if self.leak_fraction else None,
                "leak_over_budget_mean": float(np.mean(self.leak_fraction)) if self.leak_fraction else None}


def full_mc(plan: LinkPlan, dev_ab, dev_ac, trials: int, rng: np.random.Generator, pa: str = "toeplitz",
            n_docs: int = 6, sampling: str = "multinomial") -> MCSummary:
    s = MCSummary(trials)
    for _ in range(trials):
        r = run_qds(plan, dev_ab, dev_ac, rng, pa, n_docs, sampling=sampling)
        for link in (r.ab, r.ac):
            s.leak_fraction.append(link.leak / link.ec_budget)
            s.residual_error_runs += int(link.residual_errors > 0)
            if link.win_freq:
                s.win_freq.append(link.win_freq)
            if link.accepted:
                s.key_mismatch += int(not np.array_equal(link.key_alice, link.key_bob))
        if r.accepted:
            s.accepted += 1
            s.signatures.append(r.signatures)
            s.honest_ok += r.honest_ok
            s.honest_total += r.forged_tried
            s.forged_accepted += r.forged_accepted
            s.forged_tried += r.forged_tried
        else:
            s.reasons[r.ab.reason or r.ac.reason] += 1
    return s
