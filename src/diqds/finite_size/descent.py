"""Descent below a threshold on the warm-started key-length curve (shared by the Renyi and EAT optimisers).

Near a threshold the optimised key length is, to a good approximation, a straight line in the number of rounds n plus a
sawtooth of a few bits from the integer acceptance counts, provided each optimisation starts from the optimum at a
neighbouring n.  Brent's method on a cold-started or cache-started optimiser can converge on a point where a less
successful optimisation just below the root made the key length look smaller than it is; the key length at the
returned n then exceeds the target by much more than the sawtooth.  ``descend`` starts from the optimum at that n and
walks down the warm-started curve to the smallest integer n that still reaches the target.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Optional, Tuple

# (n, hint) -> (key length, hint for the next evaluation, result object)
Evaluator = Callable[[float, Any], Tuple[float, Any, Any]]

POLISH_BITS = 5.0      # thresholds whose key length exceeds the target by more than this are refined


def descend(ev: Evaluator, n0: float, ell0: float, hint0: Any, target: float, rel_step: float = 1e-3,
            max_evals: int = 40) -> Optional[Tuple[float, float, Any, Any]]:
    """Smallest integer n < n0 whose warm-started key length reaches ``target``.

    A slope from one point ``rel_step`` below n0, secant steps down until a point misses the target, then integer
    bisection between the last point that misses and the nearest point that reaches it.  Every evaluation starts from
    the optimum at the nearest point above it that reaches the target.  Returns (n, ell, hint, result) for the smallest
    such point, or None if no point below n0 reaches the target (or the slope is not positive)."""
    seen: dict = {}

    def f(n: float, hint: Any):
        n = float(n)
        if n not in seen:
            seen[n] = ev(n, hint)
        return seen[n]

    hi = (float(n0), float(ell0), hint0, None)
    n1 = float(math.floor(n0 * (1.0 - rel_step)))
    if n1 >= n0:
        return None
    e = f(n1, hint0)
    slope = (ell0 - e[0]) / (n0 - n1)
    if not slope > 0.0:
        return None
    lo: Optional[float] = None
    if e[0] >= target:
        hi = (n1, e[0], e[1], e[2])
    else:
        lo = n1
    while lo is None and len(seen) < max_evals:          # step down past the crossing
        cand = float(math.floor(hi[0] - max(1.0, 1.5 * (hi[1] - target) / slope)))
        if cand < 1.0:
            break
        e = f(cand, hi[2])
        if e[0] >= target:
            hi = (cand, e[0], e[1], e[2])
        else:
            lo = cand
    if lo is not None:
        while hi[0] - lo > 1.0 and len(seen) < max_evals:  # integer bisection
            mid = float(math.floor(0.5 * (lo + hi[0])))
            e = f(mid, hi[2])
            if e[0] >= target:
                hi = (mid, e[0], e[1], e[2])
            else:
                lo = mid
    return hi if hi[3] is not None else None
