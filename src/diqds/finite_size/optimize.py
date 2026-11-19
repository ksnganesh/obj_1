"""Optimisers for the Renyi key length and the thresholds derived from it.

renyi_optimum   maximise the key length (or another objective) over (gamma, alpha) and, for q = "opt", the noisy-
                preprocessing flip probability q.  A coarse grid over (log gamma, log(alpha-1)) seeds multi-start
                Nelder-Mead; the q = "opt" result is never worse than the q = 0 optimum (monotonicity check), and the
                returned key length is evaluated with the certified lower bound on h_alpha (renyi.h_alpha_certified).
threshold       rounds to the first signature: the smallest integer n whose optimised, certified key length reaches the
                target.  Bracketing and Brent's method in log n (relative tolerance 1e-5) run on the fast grid value
                of h_alpha; if the key length at the root exceeds the target by more than the integer-count sawtooth
                (POLISH_BITS), a warm-started descent (``polish``) finds the smallest n below it that still reaches the
                target; the certified key length (never above the grid value, and within cert_tol_bits of it) is then
                evaluated and n is stepped up until it reaches the target.  The result carries the key length AT the
                reported n, so every quoted threshold can be checked.
polish          the warm-started descent on its own (also used to refine stored thresholds).
n_min, v_min    thin wrappers (rounds for a given device, visibility for a given block size).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Callable, Optional, Sequence, Union

import numpy as np
from scipy.optimize import brentq, minimize

from ..common import KEY_BITS_PER_SIGNATURE, Honest
from .descent import POLISH_BITS, descend
from .protocols import Protocol
from .renyi import RenyiResult, RenyiSettings, key_length

QSpec = Union[float, str]          # a fixed flip probability, or "opt"
GAMMA_MAX = 0.6
LOG_AM1 = (-8.0, -0.3)


def _params(x, q_fixed: Optional[float], gamma_fixed: Optional[float]):
    i = 0
    if gamma_fixed is None:
        gamma = 10.0 ** float(np.clip(x[0], -6.0, math.log10(GAMMA_MAX)))
        i = 1
    else:
        gamma = gamma_fixed
    alpha = 1.0 + 10.0 ** float(np.clip(x[i], *LOG_AM1))
    q = q_fixed if q_fixed is not None else float(np.clip(x[i + 1], 0.0, 0.45))
    return gamma, alpha, q


def _vec(gamma, alpha, q, q_fixed, gamma_fixed):
    v = [] if gamma_fixed is not None else [math.log10(gamma)]
    v.append(math.log10(alpha - 1.0))
    if q_fixed is None:
        v.append(q)
    return np.array(v)


def _grid_seeds(n, f, q_fixed, gamma_fixed, k: int = 3):
    """Best k points of a coarse grid over (log gamma, log(alpha-1)) [and q]."""
    lg0, la0 = math.log10(min(0.3, 50.0 / math.sqrt(n))), math.log10(0.8 / math.sqrt(n))
    lgs = [None] if gamma_fixed is not None else [lg0 + d for d in (-1.0, -0.5, 0.0, 0.5)]
    las = [la0 + d for d in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    qs = [None] if q_fixed is not None else [0.0, 0.03, 0.08, 0.15]
    pts = []
    for lg in lgs:
        for la in las:
            for qq in qs:
                x = ([] if lg is None else [min(lg, math.log10(GAMMA_MAX))]) + [min(la, LOG_AM1[1])] + \
                    ([] if qq is None else [qq])
                pts.append((f(np.array(x)), x))
    pts.sort(key=lambda t: t[0])
    return [np.array(x) for _, x in pts[:k]]


def _nelder_mead(f, x0, q_fixed, gamma_fixed, scale_factor: float = 1.0):
    scale = ([] if gamma_fixed is not None else [0.3]) + [0.4] + ([] if q_fixed is not None else [0.04])
    init = np.vstack([x0, x0 + np.diag(np.array(scale) * scale_factor)])
    return minimize(f, x0, method="Nelder-Mead",
                    options={"initial_simplex": init, "xatol": 1e-4, "fatol": 1e-4, "maxiter": 1500, "maxfev": 1500})


def _optimise(n, honest, protocol, q_fixed, gamma_fixed, settings, obj, x_hint, starts):
    def f(x):
        g, a, qq = _params(x, q_fixed, gamma_fixed)
        r = key_length(n, honest, protocol, g, a, qq, settings)
        v = obj(r)
        return -v if np.isfinite(v) else 1e30

    # seeds: the warm-start hint (if any), the n-scaled default point, and (cold starts) the best coarse-grid points
    lg0, la0 = math.log10(min(0.3, 50.0 / math.sqrt(n))), math.log10(0.8 / math.sqrt(n))
    default = np.array(([] if gamma_fixed is not None else [lg0]) + [la0] + ([] if q_fixed is not None else [0.02]))
    seeds = [default]
    if x_hint is not None:
        seeds.insert(0, np.asarray(x_hint, dtype=float))
    if starts > len(seeds):
        seeds += _grid_seeds(n, f, q_fixed, gamma_fixed, k=starts - len(seeds))
    best = None
    for k, s in enumerate(seeds):
        sc = 0.25 if (x_hint is not None and k == 0) else 1.0      # a warm start explores less
        res = _nelder_mead(f, s, q_fixed, gamma_fixed, sc)
        res2 = _nelder_mead(f, res.x, q_fixed, gamma_fixed, 0.1 * sc)   # restart with a small simplex (NM can stall)
        if res2.fun < res.fun:
            res = res2
        if best is None or res.fun < best.fun:
            best = res
    return best


def renyi_optimum(n: float, honest: Honest, protocol: Protocol, q: QSpec = 0.0,
                  settings: RenyiSettings = RenyiSettings(),
                  objective: Optional[Callable[[RenyiResult], float]] = None, starts: int = 3,
                  gamma: Optional[float] = None, x_hint: Optional[Sequence[float]] = None,
                  certify: Optional[bool] = None) -> RenyiResult:
    """Maximise ``objective(result)`` (default: the key length) over gamma (unless fixed), alpha and (if q == "opt") q.

    For q == "opt" the q = 0 optimum is computed as well and the better of the two is returned, so the optimised key
    length can never fall below the one without preprocessing.  The returned result is re-evaluated with the certified
    lower bound on h_alpha (``settings.certify``); ``result.info["x"]`` is the optimiser's point (for warm starts)."""
    q_fixed = None if q == "opt" else float(q)
    obj = objective or (lambda r: r.ell)
    best = _optimise(n, honest, protocol, q_fixed, gamma, settings, obj, x_hint, starts)
    g, a, qq = _params(best.x, q_fixed, gamma)
    if q_fixed is None:                                  # monotonicity: never below the q = 0 optimum
        hint0 = list(best.x)[:-1]
        zero = _optimise(n, honest, protocol, 0.0, gamma, settings, obj, hint0, 1)
        if zero.fun <= best.fun:
            g, a, qq = _params(zero.x, 0.0, gamma)
    cert = settings.certify if certify is None else certify
    res = key_length(n, honest, protocol, g, a, qq, settings, certify=cert)
    res.info["x"] = list(_vec(g, a, qq, q_fixed, gamma))
    return res


def optimize_alpha(n, honest, protocol, gamma, q=0.0, settings=RenyiSettings()) -> RenyiResult:
    """Key length at fixed (gamma, q), maximised over alpha (used for the cross-check against Hahn et al.)."""
    return renyi_optimum(n, honest, protocol, q, settings, gamma=gamma)


@dataclass
class Threshold:
    n: float                       # smallest integer n with ell(n) >= target (inf if none below 10**hi)
    ell: float                     # certified key length at n
    result: Optional[RenyiResult]
    target: float
    evaluations: int = 0


def threshold(honest: Honest, protocol: Protocol, q: QSpec = 0.0, target: float = KEY_BITS_PER_SIGNATURE,
              settings: RenyiSettings = RenyiSettings(), lo: float = 3.0, hi: float = 11.5, rtol: float = 1e-5,
              gamma: Optional[float] = None, guess: Optional[float] = None,
              polish_bits: float = POLISH_BITS) -> Threshold:
    """Rounds to the first signature: the smallest integer n whose optimised, certified key length reaches ``target``.

    Brent's method in log10 n on the optimised key length (fast grid value of h_alpha), bracketed around ``guess`` (or
    by bisection on [lo, hi]).  At the upper end b of the final bracket the optimiser's parameters reach the target.
    If the key length there exceeds the target by more than ``polish_bits``, ``polish`` walks down the warm-started
    key-length curve to the smallest n that still reaches it.  The parameters are then re-evaluated with the certified
    h_alpha (and n is stepped up by single rounds if the certified value, which is at most cert_tol_bits lower, falls
    short).  Every evaluation uses an integer number of rounds.  Because the acceptance thresholds are integer counts,
    the key length is a sawtooth in n of a few bits; thresholds are therefore reproducible to about 1e-4 of n
    (``jitter`` measures this)."""
    cache = {}

    def ell(ln: float) -> RenyiResult:
        key = round(ln, 12)
        if key not in cache:
            ok = [k for k in cache if cache[k].ell > -0.5 * 10.0 ** k]       # skip degenerate optima far below
            near = min(ok, key=lambda k: abs(k - key)) if ok else None
            x0 = cache[near].info["x"] if near is not None else None
            cache[key] = renyi_optimum(float(round(10.0 ** ln)), honest, protocol, q, settings, gamma=gamma, x_hint=x0,
                                       starts=3 if x0 is None else 1, certify=False)
        return cache[key]

    g = lambda ln: ell(ln).ell - target
    if guess is not None and math.isfinite(guess):
        bot = top = math.log10(guess)
        step = 0.02
        while g(top) < 0:
            bot, top = top, min(top + step, hi)
            step *= 2
            if top >= hi and g(hi) < 0:
                return Threshold(math.inf, math.nan, None, target, len(cache))
        if bot == top:
            step = 0.02
            while g(bot) >= 0:
                top, bot = bot, max(bot - step, lo)
                step *= 2
                if bot <= lo and g(lo) >= 0:
                    return Threshold(10.0 ** lo, ell(lo).ell, ell(lo), target, len(cache))
    else:
        if g(hi) < 0:
            return Threshold(math.inf, math.nan, None, target, len(cache))
        if g(lo) >= 0:
            return Threshold(10.0 ** lo, ell(lo).ell, ell(lo), target, len(cache))
        bot, top = lo, hi
        while top - bot > 0.25:
            mid = 0.5 * (bot + top)
            if g(mid) >= 0:
                top = mid
            else:
                bot = mid
    xtol = rtol / math.log(10.0)
    root = brentq(g, bot, top, xtol=xtol, rtol=4 * np.finfo(float).eps, maxiter=100)
    # the cached evaluation just above the root that reaches the target
    above = sorted(k for k in cache if k >= root - 1e-12 and cache[k].ell >= target)
    b = above[0] if above else top
    base = cache[round(b, 12)]
    evals = len(cache)
    if base.ell - target > polish_bits:
        pol = _descend_renyi(honest, protocol, q, target, settings, gamma, base)
        if pol is not None:
            base, k = pol
            evals += k
    return _certify_at(base, honest, protocol, q, target, settings, gamma, evals)


def _certify_at(base: RenyiResult, honest, protocol, q, target, settings, gamma, evals: int) -> Threshold:
    """Certified key length with the parameters of ``base``; n stepped up by single rounds if it falls short."""
    n = base.n                                           # an integer: every evaluation uses an integer number of rounds
    r = key_length(n, honest, protocol, base.gamma, base.alpha, base.q, settings, certify=settings.certify)
    steps = 0
    while r.ell < target and steps < 50:                 # only if the certified value is the < 0.01 bit lower
        n += 1.0
        r = renyi_optimum(n, honest, protocol, q, settings, gamma=gamma, x_hint=base.info["x"], starts=1)
        steps += 1
    r.info["x"] = base.info["x"]
    return Threshold(n, r.ell, r, target, evals + steps + 1)


def _descend_renyi(honest, protocol, q, target, settings, gamma, base: RenyiResult):
    """Warm-started descent below base.n; returns (result at the smallest n reaching the target, evaluations)."""
    count = [0]

    def ev(n, hint):
        count[0] += 1
        r = renyi_optimum(float(n), honest, protocol, q, settings, gamma=gamma, x_hint=hint, starts=1, certify=False)
        return r.ell, r.info["x"], r
    found = descend(ev, base.n, base.ell, base.info["x"], target)
    return None if found is None else (found[3], count[0])


def polish(honest: Honest, protocol: Protocol, n0: float, q: QSpec = 0.0, target: float = KEY_BITS_PER_SIGNATURE,
           settings: RenyiSettings = RenyiSettings(), gamma: Optional[float] = None,
           x0: Optional[Sequence[float]] = None, polish_bits: float = POLISH_BITS) -> Optional[Threshold]:
    """Refine a threshold n0 whose key length exceeds the target by more than ``polish_bits``.

    The optimum at n0 (warm-started from ``x0`` if given) is the starting point of the descent; returns the refined
    Threshold, or None if the key length at n0 is within ``polish_bits`` of the target or no smaller n reaches it."""
    base = renyi_optimum(float(n0), honest, protocol, q, settings, gamma=gamma, x_hint=x0,
                         starts=1 if x0 is not None else 3, certify=False)
    if not base.ell - target > polish_bits:
        return None
    pol = _descend_renyi(honest, protocol, q, target, settings, gamma, base)
    if pol is None:
        return None
    th = _certify_at(pol[0], honest, protocol, q, target, settings, gamma, pol[1] + 1)
    return th if th.n < n0 else None


def jitter(honest: Honest, protocol: Protocol, n0: float, q: QSpec = 0.0, settings: RenyiSettings = RenyiSettings(),
           span: float = 2e-3, points: int = 21) -> dict:
    """Optimised key length on a fine grid of integer n around n0: slope and residual sawtooth (for the article)."""
    ns = np.unique(np.round(np.linspace(n0 * (1 - span), n0 * (1 + span), points)))
    hint, ells = None, []
    for n in ns:
        r = renyi_optimum(float(n), honest, protocol, q, settings, x_hint=hint, starts=3 if hint is None else 1,
                          certify=False)
        hint = r.info["x"]
        ells.append(r.ell)
    ells = np.array(ells)
    slope, icpt = np.polyfit(ns, ells, 1)
    resid = ells - (slope * ns + icpt)
    return {"n": ns.tolist(), "ell": ells.tolist(), "slope_bits_per_round": float(slope),
            "max_abs_residual_bits": float(np.max(np.abs(resid))), "rms_residual_bits": float(np.sqrt(np.mean(resid ** 2))),
            "relative_threshold_jitter": float(np.max(np.abs(resid)) / slope / n0)}


def n_min(honest: Honest, protocol: Protocol, q: QSpec = 0.0, target: float = KEY_BITS_PER_SIGNATURE,
          settings: RenyiSettings = RenyiSettings(), gamma: Optional[float] = None, **kw) -> float:
    """Smallest number of rounds whose optimised key length reaches ``target`` bits (see ``threshold``)."""
    return threshold(honest, protocol, q, target, settings, gamma=gamma, **kw).n


def v_min(n: float, protocol: Protocol, q: QSpec = 0.0, target: float = KEY_BITS_PER_SIGNATURE,
          settings: RenyiSettings = RenyiSettings(), lo: float = 0.80, hi: float = 0.999, xtol: float = 1e-6) -> float:
    """Smallest Werner visibility whose optimised key length at n rounds reaches ``target`` bits (Brent's method)."""
    hint = [None]

    def g(V, cert=False):
        r = renyi_optimum(n, Honest.werner(V), protocol, q, settings, x_hint=hint[0],
                          starts=3 if hint[0] is None else 1, certify=cert)
        hint[0] = r.info["x"]
        return r.ell - target
    if g(hi) < 0:
        return math.nan
    a = hi
    while a - 0.01 >= lo and g(a - 0.01) >= 0:
        a -= 0.01
    V = brentq(g, max(a - 0.01, lo), a, xtol=xtol)
    while g(V, cert=settings.certify) < 0:              # return a visibility whose certified key reaches the target
        V += xtol
    return V


def monotone_in_q(n: float, honest: Honest, protocol: Protocol, settings: RenyiSettings = RenyiSettings()) -> bool:
    """Check used by the tests: the q-optimised key length is at least the q = 0 one."""
    a = renyi_optimum(n, honest, protocol, "opt", settings)
    b = renyi_optimum(n, honest, protocol, 0.0, settings)
    return a.ell >= b.ell - 1e-6


__all__ = ["renyi_optimum", "optimize_alpha", "threshold", "Threshold", "n_min", "v_min", "jitter", "monotone_in_q", "replace"]
