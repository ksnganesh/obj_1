"""Baseline: the finite-size key length of Fan et al., Phys. Rev. A 114, 032617 (2026), Eq. (9) = (B12), which is
Theorem 1 of Tan et al., Quantum 6, 880 (2022) with a second-order EAT term.

    ell = n g(w_thr) - n (a ln2/2) V^2 - n a^2 K_alpha - n gamma - n (a'/4) V'^2
          - theta(e_s')/a - theta(e_s'')/a' - (alpha/a + alpha'/a' - 2) log2(1/e_EA)
          - 3 theta(e_s - e_s' - 2 e_s'') - EC_max - ceil(log2(1/e_h)) - 2 log2(1/e_PA) + 2
    a = alpha - 1, a' = alpha' - 1, theta(e) = log2((1 + sqrt(1 - e^2))/e^2),  V' = 2 log2 5
    g = affine tangent of eta(w) = (1 - gamma) s r_p(w) + gamma r_0(w) at t;  beta in [g(0), g(1)] free
    w_thr = w_hon - k sqrt(var / n) / gamma                                                    (B10), k = 3
    EC_max = n [ (1 - gamma) s h(Q') + gamma h(1 - w_hon) ] + 50 sqrt(n),  Q' = p + (1-2p) Q    (B11, s = sift)

Readings of what Fan et al. leave open:
    "loss_only"   beta optimised, margin on the loss count only (var = q_loss (1 - q_loss)).  Close to the printed
                  thresholds of Fan et al.; not covered by Tan's proof.
    "rigorous"    beta optimised, loss AND win counts bounded (var = max of the two), consistent with Tan Eq. (30).
                  DEFAULT for every comparison in the article.
    "strict"      beta = g(1), loss count only (the protocol exactly as written).
Error budget: e_PA + 2 e_s + 2 e_h <= e_snd = 5e-11, e_EA = e_PA = 1e-11, e_h = 2^-61 (61-bit tag), as in Tan et al.

Optimisation: multi-start Nelder-Mead over (gamma, t, beta, alpha, alpha', smoothing split); preprocessing p either
fixed or optimised over a grid with local refinement, the p = 0 optimum always included (so the optimised key length is
never below the p = 0 one).  Thresholds: Brent's method in log n (relative tolerance 1e-5), smallest integer n whose
key length reaches the target, refined by a warm-started descent when the key length at the root exceeds the target by
more than the integer-count sawtooth (descent.POLISH_BITS); ``threshold`` returns the key length at that n.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
from scipy.optimize import brentq, minimize

from ..common import Honest, W_MAX, W_MIN, SQRT2, h2
from .descent import POLISH_BITS, descend
from .entropy import r0, rp

VPRIME = 2.0 * math.log2(5.0)
LN2 = math.log(2.0)


@dataclass(frozen=True)
class EATSettings:
    eps_snd: float = 5.0e-11
    eps_EA: float = 1.0e-11
    eps_PA: float = 1.0e-11
    eps_h: float = 2.0 ** -61
    k_sigma: float = 3.0
    ec_sqrt: float = 50.0
    reading: str = "rigorous"            # "rigorous" | "loss_only" | "strict"
    ec_model: str = "matched"            # "matched": sift factor and p in EC_max; "printed": (B11) as printed

    @property
    def tag_bits(self) -> int:
        return math.ceil(math.log2(1.0 / self.eps_h))

    @property
    def eps_s(self) -> float:
        return (self.eps_snd - self.eps_PA - 2.0 * self.eps_h) / 2.0


def theta(eps: float) -> float:
    return math.log2((1.0 + math.sqrt(1.0 - eps * eps)) / (eps * eps))


def _eta(w, gamma, p, sift):
    return (1.0 - gamma) * sift * float(rp(w, p)) + gamma * float(r0(w))


def _deta(w, gamma, p, sift, dw=1e-7):
    lo, hi = max(w - dw, 0.75 + 1e-12), min(w + dw, W_MAX)
    return (_eta(hi, gamma, p, sift) - _eta(lo, gamma, p, sift)) / (hi - lo)


def w_threshold(n, gamma, w_hon, settings: EATSettings) -> float:
    q_loss = gamma * (1.0 - w_hon)
    var = q_loss * (1.0 - q_loss)
    if settings.reading == "rigorous":
        q_win = gamma * w_hon
        var = max(var, q_win * (1.0 - q_win))
    return w_hon - settings.k_sigma * math.sqrt(var / n) / gamma


def ec_max(n, honest: Honest, gamma, p, sift, settings: EATSettings) -> float:
    if settings.ec_model == "printed":
        return n * ((1.0 - gamma) * h2(honest.Q) + gamma * h2(1.0 - honest.w)) + settings.ec_sqrt * math.sqrt(n)
    flip = lambda Q: p + (1.0 - 2.0 * p) * Q
    hk = 0.5 * (h2(flip(honest.Q)) + h2(flip(honest.Q_x))) if sift < 1.0 else h2(flip(honest.Q))
    return n * ((1.0 - gamma) * sift * hk + gamma * h2(1.0 - honest.w)) + settings.ec_sqrt * math.sqrt(n)


@dataclass
class EATResult:
    ell: float
    n: float
    gamma: float
    t: float
    beta: float
    alpha: float
    alpha_p: float
    eps_s1: float
    eps_s2: float
    p: float
    w_thr: float
    terms: Dict[str, float] = field(default_factory=dict)
    x: list = field(default_factory=list)
    grid: dict = field(default_factory=dict)


def key_length(n, honest: Honest, gamma, t, beta_frac, am1, apm1, fs1, fs2, p=0.0, sift=0.5,
               settings: EATSettings = EATSettings()) -> EATResult:
    bad = EATResult(-math.inf, n, gamma, t, float("nan"), 1 + am1, 1 + apm1, 0, 0, p, float("nan"))
    if not (0 < gamma < 1 and 0.75 < t <= W_MAX and 0 < am1 < 1 and 0 < apm1 < 2 / VPRIME and 0 <= p < 0.5):
        return bad
    if not (fs1 > 0 and fs2 > 0 and fs1 + 2 * fs2 < 1):
        return bad
    eps_s = settings.eps_s
    wt = w_threshold(n, gamma, honest.w, settings)
    if wt <= 0.75:
        return bad
    slope = _deta(t, gamma, p, sift)
    g = lambda w: _eta(t, gamma, p, sift) + (w - t) * slope
    g0, g1 = g(0.0), g(1.0)
    beta = g1 if settings.reading == "strict" else g0 + float(np.clip(beta_frac, 0.0, 1.0)) * (g1 - g0)
    d0, d1 = (beta - g0) ** 2, (beta - g1) ** 2
    var_f = (2 - SQRT2) / (4 * gamma) * min(d0, d1) + (2 + SQRT2) / (4 * gamma) * max(d0, d1)
    Vc = math.sqrt(var_f + 2.0) + math.log2(33.0)
    fmax = g1 / gamma + (1.0 - 1.0 / gamma) * beta
    fmin = g(W_MIN)
    ex = 2.0 * math.log2(4.0) + fmax - fmin
    lnarg = ex * LN2
    lnterm = lnarg + math.log1p(math.exp(2.0 - lnarg)) if lnarg < 700 else lnarg
    logK = am1 * ex * LN2 + 3.0 * math.log(lnterm) - math.log(6.0 * (1.0 - am1) ** 3 * LN2)
    if logK > 600:
        return bad
    K = math.exp(logK)
    e1, e2 = fs1 * eps_s, fs2 * eps_s
    alpha, alphap = 1.0 + am1, 1.0 + apm1
    terms = {
        "n_g": n * g(wt),
        "second_order_V": -n * am1 * LN2 / 2.0 * Vc ** 2,
        "second_order_K": -n * am1 ** 2 * K,
        "n_gamma": -n * gamma,
        "max_entropy": -n * apm1 / 4.0 * VPRIME ** 2,
        "theta_s1": -theta(e1) / am1,
        "theta_s2": -theta(e2) / apm1,
        "eps_EA": -(alpha / am1 + alphap / apm1 - 2.0) * math.log2(1.0 / settings.eps_EA),
        "theta_rest": -3.0 * theta(eps_s - e1 - 2.0 * e2),
        "EC_max": -ec_max(n, honest, gamma, p, sift, settings),
        "hash_tag": -float(settings.tag_bits),
        "eps_PA": -2.0 * math.log2(1.0 / settings.eps_PA),
        "const": 2.0,
    }
    return EATResult(sum(terms.values()), n, gamma, t, beta, alpha, alphap, e1, e2, p, wt, terms)


def _unpack(x):
    sig = lambda z: 1.0 / (1.0 + math.exp(-max(min(z, 60.0), -60.0)))
    gamma = sig(x[0])
    t = 0.75 + (W_MAX - 0.75) * sig(x[1])
    beta_frac = sig(x[2])
    am1 = 10.0 ** x[3]
    apm1 = (2.0 / VPRIME) * sig(x[4])
    fs1 = 0.999 * sig(x[5])
    fs2 = 0.999 * (1.0 - fs1) / 2.0 * sig(x[6])
    return gamma, t, beta_frac, am1, apm1, fs1, fs2


def optimum(n, honest: Honest, p=0.0, sift=0.5, settings: EATSettings = EATSettings(), restarts: int = 5,
            seed: int = 0, x_hint=None) -> EATResult:
    """Maximise (B12) over gamma, t, beta, alpha, alpha' and the smoothing split (multi-start Nelder-Mead).
    ``x_hint``: a previous optimiser point (result.x) used as the first start."""
    rng = np.random.default_rng(seed)
    base = np.array([math.log(min(0.3, 30.0 / math.sqrt(n))) if n > 1e4 else -2.0, 2.0, 3.0,
                     -0.5 * math.log10(n) - 0.3, -0.5 * math.log(n) + 1.0, 0.0, 0.0])
    f = lambda x: -max(key_length(n, honest, *_unpack(x), p=p, sift=sift, settings=settings).ell, -1e300)
    best, bx = -math.inf, base
    starts = ([np.asarray(x_hint, dtype=float)] if x_hint is not None else []) + \
             [base + (rng.normal(0.0, 1.0, base.size) if r else 0.0) for r in range(restarts)]
    for x0 in starts:
        for tol in ((1e-6, 1e-3, 6000), (1e-7, 1e-4, 4000)):
            res = minimize(f, x0, method="Nelder-Mead",
                           options={"xatol": tol[0], "fatol": tol[1], "maxiter": tol[2], "maxfev": tol[2], "adaptive": True})
            x0 = res.x
        if -res.fun > best:
            best, bx = -res.fun, res.x
    r = key_length(n, honest, *_unpack(bx), p=p, sift=sift, settings=settings)
    r.x = list(bx)
    return r


P_GRID = (0.0, 0.01, 0.02, 0.04, 0.06, 0.09, 0.12, 0.16, 0.21)


def optimum_over_p(n, honest: Honest, sift=0.5, settings: EATSettings = EATSettings(), restarts: int = 2,
                   hint=None) -> EATResult:
    """Key length maximised over the preprocessing probability p: grid, then a bounded refinement around the best grid
    point; p = 0 is part of the grid, so the result is never below the p = 0 optimum."""
    from scipy.optimize import minimize_scalar
    hint = hint or {}
    rows = {}
    for p in P_GRID:
        rows[p] = optimum(n, honest, p, sift, settings, restarts=restarts if p == 0.0 else 1, x_hint=hint.get(p))
    pb = max(rows, key=lambda p: rows[p].ell)
    i = P_GRID.index(pb)
    best = rows[pb]
    if pb > 0.0:
        a, b = P_GRID[max(i - 1, 0)], P_GRID[min(i + 1, len(P_GRID) - 1)]
        f = lambda p: -optimum(n, honest, p, sift, settings, restarts=1, x_hint=best.x).ell
        r = minimize_scalar(f, bounds=(a, b), method="bounded", options={"xatol": 1e-3})
        cand = optimum(n, honest, float(r.x), sift, settings, restarts=1, x_hint=best.x)
        if cand.ell > best.ell:
            best = cand
    best.grid = {p: rows[p].x for p in rows}
    return best


@dataclass
class EATThreshold:
    n: float
    ell: float
    result: EATResult
    target: float


def _descend_eat(honest: Honest, sift, p, target, settings, base: EATResult) -> Optional[EATResult]:
    """Warm-started descent below base.n (see descent.descend); the result at the smallest n reaching the target."""
    def ev(n, hint):
        if p == "opt":
            r = optimum_over_p(n, honest, sift, settings, hint=hint)
            return r.ell, r.grid, r
        pp = 0.0 if p == "zero" else float(p)
        r = optimum(n, honest, pp, sift, settings, restarts=1, x_hint=hint)
        return r.ell, r.x, r
    found = descend(ev, base.n, base.ell, base.grid if p == "opt" else base.x, target)
    return None if found is None else found[3]


def polish(honest: Honest, sift, p, n0, target=138.0, settings: EATSettings = EATSettings(),
           polish_bits: float = POLISH_BITS) -> Optional[EATThreshold]:
    """Refine a threshold n0 whose key length exceeds the target by more than ``polish_bits`` (warm-started descent
    from the optimum at n0).  None if the key length at n0 is within ``polish_bits`` or no smaller n reaches it."""
    if p == "opt":
        base = optimum_over_p(float(n0), honest, sift, settings)
    else:
        base = optimum(float(n0), honest, 0.0 if p == "zero" else float(p), sift, settings, restarts=3)
    if not base.ell - target > polish_bits:
        return None
    r = _descend_eat(honest, sift, p, target, settings, base)
    return None if r is None or r.n >= n0 else EATThreshold(r.n, r.ell, r, target)


def threshold(honest: Honest, sift=0.5, p="zero", target=138.0, settings: EATSettings = EATSettings(),
              lo=3.5, hi=11.0, rtol=1e-5, restarts: int = 3, guess=None,
              polish_bits: float = POLISH_BITS) -> EATThreshold:
    """Smallest integer n whose optimised EAT key length reaches ``target`` (p = 0, a fixed p, or "opt").
    Brent's method in log10 n, bracketed around ``guess`` if given; if the key length at the root exceeds the target
    by more than ``polish_bits``, a warm-started descent finds the smallest n below it that still reaches the target."""
    hint = {"x": None, "grid": None}

    def ev(n):
        if p == "opt":
            r = optimum_over_p(n, honest, sift, settings, hint=hint["grid"])
            hint["grid"] = r.grid
        else:
            pp = 0.0 if p == "zero" else float(p)
            r = optimum(n, honest, pp, sift, settings, restarts=restarts if hint["x"] is None else 1, x_hint=hint["x"])
        hint["x"] = r.x
        return r

    cache = {}

    def g(ln):
        k = round(ln, 12)
        if k not in cache:
            cache[k] = ev(float(round(10.0 ** ln)))
        return cache[k].ell - target
    if guess is not None and math.isfinite(guess):
        bot = top = math.log10(guess)
        step = 0.02
        while g(top) < 0:
            bot, top = top, top + step
            step *= 2
            if top > hi:
                return EATThreshold(math.inf, math.nan, None, target)
        if bot == top:
            step = 0.02
            while g(bot) >= 0:
                top, bot = bot, bot - step
                step *= 2
    else:
        if g(hi) < 0:
            return EATThreshold(math.inf, math.nan, None, target)
        bot, top = lo, hi
        while top - bot > 0.25:
            mid = 0.5 * (bot + top)
            if g(mid) >= 0:
                top = mid
            else:
                bot = mid
    root = brentq(g, bot, top, xtol=rtol / math.log(10.0), maxiter=100)
    above = sorted(k for k in cache if k >= root - 1e-12 and cache[k].ell >= target)
    r = cache[above[0]] if above else cache[round(top, 12)]
    if r.ell - target > polish_bits:
        pol = _descend_eat(honest, sift, p, target, settings, r)
        if pol is not None:
            r = pol
    return EATThreshold(r.n, r.ell, r, target)


def n_min(honest: Honest, sift=0.5, p="zero", target=138.0, settings: EATSettings = EATSettings(), **kw) -> float:
    return threshold(honest, sift, p, target, settings, **kw).n


def v_min(n, sift=0.5, p="zero", target=138.0, settings: EATSettings = EATSettings(), lo=0.80, hi=0.999,
          xtol=1e-6) -> float:
    """Smallest Werner visibility whose optimised key length at n rounds reaches ``target`` (Brent's method).
    ``p``: "zero", a fixed float, or "opt"."""
    hint = {"x": None, "grid": None}

    def g(V):
        hon = Honest.werner(V)
        if p == "opt":
            r = optimum_over_p(n, hon, sift, settings, hint=hint["grid"])
            hint["grid"] = r.grid
        else:
            pp = 0.0 if p == "zero" else float(p)
            r = optimum(n, hon, pp, sift, settings, restarts=4 if hint["x"] is None else 2, x_hint=hint["x"])
        hint["x"] = r.x
        return r.ell - target
    if g(hi) < 0:
        return math.nan
    a = hi
    while a - 0.01 >= lo and g(a - 0.01) >= 0:
        a -= 0.01
    V = brentq(g, max(a - 0.01, lo), a, xtol=xtol)
    while g(V) < 0:
        V += xtol
    return V
