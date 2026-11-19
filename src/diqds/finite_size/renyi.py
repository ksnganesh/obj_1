"""Finite-size key length from the Renyi entropy accumulation theorem (REAT), written from scratch.

Sources: Arqand, Hahn, Tan, Phys. Rev. X 15, 041013 (2025) (generalised REAT); Hahn, Philip, Tan, Brown,
arXiv:2507.07365 (analytic Renyi bounds), App. D.  Key length (their Eq. D6; Theorem 1 of the article's Appendix C):

    ell = n h_alpha - n (gamma + delta_perp^low) - ell_EC - ell_EV - alpha/(alpha-1) log2(1/eps_secret) + 2
    h_alpha = inf_{w} min_{q in S_acc} [ D(q || p_w)/(alpha - 1) + q(perp) F_alpha(w) ]
    p_w = (gamma (1 - w), gamma w, 1 - gamma)          test register c in {lose, win, perp}
    ell_EC = n [ (1 - gamma) s f_key h(Q_q) + gamma f_test h(1 - w_hon) ] + 50 sqrt(n),   Q_q = Q + q - 2 q Q,
             s = sifting factor (protocols.py); f = 1 in the literature; RenyiSettings.cascade() sizes it for Cascade
    ell_EV = 64 bits (delta = 2^-61),  eps_secret = eps_sound - eps_corr

F_alpha(w) is the generation-round rate of the protocol (entropy.py / protocols.py).

Inner problem (Lemma 2), solved exactly: for fixed w the objective is sum_i q_i log2(q_i/p_i)/(alpha-1) + q_perp F,
separable and strictly convex, with one equality (sum q = 1) and box constraints (the acceptance set).  The KKT
conditions give q_i = clip(p_i c_i e^{-lambda}, lo_i, up_i) with c = (1, 1, 2^{-(alpha-1) F}) and lambda fixed by
normalisation.  ``_inner`` returns both the value at this feasible point (an upper bound on the minimum) and the
Lagrange dual function at the same multiplier (a lower bound, by weak duality); the two agree to rounding error.

Outer infimum over w, certified (``h_alpha_certified``): the fast path used inside the optimiser evaluates the inner
minimum on a grid with three zoom stages, which only gives an UPPER bound on the infimum.  The reported key lengths use
a certified LOWER bound instead, from a branch-and-bound over w in [W_MIN, W_MAX] (the whole quantum range): on an
interval [a, b] with midpoint m and half-width d,
    D(q||p_w) >= D(q||p_m) + (w - m) d/dw D(q||p_w)|_m      (D is convex in w)
    F(w) >= F(a)                                             (F is non-decreasing in w)
so  min_{w in [a,b]} J(w) >= min_{sigma = +-1} min_q [ D(q||p_m)/(alpha-1) + sigma d S(q) + q_perp F(a) ],
S(q) = [q_lose/(1-m) - q_win/m] / ((alpha-1) ln 2), linear in q: each bracket is again a separable problem solved by
the dual bound.  Intervals whose lower bound exceeds the best grid value minus the tolerance are discarded, the others
halved, until none remains; the tolerance is ``cert_tol_bits / n`` per round, so the certified key length is within
``cert_tol_bits`` of the grid value and never above the infimum.

Acceptance test: one-sided binomial tests on the counts of lost test rounds (upper), won test rounds (lower) and
generation rounds (lower), each failing for honest devices with probability <= ``tail``.  Default tail =
Phi(-3) = 1.35e-3, the per-test failure probability of the k = 3 margin of Fan et al.'s Eq. (B10), so the Renyi and EAT
bounds are compared at matched completeness.  Hahn et al. use six tails sharing 1e-3 (``RenyiSettings.hahn()``).  The
three omitted tails (lost-low, won-high, generation-high) are inactive: the minimiser of h_alpha has the fewest wins,
the most losses and the fewest generation rounds that the test accepts, so adding the opposite constraints does not
change h_alpha (tests/test_finite_size.py::test_omitted_tails_are_inactive); dropping them only enlarges S_acc, which
is conservative for soundness, and removes three ways for honest devices to fail.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy.special import xlogy
from scipy.stats import binom, norm

from ..common import LN2, W_CLASSICAL, W_MAX, W_MIN, Honest, h2
from .entropy import f_down_np
from .protocols import ONE_SETTING, Protocol

TAIL_3SIGMA = float(norm.sf(3.0))          # 1.3499e-3
COMPONENTS = ("loss", "win", "perp")


@dataclass(frozen=True)
class RenyiSettings:
    eps_sound: float = 5.0e-11             # per link (Table I: eps_snd)
    eps_corr: float = 2.0 ** -61
    tail: float = TAIL_3SIGMA
    tails: Tuple[str, ...] = ("loss_upp", "win_low", "perp_low")
    ev_bits: int = 64
    ec_sqrt: float = 50.0
    ec_f_key: float = 1.0                  # EC efficiency on key rounds (1 = Shannon limit, as in the literature)
    ec_f_test: float = 1.0                 # EC efficiency on test rounds (error rate 1 - w)
    grid: int = 161
    zoom_stages: int = 3
    certify: bool = True                   # report key lengths with the certified lower bound on h_alpha
    cert_tol_bits: float = 1e-2            # certified ell is within this many bits of the grid value

    @property
    def eps_secret(self) -> float:
        return self.eps_sound - self.eps_corr

    @property
    def completeness_AT(self) -> float:
        return self.tail * len(self.tails)

    @classmethod
    def cascade(cls, **kw) -> "RenyiSettings":
        """EC budget sized for the emulator's Cascade: measured f = 1.14-1.19 on the key rounds of the ion and 11 km links
        (Q <= 4.8 %), 1.30 on the test rounds (1 - w = 17 %); results/validation.json, section ec_efficiency.  At the
        7.8 % QBER of the 400 m link Cascade needs f = 1.23, above this budget."""
        return cls(ec_f_key=1.20, ec_f_test=1.33, **kw)

    @classmethod
    def hahn(cls, eps_sound: float = 1.0e-10) -> "RenyiSettings":
        """Hahn et al.'s Fig. 1 settings: eps_sound = 1e-10, six one-sided tails sharing eps_com^AT = 1e-3."""
        return cls(eps_sound=eps_sound, tail=1.0e-3 / 6.0,
                   tails=("loss_low", "loss_upp", "win_low", "win_upp", "perp_low", "perp_upp"))


# ---------------------------------------------------------------------------------------------- acceptance set
def _quantile_low(n: float, p: float, tail: float) -> float:
    """Smallest accepted count k such that P(K < k) <= tail (honest K ~ Bin(n, p))."""
    k = float(binom.ppf(tail, n, p))
    if not np.isfinite(k):                              # scipy fallback for extreme n
        k = n * p - norm.isf(tail) * math.sqrt(n * p * (1 - p))
    return max(k, 0.0)


def _quantile_upp(n: float, p: float, tail: float) -> float:
    k = float(binom.isf(tail, n, p))
    if not np.isfinite(k):
        k = n * p + norm.isf(tail) * math.sqrt(n * p * (1 - p))
    return min(k, n)


@dataclass(frozen=True)
class AcceptanceSet:
    n: float
    gamma: float
    w_hon: float
    lo: Tuple[float, float, float]
    up: Tuple[float, float, float]
    k_lo: Tuple[float, float, float]      # count thresholds (accept iff K >= k_lo and K <= k_up)
    k_up: Tuple[float, float, float]

    @property
    def q_hon(self) -> Tuple[float, float, float]:
        g, w = self.gamma, self.w_hon
        return (g * (1 - w), g * w, 1 - g)

    @property
    def delta_perp_low(self) -> float:
        return max((1.0 - self.gamma) - self.lo[2], 0.0)


def acceptance_set(n: float, gamma: float, w_hon: float, settings: RenyiSettings) -> AcceptanceSet:
    qh = (gamma * (1 - w_hon), gamma * w_hon, 1 - gamma)
    lo, up = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    klo, kup = [0.0, 0.0, 0.0], [float(n)] * 3
    for i, c in enumerate(COMPONENTS):
        if f"{c}_low" in settings.tails:
            klo[i] = _quantile_low(n, qh[i], settings.tail)
            lo[i] = klo[i] / n
        if f"{c}_upp" in settings.tails:
            kup[i] = _quantile_upp(n, qh[i], settings.tail)
            up[i] = kup[i] / n
    return AcceptanceSet(n, gamma, w_hon, tuple(lo), tuple(up), tuple(klo), tuple(kup))


def completeness_exact(acc: AcceptanceSet) -> float:
    """Union bound on the honest abort probability of the acceptance test (sum of the one-sided failure probs)."""
    tot = 0.0
    for i, p in enumerate(acc.q_hon):
        if acc.k_lo[i] > 0:
            tot += float(binom.cdf(acc.k_lo[i] - 1, acc.n, p))
        if acc.k_up[i] < acc.n:
            tot += float(binom.sf(acc.k_up[i], acc.n, p))
    return tot


# ---------------------------------------------------------------------------------------------- inner problem (KKT)
def _kkt_q(p: np.ndarray, c: np.ndarray, lo: np.ndarray, up: np.ndarray, iters: int = 70) -> np.ndarray:
    """Rows of q minimising sum_i q_i log(q_i/p_i)/(a-1) + linear tilt, inside the box, summing to 1."""
    pc = p * c
    a = np.full(p.shape[0], -60.0)
    b = np.full(p.shape[0], 60.0)
    for _ in range(iters):
        lam = 0.5 * (a + b)
        s = np.clip(pc * np.exp(-lam)[:, None], lo, up).sum(axis=1)
        big = s > 1.0
        a = np.where(big, lam, a)
        b = np.where(big, b, lam)
    lam = 0.5 * (a + b)
    raw = pc * np.exp(-lam)[:, None]
    q = np.clip(raw, lo, up)
    free = (raw > lo) & (raw < up)
    fixed_sum = np.where(free, 0.0, q).sum(axis=1)
    free_sum = np.where(free, pc, 0.0).sum(axis=1)
    ok = free_sum > 0
    scale = np.where(ok, (1.0 - fixed_sum) / np.where(ok, free_sum, 1.0), 0.0)
    q = np.where(free & ok[:, None], pc * scale[:, None], q)
    return q


def _inner(p: np.ndarray, t: np.ndarray, lo: np.ndarray, up: np.ndarray, alpha: float) -> Tuple[np.ndarray, np.ndarray]:
    """For every row: min over q in the box [lo, up] with sum q = 1 of sum_i q_i log2(q_i/p_i)/(alpha-1) + t_i q_i.

    Returns (primal, dual): the objective at the KKT point (feasible, an upper bound on the minimum) and the Lagrange
    dual function at the same multiplier (a lower bound by weak duality).  In natural units, with p~_i = p_i e^{-A t_i}
    and A = (alpha-1) ln 2, the objective is sum_i q_i ln(q_i/p~_i) / A, the box minimiser of the Lagrangian with
    multiplier lambda - 1 is q_i = clip(p~_i e^{-lambda}, lo_i, up_i), and the dual value is
    sum_i q_i ln(q_i/p~_i) + (lambda - 1)(sum_i q_i - 1).

    lambda is found exactly: s(lambda) = sum_i clip(p~_i e^{-lambda}) is monotone with breakpoints ln(p~_i/lo_i) and
    ln(p~_i/up_i); the breakpoint interval where s crosses 1 fixes the free components, and then
    lambda = ln(sum_free p~_i) - ln(1 - sum_fixed bounds)."""
    with np.errstate(all="ignore"):
        return _inner_impl(p, t, lo, up, alpha)


def _inner_impl(p, t, lo, up, alpha):
    A = (alpha - 1.0) * LN2
    logpt = np.log(p) - A * t                                       # (N, 3)
    llo, lup = np.log(lo), np.log(up)
    bps = np.sort(np.concatenate([logpt - llo, logpt - lup], axis=1), axis=1)      # (N, 6), +inf where lo = 0
    with np.errstate(over="ignore", invalid="ignore"):
        S = np.clip(np.exp(logpt[:, None, :] - bps[:, :, None]), lo, up).sum(axis=2)    # s at each breakpoint
    cnt = (S >= 1.0).sum(axis=1)                                   # s decreases with lambda
    rows = np.arange(p.shape[0])
    left = np.where(cnt > 0, bps[rows, np.maximum(cnt - 1, 0)], -np.inf)
    right = np.where(cnt < 6, bps[rows, np.minimum(cnt, 5)], np.inf)
    mid = np.where(np.isfinite(left) & np.isfinite(right), 0.5 * (left + right),
                   np.where(np.isfinite(left), left + 1.0, right - 1.0))
    with np.errstate(over="ignore", invalid="ignore"):
        raw_mid = np.exp(logpt - mid[:, None])
    free = (raw_mid > lo) & (raw_mid < up)
    fixed = np.where(free, 0.0, np.clip(raw_mid, lo, up)).sum(axis=1)
    anyfree = free.any(axis=1)
    lse = np.log(np.where(free, np.exp(logpt - logpt.max(axis=1, keepdims=True)), 0.0).sum(axis=1)
                 + np.where(anyfree, 0.0, 1.0)) + logpt.max(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        lam = np.where(anyfree, lse - np.log(np.maximum(1.0 - fixed, 1e-300)), mid)
    with np.errstate(over="ignore", invalid="ignore"):
        q = np.clip(np.exp(logpt - lam[:, None]), lo, up)
        logq = np.where(free, logpt - lam[:, None], np.log(q))
        term = np.where(q > 0, q * (logq - logpt), 0.0).sum(axis=1)
    primal = term / A
    dual = (term + (lam - 1.0) * (q.sum(axis=1) - 1.0)) / A
    return primal, np.minimum(dual, primal)


def _p_of_w(gamma: float, w: np.ndarray) -> np.ndarray:
    return np.stack([gamma * (1.0 - w), gamma * w, np.full_like(w, 1.0 - gamma)], axis=1)


def objective_J(w: np.ndarray, gamma: float, alpha: float, F: np.ndarray, acc: AcceptanceSet) -> np.ndarray:
    """min over q in S_acc of D(q||p_w)/(alpha-1) + q_perp F(w), for every w in the array (value at the KKT point)."""
    w = np.asarray(w, dtype=float)
    F = np.broadcast_to(np.asarray(F, dtype=float), w.shape)
    t = np.stack([np.zeros_like(w), np.zeros_like(w), F], axis=1)
    primal, _ = _inner(_p_of_w(gamma, w), t, np.array(acc.lo), np.array(acc.up), alpha)
    return primal


def h_alpha(gamma: float, alpha: float, F_func: Callable[[np.ndarray], np.ndarray], acc: AcceptanceSet,
            grid: int = 161, zoom_stages: int = 3) -> Tuple[float, float]:
    """(h_alpha, minimising w)."""
    lo_w, hi_w = W_CLASSICAL, W_MAX
    w = np.linspace(lo_w, hi_w, grid)
    J = objective_J(w, gamma, alpha, F_func(w), acc)
    best = float(np.min(J))
    i = int(np.argmin(J))
    w_best = float(w[i])
    step = w[1] - w[0]
    for _ in range(zoom_stages):
        a, b = max(lo_w, w_best - 2 * step), min(hi_w, w_best + 2 * step)
        w = np.linspace(a, b, 41)
        J = objective_J(w, gamma, alpha, F_func(w), acc)
        i = int(np.argmin(J))
        if J[i] < best:
            best, w_best = float(J[i]), float(w[i])
        step = w[1] - w[0]
    return best, w_best


@dataclass(frozen=True)
class Certificate:
    lower: float           # certified lower bound on h_alpha (inf over w in [W_MIN, W_MAX])
    upper: float           # best value found (an upper bound on the infimum)
    w_star: float
    converged: bool
    levels: int
    intervals: int


def h_alpha_certified(gamma: float, alpha: float, F_func: Callable[[np.ndarray], np.ndarray], acc: AcceptanceSet,
                      tol: float, upper_hint: float = math.inf, w_hint: float = math.nan, init: int = 160,
                      max_levels: int = 80, max_intervals: int = 1_000_000) -> Certificate:
    """Branch-and-bound lower bound on h_alpha = inf_w J(w) over the quantum range (module docstring)."""
    lo, up = np.array(acc.lo), np.array(acc.up)
    A = (alpha - 1.0) * LN2
    edges = np.unique(np.concatenate([np.linspace(W_MIN, W_CLASSICAL, 9), np.linspace(W_CLASSICAL, W_MAX, init + 1)]))
    a, b = edges[:-1], edges[1:]
    ub, w_ub = float(upper_hint), float(w_hint)
    certified = math.inf
    converged, level, count = False, 0, 0
    for level in range(1, max_levels + 1):
        m, d = 0.5 * (a + b), 0.5 * (b - a)
        p = _p_of_w(gamma, m)
        Fa, Fm = F_func(a), F_func(m)
        zero = np.zeros_like(m)
        jm, _ = _inner(p, np.stack([zero, zero, Fm], axis=1), lo, up, alpha)
        i = int(np.argmin(jm))
        if jm[i] < ub:
            ub, w_ub = float(jm[i]), float(m[i])
        c0, c1 = d / ((1.0 - m) * A), d / (m * A)
        _, lb_p = _inner(p, np.stack([c0, -c1, Fa], axis=1), lo, up, alpha)
        _, lb_m = _inner(p, np.stack([-c0, c1, Fa], axis=1), lo, up, alpha)
        lb = np.minimum(lb_p, lb_m)
        count += lb.size
        keep = lb < ub - tol
        if np.any(~keep):
            certified = min(certified, float(np.min(lb[~keep])))
        if not np.any(keep):
            converged = True
            break
        if 2 * int(keep.sum()) > max_intervals:
            certified = min(certified, float(np.min(lb[keep])))
            break
        a, b, m = a[keep], b[keep], m[keep]
        a, b = np.concatenate([a, m]), np.concatenate([m, b])
    else:
        certified = min(certified, float(np.min(lb)))
    return Certificate(min(certified, ub), ub, w_ub, converged, level, count)


# ---------------------------------------------------------------------------------------------- key length
def key_round_entropy(honest: Honest, protocol: Protocol, q: float) -> float:
    """Shannon cost per key bit: h(Q_q) for one key basis, the average over Z and X for two (Q_x from Honest)."""
    flip = lambda Q: Q + q - 2.0 * q * Q
    if protocol.key_bases == 2:
        return 0.5 * (float(h2(flip(honest.Q))) + float(h2(flip(honest.Q_x))))
    return float(h2(flip(honest.Q)))


def ec_bits(n: float, honest: Honest, protocol: Protocol, gamma: float, q: float, settings: RenyiSettings) -> float:
    return (n * ((1.0 - gamma) * protocol.sift_at(gamma) * settings.ec_f_key * key_round_entropy(honest, protocol, q)
                 + gamma * settings.ec_f_test * h2(1.0 - honest.w)) + settings.ec_sqrt * math.sqrt(n))


@dataclass
class RenyiResult:
    ell: float                         # net key length (bits); for SEEDED the PA output is ell + n
    n: float
    gamma: float
    alpha: float
    q: float
    w_star: float
    h_alpha: float
    protocol: str
    terms: Dict[str, float] = field(default_factory=dict)
    acceptance: Optional[AcceptanceSet] = None
    info: Dict[str, float] = field(default_factory=dict)


def key_length(n: float, honest: Honest, protocol: Protocol, gamma: float, alpha: float, q: float = 0.0,
               settings: RenyiSettings = RenyiSettings(), certify: bool = False) -> RenyiResult:
    """Eq. (D6).  ``certify``: replace the grid value of h_alpha by its certified lower bound (h_alpha_certified)."""
    if not (0.0 < gamma < 1.0 and alpha > 1.0 and 0.0 <= q < 0.5):
        return RenyiResult(-math.inf, n, gamma, alpha, q, float("nan"), float("nan"), protocol.key)
    acc = acceptance_set(n, gamma, honest.w, settings)

    def F_func(w):
        return protocol.gen_rate(f_down_np(w, alpha, q), alpha, gamma)

    h, w_star = h_alpha(gamma, alpha, F_func, acc, settings.grid, settings.zoom_stages)
    info = {"h_alpha_grid": h}
    if certify:
        cert = h_alpha_certified(gamma, alpha, F_func, acc, tol=settings.cert_tol_bits / n, upper_hint=h, w_hint=w_star)
        info.update({"h_alpha_certified": cert.lower, "h_alpha_upper": cert.upper, "certified": float(cert.converged),
                     "certification_gap_bits": n * (cert.upper - cert.lower), "bnb_levels": cert.levels})
        h, w_star = cert.lower, cert.w_star
    terms = {
        "n_h_alpha": n * h,
        "test_rounds": -n * (gamma + acc.delta_perp_low),
        "error_correction": -ec_bits(n, honest, protocol, gamma, q, settings),
        "error_verification": -float(settings.ev_bits),
        "privacy_amplification": -alpha / (alpha - 1.0) * math.log2(1.0 / settings.eps_secret),
        "const": 2.0,
    }
    ell = sum(terms.values())
    res = RenyiResult(ell, n, gamma, alpha, q, w_star, h, protocol.key, terms, acc)
    res.info = info
    return res


def asymptotic_rate(honest: Honest, protocol: Protocol = ONE_SETTING, q: float = 0.0, gamma: float = 0.0) -> float:
    """n -> infinity limit (alpha -> 1, no statistical margins), for orientation."""
    from .entropy import rp
    F = float(rp(honest.w, q))
    sft = protocol.sift_at(gamma) if protocol.independent_flags else protocol.sift
    Qq = honest.Q + q - 2 * q * honest.Q
    return (1 - gamma) * sft * (F - h2(Qq)) - gamma * (1 + h2(1 - honest.w))
