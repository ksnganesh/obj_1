"""Single-round entropy bounds for CHSH-based DI key distribution.

von Neumann (used by Eq. (9) of Fan et al., Phys. Rev. A 114, 032617 (2026), i.e. Tan et al. Theorem 1):
    r_0(w)    = 1 - h((1 + sqrt(u)) / 2),     u = 16 w^2 - 16 w + 3 = (S/2)^2 - 1                    (B3)
    r_p(w)    = r_0(w) + h((1 + sqrt((1-2p)^2 + 4p(1-p) u)) / 2)                                      (B2)

Renyi (Hahn, Philip, Tan, Brown, arXiv:2507.07365, Theorem 2), alpha > 1, g = sqrt(S^2/4 - 1):
    phi_mu(S) = ((1-g)/2)^mu + ((1+g)/2)^mu
    f_down(S) = 1 + alpha/(1-alpha) log2 phi_{1/alpha}(S)                  sandwiched H-down_alpha(A|X=0,E)
    f_up(S)   = 1 + (2alpha-1)/(1-alpha) log2 phi_{alpha/(2alpha-1)}(S)    sandwiched H-up_alpha(A|X=0,E)
Noisy preprocessing with flip probability q (their App. C, Eqs. C6-C7): with a = (1-g)^(1/a) + (1+g)^(1/a),
    h(S) = [ (a + D)^alpha + (a - D)^alpha ] / 2^(alpha+1),   D = sqrt(a^2 - 16 (1-g^2)^(1/alpha) (q - q^2))
    f(S) = 1 + log2( h^(S) ) / (1 - alpha),     h^ = upper concave envelope of h on [2, 2 sqrt2]
At q = 0 this reduces to f_down; at S = 2 it is the Renyi entropy of the flip, H_alpha(Bernoulli(q)).

Sifted generation rounds (Lemma 1 of the article): if a public register decides that a generation round is kept
with probability s (and replaced by a constant otherwise), the sandwiched H-down with classical conditioning gives
    H_alpha(A~ | K E) = 1/(1-alpha) log2 sum_k P(k) 2^{(1-alpha) H_alpha(A~|E, K = k)}
so the generation-round rate is  1/(1-alpha) log2( s 2^{(1-alpha) f} + 1 - s )  (-> s f as alpha -> 1).
Two key settings (random key measurements; Tan et al. Protocol 1, as in Fan et al.): s = 1/2 and H_alpha(A_x|E) >= f
for both x (the CHSH expression is symmetric under a relabelling that swaps Alice's inputs).  Independent test flags
(Lu et al., Science 391, 592 (2026)): s = (1 - sqrt(gamma/2))^2 / (1 - gamma), see protocols.py.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from ..common import LN2, SQRT2, W_CLASSICAL, W_MAX, h2

S_MIN, S_MAX = 2.0, 2.0 * SQRT2


def _u(w):
    w = np.asarray(w, dtype=float)
    return np.clip(16.0 * w * w - 16.0 * w + 3.0, 0.0, 1.0)


# ----------------------------------------------------------------------------------------------- von Neumann
def r0(w):
    u = _u(w)
    return np.where(u > 0.0, 1.0 - h2(0.5 * (1.0 + np.sqrt(u))), 0.0)


def rp(w, p):
    u = _u(w)
    z2 = (1.0 - 2.0 * p) ** 2 + 4.0 * p * (1.0 - p) * u
    return r0(w) + h2(0.5 * (1.0 + np.sqrt(z2)))


# ----------------------------------------------------------------------------------------------- Renyi, Theorem 2
def _g_of_w(w):
    S = np.clip(8.0 * np.asarray(w, dtype=float) - 4.0, S_MIN, S_MAX)
    return np.sqrt(np.clip(S * S / 4.0 - 1.0, 0.0, 1.0))


def f_down(w, alpha: float):
    g = _g_of_w(w)
    mu = 1.0 / alpha
    phi = ((1.0 - g) / 2.0) ** mu + ((1.0 + g) / 2.0) ** mu
    return 1.0 + alpha / (1.0 - alpha) * np.log2(phi)


def f_up(w, alpha: float):
    g = _g_of_w(w)
    mu = alpha / (2.0 * alpha - 1.0)
    phi = ((1.0 - g) / 2.0) ** mu + ((1.0 + g) / 2.0) ** mu
    return 1.0 + (2.0 * alpha - 1.0) / (1.0 - alpha) * np.log2(phi)


def f_vn(w):
    """alpha -> 1 limit of both: the von Neumann bound r_0."""
    return r0(w)


# ----------------------------------------------------------------------------------------------- noisy preprocessing
def h_c7(S, alpha: float, q: float):
    """Eq. (C7) of Hahn et al. (beta = 1)."""
    S = np.clip(np.asarray(S, dtype=float), S_MIN, S_MAX)
    g = np.sqrt(np.clip(S * S / 4.0 - 1.0, 0.0, 1.0))
    ia = 1.0 / alpha
    a = (1.0 - g) ** ia + (1.0 + g) ** ia
    disc = np.sqrt(np.clip(a * a - 16.0 * (1.0 - g * g) ** ia * (q - q * q), 0.0, None))
    return ((a + disc) ** alpha + np.clip(a - disc, 0.0, None) ** alpha) / 2.0 ** (alpha + 1.0)


@lru_cache(maxsize=8192)
def _envelope(alpha: float, q: float, m: int = 1601):
    """Upper concave envelope of h_c7 on [2, 2 sqrt2]; returns (vertices S, values, gap, concave).

    h_c7 is concave in S for every (alpha, q) we have tried (tests/test_finite_size.py), in which case the envelope is
    h_c7 itself (gap 0) and f_down_np evaluates h_c7 exactly; otherwise the monotone-chain upper hull is used."""
    S = np.linspace(S_MIN, S_MAX, m)
    y = h_c7(S, alpha, q)
    d2 = np.diff(y, 2)
    if float(np.max(d2)) <= 1e-14 * float(np.max(np.abs(y))):
        return S, y, 0.0, True
    hx, hy = [S[0]], [y[0]]
    for xi, yi in zip(S[1:], y[1:]):
        while len(hx) >= 2 and (hx[-1] - hx[-2]) * (yi - hy[-2]) - (hy[-1] - hy[-2]) * (xi - hx[-2]) >= 0.0:
            hx.pop()
            hy.pop()
        hx.append(xi)
        hy.append(yi)
    hx, hy = np.array(hx), np.array(hy)
    gap = float(np.max(np.interp(S, hx, hy) - y))
    return hx, hy, gap, False


def envelope_gap(alpha: float, q: float) -> float:
    """max(envelope - h) on the grid: 0 means h is already concave (the envelope changes nothing)."""
    return _envelope(float(alpha), float(q))[2]


def f_down_np(w, alpha: float, q: float):
    """Generation-round rate with noisy preprocessing (flip probability q) for one key setting."""
    if q <= 0.0:
        return f_down(w, alpha)
    S = np.clip(8.0 * np.asarray(w, dtype=float) - 4.0, S_MIN, S_MAX)
    hx, hy, _, concave = _envelope(round(float(alpha), 13), round(float(q), 13))
    h = h_c7(S, alpha, q) if concave else np.interp(S, hx, hy)
    return 1.0 + np.log2(h) / (1.0 - alpha)


def renyi_bernoulli(q: float, alpha: float) -> float:
    """H_alpha of a Bernoulli(q) bit: the S = 2 value of f_down_np."""
    if q <= 0.0:
        return 0.0
    return float(np.log2((1.0 - q) ** alpha + q ** alpha) / (1.0 - alpha))


# ----------------------------------------------------------------------------------------------- classical conditioning
def renyi_average(F_values, weights, alpha: float):
    """H-down_alpha(A|X E) = 1/(1-alpha) log2 sum_x P(x) 2^{(1-alpha) H_x}, for H_x = H-down_alpha(A|E, X = x)."""
    F = np.asarray(F_values, dtype=float)
    wts = np.asarray(weights, dtype=float).reshape((-1,) + (1,) * (F.ndim - 1))
    x = (1.0 - alpha) * LN2 * F                                # exponent in natural units
    return np.log(np.sum(wts * np.exp(x), axis=0)) / ((1.0 - alpha) * LN2)


def sifted_rate(F, alpha: float, s: float):
    """Generation-round rate when a generation round is kept with probability s (Lemma 1; stable form)."""
    F = np.asarray(F, dtype=float)
    if s >= 1.0:
        return F
    x = (1.0 - alpha) * LN2 * F                                # <= 0 for alpha > 1
    return np.log1p(s * np.expm1(x)) / ((1.0 - alpha) * LN2)


def two_setting_rate(F, alpha: float):
    """Generation-round rate for two uniformly random key settings with sifting (s = 1/2)."""
    return sifted_rate(F, alpha, 0.5)


def classical_floor(alpha: float, q: float) -> float:
    """Rate at S <= 2: zero without preprocessing, H_alpha(Bernoulli(q)) with it."""
    return renyi_bernoulli(q, alpha)


__all__ = ["r0", "rp", "f_down", "f_up", "f_vn", "h_c7", "f_down_np", "envelope_gap", "renyi_bernoulli",
           "renyi_average", "sifted_rate", "two_setting_rate", "classical_floor", "W_CLASSICAL", "W_MAX"]
