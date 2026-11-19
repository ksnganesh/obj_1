"""Two-universal hashing for privacy amplification and error verification, and what its seed costs.

Toeplitz (out_len x n), seed s of n + out_len - 1 bits, T[i, j] = s[i - j + n - 1]; y = T x mod 2 is the central part of
the linear convolution of s and x, computed with an FFT (scipy.signal.fftconvolve) in O((n + out_len) log(...)).
Modified Toeplitz [T' | I] (Hayashi and Tsurumaru, IEEE Trans. Inf. Theory 62, 2213 (2016)): x = (x1, x2) with
|x2| = out_len, y = T' x1 xor x2, T' an out_len x (n - out_len) Toeplitz matrix: seed n - 1 bits.  Still universal_2:
for x != x', if x1 = x1' the outputs differ; otherwise a collision needs T' (x1 - x1') = x2 - x2', probability 2^-out_len.
"""
from __future__ import annotations

import math
from itertools import product

import numpy as np
from scipy.signal import fftconvolve


def toeplitz_seed_length(n: int, out_len: int) -> int:
    return n + out_len - 1


def modified_seed_length(n: int, out_len: int) -> int:
    return n - 1


def toeplitz_hash(x: np.ndarray, seed: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint8)
    s = np.asarray(seed, dtype=np.uint8)
    n = x.size
    if out_len <= 0:
        return np.zeros(0, np.uint8)
    if s.size != n + out_len - 1:
        raise ValueError(f"seed must have {n + out_len - 1} bits, got {s.size}")
    if n * out_len <= 4_000_000:                                      # small: exact integer arithmetic
        full = np.convolve(s.astype(np.int64), x.astype(np.int64))
        y = full[n - 1:n - 1 + out_len]
        return (y & 1).astype(np.uint8)
    full = fftconvolve(s.astype(np.float64), x.astype(np.float64), mode="full")
    y = full[n - 1:n - 1 + out_len]
    r = np.rint(y)
    if np.max(np.abs(y - r)) > 0.3:
        raise FloatingPointError("FFT rounding error too large")
    return (r.astype(np.int64) & 1).astype(np.uint8)


def toeplitz_matrix(seed: np.ndarray, n: int, out_len: int) -> np.ndarray:
    """Explicit T for small sizes (tests): T[i, j] = s[i - j + n - 1]."""
    s = np.asarray(seed, dtype=np.uint8)
    i = np.arange(out_len)[:, None]
    j = np.arange(n)[None, :]
    return s[i - j + n - 1]


def modified_toeplitz_hash(x: np.ndarray, seed: np.ndarray, out_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.uint8)
    n = x.size
    if out_len <= 0:
        return np.zeros(0, np.uint8)
    if not 0 < out_len < n:
        raise ValueError("need 0 < out_len < n")
    if np.asarray(seed).size != n - 1:
        raise ValueError(f"seed must have {n - 1} bits")
    x1, x2 = x[:n - out_len], x[n - out_len:]
    return toeplitz_hash(x1, seed, out_len) ^ x2


def collision_probability(family: str, n: int, out_len: int) -> float:
    """max over x != x' of Pr_seed[h(x) = h(x')], by exhaustive enumeration (small n only)."""
    seed_len = toeplitz_seed_length(n, out_len) if family == "toeplitz" else modified_seed_length(n, out_len)
    h = toeplitz_hash if family == "toeplitz" else modified_toeplitz_hash
    seeds = [np.array(s, np.uint8) for s in product((0, 1), repeat=seed_len)]
    xs = [np.array(v, np.uint8) for v in product((0, 1), repeat=n)]
    table = np.array([[int("".join(map(str, h(x, s, out_len))), 2) for s in seeds] for x in xs])
    worst = 0.0
    for i in range(len(xs)):
        for j in range(i + 1, len(xs)):
            worst = max(worst, float(np.mean(table[i] == table[j])))
    return worst


def per_signature_seed(n_in: float, n_out: float, signatures: float, family: str = "toeplitz") -> float:
    if signatures <= 0:
        return math.inf
    L = toeplitz_seed_length(int(n_in), int(n_out)) if family == "toeplitz" else modified_seed_length(int(n_in), int(n_out))
    return L / signatures
