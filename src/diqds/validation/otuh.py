"""One-time universal hashing: the hash used by the messaging stage (protocol/messaging.py; Fan et al. Sec. II B,
App. A).

Hash.  h_{p,s}(M) = M(x) s(x) mod p(x), with p = f(Z) irreducible of degree n_s and s = X the n_s-bit initial state.
This is the LFSR-based Toeplitz family of App. A written in the polynomial basis (an LFSR with feedback p multiplies its
state by x modulo p, so the Toeplitz product M_W(W) s is M(x) s(x) mod p up to a fixed linear relabelling of the
output); for M != M' a collision needs p | (M - M'), which a uniform irreducible p of degree n_s does with probability
<= n_m 2^(-n_s) - the first term of Eq. (5).  The hash is linear in M, which is why a recipient who knows X_c and Z_c
can forge (protocol/messaging.py, forgery_attempt).
Keys: one 3 n_s-bit segment is cut into (X, Y, Z); the signing and verification steps are in protocol/messaging.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..common import N_S
from .gf2 import bits_to_int, message_poly, mulmod, poly_from_key, polymod


@dataclass(frozen=True)
class SegmentKeys:
    X: int
    Y: int
    Z: int


def cut(segment_bits: np.ndarray, n_s: int = N_S) -> SegmentKeys:
    """3 n_s key bits -> (X, Y, Z), each an n_s-bit int."""
    s = np.asarray(segment_bits, dtype=np.uint8)
    return SegmentKeys(bits_to_int(s[:n_s]), bits_to_int(s[n_s:2 * n_s]), bits_to_int(s[2 * n_s:3 * n_s]))


def digest(doc_bits: np.ndarray, X: int, Z: int, n_s: int = N_S) -> int:
    """h_{f(Z), X}(Doc): the n_s-bit digest.  Raises gf2.DegenerateKey if Z lies in a proper subfield."""
    p = poly_from_key(Z, n_s)
    return mulmod(polymod(message_poly(doc_bits), p), X, p)
