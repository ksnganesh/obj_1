"""GF(2)[x] arithmetic on Python ints (bit i = coefficient of x^i), Rabin's irreducibility test, Berlekamp-Massey, and
the polynomial generator p = f(Z) of the signing stage.

f(Z) follows Li et al., Phys. Rev. Applied 20, 044011 (2023), App. B.1 (after Shoup): fix a public irreducible f0 of
degree n; read Z as an element g of GF(2^n) = GF(2)[x]/(f0); return the minimal polynomial of g (Berlekamp-Massey on the
sequence a_i = [g^i mod f0](0)).  Each degree-n irreducible polynomial is the minimal polynomial of exactly n field
elements, so a uniform Z yields a uniform irreducible p using only n key bits.  With probability ~2^(-n/2) g lies in a
proper subfield (degree < n): such a key is degenerate and is skipped.
"""
from __future__ import annotations

from functools import lru_cache
from typing import List, Sequence, Tuple


def deg(a: int) -> int:
    return a.bit_length() - 1


def clmul(a: int, b: int) -> int:
    r = 0
    while b:
        if b & 1:
            r ^= a
        a <<= 1
        b >>= 1
    return r


def polymod(a: int, m: int) -> int:
    dm = deg(m)
    while a and deg(a) >= dm:
        a ^= m << (deg(a) - dm)
    return a


def mulmod(a: int, b: int, m: int) -> int:
    n = deg(m)
    a = polymod(a, m)
    r = 0
    top = 1 << n
    while b:
        if b & 1:
            r ^= a
        b >>= 1
        a <<= 1
        if a & top:
            a ^= m
    return r


def powmod(a: int, e: int, m: int) -> int:
    r, a = 1, polymod(a, m)
    while e:
        if e & 1:
            r = mulmod(r, a, m)
        a = mulmod(a, a, m)
        e >>= 1
    return r


def gcd(a: int, b: int) -> int:
    while b:
        a, b = b, polymod(a, b)
    return a


def _prime_factors(n: int) -> List[int]:
    out, d = [], 2
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            while n % d == 0:
                n //= d
        d += 1
    if n > 1:
        out.append(n)
    return out


def is_irreducible(f: int) -> bool:
    """Rabin: f of degree n is irreducible iff x^(2^n) = x mod f and gcd(x^(2^(n/q)) - x, f) = 1 for primes q | n."""
    n = deg(f)
    if n <= 0:
        return False
    if n == 1:
        return True

    def x_pow_2k(k: int) -> int:
        t = 2
        for _ in range(k):
            t = mulmod(t, t, f)
        return t

    if x_pow_2k(n) != 2:
        return False
    return all(gcd(f, x_pow_2k(n // q) ^ 2) == 1 for q in _prime_factors(n))


@lru_cache(maxsize=None)
def default_modulus(n: int) -> int:
    """First irreducible polynomial x^n + ... + 1 in increasing order of its low part (public parameter f0)."""
    for low in range(1, 1 << min(n, 24), 2):
        f = (1 << n) | low
        if is_irreducible(f):
            return f
    raise RuntimeError("no irreducible polynomial found")


def berlekamp_massey(seq: Sequence[int]) -> Tuple[List[int], int]:
    """Shortest LFSR (c_0 = 1, c_1..c_L) generating seq over GF(2)."""
    N = len(seq)
    c = [0] * (N + 1)
    b = [0] * (N + 1)
    c[0] = b[0] = 1
    L, m = 0, 1
    for i in range(N):
        d = int(seq[i])
        for j in range(1, L + 1):
            d ^= c[j] & int(seq[i - j])
        if d == 0:
            m += 1
            continue
        t = c[:]
        for j in range(N + 1 - m):
            c[j + m] ^= b[j]
        if 2 * L <= i:
            L, b, m = i + 1 - L, t, 1
        else:
            m += 1
    return c[:L + 1], L


class DegenerateKey(ValueError):
    pass


def min_poly(g: int, f0: int) -> int:
    n = deg(f0)
    seq, y = [], 1
    for _ in range(2 * n):
        seq.append(y & 1)
        y = mulmod(y, g, f0)
    c, L = berlekamp_massey(seq)
    return sum(int(c[j]) << (L - j) for j in range(L + 1))


def poly_from_key(z: int, n: int) -> int:
    """f(Z): the monic irreducible polynomial of degree n determined by the n-bit key Z."""
    P = min_poly(z, default_modulus(n))
    if deg(P) != n:
        raise DegenerateKey(f"key in a proper subfield (degree {deg(P)} < {n})")
    return P


def bits_to_int(bits) -> int:
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v


def int_to_bits(v: int, n: int):
    import numpy as np
    return np.array([(v >> (n - 1 - i)) & 1 for i in range(n)], dtype=np.uint8)


def message_poly(bits) -> int:
    """M(x) = sum_j M_j x^j (bit j of the int = M_j)."""
    v = 0
    for j, b in enumerate(bits):
        if b:
            v |= 1 << j
    return v
