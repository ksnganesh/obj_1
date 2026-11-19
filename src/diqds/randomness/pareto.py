"""Randomness-optimal protocol choice: re-optimise (gamma, q, alpha) for signatures per trusted random bit instead of
signatures per round, and trace the Pareto front of block size (rounds to the first signature) against trusted bits per
signature, for per-herald (ions, atoms) and per-attempt (photonic) platforms.

For a block of n heralded rounds the first signature is available at the end of the block, so n (or n / herald rate) is
the latency.  Larger blocks amortise the per-block costs and push bits per signature down: every point on a protocol's
curve is Pareto-optimal for that protocol, and the platform's front is the lower envelope over protocols.

Currencies (ledger.py): "private" = private input bits only (the DI-critical resource);
"trusted" = private input bits + the public-but-fresh seeds (privacy amplification, error verification).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ..common import KEY_BITS_PER_SIGNATURE
from ..finite_size.optimize import n_min as renyi_n_min, renyi_optimum
from ..finite_size.protocols import Protocol
from ..finite_size.renyi import RenyiResult, RenyiSettings
from .ledger import block_ledger
from .platforms import Platform


def bits_for(result: RenyiResult, protocol: Protocol, platform: Platform, currency: str, sampler: str = "entropy",
             pa_family: str = "toeplitz") -> float:
    led = block_ledger(protocol, result.n, result.gamma, result.q, max(result.ell, 0.0),
                       attempts_per_round=platform.attempts_per_herald if platform.per_attempt_inputs else 1.0,
                       per_attempt=platform.per_attempt_inputs, sampler=sampler, pa_family=pa_family)
    return led.private_input + (led.public_fresh if currency == "trusted" else 0.0)


def bits_per_signature_objective(protocol: Protocol, platform: Platform, currency: str, sampler: str = "entropy",
                                 pa_family: str = "toeplitz"):
    """Objective for renyi_optimum: -log10(bits per signature) when a signature exists, a feasibility ramp otherwise."""
    def obj(r: RenyiResult) -> float:
        if not np.isfinite(r.ell):
            return -1e9
        sigs = r.ell / KEY_BITS_PER_SIGNATURE
        if sigs < 1.0:
            return -30.0 + sigs
        return -math.log10(bits_for(r, protocol, platform, currency, sampler, pa_family) / sigs)
    return obj


@dataclass
class FrontPoint:
    n: float
    hours: float
    signatures: float
    bits_per_signature: float
    gamma: float
    q: float
    alpha: float
    ell: float


@dataclass
class Front:
    platform: str
    protocol: str
    currency: str
    n_first: float
    points: List[FrontPoint] = field(default_factory=list)
    key_max_points: List[FrontPoint] = field(default_factory=list)   # same n, parameters chosen for key length


def protocol_front(platform: Platform, protocol: Protocol, currency: str = "private", n_values: Optional[List[float]] = None,
                   decades: float = 2.0, npts: int = 6, settings: RenyiSettings = RenyiSettings(),
                   sampler: str = "entropy", pa_family: str = "toeplitz", n_first: Optional[float] = None) -> Front:
    hon = platform.honest
    if n_first is None:
        n_first = renyi_n_min(hon, protocol, q="opt", settings=settings)
    fr = Front(platform.key, protocol.key, currency, n_first)
    if not math.isfinite(n_first):
        return fr
    if n_values is None:
        n_values = list(np.geomspace(n_first * 1.02, n_first * 10 ** decades, npts))
    obj = bits_per_signature_objective(protocol, platform, currency, sampler, pa_family)
    for n in n_values:
        r = renyi_optimum(n, hon, protocol, q="opt", settings=settings, objective=obj, starts=3)
        k = renyi_optimum(n, hon, protocol, q="opt", settings=settings)
        for res, lst in ((r, fr.points), (k, fr.key_max_points)):
            sigs = max(res.ell, 0.0) / KEY_BITS_PER_SIGNATURE
            bps = bits_for(res, protocol, platform, currency, sampler, pa_family) / sigs if sigs >= 1 else math.inf
            lst.append(FrontPoint(n, platform.hours_for(n), sigs, bps, res.gamma, res.q, res.alpha, res.ell))
    return fr


def envelope(fronts: List[Front]) -> List[Dict]:
    """Lower envelope (minimum bits per signature) over protocols, on the union of the block sizes."""
    pts = sorted(((p.n, p.bits_per_signature, f.protocol) for f in fronts for p in f.points), key=lambda t: t[0])
    out, best = [], math.inf
    for n, b, proto in pts:
        if b < best:
            best = b
            out.append({"n": n, "bits_per_signature": b, "protocol": proto})
    return out
