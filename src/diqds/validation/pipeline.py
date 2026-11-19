"""End-to-end emulation of DI-QDS for the Monte Carlo checks of the emulator.

The protocol itself is in diqds.protocol:
    distribution stage  protocol/distribution.py   the A-B and A-C key-generation protocols (steps 1-7 of Sec. II A)
    messaging stage     protocol/messaging.py      signing, verification, certification (Sec. II B), forgery attempts
This module runs both stages and counts what the validation needs: aborts and their reasons, signatures per block, honest
signatures accepted by both recipients, and forgeries that Charlie accepts.  The forger is a dishonest Bob who either
knows Charlie's keys for some segments (``eve_known``: the window attack without privacy amplification) or guesses.
The names plan_link / LinkPlan / LinkRun / run_link are used by the privacy-amplification analysis and the emulator.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from ..common import KEY_BITS_PER_SIGNATURE, N_S
from ..protocol.distribution import (DistributionResult, LinkPlan, LinkRun, choose_inputs,  # noqa: F401
                                     distribution_stage, plan_link, run_kgp, sifting)
from ..protocol.messaging import forgery_attempt, key_sets, messaging_stage
from .otuh import cut

run_link = run_kgp          # one key-generation protocol (one link of the distribution stage)


def _inputs(protocol, n: int, gamma: float, rng: np.random.Generator):
    """Inputs of the n rounds and the key-round mask (steps 2 and 3 without the devices)."""
    X, Y, T = choose_inputs(protocol, n, gamma, rng)
    return X, Y, T, sifting(protocol, X, Y, T)


@dataclass
class QDSRun:
    ab: LinkRun
    ac: LinkRun
    signatures: int = 0
    honest_ok: int = 0
    forged_tried: int = 0
    forged_accepted: int = 0
    targeted_segments: int = 0
    targeted_forged: int = 0

    @property
    def accepted(self) -> bool:
        return self.ab.accepted and self.ac.accepted


def run_qds(plan: LinkPlan, dev_ab, dev_ac, rng: np.random.Generator, pa: str = "toeplitz", n_docs: int = 10,
            n_m: int = 1000, eve_known: Optional[Dict[int, int]] = None, target_segments: Optional[List[int]] = None,
            sampling: str = "multinomial") -> QDSRun:
    """Distribution stage, then the messaging stage and one forgery attempt for each of the first n_docs segments.
    ``eve_known``: {raw-key position of K_AC: bit} that the dishonest Bob learned from the adversarial A-C devices
    (only meaningful with pa == "none"); ``target_segments``: segments he attacks with that knowledge."""
    dist = distribution_stage(plan, dev_ab, dev_ac, rng, pa, sampling)
    out = QDSRun(dist.ab, dist.ac)
    if not out.accepted:
        return out
    w = KEY_BITS_PER_SIGNATURE
    S = dist.signatures
    out.signatures = S
    docs = list(range(min(n_docs, S)))
    if target_segments:
        docs = sorted(set(docs) | set(s for s in target_segments if s < S))
    for j in docs:
        knows = None
        if eve_known is not None and target_segments and j in target_segments:
            bits = [eve_known.get(i) for i in range(j * w, (j + 1) * w)]
            if all(v is not None for v in bits):
                knows = cut(np.array(bits, np.uint8), N_S)
        doc = rng.integers(0, 2, n_m, dtype=np.uint8)
        alice, bob, charlie = key_sets(dist, j)
        rec = messaging_stage(doc, alice, bob, charlie, index=j)
        forged = forgery_attempt(rec, bob, charlie, rng, knows) if rec.sig is not None else None
        out.honest_ok += int(rec.accepted)
        out.forged_tried += 1
        out.forged_accepted += int(bool(forged and forged.accepted))
        if target_segments and j in target_segments:
            out.targeted_segments += 1
            out.targeted_forged += int(bool(forged and forged.accepted))
    return out


__all__ = ["DistributionResult", "LinkPlan", "LinkRun", "QDSRun", "plan_link", "run_link", "run_qds", "_inputs"]
