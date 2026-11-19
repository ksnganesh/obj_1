"""One DI-QDS session: the distribution stage followed by the messaging stage for a list of documents."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

from ..common import N_S
from .distribution import DistributionResult, LinkPlan, distribution_stage
from .messaging import Channel, SignatureRecord, key_sets, messaging_stage


@dataclass
class Session:
    distribution: DistributionResult
    records: List[SignatureRecord] = field(default_factory=list)
    channel: Optional[Channel] = None

    @property
    def signed(self) -> int:
        return len(self.records)

    @property
    def accepted(self) -> int:
        return sum(r.accepted for r in self.records)


def run_session(plan: LinkPlan, dev_ab, dev_ac, documents: Sequence[np.ndarray], rng: np.random.Generator,
                pa: str = "toeplitz", sampling: str = "multinomial", channel: Optional[Channel] = None,
                n_s: int = N_S) -> Session:
    """Distribution stage on (dev_ab, dev_ac), then one messaging stage per document, document j on key segment j.
    Documents beyond the number of signatures the keys pay for are not signed."""
    dist = distribution_stage(plan, dev_ab, dev_ac, rng, pa, sampling)
    ch = channel if channel is not None else Channel(n_s=n_s)
    out = Session(dist, [], ch)
    if not dist.accepted:
        return out
    for j, doc in enumerate(list(documents)[:dist.signatures]):
        alice, bob, charlie = key_sets(dist, j, n_s)
        out.records.append(messaging_stage(np.asarray(doc, np.uint8), alice, bob, charlie, ch, n_s, index=j))
    return out
