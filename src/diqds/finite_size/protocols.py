"""The key-distribution protocols compared in the article.

All use the CHSH test (Alice x in {0,1}; Bob's test settings y in {0,1} relabelled 2, 3 below when they must be told
apart from his key setting) and differ in how generation rounds are run and who decides which round is a test round.

TWO_SETTING   random key measurements with sifting: Tan et al., Quantum 6, 880 (2022), Protocol 1, the protocol of
              Fan et al., Phys. Rev. A 114, 032617 (2026), Sec. II A.  Alice draws X uniformly in EVERY round; Bob alone
              decides T ~ Bernoulli(gamma) and draws Y in {0,1} (generation) or {2,3} (test).  Generation rounds with
              X != Y are discarded (sifting factor 1/2).  No shared randomness, so it also works when inputs must be drawn
              per attempt (photonic links).  The key comes from both of Alice's settings.
ONE_SETTING   fixed key setting with spot-checking (Hahn et al., arXiv:2507.07365; the protocol of Nadlinger et al.,
              Nature 607, 682 (2022)).  A common test flag T; generation rounds use the fixed pair (X, Y) = (0, 0)
              [Bob's y = 0 is aligned with Alice's x = 0]; test rounds draw X, Y uniformly.  No sifting.  T must be known
              to both parties before they measure and hidden from the devices and Eve until then: after a herald Alice
              can send it to Bob (an extra memory wait of L/c), but on a per-attempt (photonic) link it would have to be
              pre-shared secret randomness for every attempt.
INDEPENDENT   independent test flags, the protocol run by Lu et al., Science 391, 592 (2026) (supplement, Sec. 12.2):
              Alice tests with probability gamma_A (then x uniform in {0,1}, else x = 0); Bob tests with probability
              gamma_B (then y uniform in {0,1}, else his key setting).  Test rounds: both test (gamma = gamma_A gamma_B,
              inputs uniform).  Key rounds: Bob's key setting and x = 0, probability (1 - gamma_B)(1 - gamma_A/2); all
              other rounds are discarded.  For a given gamma the key fraction is largest at gamma_B = sqrt(gamma/2),
              gamma_A = sqrt(2 gamma) (Lu et al.'s gamma_A = 0.26, gamma_B = 0.13 is this optimum for gamma = 0.0338), so a
              non-test round is kept with probability s = (1 - sqrt(gamma/2))^2 / (1 - gamma).  No flag is exchanged, so
              there is no extra memory wait and the protocol also runs per attempt.
SEEDED        pre-shared seed with recovery (Tan et al. Protocol 2).  Alice's input X_j in every round is taken from an
              n-bit uniform pre-shared secret key; Bob decides T and, in generation rounds, sets Y = X (no sifting).
              X is never announced and privacy amplification is applied to (A, X), which "recovers" the seed's n bits:
              ell_out = n + ell_net, and the net gain ell_out - n is what the protocol adds to the key pool.

Generation-round Renyi rate (Lemma 1): F_s = 1/(1-alpha) log2(s 2^{(1-alpha) F} + 1 - s), with s the probability that
a non-test round is kept.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .entropy import sifted_rate


@dataclass(frozen=True)
class Protocol:
    key: str
    name: str
    sift: float                    # fraction of generation rounds that become raw-key bits (fixed-sifting protocols)
    common_test_flag: bool         # True: both parties must know T before measuring
    seed_bits_per_round: float     # pre-shared secret seed consumed per round (recovered by PA for SEEDED)
    short: str
    key_bases: int = 1             # 2: key bits come from both of Alice's settings (Z and X), so EC sees Q and Q_x
    independent_flags: bool = False

    def sift_at(self, gamma: Optional[float]) -> float:
        """Probability that a non-test round becomes a raw-key bit (depends on gamma for independent flags)."""
        if self.independent_flags:
            if gamma is None:
                raise ValueError("independent test flags: the sifting factor depends on gamma")
            gb = math.sqrt(gamma / 2.0)
            return (1.0 - gb) ** 2 / (1.0 - gamma)
        return self.sift

    def flag_probabilities(self, gamma: float) -> tuple:
        """(gamma_A, gamma_B) for independent flags; (gamma, gamma) otherwise (one common or one local flag)."""
        if self.independent_flags:
            return math.sqrt(2.0 * gamma), math.sqrt(gamma / 2.0)
        return gamma, gamma

    def gen_rate(self, F, alpha: float, gamma: Optional[float] = None):
        """Generation-round Renyi rate given the single-setting rate F (Lemma 1)."""
        s = self.sift_at(gamma)
        return sifted_rate(F, alpha, s) if s < 1.0 else np.asarray(F, dtype=float)

    def key_rounds_fraction(self, gamma: float) -> float:
        return (1.0 - gamma) * self.sift_at(gamma)

    def pa_input_bits(self, n: float, gamma: float) -> float:
        """Length of the privacy-amplification input (raw key; plus the seed X for SEEDED)."""
        return n * self.key_rounds_fraction(gamma) + (n * self.seed_bits_per_round)

    def output_bits(self, n: float, ell_net: float) -> float:
        """Length of the privacy-amplification output (net key plus the recovered seed for SEEDED)."""
        return ell_net + n * self.seed_bits_per_round


TWO_SETTING = Protocol("two_setting", "two key settings, sifted (Tan et al. Protocol 1; Fan et al.)", 0.5, False, 0.0,
                       "2-setting sifted", key_bases=2)
ONE_SETTING = Protocol("one_setting", "one key setting, common test flag sent after the herald (Hahn et al.; Nadlinger et al.)",
                       1.0, True, 0.0, "1-setting spot-check")
INDEPENDENT = Protocol("independent", "one key setting, independent test flags (Lu et al.)", 1.0, False, 0.0,
                       "independent flags", independent_flags=True)
SEEDED = Protocol("seeded", "pre-shared seed with recovery (Tan et al. Protocol 2)", 1.0, False, 1.0, "pre-shared seed",
                  key_bases=2)

LU_GAMMA_A, LU_GAMMA_B = 0.26, 0.13                    # Lu et al., supplement Eq. (S45)
LU_GAMMA = LU_GAMMA_A * LU_GAMMA_B                     # test fraction 0.0338

PROTOCOLS = {p.key: p for p in (TWO_SETTING, ONE_SETTING, INDEPENDENT, SEEDED)}
