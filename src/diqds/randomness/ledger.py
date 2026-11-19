"""Trusted-randomness ledger of DI-QDS: what each protocol consumes, in which currency, per round, block and signature.

DI security assumes that the parties hold random bits the devices and Eve cannot predict (Zapatero et al., npj QI 9, 10
(2023)).  Fan et al. (2026) draw them freely and never count them.  We distinguish four currencies:

    private_input   bits that choose measurement inputs, test flags or preprocessing flips.  Must be unpredictable to the
                    devices and to Eve until the measurement (flips: until privacy amplification).  Local trusted RNG.
    shared_secret   bits both parties must hold before the round and keep secret: the common test flag of ONE_SETTING when
                    it cannot be sent after a herald (per-attempt links), the seed of SEEDED.
    authentication  pre-shared secret key for Wegman-Carter tags (Wegman and Carter 1981) with key recycling (Portmann,
                    IEEE TIT 60, 4383 (2014)): the hash key is reused and every message consumes a fresh 64-bit one-time
                    pad.  64-bit tags from polynomial hashing over GF(2^64) fail with probability <= L / 2^64 <= 2^-44 for
                    messages of up to L = 2^20 words, within the eps' = 1e-11 per message of Fan et al.'s Table I.  Two
                    messages per block and link (Alice -> Bob/Charlie, back), three per signature between Bob and Charlie
                    ({Sig, Doc, Y_b}; {X_c, Y_c, Z_c}; the verdict).  The link tags can be paid from the link's own key
                    once it runs; the Bob-Charlie key must be pre-shared or come from a separate key-distribution link
                    between the recipients, which we count but do not model.
    public_fresh    bits that must be uniform and independent of the raw data but may be public once the quantum phase
                    is over: the privacy-amplification seed, the error-verification hash seed.
    (signature key) 3 n_s = 138 secret key bits per link and signature (Eq. 4) - the output, not a cost.

Biased bits (the test flag T ~ Bernoulli(gamma), flips ~ Bernoulli(q)) cost h(p) uniform bits with an entropy-efficient
sampler (arithmetic decoding), or a fixed number of bits with a dyadic sampler; Nadlinger et al. and Hahn et al. use gamma
in units of 1/256, i.e. 8 bits per round.  Both are reported ("entropy" and "dyadic8").

Per-herald vs per-attempt: memory platforms (ions, atoms) choose inputs after the herald, so inputs are drawn once per
heralded round.  The photonic link of Fan et al. (Fig. 1b) measures the local photon before the herald arrives, so
inputs are drawn for every attempt, i.e. (attempts per herald) times more.

First-signature cost: the first signature pays for its whole block, so the ledger reports block totals at the threshold
block (``first_block``) next to per-signature values at larger blocks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

from ..common import KEY_BITS_PER_SIGNATURE, h2
from ..finite_size.protocols import INDEPENDENT, ONE_SETTING, SEEDED, TWO_SETTING, Protocol

TAG_BITS = 64                          # Wegman-Carter tag, eps_auth ~ 2^-61 per message
EV_SEED_BITS = 128                     # seed of a 64-bit delta-almost-universal hash (delta = 2^-61)
MESSAGES_PER_BLOCK = 2                 # authenticated transcripts per link and block (Alice -> Bob, Bob -> Alice)
MESSAGES_PER_SIGNATURE = 3             # Bob -> Charlie {Sig, Doc, Y_b}; Charlie -> Bob {X_c, Y_c, Z_c}; Bob -> Charlie verdict


def biased_cost(p: float, sampler: str = "entropy") -> float:
    """Uniform bits needed to draw one Bernoulli(p) bit."""
    if p <= 0.0:
        return 0.0
    if sampler == "entropy":
        return float(h2(p))
    if sampler == "dyadic8":
        return 8.0
    raise ValueError(sampler)


@dataclass
class RoundCost:
    """Trusted random bits per (heralded or attempted) round, split by party and currency."""
    alice: float = 0.0
    bob: float = 0.0
    shared_secret: float = 0.0
    flips: float = 0.0                 # Alice's noisy-preprocessing flips (private, drawn after the measurement)

    @property
    def private_input(self) -> float:
        return self.alice + self.bob + self.flips

    def scaled(self, k: float) -> "RoundCost":
        return RoundCost(self.alice * k, self.bob * k, self.shared_secret * k, self.flips * k)


def per_round(protocol: Protocol, gamma: float, q: float = 0.0, sampler: str = "entropy",
              per_attempt: bool = False) -> RoundCost:
    """Inputs drawn per round.  ``per_attempt``: inputs must be chosen before the herald (photonic links)."""
    cg, cq = biased_cost(gamma, sampler), biased_cost(q, sampler)
    if protocol.key == TWO_SETTING.key:
        # Alice: X uniform every round.  Bob: T and Y (in {0,1} or {2,3}).  Flips only on sifted key rounds.
        return RoundCost(alice=1.0, bob=cg + 1.0, flips=0.5 * (1.0 - gamma) * cq)
    if protocol.key == ONE_SETTING.key:
        # Common T: drawn by Alice and sent to Bob after the herald (private until both have measured).
        # Per attempt there is no herald yet: T must be pre-shared secret randomness for every attempt.
        rc = RoundCost(alice=gamma, bob=gamma, flips=(1.0 - gamma) * cq)
        if per_attempt:
            rc.shared_secret = cg
        else:
            rc.alice += cg
        return rc
    if protocol.key == INDEPENDENT.key:
        # Each party draws its own flag (Bernoulli(gamma_A), Bernoulli(gamma_B)) and a uniform test input when testing.
        # No flag is exchanged, so the same costs apply per attempt, with no pre-shared secret.
        ga, gb = protocol.flag_probabilities(gamma)
        return RoundCost(alice=biased_cost(ga, sampler) + ga, bob=biased_cost(gb, sampler) + gb,
                         flips=(1.0 - gamma) * protocol.sift_at(gamma) * cq)
    if protocol.key == SEEDED.key:
        # Bob: T and his test setting.  Alice's input comes from the pre-shared seed (1 secret bit per round).
        return RoundCost(alice=0.0, bob=cg + gamma, shared_secret=1.0, flips=(1.0 - gamma) * cq)
    raise ValueError(protocol.key)


def pa_seed_bits(n_in: float, n_out: float, family: str = "toeplitz") -> float:
    """Seed of the privacy-amplification hash: Toeplitz n_in + n_out - 1, modified Toeplitz [T | I] n_in - 1."""
    if n_out <= 0:
        return 0.0
    if family == "toeplitz":
        return n_in + n_out - 1.0
    if family == "modified_toeplitz":
        return n_in - 1.0
    raise ValueError(family)


@dataclass
class BlockLedger:
    protocol: str
    n_rounds: float                    # heralded rounds in the block
    attempts: float                    # attempts in the block (= n_rounds for per-herald platforms)
    ell_net: float
    signatures: float
    private_input: float
    shared_secret: float
    shared_secret_recovered: float     # part of shared_secret refilled from the output (SEEDED seed)
    public_fresh: float
    auth_secret: float                 # per-block authentication keys (both links counted separately by the caller)
    details: Dict[str, float] = field(default_factory=dict)

    def per_signature(self) -> Dict[str, float]:
        s = max(self.signatures, 1e-300)
        sig_auth = MESSAGES_PER_SIGNATURE * TAG_BITS
        return {
            "private_input": self.private_input / s,
            "public_fresh": self.public_fresh / s,
            "shared_secret_net": (self.shared_secret - self.shared_secret_recovered) / s,
            "auth_secret_link": self.auth_secret / s,
            "auth_secret_bob_charlie": float(sig_auth),
            "auth_total": self.auth_secret / s + float(sig_auth),
            "trusted_total": (self.private_input + self.public_fresh) / s,
        }

    def block_totals(self) -> Dict[str, float]:
        """What the block costs in total: the cost of the FIRST signature when the block is the threshold block."""
        return {
            "private_input": self.private_input,
            "public_fresh": self.public_fresh,
            "shared_secret_gross": self.shared_secret,
            "shared_secret_net": self.shared_secret - self.shared_secret_recovered,
            "auth_secret_link": self.auth_secret,
            "auth_secret_bob_charlie_first_signature": float(MESSAGES_PER_SIGNATURE * TAG_BITS),
            "signatures": self.signatures,
        }


def block_ledger(protocol: Protocol, n: float, gamma: float, q: float, ell_net: float, attempts_per_round: float = 1.0,
                 per_attempt: bool = False, sampler: str = "entropy", pa_family: str = "toeplitz") -> BlockLedger:
    """Ledger of ONE link for a block of n heralded rounds producing ell_net net key bits."""
    attempts = n * attempts_per_round
    rc = per_round(protocol, gamma, q, sampler, per_attempt)
    # Inputs are drawn per attempt on per-attempt links; preprocessing flips are drawn per kept key bit regardless.
    k = attempts if per_attempt else n
    private = (rc.alice + rc.bob) * k + rc.flips * n
    shared = rc.shared_secret * k                             # SEEDED per attempt: one seed bit per attempt
    recovered = n * protocol.seed_bits_per_round if ell_net > 0 else 0.0   # only heralded rounds' seed bits
    n_in = protocol.pa_input_bits(n, gamma)
    n_out = protocol.output_bits(n, max(ell_net, 0.0))
    seed = pa_seed_bits(n_in, n_out, pa_family)
    public = seed + EV_SEED_BITS
    sigs = max(ell_net, 0.0) / KEY_BITS_PER_SIGNATURE
    return BlockLedger(protocol.key, n, attempts, ell_net, sigs, private, shared, recovered, public,
                       MESSAGES_PER_BLOCK * TAG_BITS,
                       {"alice": rc.alice * k, "bob": rc.bob * k, "flips": rc.flips * n, "pa_seed": seed,
                        "ev_seed": EV_SEED_BITS, "pa_input_bits": n_in, "pa_output_bits": n_out})


def first_signature_feasible(protocol: Protocol, per_attempt: bool) -> str:
    """Qualitative verdict on whether the protocol can run on the platform without per-attempt pre-shared secrets."""
    if not per_attempt:
        return "yes"
    if protocol.key == TWO_SETTING.key:
        return "yes (Bob decides T locally)"
    if protocol.key == INDEPENDENT.key:
        return "yes (each party draws its own flag)"
    if protocol.key == ONE_SETTING.key:
        return "needs h(gamma) pre-shared secret bits per attempt"
    return "needs 1 pre-shared secret seed bit per attempt; only heralded ones are recovered"
