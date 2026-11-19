"""Distribution stage of DI-QDS (Fan et al., Phys. Rev. A 114, 032617 (2026), Sec. II A).

Alice-Bob and Alice-Charlie independently run a device-independent key-generation protocol (KGP), each pair on its
own devices (no device is shared between the two links, which closes the memory loophole).  One KGP follows the
steps of Sec. II A; the function implementing each step is on the right.

 1. protocol parameters    n, gamma, the acceptance test (Fan et al.'s omega_thr test, here the
                           acceptance set of the Renyi bound), noisy-preprocessing probability p,
                           EC_max and the key length ell_key                                     plan_link -> LinkPlan
 2. data acquisition       n heralded rounds: inputs X (Alice), Y (Bob), test flag T; outputs A, B   data_acquisition
 3. sifting                which rounds give key bits: generation rounds with X = Y in the base
                           paper's two-setting protocol (mismatched rounds are dropped, which is what
                           overwriting them with A = B = 0 amounts to); every generation round in the
                           one-setting and seeded protocols                                          sifting
 4. noisy preprocessing    Alice flips each key bit with probability p                               noisy_preprocessing
 5. error correction       Bob reconstructs Alice's test and key bits with Cascade (one run on the
                           test rounds, one per key basis); abort if the disclosed bits exceed EC_max error_correction
                           then a 64-bit two-universal hash tag of Alice's whole string             error_verification
 6. parameter estimation   Bob scores the test rounds with the reconstructed bits; continue only if
                           the tag matched and the (lost, won, generation) counts are accepted   parameter_estimation
 7. privacy amplification  Toeplitz (or modified Toeplitz) hash of the key bits to ell_key bits; the
                           seeded protocol hashes (A, X); the first n output bits refill the seed privacy_amplification

run_kgp chains steps 2-7 for one link.  distribution_stage runs the A-B link and then the A-C link and returns the
keys of all three parties (DistributionResult); the messaging stage (messaging.py) cuts them into signatures.

Privacy amplification hashes the key-round bits only: the key length (Hahn et al., Eq. D6) subtracts n (gamma +
delta_perp) for the test-round outputs, which Bob learns through error correction.  The random draws always come in
the same order (inputs, device outputs, flips, Cascade, tag seed, PA seed), so a seeded run is reproducible bit for bit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..common import KEY_BITS_PER_SIGNATURE, Honest
from ..finite_size.optimize import renyi_optimum
from ..finite_size.protocols import ONE_SETTING, SEEDED, TWO_SETTING, Protocol
from ..finite_size.renyi import AcceptanceSet, RenyiSettings, acceptance_set, ec_bits, key_length
from ..randomness.ledger import biased_cost
from ..privacy_amplification.seeds import modified_toeplitz_hash, toeplitz_hash
from ..validation.cascade import cascade

EV_BITS = 64                      # error-verification tag (collision probability 2^-61 with the VHASH-style bound)
COUNTS = ("loss", "win", "perp")  # test register of the acceptance test: lost / won test rounds, generation rounds


# ------------------------------------------------------------------------------------ step 1: protocol parameters
@dataclass
class LinkPlan:
    """Step 1: the parameters of one KGP (the same for the A-B and the A-C link)."""
    protocol: Protocol
    honest: Honest              # expected (S, Q) of honest devices: sets EC_max and the acceptance set
    n: int                      # heralded rounds
    gamma: float                # testing probability
    alpha: float                # Renyi order of the bound
    q: float                    # noisy-preprocessing probability (p in Fan et al.)
    ell: int                    # key length ell_key (net of the recovered seed for SEEDED)
    acceptance: AcceptanceSet   # acceptance test on (lost, won, generation) counts
    ec_budget: float            # EC_max: bits Alice may disclose during error correction
    settings: RenyiSettings

    @property
    def ell_out(self) -> int:
        """Length of the privacy-amplification output (ell_key, plus the n recovered seed bits for SEEDED)."""
        return self.ell + (self.n if self.protocol.key == SEEDED.key else 0)


def plan_link(n: int, honest: Honest, protocol: Protocol, q="opt", settings: RenyiSettings = RenyiSettings(),
              gamma: Optional[float] = None, alpha: Optional[float] = None) -> LinkPlan:
    """Choose gamma, alpha and p by maximising the Renyi key length (or use the given gamma and alpha)."""
    if gamma is None or alpha is None:
        r = renyi_optimum(n, honest, protocol, q, settings)
    else:
        r = key_length(n, honest, protocol, gamma, alpha, 0.0 if q == "opt" else float(q), settings)
    acc = acceptance_set(n, r.gamma, honest.w, settings)
    return LinkPlan(protocol, honest, int(n), r.gamma, r.alpha, r.q, int(math.floor(max(r.ell, 0.0))), acc,
                    ec_bits(n, honest, protocol, r.gamma, r.q, settings), settings)


# ---------------------------------------------------------------------------------------- result of one KGP
@dataclass
class LinkRun:
    """Outcome of one KGP.  For the A-C link ``key_bob`` is Charlie's key."""
    accepted: bool
    reason: str                                       # why the KGP aborted ("" if it did not)
    counts: tuple                                     # (lost, won, generation) rounds, step 6
    leak: int                                         # bits disclosed by error correction, step 5
    ec_budget: float                                  # EC_max
    qber_key: float                                   # error rate between Alice's key bits and Bob's raw bits
    win_freq: float                                   # CHSH winning frequency on the test rounds
    key_alice: Optional[np.ndarray] = None            # final keys, step 7
    key_bob: Optional[np.ndarray] = None
    raw_alice: Optional[np.ndarray] = None            # key bits before privacy amplification
    raw_bob: Optional[np.ndarray] = None
    key_rounds: Optional[np.ndarray] = None           # round index of every raw key bit (public after the protocol)
    test_rounds: Optional[np.ndarray] = None
    flips: Optional[np.ndarray] = None                # noisy-preprocessing flips, step 4
    residual_errors: int = 0                          # errors left after Cascade (caught by the tag, or not)
    input_bits: float = 0.0                           # private random bits spent on inputs and flips (randomness ledger)
    pa_seed_bits: int = 0
    pe_fail: List[str] = field(default_factory=list)
    steps: Dict[str, dict] = field(default_factory=dict)   # what each step did (numbers only), for transcripts


# ----------------------------------------------------------------------------------- step 2: data acquisition
@dataclass
class RawData:
    """Step 2 output: inputs and outputs of all n rounds (kept in each party's private storage)."""
    X: np.ndarray        # Alice's inputs, {0, 1}
    Y: np.ndarray        # Bob's inputs: {0, 1} generation, {2, 3} CHSH test
    T: np.ndarray        # test flag
    A: np.ndarray        # Alice's outputs
    B: np.ndarray        # Bob's outputs


def choose_inputs(protocol: Protocol, n: int, gamma: float, rng: np.random.Generator):
    """Inputs of the n rounds.
    TWO_SETTING  Alice X ~ U{0,1} in every round; Bob T ~ Bern(gamma), Y ~ U{0,1} (generation) or U{2,3} (test).
    ONE_SETTING  common test flag T; generation rounds (X, Y) = (0, 0); test rounds X ~ U{0,1}, Y ~ U{2,3}.
    SEEDED       X from the pre-shared seed in every round; Bob T; generation Y = X; test Y ~ U{2,3}."""
    T = rng.random(n) < gamma
    u1 = rng.integers(0, 2, n).astype(np.int8)
    u2 = rng.integers(0, 2, n).astype(np.int8)
    if protocol.key == TWO_SETTING.key:
        X, Y = u1, np.where(T, 2 + u2, rng.integers(0, 2, n)).astype(np.int8)
    elif protocol.key == ONE_SETTING.key:
        X = np.where(T, u1, 0).astype(np.int8)
        Y = np.where(T, 2 + u2, 0).astype(np.int8)
    elif protocol.key == SEEDED.key:
        X = u1
        Y = np.where(T, 2 + u2, X).astype(np.int8)
    else:
        raise ValueError(protocol.key)
    return X, Y, T


def data_acquisition(plan: LinkPlan, device, rng: np.random.Generator, sampling: str = "multinomial") -> RawData:
    """Step 2: choose the inputs and let the devices (honest Aer devices or adversarial ones) produce the outputs."""
    X, Y, T = choose_inputs(plan.protocol, plan.n, plan.gamma, rng)
    a, b = device.sample(X, Y, rng, sampling)
    return RawData(X, Y, T, a.astype(np.uint8), b.astype(np.uint8))


# ---------------------------------------------------------------------------------------------- step 3: sifting
def sifting(protocol: Protocol, X: np.ndarray, Y: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Step 3: boolean mask of the key rounds, computed from the announced inputs."""
    if protocol.key == TWO_SETTING.key:
        return (~T) & (X == Y)
    return ~T


# ---------------------------------------------------------------------------------- step 4: noisy preprocessing
def noisy_preprocessing(A_raw_key: np.ndarray, p: float, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Step 4: Alice flips each key bit with probability p.  Returns (flipped key bits, flips)."""
    flips = (rng.random(A_raw_key.size) < p).astype(np.uint8)
    return A_raw_key ^ flips, flips


# ------------------------------------------------------------------------------------- step 5: error correction
def chsh_predicate(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """x (y - 2): a test round is won iff A xor B equals it (Y in {2, 3}); 0 on generation rounds."""
    yp = (Y.astype(np.int16) - 2).clip(0, 1).astype(np.uint8)
    return X.astype(np.uint8) & yp


@dataclass
class Reconciled:
    """Step 5 output: Alice's string (test bits, key bits) and Bob's reconstruction of it."""
    A_test: np.ndarray
    A_key: np.ndarray
    B_test: np.ndarray       # Bob's estimate of A_test (from B xor x (y - 2), corrected)
    B_key: np.ndarray        # Bob's estimate of A_key (from B, corrected)
    leak: int                # parity bits disclosed

    @property
    def alice(self) -> np.ndarray:
        return np.concatenate([self.A_test, self.A_key])

    @property
    def bob(self) -> np.ndarray:
        return np.concatenate([self.B_test, self.B_key])


def error_correction(plan: LinkPlan, data: RawData, key: np.ndarray, A_key: np.ndarray, rng: np.random.Generator,
                     cascade_passes: int = 4) -> Reconciled:
    """Step 5a: Cascade on the test rounds (Bob's guess b xor x(y-2), error 1 - w) and on the key rounds of each key
    basis (Bob's guess b, error Q or Q_x after preprocessing)."""
    test = data.T
    xy = chsh_predicate(data.X, data.Y)
    A_test = data.A[test]
    guess_test = data.B[test] ^ xy[test]
    corr_test, leak_test = cascade(A_test, guess_test, max(1e-3, 1.0 - plan.honest.w), rng, passes=cascade_passes)
    flipped = lambda Q: Q + plan.q - 2 * plan.q * Q                       # error rate after preprocessing
    corr_key = np.empty_like(A_key)
    leak_key = 0
    B_key_raw = data.B[key]
    xk = data.X[key]
    for basis, Qb in ((0, plan.honest.Q), (1, plan.honest.Q_x)):         # one Cascade run per key basis
        sel = np.flatnonzero(xk == basis)
        if sel.size:
            c, l = cascade(A_key[sel], B_key_raw[sel], max(1e-3, flipped(Qb)), rng, passes=cascade_passes)
            corr_key[sel] = c
            leak_key += l
    return Reconciled(A_test, A_key, corr_test, corr_key, leak_test + leak_key)


def error_verification(rec: Reconciled, rng: np.random.Generator, ev_bits: int = EV_BITS) -> bool:
    """Step 5b: Alice sends a Toeplitz hash tag of her whole string with a fresh public seed; Bob compares it."""
    full_A, full_B = rec.alice, rec.bob
    tag_seed = rng.integers(0, 2, full_A.size + ev_bits - 1, dtype=np.uint8)
    return bool(np.array_equal(toeplitz_hash(full_A, tag_seed, ev_bits), toeplitz_hash(full_B, tag_seed, ev_bits)))


# ---------------------------------------------------------------------------------- step 6: parameter estimation
def parameter_estimation(plan: LinkPlan, data: RawData, rec: Reconciled) -> Tuple[tuple, float, List[str]]:
    """Step 6: Bob scores the test rounds with his reconstruction of Alice's test bits.
    Returns ((lost, won, generation), winning frequency, list of violated bounds of the acceptance set)."""
    test = data.T
    xy = chsh_predicate(data.X, data.Y)
    wins = ((rec.B_test ^ data.B[test]) == xy[test])
    k = (int(test.sum() - wins.sum()), int(wins.sum()), int((~data.T).sum()))
    fail = []
    for i, name in enumerate(COUNTS):
        if k[i] < plan.acceptance.k_lo[i]:
            fail.append(f"{name} low")
        if k[i] > plan.acceptance.k_up[i]:
            fail.append(f"{name} high")
    return k, (float(wins.mean()) if wins.size else 0.0), fail


# ------------------------------------------------------------------------------- step 7: privacy amplification
def privacy_amplification(plan: LinkPlan, data: RawData, rec: Reconciled, rng: np.random.Generator,
                          pa: str = "toeplitz") -> Tuple[np.ndarray, np.ndarray, int]:
    """Step 7: hash the key bits (SEEDED: key bits and the seed X) to ell_out bits with a fresh public seed.
    pa = "toeplitz" | "modified" (Hayashi-Tsurumaru, seed n - 1 bits) | "none" (the insecure variant of Sec. VI).
    Returns (Alice's key, Bob's key, seed bits)."""
    if pa == "none":
        return rec.A_key.copy(), rec.B_key.copy(), 0
    if plan.protocol.key == SEEDED.key:
        in_a = np.concatenate([rec.A_key, data.X.astype(np.uint8)])
        in_b = np.concatenate([rec.B_key, data.X.astype(np.uint8)])
    else:
        in_a, in_b = rec.A_key, rec.B_key
    out = plan.ell_out
    if pa == "toeplitz":
        seed = rng.integers(0, 2, in_a.size + out - 1, dtype=np.uint8)
        ka, kb = toeplitz_hash(in_a, seed, out), toeplitz_hash(in_b, seed, out)
    elif pa == "modified":
        seed = rng.integers(0, 2, in_a.size - 1, dtype=np.uint8)
        ka, kb = modified_toeplitz_hash(in_a, seed, out), modified_toeplitz_hash(in_b, seed, out)
    else:
        raise ValueError(pa)
    if plan.protocol.key == SEEDED.key:            # the first n bits refill the seed; the rest is the net key
        ka, kb = ka[plan.n:], kb[plan.n:]
    return ka, kb, int(seed.size)


# ------------------------------------------------------------------------------------------- one complete KGP
def run_kgp(plan: LinkPlan, device, rng: np.random.Generator, pa: str = "toeplitz", sampling: str = "multinomial",
            ev_bits: int = EV_BITS, cascade_passes: int = 4) -> LinkRun:
    """Steps 2-7 of one key-generation protocol.  Aborts set ``reason`` and return early."""
    n, g, proto = plan.n, plan.gamma, plan.protocol
    data = data_acquisition(plan, device, rng, sampling)                                    # 2
    key = sifting(proto, data.X, data.Y, data.T)                                             # 3
    A_key, flips = noisy_preprocessing(data.A[key], plan.q, rng)                             # 4
    rec = error_correction(plan, data, key, A_key, rng, cascade_passes)                      # 5a
    residual = int(np.count_nonzero(rec.alice != rec.bob))
    inbits = n * biased_cost(g) + 2 * float(data.T.sum()) + int(key.sum()) * biased_cost(plan.q)
    run = LinkRun(False, "", (0, 0, 0), rec.leak, plan.ec_budget,
                  float(np.mean(A_key != data.B[key])) if key.any() else 0.0, 0.0,
                  key_rounds=np.flatnonzero(key), test_rounds=np.flatnonzero(data.T), flips=flips,
                  residual_errors=residual, input_bits=inbits)
    run.steps = {
        "1 parameters": {"protocol": proto.key, "n": n, "gamma": g, "alpha": plan.alpha, "p": plan.q,
                         "EC_max": plan.ec_budget, "ell_key": plan.ell,
                         "acceptance_lo": list(map(int, plan.acceptance.k_lo)),
                         "acceptance_up": list(map(int, plan.acceptance.k_up))},
        "2 data acquisition": {"rounds": n, "test_rounds": int(data.T.sum()),
                               "generation_rounds": int((~data.T).sum())},
        "3 sifting": {"key_rounds": int(key.sum()),
                      "discarded_generation_rounds": int((~data.T).sum() - key.sum())},
        "4 noisy preprocessing": {"flipped_bits": int(flips.sum()), "qber_key": run.qber_key},
        "5 error correction": {"bits_disclosed": int(rec.leak), "EC_max": plan.ec_budget,
                               "within_EC_max": bool(rec.leak <= plan.ec_budget),
                               "errors_left_after_cascade": residual},
    }
    if rec.leak > plan.ec_budget:
        run.reason = "error-correction budget"
        return run
    tag_ok = error_verification(rec, rng, ev_bits)                                           # 5b
    run.steps["5 error correction"].update({"tag_bits": ev_bits, "tag_match": tag_ok})
    if not tag_ok:
        run.reason = "error verification"
        return run
    run.counts, run.win_freq, run.pe_fail = parameter_estimation(plan, data, rec)            # 6
    run.steps["6 parameter estimation"] = {"lost": run.counts[0], "won": run.counts[1], "generation": run.counts[2],
                                           "win_frequency": run.win_freq, "passed": not run.pe_fail,
                                           "violations": list(run.pe_fail)}
    if run.pe_fail:
        run.reason = "acceptance test: " + ", ".join(run.pe_fail)
        return run
    run.raw_alice, run.raw_bob = A_key, rec.B_key
    if pa != "none" and plan.ell <= 0:
        run.reason = "no key"
        return run
    run.key_alice, run.key_bob, run.pa_seed_bits = privacy_amplification(plan, data, rec, rng, pa)   # 7
    run.steps["7 privacy amplification"] = {"family": pa, "input_bits": int(A_key.size), "seed_bits": run.pa_seed_bits,
                                            "key_bits": int(run.key_alice.size),
                                            "keys_agree": bool(np.array_equal(run.key_alice, run.key_bob))}
    run.accepted = True
    return run


# ----------------------------------------------------------------------------------- the distribution stage
@dataclass
class DistributionResult:
    """Output of the distribution stage: the A-B and the A-C link."""
    ab: LinkRun                 # Alice-Bob:      key_alice = Alice's copy of K_AB, key_bob = Bob's
    ac: LinkRun                 # Alice-Charlie:  key_alice = Alice's copy of K_AC, key_bob = Charlie's

    @property
    def accepted(self) -> bool:
        return self.ab.accepted and self.ac.accepted

    @property
    def reason(self) -> str:
        return self.ab.reason or self.ac.reason

    @property
    def signatures(self) -> int:
        """Signatures the keys pay for: 3 n_s = 138 bits of each link per signature."""
        if not self.accepted:
            return 0
        return min(self.ab.key_alice.size, self.ac.key_alice.size) // KEY_BITS_PER_SIGNATURE

    def keys(self) -> Dict[str, np.ndarray]:
        return {"alice_ab": self.ab.key_alice, "bob": self.ab.key_bob,
                "alice_ac": self.ac.key_alice, "charlie": self.ac.key_bob}


def distribution_stage(plan: LinkPlan, dev_ab, dev_ac, rng: np.random.Generator, pa: str = "toeplitz",
                       sampling: str = "multinomial", plan_ac: Optional[LinkPlan] = None, ev_bits: int = EV_BITS,
                       cascade_passes: int = 4) -> DistributionResult:
    """Run the KGP of the A-B link on dev_ab and then the KGP of the A-C link on dev_ac (separate devices).
    Both links run even if the first aborts, as two independent protocols would."""
    ab = run_kgp(plan, dev_ab, rng, pa, sampling, ev_bits, cascade_passes)
    ac = run_kgp(plan_ac or plan, dev_ac, rng, pa, sampling, ev_bits, cascade_passes)
    return DistributionResult(ab, ac)
