"""Can device-independent signatures skip privacy amplification?  A counter-example, a no-gain bound, and the seed cost
of every option.

Li et al., Phys. Rev. Applied 20, 044011 (2023) prove that OTUH signatures can use keys WITHOUT privacy amplification
when the devices are characterised: phase-error sampling then bounds Eve's information on every subset of the key, so
each signature's 3 n_s bits carry enough min-entropy.  In the device-independent setting the entropy bound (EAT/REAT)
holds for the WHOLE raw string only, and an untrusted device may place that entropy where it likes.

Proposition 1 (counter-example).  Let signatures use consecutive 3 n_s-bit segments of the error-corrected raw key, in an
order computable from public data, with no privacy amplification.  The devices of the Alice-Charlie link play the honest
strategy except in a window of W consecutive rounds, where both output a bit string lambda fixed by Eve (WindowDevice).
In the window the key bits equal lambda (no extra errors), and the test rounds that fall into it are won with probability
3/4 instead of w.  With W ~ 2 x 3 n_s / (1 - gamma) ~ 300 rounds this shifts the expected number of won test rounds by
about gamma W (w - 3/4) ~ 1-2, far inside the acceptance tolerance (~3 sqrt(gamma n w (1-w)) ~ 10^2), so the test passes
with essentially the honest probability.  After the protocol the round types are public, so Eve and the dishonest
recipient Bob know which key positions came from the window, hence X_c, Y_c, Z_c of every signature segment inside it,
and Bob can make Charlie accept any document for that segment: forgery probability 1.  (With noisy preprocessing q > 0
Alice's private flips hide each bit with probability q; the attack is shown at q = 0, a legitimate choice of the
protocol.)  With privacy amplification the same devices are harmless: the window removes at most W bits of min-entropy,
which the bound already accounts for, and the hashed key is close to uniform.

Proposition 2 (random segments: an i.i.d. attack).  Let each signature use 3 n_s raw bits chosen by public randomness
after the quantum phase, so that the window attack no longer works.  Take the i.i.d. devices of the attack that
minimises H(A|X=0,E) at CHSH score S (Bell-diagonal state sqrt(P+)|phi+>|0>_E + sqrt(P-)|phi->|1>_E, P+- = (1 +- g)/2,
g = sqrt(S^2/4 - 1); Alice measures Z, X; Bob (Z +- g X)/sqrt(1 + g^2) and Z for the key).  Its key rounds have no
errors, it wins the CHSH game with w = 1/2 + S/8, and Eve guesses each key bit with p_g(S) = (1 + sqrt(2 - S^2/4))/2.
A forger who knows X_c and Z_c (2 n_s bits) makes Charlie accept any document, so
    P(accept and forge) >= max_S  P_acc(S) p_g(S)^(2 n_s),
which exceeds eps = 1e-10 near the first-signature block (e.g. ~3e-10 for the recorded 11 km block) and falls below it
for larger blocks.  With noisy preprocessing p_g -> p_g - q (2 p_g - 1).

Theorem 2 (certification limit for raw segments; replaces the earlier informal Proposition 3).  Under the i.i.d.
device sigma_S of Proposition 2, accepted with probability P_acc(S), take any set T of k raw-key positions fixed
independently of the devices (e.g. sampled by public randomness after the quantum phase).  The test rounds and the key
rounds of an i.i.d. device are independent, so conditioning on acceptance does not change the key rounds, and the
guessing probability of a product cq-state is the product of the single-round ones:
    H_min(K_T | E) = k h_min(S, q),        h_min(S, q) = -log2[ p_g(S) - q (2 p_g(S) - 1) ].
(a) No device-independent argument can certify more than k h_min(S, q) bits of min-entropy for a raw k-bit segment of
    a protocol that accepts sigma_S with non-negligible probability: the certificate would be false for sigma_S.
(b) A sampling lemma (Konig and Renner, IEEE TIT 57, 4760 (2011)) gives the opposite inequality: a random subset keeps
    AT LEAST roughly its proportional share k r_cert of the certified smooth min-entropy rate r_cert of the block.
    That is what a proof can certify; (a) says what no proof can exceed.
(c) If the 2 n_s hash-key bits (X_c, Z_c) of an OTUH signature are such a segment, the forging probability bound that
    goes through the min-entropy (list-guessing: P_forge <= |list| 2^{-H_min}, with at most delta 2^{2 n_s} keys
    consistent with any (Delta, tag) for the delta-almost-XOR-universal family, delta = 2^{1-n_s} n_m) is
        eps_for <= 2^{1 - n_s} n_m 2^{2 n_s - H_min(X_c Z_c | E)}  >=  n_m 2^{1 + n_s (1 - 2 h_min(S, q))},
    which is >= 1 for EVERY n_s >= 1 when h_min(S, q) < 1/2.  For q = 0 this happens below
        S_half = 2 sqrt(2 sqrt2 - 1) = 2.7044,
    i.e. for every DIQKD experiment to date (S = 2.578-2.677).  Preprocessing raises h_min, but h_min >= 1/2 needs
    q >= q_half(S) = (p_g - 2^{-1/2}) / (2 p_g - 1), which at 11 km (q_half = 0.118) leaves no key.
(d) The explicit attack of Proposition 2 is the matching lower bound: P(accept and forge) >= P_acc(S) p_g(S, q)^(2 n_s).
So raw (unhashed) segments can be used neither sequentially (Proposition 1) nor by sampling near the first-signature
block (Proposition 2), and no min-entropy-based argument certifies them at any block size or signature length when
S < S_half.  Whether an argument that exploits the structure of the hash could do better is open.

Proposition 1, quantitatively (window_acceptance_lower_bound).  Couple the window device with the honest device round
by round (monotone coupling of Bernoulli(w) and Bernoulli(3/4) in the window's test rounds).  The window device has D
fewer wins and D more losses, D ~ Bin(W, gamma (w - 3/4)), and the same number of test rounds, so for every d >= 0
    P_acc(window W) >= P_acc^{(d)}(honest) - P(D > d),
where P_acc^{(d)} is the honest acceptance probability of the test with both count thresholds tightened by d.  Both
terms are exact binomial sums, so this is a certified lower bound on the acceptance probability as a function of W.

Seed economy that does work (seeds.py): the modified Toeplitz family [T | I] needs N - 1 instead of N + ell - 1 bits; one
seed per block serves all its signatures, so the seed per signature falls as 1/N_sig; and the seed only has to be uniform
and independent of the raw key, so it can be public randomness generated after the quantum phase (e.g. a beacon) and
leaves the private-randomness ledger.  Reusing a seed across blocks is not covered by the leftover hash lemma (Eve
prepares the next block after seeing it), so we do not consider it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
from scipy.special import gammaln

from ..common import KEY_BITS_PER_SIGNATURE, N_S
from ..validation.adversarial import WindowDevice
from ..validation.pipeline import LinkPlan


@dataclass
class WindowAttackResult:
    pa: str
    accepted: bool
    reason: str
    signatures: int
    window_rounds: int
    tests_in_window: int
    known_key_positions: int
    targeted_segments: int
    targeted_forged: int
    other_tried: int
    other_forged: int


def window_attack(plan: LinkPlan, honest_device, rng: np.random.Generator, pa: str = "none", start: int = None,
                  length: int = None, n_docs: int = 4) -> WindowAttackResult:
    """Honest A-B link, WindowDevice on the A-C link; the dishonest Bob (with Eve) forges the segments she knows.
    Without PA the targeted segments are the 3 n_s-bit segments fully inside the window; with PA Eve's raw-bit knowledge
    does not determine any hashed segment, so she can only guess (the first segments are attacked for comparison)."""
    from ..validation.otuh import cut
    from ..protocol.distribution import distribution_stage
    from ..protocol.messaging import forgery_attempt, key_sets, messaging_stage
    n = plan.n
    w = KEY_BITS_PER_SIGNATURE
    if length is None:
        length = int(math.ceil(2.2 * w / max(plan.protocol.key_rounds_fraction(plan.gamma), 1e-9)))
    if start is None:
        start = n // 2
    wdev = WindowDevice(honest_device, start, length, seed=int(rng.integers(1 << 30)))
    dist = distribution_stage(plan, honest_device, wdev, rng, pa)          # honest A-B link, attacked A-C link
    ab, ac = dist.ab, dist.ac
    tests_in = int(np.count_nonzero((ac.test_rounds >= start) & (ac.test_rounds < start + length)))
    res = WindowAttackResult(pa, ab.accepted and ac.accepted, ab.reason or ac.reason, 0, length, tests_in, 0, 0, 0, 0, 0)
    if not res.accepted:
        return res
    kr = ac.key_rounds
    pos = np.flatnonzero((kr >= start) & (kr < start + length))
    known = {int(p_): int(wdev.hidden[kr[p_]]) for p_ in pos}            # q = 0: no private flips to hide them
    res.known_key_positions = len(known)
    S = min(ab.key_alice.size, ac.key_alice.size) // w
    res.signatures = S
    if pa == "none":
        targets = [j for j in range(S) if all((j * w + i) in known for i in range(w))]
    else:
        targets = list(range(min(3, S)))
    for j in sorted(set(range(min(n_docs, S))) | set(targets)):
        if j in targets:
            bits = (np.array([known[j * w + i] for i in range(w)], np.uint8) if pa == "none"
                    else rng.integers(0, 2, w, dtype=np.uint8))          # with PA: no knowledge of the hashed bits
            knows = cut(bits, N_S)
        else:
            knows = None
        doc = rng.integers(0, 2, 1000, dtype=np.uint8)
        alice, bob, charlie = key_sets(dist, j)
        rec = messaging_stage(doc, alice, bob, charlie, index=j)             # Alice signs, Bob and Charlie verify
        forged = forgery_attempt(rec, bob, charlie, rng, knows) if rec.sig is not None else None
        ok = bool(forged and forged.accepted)                               # Charlie accepts Bob's altered document
        if j in targets:
            res.targeted_segments += 1
            res.targeted_forged += int(ok)
        else:
            res.other_tried += 1
            res.other_forged += int(ok)
    return res


# ------------------------------------------------------------------------------------- Proposition 2: i.i.d. attack
def p_guess_chsh(S):
    """Eve's probability of guessing Alice's output in one CHSH setting under the attack of minimal H(A|X=0,E)."""
    S = np.clip(np.asarray(S, dtype=float), 2.0, 2.0 * math.sqrt(2.0))
    return 0.5 * (1.0 + np.sqrt(np.clip(2.0 - S * S / 4.0, 0.0, 1.0)))


def chsh_attack_state(S):
    """(psi_ABE, observables): the tripartite pure state and the measurements of the attack at score S (QuTiP)."""
    import qutip as qt
    g = float(np.sqrt(max(S * S / 4.0 - 1.0, 0.0)))
    Pp, Pm = 0.5 * (1 + g), 0.5 * (1 - g)
    phip = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
    phim = (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) - qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()
    psi = (math.sqrt(Pp) * qt.tensor(phip, qt.basis(2, 0)) + math.sqrt(Pm) * qt.tensor(phim, qt.basis(2, 1))).unit()
    N = math.sqrt(1.0 + g * g)
    A = [qt.sigmaz(), qt.sigmax()]
    B = [qt.sigmaz(), qt.sigmax(), (qt.sigmaz() + g * qt.sigmax()) / N, (qt.sigmaz() - g * qt.sigmax()) / N]
    return psi, A, B


def chsh_attack_table(S) -> np.ndarray:
    """P[x, y, a, b] of the attack at score S, with the pipeline's labels: key y = 0 (Z), CHSH y = 2, 3."""
    import qutip as qt
    psi, A, B = chsh_attack_state(S)
    rho = psi.ptrace([0, 1])
    I = qt.qeye(2)
    P = np.zeros((2, 4, 2, 2))
    for x in range(2):
        for y in range(4):
            for a in range(2):
                for b in range(2):
                    Pa = (I + (-1) ** a * A[x]) / 2
                    Pb = (I + (-1) ** b * B[y]) / 2
                    P[x, y, a, b] = float(np.real((rho * qt.tensor(Pa, Pb)).tr()))
    return P


def eve_guess_numeric(S) -> float:
    """Helstrom guessing probability of Alice's key bit from Eve's purification (QuTiP), to check p_guess_chsh."""
    import qutip as qt
    psi, A, _ = chsh_attack_state(S)
    rho = psi * psi.dag()
    I = qt.qeye(2)
    sig = [(rho * qt.tensor((I + (-1) ** a * A[0]) / 2, I, I)).ptrace([2]) for a in range(2)]
    diff = (sig[0] - sig[1]).full()
    return float(0.5 * (1.0 + np.abs(np.linalg.eigvalsh(diff)).sum()))


@dataclass
class IIDAttack:
    n: float
    S_star: float
    w_star: float
    p_accept: float
    p_guess_bit: float
    p_guess_keys: float
    joint: float
    eps: float
    S_grid: np.ndarray
    joint_grid: np.ndarray
    segments: float = 0.0              # raw 3 n_s-bit segments in the block (what a no-PA scheme would sign)
    S_block: float = 0.0
    block_joint: float = 0.0           # P(accept and at least one of the block's segments forged)

    @property
    def breaks(self) -> bool:
        return self.joint > self.eps

    @property
    def breaks_block(self) -> bool:
        return self.block_joint > self.eps


def iid_attack_on_sampling(plan: LinkPlan, n_s: int = N_S, grid: int = 240, eps: float = 1e-10) -> IIDAttack:
    """max_S P_acc(S) p_g(S)^(2 n_s) for the i.i.d. attack against randomly sampled no-PA segments (Proposition 2)."""
    from ..validation import montecarlo as MC
    S_hon = 8.0 * plan.honest.w - 4.0
    Ss = np.linspace(2.0, min(2.0 * math.sqrt(2.0), S_hon + 0.08), grid)
    pg = p_guess_chsh(Ss)
    pg = pg - plan.q * (2.0 * pg - 1.0)
    pacc = np.array([MC.acceptance_probability(plan, 0.5 + S / 8.0) for S in Ss])
    joint = pacc * pg ** (2 * n_s)
    i = int(np.argmax(joint))
    # refine around the maximum
    lo, hi = Ss[max(i - 1, 0)], Ss[min(i + 1, grid - 1)]
    Sf = np.linspace(lo, hi, 61)
    pgf = p_guess_chsh(Sf)
    pgf = pgf - plan.q * (2.0 * pgf - 1.0)
    jf = np.array([MC.acceptance_probability(plan, 0.5 + S / 8.0) for S in Sf]) * pgf ** (2 * n_s)
    j = int(np.argmax(jf))
    S_star = float(Sf[j])
    pa = MC.acceptance_probability(plan, 0.5 + S_star / 8.0)
    # block level: the forger targets every segment of the block (independent guesses for an i.i.d. device)
    segs = plan.n * plan.protocol.key_rounds_fraction(plan.gamma) / (3 * n_s)
    with np.errstate(divide="ignore"):                  # p_g = 1 at S = 2: certain guess, log1p(-1) = -inf is fine
        blk = pacc * -np.expm1(segs * np.log1p(-(pg ** (2 * n_s))))
    k = int(np.argmax(blk))
    return IIDAttack(plan.n, S_star, 0.5 + S_star / 8.0, pa, float(pgf[j]), float(pgf[j] ** (2 * n_s)), float(jf[j]), eps,
                     Ss, joint, float(segs), float(Ss[k]), float(blk[k]))


# ------------------------------------------------------------------------------------- Theorem 2: certification limit
S_HALF = 2.0 * math.sqrt(2.0 * math.sqrt(2.0) - 1.0)          # h_min(S, 0) = 1/2


def h_min_attack(S, q: float = 0.0):
    """Min-entropy per raw key bit under the i.i.d. attack sigma_S, with noisy preprocessing q (Theorem 2)."""
    pg = p_guess_chsh(S)
    return -np.log2(pg - q * (2.0 * pg - 1.0))


def q_half(S) -> float:
    """Smallest preprocessing probability with h_min(S, q) >= 1/2."""
    pg = float(p_guess_chsh(S))
    return max(0.0, (pg - 2.0 ** -0.5) / (2.0 * pg - 1.0))


def certified_forging_bound_raw(n_s: int, n_m: float, h: float) -> float:
    """Smallest forging bound the min-entropy route can give for raw hash keys with h_min = h per bit (Theorem 2c):
    n_m 2^{1 + n_s (1 - 2h)} (capped at 1)."""
    return float(min(1.0, n_m * 2.0 ** (1.0 + n_s * (1.0 - 2.0 * h))))


def window_acceptance_lower_bound(plan: LinkPlan, W: int, w: float = None) -> Dict[str, float]:
    """Certified lower bound on the acceptance probability of the window device (Proposition 1), by coupling."""
    from scipy.stats import binom as B
    acc, n, g = plan.acceptance, plan.n, plan.gamma
    w = plan.honest.w if w is None else w
    lam = g * max(w - 0.75, 0.0)
    mu, sd = g * n, math.sqrt(n * g * (1 - g))
    t = np.arange(max(0, int(mu - 12 * sd)), min(n, int(mu + 12 * sd)) + 1)
    p_t = B.pmf(t, n, g)
    perp_ok = (n - t) >= acc.k_lo[2]
    best = {"bound": 0.0, "d": 0}
    honest = float(np.sum(p_t * perp_ok * B.sf(np.maximum(acc.k_lo[1], t - acc.k_up[0]) - 1, t, w)))
    d_max = int(B.isf(1e-15, W, lam)) + 1 if W > 0 else 0
    for d in range(0, d_max + 1):
        need = np.maximum(acc.k_lo[1] + d, t - acc.k_up[0] + d)
        a_d = float(np.sum(p_t * perp_ok * B.sf(need - 1, t, w)))
        tail = float(B.sf(d, W, lam)) if W > 0 else 0.0
        if a_d - tail > best["bound"]:
            best = {"bound": a_d - tail, "d": d}
    best.update({"W": W, "honest": honest, "expected_D": W * lam})
    return best


def segment_raw_bits(ell: float, N: float, n_s: int = N_S) -> float:
    """Raw bits that must be compressed into one signature's 3 n_s bits: m = 3 n_s N / ell (any certified scheme)."""
    return 3 * n_s * N / ell


def certified_segment_entropy_without_hashing(ell: float, N: float, n_s: int = N_S) -> float:
    """Proportional share of the block's certified entropy carried by 3 n_s raw bits, 3 n_s ell / N: roughly what a
    sampling-lemma argument (Konig-Renner) certifies for a publicly sampled segment (a LOWER bound on its entropy)."""
    return 3 * n_s * ell / N


def log2_binom(N: float, m: float) -> float:
    return float((gammaln(N + 1) - gammaln(m + 1) - gammaln(N - m + 1)) / math.log(2))


def seed_options(N: float, ell: float, n_out: float = None) -> Dict[str, Dict[str, float]]:
    """Seed bits per block and per signature for every option, for a block with N raw key bits and ell net key bits.
    ``secure``: 1 if the option is covered by the entropy bound (leftover hash lemma, with a sampling lemma for the
    sampled local hash), 0 if the window device breaks it."""
    n_out = ell if n_out is None else n_out
    sigs = ell / KEY_BITS_PER_SIGNATURE
    m = min(segment_raw_bits(ell, N), N)
    perm = float(gammaln(N + 1) / math.log(2))
    local = log2_binom(N, m) + (m - 1)
    return {
        "toeplitz": {"per_block": N + n_out - 1, "per_signature": (N + n_out - 1) / sigs, "secure": 1},
        "modified_toeplitz": {"per_block": N - 1, "per_signature": (N - 1) / sigs, "secure": 1},
        "no_pa": {"per_block": 0.0, "per_signature": 0.0, "secure": 0,
                  "certified_segment_entropy": certified_segment_entropy_without_hashing(ell, N)},
        "local_pa_chunks": {"per_block": (m - 1) * sigs, "per_signature": m - 1, "segment_raw_bits": m, "secure": 0},
        "local_pa_sampled": {"per_block": local * sigs, "per_signature": local, "segment_raw_bits": m, "secure": 1},
        "permutation_then_chunks": {"per_block": perm + (m - 1) * sigs, "per_signature": perm / sigs + m - 1,
                                    "segment_raw_bits": m, "secure": 1},
    }
