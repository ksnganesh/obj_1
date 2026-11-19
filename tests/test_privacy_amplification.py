"""Hash families, seed options, the window counter-example and the certification limit for raw segments."""
import math

import numpy as np
import pytest

from diqds.common import Honest
from diqds.finite_size.protocols import ONE_SETTING
from diqds.finite_size.renyi import RenyiSettings
from diqds.randomness import qutip_states as QS
from diqds.privacy_amplification import no_pa as NP
from diqds.privacy_amplification import seeds as SD
from diqds.validation import pipeline as P
from diqds.validation.aer_devices import TableDevice


def test_toeplitz_matches_matrix():
    rng = np.random.default_rng(0)
    for n, m in ((17, 5), (64, 20), (300, 120)):
        x = rng.integers(0, 2, n, dtype=np.uint8)
        s = rng.integers(0, 2, n + m - 1, dtype=np.uint8)
        T = SD.toeplitz_matrix(s, n, m).astype(int)
        assert np.array_equal(SD.toeplitz_hash(x, s, m), (T @ x) % 2)


def test_fft_path_agrees_with_exact():
    rng = np.random.default_rng(1)
    n, m = 5000, 1200                                   # n m > 4e6: FFT branch
    x = rng.integers(0, 2, n, dtype=np.uint8)
    s = rng.integers(0, 2, n + m - 1, dtype=np.uint8)
    exact = (np.convolve(s.astype(np.int64), x.astype(np.int64))[n - 1:n - 1 + m] & 1).astype(np.uint8)
    assert np.array_equal(SD.toeplitz_hash(x, s, m), exact)


@pytest.mark.parametrize("family", ["toeplitz", "modified_toeplitz"])
def test_universality_exhaustive(family):
    assert SD.collision_probability(family, 6, 3) <= 2 ** -3 + 1e-12


def test_log2_binom():
    for N, m in ((10, 3), (40, 20), (100, 1)):
        assert NP.log2_binom(N, m) == pytest.approx(math.log2(math.comb(N, m)), rel=1e-9)


@pytest.mark.parametrize("sigs", [1, 10, 1000])
def test_local_hashing_never_beats_global(sigs):
    ell = 138.0 * sigs
    N = 20.0 * ell
    o = NP.seed_options(N, ell)
    assert o["local_pa_sampled"]["segment_raw_bits"] == pytest.approx(N / sigs)       # no saving in raw key
    assert o["local_pa_sampled"]["per_signature"] >= o["modified_toeplitz"]["per_signature"] - 1e-9
    if sigs > 1:
        assert o["local_pa_sampled"]["per_signature"] > o["modified_toeplitz"]["per_signature"]
    assert o["modified_toeplitz"]["per_signature"] < o["toeplitz"]["per_signature"]
    assert o["no_pa"]["certified_segment_entropy"] == pytest.approx(138.0 / 20.0)


@pytest.fixture(scope="module")
def small_plan():
    hon = Honest.werner(0.98)
    return P.plan_link(400_000, hon, ONE_SETTING, q=0.0, settings=RenyiSettings.cascade()), hon


def test_window_attack_forges_without_pa(small_plan):
    plan, hon = small_plan
    assert plan.ell >= 3 * 138
    dev = TableDevice(QS.werner_probabilities(0.98))
    rng = np.random.default_rng(5)
    r = NP.window_attack(plan, dev, rng, "none")
    assert r.accepted and r.targeted_segments >= 1
    assert r.targeted_forged == r.targeted_segments              # Charlie's keys known: forgery always succeeds
    assert r.other_forged == 0
    r2 = NP.window_attack(plan, dev, rng, "toeplitz")
    assert r2.accepted and r2.targeted_forged == 0               # after hashing the window is useless


@pytest.mark.parametrize("S", [2.1, 2.45, 2.612, 2.75])
def test_chsh_attack_guessing_probability(S):
    """Proposition 2's attack: the analytic p_g equals the Helstrom value from Eve's purification (QuTiP), the device
    reaches score S and its key rounds have no errors."""
    pytest.importorskip("qutip")
    from diqds.privacy_amplification.no_pa import chsh_attack_table, eve_guess_numeric, p_guess_chsh
    assert eve_guess_numeric(S) == pytest.approx(float(p_guess_chsh(S)), abs=1e-12)
    S_dev, w_dev, Q_dev = QS.statistics(chsh_attack_table(S))
    assert S_dev == pytest.approx(S, abs=1e-9) and Q_dev == pytest.approx(0.0, abs=1e-12)


def test_iid_attack_is_a_maximum(small_plan):
    plan, hon = small_plan
    r = NP.iid_attack_on_sampling(plan, grid=80)
    assert 0.0 < r.joint <= 1.0 and r.joint >= np.max(r.joint_grid) * (1 - 1e-9)
    assert r.block_joint >= r.joint                      # targeting every segment can only help the forger
    assert 2.0 < r.S_star <= 2.0 * np.sqrt(2.0)


# ------------------------------------------------------------------------------------------- added in 0.5.0
def test_theorem2_quantities():
    from diqds.privacy_amplification import no_pa as NP
    assert float(NP.h_min_attack(NP.S_HALF)) == pytest.approx(0.5, abs=1e-12)
    assert float(NP.h_min_attack(2 * math.sqrt(2))) == pytest.approx(1.0)
    assert float(NP.h_min_attack(2.0)) == pytest.approx(0.0)
    # below S_half the min-entropy route gives no bound for any signature length
    h = float(NP.h_min_attack(2.677))
    assert h < 0.5 and all(NP.certified_forging_bound_raw(ns, 1000, h) == 1.0 for ns in (46, 82, 500))
    # preprocessing at q_half lifts h_min to 1/2
    q = NP.q_half(2.612)
    assert float(NP.h_min_attack(2.612, q)) == pytest.approx(0.5, abs=1e-9)


def test_window_lower_bound_is_below_the_estimate():
    from diqds.finite_size.operating_points import ATOM_11KM
    from diqds.finite_size.protocols import ONE_SETTING
    from diqds.finite_size.renyi import RenyiSettings
    from diqds.privacy_amplification import no_pa as NP
    from diqds.validation import montecarlo as MC
    from diqds.validation import pipeline as P
    plan = P.plan_link(1_208_000, ATOM_11KM.honest, ONE_SETTING, q=0.0, settings=RenyiSettings.cascade())
    w = ATOM_11KM.honest.w
    b0 = NP.window_acceptance_lower_bound(plan, 0)
    assert b0["bound"] == pytest.approx(MC.acceptance_probability(plan, w), rel=1e-9)
    for W in (317, 5000, 50000):
        lb = NP.window_acceptance_lower_bound(plan, W)["bound"]
        est = MC.acceptance_probability(plan, w - (W / plan.n) * (w - 0.75))
        assert lb <= est + 1e-6
    assert NP.window_acceptance_lower_bound(plan, 317)["bound"] > 0.98
