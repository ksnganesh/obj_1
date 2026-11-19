"""Finite-size bounds: entropy functions, acceptance set, Renyi key length and its certification, EAT baseline."""
import math

import numpy as np
import pytest

from diqds.common import KEY_BITS_PER_SIGNATURE, N_S, Honest, QDSBudget, h2
from diqds.finite_size import eat as EAT
from diqds.finite_size import entropy as E
from diqds.finite_size import renyi as R
from diqds.finite_size.operating_points import ATOM_11KM, ION
from diqds.finite_size.protocols import ONE_SETTING, SEEDED, TWO_SETTING

W_MAX = 0.5 + math.sqrt(2) / 4


def test_signature_length_eq5():
    b = QDSBudget()
    assert N_S == 46 and KEY_BITS_PER_SIGNATURE == 3 * N_S
    assert 2 ** (1 - N_S) * b.n_m + b.eps_snd <= b.eps            # Eq. (5) with n_s = 46 meets eps = 1e-10
    assert 2 ** (2 - N_S) * b.n_m + b.eps_snd > b.eps             # n_s = 45 does not


def test_werner_and_second_basis():
    h = Honest.werner(1.0)
    assert h.S == pytest.approx(2 * math.sqrt(2)) and h.Q == pytest.approx(0.0)
    v = Honest.werner(0.93)
    assert v.Q_x == pytest.approx(v.Q)                            # depolarising noise: both bases equal
    for e in (ION, ATOM_11KM):                                    # dephasing-dominated experiments: X basis worse
        hon = e.honest
        assert hon.Q_x > hon.Q
        assert hon.Q_x == pytest.approx(1 - hon.S / (2 * math.sqrt(2)) - hon.Q)
        assert hon.isotropic().Q_x == pytest.approx(hon.Q)          # sensitivity case
    from diqds.finite_size.operating_points import ATOM_400M
    assert ATOM_400M.honest.Q_x == pytest.approx(ATOM_400M.honest.Q)   # QBER not measured in a CHSH setting


@pytest.mark.parametrize("alpha", [1.001, 1.01, 1.1])
def test_entropy_bounds(alpha):
    ws = np.linspace(0.75, W_MAX, 9)
    fd = np.array([float(E.f_down(w, alpha)) for w in ws])
    fv = np.array([float(E.f_vn(w)) for w in ws])
    assert abs(fd[0]) < 1e-9 and abs(fd[-1] - 1) < 1e-9 and abs(fv[-1] - 1) < 1e-9
    assert np.all(np.diff(fd) > 0)                                # monotone in the winning probability
    assert np.all(fd <= fv + 1e-12)                               # Renyi (alpha > 1) below von Neumann


def test_noisy_preprocessing_helps_at_the_classical_point():
    assert float(E.f_down_np(0.75, 1.01, 0.1)) > 0.4             # private flips are entropy even at w = 3/4
    assert float(E.f_down_np(0.8, 1.01, 0.0)) == pytest.approx(float(E.f_down(0.8, 1.01)))


def test_acceptance_tails_are_three_sigma():
    st = R.RenyiSettings()
    assert st.tail == pytest.approx(0.5 * math.erfc(3 / math.sqrt(2)))
    acc = R.acceptance_set(1e6, 0.05, 0.83, st)
    c = R.completeness_exact(acc)
    assert c <= st.completeness_AT + 1e-12 and c > 0.5 * st.completeness_AT


def test_kkt_projection_is_feasible():
    """Rows of q (one per w) sum to one and stay in the box; the KKT point beats random feasible points."""
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(3), size=4)
    c = rng.random((4, 3)) + 0.2
    lo, up = np.array([0.0, 0.2, 0.3]), np.array([0.3, 1.0, 1.0])
    q = R._kkt_q(p, c, lo, up)
    assert np.allclose(q.sum(axis=1), 1.0, atol=1e-9)
    assert np.all(q >= lo - 1e-12) and np.all(q <= up + 1e-12)
    obj = lambda qq, i: float(np.sum(qq * np.log(qq / (p[i] * c[i]))))    # KL(q || p c) up to constants
    for i in range(4):
        for _ in range(200):
            r = rng.dirichlet(np.ones(3))
            if np.all(r >= lo) and np.all(r <= up):
                assert obj(q[i], i) <= obj(r, i) + 1e-9


def test_key_length_grows_and_rates():
    hon = Honest.werner(1.0)
    small = R.key_length(1e6, hon, ONE_SETTING, 0.05, 1.01)
    big = R.key_length(1e7, hon, ONE_SETTING, 0.05, 1.01)
    assert 0 < small.ell < big.ell < 1e7
    assert R.asymptotic_rate(hon) == pytest.approx(1.0, abs=1e-9)
    assert R.asymptotic_rate(Honest.werner(0.95), TWO_SETTING) == pytest.approx(
        0.5 * R.asymptotic_rate(Honest.werner(0.95), ONE_SETTING), rel=1e-9)


def test_two_basis_error_correction_cost():
    st = R.RenyiSettings()
    hon = ATOM_11KM.honest
    one = R.ec_bits(1e6, hon, ONE_SETTING, 0.05, 0.0, st)
    seeded = R.ec_bits(1e6, hon, SEEDED, 0.05, 0.0, st)
    assert seeded > one                                           # the X basis has the larger error rate
    assert R.key_round_entropy(hon, SEEDED, 0.0) == pytest.approx(0.5 * (h2(hon.Q) + h2(hon.Q_x)))


def test_eat_threshold_below_honest():
    st = EAT.EATSettings(reading="rigorous")
    hon = Honest.werner(1.0)
    r = EAT.optimum(1e9, hon, 0.0, 0.5, st, restarts=1)
    assert r.ell > 0
    assert 0.75 < r.w_thr < hon.w


# ------------------------------------------------------------------------------------------- added in 0.5.0
def test_sifted_rate_generalises_two_settings():
    F = np.linspace(0.1, 0.9, 5)
    assert np.allclose(E.sifted_rate(F, 1.01, 0.5), E.two_setting_rate(F, 1.01))
    assert np.allclose(E.sifted_rate(F, 1.0001, 0.3), 0.3 * F, atol=1e-4)      # alpha -> 1: s F
    assert np.allclose(E.sifted_rate(F, 1.01, 1.0), F)


def test_independent_flags_key_fraction():
    from diqds.finite_size.protocols import INDEPENDENT, LU_GAMMA, LU_GAMMA_A, LU_GAMMA_B
    ga, gb = INDEPENDENT.flag_probabilities(LU_GAMMA)
    assert ga == pytest.approx(LU_GAMMA_A) and gb == pytest.approx(LU_GAMMA_B)   # Lu et al.'s choice is the optimum
    key = (1 - LU_GAMMA) * INDEPENDENT.sift_at(LU_GAMMA)
    assert key == pytest.approx((1 - LU_GAMMA_B) * (1 - LU_GAMMA_A / 2))


def test_inner_dual_brackets_primal():
    rng = np.random.default_rng(5)
    for _ in range(200):
        p = rng.dirichlet(np.ones(3), size=4)
        lo = np.array([0.0, rng.uniform(0, 0.3), rng.uniform(0, 0.4)])
        up = np.array([rng.uniform(0.3, 1), 1.0, 1.0])
        if lo.sum() > 1 or up.sum() < 1:
            continue
        t = rng.normal(0, 3, size=(4, 3))
        a = 1 + 10 ** rng.uniform(-4, -1)
        primal, dual = R._inner(p, t, lo, up, a)
        assert np.all(dual <= primal + 1e-12) and np.all(primal - dual < 1e-9 * np.maximum(1, np.abs(primal)))


def test_certified_bound_is_below_every_grid_value():
    hon = ATOM_11KM.honest
    st = R.RenyiSettings()
    for n, g, a, q in ((3e5, 0.05, 1.002, 0.0), (1.208e6, 0.03, 1.0007, 0.05)):
        acc = R.acceptance_set(n, g, hon.w, st)
        F = lambda w: ONE_SETTING.gen_rate(E.f_down_np(w, a, q), a, g)
        ws = np.linspace(R.W_MIN, R.W_MAX, 20001)
        J = R.objective_J(ws, g, a, F(ws), acc)
        c = R.h_alpha_certified(g, a, F, acc, tol=1e-10)
        assert c.converged and c.lower <= J.min() + 1e-14            # a lower bound on every grid value
        assert c.upper <= J.min() + 1e-14 and c.upper - c.lower <= 1e-10 + 1e-14   # tight: within the tolerance


def test_monotone_in_preprocessing():
    from diqds.finite_size.optimize import monotone_in_q
    assert monotone_in_q(1e6, ATOM_11KM.honest, ONE_SETTING)


def test_threshold_reaches_target_at_reported_n():
    from diqds.finite_size.optimize import threshold
    th = threshold(ION.honest, ONE_SETTING, 0.0, guess=2.1e5)
    assert th.ell >= KEY_BITS_PER_SIGNATURE and th.ell < KEY_BITS_PER_SIGNATURE + 40     # no 0.5 % overshoot
    assert th.result.info["certified"] == 1.0
    lower = R.key_length(th.n * 0.99, ION.honest, ONE_SETTING, th.result.gamma, th.result.alpha, 0.0)
    assert lower.ell < KEY_BITS_PER_SIGNATURE


def test_omitted_tails_are_inactive():
    """Adding the opposite one-sided tests (lost-low, won-high, generation-high) does not change h_alpha."""
    hon = ATOM_11KM.honest
    three = R.RenyiSettings()
    six = R.RenyiSettings(tails=("loss_low", "loss_upp", "win_low", "win_upp", "perp_low", "perp_upp"))
    for n, g, a in ((5e5, 0.04, 1.001), (1.208e6, 0.03, 1.0007)):
        r3 = R.key_length(n, hon, ONE_SETTING, g, a, 0.0, three)
        r6 = R.key_length(n, hon, ONE_SETTING, g, a, 0.0, six)
        assert r6.h_alpha == pytest.approx(r3.h_alpha, rel=1e-12, abs=1e-14)


def test_concave_envelope_is_identity_where_concave():
    for a in (1.0001, 1.01, 1.1):
        for q in (0.01, 0.1, 0.3):
            assert E.envelope_gap(a, q) == 0.0


def test_photonic_operating_point_ideal_source():
    from diqds.finite_size import photonic as PH
    op = PH.operating_point("ideal", 0.0, maxiter=200)
    assert op.chsh == pytest.approx(2.631, abs=2e-3) and op.qber == pytest.approx(0.0348, abs=5e-4)


def test_descent_finds_smallest_n_on_warm_started_curve():
    """The descent walks down from an overshooting threshold to the first integer n that reaches the target, and keeps
    a threshold that is already minimal."""
    from diqds.finite_size.descent import descend
    line = lambda n: 138.0 + 0.1 * (n - 10_000.0)            # reaches 138 bits exactly at n = 10,000
    ev = lambda n, hint: (line(n), hint, n)
    found = descend(ev, 10_500.0, line(10_500.0), None, 138.0)
    assert found is not None and found[0] == 10_000.0 and found[1] >= 138.0
    assert descend(ev, 10_000.0, line(10_000.0), None, 138.0) is None
    saw = lambda n: line(n) + 4.0 * ((n // 37) % 2)           # a sawtooth on top of the line
    ev2 = lambda n, hint: (saw(n), hint, n)
    found = descend(ev2, 10_600.0, saw(10_600.0), None, 138.0)
    assert found is not None and saw(found[0]) >= 138.0 and saw(found[0] - 1) < 138.0 and found[0] < 10_600.0
