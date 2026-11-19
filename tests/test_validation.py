"""GF(2) arithmetic, OTUH signatures, Cascade, acceptance statistics and the end-to-end emulator."""
import numpy as np
import pytest

from diqds.common import N_S, Honest, h2
from diqds.finite_size.protocols import ONE_SETTING
from diqds.finite_size.renyi import RenyiSettings
from diqds.randomness import qutip_states as QS
from diqds.validation import gf2, otuh
from diqds.validation import montecarlo as MC
from diqds.validation import pipeline as P
from diqds.validation.adversarial import FractionDevice, HonestThenDeterministic, WindowDevice
from diqds.validation.aer_devices import TableDevice
from diqds.validation.cascade import cascade


def test_poly_from_key_is_irreducible():
    rng = np.random.default_rng(2)
    for _ in range(5):
        z = int(rng.integers(1, 2 ** N_S))
        p = gf2.poly_from_key(z, N_S)
        assert gf2.deg(p) == N_S and gf2.is_irreducible(p)


def test_berlekamp_massey_recovers_lfsr():
    s = [1, 0, 0, 1]                         # s_t = s_{t-1} + s_{t-4}: x^4 + x^3 + 1 is primitive
    for _ in range(40):
        s.append((s[-1] + s[-4]) % 2)
    c, L = gf2.berlekamp_massey(s)
    assert L == 4
    for t in range(L, len(s)):               # the connection polynomial reproduces the sequence
        assert sum(int(c[j]) * s[t - j] for j in range(L + 1)) % 2 == 0


def test_otuh_hash_is_linear_in_the_document():
    rng = np.random.default_rng(3)
    k = otuh.cut(rng.integers(0, 2, 3 * N_S, dtype=np.uint8))
    m1, m2 = rng.integers(0, 2, 1000, dtype=np.uint8), rng.integers(0, 2, 1000, dtype=np.uint8)
    d = lambda m: otuh.digest(m, k.X, k.Z)
    assert d(m1 ^ m2) == d(m1) ^ d(m2) and 0 <= d(m1) < 2 ** N_S


def test_cascade_corrects_and_leaks_above_shannon():
    rng = np.random.default_rng(4)
    n, Q = 40_000, 0.03
    a = rng.integers(0, 2, n, dtype=np.uint8)
    b = a ^ (rng.random(n) < Q).astype(np.uint8)
    corr, leak = cascade(a, b, Q, rng)
    assert np.array_equal(corr, a)
    assert n * h2(Q) < leak < 1.35 * n * h2(Q)


@pytest.fixture(scope="module")
def plan_and_device():
    hon = Honest.werner(0.98)
    plan = P.plan_link(400_000, hon, ONE_SETTING, q="opt", settings=RenyiSettings.cascade())
    return plan, TableDevice(QS.werner_probabilities(0.98)), hon


def test_acceptance_mc_matches_binomial(plan_and_device):
    plan, dev, hon = plan_and_device
    rng = np.random.default_rng(6)
    f = 0.05
    an = MC.acceptance_probability(plan, hon.w - f * (hon.w - 0.75))
    mc = MC.acceptance_mc(plan, FractionDevice(dev, f, seed=1), 120, rng)
    assert abs(mc - an) < 4 * np.sqrt(an * (1 - an) / 120) + 0.02


def test_end_to_end_honest(plan_and_device):
    plan, dev, hon = plan_and_device
    rng = np.random.default_rng(7)
    s = MC.full_mc(plan, dev, dev, 1, rng, n_docs=3)
    d = s.as_dict()
    assert d["accepted"] == 1 and d["honest_ok"] == d["honest_total"] == 3
    assert d["forged_accepted"] == 0 and d["key_mismatch"] == 0


def test_memory_attack_aborts(plan_and_device):
    plan, dev, hon = plan_and_device
    rng = np.random.default_rng(8)
    r = P.run_link(plan, HonestThenDeterministic(dev, 0.5), rng)
    assert not r.accepted and r.reason.startswith("acceptance test")


def test_window_device_outputs():
    dev = TableDevice(QS.werner_probabilities(1.0))
    w = WindowDevice(dev, 100, 50, seed=1)
    rng = np.random.default_rng(9)
    x = rng.integers(0, 2, 400).astype(np.int8)
    y = rng.integers(0, 4, 400).astype(np.int8)
    a, b = w.sample(x, y, rng)
    assert np.array_equal(a[100:150], w.hidden[100:150]) and np.array_equal(b[100:150], w.hidden[100:150])
