"""Randomness ledger, platform calibration (QuTiP, Qiskit Aer), SeQUeNCe herald statistics."""
import math

import numpy as np
import pytest

from diqds.common import KEY_BITS_PER_SIGNATURE, h2
from diqds.finite_size.operating_points import ATOM_11KM, ION
from diqds.finite_size.protocols import ONE_SETTING, SEEDED, TWO_SETTING
from diqds.randomness import ledger as LG
from diqds.randomness import platforms as PL
from diqds.randomness import qutip_states as QS


def test_biased_cost():
    assert LG.biased_cost(0.0) == 0.0
    assert LG.biased_cost(0.1) == pytest.approx(h2(0.1))
    assert LG.biased_cost(0.1, "dyadic8") == 8.0


def test_per_round_costs():
    g, q = 0.05, 0.1
    two = LG.per_round(TWO_SETTING, g, q)
    assert two.alice == 1.0 and two.bob == pytest.approx(h2(g) + 1.0)
    one = LG.per_round(ONE_SETTING, g, q)
    assert one.alice == pytest.approx(g + h2(g)) and one.shared_secret == 0.0
    one_pa = LG.per_round(ONE_SETTING, g, q, per_attempt=True)
    assert one_pa.shared_secret == pytest.approx(h2(g)) and one_pa.alice == pytest.approx(g)
    assert LG.per_round(SEEDED, g).shared_secret == 1.0


def test_block_ledger_arithmetic():
    n, g, q, ell = 1e6, 0.04, 0.0, 70_000
    led = LG.block_ledger(ONE_SETTING, n, g, q, ell)
    rc = LG.per_round(ONE_SETTING, g, q)
    assert led.private_input == pytest.approx((rc.alice + rc.bob) * n + rc.flips * n)
    assert led.signatures == pytest.approx(ell / KEY_BITS_PER_SIGNATURE)
    n_in = ONE_SETTING.pa_input_bits(n, g)
    assert led.details["pa_seed"] == pytest.approx(n_in + ell - 1)
    mod = LG.block_ledger(ONE_SETTING, n, g, q, ell, pa_family="modified_toeplitz")
    assert mod.public_fresh == pytest.approx(n_in - 1 + LG.EV_SEED_BITS)
    # per-attempt links pay for every attempt
    att = LG.block_ledger(TWO_SETTING, n, g, q, ell, attempts_per_round=1000.0, per_attempt=True)
    assert att.private_input == pytest.approx(1000.0 * led_two(n, g, q).private_input, rel=1e-9)


def led_two(n, g, q):
    return LG.block_ledger(TWO_SETTING, n, g, q, 1.0)


@pytest.mark.parametrize("exp,eps_r", [(ION, 1e-3), (ATOM_11KM, 5e-3)])
def test_matter_calibration_reproduces_S_and_Q(exp, eps_r):
    m = QS.calibrate_matter(exp.honest.S, exp.honest.Q, eps_r)
    S, w, Q = QS.statistics(m.probabilities())
    assert S == pytest.approx(exp.honest.S, abs=1e-6) and Q == pytest.approx(exp.honest.Q, abs=1e-6)


def test_aer_matches_qutip():
    pytest.importorskip("qiskit_aer")
    from diqds.validation.aer_devices import AerDevice, AerNoise
    m = QS.calibrate_matter(ATOM_11KM.honest.S, ATOM_11KM.honest.Q, 5e-3)
    dev = AerDevice(AerNoise.from_matter(m))
    assert np.abs(dev.P - m.probabilities()).max() < 1e-9


def test_extra_wait_lowers_S():
    m = QS.calibrate_matter(ATOM_11KM.honest.S, ATOM_11KM.honest.Q, 5e-3)
    S0 = QS.statistics(m.probabilities())[0]
    S1 = QS.statistics(m.probabilities(extra_wait=55e-6, T2_memory=1e-3))[0]
    assert S1 < S0


def test_platform_rates():
    assert PL.ION_PLATFORM.herald_rate_hz == pytest.approx(52.08, rel=1e-3)
    qd = PL.photonic_platform(0.0)
    assert qd.per_attempt_inputs and qd.attempts_per_herald > 1e4


def test_sequence_attempts_per_herald():
    pytest.importorskip("sequence")
    from diqds.randomness import sequence_links as SL
    p = PL.ATOM11_PLATFORM
    r = SL.simulate(p, 1.0, 40, "attempts", seed=3)
    sd = p.attempts_per_herald / math.sqrt(40)
    assert abs(r.mean_attempts - p.attempts_per_herald) < 4 * sd


# ------------------------------------------------------------------------------------------- added in 0.5.0
def test_independent_flags_ledger():
    from diqds.finite_size.protocols import INDEPENDENT
    g = 0.0338
    rc = LG.per_round(INDEPENDENT, g, 0.0)
    ga, gb = INDEPENDENT.flag_probabilities(g)
    assert rc.alice == pytest.approx(h2(ga) + ga) and rc.bob == pytest.approx(h2(gb) + gb)
    assert LG.per_round(INDEPENDENT, g, 0.0, per_attempt=True).shared_secret == 0.0     # no pre-shared secret


def test_first_signature_pays_for_the_block():
    led = LG.block_ledger(ONE_SETTING, 5e5, 0.04, 0.0, 140.0)
    tot = led.block_totals()
    assert tot["private_input"] == pytest.approx(led.private_input)
    assert led.per_signature()["auth_total"] == pytest.approx(LG.MESSAGES_PER_BLOCK * LG.TAG_BITS / led.signatures
                                                              + LG.MESSAGES_PER_SIGNATURE * LG.TAG_BITS)
