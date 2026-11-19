"""The two stages of the DI-QDS protocol (diqds.protocol): distribution stage and messaging stage."""
from dataclasses import replace

import numpy as np
import pytest

from diqds.common import KEY_BITS_PER_SIGNATURE, N_S, Honest
from diqds.finite_size.protocols import ONE_SETTING, TWO_SETTING
from diqds.finite_size.renyi import RenyiSettings
from diqds.randomness import qutip_states as QS
from diqds.validation.aer_devices import TableDevice
from diqds.protocol import distribution as D
from diqds.protocol import messaging as M
from diqds.protocol.session import run_session

STEPS = ["1 parameters", "2 data acquisition", "3 sifting", "4 noisy preprocessing", "5 error correction",
         "6 parameter estimation", "7 privacy amplification"]


@pytest.fixture(scope="module")
def setup():
    hon = Honest.werner(0.98)
    plan = D.plan_link(400_000, hon, ONE_SETTING, q="opt", settings=RenyiSettings.cascade())
    return plan, TableDevice(QS.werner_probabilities(0.98))


@pytest.fixture(scope="module")
def distributed(setup):
    plan, dev = setup
    return D.distribution_stage(plan, dev, dev, np.random.default_rng(1))


def random_keys(rng):
    """Matching copies after an ideal distribution stage: (Alice, Bob, Charlie) key sets for one signature."""
    k_ab = M.partition(rng.integers(0, 2, KEY_BITS_PER_SIGNATURE, dtype=np.uint8), 0)
    k_ac = M.partition(rng.integers(0, 2, KEY_BITS_PER_SIGNATURE, dtype=np.uint8), 0)
    return M.alice_key_set(k_ab, k_ac), k_ab, k_ac


# ------------------------------------------------------------------------------------------- distribution stage
def test_distribution_stage_gives_matching_keys(setup, distributed):
    plan, _ = setup
    d = distributed
    assert d.accepted and d.reason == ""
    k = d.keys()
    assert np.array_equal(k["alice_ab"], k["bob"]) and np.array_equal(k["alice_ac"], k["charlie"])
    assert k["bob"].size == k["charlie"].size == plan.ell
    assert not np.array_equal(k["bob"], k["charlie"])                     # two independent links
    assert d.signatures == plan.ell // KEY_BITS_PER_SIGNATURE >= 1
    for link in (d.ab, d.ac):
        assert list(link.steps) == STEPS
        assert link.steps["7 privacy amplification"]["keys_agree"]
        assert link.steps["5 error correction"]["within_EC_max"] and link.steps["5 error correction"]["tag_match"]


def test_sifting_keeps_the_right_rounds():
    rng = np.random.default_rng(2)
    X, Y, T = D.choose_inputs(TWO_SETTING, 40_000, 0.1, rng)
    key = D.sifting(TWO_SETTING, X, Y, T)
    assert np.array_equal(key, (~T) & (X == Y))
    assert abs(key.mean() - 0.9 / 2) < 0.01                               # half of the generation rounds survive
    X, Y, T = D.choose_inputs(ONE_SETTING, 40_000, 0.1, rng)
    key = D.sifting(ONE_SETTING, X, Y, T)
    assert np.array_equal(key, ~T) and not X[key].any() and not Y[key].any()
    assert set(np.unique(Y[T])) <= {2, 3}


def test_noisy_preprocessing_flips_with_probability_p():
    rng = np.random.default_rng(3)
    bits = rng.integers(0, 2, 200_000, dtype=np.uint8)
    out, flips = D.noisy_preprocessing(bits, 0.1, rng)
    assert np.array_equal(out ^ bits, flips) and abs(flips.mean() - 0.1) < 0.003


def test_kgp_aborts_when_error_correction_exceeds_ec_max(setup):
    plan, dev = setup
    r = D.run_kgp(replace(plan, ec_budget=100.0), dev, np.random.default_rng(4))
    assert not r.accepted and r.reason == "error-correction budget" and r.key_alice is None


def test_error_verification_catches_a_residual_error():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 2, 5000, dtype=np.uint8)
    good = D.Reconciled(a[:1000], a[1000:], a[:1000].copy(), a[1000:].copy(), 0)
    bad_key = a[1000:].copy()
    bad_key[17] ^= 1
    bad = D.Reconciled(a[:1000], a[1000:], a[:1000].copy(), bad_key, 0)
    assert D.error_verification(good, rng) and not D.error_verification(bad, rng)


# ---------------------------------------------------------------------------------------------- messaging stage
def test_alice_keys_follow_the_base_paper_relations():
    rng = np.random.default_rng(6)
    alice, bob, charlie = random_keys(rng)
    assert alice.X == charlie.X and alice.Y == bob.Y ^ charlie.Y and alice.Z == charlie.Z


def test_honest_signature_and_message_order():
    rng = np.random.default_rng(7)
    alice, bob, charlie = random_keys(rng)
    doc = M.text_to_bits("Pay Bob 100 euro, invoice 42")
    ch = M.Channel()
    rec = M.messaging_stage(doc, alice, bob, charlie, ch)
    assert rec.accepted and not rec.repudiated
    route = [(m.sender, m.receiver, m.channel, sorted(m.content)) for m in rec.transcript]
    assert route == [("Alice", "Bob", "public", ["Doc", "Sig"]),
                     ("Bob", "Charlie", "authenticated", ["Doc", "Sig", "Y_b"]),
                     ("Charlie", "Bob", "authenticated", ["X_c", "Y_c", "Z_c"]),       # only after the forward
                     ("Bob", "Charlie", "authenticated", ["decision"])]
    assert rec.transcript[-1].content["decision"] == "accept"


def test_a_signature_for_another_document_is_rejected():
    rng = np.random.default_rng(8)
    alice, bob, charlie = random_keys(rng)
    sig = M.alice_sign(M.text_to_bits("Pay Bob 100 euro"), alice)
    rec = M.messaging_stage(M.text_to_bits("Pay Bob 900 euro"), None, bob, charlie, sig=sig)
    assert not rec.bob_accepts and not rec.charlie_accepts and "aborted" in rec.note


def test_forgery_needs_charlies_x_and_z(setup, distributed):
    rng = np.random.default_rng(9)
    d = distributed
    alice, bob, charlie = M.key_sets(d, 0)
    rec = M.messaging_stage(M.text_to_bits("Pay Bob 100 euro"), alice, bob, charlie)
    guesses = [M.forgery_attempt(rec, bob, charlie, rng).accepted for _ in range(300)]
    assert rec.accepted and not any(guesses)                                   # success 2^-46 per guess
    knows = M.KeySet(X=charlie.X, Y=0, Z=charlie.Z)                            # Y_c is not needed
    f = M.forgery_attempt(rec, bob, charlie, rng, knows, altered_doc=M.text_to_bits("Pay Bob 900 euro"))
    assert f.accepted and M.bits_to_text(f.doc) == "Pay Bob 900 euro"


def test_repudiation_needs_a_channel_error():
    rng = np.random.default_rng(10)
    alice, bob, charlie = random_keys(rng)
    doc = rng.integers(0, 2, 1000, dtype=np.uint8)
    for _ in range(50):                                   # whatever Alice sends, Bob and Charlie decide alike
        rec = M.messaging_stage(doc, None, bob, charlie, sig=int(rng.integers(0, 2 ** N_S)))
        assert not rec.repudiated
    good = M.messaging_stage(doc, alice, bob, charlie)
    assert good.accepted and not good.repudiated
    noisy = M.Channel(error_rate=1.0, fields=("Y_b",))   # Bob's forwarded Y_b arrives with one bit flipped
    rec = M.messaging_stage(doc, alice, bob, charlie, noisy)
    assert rec.bob_accepts and not rec.charlie_accepts and rec.repudiated and noisy.corrupted[0][1] == "Y_b"


def test_documents_longer_than_n_m_are_refused():
    alice, _, _ = random_keys(np.random.default_rng(11))
    with pytest.raises(ValueError):
        M.alice_sign(np.zeros(1001, np.uint8), alice)


def test_session_runs_both_stages(setup):
    plan, dev = setup
    texts = ["Pay Bob 100 euro", "Meeting moved to 3 pm", "Contract 7 approved"]
    s = run_session(plan, dev, dev, [M.text_to_bits(t) for t in texts], np.random.default_rng(12))
    assert s.distribution.accepted and s.signed == 3 and s.accepted == 3
    assert [M.bits_to_text(r.doc) for r in s.records] == texts
    assert len(s.channel.transcript) == 4 * 3
