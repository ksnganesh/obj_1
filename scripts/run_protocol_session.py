"""One complete DI-QDS session, stage by stage (Fan et al. Sec. II): distribution stage, then messaging stage.

Devices: Qiskit Aer devices calibrated to the 11 km atom link (S = 2.612, Q = 2.85 %), one pair per link.
Default protocol: Fan et al.'s two-setting protocol with sifting, at 4e6 rounds per link (it needs ~1.6e6 rounds
for a first signature); --protocol one_setting runs the spot-checking protocol at the 1.208e6 recorded rounds.

    python scripts/run_protocol_session.py                       # base-paper protocol, three text documents
    python scripts/run_protocol_session.py --protocol one_setting
    python scripts/run_protocol_session.py --doc "Pay Bob 100 euro" --doc "Second document"

Prints every step and writes results/protocol_session.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diqds.common import KEY_BITS_PER_SIGNATURE, N_S  # noqa: E402
from diqds.finite_size.operating_points import ATOM_11KM  # noqa: E402
from diqds.finite_size.protocols import ONE_SETTING, SEEDED, TWO_SETTING  # noqa: E402
from diqds.finite_size.renyi import RenyiSettings  # noqa: E402
from diqds.randomness import qutip_states as QS  # noqa: E402
from diqds.validation.aer_devices import AerDevice, AerNoise  # noqa: E402
from diqds.protocol import distribution as D  # noqa: E402
from diqds.protocol import messaging as M  # noqa: E402

RES = ROOT / "results"
HON = ATOM_11KM.honest
PROTOCOLS = {"two_setting": (TWO_SETTING, 4_000_000), "one_setting": (ONE_SETTING, 1_208_000),
             "seeded": (SEEDED, 1_208_000)}
DOCS = ["Pay Bob 100 euro, invoice 42", "Meeting moved to 3 pm", "Contract 7 approved by Alice"]
HEX = "0x{:012x}"


def show_distribution(dist: D.DistributionResult) -> None:
    ab, ac = dist.ab.steps, dist.ac.steps
    rows = [("1 parameters", "gamma (test probability)", lambda s: f"{s['1 parameters']['gamma']:.4f}"),
            ("", "p (noisy preprocessing)", lambda s: f"{s['1 parameters']['p']:.4f}"),
            ("", "EC_max (bits)", lambda s: f"{s['1 parameters']['EC_max']:,.0f}"),
            ("", "ell_key (bits)", lambda s: f"{s['1 parameters']['ell_key']:,}"),
            ("2 data acquisition", "test rounds", lambda s: f"{s['2 data acquisition']['test_rounds']:,}"),
            ("", "generation rounds", lambda s: f"{s['2 data acquisition']['generation_rounds']:,}"),
            ("3 sifting", "key rounds kept", lambda s: f"{s['3 sifting']['key_rounds']:,}"),
            ("", "generation rounds dropped", lambda s: f"{s['3 sifting']['discarded_generation_rounds']:,}"),
            ("4 noisy preprocessing", "key bits flipped", lambda s: f"{s['4 noisy preprocessing']['flipped_bits']:,}"),
            ("", "key error rate", lambda s: f"{100 * s['4 noisy preprocessing']['qber_key']:.2f} %"),
            ("5 error correction", "bits disclosed (Cascade)",
             lambda s: f"{s['5 error correction']['bits_disclosed']:,}"),
            ("", "64-bit hash tag", lambda s: "match" if s['5 error correction'].get('tag_match') else "MISMATCH"),
            ("6 parameter estimation", "test rounds won / lost",
             lambda s: f"{s['6 parameter estimation']['won']:,} / {s['6 parameter estimation']['lost']:,}"),
            ("", "acceptance test", lambda s: "passed" if s['6 parameter estimation']['passed'] else "FAILED"),
            ("7 privacy amplification", "final key bits", lambda s: f"{s['7 privacy amplification']['key_bits']:,}"),
            ("", "keys of both parties equal", lambda s: str(s['7 privacy amplification']['keys_agree']))]
    print(f"\n{'':24s}{'':28s}{'Alice-Bob':>18s}{'Alice-Charlie':>18s}")
    for step, what, f in rows:
        try:
            a, c = f(ab), f(ac)
        except KeyError:
            a = c = "-"
        print(f"{step:24s}{what:28s}{a:>18s}{c:>18s}")


def show_record(rec: M.SignatureRecord, text: str) -> None:
    print(f'\n  document {rec.index + 1}: "{text}"  ({rec.doc.size} bits, key segment {rec.index})')
    if rec.sig is None:
        print("   ", rec.note)
        return

    def msg(m: M.Message, label: str = "") -> None:
        parts = [k if k == "Doc" else f"{k} = {HEX.format(v)}" if isinstance(v, int) else f"{k}: {v}"
                 for k, v in m.content.items()]
        print(f"    {label:17s}{m.sender:>7s} -> {m.receiver:<8s}{m.channel:14s} {{{', '.join(parts)}}}")

    def check(label: str, who: str, d, ok: bool) -> None:
        actual = "degenerate key" if d[1] is None else HEX.format(d[1])
        rel = "=" if ok else "!="
        print(f"    {label:17s}{who + ':':9s}expected digest {HEX.format(d[0])} {rel} actual digest {actual}"
              f"  -> {'ACCEPTS' if ok else 'REJECTS'}")

    t = rec.transcript
    msg(t[0], "1 signing")
    msg(t[1], "2 verification")
    msg(t[2])
    check("", "Bob", rec.bob_digests, rec.bob_accepts)
    msg(t[3])
    if rec.charlie_digests is None:
        print(f"    {'3 certification':17s}not reached: Bob rejected and the protocol aborted")
    else:
        check("3 certification", "Charlie", rec.charlie_digests, rec.charlie_accepts)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--protocol", choices=sorted(PROTOCOLS), default="two_setting")
    ap.add_argument("--n", type=int, default=None, help="rounds per link (default: 4e6 two_setting, 1.208e6 otherwise)")
    ap.add_argument("--doc", action="append", default=None, help="a document to sign (text, at most 125 bytes)")
    ap.add_argument("--pa", choices=["toeplitz", "modified"], default="toeplitz")
    ap.add_argument("--sampling", choices=["multinomial", "shots"], default="multinomial")
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    proto, n_default = PROTOCOLS[a.protocol]
    n = a.n or n_default
    docs = a.doc or DOCS
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    print(f"DI-QDS session: {proto.name}; atoms 11 km (S = {HON.S:.3f}, Q = {100 * HON.Q:.2f} %); "
          f"{n:,} rounds per link")
    plan = D.plan_link(n, HON, proto, q="opt", settings=RenyiSettings.cascade())
    ideal = D.plan_link(n, HON, proto, q="opt", settings=RenyiSettings())
    print(f"EC_max is sized for this code's Cascade (f = {plan.settings.ec_f_key:.2f} on key rounds, "
          f"{plan.settings.ec_f_test:.2f} on test rounds); with error correction at the Shannon limit, as in the "
          f"article's tables,\nthe same rounds would give {ideal.ell // KEY_BITS_PER_SIGNATURE:,} signatures.")
    calib = QS.calibrate_matter(HON.S, HON.Q, 5e-3)
    dev_ab, dev_ac = AerDevice(AerNoise.from_matter(calib)), AerDevice(AerNoise.from_matter(calib))   # separate devices

    print("\nDISTRIBUTION STAGE (Fan et al. Sec. II A): one key-generation protocol per link, on separate devices")
    dist = D.distribution_stage(plan, dev_ab, dev_ac, rng, a.pa, a.sampling)
    show_distribution(dist)
    if not dist.accepted:
        print(f"\nThe distribution stage aborted ({dist.reason}); no signatures.")
    else:
        print(f"\n=> K_AB and K_AC: {dist.ab.key_alice.size:,} and {dist.ac.key_alice.size:,} bits, enough for "
              f"{dist.signatures:,} signatures ({KEY_BITS_PER_SIGNATURE} bits of each link per signature)")

    records, checks = [], {}
    if dist.accepted:
        print("\nMESSAGING STAGE (Fan et al. Sec. II B): keys per signature X_a = X_c, Y_a = Y_b xor Y_c, Z_a = Z_c")
        print("  Sig = H(f(Z_a), X_a) Doc xor Y_a;  a recipient compares the expected digest Sig xor Y_b xor Y_c")
        print("  with the actual digest H(f(Z_c), X_c) Doc")
        ch = M.Channel()
        for j, text in enumerate(docs[:dist.signatures]):
            alice, bob, charlie = M.key_sets(dist, j)
            rec = M.messaging_stage(M.text_to_bits(text), alice, bob, charlie, ch, index=j)
            show_record(rec, text)
            records.append((rec, text))

        print("\nSECURITY CHECKS (Fan et al. Sec. III), on document 1")
        rec0, text0 = records[0]
        alice, bob, charlie = M.key_sets(dist, 0)
        altered = text0.replace("100", "900") if "100" in text0 else text0 + "!"
        f1 = M.forgery_attempt(rec0, bob, charlie, rng, altered_doc=M.text_to_bits(altered))
        yes = lambda ok: "accepts" if ok else "rejects"
        print(f'  forgery      Bob forwards "{altered}" and Sig xor t with a guessed t')
        print(f"               -> Charlie {yes(f1.accepted)}   (a guess succeeds with probability 2^-{N_S})")
        f2 = M.forgery_attempt(rec0, bob, charlie, rng, knows=M.KeySet(charlie.X, 0, charlie.Z),
                               altered_doc=M.text_to_bits(altered))
        print("               the same change when Bob knows X_c and Z_c (possible without privacy amplification,")
        print(f"               Sec. VI of the article): t = h(Delta) -> Charlie {yes(f2.accepted)}")
        r1 = M.messaging_stage(rec0.doc, None, bob, charlie, sig=int(rng.integers(0, 2 ** N_S)))
        print("  repudiation  Alice sends a random signature")
        print(f"               -> Bob {yes(r1.bob_accepts)}, Charlie {yes(r1.charlie_accepts)}   "
              f"(Alice repudiates only if Bob accepts and Charlie rejects: {r1.repudiated})")
        noisy = M.Channel(error_rate=1.0, fields=("Y_b",), seed=a.seed)
        r2 = M.messaging_stage(rec0.doc, alice, bob, charlie, noisy)
        print("               honest signature, but Y_b corrupted on its way to Charlie (a channel error)")
        print(f"               -> Bob {yes(r2.bob_accepts)}, Charlie {yes(r2.charlie_accepts)}   "
              f"(why eps_rep = 2 eps in Eq. 6 of Fan et al.)")
        checks = {"forgery_guess": {"altered": altered, "charlie_accepts": f1.accepted},
                  "forgery_knowing_Xc_Zc": {"altered": altered, "charlie_accepts": f2.accepted},
                  "repudiation_random_sig": {"bob": r1.bob_accepts, "charlie": r1.charlie_accepts,
                                             "repudiated": r1.repudiated},
                  "repudiation_corrupted_Yb": {"bob": r2.bob_accepts, "charlie": r2.charlie_accepts,
                                               "repudiated": r2.repudiated}}

    def link(r: D.LinkRun) -> dict:
        return {"accepted": r.accepted, "reason": r.reason, "steps": r.steps}

    out = {"protocol": proto.key, "n": n, "seed": a.seed, "pa": a.pa, "sampling": a.sampling,
           "signatures_at_shannon_limit": ideal.ell // KEY_BITS_PER_SIGNATURE,
           "operating_point": {"S": HON.S, "Q": HON.Q, "Q_x": HON.Q_x},
           "distribution": {"accepted": dist.accepted, "signatures": dist.signatures,
                            "alice_bob": link(dist.ab), "alice_charlie": link(dist.ac)},
           "messaging": [{"document": text, "bits": int(rec.doc.size), "segment": rec.index,
                          "signature": None if rec.sig is None else HEX.format(rec.sig),
                          "bob_accepts": rec.bob_accepts, "charlie_accepts": rec.charlie_accepts,
                          "bob_digests": None if rec.bob_digests is None else [HEX.format(v) if v is not None else None
                                                                               for v in rec.bob_digests],
                          "charlie_digests": None if rec.charlie_digests is None else
                          [HEX.format(v) if v is not None else None for v in rec.charlie_digests],
                          "messages": [{"step": m.step, "from": m.sender, "to": m.receiver, "channel": m.channel,
                                        "fields": sorted(m.content)} for m in rec.transcript]}
                         for rec, text in records],
           "security_checks": checks, "wall_s": time.time() - t0}
    RES.mkdir(exist_ok=True)
    (RES / "protocol_session.json").write_text(json.dumps(out, indent=1, default=float))
    print(f"\nwritten results/protocol_session.json  [{time.time() - t0:.0f} s]")


if __name__ == "__main__":
    main()
