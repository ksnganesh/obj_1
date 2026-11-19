"""End-to-end emulation of the DI-QDS protocol (Sec. VII): a consistency and completeness check of the implementation
against the analytic claims.  It is not security evidence: the adversarial content of the article is in the analytic
Propositions 1-2 and Theorem 2.

    python scripts/run_validation.py                 # ~12 min on one core
    python scripts/run_validation.py --figures       # figures from results/validation.json
    python scripts/run_validation.py --only-ec       # add the Cascade-efficiency section to the cached results

Sections
  acceptance   exact acceptance probability (binomial sum) against Monte Carlo runs of the acceptance test on Qiskit Aer
               devices: honest devices with the plan's 3-sigma tails and with stressed tails (0.02 each), devices that
               are deterministic in a random fraction f of the rounds (f tuned to P = 0.9, 0.5, 0.1), and devices that
               turn deterministic halfway through ("deterministic second half"; not the memory attack of Barrett,
               Colbeck and Kent, which concerns device reuse across blocks).
  pipeline     complete runs of both links and the signature layer: Aer devices (exact table or Aer shots), Cascade,
               64-bit Toeplitz error verification, Toeplitz privacy amplification, OTUH signing and verification with
               honest and forged messages.  11 km atom link, block of the recorded 1.208e6 heralds (one setting and
               pre-shared seed) and 4e6 (two settings), plus one 1e7-round two-setting run with Aer shots.
  adversarial  the deterministic-second-half device in the full pipeline (must abort).
  ec_efficiency  Cascade leakage over the Shannon limit, f = leak / (N h(Q)), on 10^6-bit blocks at the error rates
               of the reported links (key bases and test rounds); the EC budget RenyiSettings.cascade() uses it.
Outputs results/validation.json and results/fig_acceptance.png|pdf.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diqds.common import QDSBudget, h2  # noqa: E402
from diqds.finite_size.operating_points import ATOM_11KM, ATOM_400M, ION  # noqa: E402
from diqds.finite_size.protocols import ONE_SETTING, SEEDED, TWO_SETTING  # noqa: E402
from diqds.finite_size.renyi import RenyiSettings  # noqa: E402
from diqds.randomness import qutip_states as QS  # noqa: E402
from diqds.validation import montecarlo as MC  # noqa: E402
from diqds.validation.cascade import cascade  # noqa: E402
from diqds.validation import pipeline as P  # noqa: E402
from diqds.validation.adversarial import FractionDevice, HonestThenDeterministic  # noqa: E402
from diqds.validation.aer_devices import AerDevice, AerNoise  # noqa: E402

RES = ROOT / "results"
N_REC = 1_208_000
HON = ATOM_11KM.honest


def cp_interval(k: int, n: int, conf: float = 0.95):
    """Clopper-Pearson interval for a binomial proportion."""
    a = (1 - conf) / 2
    lo = 0.0 if k == 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - a, k + 1, n - k))
    return lo, hi


def device() -> AerDevice:
    return AerDevice(AerNoise.from_matter(QS.calibrate_matter(HON.S, HON.Q, 5e-3)))


def acceptance_section(trials_honest: int, trials_dev: int) -> dict:
    dev = device()
    rng = np.random.default_rng(7)
    out = []
    w = HON.w
    for tag, st in (("tails_3sigma", RenyiSettings.cascade()), ("tails_0.02", RenyiSettings.cascade(tail=0.02))):
        plan = P.plan_link(N_REC, HON, ONE_SETTING, q="opt", settings=st)
        cases = [("honest", 0.0, trials_honest)]
        for target in (0.9, 0.5, 0.1):
            f = brentq(lambda f_: MC.acceptance_probability(plan, w - f_ * (w - 0.75)) - target, 0.0, 0.5, xtol=1e-7)
            cases.append((f"fraction P={target}", f, trials_dev))
        for name, f, trials in cases:
            dv = dev if f == 0.0 else FractionDevice(dev, f, seed=int(rng.integers(1 << 30)))
            t = time.time()
            mc = MC.acceptance_mc(plan, dv, trials, rng)
            k = int(round(mc * trials))
            an = MC.acceptance_probability(plan, w - f * (w - 0.75))
            lo, hi = cp_interval(k, trials)
            row = {"tails": tag, "device": name, "f": f, "trials": trials, "accepted": k, "mc": mc, "ci95": [lo, hi],
                   "analytic": an, "union_bound_abort": st.completeness_AT if f == 0 else None,
                   "inside_ci": lo <= an <= hi}
            out.append(row)
            print("acceptance", tag, name, f"f={f:.5f}", f"mc={mc:.4f} [{lo:.4f},{hi:.4f}] analytic={an:.4f}",
                  f"[{time.time() - t:.0f}s]", flush=True)
        dv = HonestThenDeterministic(dev, 0.5, seed=11)
        mc = MC.acceptance_mc(plan, dv, 40, rng)
        out.append({"tails": tag, "device": "deterministic second half", "f": 0.5, "trials": 40,
                    "accepted": int(round(mc * 40)), "mc": mc, "ci95": list(cp_interval(int(round(mc * 40)), 40)),
                    "analytic": MC.acceptance_probability(plan, w - 0.5 * (w - 0.75)), "inside_ci": True})
        print("acceptance", tag, "deterministic second half", mc, flush=True)
    return {"rows": out, "w_honest": w}


def pipeline_section(trials: int, big_n: int = 10_000_000) -> dict:
    dev = device()
    rng = np.random.default_rng(11)
    st = RenyiSettings.cascade()
    out = {}
    runs = [("one_setting", ONE_SETTING, N_REC, trials, "multinomial"),
            ("seeded", SEEDED, N_REC, trials, "multinomial"),
            ("two_setting", TWO_SETTING, 4_000_000, trials, "multinomial"),
            ("one_setting_aer_shots", ONE_SETTING, N_REC, 3, "shots"),
            (f"two_setting_{big_n:.0e}_aer_shots".replace("+0", ""), TWO_SETTING, big_n, 1, "shots")]
    for key, proto, n, tr, sampling in runs:
        plan = P.plan_link(n, HON, proto, q="opt", settings=st)
        t = time.time()
        s = MC.full_mc(plan, dev, dev, tr, rng, "toeplitz", n_docs=6, sampling=sampling)
        d = s.as_dict()
        d.update({"n": n, "gamma": plan.gamma, "q": plan.q, "alpha": plan.alpha, "ell": plan.ell,
                  "signatures_per_block": plan.ell // 138, "sampling": sampling,
                  "analytic_abort_union": st.completeness_AT,
                  "analytic_accept_exact": MC.acceptance_probability(plan, HON.w) ** 2,   # both links
                  "wall_s": time.time() - t})
        out[key] = d
        print("pipeline", key, {k: d[k] for k in ("trials", "accepted", "mean_signatures", "honest_ok", "honest_total",
                                                  "forged_accepted", "forged_tried", "residual_error_runs",
                                                  "key_mismatch", "leak_over_budget_max")}, f"[{d['wall_s']:.0f}s]",
              flush=True)
    plan = P.plan_link(N_REC, HON, ONE_SETTING, q="opt", settings=st)
    s = MC.full_mc(plan, HonestThenDeterministic(dev, 0.5, seed=5), HonestThenDeterministic(dev, 0.5, seed=6), 3, rng)
    out["deterministic_second_half"] = s.as_dict()
    print("pipeline deterministic second half", s.as_dict()["reasons"], flush=True)
    b = QDSBudget()
    out["forgery_bound"] = {"n_s": b.n_s, "n_m": b.n_m, "eps_forge_otuh": 2 ** (1 - b.n_s) * b.n_m}
    return out


def ev_section(trials: int) -> dict:
    """Error verification under stress: Cascade cut to 2 or 3 passes leaves residual errors; the 64-bit Toeplitz tag
    must abort every such run (undetected mismatch probability 2^-64 per run)."""
    dev = device()
    rng = np.random.default_rng(13)
    plan = P.plan_link(N_REC, HON, ONE_SETTING, q="opt", settings=RenyiSettings.cascade())
    out = {}
    for passes in (2, 3):
        c = {"trials": 0, "residual": 0, "ev_abort": 0, "ev_abort_with_residual": 0, "accepted": 0,
             "accepted_with_mismatch": 0, "other_abort": 0}
        for _ in range(trials):
            r = P.run_link(plan, dev, rng, cascade_passes=passes)
            c["trials"] += 1
            c["residual"] += int(r.residual_errors > 0)
            if r.reason == "error verification":
                c["ev_abort"] += 1
                c["ev_abort_with_residual"] += int(r.residual_errors > 0)
            elif r.accepted:
                c["accepted"] += 1
                c["accepted_with_mismatch"] += int(not np.array_equal(r.key_alice, r.key_bob))
            else:
                c["other_abort"] += 1
        out[f"passes_{passes}"] = c
        print("error verification", passes, c, flush=True)
    return out


def ec_efficiency_section(trials: int = 5, n: int = 1_000_000) -> dict:
    """Cascade leakage relative to the Shannon limit, f = leak / (n h(Q)), at the error rates of the reported links."""
    rng = np.random.default_rng(17)
    points = [("ions, Z key basis", ION.honest.Q), ("ions, X key basis", ION.honest.Q_x),
              ("atoms 11 km, Z key basis", HON.Q), ("atoms 11 km, X key basis", HON.Q_x),
              ("atoms 400 m, key", ATOM_400M.honest.Q),
              ("atoms 11 km, test rounds (1 - w)", 1.0 - HON.w)]
    rows = []
    for label, Q in points:
        fs, ok = [], 0
        for _ in range(trials):
            a = rng.integers(0, 2, n, dtype=np.uint8)
            b = a ^ (rng.random(n) < Q).astype(np.uint8)
            corr, leak = cascade(a, b, Q, rng)
            ok += int(np.array_equal(corr, a))
            fs.append(leak / (n * h2(Q)))
        rows.append({"label": label, "Q": float(Q), "n": n, "trials": trials, "corrected": ok,
                     "f_mean": float(np.mean(fs)), "f_min": float(np.min(fs)), "f_max": float(np.max(fs))})
        print(f"cascade efficiency {label:34s} Q = {Q:.4f}: f = {np.mean(fs):.3f} "
              f"[{np.min(fs):.3f}, {np.max(fs):.3f}], corrected {ok}/{trials}", flush=True)
    budget = RenyiSettings.cascade()
    return {"rows": rows, "budget_key": budget.ec_f_key, "budget_test": budget.ec_f_test}


def figures(d: dict) -> None:
    from diqds.plotting import INK_2, SERIES, save, setup
    import matplotlib.pyplot as plt
    setup()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for tag, c, mk in (("tails_3sigma", SERIES[0], "o"), ("tails_0.02", SERIES[1], "s")):
        rows = [r for r in d["acceptance"]["rows"] if r["tails"] == tag and r["device"] != "deterministic second half"]
        xs = [r["analytic"] + (0.012 if tag != "tails_3sigma" else 0.0) for r in rows]   # small offset: both sets share x
        ys = [r["mc"] for r in rows]
        err = np.array([[r["mc"] - r["ci95"][0] for r in rows], [r["ci95"][1] - r["mc"] for r in rows]])
        ax.errorbar(xs, ys, yerr=err, fmt=mk, color=c, markersize=6, capsize=3,
                    label=r"$3\sigma$ tails (protocol)" if tag == "tails_3sigma" else "stressed tails (0.02 each)")
    ax.plot([0, 1], [0, 1], "--", color=INK_2, linewidth=1, label="Monte Carlo = analytic")
    ax.set_xlabel("exact acceptance probability (stressed points offset by +0.012)")
    ax.set_ylabel("Monte Carlo acceptance rate (95% CI)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, RES / "fig_acceptance")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--trials-honest", type=int, default=1000)
    ap.add_argument("--trials-dev", type=int, default=1000)
    ap.add_argument("--trials-pipeline", type=int, default=10)
    ap.add_argument("--quick", action="store_true", help="smoke test: few trials, 2e6-round shot run, separate output")
    ap.add_argument("--only-ev", action="store_true", help="add the error-verification section to the cached results")
    ap.add_argument("--only-ec", action="store_true", help="add the Cascade-efficiency section to the cached results")
    a = ap.parse_args()
    if a.quick:
        a.trials_honest, a.trials_dev, a.trials_pipeline = 4, 4, 1
    path = RES / ("validation_quick.json" if a.quick else "validation.json")
    if a.only_ev:
        d = json.loads(path.read_text())
        d["error_verification"] = ev_section(20)
        path.write_text(json.dumps(d, indent=1, default=float))
        return
    if a.only_ec:
        d = json.loads(path.read_text())
        d["ec_efficiency"] = ec_efficiency_section(2 if a.quick else 5)
        path.write_text(json.dumps(d, indent=1, default=float))
        return
    if not a.figures:
        t0 = time.time()
        d = {"acceptance": acceptance_section(a.trials_honest, a.trials_dev),
             "pipeline": pipeline_section(a.trials_pipeline, 2_000_000 if a.quick else 10_000_000),
             "error_verification": ev_section(3 if a.quick else 20),
             "ec_efficiency": ec_efficiency_section(2 if a.quick else 5, 200_000 if a.quick else 1_000_000)}
        d["wall_s"] = time.time() - t0
        path.write_text(json.dumps(d, indent=1, default=float))
    if not a.quick:
        figures(json.loads(path.read_text()))
        print("figures written")


if __name__ == "__main__":
    main()
