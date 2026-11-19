"""Sec. VI: whether DI signatures can skip privacy amplification (Propositions 1-2, Theorem 2), and the seed economy.

    python scripts/run_privacy_amplification.py            # ~4 min on one core; needs results/randomness_ledger.json
    python scripts/run_privacy_amplification.py --figures  # figures from results/privacy_amplification.json
    python scripts/run_privacy_amplification.py --only-seeds  # seed sections only, after a new randomness ledger

Sections
  seeds     seed bits per signature for every option (Toeplitz, modified Toeplitz, no PA with sampling, permutation) on
            every platform and protocol, at the first-signature block and at ten times that block (ledger parameters).
  window    Proposition 1: certified lower bound (coupling) and binomial estimate of the acceptance probability of the
            window device against the window length, Monte Carlo checks, and full DI-QDS runs with and without PA.
  iid       Proposition 2: the i.i.d. CHSH-optimal attack against publicly sampled segments.
  certify   Theorem 2: min-entropy per raw bit under that attack, S_half, q_half, and the smallest forging bound that
            any min-entropy argument can give for raw hash keys.
  family    seed bits per signature against block size for the 11 km atom link (Fig. 9), from the first-signature
            block of Table III.
Outputs results/privacy_amplification.json, results/fig_window_attack.png|pdf, fig_iid_attack.png|pdf, fig_seed.png|pdf.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diqds.common import KEY_BITS_PER_SIGNATURE, N_S  # noqa: E402
from diqds.finite_size.operating_points import ATOM_11KM  # noqa: E402
from diqds.finite_size.optimize import renyi_optimum  # noqa: E402
from diqds.finite_size.protocols import ONE_SETTING, PROTOCOLS  # noqa: E402
from diqds.finite_size.renyi import RenyiSettings, key_length  # noqa: E402
from diqds.randomness import ledger as LG  # noqa: E402
from diqds.randomness import qutip_states as QS  # noqa: E402
from diqds.privacy_amplification import no_pa as NP  # noqa: E402
from diqds.validation import montecarlo as MC  # noqa: E402
from diqds.validation import pipeline as P  # noqa: E402
from diqds.validation.adversarial import WindowDevice  # noqa: E402
from diqds.validation.aer_devices import AerDevice, AerNoise  # noqa: E402

RES = ROOT / "results"
PROTO = dict(PROTOCOLS) if isinstance(PROTOCOLS, dict) else {p.key: p for p in PROTOCOLS}
N_REC = 1_208_000                    # recorded 11 km block (Lu et al. 2026, 624 h)


def seeds_section() -> dict:
    led = json.loads((RES / "randomness_ledger.json").read_text())
    out = {}
    for plat, row in led.items():
        for pk, e in row.items():
            for lab in ("first", "x10"):
                d = e.get(f"{lab}_entropy")
                if not d or d["ell"] <= 0:
                    continue
                proto = PROTO[pk]
                n_in = proto.pa_input_bits(d["n"], d["gamma"])
                n_out = proto.output_bits(d["n"], d["ell"])
                opts = NP.seed_options(n_in, d["ell"], n_out)
                out[f"{plat}/{pk}/{lab}"] = {
                    "n": d["n"], "signatures": d["signatures"], "pa_input_bits": n_in, "pa_output_bits": n_out,
                    "private_input_per_signature": d["private_input"],
                    **{f"{k}_per_signature": v["per_signature"] for k, v in opts.items()},
                    "certified_segment_entropy_no_hash": opts["no_pa"]["certified_segment_entropy"],
                    "segment_raw_bits": opts["local_pa_sampled"]["segment_raw_bits"],
                    "local_sampled_over_global_modified": opts["local_pa_sampled"]["per_signature"]
                    / max(opts["modified_toeplitz"]["per_signature"], 1e-300)}
    print("seeds", len(out), "rows", flush=True)
    return out


def _device():
    m = QS.calibrate_matter(ATOM_11KM.honest.S, ATOM_11KM.honest.Q, 5e-3)
    return AerDevice(AerNoise.from_matter(m))


def window_section(trials_full: int = 10, trials_acc: int = 120) -> dict:
    dev = _device()
    rng = np.random.default_rng(2026)
    st = RenyiSettings.cascade()
    out = {"plans": {}, "capacity": {}, "mc_acceptance": [], "full_runs": {}}
    plans = {"q0": P.plan_link(N_REC, ATOM_11KM.honest, ONE_SETTING, q=0.0, settings=st),
             "qopt": P.plan_link(N_REC, ATOM_11KM.honest, ONE_SETTING, q="opt", settings=st)}
    w = ATOM_11KM.honest.w
    for lab, plan in plans.items():
        g = plan.gamma
        n_raw_key = plan.n * ONE_SETTING.key_rounds_fraction(g)
        out["plans"][lab] = {"n": plan.n, "gamma": g, "q": plan.q, "ell": plan.ell,
                             "signatures_with_pa": plan.ell // KEY_BITS_PER_SIGNATURE,
                             "signatures_claimed_without_pa": int(n_raw_key // KEY_BITS_PER_SIGNATURE),
                             "p_accept_honest": MC.acceptance_probability(plan, w),
                             "forge_prob_per_known_segment": (1 - plan.q) ** KEY_BITS_PER_SIGNATURE}
        Ws = np.unique(np.concatenate([[0], np.geomspace(100, 0.5 * plan.n, 60)]).astype(int))
        pacc = [MC.acceptance_probability(plan, w - (W / plan.n) * (w - 0.75)) for W in Ws]
        # largest window still accepted with probability >= 1/2 and >= 0.99 x honest (root of the binomial formula)
        pa_of = lambda W: MC.acceptance_probability(plan, w - (W / plan.n) * (w - 0.75))
        def w_at(target):
            if pa_of(0) < target:
                return 0
            hi = 1.0
            while pa_of(hi) >= target and hi < plan.n:
                hi *= 2
            return int(brentq(lambda W: pa_of(W) - target, 0.0, min(hi, plan.n), xtol=1.0))
        W_half, W_99 = w_at(0.5), w_at(0.99 * pacc[0])
        lb = [NP.window_acceptance_lower_bound(plan, int(W))["bound"] for W in Ws]
        lb_of = lambda W: NP.window_acceptance_lower_bound(plan, int(W))["bound"]
        def w_lb(target):
            lo_, hi_ = 0, 1
            while lb_of(hi_) >= target and hi_ < plan.n:
                lo_, hi_ = hi_, hi_ * 2
            while hi_ - lo_ > 1:
                mid = (lo_ + hi_) // 2
                lo_, hi_ = (mid, hi_) if lb_of(mid) >= target else (lo_, mid)
            return lo_
        W99_lb, Whalf_lb = w_lb(0.99 * pacc[0]), w_lb(0.5)
        out["capacity"][lab] = {"W": Ws.tolist(), "p_accept": pacc, "p_accept_lower_bound": lb, "W_half": W_half,
                                "W_99": W_99, "W_99_certified": W99_lb, "W_half_certified": Whalf_lb,
                                "lower_bound_at_317": NP.window_acceptance_lower_bound(plan, 317),
                                "segments_known_half": int(W_half * (1 - g) // KEY_BITS_PER_SIGNATURE),
                                "segments_known_99": int(W_99 * (1 - g) // KEY_BITS_PER_SIGNATURE),
                                "segments_known_99_certified": int(W99_lb * (1 - g) // KEY_BITS_PER_SIGNATURE),
                                "segments_known_half_certified": int(Whalf_lb * (1 - g) // KEY_BITS_PER_SIGNATURE)}
        print("window capacity", lab, {k: v for k, v in out["capacity"][lab].items() if k not in ("W", "p_accept")},
              flush=True)
    plan = plans["q0"]
    cap = out["capacity"]["q0"]
    for W in sorted({317, cap["W_99"], cap["W_half"], int(1.6 * cap["W_half"])}):
        if W <= 0:
            continue
        wdev = WindowDevice(dev, plan.n // 3, W, seed=int(rng.integers(1 << 30)))
        t = time.time()
        mc = MC.acceptance_mc(plan, wdev, trials_acc, rng)
        an = MC.acceptance_probability(plan, w - (W / plan.n) * (w - 0.75))
        out["mc_acceptance"].append({"W": W, "trials": trials_acc, "mc": mc, "analytic_iid_approx": an})
        print("window acceptance MC", W, mc, round(an, 4), f"[{time.time() - t:.0f}s]", flush=True)
    for pa in ("none", "toeplitz"):
        agg = {"trials": 0, "accepted": 0, "targeted_segments": 0, "targeted_forged": 0, "other_tried": 0,
               "other_forged": 0, "known_key_positions": [], "tests_in_window": [], "signatures": []}
        for _ in range(trials_full):
            r = NP.window_attack(plan, dev, rng, pa)
            agg["trials"] += 1
            if not r.accepted:
                continue
            agg["accepted"] += 1
            for k in ("targeted_segments", "targeted_forged", "other_tried", "other_forged"):
                agg[k] += getattr(r, k)
            agg["known_key_positions"].append(r.known_key_positions)
            agg["tests_in_window"].append(r.tests_in_window)
            agg["signatures"].append(r.signatures)
        agg["window_rounds"] = r.window_rounds
        out["full_runs"][pa] = agg
        print("window attack", pa, {k: v for k, v in agg.items() if not isinstance(v, list)}, flush=True)
    return out


def iid_section(trials: int = 200) -> dict:
    """Proposition 2: the i.i.d. CHSH-optimal attack against randomly sampled no-PA segments, per signature and per
    block, for the 11 km atom link and the ions at several block sizes; QuTiP check of p_g and an acceptance Monte
    Carlo of the attack device at the optimal score."""
    from diqds.finite_size.operating_points import ION
    from diqds.validation.aer_devices import TableDevice
    st = RenyiSettings.cascade()
    out = {"cases": [], "checks": {}}
    for e, ns in ((ATOM_11KM, (1.208e6, 2.416e6, 6.04e6, 1.208e7)), (ION, (1.5e6, 3.0e6, 1.5e7))):
        for n in ns:
            plan = P.plan_link(int(n), e.honest, ONE_SETTING, q="opt", settings=st)
            r = NP.iid_attack_on_sampling(plan)
            out["cases"].append({"platform": e.key, "n": n, "gamma": plan.gamma, "q": plan.q, "ell": plan.ell,
                                 "S_star": r.S_star, "p_accept": r.p_accept, "p_guess_bit": r.p_guess_bit,
                                 "p_guess_keys": r.p_guess_keys, "joint": r.joint, "breaks": bool(r.breaks),
                                 "segments": r.segments, "S_block": r.S_block, "block_joint": r.block_joint,
                                 "breaks_block": bool(r.breaks_block),
                                 "S_grid": r.S_grid.tolist(), "joint_grid": r.joint_grid.tolist()})
            print("iid attack", e.key, f"{n:.3g}", f"S*={r.S_star:.4f} joint={r.joint:.2e} block={r.block_joint:.2e}",
                  flush=True)
    c = out["cases"][0]
    plan = P.plan_link(1_208_000, ATOM_11KM.honest, ONE_SETTING, q="opt", settings=st)
    dev = TableDevice(NP.chsh_attack_table(c["S_star"]))
    rng = np.random.default_rng(31)
    mc = MC.acceptance_mc(plan, dev, trials, rng)
    out["checks"] = {"S_star": c["S_star"], "p_guess_formula": float(NP.p_guess_chsh(c["S_star"])),
                     "p_guess_qutip": NP.eve_guess_numeric(c["S_star"]), "acceptance_mc": mc, "trials": trials,
                     "acceptance_analytic": c["p_accept"]}
    print("iid checks", out["checks"], flush=True)
    return out


def certify_section() -> dict:
    """Theorem 2: what no device-independent argument can exceed for a raw segment, per platform."""
    from diqds.common import QDSBudget, h2
    from diqds.finite_size.entropy import rp
    from diqds.finite_size.operating_points import ATOM_400M, ION, QD_REALISTIC
    out = {"S_half": NP.S_HALF, "platforms": {}}
    for key, hon in (("ion", ION.honest), ("atom11km", ATOM_11KM.honest), ("atom400m", ATOM_400M.honest),
                     ("qd_realistic", QD_REALISTIC)):
        S = hon.S
        qh = NP.q_half(S)
        Qq = hon.Q + qh - 2 * qh * hon.Q
        row = {"S": S, "Q": hon.Q, "p_guess": float(NP.p_guess_chsh(S)), "h_min_q0": float(NP.h_min_attack(S)),
               "q_half": qh, "Q_at_q_half": Qq,
               "asymptotic_rate_one_setting_at_q_half": float(rp(hon.w, qh)) - float(h2(Qq)),
               "forging_bound_min_entropy_route": {f"{ns}": NP.certified_forging_bound_raw(ns, 1000, float(NP.h_min_attack(S)))
                                                   for ns in (46, 82, 200)}}
        out["platforms"][key] = row
        print("certify", key, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    # n_s needed by the min-entropy route at the q that makes h_min = h (only possible for h > 1/2)
    out["n_s_needed"] = {f"{h:g}": (math.log2(2 * 1000 / (QDSBudget().eps - QDSBudget().eps_snd)) / (2 * h - 1)
                                    if h > 0.5 else math.inf) for h in (0.45, 0.5, 0.51, 0.55, 0.6, 0.7, 0.8)}
    # the proportional share a sampling-lemma argument certifies for a 3 n_s-bit segment of the recorded blocks
    rec = json.loads((RES / "finite_size_renyi.json").read_text())["experiments"]
    out["sampling_share"] = {}
    for key in ("ion", "atom11km"):
        r = rec[key]["recorded"]["one_setting_qopt"]["result"]
        n_raw = r["n"] * (1 - r["gamma"])
        out["sampling_share"][key] = {"ell": r["ell"], "raw_bits": n_raw,
                                      "share_of_138": NP.certified_segment_entropy_without_hashing(r["ell"], n_raw)}
    return out


def family_section() -> dict:
    """Seed bits per signature against block size on the 11 km link (Fig. 9), from the first-signature block of
    Table III (finite_size_renyi.json; the first signature pays for the whole seed) to 100 times that block."""
    hon = ATOM_11KM.honest
    stored = json.loads((RES / "finite_size_renyi.json").read_text())["experiments"]["atom11km"]["rows"]["one_setting_qopt"]
    nf, P = stored["n_min"], stored["params"]
    rows = []
    for k in (1, 2, 5, 10, 30, 100):
        n = nf * k
        if k == 1:                                   # the threshold itself, with its certified parameters
            r = key_length(n, hon, ONE_SETTING, P["gamma"], P["alpha"], P["q"], RenyiSettings(), certify=True)
        else:
            r = renyi_optimum(n, hon, ONE_SETTING, "opt")
        n_in = ONE_SETTING.pa_input_bits(n, r.gamma)
        opts = NP.seed_options(n_in, r.ell, ONE_SETTING.output_bits(n, r.ell))
        led = LG.block_ledger(ONE_SETTING, n, r.gamma, r.q, r.ell)
        rows.append({"n": n, "signatures": r.ell / KEY_BITS_PER_SIGNATURE, "gamma": r.gamma,
                     "private_input_per_signature": led.per_signature()["private_input"],
                     **{k_: v["per_signature"] for k_, v in opts.items()}})
        print("family", f"{n:.3g}", {k_: round(v, 1) for k_, v in rows[-1].items()}, flush=True)
    return {"platform": "atom11km", "protocol": "one_setting", "n_first": nf, "rows": rows}


def figures(d: dict) -> None:
    from diqds.plotting import INK_2, SERIES, save, setup
    import matplotlib.pyplot as plt
    setup()
    cap = d["window"]["capacity"]["q0"]
    plan = d["window"]["plans"]["q0"]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    W = np.array(cap["W"], float)
    ax.plot(np.maximum(W, 50), cap["p_accept"], "-", color=SERIES[0], label="acceptance probability (binomial estimate)")
    if "p_accept_lower_bound" in cap:
        ax.plot(np.maximum(W, 50), cap["p_accept_lower_bound"], "--", color=SERIES[2],
                label="certified lower bound (coupling)")
    mc = [m for m in d["window"]["mc_acceptance"]]
    ax.plot([m["W"] for m in mc], [m["mc"] for m in mc], "o", color=SERIES[1], markersize=6,
            label="Monte Carlo, window device (Aer)")
    for m in mc:
        segs = int(m["W"] * (1 - plan["gamma"]) // KEY_BITS_PER_SIGNATURE)
        if m["mc"] > 0.9 and m["W"] > 5000:          # where the curves start to fall: label to the lower left
            xy, ha = (-8, -26), "right"
        else:
            xy, ha = (6, -22 if m["mc"] > 0.5 else 6), "left"
        ax.annotate(f"{segs} segments\nknown to Eve", (m["W"], m["mc"]), textcoords="offset points",
                    xytext=xy, ha=ha, fontsize=7, color=INK_2)
    ax.set_xscale("log")
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(r"length of the deterministic window, $W$ (rounds)")
    ax.set_ylabel("probability that the test passes")
    ax.legend(loc="lower left", fontsize=8)
    save(fig, RES / "fig_window_attack")
    if "iid" in d:
        fig, ax = plt.subplots(figsize=(6.4, 3.7))
        cols = {1.208e6: SERIES[0], 2.416e6: SERIES[1], 6.04e6: SERIES[2], 1.208e7: SERIES[3]}
        for c in d["iid"]["cases"]:
            if c["platform"] != "atom11km":
                continue
            ax.plot(c["S_grid"], c["joint_grid"], "-", color=cols.get(c["n"], INK_2),
                    label=f"{c['n'] / 1.208e6:.0f}× the recorded block")
            ax.plot([c["S_star"]], [c["joint"]], "o", color=cols.get(c["n"], INK_2), markersize=6)
        ax.axhline(1e-10, color=INK_2, linestyle="--", linewidth=1)
        ax.text(2.405, 1.3e-10, r"$\varepsilon = 10^{-10}$", fontsize=8, color=INK_2)
        ax.set_yscale("log")
        ax.set_ylim(1e-16, 1e-8)
        ax.set_xlim(2.4, 2.7)
        ax.set_xlabel(r"CHSH score $S$ of the i.i.d. attack device")
        ax.set_ylabel(r"$P_{\rm acc}(S)\,p_g(S)^{2n_s}$")
        ax.legend(fontsize=7.5, loc="upper right")
        save(fig, RES / "fig_iid_attack")
    fam = d["family"]["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 3.7))
    xs = [r["signatures"] for r in fam]
    series = (("toeplitz", "Toeplitz seed, one per block", SERIES[0], "o", "-"),
              ("modified_toeplitz", "modified Toeplitz seed, one per block", SERIES[2], "s", "-"),
              ("local_pa_sampled", "per-signature hash of a sampled subset", SERIES[1], "D", "-"),
              ("private_input_per_signature", "private inputs (for comparison)", INK_2, "", "--"))
    for k, lab, c, mk, ls in series:
        ax.plot(xs, [r[k] for r in fam], ls, marker=mk or None, color=c, label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("signatures per block (11 km atom link, one setting)")
    ax.set_ylabel("bits per signature")
    ax.legend(fontsize=8)
    save(fig, RES / "fig_seed")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--trials", type=int, default=10)
    ap.add_argument("--only-iid", action="store_true", help="add the Proposition 2 section to the cached results")
    ap.add_argument("--only-certify", action="store_true", help="add the Theorem 2 section to the cached results")
    ap.add_argument("--only-seeds", action="store_true",
                    help="recompute the seed sections (Table X, Fig. 9) after a new randomness ledger")
    a = ap.parse_args()
    path = RES / "privacy_amplification.json"
    if a.only_iid or a.only_certify or a.only_seeds:
        d = json.loads(path.read_text())
        if a.only_iid:
            d["iid"] = iid_section()
        if a.only_certify:
            d["certify"] = certify_section()
        if a.only_seeds:
            d["seeds"] = seeds_section()
            d["family"] = family_section()
        path.write_text(json.dumps(d, indent=1, default=float))
    elif not a.figures:
        t0 = time.time()
        d = {"seeds": seeds_section(), "window": window_section(a.trials), "iid": iid_section(),
             "certify": certify_section(), "family": family_section()}
        d["notes"] = {"n_s": N_S, "key_bits_per_signature": KEY_BITS_PER_SIGNATURE, "wall_s": time.time() - t0}
        path.write_text(json.dumps(d, indent=1, default=float))
    figures(json.loads(path.read_text()))
    print("figures written")


if __name__ == "__main__":
    main()
