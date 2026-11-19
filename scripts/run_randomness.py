"""Trusted-randomness ledger (Sec. V): platform models (QuTiP, Qiskit Aer, SeQUeNCe, photon-number model), the ledger
of each protocol on each platform, and Pareto fronts of block size against random bits per signature.

    python scripts/run_randomness.py              # everything (~40 min on one core); cached in results/randomness_*.json
    python scripts/run_randomness.py --figures    # figures from the cache only

Outputs: results/randomness_platforms.json (QuTiP/Aer calibration, photonic operating points, extra-wait sensitivity,
SeQUeNCe attempts per herald), results/randomness_ledger.json (the cost of the first signature = the whole threshold
block, and the cost per signature at ten times that block), results/randomness_pareto.json, results/fig_pareto.png|pdf,
results/fig_ledger.png|pdf.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diqds.common import Honest  # noqa: E402
from diqds.finite_size.operating_points import ATOM_11KM, ION  # noqa: E402
from diqds.finite_size.optimize import n_min, renyi_optimum, threshold  # noqa: E402
from diqds.finite_size.renyi import RenyiSettings, key_length  # noqa: E402
from diqds.finite_size.operating_points import ATOM11_T2_CONSERVATIVE  # noqa: E402
from diqds.finite_size.protocols import INDEPENDENT, ONE_SETTING, SEEDED, TWO_SETTING  # noqa: E402
from diqds.randomness import ledger as LG  # noqa: E402
from diqds.randomness import pareto as PA  # noqa: E402
from diqds.randomness import platforms as PL  # noqa: E402
from diqds.randomness import qutip_states as QS  # noqa: E402
from diqds.randomness import sequence_links as SL  # noqa: E402

RES = ROOT / "results"
PROTOS = (TWO_SETTING, ONE_SETTING, INDEPENDENT, SEEDED)
PER_ATTEMPT_OK = (TWO_SETTING.key, INDEPENDENT.key)    # run per attempt without pre-shared secret randomness
PLATS = (PL.ION_PLATFORM, PL.ATOM11_PLATFORM, PL.photonic_platform(0.0), PL.photonic_platform(10.0))
EPS_R = {"ion": 1e-3, "atom11km": 5e-3}         # assumed readout errors (the rest of the noise is fitted)


def platforms_section() -> dict:
    from diqds.validation.aer_devices import AerDevice, AerNoise
    out = {"calibration": {}, "extra_wait": {}, "sequence": {}}
    for key, exp in (("ion", ION), ("atom11km", ATOM_11KM)):
        m = QS.calibrate_matter(exp.honest.S, exp.honest.Q, EPS_R[key])
        S, w, Q = QS.statistics(m.probabilities())
        dev = AerDevice(AerNoise.from_matter(m))
        T1, T2, tg = m.aer_dephasing()
        out["calibration"][key] = {"S_target": exp.honest.S, "Q_target": exp.honest.Q, "Q_x": exp.honest.Q_x,
                                   "eps_r_assumed": EPS_R[key], "lam": m.lam, "mu": m.mu, "S_qutip": S, "Q_qutip": Q,
                                   "aer_T2_for_1us_id": T2, "max_abs_aer_minus_qutip": float(np.abs(dev.P - m.probabilities()).max())}
        L = 11e3 if key == "atom11km" else 2.0
        out["extra_wait"][key] = {f"{T2m:g}": dict(zip(("S", "w", "Q"), map(float, QS.statistics(
            m.probabilities(extra_wait=L / 2e8, T2_memory=T2m))))) for T2m in (1e-3, 1e-2, ATOM11_T2_CONSERVATIVE, 1e-1)}
        # rounds to the first signature of the one-setting protocol after that wait (no wait: finite_size_renyi.json)
        for T2m, st in out["extra_wait"][key].items():
            st["n_min_one_setting"] = n_min(Honest.from_chsh(st["S"], st["Q"]), ONE_SETTING, "opt")
        print("extra wait", key, {k: (round(v["S"], 4), round(v["n_min_one_setting"])) for k, v in out["extra_wait"][key].items()},
              flush=True)
    from diqds.finite_size import photonic as PH
    out["photonic_model"] = {}
    for src in ("ideal", "realistic"):
        op = PH.operating_point(src, 0.0)
        out["photonic_model"][src] = {"S": op.chsh, "Q": op.qber, "P_H": op.PH, "Q_bases": list(op.q_bases),
                                      "angles_alice": list(op.angles_alice), "angles_bob": list(op.angles_bob)}
    for lab, S_t in (("qd_ideal", 2.631), ("qd_realistic", 2.450)):
        pn = QS.calibrate_photonic(S_t)
        S, w, Q = QS.statistics(pn.probabilities())
        dev = AerDevice(AerNoise.from_photonic(pn))
        out["calibration"][lab] = {"S_target": S_t, "v_eff": pn.v_eff, "eta_l": pn.eta_l, "S_qutip": S, "Q_qutip": Q,
                                   "Q_photon_number_model": out["photonic_model"]["ideal" if lab == "qd_ideal" else "realistic"]["Q"],
                                   "max_abs_aer_minus_qutip": float(np.abs(dev.P - pn.probabilities()).max())}
    for p in PLATS:
        mode, nh = ("heralds", 20000) if p.per_attempt_inputs else ("attempts", 400)
        t = time.time()
        r = SL.simulate(p, 1.0, nh, mode, seed=5)
        out["sequence"][p.key] = {"mode": mode, "heralds": r.heralds, "attempts_per_herald_sim": r.mean_attempts,
                                  "attempts_per_herald_analytic": p.attempts_per_herald, "herald_rate_sim": r.herald_rate_hz,
                                  "herald_rate_analytic": p.herald_rate_hz, "draws_per_herald_sim": r.bits_per_herald,
                                  "per_attempt_inputs": p.per_attempt_inputs, "arm_loss": r.channel_loss,
                                  "arm_delay_us": r.channel_delay_s * 1e6, "wall_s": time.time() - t}
        print("sequence", p.key, out["sequence"][p.key], flush=True)
    return out


def get_stored_threshold(platform_key: str, protocol_key: str):
    """Threshold of Table III for the ion and 11 km platforms (finite_size_renyi.json), so that both tables agree."""
    f = RES / "finite_size_renyi.json"
    if not f.exists():
        return None
    row = json.loads(f.read_text()).get("experiments", {}).get(platform_key, {}).get("rows", {}).get(f"{protocol_key}_qopt")
    return row if row and math.isfinite(row.get("n_min", math.inf)) else None


def ledger_section() -> dict:
    """Cost of the first signature (the whole threshold block) and per signature at ten times that block."""
    out = {}
    for p in PLATS:
        row = {}
        for proto in PROTOS:
            stored = get_stored_threshold(p.key, proto.key)
            if stored is not None:                 # same threshold and parameters as Table III (finite_size_renyi.json)
                nf, prm = stored["n_min"], stored["params"]
                r_first = key_length(nf, p.honest, proto, prm["gamma"], prm["alpha"], prm["q"], RenyiSettings(), certify=True)
            else:
                th = threshold(p.honest, proto, "opt")
                nf, r_first = th.n, th.result
            if not math.isfinite(nf):
                row[proto.key] = {"n_first": math.inf}
                continue
            entry = {"n_first": nf, "ell_first": r_first.ell, "hours_first": p.hours_for(nf),
                     "feasible": LG.first_signature_feasible(proto, p.per_attempt_inputs)}
            r10 = renyi_optimum(nf * 10, p.honest, proto, "opt")
            for lab, (n, r) in (("first", (nf, r_first)), ("x10", (nf * 10, r10))):
                for sampler in ("entropy", "dyadic8"):
                    led = LG.block_ledger(proto, n, r.gamma, r.q, max(r.ell, 0.0),
                                          attempts_per_round=p.attempts_per_herald if p.per_attempt_inputs else 1.0,
                                          per_attempt=p.per_attempt_inputs, sampler=sampler)
                    entry[f"{lab}_{sampler}"] = {"n": n, "gamma": r.gamma, "q": r.q, "ell": r.ell,
                                                 "signatures": led.signatures, **led.per_signature(),
                                                 "block": led.block_totals(),
                                                 "shared_secret_per_signature_gross": led.shared_secret / max(led.signatures, 1e-300)}
            row[proto.key] = entry
            print("ledger", p.key, proto.key, round(nf), "first block private",
                  f"{entry['first_entropy']['block']['private_input']:.3g}", "x10 per sig",
                  round(entry["x10_entropy"]["private_input"], 1), flush=True)
        out[p.key] = row
    return out


def pareto_section(ledger: dict) -> dict:
    out = {}
    for p in PLATS:
        fronts = []
        for proto in PROTOS:
            if p.per_attempt_inputs and proto.key not in PER_ATTEMPT_OK:
                continue                       # needs pre-shared secret randomness per attempt (ledger.first_signature_feasible)
            nf = ledger[p.key][proto.key]["n_first"]
            fr = PA.protocol_front(p, proto, "private", decades=2.0, npts=6, n_first=nf)
            fronts.append(fr)
            print("pareto", p.key, proto.key, [(f"{q.n:.3g}", round(q.bits_per_signature, 1)) for q in fr.points], flush=True)
        out[p.key] = {"fronts": [{"protocol": f.protocol, "n_first": f.n_first,
                                  "points": [vars(q) for q in f.points], "key_max_points": [vars(q) for q in f.key_max_points]}
                                 for f in fronts], "envelope": PA.envelope(fronts)}
    return out


def figures(plat: dict, ledger: dict, pareto: dict) -> None:
    from diqds.plotting import INK_2, SERIES, save, setup
    import matplotlib.pyplot as plt
    setup()
    colors = {"two_setting": SERIES[0], "one_setting": SERIES[1], "seeded": SERIES[2], "independent": SERIES[3]}
    names = {"two_setting": "two settings, sifted", "one_setting": "one setting, common flag", "seeded": "pre-shared seed",
             "independent": "one setting, independent flags"}
    markers = {"two_setting": "o", "one_setting": "D", "seeded": "s", "independent": "^"}
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 6.4))
    for ax, p in zip(axes.flat, PLATS):
        d = pareto[p.key]
        for f in d["fronts"]:
            pts = [q for q in f["points"] if math.isfinite(q["bits_per_signature"])]
            km = [q for q in f["key_max_points"] if math.isfinite(q["bits_per_signature"])]
            c = colors[f["protocol"]]
            ax.plot([q["n"] for q in pts], [q["bits_per_signature"] for q in pts], "-", marker=markers[f["protocol"]],
                    color=c, label=names[f["protocol"]] + ", randomness-optimal")
            ax.plot([q["n"] for q in km], [q["bits_per_signature"] for q in km], ":", color=c, linewidth=1.4,
                    label=names[f["protocol"]] + ", key-optimal")
        ax.set_xscale("log")
        ax.set_yscale("log")
        titles = {"ion": "ions, 2 m", "atom11km": "atoms, 11 km", "qd_realistic_0km": "photonic (QD source), 0 km",
                  "qd_realistic_10km": "photonic (QD source), 10 km"}
        ax.text(0.03, 0.04, titles.get(p.key, p.name), transform=ax.transAxes, fontsize=8.5, color=INK_2)
        ax.set_xlabel(r"block size $n$ (heralded rounds)", fontsize=8)
        ax.set_ylabel("private input bits per signature", fontsize=8)
        if p.per_attempt_inputs:
            ax.text(0.97, 0.95, "inputs drawn per attempt;\ncommon-flag and seeded protocols\nneed pre-shared secrets per attempt",
                    transform=ax.transAxes, fontsize=7, color=INK_2, ha="right", va="top")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7.5)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    save(fig, RES / "fig_pareto", tight=False)
    # ledger: grouped bars (platform groups, one bar per protocol), at the first signature and at 10x that block
    plat_names = {"ion": "ions\n2 m", "atom11km": "atoms\n11 km", "qd_realistic_0km": "QD photonic\n0 km",
                  "qd_realistic_10km": "QD photonic\n10 km"}
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True)
    width = 0.2
    for ax, lab, panel in ((axes[0], "first_entropy", "(a) first signature: the whole threshold block"),
                           (axes[1], "x10_entropy", "(b) per signature, block ten times larger")):
        for j, proto in enumerate(PROTOS):
            xs, vals, plats = [], [], []
            for i, p in enumerate(PLATS):
                e = ledger[p.key][proto.key]
                if lab in e:
                    xs.append(i + (j - 1.5) * width)
                    vals.append(e[lab]["block"]["private_input"] if lab.startswith("first") else e[lab]["private_input"])
                    plats.append(p)
            bars = ax.bar(xs, vals, width * 0.92, color=colors[proto.key], label=names[proto.key], zorder=3)
            if proto.key not in PER_ATTEMPT_OK:                    # per-attempt links also need pre-shared secrets
                for bar_, p in zip(bars, plats):
                    if p.per_attempt_inputs:
                        bar_.set_hatch("///")
                        bar_.set_edgecolor("white")
        ax.set_yscale("log")
        ax.set_xticks(range(len(PLATS)))
        ax.set_xticklabels([plat_names.get(p.key, p.key) for p in PLATS], fontsize=7.5)
        ax.grid(axis="x", visible=False)
        ax.set_axisbelow(True)
        ax.text(0.01, 0.99, panel, transform=ax.transAxes, fontsize=8, color=INK_2, va="top")
        ax.axvspan(1.5, 3.5, color=INK_2, alpha=0.06, zorder=0)
        ax.text(2.5, 0.90, "inputs drawn per attempt\nhatched: plus pre-shared secret bits", ha="center", va="top",
                transform=ax.get_xaxis_transform(), fontsize=7, color=INK_2)
        ax.set_ylim(30, 1e18)                                    # room for the note above the tallest bars
    axes[0].set_ylabel("private input bits")
    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=4, fontsize=7.8)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    save(fig, RES / "fig_ledger", tight=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--section", choices=("platforms", "ledger", "pareto", "all"), default="all")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    if not a.figures:
        if a.section in ("platforms", "all"):
            (RES / "randomness_platforms.json").write_text(json.dumps(platforms_section(), indent=1, default=float))
        if a.section in ("ledger", "all"):
            (RES / "randomness_ledger.json").write_text(json.dumps(ledger_section(), indent=1, default=float))
        if a.section in ("pareto", "all"):
            led = json.loads((RES / "randomness_ledger.json").read_text())
            (RES / "randomness_pareto.json").write_text(json.dumps(pareto_section(led), indent=1, default=float))
    names = ("randomness_platforms.json", "randomness_ledger.json", "randomness_pareto.json")
    if all((RES / f).exists() for f in names):
        figures(*(json.loads((RES / f).read_text()) for f in names))
        print("figures written")


if __name__ == "__main__":
    main()
