"""Finite-size thresholds of DI-QDS under the EAT bound of Fan et al. and the Renyi bound (Sec. III-IV, VIII).

    python scripts/run_finite_size.py --section renyi        # Werner thresholds, V_min, experiments   (~30 min, 1 core)
    python scripts/run_finite_size.py --section sensitivity  # eps, EC efficiency, tails, targets, T2   (~25 min)
    python scripts/run_finite_size.py --section eat          # the EAT baseline, Table I reproduction  (~50 min)
    python scripts/run_finite_size.py --section lu           # Lu et al.'s protocol on the 11 km block   (~5 min)
    python scripts/run_finite_size.py --section jitter       # sawtooth of the key length near thresholds (~5 min)
    python scripts/run_finite_size.py --section polish       # refine stored thresholds (see below)
    python scripts/run_finite_size.py --section figures      # figures from the cached results

Every threshold is the smallest integer n (Brent's method, relative tolerance 1e-5, then a warm-started descent when
the key length at the root exceeds the target by more than the integer-count sawtooth) whose optimised key length
reaches the target; the key length AT that n (certified for the Renyi bound) is stored next to it.  ``--section
polish`` applies the descent to results stored by an earlier run.  Outputs:
results/finite_size_renyi.json, results/finite_size_sensitivity.json, results/finite_size_eat.json and the figures
results/fig_vmin.png|pdf, fig_rounds_vs_visibility.png|pdf, fig_experiments.png|pdf.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diqds.common import KEY_BITS_PER_SIGNATURE, Honest, QDSBudget  # noqa: E402
from diqds.finite_size import eat as EAT  # noqa: E402
from diqds.finite_size.operating_points import (ATOM11_T2_CONSERVATIVE, ATOM_11KM, ATOM_400M,  # noqa: E402
                                                FIG6_ATOM11_ROUNDS, FIG6_ION_ROUNDS, ION)
from diqds.finite_size.descent import POLISH_BITS  # noqa: E402
from diqds.finite_size.optimize import _vec, jitter, polish, renyi_optimum, threshold, v_min  # noqa: E402
from diqds.finite_size.protocols import (INDEPENDENT, LU_GAMMA, LU_GAMMA_A, LU_GAMMA_B, ONE_SETTING,  # noqa: E402
                                         SEEDED, TWO_SETTING)
from diqds.finite_size.renyi import RenyiSettings, completeness_exact  # noqa: E402

RES = ROOT / "results"
PAPER_VMIN = {"p0": {1e8: 0.8835, 1e9: 0.8686, 1e10: 0.8622}, "popt": {1e8: 0.8799, 1e9: 0.8608, 1e10: 0.8508}}
PAPER_P_LEGEND = {1e8: 0.035, 1e9: 0.090, 1e10: 0.155}      # threshold-optimal p of Fan et al.'s Fig. 5 legend
NS = (1e8, 1e9, 1e10)
VS = (1.0, 0.98, 0.95, 0.92, 0.90, 0.88)
EXPS = (ION, ATOM_11KM, ATOM_400M)
GUESS: dict = {}


def _save(name: str, d: dict) -> None:
    (RES / name).write_text(json.dumps(d, indent=1, default=float))


def _load(name: str) -> dict:
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else {}


def _rr(r) -> dict:
    """Compact record of a Renyi result."""
    if r is None:
        return {}
    d = {k: getattr(r, k) for k in ("ell", "n", "gamma", "alpha", "q", "w_star", "h_alpha")}
    d["terms"] = {k: float(v) for k, v in r.terms.items()}
    d["info"] = {k: v for k, v in r.info.items() if k != "x"}
    if r.acceptance is not None:
        d["completeness_union"] = completeness_exact(r.acceptance)
    return d


def _th(honest, proto, q, key, target=KEY_BITS_PER_SIGNATURE, settings=RenyiSettings(), gamma=None, hours=None) -> dict:
    t0 = time.time()
    th = threshold(honest, proto, q, target, settings, gamma=gamma, guess=GUESS.get(key))
    out = {"n_min": th.n, "ell_at_n_min": th.ell, "target": target, "params": _rr(th.result), "wall_s": time.time() - t0}
    if hours is not None and math.isfinite(th.n):
        out["hours"] = hours(th.n)
    print(f"  {key}: n_min = {th.n:.6g}, ell = {th.ell:.2f} [{time.time() - t0:.0f}s]", flush=True)
    return out


# ------------------------------------------------------------------------------------------------- Renyi
def section_renyi() -> dict:
    out = _load("finite_size_renyi.json") if "--resume" in sys.argv else {}
    out.setdefault("vmin", {})
    out.setdefault("nmin_vs_V", {})
    out["settings"] = asdict(RenyiSettings())
    for key, proto in (("two_setting", TWO_SETTING), ("one_setting", ONE_SETTING)):
        for qs, qlab in ((0.0, "q0"), ("opt", "qopt")):
            k = f"{key}_{qlab}"
            if k not in out["nmin_vs_V"]:
                out["nmin_vs_V"][k] = {f"{V:g}": _th(Honest.werner(V), proto, qs, f"werner/{k}/{V:g}") for V in VS}
                _save("finite_size_renyi.json", out)
    for key, proto in (("two_setting", TWO_SETTING), ("one_setting", ONE_SETTING)):
        for qs, qlab in ((0.0, "q0"), ("opt", "qopt")):
            k = f"{key}_{qlab}"
            if k not in out["vmin"]:
                out["vmin"][k] = {}
                for n in NS:
                    t0 = time.time()
                    out["vmin"][k][f"{n:g}"] = v_min(n, proto, qs)
                    print(f"  vmin {k} {n:g}: {out['vmin'][k][f'{n:g}']:.5f} [{time.time() - t0:.0f}s]", flush=True)
                _save("finite_size_renyi.json", out)
    out["experiments"] = experiments_renyi(out.get("experiments", {}))
    _save("finite_size_renyi.json", out)
    out["lu"] = lu_protocol()
    _save("finite_size_renyi.json", out)
    out["crosscheck_hahn"] = crosscheck_hahn()
    out["jitter"] = {k: jitter(e.honest, ONE_SETTING, out["experiments"][e.key]["rows"]["one_setting_qopt"]["n_min"], "opt")
                     for k, e in (("ion", ION), ("atom11km", ATOM_11KM))}
    _save("finite_size_renyi.json", out)
    return out


EXP_ROWS = (("one_setting_q0", ONE_SETTING, 0.0, None), ("one_setting_qopt", ONE_SETTING, "opt", None),
            ("two_setting_q0", TWO_SETTING, 0.0, None), ("two_setting_qopt", TWO_SETTING, "opt", None),
            ("independent_q0", INDEPENDENT, 0.0, None), ("independent_qopt", INDEPENDENT, "opt", None),
            ("seeded_qopt", SEEDED, "opt", None))


def experiments_renyi(prev: dict) -> dict:
    """Rounds to the first signature for the three reported experiments; signatures from the recorded blocks."""
    out = prev or {}
    for e in EXPS:
        row = out.get(e.key, {}).get("rows", {})
        for key, proto, qs, g in EXP_ROWS:
            if key not in row:
                row[key] = _th(e.honest, proto, qs, f"{e.key}/{key}", gamma=g, hours=e.hours_for)
        if "two_setting_iso_qopt" not in row:
            row["two_setting_iso_qopt"] = _th(e.honest.isotropic(), TWO_SETTING, "opt", f"{e.key}/two_setting_iso_qopt",
                                              hours=e.hours_for)
            row["two_setting_iso_qopt"]["Q_x"] = e.honest.isotropic().Q_x
        rec = {}
        for lab, proto, qs, st in (("one_setting_qopt", ONE_SETTING, "opt", RenyiSettings()),
                                   ("one_setting_qopt_cascade", ONE_SETTING, "opt", RenyiSettings.cascade()),
                                   ("independent_qopt", INDEPENDENT, "opt", RenyiSettings()),
                                   ("independent_qopt_cascade", INDEPENDENT, "opt", RenyiSettings.cascade()),
                                   ("two_setting_qopt", TWO_SETTING, "opt", RenyiSettings())):
            r = renyi_optimum(e.rounds, e.honest, proto, qs, st)
            rec[lab] = {"n": e.rounds, "ell": r.ell, "signatures": max(r.ell, 0.0) / KEY_BITS_PER_SIGNATURE,
                        "result": _rr(r)}
        out[e.key] = {"rows": row, "recorded": rec,
                      "honest": {"S": e.honest.S, "Q": e.honest.Q, "Q_x": e.honest.Q_x, "Q_x_state": e.honest.Q_x_state},
                      "rounds": e.rounds, "hours": e.hours}
        print("experiment", e.key, {k: f"{v['n_min']:.4g}" for k, v in row.items()},
              {k: round(v["signatures"], 1) for k, v in rec.items()}, flush=True)
        _save("finite_size_renyi.json", {**_load("finite_size_renyi.json"), "experiments": out})
    return out


def lu_protocol() -> dict:
    """Like-for-like with Lu et al.: their protocol (independent test flags, gamma_A = 0.26, gamma_B = 0.13, no
    preprocessing) on the recorded 11 km block, at their two soundness levels and at the QDS level 5e-11; and the
    one-setting protocol at the same levels.  Conventions: our 3-sigma tails; Hahn et al.'s completeness convention (six
    tails sharing 1e-3); and test-round outputs announced instead of reconciled (no test-round EC term), a reading of
    Lu et al.'s protocol description under which only key rounds are error-corrected."""
    e = ATOM_11KM
    out = {"gamma_A": LU_GAMMA_A, "gamma_B": LU_GAMMA_B, "gamma": LU_GAMMA, "sift": INDEPENDENT.sift_at(LU_GAMMA),
           "published": {"1e-05": 0.112, "1e-15": 0.075, "eat_1e-05": 0.034}, "rows": {}}
    for eps in (1e-5, 5e-11, 1e-15):
        for tails, mk in (("3sigma", lambda eps: RenyiSettings(eps_sound=eps)),
                          ("hahn", lambda eps: RenyiSettings.hahn(eps_sound=eps)),
                          ("announced_tests", lambda eps: RenyiSettings(eps_sound=eps, ec_f_test=0.0))):
            st = mk(eps)
            rows = {
                "lu_as_run": renyi_optimum(e.rounds, e.honest, INDEPENDENT, 0.0, st, gamma=LU_GAMMA),
                "independent_opt": renyi_optimum(e.rounds, e.honest, INDEPENDENT, "opt", st),
                "one_setting_opt": renyi_optimum(e.rounds, e.honest, ONE_SETTING, "opt", st),
            }
            for k, r in rows.items():
                out["rows"][f"{k}/{eps:g}/{tails}"] = {"eps_sound": eps, "tails": tails, "ell": r.ell,
                                                       "bits_per_event": r.ell / e.rounds,
                                                       "signatures": max(r.ell, 0.0) / KEY_BITS_PER_SIGNATURE,
                                                       "result": _rr(r)}
            print("lu", f"{eps:g}", tails, {k: round(r.ell / e.rounds, 4) for k, r in rows.items()}, flush=True)
    # Lu et al.'s protocol at the QDS level: rounds to the first signature
    out["threshold_as_run"] = _th(e.honest, INDEPENDENT, 0.0, "atom11km/lu_as_run", gamma=LU_GAMMA, hours=e.hours_for)
    return out


def crosscheck_hahn() -> dict:
    """Hahn et al.'s App. D setting on the ion block: w = 0.83, Q = 1.8 %, gamma = 13/256, eps_sound = 1e-10, six tails
    sharing 1e-3; Nadlinger et al. report 95,628 bits from 1.5e6 rounds, Hahn et al. about three times as much."""
    hon = Honest(0.83, 0.018, "hahn_fig1")
    r = renyi_optimum(1.5e6, hon, ONE_SETTING, 0.0, RenyiSettings.hahn(), gamma=13 / 256)
    out = {"n": 1.5e6, "ell_renyi": r.ell, "alpha": r.alpha, "ell_EC": -r.terms["error_correction"],
           "nadlinger_reported_ell": 95628, "ratio": r.ell / 95628, "result": _rr(r)}
    print("crosscheck", {k: v for k, v in out.items() if k != "result"}, flush=True)
    return out


# ------------------------------------------------------------------------------------------------- sensitivity
def section_sensitivity() -> dict:
    from diqds.randomness import qutip_states as QS
    out = _load("finite_size_sensitivity.json") if "--resume" in sys.argv else {}
    for e in (ION, ATOM_11KM):
        for eps in (1e-5, 1e-8, 5e-11, 1e-12, 1e-15):
            k = f"{e.key}_{eps:g}"
            if k not in out.setdefault("eps", {}):
                out["eps"][k] = {"eps_sound": eps, **_th(e.honest, ONE_SETTING, "opt", f"eps/{k}", settings=RenyiSettings(eps_sound=eps))}
        for f in (1.0, 1.1, 1.2, 1.3):
            k = f"{e.key}_{f:g}"
            if k not in out.setdefault("ec_f", {}):
                out["ec_f"][k] = {"f": f, **_th(e.honest, ONE_SETTING, "opt", f"ecf/{k}", settings=RenyiSettings(ec_f_key=f, ec_f_test=f))}
        k = f"{e.key}_cascade"
        if k not in out["ec_f"]:
            out["ec_f"][k] = {"f": "cascade (1.20 key, 1.33 test)", **_th(e.honest, ONE_SETTING, "opt", f"ecf/{k}",
                                                                          settings=RenyiSettings.cascade())}
        for lab, stt in (("3sigma", RenyiSettings()), ("hahn_1e-3_6tails", RenyiSettings.hahn(eps_sound=5e-11)),
                         ("1e-2_3tails", RenyiSettings(tail=1e-2 / 3))):
            k = f"{e.key}_{lab}"
            if k not in out.setdefault("tails", {}):
                out["tails"][k] = {"completeness_AT": stt.completeness_AT, **_th(e.honest, ONE_SETTING, "opt", f"tails/{k}", settings=stt)}
        _save("finite_size_sensitivity.json", out)
    # signature length: document length n_m and the composable 82-bit signatures of Bian et al.
    out.setdefault("targets", {})
    for e in (ION, ATOM_11KM):
        for lab, bits in (("nm_1e3", QDSBudget(n_m=1000).key_bits_per_signature),
                          ("nm_1e6", QDSBudget(n_m=10 ** 6).key_bits_per_signature),
                          ("nm_1e9", QDSBudget(n_m=10 ** 9).key_bits_per_signature), ("bian_82", 3 * 82)):
            for pk, proto in (("one_setting", ONE_SETTING), ("two_setting", TWO_SETTING)):
                if pk == "two_setting" and lab not in ("nm_1e3", "bian_82"):
                    continue
                k = f"{e.key}_{pk}_{lab}"
                if k not in out["targets"]:
                    out["targets"][k] = {"key_bits": bits, **_th(e.honest, proto, "opt", f"targets/{k}", target=bits,
                                                                 hours=e.hours_for)}
                    r = renyi_optimum(e.rounds, e.honest, proto, "opt")
                    out["targets"][k]["signatures_recorded"] = max(r.ell, 0.0) / bits
        _save("finite_size_sensitivity.json", out)
    out["n_s_of_n_m"] = {f"{nm:g}": QDSBudget(n_m=int(nm)).n_s for nm in (1e3, 8e3, 1e6, 8e6, 1e9)}
    # memory: the extra L/c wait of the one-setting protocol at 11 km, for several T2 (QuTiP Lindblad dephasing)
    m = QS.calibrate_matter(ATOM_11KM.honest.S, ATOM_11KM.honest.Q, 5e-3)
    out.setdefault("memory", {})
    for T2 in (1e-3, 1e-2, ATOM11_T2_CONSERVATIVE, 1e-1):
        k = f"{T2:g}"
        if k in out["memory"]:
            continue
        S, w, Q = QS.statistics(m.probabilities(extra_wait=11e3 / 2e8, T2_memory=T2))
        hon = Honest.from_chsh(S, Q, f"11 km after L/c, T2 = {T2:g} s")
        row = {"T2_s": T2, "S": S, "Q": Q, **_th(hon, ONE_SETTING, "opt", f"memory/{k}", hours=ATOM_11KM.hours_for)}
        r = renyi_optimum(ATOM_11KM.rounds, hon, ONE_SETTING, "opt")
        row["signatures_recorded"] = max(r.ell, 0.0) / KEY_BITS_PER_SIGNATURE
        rc = renyi_optimum(ATOM_11KM.rounds, hon, ONE_SETTING, "opt", RenyiSettings.cascade())
        row["signatures_recorded_cascade"] = max(rc.ell, 0.0) / KEY_BITS_PER_SIGNATURE
        out["memory"][k] = row
        _save("finite_size_sensitivity.json", out)
    # second key basis: Bob's KEY observables misaligned by delta (test observables ideal): S unchanged, Q measured,
    # Q_x = 1 - Q - cos(delta) S / (2 sqrt2) instead of Eq. (3)
    out.setdefault("misalignment", {})
    for e in (ION, ATOM_11KM):
        for deg in (0.0, 5.0, 10.0):
            k = f"{e.key}_{deg:g}"
            if k in out["misalignment"]:
                continue
            Qx = 1.0 - e.honest.Q - math.cos(math.radians(deg)) * e.honest.S / (2.0 * math.sqrt(2.0))
            hon = Honest(e.honest.w, e.honest.Q, f"{e.key} key misaligned {deg:g} deg", Qx=Qx)
            out["misalignment"][k] = {"delta_deg": deg, "Q_x": Qx,
                                      **_th(hon, TWO_SETTING, "opt", f"misalign/{k}", hours=e.hours_for)}
            _save("finite_size_sensitivity.json", out)
    _save("finite_size_sensitivity.json", out)
    return out


# ------------------------------------------------------------------------------------------------- EAT baseline
def _eth(honest, sift, p, key, st, target=KEY_BITS_PER_SIGNATURE, hours=None) -> dict:
    t0 = time.time()
    th = EAT.threshold(honest, sift, p, target, st, guess=GUESS.get(key))
    r = th.result
    out = {"n_min": th.n, "ell_at_n_min": th.ell, "p": getattr(r, "p", None), "gamma": getattr(r, "gamma", None),
           "wall_s": time.time() - t0}
    if hours is not None and math.isfinite(th.n):
        out["hours"] = hours(th.n)
    print(f"  {key}: n_min = {th.n:.6g}, ell = {th.ell:.2f}, p = {out['p']} [{time.time() - t0:.0f}s]", flush=True)
    return out


def section_eat() -> dict:
    st = EAT.EATSettings(reading="rigorous")
    out = _load("finite_size_eat.json") if "--resume" in sys.argv else {}
    out["settings"] = asdict(st)
    out.setdefault("vmin", {})
    for lab, pspec in (("two_setting_p0", lambda n: "zero"), ("two_setting_plegend", lambda n: PAPER_P_LEGEND[n])):
        if lab not in out["vmin"]:
            out["vmin"][lab] = {}
            for n in NS:
                t0 = time.time()
                out["vmin"][lab][f"{n:g}"] = EAT.v_min(n, 0.5, pspec(n), settings=st)
                print(f"  eat vmin {lab} {n:g}: {out['vmin'][lab][f'{n:g}']:.5f} [{time.time() - t0:.0f}s]", flush=True)
            _save("finite_size_eat.json", out)
    out.setdefault("nmin_vs_V", {})
    for key, sift in (("two_setting", 0.5), ("one_setting", 1.0)):
        for plab, p in (("p0", "zero"), ("popt", "opt")):
            k = f"{key}_{plab}"
            if k not in out["nmin_vs_V"]:
                out["nmin_vs_V"][k] = {f"{V:g}": _eth(Honest.werner(V), sift, p, f"eat/werner/{k}/{V:g}", st) for V in VS}
                _save("finite_size_eat.json", out)
    out.setdefault("experiments", {})
    for e in EXPS:
        row = out["experiments"].setdefault(e.key, {})
        for plab, p in (("one_setting_p0", "zero"), ("one_setting_popt", "opt")):
            if plab not in row:
                row[plab] = _eth(e.honest, 1.0, p, f"eat/{e.key}/{plab}", st, hours=e.hours_for)
                _save("finite_size_eat.json", out)
    # Fan et al.'s own two-setting protocol at 11 km (their Fig. 6 used the one-setting key generation of
    # Nadlinger et al.)
    out.setdefault("two_setting_11km", {})
    for lab, h in (("Q_x_state", ATOM_11KM.honest), ("Q_x_eq_Q", ATOM_11KM.honest.isotropic())):
        for plab, p in (("p0", "zero"), ("popt", "opt")):
            k = f"{lab}_{plab}"
            if k not in out["two_setting_11km"]:
                out["two_setting_11km"][k] = _eth(h, 0.5, p, f"eat/two11/{k}", st, hours=ATOM_11KM.hours_for)
                _save("finite_size_eat.json", out)
    # signatures from the recorded blocks with the EAT bound (one setting, p optimised)
    out["recorded"] = {}
    for e in (ION, ATOM_11KM):
        r = EAT.optimum_over_p(e.rounds, e.honest, 1.0, st)
        out["recorded"][e.key] = {"n": e.rounds, "ell": r.ell, "p": r.p, "signatures": max(r.ell, 0.0) / KEY_BITS_PER_SIGNATURE}
        print("eat recorded", e.key, out["recorded"][e.key], flush=True)
    _save("finite_size_eat.json", out)
    return out


# ------------------------------------------------------------------------------------------------- polish
def section_polish(which=("renyi", "eat")) -> None:
    """Refine stored thresholds whose key length exceeds the target by more than POLISH_BITS.

    ``threshold`` applies the same warm-started descent itself (diqds.finite_size.descent); this section applies it to
    results computed before, without repeating the Brent search.  Each refined entry keeps the value before its first
    refinement in ``polished_from``."""
    def refine(entry, honest, proto, q, settings=RenyiSettings(), gamma=None, hours=None, label=""):
        tgt = entry.get("target", KEY_BITS_PER_SIGNATURE)
        if not (math.isfinite(entry.get("n_min", math.inf)) and entry["ell_at_n_min"] - tgt > POLISH_BITS):
            return
        P = entry["params"]
        x0 = list(_vec(P["gamma"], P["alpha"], P["q"], None if q == "opt" else float(q), gamma))
        t0 = time.time()
        th = polish(honest, proto, entry["n_min"], q, tgt, settings, gamma=gamma, x0=x0)
        if th is None:
            print(f"  polish {label}: kept n = {entry['n_min']:.6g} (ell = {entry['ell_at_n_min']:.1f})", flush=True)
            return
        print(f"  polish {label}: {entry['n_min']:.6g} -> {th.n:.6g}, ell {entry['ell_at_n_min']:.1f} -> {th.ell:.1f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        entry.setdefault("polished_from", entry["n_min"])
        entry.update({"n_min": th.n, "ell_at_n_min": th.ell, "params": _rr(th.result)})
        if hours is not None:
            entry["hours"] = hours(th.n)

    def refine_eat(entry, honest, sift, p, st, hours=None, label=""):
        tgt = KEY_BITS_PER_SIGNATURE
        if not (math.isfinite(entry.get("n_min", math.inf)) and entry["ell_at_n_min"] - tgt > POLISH_BITS):
            return
        t0 = time.time()
        th = EAT.polish(honest, sift, p, entry["n_min"], tgt, st)
        if th is None:
            print(f"  polish {label}: kept n = {entry['n_min']:.6g} (ell = {entry['ell_at_n_min']:.1f})", flush=True)
            return
        print(f"  polish {label}: {entry['n_min']:.6g} -> {th.n:.6g}, ell {entry['ell_at_n_min']:.1f} -> {th.ell:.1f} "
              f"[{time.time() - t0:.0f}s]", flush=True)
        entry.setdefault("polished_from", entry["n_min"])
        entry.update({"n_min": th.n, "ell_at_n_min": th.ell, "p": th.result.p, "gamma": th.result.gamma})
        if hours is not None:
            entry["hours"] = hours(th.n)

    PK = {"two_setting": TWO_SETTING, "one_setting": ONE_SETTING}
    EK = {e.key: e for e in EXPS}
    if "renyi" in which:
        ren = _load("finite_size_renyi.json")
        for k, d in ren.get("nmin_vs_V", {}).items():
            pk, ql = k.rsplit("_", 1)
            for V, entry in d.items():
                refine(entry, Honest.werner(float(V)), PK[pk], 0.0 if ql == "q0" else "opt", label=f"werner/{k}/{V}")
        _save("finite_size_renyi.json", ren)
        for e in EXPS:
            rows = ren["experiments"][e.key]["rows"]
            for key, proto, qs, g in EXP_ROWS:
                refine(rows[key], e.honest, proto, qs, gamma=g, hours=e.hours_for, label=f"{e.key}/{key}")
            refine(rows["two_setting_iso_qopt"], e.honest.isotropic(), TWO_SETTING, "opt", hours=e.hours_for,
                   label=f"{e.key}/two_setting_iso_qopt")
            _save("finite_size_renyi.json", ren)
        if "threshold_as_run" in ren.get("lu", {}):
            refine(ren["lu"]["threshold_as_run"], ATOM_11KM.honest, INDEPENDENT, 0.0, gamma=LU_GAMMA,
                   hours=ATOM_11KM.hours_for, label="atom11km/lu_as_run")
        _save("finite_size_renyi.json", ren)
        sen = _load("finite_size_sensitivity.json")
        tails = {"3sigma": RenyiSettings(), "hahn_1e-3_6tails": RenyiSettings.hahn(eps_sound=5e-11),
                 "1e-2_3tails": RenyiSettings(tail=1e-2 / 3)}
        for k, entry in sen.get("eps", {}).items():
            refine(entry, EK[k.split("_", 1)[0]].honest, ONE_SETTING, "opt", RenyiSettings(eps_sound=entry["eps_sound"]),
                   label=f"eps/{k}")
        for k, entry in sen.get("ec_f", {}).items():
            ek, f = k.split("_", 1)
            st = RenyiSettings.cascade() if f == "cascade" else RenyiSettings(ec_f_key=float(f), ec_f_test=float(f))
            refine(entry, EK[ek].honest, ONE_SETTING, "opt", st, label=f"ecf/{k}")
        for k, entry in sen.get("tails", {}).items():
            ek, lab = k.split("_", 1)
            refine(entry, EK[ek].honest, ONE_SETTING, "opt", tails[lab], label=f"tails/{k}")
        for k, entry in sen.get("targets", {}).items():
            e = EK[k.split("_", 1)[0]]
            refine(entry, e.honest, TWO_SETTING if "_two_setting_" in k else ONE_SETTING, "opt", hours=e.hours_for,
                   label=f"targets/{k}")
        for k, entry in sen.get("memory", {}).items():
            hon = Honest.from_chsh(entry["S"], entry["Q"], f"11 km after L/c, T2 = {k} s")
            refine(entry, hon, ONE_SETTING, "opt", hours=ATOM_11KM.hours_for, label=f"memory/{k}")
        for k, entry in sen.get("misalignment", {}).items():
            e = EK[k.split("_", 1)[0]]
            hon = Honest(e.honest.w, e.honest.Q, f"{e.key} key misaligned {entry['delta_deg']:g} deg", Qx=entry["Q_x"])
            refine(entry, hon, TWO_SETTING, "opt", hours=e.hours_for, label=f"misalign/{k}")
        _save("finite_size_sensitivity.json", sen)
    if "eat" in which:
        eat = _load("finite_size_eat.json")
        st = EAT.EATSettings(reading="rigorous")
        for k, d in eat.get("nmin_vs_V", {}).items():
            pk, pl = k.rsplit("_", 1)
            for V, entry in d.items():
                refine_eat(entry, Honest.werner(float(V)), 0.5 if pk == "two_setting" else 1.0,
                           "zero" if pl == "p0" else "opt", st, label=f"eat/werner/{k}/{V}")
                _save("finite_size_eat.json", eat)
        for e in EXPS:
            row = eat.get("experiments", {}).get(e.key, {})
            for plab, p in (("one_setting_p0", "zero"), ("one_setting_popt", "opt")):
                if plab in row:
                    refine_eat(row[plab], e.honest, 1.0, p, st, hours=e.hours_for, label=f"eat/{e.key}/{plab}")
            _save("finite_size_eat.json", eat)
        for lab, h in (("Q_x_state", ATOM_11KM.honest), ("Q_x_eq_Q", ATOM_11KM.honest.isotropic())):
            for plab, p in (("p0", "zero"), ("popt", "opt")):
                k = f"{lab}_{plab}"
                if k in eat.get("two_setting_11km", {}):
                    refine_eat(eat["two_setting_11km"][k], h, 0.5, p, st, hours=ATOM_11KM.hours_for, label=f"eat/two11/{k}")
        _save("finite_size_eat.json", eat)


# ------------------------------------------------------------------------------------------------- figures
def figures() -> dict:
    from diqds.plotting import INK_2, SERIES, save, setup
    import matplotlib.pyplot as plt
    import numpy as np
    setup()
    ren, eat = _load("finite_size_renyi.json"), _load("finite_size_eat.json")
    summary = {}
    # (a) V_min at three block sizes
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    rows = [("EAT bound of Fan et al., two settings", eat["vmin"]["two_setting_plegend"], SERIES[0], "o", "--"),
            ("Rényi bound, two settings", ren["vmin"]["two_setting_qopt"], SERIES[0], "s", "-"),
            ("Rényi bound, one setting", ren["vmin"]["one_setting_qopt"], SERIES[1], "D", "-")]
    xs = list(NS)
    for lab, d, c, m, ls in rows:
        ys = [d[f"{n:g}"] for n in xs]
        ax.plot(xs, ys, ls, marker=m, color=c, label=lab)
    ax.plot(xs, [PAPER_VMIN["popt"][n] for n in xs], linestyle="none", marker="x", color=INK_2,
            label="printed by Fan et al.")
    ax.set_xscale("log")
    ax.set_xlabel(r"rounds per link, $n$")
    ax.set_ylabel("minimum visibility for one signature")
    ax.legend(loc="upper right")
    save(fig, RES / "fig_vmin")
    # (b) rounds to the first signature against visibility
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    curves = [("EAT, two settings", {V: v["n_min"] for V, v in eat["nmin_vs_V"]["two_setting_popt"].items()}, SERIES[0], "o", "--"),
              ("Rényi, two settings", {V: v["n_min"] for V, v in ren["nmin_vs_V"]["two_setting_qopt"].items()}, SERIES[0], "s", "-"),
              ("EAT, one setting", {V: v["n_min"] for V, v in eat["nmin_vs_V"]["one_setting_popt"].items()}, SERIES[1], "o", "--"),
              ("Rényi, one setting", {V: v["n_min"] for V, v in ren["nmin_vs_V"]["one_setting_qopt"].items()}, SERIES[1], "D", "-")]
    for lab, d, c, m, ls in curves:
        vs = [V for V in VS if math.isfinite(d[f"{V:g}"])]
        ax.plot(vs, [d[f"{V:g}"] for V in vs], ls, marker=m, color=c, label=lab)
    ax.set_yscale("log")
    ax.invert_xaxis()
    ax.set_xlabel(r"visibility $V$ (Werner state)")
    ax.set_ylabel("rounds per link to the first signature")
    ax.legend(loc="upper left", fontsize=8)
    save(fig, RES / "fig_rounds_vs_visibility")
    # (c) experiments
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    names = ["ions, 2 m", "atoms, 11 km", "atoms, 400 m"]
    keys = [e.key for e in EXPS]
    x = np.arange(len(keys))
    E = ren["experiments"]
    bars = [("EAT, one setting", [eat["experiments"][k]["one_setting_popt"]["n_min"] for k in keys], SERIES[0]),
            ("Rényi, one setting (common flag)", [E[k]["rows"]["one_setting_qopt"]["n_min"] for k in keys], SERIES[1]),
            ("Rényi, independent flags", [E[k]["rows"]["independent_qopt"]["n_min"] for k in keys], SERIES[3]),
            ("Rényi, two settings", [E[k]["rows"]["two_setting_qopt"]["n_min"] for k in keys], SERIES[2])]
    wdt = 0.2
    for i, (lab, ys, c) in enumerate(bars):
        ax.bar(x + (i - 1.5) * wdt, ys, wdt * 0.92, color=c, label=lab, zorder=3)
    for i, (xi, e) in enumerate(zip(x, EXPS)):
        ax.hlines(e.rounds, xi - 0.42, xi + 0.42, colors="#0b0b0b", linewidth=2, zorder=5,
                  label="rounds the experiment recorded" if i == 0 else None)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("rounds per link to the first signature")
    ax.set_ylim(1e3, 1e8)
    handles, labels_ = ax.get_legend_handles_labels()
    fig.legend(handles, labels_, loc="lower center", ncol=2, fontsize=7.8)
    fig.tight_layout(rect=(0, 0.17, 1, 1))
    save(fig, RES / "fig_experiments", tight=False)
    summary["paper"] = {"vmin_popt": {f"{k:g}": v for k, v in PAPER_VMIN["popt"].items()},
                        "vmin_p0": {f"{k:g}": v for k, v in PAPER_VMIN["p0"].items()},
                        "fig6_ion_rounds": FIG6_ION_ROUNDS, "fig6_atom11_rounds": FIG6_ATOM11_ROUNDS}
    return summary


def _guesses() -> dict:
    """Starting points for the threshold searches from cached results (speed only; the result does not depend on it)."""
    g = {}
    for name in ("_guesses.json",):
        g.update(_load(name))
    ren, sen, eat = _load("finite_size_renyi.json"), _load("finite_size_sensitivity.json"), _load("finite_size_eat.json")
    for k, d in ren.get("nmin_vs_V", {}).items():
        for V, v in d.items():
            if isinstance(v, dict):
                g[f"werner/{k}/{V}"] = v["n_min"]
    for ek, d in ren.get("experiments", {}).items():
        for k, v in d.get("rows", {}).items():
            g[f"{ek}/{k}"] = v["n_min"]
    for sec in ("eps", "ec_f", "tails", "targets", "memory", "misalignment"):
        for k, v in sen.get(sec, {}).items():
            if isinstance(v, dict) and "n_min" in v:
                g[f"{ {'ec_f': 'ecf', 'misalignment': 'misalign'}.get(sec, sec)}/{k}"] = v["n_min"]
    for k, d in eat.get("nmin_vs_V", {}).items():
        for V, v in d.items():
            if isinstance(v, dict):
                g[f"eat/werner/{k}/{V}"] = v["n_min"]
    return g


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", choices=("renyi", "lu", "jitter", "sensitivity", "eat", "polish", "figures"),
                    default="figures")
    ap.add_argument("--only", choices=("renyi", "eat"), help="polish: refine only the Renyi (incl. sensitivity) or EAT results")
    ap.add_argument("--resume", action="store_true", help="keep finished items of an interrupted run")
    args = ap.parse_args()
    RES.mkdir(exist_ok=True)
    GUESS.update(_guesses())
    if args.section == "renyi":
        section_renyi()
    elif args.section == "jitter":
        out = _load("finite_size_renyi.json")
        E = out["experiments"]
        jit = {}
        for k, e in (("ion", ION), ("atom11km", ATOM_11KM)):
            jit[k] = jitter(e.honest, ONE_SETTING, E[e.key]["rows"]["one_setting_qopt"]["n_min"], "opt")
            jit[f"{k}_two_setting"] = jitter(e.honest, TWO_SETTING, E[e.key]["rows"]["two_setting_qopt"]["n_min"], "opt")
            print("jitter", k, {kk: v["relative_threshold_jitter"] for kk, v in jit.items()}, flush=True)
        out["jitter"] = jit
        _save("finite_size_renyi.json", out)
    elif args.section == "lu":
        out = _load("finite_size_renyi.json")
        out["lu"] = lu_protocol()
        _save("finite_size_renyi.json", out)
    elif args.section == "sensitivity":
        section_sensitivity()
    elif args.section == "polish":
        section_polish((args.only,) if args.only else ("renyi", "eat"))
    elif args.section == "eat":
        section_eat()
    else:
        _save("finite_size_summary.json", figures())


if __name__ == "__main__":
    main()
