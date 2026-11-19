"""Every table of the article and every number quoted in its text, from the cached results.

    python scripts/make_tables.py

Writes results/tables.md (the tables, in the article's numbering) and results/article_numbers.json (the formatted
numbers and table rows the article uses).  Missing result files are skipped, so the script also runs on a partial cache.
"""
from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RES = ROOT / "results"

from diqds.common import KEY_BITS_PER_SIGNATURE, QDSBudget, h2  # noqa: E402
from diqds.finite_size.operating_points import ATOM_11KM, ATOM_400M, ION  # noqa: E402
from diqds.finite_size.protocols import INDEPENDENT, LU_GAMMA  # noqa: E402


def load(name):
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else {}


def get(d, path, default=None):
    for k in path.split("/"):
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def count_tests():
    """Number of tests pytest collects (quoted in Appendix E and the README)."""
    import re
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests"], capture_output=True,
                       text=True, cwd=ROOT)
    m = re.search(r"(\d+) tests? collected", r.stdout)
    return int(m.group(1)) if m else None


# ------------------------------------------------------------------------------------------------ number formats
SUP = str.maketrans("0123456789-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁻")


def sci(x, d=3, unit=False):
    """4.98×10^{5} (article markup); integers below 1e5 with thousands separators."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    x = float(x)
    if x == 0:
        return "0"
    if abs(x) < 1e5 and abs(x) >= 1 and not unit:
        return f"{x:,.0f}"
    e = int(math.floor(math.log10(abs(x))))
    m = x / 10 ** e
    ms = f"{m:.{d - 1}f}"                      # keep trailing zeros: d significant figures
    if float(ms) >= 10.0:
        e += 1
        ms = f"{x / 10 ** e:.{d - 1}f}"
    es = str(e).replace("-", "−")
    return f"{ms}×10^{{{es}}}"


def num(x, d=0):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{x:,.{d}f}"


def sg(x):
    """Signature counts: only complete signatures count, so round down."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    return f"{max(0, math.floor(float(x) + 1e-9)):,}"


def pct(x, d=1):
    return f"{100 * x:.{d}f} %"


def eps_s(x):
    e = int(math.floor(math.log10(x) + 1e-9))
    m = x / 10 ** e
    es = str(e).replace("-", "−")
    return f"10^{{{es}}}" if abs(m - 1) < 1e-9 else f"{m:g}×10^{{{es}}}"


def hours(x):
    if x is None or not math.isfinite(x):
        return "–"
    return sci(x, 2, True) if x >= 1e5 else (f"{x:,.1f}" if x < 10 else f"{x:,.0f}")


def md(s):
    """Article markup -> Markdown-ish plain text (citation tokens {ref:key} become [key])."""
    s = re.sub(r"\{ref:([^}]*)\}", lambda m: "[" + m.group(1).replace(",", ", ") + "]", str(s))
    s = re.sub(r"\^\{([^}]*)\}", lambda m: "^" + m.group(1), s)
    s = re.sub(r"_\{([^}]*)\}", lambda m: "_" + m.group(1), s)
    return s.replace("*", "")


N = {}          # numbers quoted in the text
T = {}          # tables: name -> {"caption": str, "rows": [[...]]}


def table(name, caption, rows):
    T[name] = {"caption": caption, "rows": rows}


# ------------------------------------------------------------------------------------------------ finite size
def finite_size():
    ren, eat, sen = load("finite_size_renyi.json"), load("finite_size_eat.json"), load("finite_size_sensitivity.json")
    if not ren or not eat:
        return
    NS = ("1e+08", "1e+09", "1e+10")
    printed = {"p0": (0.8835, 0.8686, 0.8622), "popt": (0.8799, 0.8608, 0.8508)}
    rows = [["Bound", "n = 10^{8}", "n = 10^{9}", "n = 10^{10}"],
            ["Ref. {ref:fan}, printed (p = 0)", *[f"{v:.4f}" for v in printed["p0"]]],
            ["Ref. {ref:fan}, printed (p optimised)", *[f"{v:.4f}" for v in printed["popt"]]]]
    for lab, key, src in (("EAT bound, this implementation (p = 0)", "two_setting_p0", eat),
                          ("EAT bound, this implementation (p of Ref. {ref:fan})", "two_setting_plegend", eat),
                          ("Rényi, two settings (q = 0)", "two_setting_q0", ren),
                          ("Rényi, two settings (q optimised)", "two_setting_qopt", ren),
                          ("Rényi, one setting, common flag (q = 0)", "one_setting_q0", ren),
                          ("Rényi, one setting, common flag (q optimised)", "one_setting_qopt", ren)):
        d = get(src, f"vmin/{key}", {})
        rows.append([lab, *[f"{d[n]:.4f}" if n in d else "–" for n in NS]])
    table("T_vmin", "Minimum Werner visibility for one signature (138 key bits per link) at ε = 10^{−10}.", rows)
    ev = get(eat, "vmin/two_setting_p0", {})
    ep = get(eat, "vmin/two_setting_plegend", {})
    if ev and ep:
        N["vmin_repro_maxdiff"] = f"{max(max(abs(ev[n] - printed['p0'][i]) for i, n in enumerate(NS)), max(abs(ep[n] - printed['popt'][i]) for i, n in enumerate(NS))):.4f}"
    # rounds to the first signature, Werner
    VS = ("1", "0.98", "0.95", "0.92", "0.9", "0.88")
    hdr = ["V", "EAT 2 sett. p = 0", "EAT 2 sett. p opt.", "Rényi 2 sett. q = 0", "Rényi 2 sett. q opt.",
           "EAT 1 sett. p = 0", "EAT 1 sett. p opt.", "Rényi 1 sett. q = 0", "Rényi 1 sett. q opt."]
    rows = [hdr]
    cols = [("eat", "two_setting_p0"), ("eat", "two_setting_popt"), ("ren", "two_setting_q0"), ("ren", "two_setting_qopt"),
            ("eat", "one_setting_p0"), ("eat", "one_setting_popt"), ("ren", "one_setting_q0"), ("ren", "one_setting_qopt")]
    ratios_q0, ratios_opt, set_ratio, ells = [], [], [], []
    for V in VS:
        r = [V]
        vals = {}
        for src, key in cols:
            d = get(eat if src == "eat" else ren, f"nmin_vs_V/{key}/{V}", {})
            vals[(src, key)] = d.get("n_min")
            if d.get("ell_at_n_min") is not None and math.isfinite(d.get("n_min", math.inf)):
                ells.append(d["ell_at_n_min"])
            r.append(sci(d.get("n_min")))
        rows.append(r)
        for sk in ("two_setting", "one_setting"):
            a, b = vals.get(("eat", f"{sk}_p0")), vals.get(("ren", f"{sk}_q0"))
            if a and b:
                ratios_q0.append(a / b)
            a, b = vals.get(("eat", f"{sk}_popt")), vals.get(("ren", f"{sk}_qopt"))
            if a and b:
                ratios_opt.append(a / b)
        for src, sfx in (("eat", "popt"), ("ren", "qopt")):
            a, b = vals.get((src, f"two_setting_{sfx}")), vals.get((src, f"one_setting_{sfx}"))
            if a and b:
                set_ratio.append(a / b)
    table("T_werner", "Heralded rounds per link to the first signature, Werner devices. EAT: bound of Ref. {ref:fan} without "
          "(p = 0) and with optimised preprocessing; Rényi: Eqs. (5)–(7) without (q = 0) and with optimised preprocessing. "
          "Two-setting entries rest on the per-basis bound of Lemma 1.", rows)
    if ratios_q0:
        N["gain_q0"] = f"{min(ratios_q0):.1f}–{max(ratios_q0):.1f}"
    if ratios_opt:
        N["gain_opt"] = f"{min(ratios_opt):.1f}–{max(ratios_opt):.1f}"
    if set_ratio:
        N["set_ratio"] = f"{min(set_ratio):.1f}–{max(set_ratio):.1f}"

    # experiments (projections)
    E = ren.get("experiments", {})
    names = {"ion": "Ions, 2 m {ref:nadlinger}", "atom11km": "Atoms, 11 km {ref:lu}", "atom400m": "Atoms, 400 m {ref:zhang22}"}
    rows = [["Data set (reported S / Q / Q_{x})", "EAT, one setting", "Rényi, one setting, common flag",
             "Rényi, independent flags", "Rényi, two settings (Eq. 4 / Q_{x} = Q)",
             "Signatures from the recorded block (f = 1 / Cascade budget)"]]
    for k in ("ion", "atom11km", "atom400m"):
        d = E.get(k)
        if not d:
            continue
        h = d["honest"]
        ee = get(eat, f"experiments/{k}/one_setting_popt", {})
        rr = d["rows"]
        rec = d["recorded"]
        rows.append([f"{names[k]}: {h['S']:.3f} / {pct(h['Q'], 2)} / {pct(h['Q_x'], 2)}", sci(ee.get("n_min")),
                     sci(rr["one_setting_qopt"]["n_min"]), sci(rr["independent_qopt"]["n_min"]),
                     f"{sci(rr['two_setting_qopt']['n_min'])} / {sci(rr['two_setting_iso_qopt']['n_min'])}",
                     f"{sg(rec['one_setting_qopt']['signatures'])} / {sg(rec['one_setting_qopt_cascade']['signatures'])}"])
        hours = lambda key: get(rr, f"{key}/hours")
        N[f"{k}_one"] = sci(rr["one_setting_qopt"]["n_min"])
        N[f"{k}_one_q0"] = sci(rr["one_setting_q0"]["n_min"])
        N[f"{k}_one_hours"] = globals()["hours"](hours("one_setting_qopt"))
        N[f"{k}_ind"] = sci(rr["independent_qopt"]["n_min"])
        N[f"{k}_ind_q0"] = sci(rr["independent_q0"]["n_min"])
        N[f"{k}_two"] = sci(rr["two_setting_qopt"]["n_min"])
        N[f"{k}_two_iso"] = sci(rr["two_setting_iso_qopt"]["n_min"])
        N[f"{k}_seeded"] = sci(rr["seeded_qopt"]["n_min"])
        N[f"{k}_eat"] = sci(ee.get("n_min"))
        N[f"{k}_eat_p0"] = sci(get(eat, f"experiments/{k}/one_setting_p0/n_min"))
        N[f"{k}_eat_hours"] = globals()["hours"](ee.get("hours"))
        N[f"{k}_eat_p0_hours"] = globals()["hours"](get(eat, f"experiments/{k}/one_setting_p0/hours"))
        N[f"{k}_sigs"] = sg(rec["one_setting_qopt"]["signatures"])
        N[f"{k}_sigs_cascade"] = sg(rec["one_setting_qopt_cascade"]["signatures"])
        N[f"{k}_sigs_ind"] = sg(rec["independent_qopt"]["signatures"])
        N[f"{k}_sigs_ind_cascade"] = sg(rec["independent_qopt_cascade"]["signatures"])
        N[f"{k}_sigs_two"] = sg(rec["two_setting_qopt"]["signatures"])
        N[f"{k}_one_hours_raw"] = hours("one_setting_qopt")
        N[f"{k}_Qx"] = pct(h["Q_x"], 2)
        N[f"{k}_ell_one"] = f"{rr['one_setting_qopt']['ell_at_n_min']:.1f}"
        ratio = rr["one_setting_qopt"]["n_min"] / ee["n_min"] if ee.get("n_min") else None
        N[f"{k}_one_over_eat"] = f"{ratio:.2f}" if ratio else "–"
    table("T_experiments", "Projections for runs with the reported (S, Q) and number of heralded rounds of three DIQKD "
          "experiments {ref:nadlinger,zhang22,lu}, under the protocol named in each column (none is the experiment's own analysis): rounds "
          "to the first signature (q optimised) and signatures that the recorded block would support with the one-setting "
          "protocol. EAT: bound of Ref. {ref:fan}, one setting. Error correction at the Shannon limit except in the last "
          "column's second entry (f = 1.20 on key rounds, 1.33 on test rounds, the budget our Cascade meets). No memory "
          "penalty for the common flag (Table IV gives it). Q_{x} from Eq. (4) except for the 400 m link (Sec. II C).", rows)
    # EAT recorded / ratios
    for k in ("ion", "atom11km"):
        N[f"{k}_eat_sigs"] = sg(get(eat, f"recorded/{k}/signatures"))
    if E.get("ion"):
        e1, r1 = get(eat, "recorded/ion/signatures"), get(E, "ion/recorded/one_setting_qopt/signatures")
        if e1 and r1:
            N["ion_sig_ratio"] = f"{r1 / e1:.1f}"
    # Fan et al.'s two-setting protocol at 11 km with the EAT
    t11 = eat.get("two_setting_11km", {})
    for k, v in t11.items():
        N[f"eat_two11_{k}"] = sci(v["n_min"])
    # Lu et al. like-for-like
    lu = ren.get("lu", {})
    if lu:
        rows = [["ε_{snd} per link", "Published (Lu et al., their protocol)",
                 "Lu et al.'s protocol, this implementation (γ_{A} = 0.26, γ_{B} = 0.13, q = 0)",
                 "Independent flags, γ and q optimised", "One setting, common flag, q optimised"]]
        pub = {"1e-05": "0.112", "1e-15": "0.075", "5e-11": "–"}
        for eps in ("1e-05", "5e-11", "1e-15"):
            g = lambda k, t="3sigma": lu["rows"].get(f"{k}/{eps}/{t}", {}).get("bits_per_event")
            sig = lambda k, t="3sigma": lu["rows"].get(f"{k}/{eps}/{t}", {}).get("signatures")
            rows.append([eps_s(float(eps)), pub[eps],
                         f"{g('lu_as_run'):.3f} ({sg(sig('lu_as_run'))} sig.)",
                         f"{g('independent_opt'):.3f} ({sg(sig('independent_opt'))} sig.)",
                         f"{g('one_setting_opt'):.3f} ({sg(sig('one_setting_opt'))} sig.)"])
            for key in ("lu_as_run", "independent_opt", "one_setting_opt"):
                N[f"lu_{key}_{eps}"] = f"{g(key):.3f}"
                N[f"lu_{key}_{eps}_sigs"] = sg(sig(key))
                N[f"lu_{key}_{eps}_hahn"] = f"{g(key, 'hahn'):.3f}"
                if g(key, "announced_tests") is not None:
                    N[f"lu_{key}_{eps}_ann"] = f"{g(key, 'announced_tests'):.3f}"
        table("T_lu", "Bits per heralded event and signatures (138 key bits each) from the recorded 11 km block "
              "(1.208×10^{6} events), our implementation of three protocols at three soundness levels, against Lu et al.'s "
              "published Rényi analysis of their own protocol.", rows)
        N["lu_threshold_as_run"] = sci(get(lu, "threshold_as_run/n_min"))
        N["lu_gain_one_vs_ind_5e11"] = f"{100 * (lu['rows']['one_setting_opt/5e-11/3sigma']['ell'] / lu['rows']['independent_opt/5e-11/3sigma']['ell'] - 1):.0f}"
        N["lu_published_sigs"] = f"{sg(0.075 * 1.208e6 / 138)}–{sg(0.112 * 1.208e6 / 138)}"
    hc = ren.get("crosscheck_hahn", {})
    if hc:
        N["hahn_ell"] = num(hc["ell_renyi"])
        N["hahn_ratio"] = f"{hc['ratio']:.2f}"
        N["nad_sigs"] = sg(95628 / KEY_BITS_PER_SIGNATURE)
        N["hahn_sigs"] = sg(hc["ell_renyi"] / KEY_BITS_PER_SIGNATURE)
    jit = ren.get("jitter", {})
    one = [v for k, v in jit.items() if "two_setting" not in k]
    two = [v for k, v in jit.items() if "two_setting" in k]
    fmt = lambda x: f"{x:.0e}".replace("e-0", "×10^{−").replace("e-", "×10^{−") + "}"
    if one:
        N["jitter_rel"] = fmt(max(v["relative_threshold_jitter"] for v in one))
        N["jitter_bits"] = f"{max(v['max_abs_residual_bits'] for v in one):.0f}"
    if two:
        N["jitter_two_rel"] = fmt(max(v["relative_threshold_jitter"] for v in two))
        N["jitter_two_bits"] = f"{max(v['max_abs_residual_bits'] for v in two):.0f}"
    # thresholds refined by the warm-started descent (run_finite_size.py --section polish)
    polished = []

    def walk(d, path=""):
        if isinstance(d, dict):
            if "polished_from" in d and "n_min" in d:
                polished.append((path, float(d["polished_from"]), float(d["n_min"])))
            for k, v in d.items():
                walk(v, f"{path}/{k}")
    for doc in (ren, sen, eat):
        walk(doc)
    ion2 = get(ren, "experiments/ion/rows/two_setting_qopt") or {}
    if "polished_from" in ion2:
        N["polish_ion_two"] = f"{ion2['polished_from'] - ion2['n_min']:,.0f}"
        others = [(a - b) / a for _, a, b in polished if a != ion2["polished_from"]]
        N["polish_max_rel"] = fmt(max(others)) if others else "0"
        N["polish_count"] = str(len(polished))
    # sensitivity
    if sen:
        rows = [["Parameter", "Value", "Ions", "Atoms, 11 km"]]
        for eps in (1e-5, 1e-8, 5e-11, 1e-12, 1e-15):
            rows.append(["soundness per link ε_{snd}", eps_s(eps), sci(get(sen, f"eps/ion_{eps:g}/n_min")),
                         sci(get(sen, f"eps/atom11km_{eps:g}/n_min"))])
        for f in ("1", "1.1", "1.2", "1.3"):
            rows.append(["error-correction efficiency f", f, sci(get(sen, f"ec_f/ion_{f}/n_min")),
                         sci(get(sen, f"ec_f/atom11km_{f}/n_min"))])
        rows.append(["error-correction efficiency f", "Cascade budget (1.20 key, 1.33 test)",
                     sci(get(sen, "ec_f/ion_cascade/n_min")), sci(get(sen, "ec_f/atom11km_cascade/n_min"))])
        for lab, txt in (("3sigma", "3 one-sided tails, Φ(−3) each (this work)"), ("hahn_1e-3_6tails", "6 tails sharing 10^{−3} (Ref. {ref:hahn})"),
                         ("1e-2_3tails", "3 tails sharing 10^{−2}")):
            ca = get(sen, f"tails/ion_{lab}/completeness_AT")
            rows.append([f"acceptance tails (ε^{{AT}}_{{com}} = {ca:.2g})" if ca else "acceptance tails", txt,
                         sci(get(sen, f"tails/ion_{lab}/n_min")), sci(get(sen, f"tails/atom11km_{lab}/n_min"))])
        for T2 in ("0.001", "0.01", "0.0275", "0.1"):
            m = get(sen, f"memory/{T2}")
            if m:
                rows.append(["memory T_{2} over the extra L/c (S after the wait)", f"{float(T2) * 1e3:g} ms (S = {m['S']:.3f})",
                             "–", sci(m["n_min"])])
        for lab, txt in (("nm_1e6", "n_{m} = 10^{6} bits: n_{s} = 56"), ("nm_1e9", "n_{m} = 10^{9} bits: n_{s} = 66"),
                         ("bian_82", "composable, n_{s} = 82 {ref:bian}")):
            rows.append(["signature length", txt, sci(get(sen, f"targets/ion_one_setting_{lab}/n_min")),
                         sci(get(sen, f"targets/atom11km_one_setting_{lab}/n_min"))])
        table("T_sensitivity", "Sensitivity of the rounds to the first signature (one setting, common flag, q optimised) "
              "for the ion and 11 km atom statistics.", rows)
        a = [get(sen, f"eps/{k}_{e:g}/n_min") for k in ("ion", "atom11km") for e in (1e-5, 1e-15)]
        if all(a):
            N["eps_factor"] = f"{max(a[1] / a[0], a[3] / a[2]):.2f}"
        f12 = get(sen, "ec_f/atom11km_1.2/n_min"), get(sen, "ec_f/atom11km_1/n_min")
        if all(f12):
            N["f12_factor_11km"] = f"{f12[0] / f12[1]:.2f}"
            N["a11_f12"] = sci(f12[0])
        N["a11_cascade"] = sci(get(sen, "ec_f/atom11km_cascade/n_min"))
        N["ion_cascade"] = sci(get(sen, "ec_f/ion_cascade/n_min"))
        tl = [get(sen, f"tails/{k}_{lab}/n_min") / get(sen, f"tails/{k}_3sigma/n_min") - 1
              for k in ("ion", "atom11km") for lab in ("hahn_1e-3_6tails", "1e-2_3tails") if get(sen, f"tails/{k}_{lab}/n_min")]
        if tl:
            N["tails_range"] = f"{100 * min(tl):+.0f} % to {100 * max(tl):+.0f} %".replace("-", "−")
        worst = [get(sen, f"ec_f/{k}_1.3/n_min") for k in ("ion", "atom11km")]
        N["worst_ion"], N["worst_11km"] = sci(worst[0]), sci(worst[1])
        mem = sen.get("memory", {})
        if mem:
            m = mem.get("0.0275", {})
            N["T2_cons_ms"] = "27.5"
            N["T2_S"] = f"{m.get('S', 0):.3f}"
            N["T2_nmin"] = sci(m.get("n_min"))
            N["T2_sigs"] = sg(m.get("signatures_recorded"))
            N["T2_sigs_cascade"] = sg(m.get("signatures_recorded_cascade"))
            for T2 in ("0.001", "0.01", "0.1"):
                N[f"T2_{T2}_S"] = f"{mem[T2]['S']:.3f}"
                N[f"T2_{T2}_nmin"] = sci(mem[T2]["n_min"])
        # targets and n_m
        for k in ("ion", "atom11km"):
            for lab in ("nm_1e6", "nm_1e9", "bian_82"):
                v = get(sen, f"targets/{k}_one_setting_{lab}")
                if v:
                    N[f"{k}_{lab}_nmin"] = sci(v["n_min"])
                    N[f"{k}_{lab}_sigs"] = sg(v["signatures_recorded"])
            v = get(sen, f"targets/{k}_two_setting_bian_82")
            if v:
                N[f"{k}_two_bian_nmin"] = sci(v["n_min"])
            base = get(sen, f"targets/{k}_one_setting_nm_1e3/n_min")
            b82 = get(sen, f"targets/{k}_one_setting_bian_82/n_min")
            if base and b82:
                N[f"{k}_bian_shift"] = f"{100 * (b82 / base - 1):.1f}"
        N["n_s_of_n_m"] = {f"{nm:g}": QDSBudget(n_m=int(nm)).n_s for nm in (1e3, 8e3, 1e6, 8e6, 1e9)}
        mis = sen.get("misalignment", {})
        for k, v in mis.items():
            N[f"mis_{k}_Qx"] = pct(v["Q_x"], 2)
            N[f"mis_{k}_nmin"] = sci(v["n_min"])

    ells_400 = []                    # the 400 m link: larger steps of the integer acceptance counts (high error rate)
    for ek, d in E.items():
        for v in d.get("rows", {}).values():
            if math.isfinite(v.get("n_min", math.inf)):
                (ells_400 if ek == "atom400m" else ells).append(v["ell_at_n_min"])
    for ek, d in eat.get("experiments", {}).items():
        for v in d.values():
            if isinstance(v, dict) and math.isfinite(v.get("n_min", math.inf)) and "ell_at_n_min" in v:
                (ells_400 if ek == "atom400m" else ells).append(v["ell_at_n_min"])
    for v in eat.get("two_setting_11km", {}).values():
        if isinstance(v, dict) and math.isfinite(v.get("n_min", math.inf)):
            ells.append(v["ell_at_n_min"])
    for sec in ("eps", "ec_f", "tails", "memory", "misalignment"):
        for v in sen.get(sec, {}).values():
            if isinstance(v, dict) and math.isfinite(v.get("n_min", math.inf)) and v.get("target", 138) == 138:
                ells.append(v["ell_at_n_min"])
    for d in eat.get("nmin_vs_V", {}).values():
        for v in d.values():
            if isinstance(v, dict) and math.isfinite(v.get("n_min", math.inf)):
                ells.append(v["ell_at_n_min"])
    if ells:
        N["ell_at_threshold_range"] = f"{math.floor(min(ells)):.0f}–{math.ceil(max(ells)):.0f}"
    if ells_400:
        N["ell_at_threshold_range_400m"] = f"{math.floor(min(ells_400)):.0f}–{math.ceil(max(ells_400)):.0f}"

# ------------------------------------------------------------------------------------------------ randomness
def randomness():
    led, plat, par = load("randomness_ledger.json"), load("randomness_platforms.json"), load("randomness_pareto.json")
    names = {"ion": "Ions, 2 m", "atom11km": "Atoms, 11 km", "qd_realistic_0km": "Photonic, 0 km",
             "qd_realistic_10km": "Photonic, 10 km"}
    pnames = {"two_setting": "two settings", "one_setting": "one setting, common flag",
              "independent": "one setting, independent flags", "seeded": "pre-shared seed"}
    if led:
        rows = [["Platform", "Protocol", "First signature (rounds)", "Private bits, first signature (whole block)",
                 "Private bits per signature, 10× block", "Public seed per signature, 10× block",
                 "Pre-shared secret, first signature", "Auth. key per signature, 10× block"]]
        for pk, row in led.items():
            for proto, e in row.items():
                if not math.isfinite(e.get("n_first", math.inf)):
                    continue
                f, x = e["first_entropy"], e["x10_entropy"]
                pre = f["block"]["shared_secret_net"]
                rows.append([names.get(pk, pk), pnames[proto], sci(e["n_first"]), sci(f["block"]["private_input"], unit=True),
                             sci(x["private_input"]), sci(x["public_fresh"]), sci(pre) if pre > 0 else "–",
                             num(x["auth_total"])])
                N[f"led_{pk}_{proto}_first"] = sci(f["block"]["private_input"], unit=True)
                N[f"led_{pk}_{proto}_x10"] = num(x["private_input"]) if x["private_input"] < 1e5 else sci(x["private_input"])
                N[f"led_{pk}_{proto}_seed"] = num(x["public_fresh"])
                N[f"led_{pk}_{proto}_auth"] = num(x["auth_total"])
                N[f"led_{pk}_{proto}_dyadic_first"] = sci(e["first_dyadic8"]["block"]["private_input"], unit=True)
        table("T_ledger", "Trusted bits per link, entropy-efficient sampler, key-optimal parameters. The first signature "
              "pays for its whole threshold block; the other columns are per signature at ten times that block. "
              "Auth.: authentication key, 64-bit Wegman–Carter tags with key recycling, two per block and link plus three per "
              "signature between Bob and Charlie.", rows)
        for key, prs in (("priv_one_range", ("one_setting",)), ("priv_ind_range", ("independent",)),
                         ("priv_seeded_range", ("seeded",))):
            x10 = [led[p][pr]["x10_entropy"]["private_input"] for p in ("ion", "atom11km") for pr in prs
                   if pr in led.get(p, {}) and "x10_entropy" in led[p][pr]]
            if x10:
                N[key] = f"{min(x10):,.0f}–{max(x10):,.0f}"
        x10t = [led[p]["two_setting"]["x10_entropy"]["private_input"] for p in ("ion", "atom11km") if "two_setting" in led.get(p, {})]
        if x10t:
            N["priv_two_range"] = f"{min(x10t):,.0f}–{max(x10t):,.0f}"
        ph = [led[p][pr]["x10_entropy"]["private_input"] for p in ("qd_realistic_0km", "qd_realistic_10km")
              for pr in ("two_setting", "independent") if pr in led.get(p, {}) and "x10_entropy" in led[p][pr]]
        if ph:
            N["priv_photonic_range"] = f"{sci(min(ph))}–{sci(max(ph))}"
        seeds = [led[p][pr]["x10_entropy"]["public_fresh"] for p in ("ion", "atom11km") for pr in ("one_setting",)
                 if pr in led.get(p, {})]
        if seeds:
            N["seed_x10_range"] = f"{min(seeds):,.0f}–{max(seeds):,.0f}"
    if plat:
        rows = [["Platform", "Target S / Q", "Fitted noise", "Model S / Q", "max |P_{Aer} − P_{QuTiP}|"]]
        c = plat.get("calibration", {})
        for k, lab in (("ion", "Ions, 2 m"), ("atom11km", "Atoms, 11 km")):
            if k in c:
                v = c[k]
                rows.append([lab, f"{v['S_target']:.3f} / {pct(v['Q_target'], 2)}",
                             f"λ = {v['lam']:.4f}, μ = {v['mu']:.4f}, ε_{{r}} = {v['eps_r_assumed']:g}",
                             f"{v['S_qutip']:.4f} / {pct(v['Q_qutip'], 2)}", f"{v['max_abs_aer_minus_qutip']:.1e}"])
        if "qd_realistic" in c:
            v = c["qd_realistic"]
            rows.append(["Photonic (QD, realistic)", f"{v['S_target']:.3f} / {pct(v['Q_photon_number_model'], 2)}",
                         f"v_{{eff}} = {v['v_eff']:.4f}, η = {v['eta_l']:g}", f"{v['S_qutip']:.4f} / {pct(v['Q_qutip'], 2)}",
                         f"{v['max_abs_aer_minus_qutip']:.1e}"])
        table("T_devices", "Device models fitted to the reported statistics; the last column compares the Qiskit Aer and "
              "QuTiP probability tables (a software cross-check).", rows)
        rows = [["Platform", "Attempts per herald (simulated / analytic)", "Heralds per second (simulated / analytic)",
                 "Inputs drawn"]]
        for k, v in plat.get("sequence", {}).items():
            rows.append([names.get(k, k), f"{v['attempts_per_herald_sim']:,.0f} / {v['attempts_per_herald_analytic']:,.0f}",
                         f"{v['herald_rate_sim']:.3g} / {v['herald_rate_analytic']:.3g}",
                         "per attempt" if v["per_attempt_inputs"] else "per herald"])
        table("T_sequence", "SeQUeNCe discrete-event simulation of the attempt loop.", rows)
        pm = plat.get("photonic_model", {})
        for k, v in pm.items():
            N[f"qd_{k}_S"] = f"{v['S']:.3f}"
            N[f"qd_{k}_Q"] = pct(v["Q"], 2)
    if par:
        N["pareto"] = {k: v.get("envelope", []) for k, v in par.items()}


# ------------------------------------------------------------------------------------------------ privacy amplification
def privacy():
    pa = load("privacy_amplification.json")
    if not pa:
        return
    w = pa.get("window", {})
    plan, cap = get(w, "plans/q0", {}), get(w, "capacity/q0", {})
    fr = get(w, "full_runs", {})
    if plan and cap:
        lb317 = cap.get("lower_bound_at_317", {})
        rows = [["Quantity", "Value"],
                ["Signatures with privacy amplification", sg(plan["signatures_with_pa"])],
                ["Segments a no-PA scheme would sign", sg(plan["signatures_claimed_without_pa"])],
                ["Acceptance probability, honest devices (exact)", f"{plan['p_accept_honest']:.4f}"],
                ["Acceptance probability, window of 317 rounds: certified lower bound (coupling)", f"{lb317.get('bound', math.nan):.4f}"],
                ["Longest window with acceptance ≥ 0.99 × honest: certified / estimate",
                 f"{num(cap.get('W_99_certified'))} / {num(cap['W_99'])} rounds ({num(cap.get('segments_known_99_certified'))} / {num(cap['segments_known_99'])} segments known to Eve)"],
                ["Longest window with acceptance ≥ 1/2: certified / estimate",
                 f"{num(cap.get('W_half_certified'))} / {num(cap['W_half'])} rounds ({num(cap.get('segments_known_half_certified'))} / {num(cap['segments_known_half'])} segments)"]]
        if "none" in fr:
            n_ = fr["none"]
            rows.append([f"{n_['trials']} complete runs, {n_['window_rounds']}-round window, no PA",
                         f"{n_['accepted']}/{n_['trials']} accepted; {n_['targeted_forged']} of {n_['targeted_segments']} targeted segments forged; {n_['other_forged']} of {n_['other_tried']} others"])
        if "toeplitz" in fr:
            t_ = fr["toeplitz"]
            rows.append(["Same devices with Toeplitz PA", f"{t_['accepted']}/{t_['trials']} accepted; {t_['targeted_forged'] + t_['other_forged']} of {t_['targeted_segments'] + t_['other_tried']} forged"])
        table("T_window", "Proposition 1 on Aer devices calibrated to the 11 km link (one setting, n = 1.208×10^{6}, q = 0, "
              "Cascade error correction).", rows)
        for k in ("W_99", "W_half", "W_99_certified", "W_half_certified", "segments_known_99", "segments_known_half",
                  "segments_known_99_certified", "segments_known_half_certified"):
            N[f"win_{k}"] = num(cap.get(k))
        N["win_lb317"] = f"{lb317.get('bound', math.nan):.4f}"
        N["win_honest"] = f"{plan['p_accept_honest']:.4f}"
        N["win_sigs_pa"] = sg(plan["signatures_with_pa"])
        N["win_segments"] = sg(plan["signatures_claimed_without_pa"])
        if "none" in fr:
            N["win_runs"] = f"{fr['none']['accepted']}/{fr['none']['trials']}"
            N["win_forged"] = f"{fr['none']['targeted_forged']} of {fr['none']['targeted_segments']}"
    iid = pa.get("iid", {})
    if iid:
        rows = [["Link", "n", "S^{*}", "P_{acc}", "p_{g}", "P(sig.)", "Segments", "P(block)"]]
        for c in iid["cases"]:
            rows.append([{"atom11km": "Atoms, 11 km", "ion": "Ions, 2 m"}[c["platform"]], sci(c["n"]), f"{c['S_star']:.3f}",
                         f"{c['p_accept']:.2f}", f"{c['p_guess_bit']:.3f}", sci(c["joint"], 2, True), num(c["segments"]),
                         sci(c["block_joint"], 2, True)])
        table("T_iid", "Proposition 2 on the one-setting protocol (Cascade budget, q optimised). P(sig.): accept and forge one "
              "targeted signature; P(block): accept and forge at least one segment of the block (per-session notion, "
              "Sec. II B). P_{acc} is the acceptance probability of the 3σ test.", rows)
        c0 = iid["cases"][0]
        N["iid_S"] = f"{c0['S_star']:.3f}"
        N["iid_pacc"] = f"{c0['p_accept']:.2f}"
        N["iid_pg"] = f"{c0['p_guess_bit']:.4f}"
        N["iid_joint"] = sci(c0["joint"], 2, True)
        N["iid_block"] = sci(c0["block_joint"], 2, True)
        brk = [c for c in iid["cases"] if c["platform"] == "atom11km" and c["breaks"]]
        N["iid_breaks_upto"] = f"{max(c['n'] for c in brk) / 1.208e6:.0f}" if brk else "0"
        ions = [c for c in iid["cases"] if c["platform"] == "ion"]
        if ions:
            N["iid_ion_max"] = sci(max(c["joint"] for c in ions), 2, True)
        ch = iid.get("checks", {})
        if ch:
            N["iid_mc"] = f"{ch['acceptance_mc']:.3f}"
            N["iid_mc_trials"] = num(ch["trials"])
            N["iid_mc_analytic"] = f"{ch['acceptance_analytic']:.3f}"
    cert = pa.get("certify", {})
    if cert:
        rows = [["Data set", "S", "p_{g}(S)", "h_{min} per raw bit", "q_{1/2}", "Key rate at q_{1/2} (asymptotic)",
                 "Min-entropy forging bound, n_{s} = 46 / 82 / 200"]]
        nm = {"ion": "Ions, 2 m", "atom11km": "Atoms, 11 km", "atom400m": "Atoms, 400 m", "qd_realistic": "Photonic (QD)"}
        for k, v in cert["platforms"].items():
            fb = v["forging_bound_min_entropy_route"]
            rows.append([nm[k], f"{v['S']:.3f}", f"{v['p_guess']:.4f}", f"{v['h_min_q0']:.3f}", f"{v['q_half']:.3f}",
                         f"{v['asymptotic_rate_one_setting_at_q_half']:.3f}".replace("-", "−"), " / ".join("1" if fb[s] >= 1 else sci(fb[s], 2, True) for s in ("46", "82", "200"))])
            N[f"thm2_{k}_h"] = f"{v['h_min_q0']:.3f}"
            N[f"thm2_{k}_qhalf"] = f"{v['q_half']:.3f}"
            N[f"thm2_{k}_rate"] = f"{v['asymptotic_rate_one_setting_at_q_half']:.3f}".replace("-", "−")
        table("T_thm2", "Theorem 2 at the reported scores: min-entropy per raw key bit under the accepted i.i.d. attack, "
              "the preprocessing q_{1/2} that would lift it to 1/2, the asymptotic one-setting key rate at that q, and the "
              "smallest forging bound any min-entropy argument can give for raw hash keys (n_{m} = 1000).", rows)
        N["S_half"] = f"{cert['S_half']:.3f}"
        nsn = cert.get("n_s_needed", {})
        N["ns_needed_055"] = f"{nsn.get('0.55', math.nan):.0f}"
        N["ns_needed_051"] = f"{nsn.get('0.51', math.nan):.0f}"
        sh = cert.get("sampling_share", {})
        for k, v in sh.items():
            N[f"share_{k}"] = f"{v['share_of_138']:.0f}"
    seeds = pa.get("seeds", {})
    fam = pa.get("family", {})
    key = "atom11km/one_setting/x10"
    if key in seeds:
        s = seeds[key]
        rows = [["Option", "Seed per block", "Bits per signature (11 km, 10× block)", "Secure"],
                ["Toeplitz", "N + ℓ − 1", num(s["toeplitz_per_signature"]), "yes"],
                ["Modified Toeplitz [T | I] {ref:hayashi}", "N − 1", num(s["modified_toeplitz_per_signature"]), "yes"],
                ["No privacy amplification", "0", "0", "no (Props. 1–2, Theorem 2)"],
                ["Per-signature hash, consecutive chunks", "m − 1 per signature", num(s["local_pa_chunks_per_signature"]), "no (Prop. 1)"],
                ["Per-signature hash, sampled subsets", "log_{2} C(N, m) + m − 1 per signature", num(s["local_pa_sampled_per_signature"]), "yes"],
                ["Public permutation, then chunks", "log_{2} N! + N_{sig}(m − 1)", num(s["permutation_then_chunks_per_signature"]), "yes"]]
        table("T_seeds", "Seed options; m = 138N/ℓ raw bits per signature.", rows)
        N["seed_toep"] = num(s["toeplitz_per_signature"])
        N["seed_mod"] = num(s["modified_toeplitz_per_signature"])
        N["seed_sampled"] = num(s["local_pa_sampled_per_signature"])
    first = seeds.get("atom11km/one_setting/first")
    if first:                                    # the first signature pays for the whole seed
        N["seed_first"] = sci(first["toeplitz_per_signature"])
    elif fam:
        N["seed_first"] = sci(fam["rows"][0]["toeplitz"])


# ------------------------------------------------------------------------------------------------ validation
def validation():
    v, ps = load("validation.json"), load("protocol_session.json")
    if not v:
        return
    rows = [["Tails", "Device", "f", "Runs", "MC acceptance", "95 % CI (Clopper–Pearson)", "Exact"]]
    inside = 0
    tot = 0
    for r in v["acceptance"]["rows"]:
        tag = "3σ" if r["tails"] == "tails_3sigma" else "0.02"
        dev = r["device"].replace("fraction P=", "deterministic fraction, target ")
        rows.append([tag, dev, f"{r['f']:.4f}", num(r["trials"]), f"{r['mc']:.3f}", f"{r['ci95'][0]:.3f}–{r['ci95'][1]:.3f}",
                     f"{r['analytic']:.4f}"])
        if r["device"] != "deterministic second half":
            tot += 1
            inside += int(r["ci95"][0] <= r["analytic"] <= r["ci95"][1])
    table("T_acceptance", "Acceptance test: Monte Carlo on Aer devices against the exact binomial probability. Deterministic "
          "fraction: devices deterministic in a random fraction f of rounds, tuned to an exact acceptance of 0.9, 0.5 and "
          "0.1; deterministic second half: devices honest for the first half of the block and deterministic afterwards.", rows)
    N["acc_inside"] = f"{inside} of {tot}"
    hon = [r for r in v["acceptance"]["rows"] if r["device"] == "honest" and r["tails"] == "tails_3sigma"]
    if hon:
        N["acc_honest_mc"] = f"{hon[0]['mc']:.3f}"
        N["acc_honest_exact"] = f"{hon[0]['analytic']:.4f}"
    pl = v["pipeline"]
    rows = [["Run", "n per link", "Accepted", "Signatures/block", "Honest verified", "Altered documents accepted",
             "Max leak/budget"]]
    labels = (("one_setting", "One setting"), ("seeded", "Pre-shared seed"), ("two_setting", "Two settings"),
              ("one_setting_aer_shots", "One setting, Aer shots"), ("two_setting_1e7_aer_shots", "Two settings, Aer shots"),
              ("deterministic_second_half", "Deterministic second half"))
    n_hon = n_forge = 0
    for k, lab in labels:
        d = pl.get(k)
        if not d:
            continue
        if k == "deterministic_second_half":
            rows.append([lab, sci(d.get("n", 1.208e6)), f"{d['accepted']}/{d['trials']}", "–", "–", "–", "–"])
            continue
        rows.append([lab, sci(d["n"]), f"{d['accepted']}/{d['trials']}", sg(d["signatures_per_block"]),
                     f"{d['honest_ok']}/{d['honest_total']}", f"{d['forged_accepted']}/{d['forged_tried']}",
                     f"{d['leak_over_budget_max']:.2f}"])
        n_hon += d["honest_total"]
        n_forge += d["forged_tried"]
        N[f"run_{k}_sigs"] = sg(d["signatures_per_block"])
    table("T_runs", "Complete DI-QDS runs on the 11 km link (both links, Cascade, verification, Toeplitz privacy "
          "amplification, OTUH); a consistency and completeness check of the implementation.", rows)
    N["runs_honest"] = num(n_hon)
    N["runs_forge"] = num(n_forge)
    ec = v.get("ec_efficiency", {}).get("rows", [])
    if ec:
        key = [r for r in ec if "key" in r["label"] or "basis" in r["label"]]
        N["ec_key_range"] = f"{min(r['f_mean'] for r in key if r['Q'] < 0.06):.2f}–{max(r['f_mean'] for r in key if r['Q'] < 0.06):.2f}"
        N["ec_test"] = f"{[r for r in ec if 'test' in r['label']][0]['f_mean']:.2f}"
        N["ec_400m"] = f"{[r for r in ec if '400' in r['label']][0]['f_mean']:.2f}"
    evv = v.get("error_verification", {}).get("passes_2", {})
    if evv:
        N["ev_residual"] = f"{evv['residual']} of {evv['trials']}"
        N["ev_pass"] = num(evv["accepted"])
    if ps:
        N["session_sigs"] = sg(get(ps, "operating_point/signatures") or get(ps, "messaging/signatures"))


def main():
    for section in (finite_size, randomness, privacy, validation):
        try:
            section()
        except (KeyError, TypeError, IndexError) as exc:        # a result file from an older version or still running
            print(f"warning: {section.__name__} skipped ({type(exc).__name__}: {exc})")
    nt = count_tests()
    if nt:
        N["n_tests"] = str(nt)
    (RES / "article_numbers.json").write_text(json.dumps({"numbers": N, "tables": T}, indent=1, ensure_ascii=False,
                                                         default=float))
    out = ["# Tables of the article\n", "Generated by `scripts/make_tables.py` from the cached results in `results/`.\n"]
    for name, t in T.items():
        out.append(f"\n## {name}: {md(t['caption'])}\n")
        rows = t["rows"]
        out.append("| " + " | ".join(md(c) for c in rows[0]) + " |")
        out.append("|" + "---|" * len(rows[0]))
        for r in rows[1:]:
            out.append("| " + " | ".join(md(c) for c in r) + " |")
    (RES / "tables.md").write_text("\n".join(out) + "\n")
    print(f"wrote results/tables.md ({len(T)} tables) and results/article_numbers.json ({len(N)} numbers)")


if __name__ == "__main__":
    main()
