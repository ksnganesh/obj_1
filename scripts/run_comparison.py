"""Comparison with published results (Sec. VIII): other papers' numbers next to ours, from the cached results.

Published values (sources):
  [Fan]  Fan, Bian, Li, Wen, Yin, Chen, Phys. Rev. A 114, 032617 (2026).  Table I: eps = 1e-10, eps_snd = 5e-11,
         n_m = 1e3, n_s = 46.  Minimum visibility for signing (text and Fig. 5): p = 0: 0.8835, 0.8686, 0.8622 at
         n = 1e8, 1e9, 1e10; p optimised: 0.8799, 0.8608, 0.8508.  Experiments (Sec. IV, Fig. 6, key generation of
         Nadlinger et al.): ions, 1.5e6 rounds in ~8 h are enough; atoms 11 km, 1.2e6 events in 624 h are not, the
         extrapolation n = 2.4e6 ("~1248 h"; 1,240 h at the reported 0.53 events/s) is; atoms 400 m need of order 1e7 h.
  [Bian] Bian, Li, Lu, Li, Yin, Chen, arXiv:2610.39360 (2026), composable DI-QDS: EAT key generation; related-key
         forging bound Pr[forge] <= sqrt(7 m~ / 2^n); eps_QDS = 1e-10 is reached with a signature length n = 82 bits for
         L_Doc = 1e3; multiplexed photonic links.
  [Nad]  Nadlinger et al., Nature 607, 682 (2022): 95,628 key bits from 1.5e6 Bell pairs in ~8 h.
  [Hahn] Hahn, Philip, Tan, Brown, arXiv:2507.07365 (2025): for the setting of [Nad], about 3x the finite-size key rate at
         n = 1.5e6 and nearly 3x smaller minimum n.
  [Zhang23] Zhang, Zeng, Ye, Lo, Ma, Phys. Rev. Lett. 131, 140801 (2023): the complementarity approach reduces the data
         requirement of the ion experiment to less than one third.
  [Lu]   Lu et al., Science 391, 592 (2026): 11 km, 1.208e6 events in 624 h; 0.112 bit/event at eps_snd = 1e-5 (Renyi
         EAT), 0.075 at 1e-15, 0.034 with the original EAT at 1e-5; independent test flags gamma_A = 0.26,
         gamma_B = 0.13 (supplement Eq. S45); asymptotic rates up to 100 km from at most a few thousand events per
         distance (QBER 7.16 % at 100 km), no finite-size key beyond 11 km.
Ours: finite_size_renyi.json, finite_size_sensitivity.json and finite_size_eat.json (run_finite_size.py).

    python scripts/run_comparison.py            # seconds; writes results/comparison.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from diqds.common import KEY_BITS_PER_SIGNATURE  # noqa: E402
from diqds.finite_size.operating_points import ATOM_11KM  # noqa: E402

RES = ROOT / "results"
BIAN_SIGNATURE_BITS = 82
PUBLISHED = {
    "fan": {"ref": "Fan et al., PRA 114, 032617 (2026)",
            "vmin_p0": {"1e+08": 0.8835, "1e+09": 0.8686, "1e+10": 0.8622},
            "vmin_popt": {"1e+08": 0.8799, "1e+09": 0.8608, "1e+10": 0.8508},
            "ion": {"rounds": 1.5e6, "hours": 8.0, "signs": True},
            "atom11km": {"recorded": 1.2e6, "recorded_hours": 624.0, "signs_with_recorded": False,
                         "extrapolated_rounds": 2.4e6, "extrapolated_hours_quoted": 1248.0,
                         "extrapolated_hours_at_reported_rate": ATOM_11KM.hours_for(2.4e6)},
            "atom400m": {"hours_order": 1e7},
            "photonic_reach_km": {"1e14": 17.82, "2e14": 28.87},
            "n_s": 46, "eps": 1e-10, "eps_snd": 5e-11, "n_m": 1000},
    "bian": {"ref": "Bian et al., arXiv:2610.39360 (2026)", "signature_bits": BIAN_SIGNATURE_BITS, "eps_qds": 1e-10,
             "L_doc": 1000, "forging_bound": "sqrt(7 m~ / 2^n), related-key", "finite_size": "EAT"},
    "nadlinger": {"ref": "Nadlinger et al., Nature 607, 682 (2022)", "key_bits": 95628, "rounds": 1.5e6, "hours": 8.0},
    "hahn": {"ref": "Hahn et al., arXiv:2507.07365 (2025)", "key_factor": 3.0, "nmin_factor": 3.0},
    "zhang23": {"ref": "Zhang et al., PRL 131, 140801 (2023)", "ion_data_requirement_factor": "< 1/3"},
    "lu": {"ref": "Lu et al., Science 391, 592 (2026)", "events": 1.208e6, "hours": 624.0,
           "bits_per_event": {"renyi_1e-05": 0.112, "renyi_1e-15": 0.075, "eat_1e-05": 0.034}, "key_kbit": 135.3,
           "gamma_A": 0.26, "gamma_B": 0.13, "qber_100km": 0.0716,
           "events_per_distance_beyond_11km": {"20": 4351, "50": 1193, "70": 1166, "100": 920}},
}


def load(name: str) -> dict:
    f = RES / name
    return json.loads(f.read_text()) if f.exists() else {}


def main() -> None:
    ren, sen, eat = load("finite_size_renyi.json"), load("finite_size_sensitivity.json"), load("finite_size_eat.json")
    E = ren["experiments"]
    ours = {
        "vmin": ren["vmin"], "eat_vmin": eat["vmin"],
        "experiments": {k: {"rows": {r: {"n_min": v["n_min"], "hours": v.get("hours"), "ell_at_n_min": v["ell_at_n_min"]}
                                     for r, v in d["rows"].items()},
                            "recorded": {r: {"ell": v["ell"], "signatures": v["signatures"]} for r, v in d["recorded"].items()}}
                        for k, d in E.items()},
        "eat_experiments": eat["experiments"], "eat_recorded": eat.get("recorded", {}),
        "eat_two_setting_11km": eat.get("two_setting_11km", {}),
        "crosscheck_hahn": {k: v for k, v in ren["crosscheck_hahn"].items() if k != "result"},
        "lu": {k: v for k, v in ren["lu"].items() if k != "rows"},
        "lu_rows": {k: {kk: vv for kk, vv in v.items() if kk != "result"} for k, v in ren["lu"]["rows"].items()},
        "targets": {k: {kk: v[kk] for kk in ("key_bits", "n_min", "hours", "signatures_recorded") if kk in v}
                    for k, v in sen.get("targets", {}).items()},
        "n_s_of_n_m": sen.get("n_s_of_n_m", {}),
        "memory": {k: {kk: v[kk] for kk in ("T2_s", "S", "n_min", "signatures_recorded", "signatures_recorded_cascade")}
                   for k, v in sen.get("memory", {}).items()},
    }
    out = {"published": PUBLISHED, "ours": ours, "key_bits_per_signature": KEY_BITS_PER_SIGNATURE}
    (RES / "comparison.json").write_text(json.dumps(out, indent=1, default=float))
    print("written results/comparison.json")


if __name__ == "__main__":
    main()
