"""Finite-size security of the DI key-distribution stage.

    entropy.py            von Neumann (B2-B3) and Renyi (Hahn et al. Theorem 2, App. C) single-round rates; Lemma 1
    protocols.py          TWO_SETTING (Fan et al.), ONE_SETTING (common flag), INDEPENDENT (Lu et al.), SEEDED (Tan et al.)
    renyi.py              REAT key length (Hahn et al. Eq. D6): exact inner problem, certified outer minimisation
    eat.py                Fan et al.'s Eq. (9) (Tan et al. Theorem 1), readings "rigorous" / "loss_only" / "strict"
    optimize.py           maximisers, exact thresholds n_min(V), V_min(n), randomness-aware objectives
    operating_points.py   Werner states, the reported experiments, photonic statistics
"""
from .operating_points import ATOM_11KM, ATOM_400M, EXPERIMENTS, ION, QD_IDEAL, QD_REALISTIC  # noqa: F401
from .optimize import n_min, optimize_alpha, renyi_optimum, threshold, v_min  # noqa: F401
from .protocols import INDEPENDENT, ONE_SETTING, PROTOCOLS, SEEDED, TWO_SETTING, Protocol  # noqa: F401
from .renyi import RenyiResult, RenyiSettings, acceptance_set, completeness_exact, key_length  # noqa: F401
