"""Honest-device operating points: Werner states (Figs. 3-5 of Fan et al.) and the reported experiments.

Experiments (headline numbers of each paper; Fan et al.'s Fig. 6 uses these three with the key generation of
Nadlinger et al.):
    Nadlinger et al., Nature 607, 682 (2022)   88Sr+ ions, ~2 m     S = 2.677(6),  Q = 1.44(2) %   1.5e6 rounds, ~8 h
    Zhang et al., Nature 607, 687 (2022)       87Rb atoms, 400 m    S = 2.578(75), Q = 7.8(9) %    3342 rounds, 75 h
    Lu et al., Science 391, 592 (2026)         87Rb atoms, 11 km    S = 2.612(31), Q = 2.85(2) %   1.208e6 events, 624 h
         (mean heralded rate 0.53 /s; per-trial overhead 12 us + 3L/2c; duty cycle ~0.15; their own analysis:
          0.112 bit/event at eps_snd = 1e-5 and 0.075 bit/event at 1e-15, Renyi EAT, independent test flags with
          gamma_A = 0.26, gamma_B = 0.13 (supplement Eq. S45); see protocols.INDEPENDENT)
         Memory coherence: Hahn-echo decoupling over the 55 us wait at 11 km; Table S1 of the supplement attributes
         < 0.001 of the entanglement infidelity to atomic coherence.  For pure dephasing, an infidelity of 0.001 over
         55 us corresponds to T2 = 55 us / (2 x 0.001) = 27.5 ms, the conservative value used for the extra L/c wait of
         the one-setting protocol (ATOM11_T2_CONSERVATIVE).
Photonic link with quantum-dot sources (Fan et al. Fig. 1b, App. C-D, Fig. 7), honest statistics at L = 0 from the
photon-number-resolved model of that architecture (photonic.py, which reproduces these two points):
    ideal source        S = 2.631, Q = 3.48 %
    realistic source    S = 2.450, Q = 6.65 %    (V_HOM = 0.975, g2 = 0.01, Eq. C13 with kappa = 1.5446)
"""
from __future__ import annotations

from dataclasses import dataclass

from ..common import chsh_to_w, Honest


@dataclass(frozen=True)
class Experiment:
    key: str
    ref: str
    platform: str
    distance_km: float
    honest: Honest
    rounds: float
    hours: float

    @property
    def rounds_per_hour(self) -> float:
        return self.rounds / self.hours

    def hours_for(self, n: float) -> float:
        return n / self.rounds_per_hour


ION = Experiment("ion", "Nadlinger et al. 2022", "88Sr+ ions", 0.002, Honest.from_chsh(2.677, 0.0144, "ions (Nadlinger et al.)"),
                 1.5e6, 8.0)
# Zhang et al. measured their QBER in two key bases (Q_0 = 7.81 %, Q_1 = 7.77 %) that are NOT Alice's CHSH settings (key: Alice and
# Bob at -/+22.5 deg; CHSH: Alice at -45 and 0 deg), so the state-noise inference of Q_x does not apply: Q_x = Q.
ATOM_400M = Experiment("atom400m", "Zhang et al. 2022", "87Rb atoms", 0.4,
                       Honest(chsh_to_w(2.578), 0.078, "atoms 400 m (Zhang et al.)", Qx=0.078), 3342.0, 75.0)
ATOM_11KM = Experiment("atom11km", "Lu et al. 2026", "87Rb atoms", 11.0, Honest.from_chsh(2.612, 0.0285, "atoms 11 km (Lu et al.)"),
                       1.208e6, 624.0)
EXPERIMENTS = {e.key: e for e in (ION, ATOM_400M, ATOM_11KM)}

QD_IDEAL = Honest.from_chsh(2.631, 0.0348, "quantum dot, ideal source, L = 0")
QD_REALISTIC = Honest.from_chsh(2.450, 0.0665, "quantum dot, realistic source, L = 0")

# Markers of Fan et al.'s Fig. 6
FIG6_ION_ROUNDS = 1.5e6           # "~8 h of operation"
FIG6_ATOM11_ROUNDS = 2.4e6        # "~1248 h" (extrapolated; 2.4e6 events at the reported 0.53 /s is ~1,240 h)

ATOM11_T2_CONSERVATIVE = 27.5e-3  # s; from < 0.001 atomic-coherence infidelity over the 55 us wait (see above)
