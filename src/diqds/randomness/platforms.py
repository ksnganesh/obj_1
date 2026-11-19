"""Physical platforms: when inputs must be drawn (per herald or per attempt), attempt period, success probability.

Ions   Nadlinger et al. (arXiv:2109.14600, SI): attempts at 1 MHz; 1,133,526,525 attempts for 180,000 heralds
           (p = 1.588e-4, 6297 attempts per herald); the DIQKD run collected 1.5e6 rounds in ~8 h (52 /s), which fixes the
           duty cycle (re-cooling etc.) at 0.328.  Two-photon (coincidence) heralding.  Inputs after the herald.
Atoms  Lu et al. (arXiv:2602.09596): per-trial time 12 us + 3L/2c (94.5 us at 11 km), duty cycle ~0.15, mean herald
           rate 0.53 /s, hence p = 3.34e-4.  Single-photon-interference heralding (p_s = a_A eta_A + a_B eta_B).
           Inputs after the herald.
Photonic   Fan et al. Fig. 1(b), App. D: quantum-dot sources at nu = 75 MHz, local beam splitter T = 0.01, conversion
           eta_conv = 0.5, fibre attenuation length L0 = 22 km:  P_H(L) = T^2 eta_C(L)^2 / 2,  eta_C = eta_conv e^{-L/2L0}
           (one herald pattern).  The local photon is measured before the herald can
           arrive, so inputs are drawn for EVERY attempt.  Honest statistics: the L = 0 values of the photon-number model
           (finite_size/photonic.py; realistic source S = 2.450, Q = 6.65 %; ideal S = 2.631, Q = 3.48 %), held fixed in
           L (their degradation
           with L is a second-order effect for the randomness budget; the attempt count is not).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..common import Honest
from ..finite_size.operating_points import ATOM_11KM, ION, QD_IDEAL, QD_REALISTIC

C_FIBRE = 2.0e8                     # m/s


@dataclass(frozen=True)
class Platform:
    key: str
    name: str
    honest: Honest
    per_attempt_inputs: bool
    attempt_period_s: float
    p_herald: float
    duty_cycle: float
    distance_km: float
    scheme: str
    source: str

    @property
    def attempts_per_herald(self) -> float:
        return 1.0 / self.p_herald

    @property
    def herald_rate_hz(self) -> float:
        return self.duty_cycle * self.p_herald / self.attempt_period_s

    def hours_for(self, n_heralds: float) -> float:
        return n_heralds / self.herald_rate_hz / 3600.0


ION_ATTEMPTS, ION_HERALDS = 1_133_526_525, 180_000
ION_P = ION_HERALDS / ION_ATTEMPTS
ION_PLATFORM = Platform(
    "ion", "88Sr+ ions, 2 m (Nadlinger et al.)", ION.honest, False, 1.0e-6, ION_P,
    duty_cycle=(1.5e6 / (8 * 3600.0)) * 1.0e-6 / ION_P, distance_km=0.002, scheme="two-photon coincidence",
    source="Nadlinger et al. 2022 (SI: 1 MHz attempts, 1.13e9 attempts / 1.8e5 heralds; 1.5e6 rounds in 8 h)")

ATOM11_PERIOD = 12e-6 + 3.0 * 11e3 / (2.0 * C_FIBRE)
ATOM11_PLATFORM = Platform(
    "atom11km", "87Rb atoms, 11 km (Lu et al.)", ATOM_11KM.honest, False, ATOM11_PERIOD,
    p_herald=0.53 * ATOM11_PERIOD / 0.15, duty_cycle=0.15, distance_km=11.0, scheme="single-photon interference",
    source="Lu et al. 2026 (12 us + 3L/2c per trial, duty ~0.15, 0.53 heralds/s)")


@dataclass(frozen=True)
class PhotonicParams:
    T: float = 0.01
    eta_conv: float = 0.5
    L0_km: float = 22.0
    rate_hz: float = 75e6

    def eta_c(self, L_km: float) -> float:
        return self.eta_conv * math.exp(-L_km / (2.0 * self.L0_km))

    def p_herald(self, L_km: float) -> float:
        return self.T ** 2 * self.eta_c(L_km) ** 2 / 2.0


PHOTONIC = PhotonicParams()


def photonic_platform(L_km: float = 0.0, source: str = "realistic", params: PhotonicParams = PHOTONIC) -> Platform:
    honest = QD_REALISTIC if source == "realistic" else QD_IDEAL
    return Platform(f"qd_{source}_{L_km:g}km", f"quantum dots ({source}), {L_km:g} km", honest, True, 1.0 / params.rate_hz,
                    params.p_herald(L_km), 1.0, L_km, "photonic, heralded by {d1-, d2+}",
                    "Fan et al. App. D; statistics from the photon-number model (finite_size/photonic.py) at L = 0")


PLATFORMS = {p.key: p for p in (ION_PLATFORM, ATOM11_PLATFORM, photonic_platform(0.0), photonic_platform(10.0))}
