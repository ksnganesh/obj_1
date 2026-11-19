"""QuTiP effective two-qubit states of the three platforms, and the Qiskit Aer noise parameters they imply.

Matter qubits (ions, atoms).  |Phi+>, then a depolarising channel of strength lam (rho -> (1-lam) rho + lam I/4, models
imperfect heralded entanglement), then pure dephasing of Bob's memory while it waits for the inputs, obtained by
integrating the Lindblad equation (qutip.mesolve, collapse operator sqrt(1/(2 T2)) sigma_z on qubit B) for a storage time
t: coherences decay by e^{-t/T2} = 1 - 2 mu.  Measurements along sigma(theta) = cos(theta) Z + sin(theta) X with a
symmetric readout error eps_r.  Settings (as in Fan et al.): Alice x in {0, 1} -> theta = 0, pi/2;
Bob y in {0, 1} -> 0, pi/2 (key, aligned with Alice), y in {2, 3} -> +pi/4, -pi/4 (CHSH).  Win iff a xor b = x (y - 2).

With these settings S = sqrt2 (<ZZ> + <XX>) and Q = (1 - <ZZ>)/2, where <ZZ> = (1-lam)(1-2eps_r)^2 and
<XX> = (1-lam)(1-2mu)(1-2eps_r)^2.  Given (S, Q) and an assumed readout error the calibration is therefore closed form:
    1 - lam = (1 - 2Q)/(1 - 2eps_r)^2,     1 - 2mu = (S/sqrt2 - (1 - 2Q)) / (1 - 2Q).
The dephasing mu already present in the reported statistics is applied as a fixed Z-flip channel.  ONE_SETTING needs
the common test flag at Bob's node before he measures; if it is sent after the herald (to keep it from Eve), Bob's memory
waits an extra L/c, modelled by Lindblad dephasing (qutip.mesolve) with an assumed memory coherence time T2_memory.

Photonic link.  Heralded state (D1) |+-> - t/(1-t) |-+> (t = 1/2: a maximally entangled state; up to local unitaries
we use |Phi+>), partially distinguishable photons as a mixture with visibility V_eff = V_HOM - kappa g2 (Eq. C13), and a
local polarisation measurement with one threshold detector of efficiency eta_l and no-click binning:
E_0 = eta_l |theta><theta|, E_1 = 1 - E_0.  In Aer the binning is an asymmetric ReadoutError [[eta_l, 1-eta_l], [0, 1]].
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

import numpy as np

ALICE_ANGLES = (0.0, math.pi / 2)
BOB_ANGLES = (0.0, math.pi / 2, math.pi / 4, -math.pi / 4)


def _qt():
    import qutip as qt
    return qt


def bell_phi_plus():
    qt = _qt()
    return (qt.tensor(qt.basis(2, 0), qt.basis(2, 0)) + qt.tensor(qt.basis(2, 1), qt.basis(2, 1))).unit()


def depolarised(lam: float):
    qt = _qt()
    return (1.0 - lam) * qt.ket2dm(bell_phi_plus()) + lam * qt.tensor(qt.qeye(2), qt.qeye(2)) / 4.0


def dephase_bob(rho, t_store: float, T2: float):
    """Lindblad pure dephasing of qubit B for time t_store (qutip.mesolve)."""
    qt = _qt()
    if t_store <= 0 or not math.isfinite(T2):
        return rho
    c_op = math.sqrt(1.0 / (2.0 * T2)) * qt.tensor(qt.qeye(2), qt.sigmaz())
    H = 0 * qt.tensor(qt.qeye(2), qt.qeye(2))
    res = qt.mesolve(H, rho, [0.0, t_store], c_ops=[c_op], e_ops=[])
    return res.states[-1]


def sigma(theta: float):
    qt = _qt()
    return math.cos(theta) * qt.sigmaz() + math.sin(theta) * qt.sigmax()


def povm_projective(theta: float, eps_r: float = 0.0):
    qt = _qt()
    P0 = 0.5 * (qt.qeye(2) + sigma(theta))
    P1 = 0.5 * (qt.qeye(2) - sigma(theta))
    return [(1 - eps_r) * P0 + eps_r * P1, (1 - eps_r) * P1 + eps_r * P0]


def povm_no_click(theta: float, eta: float):
    qt = _qt()
    P0 = 0.5 * (qt.qeye(2) + sigma(theta))
    return [eta * P0, qt.qeye(2) - eta * P0]


def joint_probabilities(rho, povm_a: Sequence, povm_b: Sequence) -> np.ndarray:
    """P[x, y, a, b] for POVM lists povm_a[x] = [E_0, E_1], povm_b[y] = [F_0, F_1]."""
    qt = _qt()
    P = np.zeros((len(povm_a), len(povm_b), 2, 2))
    for x, Ea in enumerate(povm_a):
        for y, Fb in enumerate(povm_b):
            for a in (0, 1):
                for b in (0, 1):
                    P[x, y, a, b] = float(np.real(qt.expect(qt.tensor(Ea[a], Fb[b]), rho)))
    return P


def statistics(P: np.ndarray) -> Tuple[float, float, float]:
    """(S, w, Q): CHSH value, winning probability, and the key-round error of the aligned pair (x, y) = (0, 0)."""
    w = 0.0
    for x in (0, 1):
        for yp in (0, 1):
            w += 0.25 * sum(P[x, 2 + yp, a, b] for a in (0, 1) for b in (0, 1) if (a ^ b) == (x & yp))
    Q = P[0, 0, 0, 1] + P[0, 0, 1, 0]
    return 8.0 * w - 4.0, w, Q


def dephase_fixed(rho, mu: float):
    """Z-flip of qubit B with probability mu (the dephasing already present in the reported statistics)."""
    qt = _qt()
    Zb = qt.tensor(qt.qeye(2), qt.sigmaz())
    return (1.0 - mu) * rho + mu * Zb * rho * Zb


@dataclass(frozen=True)
class MatterNoise:
    lam: float            # two-qubit depolarising strength
    mu: float             # Z-flip probability of Bob's qubit already contained in the reported (S, Q)
    eps_r: float          # symmetric readout error per party

    def state(self, extra_wait: float = 0.0, T2_memory: float = math.inf):
        """Calibrated state, optionally after an extra storage time with memory coherence time T2_memory (mesolve)."""
        rho = dephase_fixed(depolarised(self.lam), self.mu)
        return dephase_bob(rho, extra_wait, T2_memory)

    def probabilities(self, extra_wait: float = 0.0, T2_memory: float = math.inf) -> np.ndarray:
        rho = self.state(extra_wait, T2_memory)
        pa = [povm_projective(t, self.eps_r) for t in ALICE_ANGLES]
        pb = [povm_projective(t, self.eps_r) for t in BOB_ANGLES]
        return joint_probabilities(rho, pa, pb)

    def aer_dephasing(self, gate_time: float = 1.0e-6):
        """(T1, T2, time) of an Aer thermal-relaxation error on an id gate that reproduces mu (T1 effectively infinite)."""
        if self.mu <= 0:
            return 1e9, 1e9, gate_time
        T2 = -gate_time / math.log(1.0 - 2.0 * self.mu)
        return 1e9, T2, gate_time


def calibrate_matter(S: float, Q: float, eps_r: float = 0.0) -> MatterNoise:
    zz = 1.0 - 2.0 * Q
    xx = S / math.sqrt(2.0) - zz
    r2 = (1.0 - 2.0 * eps_r) ** 2
    lam = 1.0 - zz / r2
    one_minus_2mu = xx / zz
    if not (0.0 <= lam < 1.0 and 0.0 < one_minus_2mu <= 1.0):
        raise ValueError(f"(S, Q, eps_r) = ({S}, {Q}, {eps_r}) is outside this noise family")
    return MatterNoise(lam=lam, mu=(1.0 - one_minus_2mu) / 2.0, eps_r=eps_r)


@dataclass(frozen=True)
class PhotonicNoise:
    v_eff: float          # visibility of the heralded state (white-noise mixture)
    eta_l: float = 0.97   # local efficiency (source extraction and detection), no-click binning

    def probabilities(self) -> np.ndarray:
        qt = _qt()
        rho = self.v_eff * qt.ket2dm(bell_phi_plus()) + (1 - self.v_eff) * qt.tensor(qt.qeye(2), qt.qeye(2)) / 4.0
        pa = [povm_no_click(t, self.eta_l) for t in ALICE_ANGLES]
        pb = [povm_no_click(t, self.eta_l) for t in BOB_ANGLES]
        return joint_probabilities(rho, pa, pb)


def calibrate_photonic(S_target: float, eta_l: float = 0.97) -> PhotonicNoise:
    """Visibility that reproduces S_target with standard angles and no-click binning (bisection)."""
    lo, hi = 0.5, 1.0
    if statistics(PhotonicNoise(1.0, eta_l).probabilities())[0] < S_target:
        return PhotonicNoise(1.0, eta_l)
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if statistics(PhotonicNoise(mid, eta_l).probabilities())[0] >= S_target:
            hi = mid
        else:
            lo = mid
    return PhotonicNoise(hi, eta_l)


def werner_probabilities(V: float) -> np.ndarray:
    P = np.zeros((2, 4, 2, 2))
    for x, th in enumerate(ALICE_ANGLES):
        for y, ph in enumerate(BOB_ANGLES):
            c = V * math.cos(th - ph)
            for a in (0, 1):
                for b in (0, 1):
                    P[x, y, a, b] = 0.25 * (1 + (-1) ** (a + b) * c)
    return P
