"""Honest devices on Qiskit Aer: the 2 x 4 setting pairs, a NoiseModel with depolarising, thermal-relaxation and readout
errors calibrated to a platform's (S, Q), exact outcome distributions and shot sampling.

Circuit per setting pair (x, y): h(0); cx(0, 1)            -> |Phi+>, depolarising_error(lam, 2) attached to the cx
                                  id(1)                     -> thermal_relaxation_error(T1 >> T2, T2, t): Z-flip mu on Bob's qubit
                                  ry(-theta_x, 0); ry(-theta_y, 1); measure   -> sigma(theta) = cos(theta) Z + sin(theta) X
                                  ReadoutError on both measurements (symmetric eps_r, or no-click binning [[eta, 1-eta], [0, 1]])
Alice theta = (0, pi/2); Bob theta = (0, pi/2, pi/4, -pi/4): y = 0, 1 key settings (aligned with Alice), y = 2, 3 CHSH.
Qubit 0 = Alice = classical bit 0 (Qiskit prints little-endian: 'c1c0').

Exact probabilities: the density-matrix simulator saves the pre-measurement probabilities (save_probabilities) and the
readout confusion matrices are applied to them, which is what ReadoutError does to shots.  Sampling: Aer shots grouped
by setting pair (rounds with equal settings are i.i.d.), or multinomial draws from the exact table for very long runs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..randomness.qutip_states import ALICE_ANGLES, BOB_ANGLES, MatterNoise, PhotonicNoise, statistics


@dataclass(frozen=True)
class AerNoise:
    lam: float = 0.0                       # two-qubit depolarising strength on the cx
    mu: float = 0.0                        # Z-flip probability on Bob's qubit (via thermal relaxation on an id gate)
    readout_a: Tuple[float, float] = (0.0, 0.0)   # (P(read 1 | 0), P(read 0 | 1)) for Alice
    readout_b: Tuple[float, float] = (0.0, 0.0)
    gate_time: float = 1.0e-6

    @classmethod
    def from_matter(cls, m: MatterNoise) -> "AerNoise":
        return cls(lam=m.lam, mu=m.mu, readout_a=(m.eps_r, m.eps_r), readout_b=(m.eps_r, m.eps_r))

    @classmethod
    def from_photonic(cls, p: PhotonicNoise) -> "AerNoise":
        # white noise of visibility v_eff = depolarising 1 - v_eff; no-click binning: true 0 read as 1 w.p. 1 - eta
        return cls(lam=1.0 - p.v_eff, mu=0.0, readout_a=(1.0 - p.eta_l, 0.0), readout_b=(1.0 - p.eta_l, 0.0))

    @classmethod
    def werner(cls, V: float) -> "AerNoise":
        return cls(lam=1.0 - V)

    @property
    def T2(self) -> float:
        return -self.gate_time / math.log(1.0 - 2.0 * self.mu) if self.mu > 0 else math.inf


def _confusion(e: Tuple[float, float]) -> np.ndarray:
    e01, e10 = e
    return np.array([[1.0 - e01, e01], [e10, 1.0 - e10]])        # rows: true outcome, columns: reported outcome


class AerDevice:
    """Honest i.i.d. devices simulated with Qiskit Aer."""

    def __init__(self, noise: AerNoise, seed: Optional[int] = None):
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, thermal_relaxation_error

        self.noise = noise
        nm = NoiseModel()
        if noise.lam > 0:
            nm.add_quantum_error(depolarizing_error(noise.lam, 2), ["cx"], [0, 1])
        if noise.mu > 0:
            nm.add_quantum_error(thermal_relaxation_error(1e9, noise.T2, noise.gate_time), ["id"], [1])
        for q, e in ((0, noise.readout_a), (1, noise.readout_b)):
            if e[0] > 0 or e[1] > 0:
                nm.add_readout_error(ReadoutError(_confusion(e)), [q])
        self.noise_model = nm
        self.backend = AerSimulator(noise_model=nm)
        self.dm_backend = AerSimulator(method="density_matrix", noise_model=nm)
        self._seed = seed
        self.circuits, self.prob_circuits = {}, {}
        for x, ta in enumerate(ALICE_ANGLES):
            for y, tb in enumerate(BOB_ANGLES):
                for meas in (True, False):
                    qc = QuantumCircuit(2, 2)
                    qc.h(0)
                    qc.cx(0, 1)
                    qc.id(1)
                    qc.ry(-ta, 0)
                    qc.ry(-tb, 1)
                    if meas:
                        qc.measure([0, 1], [0, 1])
                        self.circuits[(x, y)] = transpile(qc, self.backend, optimization_level=0)
                    else:
                        qc.save_probabilities()
                        self.prob_circuits[(x, y)] = transpile(qc, self.dm_backend, optimization_level=0)
        self.P = self._exact()

    def _exact(self) -> np.ndarray:
        Ra, Rb = _confusion(self.noise.readout_a), _confusion(self.noise.readout_b)
        P = np.zeros((2, 4, 2, 2))
        for (x, y), qc in self.prob_circuits.items():
            pr = np.asarray(self.dm_backend.run(qc).result().data(0)["probabilities"])
            M = np.array([[pr[0], pr[2]], [pr[1], pr[3]]])         # M[a, b], index = a + 2b
            P[x, y] = Ra.T @ M @ Rb                                   # reported (a, b)
        return P

    def statistics(self) -> Tuple[float, float, float]:
        return statistics(self.P)

    def sample(self, x: np.ndarray, y: np.ndarray, rng: np.random.Generator, mode: str = "shots") -> Tuple[np.ndarray, np.ndarray]:
        a = np.empty(x.size, np.uint8)
        b = np.empty(x.size, np.uint8)
        for (xx, yy), qc in self.circuits.items():
            idx = np.flatnonzero((x == xx) & (y == yy))
            if idx.size == 0:
                continue
            if mode == "shots":
                seed = int(rng.integers(0, 2 ** 31 - 1)) if self._seed is None else self._seed + 97 * xx + 13 * yy
                counts = self.backend.run(qc, shots=int(idx.size), seed_simulator=seed).result().get_counts()
                outs = np.concatenate([np.full(c, int(k[::-1][0]) + 2 * int(k[::-1][1]), np.uint8) for k, c in counts.items()])
                rng.shuffle(outs)
            else:
                p = self.P[xx, yy].T.reshape(4)                       # index a + 2b
                outs = rng.choice(4, size=idx.size, p=p / p.sum()).astype(np.uint8)
            a[idx] = outs & 1
            b[idx] = outs >> 1
        return a, b


class TableDevice:
    """Honest i.i.d. devices from an explicit probability table P[x, y, a, b] (NumPy sampler)."""

    def __init__(self, P: np.ndarray):
        self.P = np.asarray(P, dtype=float)

    def statistics(self):
        return statistics(self.P)

    def sample(self, x, y, rng, mode: str = "multinomial"):
        a = np.empty(x.size, np.uint8)
        b = np.empty(x.size, np.uint8)
        for xx in range(self.P.shape[0]):
            for yy in range(self.P.shape[1]):
                idx = np.flatnonzero((x == xx) & (y == yy))
                if idx.size:
                    p = self.P[xx, yy].T.reshape(4)
                    o = rng.choice(4, size=idx.size, p=p / p.sum())
                    a[idx] = o & 1
                    b[idx] = o >> 1
        return a, b
