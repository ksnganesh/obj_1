"""Discrete-event simulation of heralded rounds on SeQUeNCe's kernel, with a meter of the trusted random bits drawn.

What SeQUeNCe provides here: the event kernel (Timeline, Entity, Event, Process; integer picoseconds) and the fibre model
(sequence.components.optical_channel.QuantumChannel: delay = distance / light speed, loss = 1 - 10^(-att * distance / 10)).
Each end node attempts entanglement every ``attempt_period / duty_cycle`` seconds (the duty cycle spreads the attempts so
the long-run herald rate matches the reported one); the heralding station sits in the middle; a successful attempt
sends a herald back to both nodes after the photon's and the classical message's travel times through the fibre.

Randomness meter.  Inputs are drawn either at every attempt (photonic links: the local photon is measured before the
herald exists) or once per herald (memories wait for the herald).  The meter records the private input bits drawn,
from which the bits per heralded round and the required RNG throughput follow.

Two modes: "attempts" schedules every attempt (exact, used for p >~ 1e-4), "heralds" draws the geometric number of
attempts until the next success and schedules only the herald (exact in distribution, used for photonic links whose
p ~ 1e-5 would need ~1e5 events per herald).  The two are cross-checked in the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from .platforms import Platform

LIGHT_SPEED_M_PER_PS = 2e-4        # 2e8 m/s, SeQUeNCe's default fibre light speed


@dataclass
class HeraldRun:
    platform: str
    mode: str
    heralds: int
    attempts: int
    elapsed_s: float
    bits_drawn: float
    attempts_per_herald: List[int] = field(default_factory=list)
    channel_loss: float = float("nan")
    channel_delay_s: float = float("nan")

    @property
    def mean_attempts(self) -> float:
        return self.attempts / max(self.heralds, 1)

    @property
    def herald_rate_hz(self) -> float:
        return self.heralds / self.elapsed_s

    @property
    def bits_per_herald(self) -> float:
        return self.bits_drawn / max(self.heralds, 1)

    @property
    def rng_throughput_bps(self) -> float:
        return self.bits_drawn / self.elapsed_s


def fibre_channel(distance_km: float, attenuation_db_per_km: float = 0.2):
    """A SeQUeNCe QuantumChannel for one arm (node -> heralding station), initialised to read delay and loss."""
    from sequence.components.optical_channel import QuantumChannel
    from sequence.kernel.timeline import Timeline
    tl = Timeline()
    qc = QuantumChannel("arm", tl, attenuation=attenuation_db_per_km / 1000.0, distance=max(distance_km, 1e-6) * 1000.0,
                        light_speed=LIGHT_SPEED_M_PER_PS)
    qc.init()
    return qc


def simulate(platform: Platform, bits_per_round: float, n_heralds: int = 200, mode: str = "attempts",
             seed: int = 0) -> HeraldRun:
    """Run ``n_heralds`` heralded rounds of ``platform`` on SeQUeNCe's kernel; ``bits_per_round`` inputs per draw."""
    from sequence.kernel.entity import Entity
    from sequence.kernel.event import Event
    from sequence.kernel.process import Process
    from sequence.kernel.timeline import Timeline

    arm = fibre_channel(platform.distance_km / 2.0)
    herald_delay_ps = 2 * int(arm.delay)                          # photon to the station + herald back
    period_ps = max(1, int(round(platform.attempt_period_s / platform.duty_cycle * 1e12)))
    p = platform.p_herald
    rng = np.random.default_rng(seed)

    class EndNodes(Entity):
        def __init__(self, name, tl):
            super().__init__(name, tl)
            self.attempts = 0
            self.heralds = 0
            self.bits = 0.0
            self.last_attempts = 0
            self.per_herald: List[int] = []
            self.done = False

        def init(self):
            self._next(1)

        def _next(self, cycles):
            proc = Process(self, "attempt" if mode == "attempts" else "batch", [])
            self.timeline.schedule(Event(self.timeline.now() + cycles * period_ps, proc))

        def attempt(self):
            if self.done:
                return
            self.attempts += 1
            self.last_attempts += 1
            if platform.per_attempt_inputs:
                self.bits += bits_per_round
            if rng.random() < p:
                self.timeline.schedule(Event(self.timeline.now() + herald_delay_ps, Process(self, "herald", [])))
            else:
                self._next(1)

        def batch(self):
            k = int(rng.geometric(p))                              # attempts up to and including the success
            self.attempts += k
            self.last_attempts += k
            if platform.per_attempt_inputs:
                self.bits += k * bits_per_round
            self.timeline.schedule(Event(self.timeline.now() + (k - 1) * period_ps + herald_delay_ps,
                                         Process(self, "herald", [])))

        def herald(self):
            self.heralds += 1
            self.per_herald.append(self.last_attempts)
            self.last_attempts = 0
            if not platform.per_attempt_inputs:
                self.bits += bits_per_round
            if self.heralds >= n_heralds:
                self.done = True
                return
            self._next(1)

    tl = Timeline()
    nodes = EndNodes("ends", tl)
    tl.init()
    tl.run()
    return HeraldRun(platform.key, mode, nodes.heralds, nodes.attempts, tl.now() * 1e-12, nodes.bits,
                     nodes.per_herald, float(arm.loss), arm.delay * 1e-12)


def analytic(platform: Platform, bits_per_round: float) -> dict:
    """Expectations the simulation must reproduce."""
    k = platform.attempts_per_herald if platform.per_attempt_inputs else 1.0
    return {"attempts_per_herald": platform.attempts_per_herald, "herald_rate_hz": platform.herald_rate_hz,
            "bits_per_herald": bits_per_round * k, "rng_throughput_bps": bits_per_round * k * platform.herald_rate_hz}
