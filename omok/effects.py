"""Easing curves, tweened values and a small particle pool.

Deliberately free of arcade imports.  The window turns these numbers into
sprites, but the numbers themselves are plain numpy, which keeps the physics
testable without an OpenGL context and keeps the drawing code short.

The particle pool is fixed-capacity and compacted in place: a burst of a few
hundred sparks allocates nothing per frame, which matters when it fires on
every stone placed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

TAU = math.tau


# ------------------------------------------------------------------ easing
# All take t in [0, 1] and return the eased fraction.  ``back`` and
# ``elastic`` deliberately overshoot past 1.0 -- that overshoot is what makes a
# dropped stone feel like it has weight.
def linear(t: float) -> float:
    return t


def ease_out_quad(t: float) -> float:
    return 1.0 - (1.0 - t) ** 2


def ease_out_cubic(t: float) -> float:
    return 1.0 - (1.0 - t) ** 3


def ease_in_out_cubic(t: float) -> float:
    return 4.0 * t ** 3 if t < 0.5 else 1.0 - ((-2.0 * t + 2.0) ** 3) / 2.0


def ease_out_back(t: float, overshoot: float = 1.9) -> float:
    c3 = overshoot + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + overshoot * (t - 1.0) ** 2


def ease_out_elastic(t: float) -> float:
    if t <= 0.0 or t >= 1.0:
        return t
    return 2.0 ** (-9.0 * t) * math.sin((t * 10.0 - 0.75) * (TAU / 3.0)) + 1.0


def pulse(clock: float, period: float = 1.4) -> float:
    """A smooth 0..1 breathing curve, for anything that should feel alive."""
    return 0.5 - 0.5 * math.cos(clock * TAU / period)


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return low if value < low else high if value > high else value


def mix(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# ------------------------------------------------------------------ tweens
class Animated:
    """A scalar that eases toward whatever it is last set to.

    Used for the evaluation bar and the search progress, so a number that jumps
    when a search returns slides into place instead.
    """

    def __init__(self, value: float = 0.0, duration: float = 0.4,
                 ease=ease_out_cubic) -> None:
        self.value = self.start = self.target = float(value)
        self.duration = duration
        self.ease = ease
        self.elapsed = duration

    def set(self, target: float, snap: bool = False) -> None:
        target = float(target)
        if snap:
            self.value = self.start = self.target = target
            self.elapsed = self.duration
            return
        if target == self.target:
            return
        self.start, self.target, self.elapsed = self.value, target, 0.0

    def update(self, dt: float) -> None:
        if self.elapsed >= self.duration:
            self.value = self.target
            return
        self.elapsed += dt
        t = clamp(self.elapsed / self.duration)
        self.value = mix(self.start, self.target, self.ease(t))

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration


@dataclass
class Clip:
    """A one-shot timer.  ``progress`` runs 0 -> 1, then it is ``done``."""

    duration: float
    elapsed: float = 0.0
    data: object = None

    def update(self, dt: float) -> bool:
        self.elapsed += dt
        return not self.done

    @property
    def progress(self) -> float:
        return clamp(self.elapsed / self.duration) if self.duration > 0 else 1.0

    @property
    def done(self) -> bool:
        return self.elapsed >= self.duration


# --------------------------------------------------------------- particles
class Particles:
    """A fixed-capacity pool of sparks, integrated as whole numpy arrays."""

    def __init__(self, capacity: int = 900, seed: int | None = None) -> None:
        self.capacity = capacity
        self.count = 0
        self.rng = np.random.default_rng(seed)
        zeros = lambda: np.zeros(capacity, dtype=np.float32)  # noqa: E731
        self.x, self.y = zeros(), zeros()
        self.vx, self.vy = zeros(), zeros()
        self.age, self.life = zeros(), np.ones(capacity, dtype=np.float32)
        self.size, self.grow = zeros(), zeros()
        self.angle, self.spin = zeros(), zeros()
        self.gravity, self.drag = zeros(), zeros()
        self.red, self.green, self.blue = zeros(), zeros(), zeros()
        self.fade = np.ones(capacity, dtype=np.float32)

    def __len__(self) -> int:
        return self.count

    def clear(self) -> None:
        self.count = 0

    def emit(self, count: int, x: float, y: float, *, colour: tuple[int, int, int],
             speed: tuple[float, float], life: tuple[float, float],
             size: tuple[float, float], direction: float = 0.0, arc: float = TAU,
             gravity: float = 0.0, drag: float = 1.6, grow: float = 0.0,
             spin: float = 0.0, fade: float = 1.0, spread: float = 0.0) -> None:
        """Add ``count`` particles.  Ranges are (min, max) and sampled uniformly.

        ``direction``/``arc`` aim the burst: the default full circle is a puff,
        a narrow arc pointing up is a fountain.
        """
        count = int(min(count, self.capacity - self.count))
        if count <= 0:
            return
        lo, hi = self.count, self.count + count
        rng = self.rng
        theta = direction + rng.uniform(-arc / 2, arc / 2, count)
        magnitude = rng.uniform(*speed, count)
        offset = rng.uniform(0.0, spread, count) if spread else 0.0
        self.x[lo:hi] = x + np.cos(theta) * offset
        self.y[lo:hi] = y + np.sin(theta) * offset
        self.vx[lo:hi] = np.cos(theta) * magnitude
        self.vy[lo:hi] = np.sin(theta) * magnitude
        self.age[lo:hi] = 0.0
        self.life[lo:hi] = rng.uniform(*life, count)
        self.size[lo:hi] = rng.uniform(*size, count)
        self.grow[lo:hi] = grow
        self.angle[lo:hi] = rng.uniform(0.0, 360.0, count)
        self.spin[lo:hi] = rng.uniform(-spin, spin, count) if spin else 0.0
        self.gravity[lo:hi] = gravity
        self.drag[lo:hi] = drag
        self.red[lo:hi], self.green[lo:hi], self.blue[lo:hi] = colour
        self.fade[lo:hi] = fade
        self.count = hi

    def update(self, dt: float) -> None:
        n = self.count
        if n == 0:
            return
        s = slice(0, n)
        # Exponential drag, so a spark shoots out and then coasts to a stop.
        damping = np.exp(-self.drag[s] * dt, dtype=np.float32)
        self.vx[s] *= damping
        self.vy[s] *= damping
        self.vy[s] -= self.gravity[s] * dt
        self.x[s] += self.vx[s] * dt
        self.y[s] += self.vy[s] * dt
        self.angle[s] += self.spin[s] * dt
        self.age[s] += dt
        alive = self.age[s] < self.life[s]
        if not alive.all():
            self._compact(alive)

    def _compact(self, alive: np.ndarray) -> None:
        keep = np.nonzero(alive)[0]
        n = len(keep)
        for array in (self.x, self.y, self.vx, self.vy, self.age, self.life,
                      self.size, self.grow, self.angle, self.spin, self.gravity,
                      self.drag, self.red, self.green, self.blue, self.fade):
            array[:n] = array[keep]
        self.count = n

    # -- what the renderer needs ------------------------------------------
    def view(self) -> tuple[np.ndarray, ...]:
        """(x, y, radius, alpha 0..1, angle, r, g, b) for the live particles."""
        n = self.count
        s = slice(0, n)
        t = self.age[s] / np.maximum(self.life[s], 1e-6)
        alpha = np.power(np.clip(1.0 - t, 0.0, 1.0), self.fade[s])
        radius = self.size[s] * (1.0 + self.grow[s] * t)
        return (self.x[s], self.y[s], radius, alpha, self.angle[s],
                self.red[s], self.green[s], self.blue[s])
