"""Sensor noise models on the measured states, for robustness tests (evaluation and closed loop).

Kinds, each with one level s (for the additive kinds: in units of the per-state std of the training
statistics, so the noise on state j has std s * std_j):
  gauss   : i.i.d. Gaussian, independent per step and per state (the noise of train_a.py --input-noise).
  bias    : constant offset b ~ N(0, (s std)^2) per flight (or window) and per state, held over all steps.
  colored : AR(1) per state, e_k = rho e_{k-1} + sqrt(1 - rho^2) w_k, w_k ~ N(0, 1), stationary start
            e_0 ~ N(0, 1), so the marginal std is s * std at every step; rho per 50 ms step (default 0.95,
            correlation time about 1 s).
  dropout : s is the probability p in [0, 1] that a measurement is lost, independently per step and per
            state; a lost value keeps the last received one (sample and hold), no additive noise. The first
            measurement (of a flight or of a window) is always received.

Two uses, same models:
  - whole windows (numpy, evaluation): real = draw(kind, rng, (N, L, d), rho) once, then
    corrupt(x, kind, s, std, real) for every level s; the realization does not depend on s (nested dropout
    masks), so the levels and the models see the same noise;
  - step by step (closed loop): SensorNoise(kind, s, std, rng, rho), called once per measurement on the
    (n, d) states of the n flights; it keeps the state of each flight (bias, AR(1) value, held values).
"""
import numpy as np

KINDS = ("gauss", "bias", "colored", "dropout")
ADDITIVE = ("gauss", "bias", "colored")


def _check(kind, level=0.0, rho=0.0):
    if kind not in KINDS:
        raise ValueError(f"unknown noise kind {kind!r}, expected one of {KINDS}")
    if kind == "dropout" and not 0.0 <= level <= 1.0:
        raise ValueError(f"dropout level is a probability, got {level}")
    if kind == "colored" and not 0.0 <= rho < 1.0:
        raise ValueError(f"AR(1) coefficient must be in [0, 1), got {rho}")


def draw(kind, rng, shape, rho=0.95):
    """Level-independent realization over windows, shape (N, L, d) with time on axis 1: unit-std noise for the
    additive kinds, uniforms in [0, 1) for dropout (a step is lost where real < p)."""
    _check(kind, rho=rho)
    if kind == "gauss":
        return rng.standard_normal(shape)
    if kind == "bias":
        return np.broadcast_to(rng.standard_normal((shape[0], 1, *shape[2:])), shape)
    if kind == "colored":
        w = rng.standard_normal(shape)
        e = np.empty(shape)
        e[:, 0] = w[:, 0]
        c = np.sqrt(1 - rho ** 2)
        for k in range(1, shape[1]):
            e[:, k] = rho * e[:, k - 1] + c * w[:, k]
        return e
    return rng.random(shape)


def hold(x, lost):
    """Sample and hold along axis 1: lost steps take the last received value (step 0 always received)."""
    lost = lost.copy()
    lost[:, 0] = False
    k = np.where(lost, 0, np.arange(x.shape[1]).reshape(1, -1, *([1] * (x.ndim - 2))))
    return np.take_along_axis(x, np.maximum.accumulate(k, axis=1), axis=1)


def corrupt(x, kind, level, std, real):
    """Measured windows from the true ones x (N, L, d), for the realization real = draw(kind, ...)."""
    _check(kind, level)
    if kind in ADDITIVE:
        return x + level * std * real
    return hold(x, real < level) if level > 0 else x.copy()


class SensorNoise:
    """Step-by-step sensor noise for a batch of flights: meas = noise(s) for the true states s (n, d) of each
    50 ms measurement, in time order. The random draws do not depend on the states, so a given rng seed gives
    the same realization (offsets, AR(1) series, lost steps) whatever the controller."""

    def __init__(self, kind, level, std, rng, rho=0.95):
        _check(kind, level, rho)
        self.kind, self.level, self.std, self.rng, self.rho = kind, level, std, rng, rho
        self.e = None                                    # bias or AR(1) value (n, d), held values for dropout

    def __call__(self, s):
        kind, rng = self.kind, self.rng
        if kind == "gauss":
            return s + self.level * self.std * rng.standard_normal(s.shape)
        if kind == "dropout":
            if self.e is None:
                self.e = s.copy()
            else:
                self.e = np.where(rng.random(s.shape) < self.level, self.e, s)
            return self.e.copy()
        if self.e is None:
            self.e = rng.standard_normal(s.shape)
        elif kind == "colored":
            self.e = self.rho * self.e + np.sqrt(1 - self.rho ** 2) * rng.standard_normal(s.shape)
        return s + self.level * self.std * self.e

    def describe(self):
        """Settings for the output files."""
        d = dict(kind=self.kind, level=self.level)
        if self.kind == "colored":
            d["rho"] = self.rho
        if self.kind == "dropout":
            d["p_drop"] = self.level
        return d
