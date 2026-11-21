"""Realistic sensor suite of a small fixed-wing aircraft, for the partially observed tests (train_a.py --obs realistic,
eval_skyjepa.py --obs realistic) and the realistic-noise robustness test (eval_skyjepa.py --realistic-noise 1).

What a small autopilot measures, per 50 ms step, for the 9 state quantities (Va, alpha, beta, p, q, r, phi, theta,
vertical speed):
  - alpha and beta are NOT measured (no vanes on most small aircraft): their slot holds the training mean
    (0 in normalized units), so the model has to infer them;
  - the others carry white noise of REAL_STD plus a constant offset per window of REAL_BIAS (both physical units):
    pitot airspeed 0.5 m/s, gyro rates 1 deg/s, attitude estimate 1 deg, barometric vertical speed 0.3 m/s;
    offsets of half these values (calibration, estimator bias).
Only the past (the measured context) is observed this way: the training targets and the evaluation truth stay the
clean simulator states.
"""
import numpy as np
import torch

DEG = np.pi / 180
REAL_STD = np.array([0.5, 1 * DEG, 1 * DEG, 1 * DEG, 1 * DEG, 1 * DEG, 1 * DEG, 1 * DEG, 0.3])
REAL_BIAS = 0.5 * REAL_STD
UNMEASURED = [1, 2]                                   # alpha, beta
OBS_KINDS = ("none", "realistic")


def draw_np(rng, shape):
    """Unit realization (white part (N, L, 9), offset part (N, 1, 9)) of the sensor errors."""
    return rng.standard_normal(shape), rng.standard_normal((shape[0], 1, shape[2]))


def observe_np(past, mean, real, scale=1.0, mask=True):
    """Measured past (physical units) from the true past states (N, L, 9): scale x (white noise + offset), and the
    unmeasured quantities replaced by `mean` (the model's training mean of each state) if mask."""
    white, off = real
    out = past + scale * (REAL_STD * white + REAL_BIAS * off)
    if mask:
        out[..., UNMEASURED] = np.asarray(mean)[UNMEASURED]
    return out


def observe_torch(past_n, std, mask=True):
    """Same, in normalized units, on a training batch (B, L, 9) (in place on a copy): the noise is divided by the
    per-state std of the training statistics, the unmeasured quantities are set to 0 (= the training mean)."""
    s = torch.as_tensor(REAL_STD, dtype=past_n.dtype, device=past_n.device) / std
    b = torch.as_tensor(REAL_BIAS, dtype=past_n.dtype, device=past_n.device) / std
    out = past_n + s * torch.randn_like(past_n) + b * torch.randn_like(past_n[:, :1])
    if mask:
        out[..., UNMEASURED] = 0.0
    return out
