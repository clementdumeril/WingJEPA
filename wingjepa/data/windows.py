"""Training windows for the world models (docs/model_design.md).

Per time step (20 Hz), 16 quantities:
  state (9)     : Va, alpha, beta, p, q, r, phi, theta, vertical speed
  controls (4)  : elevator, aileron, rudder, throttle
  wind (3)      : ESTIMATED wind expressed relative to the heading (along-track, cross-track, vertical)
Heading and position are not given to the model (the dynamics do not depend on them); altitude is
the integral of the vertical speed.

A window = H past steps + T future steps, entirely within the valid part of the flight.
"""
from pathlib import Path

import numpy as np

STATE = ["Va", "alpha", "beta", "p", "q", "r", "phi", "theta", "hdot"]
CONTROLS = ["de_cmd", "da_cmd", "dr_cmd", "dt_cmd"]
WIND = ["wind_along", "wind_cross", "wind_down"]
FEATURES = STATE + CONTROLS + WIND
N_STATE, N_CTRL, N_WIND = len(STATE), len(CONTROLS), len(WIND)
# known aircraft parameters (weighing, pendulum test), given to the v2 models and to the model C equations
AIRCRAFT = ["mass", "Ixx", "Iyy", "Izz", "Ixz"]


def load_set(data_dir, set_name):
    """Concatenate the shards of a set: data (n, K, C), channels, metadata (dict of (n,) arrays)."""
    files = sorted(Path(data_dir).glob(f"{set_name}_shard*.npz"))
    if not files:
        raise FileNotFoundError(f"no {set_name}_shard*.npz file in {data_dir}")
    zs = [np.load(f, allow_pickle=False) for f in files]
    ch = list(zs[0]["channels"])
    data = np.concatenate([z["data"] for z in zs])
    meta = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files if k not in ("data", "channels")}
    return data, ch, meta


def features(data, ch, dt=0.05):
    """(n, K, 16) float32 in FEATURES order, and the mask of valid steps (n, K)."""
    c = {k: data[:, :, i].astype(np.float64) for i, k in enumerate(ch)}
    psi = c["psi"]
    along = c["wind_n_est"] * np.cos(psi) + c["wind_e_est"] * np.sin(psi)
    cross = -c["wind_n_est"] * np.sin(psi) + c["wind_e_est"] * np.cos(psi)
    hdot = np.gradient(c["h"], dt, axis=1)
    cols = [c["Va"], c["alpha"], c["beta"], c["p"], c["q"], c["r"], c["phi"], c["theta"], hdot,
            c["de_cmd"], c["da_cmd"], c["dr_cmd"], c["dt_cmd"], along, cross, c["wind_d_est"]]
    X = np.stack(cols, -1).astype(np.float32)
    alive = c["alive"] > 0.5
    # the derivative of h is only reliable if both neighbours are valid
    alive[:, 1:-1] &= alive[:, :-2] & alive[:, 2:]
    return X, alive, c["h"].astype(np.float32)


def aircraft_params(meta):
    """(n, 5) float32: mass and inertias of each flight, in AIRCRAFT order."""
    return np.stack([meta[k] for k in AIRCRAFT], 1).astype(np.float32)


def true_wind_features(data, ch):
    """True wind (mean + gusts, without turbulence) in the heading frame, (n, K, 3): used for the
    "known wind" evaluation, to separate the model error from what is unpredictable."""
    c = {k: data[:, :, i].astype(np.float64) for i, k in enumerate(ch)}
    psi = c["psi"]
    along = c["wind_n"] * np.cos(psi) + c["wind_e"] * np.sin(psi)
    cross = -c["wind_n"] * np.sin(psi) + c["wind_e"] * np.cos(psi)
    return np.stack([along, cross, c["wind_d"]], -1).astype(np.float32)


def window_starts(alive, length):
    """List of (flight, start) for the windows of `length` steps that are entirely valid."""
    n, K = alive.shape
    if K < length:
        return np.zeros((0, 2), np.int64)
    # number of valid steps in each sliding window
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(alive, 1)], 1)
    full = (cs[:, length:] - cs[:, :-length]) == length
    f, s = np.nonzero(full)
    return np.stack([f, s], 1).astype(np.int64)


class Normalizer:
    """Per-quantity standardization (zero mean, unit variance), statistics computed on the training flights."""

    def __init__(self, mean, std):
        self.mean, self.std = np.asarray(mean, np.float32), np.asarray(std, np.float32)

    @classmethod
    def fit(cls, X, alive):
        v = X[alive]
        return cls(v.mean(0), np.maximum(v.std(0), 1e-6))

    def __call__(self, X):
        return (X - self.mean) / self.std

    def state_inverse(self, S):
        return S * self.std[:N_STATE] + self.mean[:N_STATE]

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist(), "features": FEATURES}

    @classmethod
    def from_dict(cls, d):
        return cls(d["mean"], d["std"])


def split_mask(meta, split, seed=0):
    """Mask of the flights in a split (train / val / test), as recorded at generation time (set 'train').
    The other sets have no split (everything is 'test'): for local trials on a small set,
    one is drawn (80/10/10) with a fixed seed."""
    s = meta["split"]
    if (s == "train").any():
        return s == split
    draw = np.random.default_rng(seed).choice(["train", "val", "test"], size=len(s), p=[0.8, 0.1, 0.1])
    return draw == split


class WindowSampler:
    """Draws batches of windows (past H + future T) directly from an in-memory array.
    P (n, 5), optional: aircraft parameters of each flight (aircraft_params)."""

    def __init__(self, Xn, alive, H, T, seed=0, P=None):
        self.X, self.H, self.T, self.P = Xn, H, T, P
        self.idx = window_starts(alive, H + T)
        self.rng = np.random.default_rng(seed)
        self.offsets = np.arange(H + T)

    def __len__(self):
        return len(self.idx)

    def gather(self, rows):
        f, s = self.idx[rows, 0], self.idx[rows, 1]
        return self.X[f[:, None], s[:, None] + self.offsets]          # (B, H+T, 16)

    def aircraft(self, rows):
        return self.P[self.idx[rows, 0]]                               # (B, 5)

    def sample_rows(self, B):
        return self.rng.integers(0, len(self.idx), B)

    def fixed_rows(self, B, seed=123):
        return np.random.default_rng(seed).choice(len(self.idx), min(B, len(self.idx)), replace=False)

    def batch(self, B):
        return self.gather(self.sample_rows(B))

    def fixed(self, B, seed=123):
        return self.gather(self.fixed_rows(B, seed))
