"""Fenêtres d'entraînement pour les world models (docs/model_design.md).

Par pas de temps (20 Hz), 16 grandeurs :
  état (9)      : Va, alpha, beta, p, q, r, phi, theta, vitesse verticale
  commandes (4) : profondeur, ailerons, direction, gaz
  vent (3)      : vent ESTIMÉ exprimé par rapport au cap (dans l'axe, travers, vertical)
Le cap et la position ne sont pas donnés au modèle (la dynamique n'en dépend pas) ; l'altitude est
l'intégrale de la vitesse verticale.

Une fenêtre = H pas de passé + T pas de futur, entièrement dans la partie valide du vol.
"""
from pathlib import Path

import numpy as np

STATE = ["Va", "alpha", "beta", "p", "q", "r", "phi", "theta", "hdot"]
CONTROLS = ["de_cmd", "da_cmd", "dr_cmd", "dt_cmd"]
WIND = ["wind_along", "wind_cross", "wind_down"]
FEATURES = STATE + CONTROLS + WIND
N_STATE, N_CTRL, N_WIND = len(STATE), len(CONTROLS), len(WIND)


def load_set(data_dir, set_name):
    """Concatène les shards d'un set : données (n, K, C), canaux, métadonnées (dict de tableaux (n,))."""
    files = sorted(Path(data_dir).glob(f"{set_name}_shard*.npz"))
    if not files:
        raise FileNotFoundError(f"aucun fichier {set_name}_shard*.npz dans {data_dir}")
    zs = [np.load(f, allow_pickle=False) for f in files]
    ch = list(zs[0]["channels"])
    data = np.concatenate([z["data"] for z in zs])
    meta = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files if k not in ("data", "channels")}
    return data, ch, meta


def features(data, ch, dt=0.05):
    """(n, K, 16) float32 dans l'ordre FEATURES, et le masque des pas valides (n, K)."""
    c = {k: data[:, :, i].astype(np.float64) for i, k in enumerate(ch)}
    psi = c["psi"]
    along = c["wind_n_est"] * np.cos(psi) + c["wind_e_est"] * np.sin(psi)
    cross = -c["wind_n_est"] * np.sin(psi) + c["wind_e_est"] * np.cos(psi)
    hdot = np.gradient(c["h"], dt, axis=1)
    cols = [c["Va"], c["alpha"], c["beta"], c["p"], c["q"], c["r"], c["phi"], c["theta"], hdot,
            c["de_cmd"], c["da_cmd"], c["dr_cmd"], c["dt_cmd"], along, cross, c["wind_d_est"]]
    X = np.stack(cols, -1).astype(np.float32)
    alive = c["alive"] > 0.5
    # la dérivée de h n'est fiable que si les deux voisins sont valides
    alive[:, 1:-1] &= alive[:, :-2] & alive[:, 2:]
    return X, alive, c["h"].astype(np.float32)


def window_starts(alive, length):
    """Liste (vol, début) des fenêtres de `length` pas entièrement valides."""
    n, K = alive.shape
    if K < length:
        return np.zeros((0, 2), np.int64)
    # nombre de pas valides dans chaque fenêtre glissante
    cs = np.concatenate([np.zeros((n, 1)), np.cumsum(alive, 1)], 1)
    full = (cs[:, length:] - cs[:, :-length]) == length
    f, s = np.nonzero(full)
    return np.stack([f, s], 1).astype(np.int64)


class Normalizer:
    """Centrage-réduction par grandeur, statistiques calculées sur les vols d'entraînement."""

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
    """Masque des vols d'un split (train / val / test), tel qu'enregistré à la génération (set 'train').
    Les autres sets n'ont pas de découpage (tout est 'test') : pour les essais locaux sur un petit set,
    on en tire un (80/10/10) avec une graine fixe."""
    s = meta["split"]
    if (s == "train").any():
        return s == split
    draw = np.random.default_rng(seed).choice(["train", "val", "test"], size=len(s), p=[0.8, 0.1, 0.1])
    return draw == split


class WindowSampler:
    """Tire des lots de fenêtres (passé H + futur T) directement depuis un tableau en mémoire."""

    def __init__(self, Xn, alive, H, T, seed=0):
        self.X, self.H, self.T = Xn, H, T
        self.idx = window_starts(alive, H + T)
        self.rng = np.random.default_rng(seed)
        self.offsets = np.arange(H + T)

    def __len__(self):
        return len(self.idx)

    def gather(self, rows):
        f, s = self.idx[rows, 0], self.idx[rows, 1]
        return self.X[f[:, None], s[:, None] + self.offsets]          # (B, H+T, 16)

    def batch(self, B):
        return self.gather(self.rng.integers(0, len(self.idx), B))

    def fixed(self, B, seed=123):
        rows = np.random.default_rng(seed).choice(len(self.idx), min(B, len(self.idx)), replace=False)
        return self.gather(rows)
