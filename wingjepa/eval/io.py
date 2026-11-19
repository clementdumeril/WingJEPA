"""Loading a trained run and running its open-loop predictions (scripts/eval_skyjepa.py, scripts/closed_loop.py)."""
import json
from pathlib import Path

import numpy as np
import torch

from wingjepa.data.windows import Normalizer
from wingjepa.models.jepa import model_from_config

DT = 0.05


def load_model(d):
    """Run directory (config.json, model.pt) -> model in eval mode, its config, its normalizer."""
    cfg = json.load(open(Path(d) / "config.json"))
    m = model_from_config(cfg)
    m.load_state_dict(torch.load(Path(d) / "model.pt"))
    m.eval()
    return m, cfg, Normalizer.from_dict(cfg["norm"])


@torch.no_grad()
def predict(model, norm, seq, H, hold_wind, ac=None, batch=512):
    """seq : (N, H+L, 16) unnormalized, ac : (N, 5) aircraft -> predicted states (N, L, 9) unnormalized."""
    out = []
    for i in range(0, len(seq), batch):
        s = torch.from_numpy(norm(seq[i:i + batch]))
        a = None if ac is None else torch.from_numpy(ac[i:i + batch])
        out.append(model.predict_states(s, H, hold_wind, a)[0].numpy())
    return norm.state_inverse(np.concatenate(out))


def starts(alive, n_past, n_future, stride):
    """(flight, start) of the rollouts: every `stride` steps, windows of n_past + n_future valid steps."""
    out = []
    for f in range(alive.shape[0]):
        for s in range(0, alive.shape[1] - n_past - n_future + 1, stride):
            if alive[f, s:s + n_past + n_future].all():
                out.append((f, s))
    return np.array(out)
