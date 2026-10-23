"""SkyJEPA-style metric: 3D position error of open-loop predictions, in meters.

The models predict airspeed, angles, rates and vertical speed, not the position. The position is
rebuilt from the prediction: the heading is integrated from the predicted rates
(psi_dot = (q sin phi + r cos phi) / cos theta), the air velocity is rotated to north-east and the last
wind estimate (held, as the model sees it) is added, then everything is integrated over time; altitude
is the integral of the predicted vertical speed. Truth: the simulator's north, east and altitude.
The "constant state" baseline goes through the same integration with the last measured state held.

Writes <model_dir>/eval_position_<set>.json and prints the RMS position error at 1, 5, 10 and 20 s.
Usage: python scripts/position_error.py --data-dir <data_dir> --models runs/a0_v2T100_s0 runs/jepa_v2T100_s0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import DT, load_model, predict  # noqa: E402
from wingjepa.data.windows import N_STATE, STATE, aircraft_params, features, load_set, split_mask  # noqa: E402

TIMES = [1, 5, 10, 20]


def starts(alive, n_past, n_future, stride):
    out = []
    for f in range(alive.shape[0]):
        for s in range(0, alive.shape[1] - n_past - n_future + 1, stride):
            if alive[f, s:s + n_past + n_future].all():
                out.append((f, s))
    return np.array(out)


def integrate(states, psi0, pos0, wind_ne):
    """states (N, L, 9) -> positions (N, L, 3): north, east, altitude."""
    g = {k: states[:, :, i] for i, k in enumerate(STATE)}
    sf, cf, st, ct = np.sin(g["phi"]), np.cos(g["phi"]), np.sin(g["theta"]), np.cos(g["theta"])
    psi = psi0[:, None] + np.cumsum((g["q"] * sf + g["r"] * cf) / np.maximum(ct, 0.05), 1) * DT
    sp, cp = np.sin(psi), np.cos(psi)
    ca, sa, cb, sb = np.cos(g["alpha"]), np.sin(g["alpha"]), np.cos(g["beta"]), np.sin(g["beta"])
    u, v, w = g["Va"] * ca * cb, g["Va"] * sb, g["Va"] * sa * cb
    vn = ct * cp * u + (sf * st * cp - cf * sp) * v + (cf * st * cp + sf * sp) * w + wind_ne[:, :1]
    ve = ct * sp * u + (sf * st * sp + cf * cp) * v + (cf * st * sp - sf * cp) * w + wind_ne[:, 1:]
    n = pos0[:, :1] + np.cumsum(vn, 1) * DT
    e = pos0[:, 1:2] + np.cumsum(ve, 1) * DT
    h = pos0[:, 2:3] + np.cumsum(g["hdot"], 1) * DT
    return np.stack([n, e, h], -1)


def summarize(err):
    """err (N, L) 3D position error -> RMS and median at TIMES."""
    L = err.shape[1]
    return {f"{t}s": dict(rms=float(np.sqrt(np.mean(err[:, int(t / DT) - 1] ** 2))),
                          median=float(np.median(err[:, int(t / DT) - 1])))
            for t in TIMES if int(t / DT) <= L}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--set", default="test_long")
    ap.add_argument("--horizon-s", type=float, default=20.0)
    ap.add_argument("--stride-s", type=float, default=10.0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, _ = features(data, ch)
    P = aircraft_params(meta)
    raw = {k: data[:, :, ch.index(k)].astype(np.float64) for k in ("psi", "north", "east", "h", "wind_n_est", "wind_e_est")}
    if (meta["split"] == "train").any():
        groups = {"vols_de_test": split_mask(meta, "test")}
    else:
        test = meta["is_test_aircraft"].astype(bool)
        groups = {"avions_entrainement": ~test, "avions_de_test": test}
    L = int(a.horizon_s / DT)
    models = [(d, *load_model(d)) for d in a.models]
    Hm = max(cfg["H"] for _, _, cfg, _ in models)

    for gname, gm in groups.items():
        st = starts(alive & gm[:, None], Hm, L, int(a.stride_s / DT))
        f, s0 = st[:, 0], st[:, 1] + Hm - 1                                 # last measured step
        seq = X[f[:, None], st[:, 1:2] + np.arange(Hm + L)]
        idx = s0[:, None] + 1 + np.arange(L)
        true = np.stack([raw["north"][f[:, None], idx], raw["east"][f[:, None], idx], raw["h"][f[:, None], idx]], -1)
        psi0 = raw["psi"][f, s0]
        pos0 = np.stack([raw["north"][f, s0], raw["east"][f, s0], raw["h"][f, s0]], -1)
        wind = np.stack([raw["wind_n_est"][f, s0], raw["wind_e_est"][f, s0]], -1)
        ref = integrate(np.repeat(seq[:, Hm - 1:Hm, :N_STATE], L, 1).astype(np.float64), psi0, pos0, wind)
        ref_err = summarize(np.linalg.norm(ref - true, axis=-1))
        print(f"\n== {gname}: {len(st)} predictions of {a.horizon_s:.0f} s, RMS 3D position error [m]")
        print("   " + "model".ljust(34) + "".join(f"{t:>8d} s" for t in TIMES))
        print("   " + "baseline: constant state".ljust(34) + "".join(f"{v['rms']:10.1f}" for v in ref_err.values()))
        for d, model, cfg, norm in models:
            H = cfg["H"]
            uses_ac = bool(cfg.get("aircraft_input")) or cfg.get("head") == "physics"
            pred = predict(model, norm, seq[:, Hm - H:], H, True, P[f] if uses_ac else None).astype(np.float64)
            res = summarize(np.linalg.norm(integrate(pred, psi0, pos0, wind) - true, axis=-1))
            print("   " + Path(d).name.ljust(34) + "".join(f"{v['rms']:10.1f}" for v in res.values()))
            out = Path(d) / f"eval_position_{a.set}.json"
            prev = json.load(open(out)) if out.exists() else {}
            prev[gname] = dict(n_predictions=len(st), horizon_s=a.horizon_s, model=res, reference=ref_err)
            json.dump(prev, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
