"""Does a world model respond to its commands like the aircraft? The quantity MPPI relies on.

For test windows of a calm-air set (data_advanced), each command channel (elevator, aileron, rudder, throttle) is
offset by +-delta over the whole future (the MPPI horizon, 1 s by default), and the response
  dS/du = (S(u + delta) - S(u - delta)) / (2 delta)
of the predicted states is compared with the TRUE response, obtained by re-simulating the same aircraft from the
same recorded state (scripts/generate_flights.py metadata: mass, inertias, motor, CG, servo time constant,
aerodynamic factors; steady wind, no gusts) with the same offset commands. Differences of two simulations cancel
the replay error to first order; the replay error itself is printed (unperturbed re-simulation vs recording).

Per model, channel and state of interest, at 0.25, 0.5 and 1 s: median model and true response, correlation across
windows between model and true response, fraction of windows where they have the same sign. Units: deg, deg/s,
m/s per 1 deg of surface or per 0.1 of throttle. Writes <model_dir>/command_response<tag>.json.

Usage: python scripts/command_response.py --data-dir data/data_advanced --config configs/data_advanced.yaml \
           --models runs/mlp_adv_s0 runs/jepa_adv_s0 runs/jepapi_adv_s0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.windows import N_STATE, STATE, aircraft_params, features, load_set  # noqa: E402
from wingjepa.eval.io import DT, load_model, predict, starts  # noqa: E402
from wingjepa.sim import common as sc  # noqa: E402
from wingjepa.sim.advanced_plane import AdvancedPlane  # noqa: E402

DEG = 180 / np.pi
# channel -> (index among the commands, offset, display unit of the command, states of interest)
CHANNELS = {"elevator": (0, np.deg2rad(2.0), np.deg2rad(1.0), ["q", "theta", "alpha", "hdot"]),
            "aileron": (1, np.deg2rad(2.0), np.deg2rad(1.0), ["p", "phi"]),
            "rudder": (2, np.deg2rad(2.0), np.deg2rad(1.0), ["r", "beta"]),
            "throttle": (3, 0.05, 0.1, ["Va", "hdot"])}
TIMES = [0.25, 0.5, 1.0]
SHOW = lambda k: 1.0 if k in ("Va", "hdot") else DEG              # display factor of each state


def aircraft(p, meta, f):
    """AdvancedPlane batch rebuilt from the metadata of flights f."""
    aero = {k[len("aero_scale_"):]: meta[k][f] for k in meta if k.startswith("aero_scale_")}
    ov = dict(mass=meta["mass"][f], Ixx=meta["Ixx"][f], Iyy=meta["Iyy"][f], Izz=meta["Izz"][f], Ixz=meta["Ixz"][f],
              motor_k=meta["power_scale"][f], cg_shift=meta["cg_shift"][f], aero_scale=aero)
    if "servo_tau" in meta:
        ov["servo_tau"] = meta["servo_tau"][f]
    return AdvancedPlane(p, len(f), None, ov)


def simulate(ac, Y0, cmds, wind, dt_sim):
    """Re-simulation over the L steps of cmds (N, L, 4) -> the 9 model quantities after each 50 ms step (N, L, 9)."""
    Y, out, nsub = Y0.copy(), [], int(round(DT / dt_sim))
    zero = np.zeros((len(Y), 3))
    for k in range(cmds.shape[1]):
        h0 = Y[:, sc.H].copy()
        for _ in range(nsub):
            Y = ac.step(Y, cmds[:, k, :3], cmds[:, k, 3], zero, wind, dt_sim)
        u, v, w = Y[:, sc.U], Y[:, sc.V], Y[:, sc.W]
        Va = np.sqrt(u * u + v * v + w * w)
        out.append(np.stack([Va, np.arctan2(w, u), np.arcsin(np.clip(v / Va, -1, 1)), Y[:, sc.P], Y[:, sc.Q],
                             Y[:, sc.R], Y[:, sc.PHI], Y[:, sc.TH], (Y[:, sc.H] - h0) / DT], 1))
    return np.stack(out, 1)


def stats(m, t):
    """Median model / true response, correlation across windows, same-sign fraction."""
    ok = np.isfinite(m) & np.isfinite(t)
    m, t = m[ok], t[ok]
    corr = float(np.corrcoef(m, t)[0, 1]) if len(m) > 2 and m.std() > 0 and t.std() > 0 else float("nan")
    return dict(model=float(np.median(m)), true=float(np.median(t)), corr=corr,
                same_sign=float(np.mean(np.sign(m) == np.sign(t))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--config", default="configs/data_advanced.yaml", help="data config (aircraft, sim dt)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--set", default="test_long")
    ap.add_argument("--horizon-s", type=float, default=1.0)
    ap.add_argument("--max-rollouts", type=int, default=500, help="per aircraft group")
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = sc.load_config(a.config)
    p = sc.load_config(cfg["aircraft"]["config"])
    dt_sim = cfg["sim"]["dt"]
    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, _ = features(data, ch)
    P = aircraft_params(meta)
    col = {k: ch.index(k) for k in ch}
    L = int(round(a.horizon_s / DT))
    models = [(d, *load_model(d)) for d in a.models]
    Hm = max(c["H"] for _, _, c, _ in models)
    rng = np.random.default_rng(0)
    test = meta["is_test_aircraft"].astype(bool)
    results = {d: {} for d, *_ in models}
    for gname, gm in (("avions_entrainement", ~test), ("avions_de_test", test)):
        st = starts(alive & gm[:, None], Hm, L, int(5 / DT))
        if len(st) > a.max_rollouts:
            st = st[np.sort(rng.choice(len(st), a.max_rollouts, replace=False))]
        f, s0 = st[:, 0], st[:, 1] + Hm - 1                                  # last measured step
        seq = X[f[:, None], st[:, 1:2] + np.arange(Hm + L)].astype(np.float32)
        Y0 = np.stack([data[f, s0, col[k]] for k in ("u", "v", "w", "p", "q", "r", "phi", "theta", "psi",
                                                     "north", "east", "h", "de", "da", "dr")], 1).astype(np.float64)
        Y0 = np.c_[Y0, data[f, s0, col["omega_rpm"]] * 2 * np.pi / 60]
        wind = np.stack([meta[f"wind_steady_{k}"][f] for k in "ned"], 1)
        cmds = np.stack([data[f[:, None], s0[:, None] + np.arange(L), col[k]]
                         for k in ("de_cmd", "da_cmd", "dr_cmd", "dt_cmd")], -1).astype(np.float64)
        ac = aircraft(p, meta, f)
        replay = simulate(ac, Y0, cmds, wind, dt_sim)
        rec = seq[:, Hm:, :N_STATE]
        err = np.median(np.abs(replay - rec), 0)                           # (L, 9)
        print(f"\n== {gname}: {len(st)} windows of {a.horizon_s:g} s; replay error at {a.horizon_s:g} s (median): "
              + ", ".join(f"{k} {err[-1, i] * SHOW(k):.2f}" for i, k in enumerate(STATE)))
        truth = {}
        for name, (c, dlt, _, _) in CHANNELS.items():
            up, dn = cmds.copy(), cmds.copy()
            up[:, :, c] += dlt
            dn[:, :, c] -= dlt
            truth[name] = (simulate(ac, Y0, up, wind, dt_sim) - simulate(ac, Y0, dn, wind, dt_sim)) / (2 * dlt)
        for d, model, mc, norm in models:
            H = mc["H"]
            acp = P[f] if (mc.get("aircraft_input") or mc.get("head") == "physics") else None
            sq = seq[:, Hm - H:]
            res, line = {}, []
            for name, (c, dlt, unit, keys) in CHANNELS.items():
                up, dn = sq.copy(), sq.copy()
                up[:, H - 1:H + L - 1, N_STATE + c] += dlt                   # commands applied over the horizon
                dn[:, H - 1:H + L - 1, N_STATE + c] -= dlt
                resp = (predict(model, norm, up, H, True, acp) - predict(model, norm, dn, H, True, acp)) / (2 * dlt)
                res[name] = {}
                for key in keys:
                    i = STATE.index(key)
                    res[name][key] = {f"{t:g}": stats(resp[:, int(round(t / DT)) - 1, i] * unit * SHOW(key),
                                                      truth[name][:, int(round(t / DT)) - 1, i] * unit * SHOW(key))
                                      for t in TIMES}
                key = keys[0]
                s = res[name][key]["0.5"]
                line.append(f"{name[:4]}->{key} {s['model']:+7.2f}/{s['true']:+7.2f} r {s['corr']:+.2f}")
            results[d][gname] = res
            print(f"   {Path(d).name:24s} " + "  ".join(line) + "   (model/true at 0.5 s, corr)")
    for d, r in results.items():
        json.dump(r, open(Path(d) / f"command_response{a.tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
