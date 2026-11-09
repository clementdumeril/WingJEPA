"""Open-loop metrics of SkyJEPA (arXiv 2606.23444, Sec. VI-A/B), computed on our fixed-wing test flights.

For each model and each aircraft group of the set:
  - curves vs rollout step k (Fig. 8 and Table III of the paper): RMS 3D position error [m], RMS inertial velocity
    error [m/s], RMS attitude error [deg] (geodesic angle ||log(R_true^T R_pred)||), normalized state RMSE
    e_k = sqrt(mean over the 9 states of ((pred - true) / std)^2);
  - compounding ratio CR_k = e_k(open-loop rollout) / e_k(teacher-forced), where the teacher-forced prediction of
    step t+k is the one-step prediction from the true window ending at t+k-1, and error growth
    ER_k = mean(e_k - e_{k-1}) (Fig. 6, Eqs. 29-30);
  - temporal straightening (Fig. 7): mean cosine similarity of consecutive latent velocities z_{k+1} - z_k of the
    predicted latent trajectory, per rollout (mean and quartiles);
  - observation-noise robustness (Fig. 9): Gaussian noise of sigma * std added to the past states only, median over
    rollouts of the normalized state RMSE over the first --noise-horizon-s seconds.
Kinematics as in scripts/position_error.py: heading integrated from the predicted rates, air velocity rotated to
north-east-up plus the last wind estimate held; truth from the simulator (north, east, altitude, heading).

Writes <model_dir>/eval_skyjepa_<set>.json. Usage:
  python scripts/eval_skyjepa.py --data-dir <data_dir> --models runs/a0_calm_sky_s0 runs/jepa_calm_sky_s0
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
from position_error import starts  # noqa: E402
from wingjepa.data.windows import N_STATE, STATE, aircraft_params, features, load_set, split_mask  # noqa: E402

DEG = 180 / np.pi
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.5]


def kinematics(states, psi0, wind_ne):
    """states (N, L, 9) -> heading (N, L), velocity north-east-up (N, L, 3), as in position_error.integrate."""
    g = {k: states[:, :, i] for i, k in enumerate(STATE)}
    sf, cf, st, ct = np.sin(g["phi"]), np.cos(g["phi"]), np.sin(g["theta"]), np.cos(g["theta"])
    psi = psi0[:, None] + np.cumsum((g["q"] * sf + g["r"] * cf) / np.maximum(ct, 0.05), 1) * DT
    sp, cp = np.sin(psi), np.cos(psi)
    ca, sa, cb, sb = np.cos(g["alpha"]), np.sin(g["alpha"]), np.cos(g["beta"]), np.sin(g["beta"])
    u, v, w = g["Va"] * ca * cb, g["Va"] * sb, g["Va"] * sa * cb
    vn = ct * cp * u + (sf * st * cp - cf * sp) * v + (cf * st * cp + sf * sp) * w + wind_ne[:, :1]
    ve = ct * sp * u + (sf * st * sp + cf * cp) * v + (cf * st * sp - sf * cp) * w + wind_ne[:, 1:]
    return psi, np.stack([vn, ve, g["hdot"]], -1)


def rotation(phi, theta, psi):
    """Body-to-NED rotation matrices (..., 3, 3) from ZYX Euler angles."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(theta), np.sin(theta), np.cos(psi), np.sin(psi)
    R = np.stack([ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp,
                  ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp,
                  -st, sf * ct, cf * ct], -1)
    return R.reshape(*phi.shape, 3, 3)


def attitude_error(pred, true, psi_pred, psi_true):
    """Geodesic angle between true and predicted attitude, degrees, (N, L)."""
    Rp = rotation(pred[:, :, STATE.index("phi")], pred[:, :, STATE.index("theta")], psi_pred)
    Rt = rotation(true[:, :, STATE.index("phi")], true[:, :, STATE.index("theta")], psi_true)
    tr = np.einsum("nkji,nkji->nk", Rt, Rp)                          # trace(Rt^T Rp)
    return np.arccos(np.clip((tr - 1) / 2, -1, 1)) * DEG


def norm_err(pred, true, std):
    """Normalized state error per rollout and step, (N, L)."""
    return np.sqrt(np.mean(((pred - true) / std) ** 2, -1))


def rms(e):
    return np.sqrt(np.mean(e ** 2, 0))


@torch.no_grad()
def straightening(model, norm, seq, H, ac, batch=512):
    """Mean cosine of consecutive latent velocities of the predicted latent trajectory, per rollout."""
    out = []
    for i in range(0, len(seq), batch):
        s = torch.from_numpy(norm(seq[i:i + batch]))
        a = None if ac is None else torch.from_numpy(ac[i:i + batch])
        z0, zs, _ = model.latents(s, H, True, a)
        z = torch.cat([z0[:, None], zs], 1)
        v = z[:, 1:] - z[:, :-1]
        c = torch.nn.functional.cosine_similarity(v[:, 1:], v[:, :-1], dim=-1)
        out.append(c.mean(1).numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--set", default="test_long")
    ap.add_argument("--horizon-s", type=float, default=5.0, help="rollout length of the curves")
    ap.add_argument("--stride-s", type=float, default=5.0)
    ap.add_argument("--noise-horizon-s", type=float, default=1.0)
    ap.add_argument("--max-rollouts", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)

    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, _ = features(data, ch)
    P = aircraft_params(meta, oracle=True)          # columns 5: only read by an oracle physics head
    raw = {k: data[:, :, ch.index(k)].astype(np.float64)
           for k in ("psi", "north", "east", "h", "wind_n_est", "wind_e_est")}
    if (meta["split"] == "train").any():
        groups = {"vols_de_test": split_mask(meta, "test")}
    else:
        test = meta["is_test_aircraft"].astype(bool)
        groups = {"avions_entrainement": ~test, "avions_de_test": test}
    L, Ln = int(a.horizon_s / DT), int(a.noise_horizon_s / DT)
    models = [(d, *load_model(d)) for d in a.models]
    Hm = max(cfg["H"] for _, _, cfg, _ in models)

    for gname, gm in groups.items():
        st = starts(alive & gm[:, None], Hm, L, int(a.stride_s / DT))
        if len(st) > a.max_rollouts:
            st = st[np.sort(rng.choice(len(st), a.max_rollouts, replace=False))]
        f, s0 = st[:, 0], st[:, 1] + Hm - 1                                 # last measured step
        seq = X[f[:, None], st[:, 1:2] + np.arange(Hm + L)]
        idx = s0[:, None] + 1 + np.arange(L)
        true = seq[:, Hm:, :N_STATE].astype(np.float64)
        pos_true = np.stack([raw["north"][f[:, None], idx], raw["east"][f[:, None], idx],
                             raw["h"][f[:, None], idx]], -1)
        vel_true = np.gradient(_ned(raw, f, s0, L), DT, axis=1)[:, 1:]          # ground velocity, (N, L, 3)
        psi_true = raw["psi"][f[:, None], idx]
        psi0 = raw["psi"][f, s0]
        pos0 = np.stack([raw["north"][f, s0], raw["east"][f, s0], raw["h"][f, s0]], -1)
        wind = np.stack([raw["wind_n_est"][f, s0], raw["wind_e_est"][f, s0]], -1)
        noise = rng.standard_normal(seq[:, :Hm, :N_STATE].shape).astype(np.float32)
        print(f"\n== {gname}: {len(st)} rollouts of {a.horizon_s:g} s")

        def physical(pred):
            psi, vel = kinematics(pred, psi0, wind)
            pos = pos0[:, None] + np.cumsum(vel, 1) * DT
            return (np.linalg.norm(pos - pos_true, axis=-1), np.linalg.norm(vel - vel_true, axis=-1),
                    attitude_error(pred, true, psi, psi_true))

        for d, model, cfg, norm in models:
            H = cfg["H"]
            ac = P[f] if (cfg.get("aircraft_input") or cfg.get("head") == "physics") else None
            std = norm.std[:N_STATE]
            sq = seq[:, Hm - H:]
            pred = predict(model, norm, sq, H, True, ac).astype(np.float64)
            e_roll = norm_err(pred, true, std)
            tf = np.stack([predict(model, norm, sq[:, k:k + H + 1], H, True, ac)[:, 0] for k in range(L)], 1)
            e_tf = norm_err(tf.astype(np.float64), true, std)
            pos_e, vel_e, att_e = physical(pred)
            straight = straightening(model, norm, sq, H, ac)
            robust = {}
            for s in SIGMAS:
                noisy = sq.copy()
                noisy[:, :H, :N_STATE] += s * norm.std[:N_STATE] * noise[:, Hm - H:]
                pn = predict(model, norm, noisy[:, :H + Ln], H, True, ac).astype(np.float64)
                e = np.sqrt(np.mean(((pn - true[:, :Ln]) / std) ** 2, (1, 2)))
                robust[f"{s:g}"] = dict(median=float(np.median(e)), q25=float(np.percentile(e, 25)),
                                        q75=float(np.percentile(e, 75)))
            res = dict(
                n_rollouts=len(st), dt=DT,
                position_rmse_m=rms(pos_e).tolist(), velocity_rmse_ms=rms(vel_e).tolist(),
                attitude_rmse_deg=rms(att_e).tolist(), state_rmse_norm=rms(e_roll).tolist(),
                state_rmse_norm_teacher_forced=rms(e_tf).tolist(),
                compounding_ratio=(rms(e_roll) / np.maximum(rms(e_tf), 1e-9)).tolist(),
                error_growth=np.r_[np.nan, np.mean(np.diff(e_roll, axis=1), 0)].tolist(),
                straightening=dict(mean=float(straight.mean()), q25=float(np.percentile(straight, 25)),
                                   q75=float(np.percentile(straight, 75))),
                noise_robustness=robust)
            k1, kL = int(1 / DT) - 1, L - 1
            print(f"   {Path(d).name:34s} pos {res['position_rmse_m'][k1]:6.2f} / {res['position_rmse_m'][kL]:6.1f} m"
                  f"  att {res['attitude_rmse_deg'][k1]:5.1f} / {res['attitude_rmse_deg'][kL]:5.1f} deg"
                  f"  CR {res['compounding_ratio'][kL]:5.2f}  straight {straight.mean():5.2f}"
                  f"  noise0 {robust['0']['median']:.3f}  (1 s / {a.horizon_s:g} s)")
            out = Path(d) / f"eval_skyjepa_{a.set}.json"
            prev = json.load(open(out)) if out.exists() else {}
            prev[gname] = res
            json.dump(prev, open(out, "w"), indent=1)


def _ned(raw, f, s0, L):
    """True north, east, altitude from the last measured step to the end, (N, L+1, 3)."""
    idx = s0[:, None] + np.arange(L + 1)
    return np.stack([raw["north"][f[:, None], idx], raw["east"][f[:, None], idx], raw["h"][f[:, None], idx]], -1)


if __name__ == "__main__":
    main()
