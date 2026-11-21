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
    rollouts of the normalized state RMSE over the first --noise-horizon-s seconds. --noise-kinds adds noise kinds
    not seen in training (wingjepa/data/noise.py), on the same rollouts and levels: bias (constant offset per
    rollout and state), colored (AR(1), coefficient --noise-rho per step) and dropout (the level is then the
    probability that a measurement is lost, sample and hold). The Gaussian test is always run (key
    noise_robustness); the other kinds go to noise_robustness_<kind>, their settings to noise_settings (both absent
    by default, so the default output is unchanged). Each kind has its own fixed seed, and the same realization
    is used for every model and level.
Options for the comparisons across datasets and the partially observed tests (all off by default):
  --std-from RUN: one common per-state std (that run's training statistics) for the normalized errors and the noise
    levels of every model, instead of each model's own: models trained on different datasets then share units.
  --realistic-noise 1: realistic sensor noise (wingjepa/data/sensors.py: white noise + offset per rollout, physical
    units) at 0, 1, 2 and 4 times its nominal level on the past states, all states measured; position and attitude
    errors in metres and degrees at 1, 1.5, 3 and 5 s, and normalized state error at 1 s (key noise_realistic).
  --obs realistic: every rollout starts from what the aircraft measures (alpha, beta unmeasured, realistic noise,
    fixed draw), for the models trained with train_a.py --obs realistic; the truth stays clean.
  --z-ablation zero|shuffle: for the physics-head models only (others skipped), the latent read by the head is set
    to zero or shuffled across rollouts (same permutation for all its steps): how much the head uses the latent.
Kinematics: heading integrated from the predicted rates, air velocity rotated to north-east-up plus the last wind
estimate held; truth from the simulator (north, east, altitude, heading).

Writes <model_dir>/eval_skyjepa_<set><tag>.json (--tag, empty by default). Usage:
  python scripts/eval_skyjepa.py --data-dir data/data_advanced --models runs/a0_adv_s0 runs/jepa_adv_s0 \
      [--noise-kinds gauss bias colored dropout --noise-rho 0.95]
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.noise import KINDS, corrupt, draw  # noqa: E402
from wingjepa.data.sensors import OBS_KINDS, REAL_BIAS, REAL_STD, draw_np, observe_np  # noqa: E402
from wingjepa.data.windows import N_STATE, STATE, aircraft_params, features, load_set, split_mask  # noqa: E402
from wingjepa.eval.io import DT, load_model, predict, starts  # noqa: E402

DEG = 180 / np.pi
SIGMAS = [0.0, 0.05, 0.1, 0.2, 0.5]
REAL_MULTS = [0.0, 1.0, 2.0, 4.0]                 # multiples of the nominal realistic sensor noise
REPORT_S = [1.0, 1.5, 3.0, 5.0]                   # horizons of the realistic-noise report


def ablate_latent(model, how):
    """Physics-head model whose head reads a zeroed or shuffled latent (instance override of model.latents)."""
    orig = model.latents

    def latents(seq, H, hold_wind=True, ac=None):
        z0, zs, U = orig(seq, H, hold_wind, ac)
        if how == "zero":
            return torch.zeros_like(z0), torch.zeros_like(zs), U
        perm = torch.randperm(len(z0), generator=torch.Generator().manual_seed(0))
        return z0[perm], zs[perm], U

    model.latents = latents
    return model


def kinematics(states, psi0, wind_ne):
    """states (N, L, 9) -> heading (N, L), velocity north-east-up (N, L, 3): heading integrated from the rates,
    air velocity rotated to north-east plus the wind wind_ne (N, 2), vertical speed as predicted."""
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
    ap.add_argument("--noise-kinds", nargs="+", default=["gauss"], choices=KINDS,
                    help="noise kinds of the robustness test (gauss is always run)")
    ap.add_argument("--noise-rho", type=float, default=0.95, help="AR(1) coefficient per step of the colored noise")
    ap.add_argument("--max-rollouts", type=int, default=2000)
    ap.add_argument("--tag", default="", help="suffix of the output file, e.g. _on_advanced to evaluate a model on "
                    "another dataset without overwriting its own eval_skyjepa_<set>.json")
    ap.add_argument("--std-from", default=None, help="run whose training std is the common unit of every model")
    ap.add_argument("--realistic-noise", type=int, default=0, help="1: realistic sensor-noise test (noise_realistic)")
    ap.add_argument("--obs", choices=OBS_KINDS, default="none", help="realistic: partially observed rollouts")
    ap.add_argument("--z-ablation", choices=["none", "zero", "shuffle"], default="none",
                    help="physics-head models: latent read by the head zeroed or shuffled across rollouts")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)
    kinds = [k for k in KINDS if k != "gauss" and k in a.noise_kinds]
    rng_kind = {k: np.random.default_rng(1 + KINDS.index(k)) for k in kinds}     # apart from rng: gauss unchanged
    settings = dict(levels="std of each state; dropout: probability of a lost measurement",
                    kinds=["gauss"] + kinds, **({"rho": a.noise_rho} if "colored" in kinds else {}))

    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, _ = features(data, ch)
    P = aircraft_params(meta)
    raw = {k: data[:, :, ch.index(k)].astype(np.float64)
           for k in ("psi", "north", "east", "h", "wind_n_est", "wind_e_est")}
    if (meta["split"] == "train").any():
        groups = {"vols_de_test": split_mask(meta, "test")}
    else:
        test = meta["is_test_aircraft"].astype(bool)
        groups = {"avions_entrainement": ~test, "avions_de_test": test}
    L, Ln = int(a.horizon_s / DT), int(a.noise_horizon_s / DT)
    models = [(d, *load_model(d)) for d in a.models]
    if a.z_ablation != "none":
        skip = [d for d, _, cfg, _ in models if cfg.get("head") != "physics"]
        if skip:
            print(f"--z-ablation: no physics head, skipped: {' '.join(skip)}")
        models = [(d, ablate_latent(m, a.z_ablation), cfg, nm) for d, m, cfg, nm in models
                  if cfg.get("head") == "physics"]
    std_common = None
    if a.std_from:
        std_common = np.asarray(json.load(open(Path(a.std_from) / "config.json"))["norm"]["std"][:N_STATE])
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
        past = seq[:, :Hm, :N_STATE]                                       # measured past, longest history
        reals = {"gauss": rng.standard_normal(past.shape).astype(np.float32)}
        reals.update({k: draw(k, rng_kind[k], past.shape, a.noise_rho) for k in kinds})
        # sensor-error realizations, own generators: the default draws above are unchanged
        real_obs = draw_np(np.random.default_rng(11), past.shape) if a.obs != "none" else None
        real_rn = draw_np(np.random.default_rng(12), past.shape) if a.realistic_noise else None
        print(f"\n== {gname}: {len(st)} rollouts of {a.horizon_s:g} s")

        def physical(pred):
            psi, vel = kinematics(pred, psi0, wind)
            pos = pos0[:, None] + np.cumsum(vel, 1) * DT
            return (np.linalg.norm(pos - pos_true, axis=-1), np.linalg.norm(vel - vel_true, axis=-1),
                    attitude_error(pred, true, psi, psi_true))

        for d, model, cfg, norm in models:
            H = cfg["H"]
            ac = P[f] if (cfg.get("aircraft_input") or cfg.get("head") == "physics") else None
            std = norm.std[:N_STATE] if std_common is None else std_common
            sq = seq[:, Hm - H:]
            past_m = past
            if real_obs is not None:              # what the aircraft measures (model's own mean in the blanks)
                past_m = observe_np(past, norm.mean[:N_STATE], real_obs)
                sq = sq.copy()
                sq[:, :H, :N_STATE] = past_m[:, Hm - H:]
            pred = predict(model, norm, sq, H, True, ac).astype(np.float64)
            e_roll = norm_err(pred, true, std)
            tf = np.stack([predict(model, norm, sq[:, k:k + H + 1], H, True, ac)[:, 0] for k in range(L)], 1)
            e_tf = norm_err(tf.astype(np.float64), true, std)
            pos_e, vel_e, att_e = physical(pred)
            straight = straightening(model, norm, sq, H, ac) if getattr(model, "latent", True) else np.full(1, np.nan)
            robust = {}
            for kind, real in reals.items():
                r = robust[kind] = {}
                for s in SIGMAS:
                    noisy = sq.copy()
                    noisy[:, :H, :N_STATE] = corrupt(past_m, kind, s, std, real)[:, Hm - H:]
                    pn = predict(model, norm, noisy[:, :H + Ln], H, True, ac).astype(np.float64)
                    e = np.sqrt(np.mean(((pn - true[:, :Ln]) / std) ** 2, (1, 2)))
                    r[f"{s:g}"] = dict(median=float(np.median(e)), q25=float(np.percentile(e, 25)),
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
                noise_robustness=robust["gauss"])
            if real_rn is not None:               # realistic sensor noise, physical units
                rr, ks = {}, [int(round(t / DT)) - 1 for t in REPORT_S if int(round(t / DT)) <= L]
                for mult in REAL_MULTS:
                    noisy = sq.copy()
                    noisy[:, :H, :N_STATE] = observe_np(past_m, norm.mean[:N_STATE], real_rn, mult,
                                                        mask=False)[:, Hm - H:]
                    pn = predict(model, norm, noisy, H, True, ac).astype(np.float64)
                    pe, _, ae = physical(pn)
                    en = np.sqrt(np.mean(((pn[:, :Ln] - true[:, :Ln]) / std) ** 2, (1, 2)))
                    rp, ra = rms(pe), rms(ae)
                    rr[f"{mult:g}"] = dict(position_m={f"{(k + 1) * DT:g}": float(rp[k]) for k in ks},
                                           attitude_deg={f"{(k + 1) * DT:g}": float(ra[k]) for k in ks},
                                           state_norm_median=float(np.median(en)))
                res["noise_realistic"] = dict(levels=rr, std=REAL_STD.tolist(), bias=REAL_BIAS.tolist())
            if a.obs != "none" or std_common is not None or a.z_ablation != "none":
                res["settings"] = dict(obs=a.obs, std_from=a.std_from, z_ablation=a.z_ablation)
            if kinds:                                                   # default run: same keys as before
                res.update(noise_settings=settings, **{f"noise_robustness_{k}": robust[k] for k in kinds})
            k1, kL = int(1 / DT) - 1, L - 1
            print(f"   {Path(d).name:34s} pos {res['position_rmse_m'][k1]:6.2f} / {res['position_rmse_m'][kL]:6.1f} m"
                  f"  att {res['attitude_rmse_deg'][k1]:5.1f} / {res['attitude_rmse_deg'][kL]:5.1f} deg"
                  f"  CR {res['compounding_ratio'][kL]:5.2f}  straight {straight.mean():5.2f}"
                  f"  noise0 {robust['gauss']['0']['median']:.3f}  (1 s / {a.horizon_s:g} s)"
                  + "".join(f"  {k}{SIGMAS[-1]:g} {robust[k][f'{SIGMAS[-1]:g}']['median']:.3f}" for k in kinds)
                  + ("".join(f"  real{m}x {res['noise_realistic']['levels'][m]['position_m'].get('1.5', np.nan):.2f}m"
                             for m in ("1", "4")) if real_rn is not None else ""))
            out = Path(d) / f"eval_skyjepa_{a.set}{a.tag}.json"
            prev = json.load(open(out)) if out.exists() else {}
            prev[gname] = res
            json.dump(prev, open(out, "w"), indent=1)


def _ned(raw, f, s0, L):
    """True north, east, altitude from the last measured step to the end, (N, L+1, 3)."""
    idx = s0[:, None] + np.arange(L + 1)
    return np.stack([raw["north"][f[:, None], idx], raw["east"][f[:, None], idx], raw["h"][f[:, None], idx]], -1)


if __name__ == "__main__":
    main()
