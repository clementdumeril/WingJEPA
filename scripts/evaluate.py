"""Open-loop evaluation of a world model: starting from 0.5 s of real past, the model predicts
on its own for L seconds, receiving only the controls actually applied and the wind.

Four predictions are compared to the truth:
  - modele_vent_estime : last estimated wind held constant (realistic case, as in use);
  - modele_vent_vrai   : true wind (mean + gusts) at each step, turbulence remaining unknown;
                         separates the model error from what is unpredictable;
  - reference          : "nothing changes" (constant state, constant vertical speed);
  - reference_altitude_constante : constant state and zero vertical speed (project log, section 9: flights
                         stay close to their altitude setpoint, this reference is hard to beat at 60 s).
Results per aircraft group and per turbulence level.
Model C (physics head): in addition, "attribution": the correction learned at the first step is compared to
what the prior is really missing (true simulator forces and moments - prior, at the same state); R2 per
component (1 = the correction learned exactly the missing physics, 0 = nothing, < 0 = worse than nothing).
Constrained head (--correction param): also "identification", does the per-flight coefficient change follow
the true factor of each aircraft (aero factors on CD ... Cn, motor power)? Spearman correlation per group.

Useful horizon = first instant where |altitude error| > 5 m, |speed error| > 1 m/s or
|phi or theta error| > 10 deg.

Usage: python scripts/evaluate.py --model-dir runs/jepa_s0 --data-dir <out_dir>/data_v1 --set test_long
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.windows import (N_CTRL, N_STATE, STATE, Normalizer, aircraft_params, features,  # noqa: E402
                                   load_set, split_mask, true_wind_features, window_starts)
from wingjepa.models.jepa import model_from_config  # noqa: E402
from wingjepa.models.physics import COEFS, TERMS  # noqa: E402

DT = 0.05
DEG = 180 / np.pi
ANGLES = {"alpha", "beta", "p", "q", "r", "phi", "theta"}
THRESH = dict(h=5.0, Va=1.0, att=10.0)
TURB = ["none", "light", "moderate", "severe"]
WHO = ["modele_vent_estime", "modele_vent_vrai", "reference", "reference_altitude_constante"]
BODY = ["CX", "CY", "CZ", "Cl", "Cm", "Cn"]


def load_model(d):
    cfg = json.load(open(Path(d) / "config.json"))
    m = model_from_config(cfg)
    m.load_state_dict(torch.load(Path(d) / "model.pt"))
    m.eval()
    return m, cfg, Normalizer.from_dict(cfg["norm"])


def make_rollouts(X, alive, h, H, L, stride):
    """Starts (flight, s) such that H + L steps are valid: sequences (N, H+L, 16), altitudes, flight index."""
    seqs, hs, fl = [], [], []
    for f in range(X.shape[0]):
        for s in range(0, X.shape[1] - H - L + 1, stride):
            if alive[f, s:s + H + L].all():
                seqs.append(X[f, s:s + H + L])
                hs.append(h[f, s + H - 1:s + H + L])
                fl.append(f)
    if not seqs:
        return None, None, None
    return np.stack(seqs), np.stack(hs), np.array(fl)


@torch.no_grad()
def predict(model, norm, seq, H, hold_wind, ac=None, batch=512):
    """seq : (N, H+L, 16) unnormalized, ac : (N, 5) aircraft -> predicted states (N, L, 9) unnormalized."""
    out = []
    for i in range(0, len(seq), batch):
        s = torch.from_numpy(norm(seq[i:i + batch]))
        a = None if ac is None else torch.from_numpy(ac[i:i + batch])
        out.append(model.predict_states(s, H, hold_wind, a)[0].numpy())
    return norm.state_inverse(np.concatenate(out))


@torch.no_grad()
def attribution(model, norm, X, alive, data, ch, P, H, n=20000, seed=0):
    """Correction learned at the first step vs physics really missing from the prior (R2 per component).
    Constrained head: the correction is the change of the body-axis coefficients due to the per-flight
    coefficients, at the same state."""
    head = model.decoder
    st = window_starts(alive, H + 1)
    st = st[np.random.default_rng(seed).choice(len(st), min(n, len(st)), replace=False)]
    f, s = st[:, 0], st[:, 1] + H - 1                                   # last observed step
    seq = torch.from_numpy(norm(X[f[:, None], st[:, 1:2] + np.arange(H + 1)]))
    ac = torch.from_numpy(P[f])
    z0 = model.encoder(seq[:, :H, :N_STATE], model.ctx(ac))
    U = model.future_inputs(seq, H, True)
    corr, _ = head.correction(z0, seq[:, H - 1, :N_STATE], U[:, 0])
    c = {k: torch.from_numpy(data[f, s, ch.index(k)].astype(np.float32)) for k in
         ["Va", "alpha", "beta", "p", "q", "r", "de_cmd", "da_cmd", "dr_cmd", "dt_cmd", "thrust",
          "Fx_aero", "Fy_aero", "Fz_aero", "Mx_aero", "My_aero", "Mz_aero", "Mx_prop", "My_prop", "Mz_prop"]}
    Va = c["Va"].clamp_min(3.0)
    qS = 0.5 * head.rho * Va * Va * head.S
    true = torch.stack([(c["Fx_aero"] + c["thrust"]) / qS, c["Fy_aero"] / qS, c["Fz_aero"] / qS,
                        (c["Mx_aero"] + c["Mx_prop"]) / (qS * head.b), (c["My_aero"] + c["My_prop"]) / (qS * head.c),
                        (c["Mz_aero"] + c["Mz_prop"]) / (qS * head.b)], -1)
    args = (c["alpha"], c["beta"], c["de_cmd"], c["da_cmd"], c["dr_cmd"], c["p"], c["q"], c["r"], c["dt_cmd"], Va)
    prior = head.prior_body(*args)
    miss = (true - prior).numpy()
    dC = (head.prior_body(*args, dcoef=corr) - prior if head.mode == "param" else corr).numpy()
    out = {}
    for i, k in enumerate(BODY):
        out[k] = dict(r2=float(1 - np.mean((dC[:, i] - miss[:, i]) ** 2) / max(miss[:, i].var(), 1e-12)),
                      rms_manquant=float(np.sqrt(np.mean(miss[:, i] ** 2))),
                      rms_correction=float(np.sqrt(np.mean(dC[:, i] ** 2))))
    return out


@torch.no_grad()
def identification(model, norm, X, alive, P, meta, H, n=20000, seed=0):
    """Constrained head: per-flight multiplier of each coefficient group learned from z_0 (least-squares
    fit dc = m * c over the group's terms) vs the true factor of the aircraft minus 1. Spearman correlation
    over windows (several per flight, so no p-value), and the spread of both."""
    from scipy.stats import spearmanr
    head = model.decoder
    st = window_starts(alive, H)
    st = st[np.random.default_rng(seed).choice(len(st), min(n, len(st)), replace=False)]
    f = st[:, 0]
    seq = torch.from_numpy(norm(X[f[:, None], st[:, 1:2] + np.arange(H)]))
    z0 = model.encoder(seq[:, :, :N_STATE], model.ctx(torch.from_numpy(P[f])))
    dc = (head.correction(z0)[0] * head.coef_scale).numpy()
    c = head.coefficients().numpy()
    out, i = {}, 0
    for k in COEFS:
        j = slice(i, i + len(TERMS[k]))
        m = dc[:, j] @ c[j] / max(float(c[j] @ c[j]), 1e-12)
        true = (meta["power_scale"] if k == "T" else meta[f"aero_scale_{k}"])[f] - 1
        out[k] = dict(spearman=float(spearmanr(m, true)[0]), std_learned=float(m.std()),
                      std_true=float(true.std()))
        i += len(TERMS[k])
    return out


def errors(pred, seq, hs, H):
    """Errors (N, L) per quantity, angles in degrees; altitude = integral of the vertical speed."""
    true = seq[:, H:, :N_STATE]
    e = {k: (pred[:, :, i] - true[:, :, i]) * (DEG if k in ANGLES else 1.0) for i, k in enumerate(STATE)}
    e["h"] = hs[:, :1] + np.cumsum(pred[:, :, STATE.index("hdot")], 1) * DT - hs[:, 1:]
    return e


def useful_horizon(e):
    bad = ((np.abs(e["h"]) > THRESH["h"]) | (np.abs(e["Va"]) > THRESH["Va"])
           | (np.abs(e["phi"]) > THRESH["att"]) | (np.abs(e["theta"]) > THRESH["att"]))
    first = np.where(bad.any(1), bad.argmax(1), bad.shape[1])
    return (first + 1) * DT, ~bad.any(1)


def summarize(e, m, horizons):
    L = e["h"].shape[1]
    out = {k: {f"{t:g}s": float(np.sqrt(np.mean(e[k][m, int(t / DT) - 1] ** 2)))
               for t in horizons if int(t / DT) <= L}
           for k in ["h", "Va", "alpha", "beta", "phi", "theta", "p", "q", "r", "hdot"]}
    hz, cens = useful_horizon({k: v[m] for k, v in e.items()})
    out["horizon_utile_s"] = dict(mediane=float(np.median(hz)), q25=float(np.percentile(hz, 25)),
                                  q75=float(np.percentile(hz, 75)), jamais_depasse=float(cens.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--horizon-s", type=float, default=None, help="default: 25 s (30 s flights), 60 s (long flights)")
    ap.add_argument("--stride-s", type=float, default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    model, cfg, norm = load_model(a.model_dir)
    H = cfg["H"]
    uses_ac = bool(cfg.get("aircraft_input")) or cfg.get("head") == "physics"

    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, h = features(data, ch)
    P = aircraft_params(meta)
    X_true = X.copy()
    X_true[:, :, N_STATE + N_CTRL:] = true_wind_features(data, ch)
    long_set = data.shape[1] * DT > 60
    L = int((a.horizon_s or (60.0 if long_set else 25.0)) / DT)
    stride = int((a.stride_s or (10.0 if long_set else 2.5)) / DT)
    horizons = [1, 5, 20, 60]

    groups = {}
    if (meta["split"] == "train").any():
        groups["vols_de_test"] = split_mask(meta, "test", cfg["seed"])
    elif meta["is_test_aircraft"].any():
        groups["avions_entrainement"] = ~meta["is_test_aircraft"].astype(bool)
        groups["avions_de_test"] = meta["is_test_aircraft"].astype(bool)
    else:
        groups["tous"] = np.ones(len(X), bool)

    results = dict(model_dir=a.model_dir, set=a.set, horizon_s=L * DT, seuils=THRESH)
    for gname, gm in groups.items():
        idx = np.flatnonzero(gm)
        seq, hs, fl = make_rollouts(X[idx], alive[idx], h[idx], H, L, stride)
        if seq is None:
            print(f"[{gname}] no complete sequence of {L * DT:.0f} s")
            continue
        seq_true, _, _ = make_rollouts(X_true[idx], alive[idx], h[idx], H, L, stride)
        turb = meta["turb_level"][idx][fl]
        ac = P[idx][fl] if uses_ac else None
        const = np.repeat(seq[:, H - 1:H, :N_STATE], L, 1)
        level = const.copy()
        level[:, :, STATE.index("hdot")] = 0.0
        err = {"modele_vent_estime": errors(predict(model, norm, seq, H, True, ac), seq, hs, H),
               "modele_vent_vrai": errors(predict(model, norm, seq_true, H, False, ac), seq, hs, H),
               "reference": errors(const, seq, hs, H),
               "reference_altitude_constante": errors(level, seq, hs, H)}
        res = {}
        print(f"\n== {gname}: {len(seq)} sequences of {L * DT:.0f} s")
        print("   turbulence  n     median useful horizon [s]           RMS altitude [m]                  RMS speed 5 s [m/s]")
        print("                     est. / true wind / ref. / const. alt   est. / true wind / ref. / const. alt   ...")
        tmax = "60s" if L * DT >= 60 else "20s"
        for tl in ["toutes"] + TURB:
            m = np.ones(len(seq), bool) if tl == "toutes" else (turb == tl)
            if m.sum() == 0:
                continue
            res[tl] = {w: summarize(err[w], m, horizons) for w in WHO}
            hz = " / ".join(f"{res[tl][w]['horizon_utile_s']['mediane']:.2f}" for w in WHO)
            hh = " / ".join(f"{res[tl][w]['h'].get(tmax, float('nan')):.0f}" for w in WHO)
            va = " / ".join(f"{res[tl][w]['Va']['5s']:.2f}" for w in WHO)
            print(f"   {tl:10s} {int(m.sum()):4d}   {hz:36s}  at {tmax}: {hh:28s}  {va}")
        results[gname] = dict(n_sequences=len(seq), par_turbulence=res)
        if cfg.get("head") == "physics":
            at = attribution(model, norm, X[idx], alive[idx], data[idx], ch, P[idx], H)
            results[gname]["attribution"] = at
            print("   attribution (learned correction vs physics missing from the prior, first step):")
            for k, v in at.items():
                print(f"      {k:3s} R2 {v['r2']:+.2f}   RMS manquant {v['rms_manquant']:.4f}   "
                      f"RMS correction {v['rms_correction']:.4f}")
        if cfg.get("correction") == "param":
            ident = identification(model, norm, X[idx], alive[idx], P[idx],
                                   {k: v[idx] for k, v in meta.items() if k.startswith(("aero_scale", "power"))}, H)
            results[gname]["identification"] = ident
            print("   identification (learned per-flight multiplier vs true aircraft factor - 1):")
            for k, v in ident.items():
                print(f"      {k:3s} Spearman {v['spearman']:+.2f}   spread learned {v['std_learned']:.3f}   "
                      f"true {v['std_true']:.3f}")

    out = Path(a.model_dir) / f"eval_{a.set}.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\ndetailed results (all quantities, 1/5/20/60 s): {out}")


if __name__ == "__main__":
    main()
