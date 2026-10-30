"""Diagnostics of the already-trained world models (docs/journal.md, section 8).

Q1. Why is A0 better than JEPA at short range?
    - error per quantity at 0.25 / 0.5 / 1 / 2 / 5 s (normalized by the standard deviation of the quantity);
    - "probe": a small new decoder trained to read the PRESENT state from the encoder latent z(t).
      If the present state is poorly readable, the information is lost at encoding time.
Q2. Why is JEPA better on altitude at long range?
    - additional references: constant altitude (zero vertical speed);
    - spread of predicted / true vertical speeds as a function of horizon: a model that "shrinks"
      its predictions towards the mean drifts little in altitude without actually knowing the dynamics.
Q3. Why does no model generalize to extreme aircraft?
    - correlation (Spearman) between the useful horizon and the deviation of each aircraft parameter from
      its nominal value, on the test aircraft;
    - which quantities degrade at 1 s between training aircraft and test aircraft.

Usage: python scripts/diagnose.py --data-dir <out_dir>/data_v1 --models runs/a0_T100_s0 runs/jepa_T100_s0
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import DT, errors, load_model, make_rollouts, predict, useful_horizon  # noqa: E402
from wingjepa.data.windows import (N_STATE, STATE, WindowSampler, aircraft_params, features,  # noqa: E402
                                   load_set, split_mask)
from wingjepa.models.jepa import MLPDecoder  # noqa: E402

NOMINAL = dict(mass=6.804, Ixx=0.672485702, Iyy=0.889416574, Izz=1.57817209)


def probe_present_state(model, norm, X, alive, P, tr, va, H, steps=1500):
    """Normalized (validation) error of a new decoder that reads the present state from z(t)."""
    torch.manual_seed(0)
    Xn = norm(X)
    St = WindowSampler(Xn[tr], alive[tr], H, 1, seed=0, P=P[tr])
    Sv = WindowSampler(Xn[va], alive[va], H, 1, seed=1, P=P[va])
    vr = Sv.fixed_rows(4096)
    vs, vac = torch.from_numpy(Sv.gather(vr)), torch.from_numpy(Sv.aircraft(vr))
    probe = MLPDecoder(model.encoder.out.out_features)
    opt = torch.optim.Adam(probe.parameters(), 3e-3)
    for _ in range(steps):
        rows = St.sample_rows(1024)
        s, ac = torch.from_numpy(St.gather(rows)), torch.from_numpy(St.aircraft(rows))
        with torch.no_grad():
            z = model.encoder(s[:, :H, :N_STATE], model.ctx(ac))
        loss = ((probe(z) - s[:, H - 1, :N_STATE]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        e = ((probe(model.encoder(vs[:, :H, :N_STATE], model.ctx(vac))) - vs[:, H - 1, :N_STATE]) ** 2).mean(0)
    return dict(zip(STATE, e.numpy().round(3).tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    ap.add_argument("--train-set", default="train")
    ap.add_argument("--long-set", default="test_long")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    dtr, ch, mtr = load_set(a.data_dir, a.train_set)
    Xtr, atr, _ = features(dtr, ch)
    Ptr = aircraft_params(mtr)
    dl, chl, ml = load_set(a.data_dir, a.long_set)
    Xl, al, hl = features(dl, chl)
    Pl = aircraft_params(ml)
    std = Xtr[atr].std(0)[:N_STATE]
    test_ac = ml["is_test_aircraft"].astype(bool) if ml["is_test_aircraft"].any() else np.zeros(len(Xl), bool)

    for d in a.models:
        model, cfg, norm = load_model(d)
        H = cfg["H"]
        tr, va = split_mask(mtr, "train", cfg["seed"]), split_mask(mtr, "val", cfg["seed"])
        print(f"\n################ {d}")

        # ---------- Q1: short range
        print("[Q1] RMS error / standard deviation of the quantity, training aircraft (long flights):")
        L = int(60 / DT)
        out = {}
        for gname, gm in (("entrainement", ~test_ac), ("test", test_ac)):
            idx = np.flatnonzero(gm)
            if not len(idx):
                continue
            seq, hs, fl = make_rollouts(Xl[idx], al[idx], hl[idx], H, L, int(10 / DT))
            pred = predict(model, norm, seq, H, True, Pl[idx][fl])
            out[gname] = (seq, hs, fl, idx, pred)
        seq, hs, fl, idx, pred = out["entrainement"]
        true = seq[:, H:, :N_STATE]
        hor = [0.25, 0.5, 1, 2, 5]
        print("      quantity " + "".join(f"{t:>8g}s" for t in hor))
        for i, k in enumerate(STATE):
            r = [np.sqrt(np.mean((pred[:, int(t / DT) - 1, i] - true[:, int(t / DT) - 1, i]) ** 2)) / std[i] for t in hor]
            print(f"      {k:8s} " + "".join(f"{v:9.3f}" for v in r))
        pr = probe_present_state(model, norm, Xtr, atr, Ptr, tr, va, H)
        print(f"      probe (present state read from z(t), normalized error, 0 = perfect, 1 = nothing): {pr}")
        json.dump(dict(probe=pr, data_dir=Path(a.data_dir).name), open(Path(d) / "probe.json", "w"), indent=1)

        # ---------- Q2: altitude at long range
        print("[Q2] altitude at long range, training aircraft:")
        e = errors(pred, seq, hs, H)
        hdot_i = STATE.index("hdot")
        zero = np.zeros_like(pred)
        zero[:, :, :] = seq[:, H - 1:H, :N_STATE]
        zero[:, :, hdot_i] = 0.0                                          # constant altitude
        e0 = errors(zero, seq, hs, H)
        const = np.repeat(seq[:, H - 1:H, :N_STATE], L, 1)
        ec = errors(const, seq, hs, H)
        for t in (5, 20, 60):
            k = int(t / DT) - 1
            rms = lambda x: np.sqrt(np.mean(x[:, k] ** 2))
            ratio = pred[:, k, hdot_i].std() / true[:, k, hdot_i].std()
            print(f"      {t:3d} s : RMS altitude model {rms(e['h']):6.1f} m | constant altitude {rms(e0['h']):6.1f} m | "
                  f"constant vertical speed {rms(ec['h']):6.1f} m | predicted / true vz spread {ratio:.2f} | "
                  f"vz bias {np.mean(pred[:, k, hdot_i] - true[:, k, hdot_i]):+.2f} m/s")

        # ---------- Q3: generalization
        if "test" in out:
            seq_t, hs_t, fl_t, idx_t, pred_t = out["test"]
            et = errors(pred_t, seq_t, hs_t, H)
            hz, _ = useful_horizon(et)
            print(f"[Q3] test aircraft: median useful horizon {np.median(hz):.2f} s; Spearman correlation "
                  "between useful horizon and deviation of the parameter from its nominal value (negative = the deviation hurts):")
            fid = idx_t[fl_t]
            devs = {k: np.abs(ml[k][fid] / v - 1) for k, v in NOMINAL.items()}
            devs["power"] = np.abs(ml["power_scale"][fid] - 1)
            devs["cg_shift"] = np.abs(ml["cg_shift"][fid])
            for c in ("CD", "CY", "CL", "Cl", "Cm", "Cn"):
                devs[f"aero_{c}"] = np.abs(ml[f"aero_scale_{c}"][fid] - 1)
            for k, v in devs.items():
                rho, pval = spearmanr(v, hz)
                print(f"      {k:10s} rho = {rho:+.2f} (p = {pval:.3f})")
            seq_e, hs_e, _, _, pred_e = out["entrainement"]
            ee = errors(pred_e, seq_e, hs_e, H)
            k1 = int(1 / DT) - 1
            print("      RMS error at 1 s, training aircraft -> test aircraft:")
            for k in ("Va", "alpha", "beta", "phi", "theta", "p", "q", "r", "hdot"):
                print(f"         {k:6s} {np.sqrt(np.mean(ee[k][:, k1] ** 2)):7.3f} -> {np.sqrt(np.mean(et[k][:, k1] ** 2)):7.3f}")


if __name__ == "__main__":
    main()
