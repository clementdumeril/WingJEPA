"""Diagnostics des world models déjà entraînés (docs/journal.md, § 8).

Q1. Pourquoi A0 est meilleur à court terme que JEPA ?
    - erreur par grandeur à 0,25 / 0,5 / 1 / 2 / 5 s (normalisée par l'écart-type de la grandeur) ;
    - « sonde » : un petit décodeur neuf appris à lire l'état PRÉSENT dans le latent z(t) de l'encodeur.
      Si l'état présent est mal lisible, l'information est perdue dès l'encodage.
Q2. Pourquoi JEPA est meilleur sur l'altitude à long terme ?
    - références supplémentaires : altitude constante (vitesse verticale nulle) ;
    - dispersion des vitesses verticales prédites / vraies selon l'horizon : un modèle qui « rétrécit »
      ses prédictions vers la moyenne dérive peu en altitude sans pour autant connaître la dynamique.
Q3. Pourquoi aucun modèle ne généralise aux avions extrêmes ?
    - corrélation (Spearman) entre l'horizon utile et l'écart de chaque paramètre de l'avion à la
      valeur nominale, sur les avions de test ;
    - quelles grandeurs se dégradent à 1 s entre avions d'entraînement et avions de test.

Usage : python scripts/diagnose.py --data-dir <out_dir>/data_v1 --models runs/a0_T100_s0 runs/jepa_T100_s0
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
from evaluate import DT, errors, make_rollouts, predict, useful_horizon  # noqa: E402
from wingjepa.data.windows import (N_STATE, STATE, Normalizer, WindowSampler, features, load_set,  # noqa: E402
                                   split_mask)
from wingjepa.models.jepa import MLPDecoder, WorldModel  # noqa: E402

NOMINAL = dict(mass=6.804, Ixx=0.672485702, Iyy=0.889416574, Izz=1.57817209)


def load_model(d):
    cfg = json.load(open(Path(d) / "config.json"))
    m = WorldModel(z_dim=cfg["z_dim"])
    m.load_state_dict(torch.load(Path(d) / "model.pt"))
    m.eval()
    return m, cfg, Normalizer.from_dict(cfg["norm"])


def probe_present_state(model, norm, X, alive, tr, va, H, steps=1500):
    """Erreur normalisée (validation) d'un décodeur neuf qui lit l'état présent dans z(t)."""
    torch.manual_seed(0)
    Xn = norm(X)
    St = WindowSampler(Xn[tr], alive[tr], H, 1, seed=0)
    vs = torch.from_numpy(WindowSampler(Xn[va], alive[va], H, 1, seed=1).fixed(4096))
    probe = MLPDecoder(model.encoder.out.out_features)
    opt = torch.optim.Adam(probe.parameters(), 3e-3)
    for _ in range(steps):
        s = torch.from_numpy(St.batch(1024))
        with torch.no_grad():
            z = model.encoder(s[:, :H, :N_STATE])
        loss = ((probe(z) - s[:, H - 1, :N_STATE]) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        e = ((probe(model.encoder(vs[:, :H, :N_STATE])) - vs[:, H - 1, :N_STATE]) ** 2).mean(0)
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
    dl, chl, ml = load_set(a.data_dir, a.long_set)
    Xl, al, hl = features(dl, chl)
    std = Xtr[atr].std(0)[:N_STATE]
    test_ac = ml["is_test_aircraft"].astype(bool) if ml["is_test_aircraft"].any() else np.zeros(len(Xl), bool)

    for d in a.models:
        model, cfg, norm = load_model(d)
        H = cfg["H"]
        tr, va = split_mask(mtr, "train", cfg["seed"]), split_mask(mtr, "val", cfg["seed"])
        print(f"\n################ {d}")

        # ---------- Q1 : court terme
        print("[Q1] erreur RMS / écart-type de la grandeur, avions d'entraînement (vols longs) :")
        L = int(60 / DT)
        out = {}
        for gname, gm in (("entrainement", ~test_ac), ("test", test_ac)):
            idx = np.flatnonzero(gm)
            if not len(idx):
                continue
            seq, hs, fl = make_rollouts(Xl[idx], al[idx], hl[idx], H, L, int(10 / DT))
            pred = predict(model, norm, seq, H, True)
            out[gname] = (seq, hs, fl, idx, pred)
        seq, hs, fl, idx, pred = out["entrainement"]
        true = seq[:, H:, :N_STATE]
        hor = [0.25, 0.5, 1, 2, 5]
        print("      grandeur " + "".join(f"{t:>8g}s" for t in hor))
        for i, k in enumerate(STATE):
            r = [np.sqrt(np.mean((pred[:, int(t / DT) - 1, i] - true[:, int(t / DT) - 1, i]) ** 2)) / std[i] for t in hor]
            print(f"      {k:8s} " + "".join(f"{v:9.3f}" for v in r))
        pr = probe_present_state(model, norm, Xtr, atr, tr, va, H)
        print(f"      sonde (état présent lu dans z(t), erreur normalisée, 0 = parfait, 1 = rien) : {pr}")

        # ---------- Q2 : altitude à long terme
        print("[Q2] altitude à long terme, avions d'entraînement :")
        e = errors(pred, seq, hs, H)
        hdot_i = STATE.index("hdot")
        zero = np.zeros_like(pred)
        zero[:, :, :] = seq[:, H - 1:H, :N_STATE]
        zero[:, :, hdot_i] = 0.0                                          # altitude constante
        e0 = errors(zero, seq, hs, H)
        const = np.repeat(seq[:, H - 1:H, :N_STATE], L, 1)
        ec = errors(const, seq, hs, H)
        for t in (5, 20, 60):
            k = int(t / DT) - 1
            rms = lambda x: np.sqrt(np.mean(x[:, k] ** 2))
            ratio = pred[:, k, hdot_i].std() / true[:, k, hdot_i].std()
            print(f"      {t:3d} s : RMS altitude modèle {rms(e['h']):6.1f} m | altitude constante {rms(e0['h']):6.1f} m | "
                  f"vitesse verticale constante {rms(ec['h']):6.1f} m | dispersion vz prédite / vraie {ratio:.2f} | "
                  f"biais vz {np.mean(pred[:, k, hdot_i] - true[:, k, hdot_i]):+.2f} m/s")

        # ---------- Q3 : généralisation
        if "test" in out:
            seq_t, hs_t, fl_t, idx_t, pred_t = out["test"]
            et = errors(pred_t, seq_t, hs_t, H)
            hz, _ = useful_horizon(et)
            print(f"[Q3] avions de test : horizon utile médian {np.median(hz):.2f} s ; corrélation de Spearman "
                  "entre horizon utile et écart du paramètre à sa valeur nominale (négatif = l'écart nuit) :")
            fid = idx_t[fl_t]
            devs = {k: np.abs(ml[k][fid] / v - 1) for k, v in NOMINAL.items()}
            devs["puissance"] = np.abs(ml["power_scale"][fid] - 1)
            devs["centrage"] = np.abs(ml["cg_shift"][fid])
            for c in ("CD", "CY", "CL", "Cl", "Cm", "Cn"):
                devs[f"aero_{c}"] = np.abs(ml[f"aero_scale_{c}"][fid] - 1)
            for k, v in devs.items():
                rho, pval = spearmanr(v, hz)
                print(f"      {k:10s} rho = {rho:+.2f} (p = {pval:.3f})")
            seq_e, hs_e, _, _, pred_e = out["entrainement"]
            ee = errors(pred_e, seq_e, hs_e, H)
            k1 = int(1 / DT) - 1
            print("      erreur RMS à 1 s, avions d'entraînement -> avions de test :")
            for k in ("Va", "alpha", "beta", "phi", "theta", "p", "q", "r", "hdot"):
                print(f"         {k:6s} {np.sqrt(np.mean(ee[k][:, k1] ** 2)):7.3f} -> {np.sqrt(np.mean(et[k][:, k1] ** 2)):7.3f}")


if __name__ == "__main__":
    main()
