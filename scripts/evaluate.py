"""Évaluation en boucle ouverte d'un world model : à partir de 0,5 s de passé réel, le modèle prédit
seul pendant L secondes en ne recevant que les commandes réellement appliquées et le vent.

Trois prédictions sont comparées à la vérité :
  - modele_vent_estime : dernier vent estimé tenu constant (cas réaliste, comme en usage) ;
  - modele_vent_vrai   : vrai vent (moyen + rafales) à chaque pas, la turbulence restant inconnue ;
                         sépare l'erreur du modèle de ce qui est imprévisible ;
  - reference          : « rien ne change » (état constant, vitesse verticale constante).
Résultats par groupe d'avions et par niveau de turbulence.

Horizon utile = premier instant où |erreur d'altitude| > 5 m, |erreur de vitesse| > 1 m/s ou
|erreur de phi ou theta| > 10 deg.

Usage : python scripts/evaluate.py --model-dir runs/jepa_s0 --data-dir <out_dir>/data_v1 --set test_long
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.windows import (N_CTRL, N_STATE, STATE, Normalizer, features, load_set,  # noqa: E402
                                   split_mask, true_wind_features)
from wingjepa.models.jepa import WorldModel  # noqa: E402

DT = 0.05
DEG = 180 / np.pi
ANGLES = {"alpha", "beta", "p", "q", "r", "phi", "theta"}
THRESH = dict(h=5.0, Va=1.0, att=10.0)
TURB = ["none", "light", "moderate", "severe"]
WHO = ["modele_vent_estime", "modele_vent_vrai", "reference"]


def make_rollouts(X, alive, h, H, L, stride):
    """Départs (vol, s) tels que H + L pas soient valides : séquences (N, H+L, 16), altitudes, n° de vol."""
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
def predict(model, norm, seq, H, hold_wind, batch=512):
    """seq : (N, H+L, 16) non normalisé -> états prédits (N, L, 9) non normalisés."""
    out = []
    for i in range(0, len(seq), batch):
        s = torch.from_numpy(norm(seq[i:i + batch]))
        out.append(model.decoder(model.predict_latents(s, H, hold_wind)).numpy())
    return norm.state_inverse(np.concatenate(out))


def errors(pred, seq, hs, H):
    """Erreurs (N, L) par grandeur, angles en degrés ; altitude = intégrale de la vitesse verticale."""
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
    ap.add_argument("--horizon-s", type=float, default=None, help="défaut : 25 s (vols de 30 s), 60 s (vols longs)")
    ap.add_argument("--stride-s", type=float, default=None)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)

    cfg = json.load(open(Path(a.model_dir) / "config.json"))
    H = cfg["H"]
    model = WorldModel(z_dim=cfg["z_dim"])
    model.load_state_dict(torch.load(Path(a.model_dir) / "model.pt"))
    model.eval()
    norm = Normalizer.from_dict(cfg["norm"])

    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, h = features(data, ch)
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
            print(f"[{gname}] aucune séquence complète de {L * DT:.0f} s")
            continue
        seq_true, _, _ = make_rollouts(X_true[idx], alive[idx], h[idx], H, L, stride)
        turb = meta["turb_level"][idx][fl]
        err = {"modele_vent_estime": errors(predict(model, norm, seq, H, True), seq, hs, H),
               "modele_vent_vrai": errors(predict(model, norm, seq_true, H, False), seq, hs, H),
               "reference": errors(np.repeat(seq[:, H - 1:H, :N_STATE], L, 1), seq, hs, H)}
        res = {}
        print(f"\n== {gname} : {len(seq)} séquences de {L * DT:.0f} s")
        print("   turbulence  n     horizon utile médian [s]        RMS altitude [m]              RMS vitesse 5 s [m/s]")
        print("                     estimé / vrai vent / réf.       estimé / vrai vent / réf.     estimé / vrai vent / réf.")
        tmax = f"{min(60, int(L * DT)):g}s"
        for tl in ["toutes"] + TURB:
            m = np.ones(len(seq), bool) if tl == "toutes" else (turb == tl)
            if m.sum() == 0:
                continue
            res[tl] = {w: summarize(err[w], m, horizons) for w in WHO}
            hz = " / ".join(f"{res[tl][w]['horizon_utile_s']['mediane']:.2f}" for w in WHO)
            hh = " / ".join(f"{res[tl][w]['h'].get(tmax, float('nan')):.0f}" for w in WHO)
            va = " / ".join(f"{res[tl][w]['Va']['5s']:.2f}" for w in WHO)
            print(f"   {tl:10s} {int(m.sum()):4d}   {hz:28s}  à {tmax}: {hh:22s}  {va}")
        results[gname] = dict(n_sequences=len(seq), par_turbulence=res)

    out = Path(a.model_dir) / f"eval_{a.set}.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nrésultats détaillés (toutes les grandeurs, 1/5/20/60 s) : {out}")


if __name__ == "__main__":
    main()
