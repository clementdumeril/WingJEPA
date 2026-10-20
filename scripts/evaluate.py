"""Évaluation en boucle ouverte d'un world model : à partir de 0,5 s de passé réel, le modèle prédit
seul pendant L secondes en ne recevant que les commandes réellement appliquées (et le dernier vent
estimé, tenu constant). On compare à la vérité et à une référence triviale (« rien ne change » :
état constant, vitesse verticale constante).

Sorties : erreurs RMS à 1, 5, 20, 60 s par grandeur, et « horizon utile » = premier instant où
|erreur d'altitude| > 5 m, |erreur de vitesse| > 1 m/s ou |erreur de phi ou theta| > 10 deg.

Usage : python scripts/evaluate.py --model-dir runs/model_a --data-dir <out_dir>/data_v1 --set test_long
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.windows import N_STATE, STATE, Normalizer, features, load_set, split_mask  # noqa: E402
from wingjepa.models.jepa import WorldModel  # noqa: E402

DT = 0.05
DEG = 180 / np.pi
ANGLES = {"alpha", "beta", "p", "q", "r", "phi", "theta"}
THRESH = dict(h=5.0, Va=1.0, att=10.0)


def make_rollouts(X, alive, h, H, L, stride):
    """Départs (vol, s) tels que H + L pas soient valides ; renvoie séquences (N, H+L, 16) et altitudes."""
    seqs, hs = [], []
    for f in range(X.shape[0]):
        for s in range(0, X.shape[1] - H - L + 1, stride):
            if alive[f, s:s + H + L].all():
                seqs.append(X[f, s:s + H + L])
                hs.append(h[f, s + H - 1:s + H + L])
    if not seqs:
        return None, None
    return np.stack(seqs), np.stack(hs)


@torch.no_grad()
def predict(model, norm, seq, H, batch=512):
    """seq : (N, H+L, 16) non normalisé -> états prédits (N, L, 9) non normalisés."""
    out = []
    for i in range(0, len(seq), batch):
        s = torch.from_numpy(norm(seq[i:i + batch]))
        out.append(model.decoder(model.predict_latents(s, H)).numpy())
    return norm.state_inverse(np.concatenate(out))


def errors(pred, seq, hs, H):
    """Erreurs (N, L) par grandeur, angles en degrés ; altitude = intégrale de la vitesse verticale."""
    true = seq[:, H:, :N_STATE]
    e = {}
    for i, k in enumerate(STATE):
        f = DEG if k in ANGLES else 1.0
        e[k] = (pred[:, :, i] - true[:, :, i]) * f
    h_pred = hs[:, :1] + np.cumsum(pred[:, :, STATE.index("hdot")], 1) * DT
    e["h"] = h_pred - hs[:, 1:]
    return e


def useful_horizon(e):
    bad = ((np.abs(e["h"]) > THRESH["h"]) | (np.abs(e["Va"]) > THRESH["Va"])
           | (np.abs(e["phi"]) > THRESH["att"]) | (np.abs(e["theta"]) > THRESH["att"]))
    first = np.where(bad.any(1), bad.argmax(1), bad.shape[1])
    return (first + 1) * DT, ~bad.any(1)


def summarize(e, horizons):
    L = e["h"].shape[1]
    out = {}
    for k in ["h", "Va", "alpha", "beta", "phi", "theta", "p", "q", "r", "hdot"]:
        out[k] = {f"{t:g}s": float(np.sqrt(np.mean(e[k][:, int(t / DT) - 1] ** 2)))
                  for t in horizons if int(t / DT) <= L}
    hz, cens = useful_horizon(e)
    out["horizon_utile_s"] = dict(mediane=float(np.median(hz)), q25=float(np.percentile(hz, 25)),
                                  q75=float(np.percentile(hz, 75)), jamais_depasse=float(cens.mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--set", required=True)
    ap.add_argument("--horizon-s", type=float, default=None, help="défaut : 25 s (train), 60 s (test_long)")
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
    long_set = data.shape[1] * DT > 60
    L = int((a.horizon_s or (60.0 if long_set else 25.0)) / DT)
    stride = int((a.stride_s or (10.0 if long_set else 2.5)) / DT)
    horizons = [1, 5, 20, 60]

    groups = {}
    if a.set == "train" or (meta["split"] == "train").any():
        groups["vols_de_test"] = split_mask(meta, "test", cfg["seed"])
    elif "is_test_aircraft" in meta and meta["is_test_aircraft"].any():
        groups["avions_entrainement"] = ~meta["is_test_aircraft"]
        groups["avions_de_test"] = meta["is_test_aircraft"].astype(bool)
    else:
        groups["tous"] = np.ones(len(X), bool)

    results = dict(model_dir=a.model_dir, set=a.set, horizon_s=L * DT, seuils=THRESH)
    for name, m in groups.items():
        seq, hs = make_rollouts(X[m], alive[m], h[m], H, L, stride)
        if seq is None:
            print(f"[{name}] aucune séquence complète de {L * DT:.0f} s")
            continue
        pred = predict(model, norm, seq, H)
        persist = np.repeat(seq[:, H - 1:H, :N_STATE], L, 1)       # référence : rien ne change
        res = dict(n_sequences=len(seq), modele=summarize(errors(pred, seq, hs, H), horizons),
                   reference_constante=summarize(errors(persist, seq, hs, H), horizons))
        results[name] = res
        print(f"== {name} : {len(seq)} séquences de {L * DT:.0f} s")
        for who in ("modele", "reference_constante"):
            r = res[who]
            hz = r["horizon_utile_s"]
            print(f"   {who:20s} horizon utile médian {hz['mediane']:.2f} s (q25 {hz['q25']:.2f}, q75 {hz['q75']:.2f}, "
                  f"jamais dépassé {hz['jamais_depasse']:.2f})")
            for k in ("h", "Va", "phi", "theta", "alpha", "beta"):
                print(f"      RMS {k:6s} " + "  ".join(f"{t}: {v:.3g}" for t, v in r[k].items()))

    out = Path(a.model_dir) / f"eval_{a.set}.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"résultats : {out}")


if __name__ == "__main__":
    main()
