"""Évaluation en boucle ouverte d'un world model : à partir de 0,5 s de passé réel, le modèle prédit
seul pendant L secondes en ne recevant que les commandes réellement appliquées et le vent.

Trois prédictions sont comparées à la vérité :
  - modele_vent_estime : dernier vent estimé tenu constant (cas réaliste, comme en usage) ;
  - modele_vent_vrai   : vrai vent (moyen + rafales) à chaque pas, la turbulence restant inconnue ;
                         sépare l'erreur du modèle de ce qui est imprévisible ;
  - reference          : « rien ne change » (état constant, vitesse verticale constante) ;
  - reference_altitude_constante : état constant et vitesse verticale nulle (journal, § 9 : les vols restent
                         près de leur consigne d'altitude, cette référence est dure à battre à 60 s).
Résultats par groupe d'avions et par niveau de turbulence.
Modèle C (tête physique) : en plus, « attribution » : la correction apprise au premier pas est comparée à ce
qui manque vraiment à l'a priori (forces et moments vrais du simulateur - a priori, au même état) ; R2 par
composante (1 = la correction a appris exactement la physique manquante, 0 = rien, < 0 = pire que rien).

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
from wingjepa.data.windows import (N_CTRL, N_STATE, STATE, Normalizer, aircraft_params, features,  # noqa: E402
                                   load_set, split_mask, true_wind_features, window_starts)
from wingjepa.models.jepa import model_from_config  # noqa: E402

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
def predict(model, norm, seq, H, hold_wind, ac=None, batch=512):
    """seq : (N, H+L, 16) non normalisé, ac : (N, 5) avion -> états prédits (N, L, 9) non normalisés."""
    out = []
    for i in range(0, len(seq), batch):
        s = torch.from_numpy(norm(seq[i:i + batch]))
        a = None if ac is None else torch.from_numpy(ac[i:i + batch])
        out.append(model.predict_states(s, H, hold_wind, a)[0].numpy())
    return norm.state_inverse(np.concatenate(out))


@torch.no_grad()
def attribution(model, norm, X, alive, data, ch, P, H, n=20000, seed=0):
    """Correction apprise au premier pas vs physique vraiment manquante à l'a priori (R2 par composante)."""
    head = model.decoder
    st = window_starts(alive, H + 1)
    st = st[np.random.default_rng(seed).choice(len(st), min(n, len(st)), replace=False)]
    f, s = st[:, 0], st[:, 1] + H - 1                                   # dernier pas observé
    seq = torch.from_numpy(norm(X[f[:, None], st[:, 1:2] + np.arange(H + 1)]))
    ac = torch.from_numpy(P[f])
    z0 = model.encoder(seq[:, :H, :N_STATE], model.ctx(ac))
    U = model.future_inputs(seq, H, True)
    dC, _ = head.correction(z0, seq[:, H - 1, :N_STATE], U[:, 0])
    c = {k: torch.from_numpy(data[f, s, ch.index(k)].astype(np.float32)) for k in
         ["Va", "alpha", "beta", "p", "q", "r", "de_cmd", "da_cmd", "dr_cmd", "dt_cmd", "thrust",
          "Fx_aero", "Fy_aero", "Fz_aero", "Mx_aero", "My_aero", "Mz_aero", "Mx_prop", "My_prop", "Mz_prop"]}
    Va = c["Va"].clamp_min(3.0)
    qS = 0.5 * head.rho * Va * Va * head.S
    true = torch.stack([(c["Fx_aero"] + c["thrust"]) / qS, c["Fy_aero"] / qS, c["Fz_aero"] / qS,
                        (c["Mx_aero"] + c["Mx_prop"]) / (qS * head.b), (c["My_aero"] + c["My_prop"]) / (qS * head.c),
                        (c["Mz_aero"] + c["Mz_prop"]) / (qS * head.b)], -1)
    prior = head.prior_body(c["alpha"], c["beta"], c["de_cmd"], c["da_cmd"], c["dr_cmd"], c["p"], c["q"], c["r"],
                            c["dt_cmd"], Va)
    miss = (true - prior).numpy()
    dC = dC.numpy()
    out = {}
    for i, k in enumerate(BODY):
        out[k] = dict(r2=float(1 - np.mean((dC[:, i] - miss[:, i]) ** 2) / max(miss[:, i].var(), 1e-12)),
                      rms_manquant=float(np.sqrt(np.mean(miss[:, i] ** 2))),
                      rms_correction=float(np.sqrt(np.mean(dC[:, i] ** 2))))
    return out


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
            print(f"[{gname}] aucune séquence complète de {L * DT:.0f} s")
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
        print(f"\n== {gname} : {len(seq)} séquences de {L * DT:.0f} s")
        print("   turbulence  n     horizon utile médian [s]            RMS altitude [m]                  RMS vitesse 5 s [m/s]")
        print("                     estimé / vrai vent / réf. / alt. cst   estimé / vrai vent / réf. / alt. cst   ...")
        tmax = "60s" if L * DT >= 60 else "20s"
        for tl in ["toutes"] + TURB:
            m = np.ones(len(seq), bool) if tl == "toutes" else (turb == tl)
            if m.sum() == 0:
                continue
            res[tl] = {w: summarize(err[w], m, horizons) for w in WHO}
            hz = " / ".join(f"{res[tl][w]['horizon_utile_s']['mediane']:.2f}" for w in WHO)
            hh = " / ".join(f"{res[tl][w]['h'].get(tmax, float('nan')):.0f}" for w in WHO)
            va = " / ".join(f"{res[tl][w]['Va']['5s']:.2f}" for w in WHO)
            print(f"   {tl:10s} {int(m.sum()):4d}   {hz:36s}  à {tmax}: {hh:28s}  {va}")
        results[gname] = dict(n_sequences=len(seq), par_turbulence=res)
        if cfg.get("head") == "physics":
            at = attribution(model, norm, X[idx], alive[idx], data[idx], ch, P[idx], H)
            results[gname]["attribution"] = at
            print("   attribution (correction apprise vs physique manquante à l'a priori, premier pas) :")
            for k, v in at.items():
                print(f"      {k:3s} R2 {v['r2']:+.2f}   RMS manquant {v['rms_manquant']:.4f}   "
                      f"RMS correction {v['rms_correction']:.4f}")

    out = Path(a.model_dir) / f"eval_{a.set}.json"
    json.dump(results, open(out, "w"), indent=1)
    print(f"\nrésultats détaillés (toutes les grandeurs, 1/5/20/60 s) : {out}")


if __name__ == "__main__":
    main()
