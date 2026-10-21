"""Entraînement du modèle A (boîte noire), de la référence A0 (sans JEPA) et du modèle C (tête physique).

  --mode jepa    : étape 1 = encodeur + prédicteur, perte JEPA + lambda * SIGReg (latents prédits) ;
                   étape 2 = décodeur seul (encodeur et prédicteur figés), erreur sur l'état.
  --mode a0      : même réseau entraîné d'un bloc sur l'erreur d'état (prédiction directe, pas de JEPA).
  --mode joint   : JEPA + SIGReg + erreur d'état ensemble, décodeur appris en même temps (pas figé).
  --mode c_direct: comme a0, avec la tête physique (--prior) à la place du décodeur MLP ; perte = erreur
                   d'état + --lambda-res * taille de la correction apprise (idée d'APHYNITY).
  --mode c_jepa  : tronc (encodeur + prédicteur) d'un run jepa (--trunk) figé, tête physique seule apprise
                   (comme la sonde physique de SkyJEPA), même perte que c_direct.
Horizon d'entraînement : de --T pas au départ jusqu'à --T-max pas (curriculum linéaire atteint à
--curriculum-frac de l'entraînement) ; la validation utilise toujours --T-max.
On garde les poids les meilleurs sur les vols de validation.

Entrées v2 par défaut (2 s de passé, encodeur à 5 couches, masse et inerties en entrée). Pour refaire un
run v1 : --H 10 --channels 16,16,32 --dilations 1,2,4 --aircraft-input 0.

Usage : python scripts/train_a.py --data-dir <out_dir>/data_v1 --out runs/model_a --mode jepa
        python scripts/train_a.py ... --mode c_direct --prior <data_dir>/priors/ajuste.json
        python scripts/train_a.py ... --mode c_jepa --prior <...>/ajuste.json --trunk runs/jepa_v2T100_s0
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.windows import (FEATURES, N_STATE, Normalizer, WindowSampler, aircraft_params,  # noqa: E402
                                   features, load_set, split_mask)
from wingjepa.models.jepa import model_from_config, sigreg  # noqa: E402


def lr_at(step, total, peak, warmup, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    x = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(x, 1.0)))


def run_stage(name, params, loss_fn, val_fn, steps, a, log, save_best):
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)
    best, t0, skipped = float("inf"), time.time(), 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, a.lr, a.warmup, a.lr_floor)
        loss, parts = loss_fn(step)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, a.clip)
        # un lot à perte ou gradient non fini est sauté : un seul NaN suffirait à détruire les poids
        if torch.isfinite(loss) and torch.isfinite(gnorm):
            opt.step()
        else:
            skipped += 1
        if step % a.eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                val = float(val_fn())
            if val < best:
                best = val
                save_best()
            rec = dict(stage=name, step=step, train=float(loss), val=float(val), best=float(best),
                       lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1), skipped=skipped, **parts)
            log.append(rec)
            print(json.dumps(rec), flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--set", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["jepa", "a0", "joint", "c_direct", "c_jepa"], default="jepa")
    ap.add_argument("--H", type=int, default=40, help="pas de passé (0,05 s)")
    ap.add_argument("--channels", default="16,16,32,32,32", help="canaux des couches de l'encodeur")
    ap.add_argument("--dilations", default="1,2,4,8,16", help="champ réceptif = 1 + 2 * somme")
    ap.add_argument("--aircraft-input", type=int, default=1, help="masse et inerties en entrée (1) ou non (0)")
    ap.add_argument("--T", type=int, default=20, help="horizon d'entraînement au départ (pas de 0,05 s)")
    ap.add_argument("--T-max", type=int, default=None, help="horizon final (défaut : --T, pas de curriculum)")
    ap.add_argument("--curriculum-frac", type=float, default=0.5)
    ap.add_argument("--lambda-state", type=float, default=1.0, help="poids de l'erreur d'état (mode joint)")
    ap.add_argument("--prior", default=None, help="a priori de la tête physique (scripts/fit_prior.py)")
    ap.add_argument("--trunk", default=None, help="run jepa dont on reprend le tronc figé (mode c_jepa)")
    ap.add_argument("--lambda-res", type=float, default=1e-3, help="pénalité sur la taille de la correction")
    ap.add_argument("--out-scale", type=float, default=0.1, help="échelle de la correction (coefficients)")
    ap.add_argument("--n-sub", type=int, default=1, help="sous-pas RK4 par pas de 0,05 s (tête physique)")
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--steps1", type=int, default=20000)
    ap.add_argument("--steps2", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--lr-floor", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--clip", type=float, default=0.5)
    ap.add_argument("--lambda-sig", type=float, default=0.02)
    ap.add_argument("--sigreg-proj", type=int, default=128)
    ap.add_argument("--sigreg-steps", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--val-windows", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    physics = a.mode in ("c_direct", "c_jepa")
    a.head = "physics" if physics else "mlp"
    if physics and not a.prior:
        ap.error("--prior est requis pour les modes c_direct et c_jepa")
    a.prior_file, a.prior = a.prior, (json.load(open(a.prior)) if physics else None)
    trunk_cfg = None
    if a.mode == "c_jepa":
        if not a.trunk:
            ap.error("--trunk est requis pour le mode c_jepa")
        trunk_cfg = json.load(open(Path(a.trunk) / "config.json"))
        for k in ("H", "z_dim", "channels", "dilations", "aircraft_input"):     # architecture du tronc
            setattr(a, k, trunk_cfg.get(k, getattr(a, k)))

    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, _ = features(data, ch)
    P = aircraft_params(meta)
    tr, va = split_mask(meta, "train", a.seed), split_mask(meta, "val", a.seed)
    norm = Normalizer.from_dict(trunk_cfg["norm"]) if trunk_cfg else Normalizer.fit(X[tr], alive[tr])
    Xn = norm(X)
    T_max = a.T_max or a.T
    S_tr = WindowSampler(Xn[tr], alive[tr], a.H, T_max, seed=a.seed, P=P[tr])
    S_va = WindowSampler(Xn[va], alive[va], a.H, T_max, seed=a.seed + 1, P=P[va])
    vrows = S_va.fixed_rows(a.val_windows)
    val_seq, val_ac = torch.from_numpy(S_va.gather(vrows)), torch.from_numpy(S_va.aircraft(vrows))
    print(f"vols : {tr.sum()} entraînement, {va.sum()} validation ; fenêtres : {len(S_tr)} / {len(S_va)}", flush=True)

    model = model_from_config(dict(vars(a), norm=norm.to_dict()))
    if trunk_cfg:
        trunk = model_from_config(trunk_cfg)
        trunk.load_state_dict(torch.load(Path(a.trunk) / "model.pt"))
        model.encoder.load_state_dict(trunk.encoder.state_dict())
        model.predictor.load_state_dict(trunk.predictor.state_dict())
    print(f"paramètres : {model.n_params()} (total {sum(model.n_params().values())}) ; "
          f"champ réceptif de l'encodeur {model.encoder.receptive_field} pas pour H = {a.H}", flush=True)
    H = a.H

    def T_at(step, steps):
        """Curriculum : de a.T à T_max pas, linéairement, atteint à curriculum_frac de l'étape."""
        x = min(1.0, step / max(1, a.curriculum_frac * steps))
        return int(round(a.T + (T_max - a.T) * x))

    def nxt(T):
        rows = S_tr.sample_rows(a.batch)
        return torch.from_numpy(S_tr.gather(rows))[:, :H + T], torch.from_numpy(S_tr.aircraft(rows))

    def state_err(pred, seq):
        return ((pred - seq[:, H:, :N_STATE]) ** 2).mean()

    def state_loss(seq, ac, frozen=False):
        """Erreur d'état normalisée et taille de la correction physique (0 pour le MLP)."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            z0, zs, U = model.latents(seq, H, True, ac)
        pred, raw = model.head_states(z0, zs, U, seq, H, ac)
        res = (raw ** 2).mean() if raw is not None else torch.zeros(())
        return state_err(pred, seq), res

    val_state = lambda: state_loss(val_seq, val_ac)[0]
    log = []
    ckpt = lambda: torch.save(model.state_dict(), out / "model.pt")

    def jepa_loss(seq, ac):
        z0, zp, U = model.latents(seq, H, True, ac)
        zt = model.target_latents(seq, H, ac)
        pred = ((zp - zt) ** 2).mean()
        # SIGReg sur les latents prédits, moyenné dans le temps : estimé sur quelques pas tirés au hasard
        ks = torch.randperm(zp.shape[1])[:a.sigreg_steps]
        reg = torch.stack([sigreg(zp[:, k], n_proj=a.sigreg_proj) for k in ks]).mean()
        return pred + a.lambda_sig * reg, (z0, zp, U), dict(pred=pred.item(), sigreg=reg.item())

    trunk_params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    if a.mode == "jepa":
        def loss1(step):
            T = T_at(step, a.steps1)
            loss, _, parts = jepa_loss(*nxt(T))
            return loss, dict(parts, T=T)

        def val1():
            # erreur de prédiction normalisée par la dispersion des latents cibles : indépendante de la
            # taille du lot, et pénalise l'effondrement (si tous les z se ressemblent, elle explose)
            zp, zt = model.predict_latents(val_seq, H, True, val_ac), model.target_latents(val_seq, H, val_ac)
            return ((zp - zt) ** 2).mean() / zt.reshape(-1, zt.shape[-1]).var(0).mean()

        best1 = run_stage("jepa", trunk_params, loss1, val1, a.steps1, a, log, ckpt)
        model.load_state_dict(torch.load(out / "model.pt"))
        for p_ in trunk_params:
            p_.requires_grad_(False)                                  # étape 2 : tronc figé

        def loss2(step):
            return state_loss(*nxt(T_max), frozen=True)[0], {}

        best2 = run_stage("decodeur", list(model.decoder.parameters()), loss2, val_state, a.steps2, a, log, ckpt)
        summary = dict(best_val_jepa=best1, best_val_state=best2)
    elif a.mode == "joint":
        steps = a.steps1 + a.steps2

        def loss_joint(step):
            T = T_at(step, steps)
            seq, ac = nxt(T)
            lj, (z0, zp, U), parts = jepa_loss(seq, ac)
            ls = state_err(model.head_states(z0, zp, U, seq, H, ac)[0], seq)
            return lj + a.lambda_state * ls, dict(parts, state=ls.item(), T=T)

        best = run_stage("joint", list(model.parameters()), loss_joint, val_state, steps, a, log, ckpt)
        summary = dict(best_val_state=best)
    elif a.mode == "c_jepa":
        for p_ in trunk_params:
            p_.requires_grad_(False)

        def loss_cj(step):
            T = T_at(step, a.steps2)
            ls, res = state_loss(*nxt(T), frozen=True)
            return ls + a.lambda_res * res, dict(state=ls.item(), res=res.item(), T=T)

        best = run_stage("tete_physique", list(model.decoder.parameters()), loss_cj, val_state, a.steps2,
                         a, log, ckpt)
        summary = dict(best_val_state=best)
    else:                                                             # a0, c_direct
        steps = a.steps1 + a.steps2

        def loss_direct(step):
            T = T_at(step, steps)
            ls, res = state_loss(*nxt(T))
            return ls + a.lambda_res * res, dict(state=ls.item(), res=res.item(), T=T)

        best = run_stage(a.mode, list(model.parameters()), loss_direct, val_state, steps, a, log, ckpt)
        summary = dict(best_val_state=best)

    cfg = dict(vars(a), features=FEATURES, n_params=model.n_params(), norm=norm.to_dict(), **summary)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    json.dump(log, open(out / "log.json", "w"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
