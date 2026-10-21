"""Entraînement du modèle A (boîte noire) et de la référence A0 (sans JEPA).

  --mode jepa : étape 1 = encodeur + prédicteur, perte JEPA + lambda * SIGReg (latents prédits) ;
                étape 2 = décodeur seul (encodeur et prédicteur figés), erreur sur l'état.
  --mode a0   : même réseau entraîné d'un bloc sur l'erreur d'état (prédiction directe, pas de JEPA).
  --mode joint: JEPA + SIGReg + erreur d'état ensemble, décodeur appris en même temps (pas figé).
Horizon d'entraînement : de --T pas au départ jusqu'à --T-max pas (curriculum linéaire atteint à
--curriculum-frac de l'entraînement) ; la validation utilise toujours --T-max.
On garde les poids les meilleurs sur les vols de validation.

Usage : python scripts/train_a.py --data-dir <out_dir>/data_v1 --out runs/model_a --mode jepa
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
from wingjepa.data.windows import (FEATURES, N_STATE, Normalizer, WindowSampler, features,  # noqa: E402
                                   load_set, split_mask)
from wingjepa.models.jepa import WorldModel, sigreg  # noqa: E402


def lr_at(step, total, peak, warmup, floor):
    if step < warmup:
        return peak * (step + 1) / warmup
    x = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(x, 1.0)))


def run_stage(name, params, loss_fn, val_fn, steps, a, log, save_best):
    opt = torch.optim.Adam(params, lr=a.lr, weight_decay=a.weight_decay)
    best, t0 = float("inf"), time.time()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, a.lr, a.warmup, a.lr_floor)
        loss, parts = loss_fn(step)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        if step % a.eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                val = float(val_fn())
            if val < best:
                best = val
                save_best()
            rec = dict(stage=name, step=step, train=float(loss), val=float(val), best=float(best),
                       lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1), **parts)
            log.append(rec)
            print(json.dumps(rec), flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--set", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["jepa", "a0", "joint"], default="jepa")
    ap.add_argument("--H", type=int, default=10)
    ap.add_argument("--T", type=int, default=20, help="horizon d'entraînement au départ (pas de 0,05 s)")
    ap.add_argument("--T-max", type=int, default=None, help="horizon final (défaut : --T, pas de curriculum)")
    ap.add_argument("--curriculum-frac", type=float, default=0.5)
    ap.add_argument("--lambda-state", type=float, default=1.0, help="poids de l'erreur d'état (mode joint)")
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

    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, _ = features(data, ch)
    tr, va = split_mask(meta, "train", a.seed), split_mask(meta, "val", a.seed)
    norm = Normalizer.fit(X[tr], alive[tr])
    Xn = norm(X)
    T_max = a.T_max or a.T
    S_tr = WindowSampler(Xn[tr], alive[tr], a.H, T_max, seed=a.seed)
    S_va = WindowSampler(Xn[va], alive[va], a.H, T_max, seed=a.seed + 1)
    val_seq = torch.from_numpy(S_va.fixed(a.val_windows))
    print(f"vols : {tr.sum()} entraînement, {va.sum()} validation ; fenêtres : {len(S_tr)} / {len(S_va)}", flush=True)

    model = WorldModel(z_dim=a.z_dim)
    print(f"paramètres : {model.n_params()} (total {sum(model.n_params().values())})", flush=True)
    H = a.H

    def T_at(step, steps):
        """Curriculum : de a.T à T_max pas, linéairement, atteint à curriculum_frac de l'étape."""
        x = min(1.0, step / max(1, a.curriculum_frac * steps))
        return int(round(a.T + (T_max - a.T) * x))

    def nxt(T):
        return torch.from_numpy(S_tr.batch(a.batch))[:, :H + T]

    state_loss = lambda zf, seq: ((model.decoder(zf) - seq[:, H:, :N_STATE]) ** 2).mean()
    log = []
    ckpt = lambda: torch.save(model.state_dict(), out / "model.pt")

    def jepa_loss(seq):
        zp, zt = model.predict_latents(seq, H), model.target_latents(seq, H)
        pred = ((zp - zt) ** 2).mean()
        # SIGReg sur les latents prédits, moyenné dans le temps : estimé sur quelques pas tirés au hasard
        ks = torch.randperm(zp.shape[1])[:a.sigreg_steps]
        reg = torch.stack([sigreg(zp[:, k], n_proj=a.sigreg_proj) for k in ks]).mean()
        return pred + a.lambda_sig * reg, zp, dict(pred=pred.item(), sigreg=reg.item())

    if a.mode == "jepa":
        def loss1(step):
            T = T_at(step, a.steps1)
            loss, zp, parts = jepa_loss(nxt(T))
            return loss, dict(parts, T=T)

        def val1():
            # erreur de prédiction normalisée par la dispersion des latents cibles : indépendante de la
            # taille du lot, et pénalise l'effondrement (si tous les z se ressemblent, elle explose)
            zp, zt = model.predict_latents(val_seq, H), model.target_latents(val_seq, H)
            return ((zp - zt) ** 2).mean() / zt.reshape(-1, zt.shape[-1]).var(0).mean()

        best1 = run_stage("jepa", list(model.encoder.parameters()) + list(model.predictor.parameters()),
                          loss1, val1, a.steps1, a, log, ckpt)
        model.load_state_dict(torch.load(out / "model.pt"))
        for p in list(model.encoder.parameters()) + list(model.predictor.parameters()):
            p.requires_grad_(False)                                   # étape 2 : tronc figé

        def loss2(step):
            seq = nxt(T_max)
            with torch.no_grad():
                zf = model.predict_latents(seq, H)
            return state_loss(zf, seq), {}

        def val2():
            return state_loss(model.predict_latents(val_seq, H), val_seq)

        best2 = run_stage("decodeur", list(model.decoder.parameters()), loss2, val2, a.steps2, a, log, ckpt)
        summary = dict(best_val_jepa=best1, best_val_state=best2)
    elif a.mode == "joint":
        steps = a.steps1 + a.steps2

        def loss_joint(step):
            T = T_at(step, steps)
            seq = nxt(T)
            lj, zp, parts = jepa_loss(seq)
            ls = state_loss(zp, seq)
            return lj + a.lambda_state * ls, dict(parts, state=ls.item(), T=T)

        best = run_stage("joint", list(model.parameters()), loss_joint,
                         lambda: state_loss(model.predict_latents(val_seq, H), val_seq), steps, a, log, ckpt)
        summary = dict(best_val_state=best)
    else:
        steps = a.steps1 + a.steps2

        def loss_a0(step):
            T = T_at(step, steps)
            seq = nxt(T)
            return state_loss(model.predict_latents(seq, H), seq), dict(T=T)

        best = run_stage("a0", list(model.parameters()), loss_a0,
                         lambda: state_loss(model.predict_latents(val_seq, H), val_seq),
                         steps, a, log, ckpt)
        summary = dict(best_val_state=best)

    cfg = dict(vars(a), features=FEATURES, n_params=model.n_params(), norm=norm.to_dict(), **summary)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    json.dump(log, open(out / "log.json", "w"))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
