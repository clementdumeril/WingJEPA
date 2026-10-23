"""Training of model A (black box), of the A0 reference (no JEPA) and of model C (physics head).

  --mode jepa    : stage 1 = encoder + predictor, JEPA loss + lambda * SIGReg (predicted latents);
                   stage 2 = decoder alone (encoder and predictor frozen), state error.
  --mode a0      : same network trained in one block on the state error (direct prediction, no JEPA).
  --mode jepa_ft : JEPA as pre-training: stage 1 as in jepa, then stage 2 fine-tunes the whole network
                   (encoder, predictor, decoder) on the state error, at peak learning rate --ft-lr.
  --mode joint   : JEPA + SIGReg + state error together, decoder learned at the same time (not frozen).
  --mode c_direct: like a0, with the physics head (--prior) instead of the MLP decoder; loss = state
                   error + --lambda-res * size of the learned correction (APHYNITY idea).
  --mode c_jepa  : trunk (encoder + predictor) of a jepa run (--trunk) frozen, only the physics head is
                   learned (like the SkyJEPA physics probe), same loss as c_direct.
Training horizon: from --T steps at the start up to --T-max steps (linear curriculum reached at
--curriculum-frac of the training); validation always uses --T-max.
The best weights on the validation flights are kept.
--ema TAU (> 0): the JEPA targets come from a slow copy of the encoder, updated after every step as
target = TAU * target + (1 - TAU) * encoder, with no gradient through the targets (I-JEPA / V-JEPA).

v2 inputs by default (2 s of past, 5-layer encoder, mass and inertias as input). To redo a v1 run:
--H 10 --channels 16,16,32 --dilations 1,2,4 --aircraft-input 0.

Usage: python scripts/train_a.py --data-dir <out_dir>/data_v1 --out runs/model_a --mode jepa
       python scripts/train_a.py ... --mode c_direct --prior <data_dir>/priors/ajuste.json
       python scripts/train_a.py ... --mode c_jepa --prior <...>/ajuste.json --trunk runs/jepa_v2T100_s0
"""
import argparse
import copy
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


def run_stage(name, params, loss_fn, val_fn, steps, a, log, save_best, peak=None, after_step=None):
    peak = peak or a.lr
    opt = torch.optim.Adam(params, lr=peak, weight_decay=a.weight_decay)
    best, t0, skipped = float("inf"), time.time(), 0
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, peak, a.warmup, a.lr_floor)
        loss, parts = loss_fn(step)
        opt.zero_grad()
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, a.clip)
        # a batch with a non-finite loss or gradient is skipped: a single NaN would be enough to destroy the weights
        if torch.isfinite(loss) and torch.isfinite(gnorm):
            opt.step()
            if after_step:
                after_step()
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
    ap.add_argument("--mode", choices=["jepa", "jepa_ft", "a0", "joint", "c_direct", "c_jepa"], default="jepa")
    ap.add_argument("--H", type=int, default=40, help="past steps (0.05 s)")
    ap.add_argument("--channels", default="16,16,32,32,32", help="channels of the encoder layers")
    ap.add_argument("--dilations", default="1,2,4,8,16", help="receptive field = 1 + 2 * sum")
    ap.add_argument("--aircraft-input", type=int, default=1, help="mass and inertias as input (1) or not (0)")
    ap.add_argument("--T", type=int, default=20, help="training horizon at the start (steps of 0.05 s)")
    ap.add_argument("--T-max", type=int, default=None, help="final horizon (default: --T, no curriculum)")
    ap.add_argument("--curriculum-frac", type=float, default=0.5)
    ap.add_argument("--lambda-state", type=float, default=1.0, help="weight of the state error (joint mode)")
    ap.add_argument("--prior", default=None, help="prior of the physics head (scripts/fit_prior.py)")
    ap.add_argument("--trunk", default=None, help="jepa run whose frozen trunk is reused (c_jepa mode)")
    ap.add_argument("--lambda-res", type=float, default=1e-3, help="penalty on the size of the correction")
    ap.add_argument("--out-scale", type=float, default=0.1, help="scale of the correction (coefficients)")
    ap.add_argument("--n-sub", type=int, default=1, help="RK4 sub-steps per 0.05 s step (physics head)")
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
    ap.add_argument("--ema", type=float, default=0.0, help="slow target encoder (e.g. 0.996); 0 = off")
    ap.add_argument("--ft-lr", type=float, default=1e-3, help="peak learning rate of the fine-tuning (jepa_ft)")
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
        ap.error("--prior is required for the c_direct and c_jepa modes")
    a.prior_file, a.prior = a.prior, (json.load(open(a.prior)) if physics else None)
    trunk_cfg = None
    if a.mode == "c_jepa":
        if not a.trunk:
            ap.error("--trunk is required for the c_jepa mode")
        trunk_cfg = json.load(open(Path(a.trunk) / "config.json"))
        for k in ("H", "z_dim", "channels", "dilations", "aircraft_input"):     # trunk architecture
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
    print(f"flights: {tr.sum()} training, {va.sum()} validation; windows: {len(S_tr)} / {len(S_va)}", flush=True)

    model = model_from_config(dict(vars(a), norm=norm.to_dict()))
    if trunk_cfg:
        trunk = model_from_config(trunk_cfg)
        trunk.load_state_dict(torch.load(Path(a.trunk) / "model.pt"))
        model.encoder.load_state_dict(trunk.encoder.state_dict())
        model.predictor.load_state_dict(trunk.predictor.state_dict())
    print(f"parameters: {model.n_params()} (total {sum(model.n_params().values())}); "
          f"encoder receptive field {model.encoder.receptive_field} steps for H = {a.H}", flush=True)
    H = a.H

    def T_at(step, steps):
        """Curriculum: from a.T to T_max steps, linearly, reached at curriculum_frac of the stage."""
        x = min(1.0, step / max(1, a.curriculum_frac * steps))
        return int(round(a.T + (T_max - a.T) * x))

    def nxt(T):
        rows = S_tr.sample_rows(a.batch)
        return torch.from_numpy(S_tr.gather(rows))[:, :H + T], torch.from_numpy(S_tr.aircraft(rows))

    def state_err(pred, seq):
        return ((pred - seq[:, H:, :N_STATE]) ** 2).mean()

    def state_loss(seq, ac, frozen=False):
        """Normalized state error and size of the physics correction (0 for the MLP)."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            z0, zs, U = model.latents(seq, H, True, ac)
        pred, raw = model.head_states(z0, zs, U, seq, H, ac)
        res = (raw ** 2).mean() if raw is not None else torch.zeros(())
        return state_err(pred, seq), res

    val_state = lambda: state_loss(val_seq, val_ac)[0]
    log = []
    ckpt = lambda: torch.save(model.state_dict(), out / "model.pt")

    target_enc = None
    if a.ema > 0:
        target_enc = copy.deepcopy(model.encoder)
        for p_ in target_enc.parameters():
            p_.requires_grad_(False)

    @torch.no_grad()
    def ema_update():
        for pt, p_ in zip(target_enc.parameters(), model.encoder.parameters()):
            pt.mul_(a.ema).add_(p_.detach(), alpha=1 - a.ema)

    after = ema_update if target_enc is not None else None

    def jepa_loss(seq, ac):
        z0, zp, U = model.latents(seq, H, True, ac)
        zt = model.target_latents(seq, H, ac, encoder=target_enc)
        pred = ((zp - zt) ** 2).mean()
        # SIGReg on the predicted latents, averaged over time: estimated on a few randomly drawn steps
        ks = torch.randperm(zp.shape[1])[:a.sigreg_steps]
        reg = torch.stack([sigreg(zp[:, k], n_proj=a.sigreg_proj) for k in ks]).mean()
        return pred + a.lambda_sig * reg, (z0, zp, U), dict(pred=pred.item(), sigreg=reg.item())

    trunk_params = list(model.encoder.parameters()) + list(model.predictor.parameters())
    if a.mode in ("jepa", "jepa_ft"):
        def loss1(step):
            T = T_at(step, a.steps1)
            loss, _, parts = jepa_loss(*nxt(T))
            return loss, dict(parts, T=T)

        def val1():
            # prediction error normalized by the spread of the target latents: independent of the
            # batch size, and penalizes collapse (if all the z look alike, it blows up)
            zp = model.predict_latents(val_seq, H, True, val_ac)
            zt = model.target_latents(val_seq, H, val_ac, encoder=target_enc)
            return ((zp - zt) ** 2).mean() / zt.reshape(-1, zt.shape[-1]).var(0).mean()

        best1 = run_stage("jepa", trunk_params, loss1, val1, a.steps1, a, log, ckpt, after_step=after)
        model.load_state_dict(torch.load(out / "model.pt"))
        if a.mode == "jepa":
            for p_ in trunk_params:
                p_.requires_grad_(False)                              # stage 2: trunk frozen

            def loss2(step):
                return state_loss(*nxt(T_max), frozen=True)[0], {}

            best2 = run_stage("decodeur", list(model.decoder.parameters()), loss2, val_state, a.steps2, a, log, ckpt)
        else:                                                         # stage 2: fine-tune everything
            def loss2(step):
                return state_loss(*nxt(T_max))[0], {}

            best2 = run_stage("finetune", list(model.parameters()), loss2, val_state, a.steps2, a, log, ckpt,
                              peak=a.ft_lr)
        summary = dict(best_val_jepa=best1, best_val_state=best2)
    elif a.mode == "joint":
        steps = a.steps1 + a.steps2

        def loss_joint(step):
            T = T_at(step, steps)
            seq, ac = nxt(T)
            lj, (z0, zp, U), parts = jepa_loss(seq, ac)
            ls = state_err(model.head_states(z0, zp, U, seq, H, ac)[0], seq)
            return lj + a.lambda_state * ls, dict(parts, state=ls.item(), T=T)

        best = run_stage("joint", list(model.parameters()), loss_joint, val_state, steps, a, log, ckpt,
                         after_step=after)
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
