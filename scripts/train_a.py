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
  --correction param (c_direct, c_jepa): constrained physics head, no free forces; the 38 prior coefficients
                   are changed once per flight from z_0 (models/physics.py). Default free: 6 body-axis
                   coefficients added to the prior at every step.
Training horizon: from --T steps at the start up to --T-max steps (linear curriculum reached at
--curriculum-frac of the training); validation always uses --T-max.
The best weights on the validation flights are kept.
--ema TAU (> 0): the JEPA targets come from a slow copy of the encoder, updated after every step as
target = TAU * target + (1 - TAU) * encoder, with no gradient through the targets (I-JEPA / V-JEPA).

v2 inputs by default (2 s of past, 5-layer encoder, mass and inertias as input). To redo a v1 run:
--H 10 --channels 16,16,32 --dilations 1,2,4 --aircraft-input 0.

Restart: the training state is saved every --checkpoint-min minutes in <out>/resume.pt. A run stopped by
the time limit, resubmitted with the same command, continues where it stopped (same steps, same random
draws); the file is deleted when the run finishes. --device cuda trains on a GPU (needs a CUDA build of
torch, cluster/setup_gpu_env.sh); model.pt is always saved on the CPU.

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


class Resume:
    """Training state saved in <out>/resume.pt (weights, optimizer, position in the stage, random
    generators, log), at most every `every_min` minutes and after each new best. Loaded at start if
    present, so that a resubmitted run continues where it stopped."""

    def __init__(self, path, every_min, key, modules, sampler, log):
        self.path, self.every, self.key = path, every_min * 60, key
        self.modules, self.sampler, self.log = modules, sampler, log
        self.done, self.stage, self.last = {}, None, time.time()
        if path.exists():
            st = torch.load(path, map_location="cpu", weights_only=False)
            if st["key"] != key:
                sys.exit(f"{path} was saved by a run with other options: delete it to start over")
            for m, sd in zip(modules, st["modules"]):
                m.load_state_dict(sd)
            log[:] = st["log"]
            sampler.rng.bit_generator.state = st["sampler_rng"]
            torch.set_rng_state(st["torch_rng"])
            if "cuda_rng" in st and torch.cuda.is_available():
                torch.cuda.set_rng_state(st["cuda_rng"])
            self.done, self.stage = st["done"], st["stage"]
            print(f"resuming from {path}: stages done {list(self.done)}, "
                  f"stage {self.stage['name']} from step {self.stage['step']}", flush=True)

    def finished(self, name):
        """Best validation value of a stage completed before the restart, None otherwise."""
        return self.done.get(name)

    def restore(self, name, opt):
        """(first step, best, skipped batches, seconds already spent) of stage `name`."""
        st = self.stage
        if st is None or st["name"] != name:
            return 0, float("inf"), 0, 0.0
        opt.load_state_dict(st["opt"])
        return st["step"], st["best"], st["skipped"], st["sec"]

    def save(self, name, step, opt, best, skipped, sec, force=False):
        if self.every <= 0 or not (force or time.time() - self.last >= self.every):
            return
        st = dict(key=self.key, modules=[m.state_dict() for m in self.modules], log=self.log, done=self.done,
                  sampler_rng=self.sampler.rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                  stage=dict(name=name, step=step, opt=opt.state_dict(), best=best, skipped=skipped, sec=sec))
        if torch.cuda.is_available():
            st["cuda_rng"] = torch.cuda.get_rng_state()
        tmp = self.path.with_suffix(".tmp")
        torch.save(st, tmp)
        os.replace(tmp, self.path)                  # never a half-written file, even if the job is killed
        self.last = time.time()

    def end_stage(self, name, best):
        self.done[name] = best

    def clear(self):
        self.path.unlink(missing_ok=True)


def run_stage(name, params, loss_fn, val_fn, steps, a, log, save_best, resume, peak=None, after_step=None):
    if resume.finished(name) is not None:           # completed before the restart
        return resume.finished(name)
    peak = peak or a.lr
    opt = torch.optim.Adam(params, lr=peak, weight_decay=a.weight_decay)
    start, best, skipped, sec = resume.restore(name, opt)
    t0 = time.time() - sec
    for step in range(start, steps):
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
        improved = False
        if step % a.eval_every == 0 or step == steps - 1:
            with torch.no_grad():
                val = float(val_fn())
            if val < best:
                best, improved = val, True
                save_best()
            rec = dict(stage=name, step=step, train=loss.item(), val=float(val), best=float(best),
                       lr=opt.param_groups[0]["lr"], sec=round(time.time() - t0, 1), skipped=skipped, **parts)
            log.append(rec)
            print(json.dumps(rec), flush=True)
        # saved with each new best as well, so that model.pt and the saved best always agree
        resume.save(name, step + 1, opt, best, skipped, time.time() - t0, force=improved)
    resume.end_stage(name, best)
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
    ap.add_argument("--correction", choices=["free", "param"], default="free",
                    help="physics head: free forces at every step, or prior coefficients changed once per flight")
    ap.add_argument("--n-sub", type=int, default=1, help="RK4 sub-steps per 0.05 s step (physics head)")
    ap.add_argument("--compile", type=int, default=0, help="1: compile the physics-head step (torch.compile)")
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
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto (cuda if a GPU is visible)")
    ap.add_argument("--checkpoint-min", type=float, default=15.0,
                    help="minutes between saves of the training state (resume.pt); 0 = off")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(a.device)
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
    val_seq = torch.from_numpy(S_va.gather(vrows)).to(dev)
    val_ac = torch.from_numpy(S_va.aircraft(vrows)).to(dev)
    print(f"flights: {tr.sum()} training, {va.sum()} validation; windows: {len(S_tr)} / {len(S_va)}; "
          f"device {dev}", flush=True)

    model = model_from_config(dict(vars(a), norm=norm.to_dict())).to(dev)
    if a.compile and physics:
        model.decoder.compile_step()
    # config written now as well (rewritten with the results at the end): a run stopped by the time
    # limit keeps its best weights and can still be evaluated
    json.dump(dict(vars(a), features=FEATURES, n_params=model.n_params(), norm=norm.to_dict()),
              open(out / "config.json", "w"), indent=1)
    if trunk_cfg:
        trunk = model_from_config(trunk_cfg)
        trunk.load_state_dict(torch.load(Path(a.trunk) / "model.pt", map_location="cpu"))
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
        return (torch.from_numpy(S_tr.gather(rows)[:, :H + T]).to(dev),
                torch.from_numpy(S_tr.aircraft(rows)).to(dev))

    def state_err(pred, seq):
        return ((pred - seq[:, H:, :N_STATE]) ** 2).mean()

    def state_loss(seq, ac, frozen=False):
        """Normalized state error and size of the physics correction (0 for the MLP)."""
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            z0, zs, U = model.latents(seq, H, True, ac)
        pred, raw = model.head_states(z0, zs, U, seq, H, ac)
        res = (raw ** 2).mean() if raw is not None else pred.new_zeros(())
        return state_err(pred, seq), res

    val_state = lambda: state_loss(val_seq, val_ac)[0]
    log = []
    ckpt = lambda: torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")

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
    # what must match for a restart: all the options but the machine-side ones
    key = json.dumps({k: v for k, v in vars(a).items() if k not in ("threads", "device", "compile", "checkpoint_min")},
                     sort_keys=True, default=str)
    resume = Resume(out / "resume.pt", a.checkpoint_min, key, [model] + ([target_enc] if target_enc else []), S_tr, log)

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

        resumed_in_stage2 = resume.finished("jepa") is not None       # the weights are already those of stage 2
        best1 = run_stage("jepa", trunk_params, loss1, val1, a.steps1, a, log, ckpt, resume, after_step=after)
        if not resumed_in_stage2:
            model.load_state_dict(torch.load(out / "model.pt"))
        if a.mode == "jepa":
            for p_ in trunk_params:
                p_.requires_grad_(False)                              # stage 2: trunk frozen

            def loss2(step):
                return state_loss(*nxt(T_max), frozen=True)[0], {}

            best2 = run_stage("decodeur", list(model.decoder.parameters()), loss2, val_state, a.steps2, a, log, ckpt,
                              resume)
        else:                                                         # stage 2: fine-tune everything
            def loss2(step):
                return state_loss(*nxt(T_max))[0], {}

            best2 = run_stage("finetune", list(model.parameters()), loss2, val_state, a.steps2, a, log, ckpt,
                              resume, peak=a.ft_lr)
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
                         resume, after_step=after)
        summary = dict(best_val_state=best)
    elif a.mode == "c_jepa":
        for p_ in trunk_params:
            p_.requires_grad_(False)

        def loss_cj(step):
            T = T_at(step, a.steps2)
            ls, res = state_loss(*nxt(T), frozen=True)
            return ls + a.lambda_res * res, dict(state=ls.item(), res=res.item(), T=T)

        best = run_stage("tete_physique", list(model.decoder.parameters()), loss_cj, val_state, a.steps2,
                         a, log, ckpt, resume)
        summary = dict(best_val_state=best)
    else:                                                             # a0, c_direct
        steps = a.steps1 + a.steps2

        def loss_direct(step):
            T = T_at(step, steps)
            ls, res = state_loss(*nxt(T))
            return ls + a.lambda_res * res, dict(state=ls.item(), res=res.item(), T=T)

        best = run_stage(a.mode, list(model.parameters()), loss_direct, val_state, steps, a, log, ckpt, resume)
        summary = dict(best_val_state=best)

    cfg = dict(vars(a), features=FEATURES, n_params=model.n_params(), norm=norm.to_dict(), **summary)
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    json.dump(log, open(out / "log.json", "w"))
    resume.clear()
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
