"""Training of the world models of the SkyJEPA test suite: JEPA (model A), the A0 reference (no JEPA), the physics
head (model C) and the plain MLP baseline.

  --mode jepa    : stage 1 = encoder + predictor, JEPA loss + lambda * SIGReg (predicted latents);
                   stage 2 = decoder alone (encoder and predictor frozen), state error.
  --mode a0      : same network trained in one block on the state error (direct prediction, no JEPA).
  --mode mlp_direct: plain MLP dynamics model (Nagabandi et al. 2018, models/mlp_direct.py): flattened window of
                   the last H states and commands + next command and wind -> state increment, rolled out
                   autoregressively; same loss, optimizer, schedule and data as a0. Widths: --mlp-hidden.
  --mode c_direct: like a0, with the physics head (--prior) instead of the MLP decoder; loss = state
                   error + --lambda-res * size of the learned correction (APHYNITY idea).
  --mode c_jepa  : trunk (encoder + predictor) of a jepa or a0 run (--trunk) frozen, only the physics head is
                   learned (like the SkyJEPA physics probe), same loss as c_direct.
  --lambda-phys L (a0): physics in the loss instead of the architecture (Pred + Phys Reg in SkyJEPA): the MLP
                   predictions are also penalized, with weight L, when they violate the kinematics known for
                   any aircraft (attitude kinematics from p, q, r; vertical speed from Va, alpha, beta, attitude).
  --input-noise S: noise augmentation: Gaussian noise of sigma * std (normalized units), sigma drawn uniformly
                   in [0, S] for each window, added to the past states (the H context steps) of every training
                   batch; the future states (targets) and the validation batches stay clean. Same noise model
                   as the robustness tests, so S = 0.5 covers all the tested levels.
--action-encoder 1: as in SkyJEPA (eq. 7-8), the predictor receives at each step the encoding of the
                   commands applied up to that step (TCN of --act-channels) instead of the raw command; the
                   state latent and the JEPA targets stay states only. A c_jepa run takes this from its trunk.
Training horizon: from --T steps at the start up to --T-max steps (linear curriculum reached at
--curriculum-frac of the training); validation always uses --T-max.
The best weights on the validation flights are kept.

Options of the earlier Ultra Stick studies (other modes, constrained and port-Hamiltonian heads, command-history
encoders, context latent, EMA target encoder...) are in the tag ultrastick-final; config.json still records them
at their "off" value (REMOVED below), so that runs started before their removal resume and read the same.

v2 inputs by default (2 s of past, 5-layer encoder, mass and inertias as input). The SkyJEPA suite uses
--H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 --action-encoder 1.

Restart: the training state is saved every --checkpoint-min minutes in <out>/resume.pt. A run stopped by the
time limit, resubmitted with the same command, continues where it stopped (same steps, same random
draws); the file is deleted when the run finishes. --device cuda trains on a GPU (needs a CUDA build of
torch, cluster/setup_gpu_env.sh); model.pt is always saved on the CPU.

Usage: python scripts/train_a.py --data-dir data/data_advanced --out runs/a0_adv_s0 --mode a0
       python scripts/train_a.py ... --mode c_direct --prior configs/priors/advanced_plane_nul.json
       python scripts/train_a.py ... --mode c_jepa --prior configs/priors/advanced_plane_nul.json --trunk runs/jepa_adv_s0
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
from wingjepa.data.windows import (FEATURES, N_CTRL, N_STATE, Normalizer, WindowSampler, aircraft_params,  # noqa: E402
                                   features, load_set, split_mask)
from wingjepa.models.jepa import model_from_config, sigreg  # noqa: E402

MODES = ["jepa", "a0", "c_direct", "c_jepa", "mlp_direct"]
# options removed with the earlier studies, at the value they had in every run of this pipeline: still written in
# config.json and in the restart key, so both stay identical to those of the runs started before the removal
REMOVED = dict(action_history=0, target_window=0, context_dim=0, context_channels="8,8,16", id_jepa=0.0,
               id_gap_s=2.0, lambda_state=1.0, oracle_params=0, correction="free", param_bound=0.0,
               head_kind="physics", ema=0.0, ft_lr=1e-3)


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


def run_stage(name, params, loss_fn, val_fn, steps, a, log, save_best, resume):
    if resume.finished(name) is not None:           # completed before the restart
        return resume.finished(name)
    peak = a.lr
    opt = torch.optim.Adam(params, lr=peak, weight_decay=a.weight_decay)
    start, best, skipped, sec = resume.restore(name, opt)
    t0, streak = time.time() - sec, 0
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
            streak = 0
        else:
            skipped, streak = skipped + 1, streak + 1
            # every batch non-finite: the weights no longer move, the run is dead (stop instead of idling)
            if streak >= a.max_skip_streak:
                sys.exit(f"{name}: {streak} batches in a row with a non-finite loss or gradient at step {step}, "
                         f"training diverged (best validation {best:.4g}, kept in model.pt)")
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


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--set", default="train")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=MODES, default="jepa")
    ap.add_argument("--H", type=int, default=40, help="past steps (0.05 s)")
    ap.add_argument("--channels", default="16,16,32,32,32", help="channels of the encoder layers")
    ap.add_argument("--dilations", default="1,2,4,8,16", help="receptive field = 1 + 2 * sum")
    ap.add_argument("--aircraft-input", type=int, default=1, help="mass and inertias as input (1) or not (0)")
    ap.add_argument("--action-encoder", type=int, default=0,
                    help="1: the predictor reads the commands through an action encoder (SkyJEPA's Enc_phi)")
    ap.add_argument("--act-channels", default="4,4,8", help="channels of the action encoder layers")
    ap.add_argument("--T", type=int, default=20, help="training horizon at the start (steps of 0.05 s)")
    ap.add_argument("--T-max", type=int, default=None, help="final horizon (default: --T, no curriculum)")
    ap.add_argument("--curriculum-frac", type=float, default=0.5)
    ap.add_argument("--prior", default=None, help="prior of the physics head (configs/priors/*.json)")
    ap.add_argument("--trunk", default=None, help="jepa or a0 run whose frozen trunk is reused (c_jepa mode)")
    ap.add_argument("--lambda-res", type=float, default=1e-3, help="penalty on the size of the correction")
    ap.add_argument("--lambda-phys", type=float, default=0.0,
                    help="physics in the loss (a0 mode, MLP head): weight of the kinematic residuals of the "
                         "predicted states (attitude kinematics, vertical speed), like Pred + Phys Reg in SkyJEPA")
    ap.add_argument("--input-noise", type=float, default=0.0,
                    help="noise augmentation: max std (normalized units) of the noise added to the past states")
    ap.add_argument("--out-scale", type=float, default=0.1, help="scale of the correction (coefficients)")
    ap.add_argument("--max-skip-streak", type=int, default=500,
                    help="stop when this many batches in a row have a non-finite loss or gradient")
    ap.add_argument("--n-sub", type=int, default=1, help="RK4 sub-steps per 0.05 s step (physics head)")
    ap.add_argument("--compile", type=int, default=0, help="1: compile the physics-head step (torch.compile)")
    ap.add_argument("--mlp-hidden", default="50,50",
                    help="hidden widths of the mlp_direct model (50,50: about the a0 size at H = 11)")
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
    ap.add_argument("--device", default="cpu", help="cpu, cuda, or auto (cuda if a GPU is visible)")
    ap.add_argument("--checkpoint-min", type=float, default=15.0,
                    help="minutes between saves of the training state (resume.pt); 0 = off")
    ap.set_defaults(**REMOVED)                      # no command-line flag: fixed values, see REMOVED
    return ap


def resolve(ap, a):
    """Options derived from the command line, in place: head, prior (file name and loaded content) and, for c_jepa,
    the architecture of the trunk. Returns the trunk config (None outside c_jepa)."""
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
        a.action_encoder = trunk_cfg.get("action_encoder", 0)
        a.act_channels = trunk_cfg.get("act_channels", a.act_channels)
    return trunk_cfg


def main(argv=None):
    ap = build_parser()
    a = ap.parse_args(argv)

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    torch.set_num_threads(a.threads)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(a.device)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    trunk_cfg = resolve(ap, a)
    physics = a.head == "physics"

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
          f"encoder receptive field {getattr(model, 'encoder', model).receptive_field} steps for H = {a.H}",
          flush=True)
    H = a.H

    def T_at(step, steps):
        """Curriculum: from a.T to T_max steps, linearly, reached at curriculum_frac of the stage."""
        x = min(1.0, step / max(1, a.curriculum_frac * steps))
        return int(round(a.T + (T_max - a.T) * x))

    def nxt(T, keep_clean=False):
        """Batch of windows (B, H+T, 16) and aircraft parameters. keep_clean: also the windows without the
        noise augmentation (JEPA stage: the targets encode the clean future, as in a denoising JEPA)."""
        rows = S_tr.sample_rows(a.batch)
        seq = torch.from_numpy(S_tr.gather(rows)[:, :H + T]).to(dev)
        clean = seq
        if a.input_noise > 0:                     # noise augmentation of the past states only
            clean = seq.clone()
            sig = a.input_noise * torch.rand(seq.shape[0], 1, 1, device=dev)      # level drawn per window
            seq[:, :H, :N_STATE] += sig * torch.randn_like(seq[:, :H, :N_STATE])
        ac = torch.from_numpy(S_tr.aircraft(rows)).to(dev)
        return (seq, ac, clean) if keep_clean else (seq, ac)

    def state_err(pred, seq):
        return ((pred - seq[:, H:, :N_STATE]) ** 2).mean()

    def state_loss(seq, ac, frozen=False):
        """Normalized state error and size of the physics correction (0 for the MLP)."""
        if not getattr(model, "latent", True):                       # mlp_direct: direct state rollout
            pred = model.predict_states(seq, H, True, ac)[0]
            return state_err(pred, seq), pred.new_zeros(()), pred
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen):
            z0, zs, U = model.latents(seq, H, True, ac)
        pred, raw = model.head_states(z0, zs, U, seq, H, ac)
        res = (raw ** 2).mean() if raw is not None else pred.new_zeros(())
        return state_err(pred, seq), res, pred

    mean_t, std_t = (torch.as_tensor(v, device=dev) for v in (norm.mean, norm.std))
    i_wd = N_STATE + N_CTRL + 2                                       # wind_down among the 16 quantities

    def phys_residual(pred, seq, dt=0.05):
        """Physics in the loss: how much the predicted states (B, T, 9, normalized) violate the kinematics,
        known exactly whatever the aircraft: attitude kinematics phi' = p + (q sin phi + r cos phi) tan theta,
        theta' = q cos phi - r sin phi (finite differences between consecutive predictions, trapezoidal rule,
        from the last observed state), and the vertical speed given by Va, alpha, beta and the attitude.
        Each residual is divided by the spread of the matching quantity (p, q, vertical speed)."""
        s = torch.cat([seq[:, H - 1:H, :N_STATE], pred], 1) * std_t[:N_STATE] + mean_t[:N_STATE]
        Va, al, be, p, q, r, ph, th, hd = s.unbind(-1)
        th = th.clamp(-1.4, 1.4)
        sf, cf, st, ct = torch.sin(ph), torch.cos(ph), torch.sin(th), torch.cos(th)
        f_ph, f_th = p + (q * sf + r * cf) * torch.tan(th), q * cf - r * sf
        d = ph[:, 1:] - ph[:, :-1]
        r_ph = torch.atan2(torch.sin(d), torch.cos(d)) / dt - 0.5 * (f_ph[:, 1:] + f_ph[:, :-1])
        r_th = (th[:, 1:] - th[:, :-1]) / dt - 0.5 * (f_th[:, 1:] + f_th[:, :-1])
        u, v, w = Va * torch.cos(al) * torch.cos(be), Va * torch.sin(be), Va * torch.sin(al) * torch.cos(be)
        wd = seq[:, H:, i_wd] * std_t[i_wd] + mean_t[i_wd]
        r_hd = hd[:, 1:] - ((u * st - v * sf * ct - w * cf * ct)[:, 1:] - wd)        # predicted steps only
        return ((r_ph / std_t[3]) ** 2 + (r_th / std_t[4]) ** 2 + (r_hd / std_t[8]) ** 2).mean() / 3

    if a.lambda_phys > 0:      # check: the true states satisfy the kinematics (up to the finite differences)
        print(f"physics residual of the true validation states: "
              f"{phys_residual(val_seq[:, H:, :N_STATE], val_seq).item():.3g}", flush=True)

    val_state = lambda: state_loss(val_seq, val_ac)[0]
    log = []
    ckpt = lambda: torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")

    # what must match for a restart: all the options but the machine-side ones; options added after
    # resume.pt existed are left out at their default, so that the runs started before can still resume
    machine = ("threads", "device", "compile", "checkpoint_min", "max_skip_streak")
    later = dict(param_bound=0.0, action_history=0, target_window=0, action_encoder=0, act_channels="4,4,8",
                 lambda_phys=0.0, input_noise=0.0, oracle_params=0, head_kind="physics", mlp_hidden="50,50",
                 context_dim=0, context_channels="8,8,16", id_jepa=0.0, id_gap_s=2.0)
    key = json.dumps({k: v for k, v in vars(a).items() if k not in machine and later.get(k, object()) != v},
                     sort_keys=True, default=str)
    resume = Resume(out / "resume.pt", a.checkpoint_min, key, [model], S_tr, log)

    def jepa_loss(seq, ac, clean=None):
        """clean: windows without the noise augmentation, encoded as targets (default seq itself)."""
        z0, zp, U = model.latents(seq, H, True, ac)
        zt = model.target_latents(seq if clean is None else clean, H, ac)
        pred = ((zp - zt) ** 2).mean()
        # SIGReg on the predicted latents, averaged over time: estimated on a few randomly drawn steps
        ks = torch.randperm(zp.shape[1])[:a.sigreg_steps]
        reg = torch.stack([sigreg(zp[:, k], n_proj=a.sigreg_proj) for k in ks]).mean()
        return pred + a.lambda_sig * reg, (z0, zp, U), dict(pred=pred.item(), sigreg=reg.item())

    trunk_params = (list(model.encoder.parameters()) + list(model.predictor.parameters())
                    if getattr(model, "latent", True) else [])
    if a.mode == "jepa":
        def loss1(step):
            T = T_at(step, a.steps1)
            loss, _, parts = jepa_loss(*nxt(T, keep_clean=True))
            return loss, dict(parts, T=T)

        def val1():
            # prediction error normalized by the spread of the target latents: independent of the
            # batch size, and penalizes collapse (if all the z look alike, it blows up)
            zp = model.predict_latents(val_seq, H, True, val_ac)
            zt = model.target_latents(val_seq, H, val_ac)
            return ((zp - zt) ** 2).mean() / zt.reshape(-1, zt.shape[-1]).var(0).mean()

        resumed_in_stage2 = resume.finished("jepa") is not None       # the weights are already those of stage 2
        best1 = run_stage("jepa", trunk_params, loss1, val1, a.steps1, a, log, ckpt, resume)
        if not resumed_in_stage2:
            model.load_state_dict(torch.load(out / "model.pt"))
        for p_ in trunk_params:
            p_.requires_grad_(False)                                  # stage 2: trunk frozen

        def loss2(step):
            return state_loss(*nxt(T_max), frozen=True)[0], {}

        best2 = run_stage("decodeur", list(model.decoder.parameters()), loss2, val_state, a.steps2, a, log, ckpt,
                          resume)
        summary = dict(best_val_jepa=best1, best_val_state=best2)
    elif a.mode == "c_jepa":
        for p_ in trunk_params:
            p_.requires_grad_(False)

        def loss_cj(step):
            T = T_at(step, a.steps2)
            ls, res, _ = state_loss(*nxt(T), frozen=True)
            return ls + a.lambda_res * res, dict(state=ls.item(), res=res.item(), T=T)

        best = run_stage("tete_physique", list(model.decoder.parameters()), loss_cj, val_state, a.steps2,
                         a, log, ckpt, resume)
        summary = dict(best_val_state=best)
    else:                                                             # a0, c_direct, mlp_direct
        steps = a.steps1 + a.steps2

        def loss_direct(step):
            T = T_at(step, steps)
            seq, ac = nxt(T)
            ls, res, pred = state_loss(seq, ac)
            if a.lambda_phys > 0:
                ph = phys_residual(pred, seq)
                return ls + a.lambda_res * res + a.lambda_phys * ph, dict(state=ls.item(), phys=ph.item(), T=T)
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
