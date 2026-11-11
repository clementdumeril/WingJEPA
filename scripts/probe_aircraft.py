"""Can the aircraft be identified? Probes that read the aircraft parameters of each flight back from
(a) the latent z_t of each trained world model and (b) the raw history of measured states and commands,
for several history lengths.

Every flight has its own aircraft (mass, inertias, motor power, CG, six aerodynamic scale factors). A physics
head can only produce the right forces if this aircraft is identified from the past window, so a latent from
which the parameters cannot be read is a likely cause of a weak physics probe. Probes are fitted on windows of
the training flights and scored (R^2 per parameter) on the long test flights, training-range and extreme aircraft
separately. Two readers: ridge regression (linear) and a small MLP.

  latents   z_t of each model, from its own context window (H steps), as fed to its head
  latents+1 latent z_{t+1} after the first predictor step, the first place where the past commands (action
            encoder of the predictor) meet the state latent (the encoder alone reads the states only, unless
            the run has --action-history 1 or 2)
  history   flattened window of the last L steps of the 9 states + 4 commands, L in --hist-s; tells how much a
            longer context could help, independently of any model. Also with the 9 states only: tells how much
            of the identification needs the commands

Writes <out>.json and prints the R^2 tables. Usage:
  python scripts/probe_aircraft.py --data-dir data/data_calm --models runs/a0_calm_sky_s0 runs/jepa_calm_sky_s0 \
      runs/c_direct_calm_sky_nul_s0 --out results/probe_aircraft
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_model  # noqa: E402
from wingjepa.data.windows import N_CTRL, N_STATE, ORACLE, aircraft_params, features, load_set, window_starts  # noqa: E402

DT = 0.05
TARGETS = ["mass", "Ixx", "Iyy", "Izz"] + ORACLE


def targets(meta):
    return np.stack([meta[k] for k in TARGETS], 1).astype(np.float64)


def pick(alive, L, n_max, rng, stride=1):
    """(flight, start) of fully valid windows of L steps, at most n_max, starts on a stride."""
    st = window_starts(alive, L)
    st = st[st[:, 1] % stride == 0]
    if len(st) > n_max:
        st = st[np.sort(rng.choice(len(st), n_max, replace=False))]
    return st


def windows(Xn, st, L, end):
    """Windows of the L steps ending at step `end` of each (flight, start) (start + end - L + 1 .. start + end)."""
    idx = st[:, 1:2] + end - L + 1 + np.arange(L)
    return Xn[st[:, :1], idx]


def ridge(Ftr, Ytr, lam=1e-2):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    A = np.c_[(Ftr - mu) / sd, np.ones(len(Ftr))]
    W = np.linalg.solve(A.T @ A + lam * len(A) * np.eye(A.shape[1]), A.T @ Ytr)
    return lambda F: np.c_[(F - mu) / sd, np.ones(len(F))] @ W


def mlp(Ftr, Ytr, steps=3000, seed=0):
    torch.manual_seed(seed)
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
    X = torch.tensor((Ftr - mu) / sd, dtype=torch.float32)
    Y = torch.tensor(Ytr, dtype=torch.float32)
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                        nn.Linear(128, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)
    g = torch.Generator().manual_seed(seed)
    for _ in range(steps):
        i = torch.randint(0, len(X), (1024,), generator=g)
        loss = ((net(X[i]) - Y[i]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    @torch.no_grad()
    def f(F):
        return net(torch.tensor((F - mu) / sd, dtype=torch.float32)).numpy().astype(np.float64)
    return f


def r2(pred, y):
    """R^2 per parameter, against the variance of the scored group."""
    return 1 - ((pred - y) ** 2).mean(0) / np.maximum(y.var(0), 1e-12)


def score(F_fit, Y_fit, groups, steps):
    """{reader: {group: {param: R^2}}}; Y standardized on the fit set."""
    ym, ys = Y_fit.mean(0), Y_fit.std(0) + 1e-12
    out = {}
    for name, fit in (("ridge", ridge), ("mlp", lambda F, Y: mlp(F, Y, steps))):
        f = fit(F_fit, (Y_fit - ym) / ys)
        out[name] = {g: dict(zip(TARGETS, r2(f(F), (Y - ym) / ys).round(3).tolist())) for g, (F, Y) in groups.items()}
    return out


@torch.no_grad()
def encode(model, xw, ac, batch=4096):
    z = [model.encode(torch.from_numpy(xw[i:i + batch]), None if ac is None else torch.from_numpy(ac[i:i + batch]))
         for i in range(0, len(xw), batch)]
    return torch.cat(z).numpy().astype(np.float64)


@torch.no_grad()
def step1(model, seq, H, ac, batch=4096):
    """Latent after the first predictor step, z_{t+1}: seq (n, H + 1, 16) normalized."""
    z = []
    for i in range(0, len(seq), batch):
        a = None if ac is None else torch.from_numpy(ac[i:i + batch])
        z.append(model.latents(torch.from_numpy(seq[i:i + batch]), H, True, a)[1][:, 0])
    return torch.cat(z).numpy().astype(np.float64)


def show(title, res):
    print(f"\n== {title}")
    for reader, gr in res.items():
        for g, d in gr.items():
            aero = np.mean([d[f"aero_scale_{k}"] for k in ("CD", "CY", "CL", "Cl", "Cm", "Cn")])
            print(f"   {reader:5s} {g:20s} mass {d['mass']:6.2f}  Iyy {d['Iyy']:6.2f}  motor {d['power_scale']:6.2f}"
                  f"  CG {d['cg_shift']:6.2f}  CL {d['aero_scale_CL']:6.2f}  Cm {d['aero_scale_Cm']:6.2f}"
                  f"  aero mean {aero:6.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="*", default=[])
    ap.add_argument("--hist-s", type=float, nargs="*", default=[0.55, 2.0, 5.0])
    ap.add_argument("--n-fit", type=int, default=40000, help="training windows used to fit the probes")
    ap.add_argument("--steps", type=int, default=3000, help="MLP probe steps")
    ap.add_argument("--out", default="results/probe_aircraft")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "4")))
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    rng = np.random.default_rng(0)

    data, ch, meta_tr = load_set(a.data_dir, "train")
    Xtr, al_tr, _ = features(data, ch)
    data, ch, meta_te = load_set(a.data_dir, "test_long")
    Xte, al_te, _ = features(data, ch)
    del data
    Ytr, Yte = targets(meta_tr), targets(meta_te)
    test = meta_te["is_test_aircraft"].astype(bool)
    Lmax = max([int(round(s / DT)) for s in a.hist_s] + [1])
    models = [(d, *load_model(d)) for d in a.models]
    Lmax = max([Lmax] + [cfg["H"] for _, _, cfg, _ in models])
    # same windows for every reader: the longest history ending at the same step
    # one step more than the context: the command of the last context step, for the first predictor step
    st_fit = pick(al_tr, Lmax + 1, a.n_fit, rng)
    st_te = pick(al_te, Lmax + 1, 10 ** 9, rng, stride=int(1 / DT))
    g_te = {"avions_entrainement": ~test[st_te[:, 0]], "avions_de_test": test[st_te[:, 0]]}
    end = Lmax - 1
    print(f"fit windows {len(st_fit)}, test windows {g_te['avions_entrainement'].sum()} training-range, "
          f"{g_te['avions_de_test'].sum()} extreme", flush=True)
    results = dict(targets=TARGETS, n_fit=len(st_fit), history={}, latents={}, latents_after_step={}, context={})

    norm0 = models[0][3] if models else None
    for s in a.hist_s:
        L = int(round(s / DT))
        for nc, suffix, what in ((N_STATE + N_CTRL, "", "9 states + 4 commands"), (N_STATE, "_states", "9 states only")):
            def feats(X, st):
                xw = windows(X, st, L, end)[:, :, :nc]
                if norm0 is not None:
                    xw = (xw - norm0.mean[:nc]) / norm0.std[:nc]
                return xw.reshape(len(st), -1).astype(np.float64)
            F_fit = feats(Xtr, st_fit)
            F_te = feats(Xte, st_te)
            groups = {g: (F_te[m], Yte[st_te[m, 0]]) for g, m in g_te.items()}
            res = score(F_fit, Ytr[st_fit[:, 0]], groups, a.steps)
            results["history"][f"{s:g}s{suffix}"] = res
            show(f"raw history, {s:g} s ({L} steps of {what})", res)

    for d, model, cfg, norm in models:
        if not getattr(model, "latent", True):                       # mlp_direct: no latent to probe
            print(f"\n== {Path(d).name}: no latent, skipped")
            continue
        H = cfg["H"]
        uses_ac = bool(cfg.get("aircraft_input"))
        def lat(X, st, meta):
            xw = norm(windows(X, st, H, end)).astype(np.float32)
            ac = np.stack([meta[k] for k in ("mass", "Ixx", "Iyy", "Izz", "Ixz")], 1)[st[:, 0]].astype(np.float32)
            return encode(model, xw, ac if uses_ac else None)
        F_fit, F_te = lat(Xtr, st_fit, meta_tr), lat(Xte, st_te, meta_te)
        groups = {g: (F_te[m], Yte[st_te[m, 0]]) for g, m in g_te.items()}
        res = score(F_fit, Ytr[st_fit[:, 0]], groups, a.steps)
        results["latents"][Path(d).name] = res
        show(f"latent z_t of {Path(d).name} ({H} steps of context)", res)

        def lat1(X, st, meta):
            seq = norm(windows(X, st, H + 1, end + 1)).astype(np.float32)
            ac = aircraft_params(meta, oracle=True)[st[:, 0]] if uses_ac else None
            return step1(model, seq, H, ac)
        F_fit, F_te = lat1(Xtr, st_fit, meta_tr), lat1(Xte, st_te, meta_te)
        groups = {g: (F_te[m], Yte[st_te[m, 0]]) for g, m in g_te.items()}
        res = score(F_fit, Ytr[st_fit[:, 0]], groups, a.steps)
        results["latents_after_step"][Path(d).name] = res
        show(f"latent z_t+1 of {Path(d).name} (after the first predictor step, past commands included)", res)

        if getattr(model, "context_enc", None) is not None:          # context latent c, read from the same window
            def ctx(X, st):
                xw = norm(windows(X, st, H, end)).astype(np.float32)
                with torch.no_grad():
                    c = [model.context_of(torch.from_numpy(xw[i:i + 4096])) for i in range(0, len(xw), 4096)]
                return torch.cat(c).numpy().astype(np.float64)
            F_fit, F_te = ctx(Xtr, st_fit), ctx(Xte, st_te)
            groups = {g: (F_te[m], Yte[st_te[m, 0]]) for g, m in g_te.items()}
            res = score(F_fit, Ytr[st_fit[:, 0]], groups, a.steps)
            results["context"][Path(d).name] = res
            show(f"context latent c of {Path(d).name} ({model.context_dim} values, {H} steps of states and commands)",
                 res)

    out = Path(a.out).with_suffix(".json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    print("written", out)


if __name__ == "__main__":
    main()
