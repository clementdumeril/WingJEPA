"""Closed-loop test: the world models fly the aircraft (MPPI), compared with the TECS autopilot.

Same batch of aircraft, wind, references and sensor noise for every controller (fixed seeds). Each flight:
trim, 10 s of settling under TECS, then `--duration` s where the controller tracks piecewise-constant references
of altitude, airspeed and bank angle (a new target every 10 s). Half of the aircraft from the training ranges,
half "extreme" (outside them, as in the test sets). No envelope protection: a flight that leaves the simulator
limits is lost (frozen).

MPPI (model-predictive path integral control), every 50 ms:
  - N candidate command sequences over `--horizon` steps = nominal sequence + smoothed Gaussian noise;
  - the world model predicts the states of each one from the measured past (0.55 s, noisy if --noise > 0);
  - cost: altitude (integrated predicted climb rate), climb rate towards a target, airspeed, bank, sideslip,
    envelope penalties (angle of attack, speed, attitude) and command rate;
  - new nominal = softmax-weighted mean of the candidates; its first command is applied, the rest shifted.
The physics head receives the true mass and inertia of the aircraft, as in training and evaluation.

Sensor noise --noise s: Gaussian noise of s * std (training statistics) on the 9 measured states, for every
controller (TECS included); altitude itself is measured without noise.

Usage (cluster, GPU):
  python scripts/closed_loop.py --controllers tecs runs/c_direct_calm_sky_nul_s0 runs/c_jepa_calm_sky_nul_s0 \
      runs/a0_calm_sky_s0 --out runs/closed_loop/calm_noise0.json [--noise 0.2] [--turbulence moderate --gust 3]
Writes <out>.json (metrics per controller and aircraft group) and <out>_traces.npz (time series).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import load_model  # noqa: E402
from generate_flights import sample_overrides  # noqa: E402
from wingjepa.data.windows import N_CTRL, N_STATE  # noqa: E402
from wingjepa.sim import ultrastick as us  # noqa: E402
from wingjepa.sim.pilot import TECS, lateral  # noqa: E402
from wingjepa.sim.wind import Wind  # noqa: E402

DT = 0.05
LEVELS = ("none", "light", "moderate", "severe")


def references(rng, n, K, h0, V0, seg_s=10.0):
    """Piecewise-constant targets (n, K): altitude steps of +-25 m, airspeed 17-25 m/s, bank 0 / +-20 / +-30 deg."""
    nseg = int(np.ceil(K * DT / seg_s))
    dh = rng.uniform(-25, 25, (n, nseg))
    dh[:, 0] = 0
    h_seg = h0[:, None] + np.cumsum(dh, 1)
    V_seg = rng.uniform(17, 25, (n, nseg))
    V_seg[:, 0] = V0
    phi_seg = rng.choice(np.deg2rad([0, 0, -20, 20, -30, 30]), (n, nseg))
    phi_seg[:, 0] = 0
    idx = np.minimum((np.arange(K) * DT // seg_s).astype(int), nseg - 1)
    return h_seg[:, idx], V_seg[:, idx], phi_seg[:, idx]


class Flights:
    """Batch of simulated aircraft, stepped at 20 Hz (RK4 at cfg sim.dt inside)."""

    def __init__(self, cfg, n_train, n_test, seed, duration, turbulence, gust, noise, std_state):
        self.cfg, sim = cfg, cfg["sim"]
        p = us.load_config(cfg["aircraft"]["config"])
        self.p, self.lim = p, sim["limits"]
        rng = np.random.default_rng(seed)
        n = n_train + n_test
        self.n, self.is_test = n, np.r_[np.zeros(n_train, bool), np.ones(n_test, bool)]
        ov = sample_overrides(rng, n, p, cfg["aircraft"]["ranges"], self.is_test)
        self.ac = us.UltraStick(p, n, us.FaserTables(p["aero_file"]), ov)
        self.params = np.stack([ov["mass"], ov["Ixx"], ov["Iyy"], ov["Izz"], ov["Ixz"]], 1).astype(np.float32)
        # same columns as windows.aircraft_params(meta, oracle=True); 5: only read by an oracle physics head
        extra = [ov["power_scale"], ov["cg_shift"]] + [ov["aero_scale"][k] for k in ("CD", "CY", "CL", "Cl", "Cm", "Cn")]
        extra = [(e - 1) / 0.3 for e in extra[:1]] + [extra[1] / 0.05] + [(e - 1) / 0.3 for e in extra[2:]]
        self.params = np.c_[self.params, np.stack(extra, 1)].astype(np.float32)
        self.h0, self.V0 = rng.uniform(150, 300, n), rng.uniform(17, 25, n)
        psi0 = rng.uniform(-np.pi, np.pi, n)
        self.Y, self.cmd0, self.thr0, ok = self.ac.trim(self.V0, h=self.h0, psi=psi0)
        self.warm = int(round(sim["warmup_s"] / DT))
        self.K = int(round(duration / DT))
        self.h_ref, self.V_ref, self.phi_ref = references(rng, n, self.K, self.h0, self.V0)
        wcfg = dict(cfg["wind"], turbulence_probs={k: float(k == turbulence) for k in LEVELS},
                    gust_amplitude_max=float(gust))
        self.wind = Wind(np.random.default_rng(seed + 1), n, wcfg, sim["warmup_s"] + duration + 1)
        self.rng_noise = np.random.default_rng(seed + 2)
        self.noise, self.std_state = noise, std_state
        self.sub = int(round(DT / sim["dt"]))
        self.dt_sim = sim["dt"]
        self.alive = ok.copy()
        self.turb = np.zeros((n, 3))
        self.wind_est = self.wind.steady.copy()
        self.t = 0.0
        self.h_prev = self.Y[:, us.H].copy()
        self.thr = self.thr0.copy()

    def gust_body(self, Y, t):
        Cbn = np.swapaxes(us.dcm_body_to_ned(Y[:, us.PHI], Y[:, us.TH], Y[:, us.PSI]), 1, 2)
        gn = self.wind.gust(t)
        return np.einsum("nij,nj->ni", Cbn, gn) + self.turb, gn

    def measure(self):
        """True and measured (noisy) state features (n, 9), wind features (n, 3), altitude (n,)."""
        gb, _ = self.gust_body(self.Y, self.t)
        F = self.ac.forces(self.Y, self.thr, gb)
        Y, h = self.Y, self.Y[:, us.H]
        hdot = (h - self.h_prev) / DT
        s = np.stack([F["Va"], F["alpha"], F["beta"], Y[:, us.P], Y[:, us.Q], Y[:, us.R],
                      Y[:, us.PHI], Y[:, us.TH], hdot], 1)
        psi = Y[:, us.PSI]
        we = self.wind_est
        w = np.stack([we[:, 0] * np.cos(psi) + we[:, 1] * np.sin(psi),
                      -we[:, 0] * np.sin(psi) + we[:, 1] * np.cos(psi), we[:, 2]], 1)
        s_meas = s + self.noise * self.std_state * self.rng_noise.standard_normal(s.shape)
        return s, s_meas, w, h.copy()

    def step(self, cmd, thr):
        """Apply (cmd (n, 3) rad, thr (n,)) for 50 ms; envelope exits freeze the flight."""
        lim = self.lim
        cmd = np.clip(cmd, self.p["act_min"], self.p["act_max"])
        thr = np.clip(thr, 0.0, 1.0)
        self.h_prev = self.Y[:, us.H].copy()
        self.thr = thr
        for _ in range(self.sub):
            self.turb = self.wind.step_turbulence(
                self.ac.forces(self.Y, thr, self.gust_body(self.Y, self.t)[0])["Va"], self.Y[:, us.H], self.dt_sim)
            gb, gn = self.gust_body(self.Y, self.t)
            self.wind_est = self.wind.step_estimate(self.wind.steady + gn, self.dt_sim)
            Yn = self.ac.step(self.Y, cmd, thr, gb, self.wind.steady, self.dt_sim)
            with np.errstate(all="ignore"):
                vr = Yn[:, us.U:us.W + 1] - gb
                Va = np.linalg.norm(vr, axis=1)
                a = np.arctan2(vr[:, 2], vr[:, 0])
                b = np.arcsin(np.clip(vr[:, 1] / np.maximum(Va, 1e-3), -1, 1))
                bad = (~np.isfinite(Yn).all(1) | (Va < lim["Va_min"]) | (Va > lim["Va_max"])
                       | (a > np.deg2rad(lim["alpha_max_deg"])) | (a < np.deg2rad(lim["alpha_min_deg"]))
                       | (np.abs(b) > np.deg2rad(lim["beta_max_deg"]))
                       | (np.abs(Yn[:, us.PHI]) > np.deg2rad(lim["phi_max_deg"]))
                       | (np.abs(Yn[:, us.TH]) > np.deg2rad(lim["theta_max_deg"])) | (Yn[:, us.H] < lim["h_min"]))
            self.alive &= ~bad
            self.Y = np.where(self.alive[:, None], Yn, self.Y)
            self.t += self.dt_sim
        return cmd, thr


class TecsController:
    """The data-generation autopilot (TECS + lateral pilot), on the measured states."""

    def __init__(self, fl, cfg):
        pil = cfg["pilot"]
        self.k_lat, self.g = pil["lateral"]["gains"], fl.p["g"]
        self.tecs = TECS(fl.n, pil["tecs"], thr_ff=fl.thr0, theta_ff=fl.Y[:, us.TH], de_ff=fl.cmd0[:, 0], g=self.g)
        self.cmd0 = fl.cmd0
        self.alpha_f = None

    def __call__(self, s, h, h_ref, V_ref, phi_ref, hist=None, u_prev=None):
        Va, alpha, beta, p, q, r, phi, th, hdot = s.T
        Va = np.maximum(Va, 1.0)
        self.alpha_f = alpha.copy() if self.alpha_f is None else self.alpha_f + DT / (0.5 + DT) * (alpha - self.alpha_f)
        gamma = np.arcsin(np.clip(hdot / Va, -1, 1))
        psidot = self.g * np.tan(np.clip(phi, -1.3, 1.3)) / Va
        q_ref, r_ref = psidot * np.cos(th) * np.sin(phi), psidot * np.cos(th) * np.cos(phi)
        no = np.zeros(len(Va), bool)
        de, thr = self.tecs(Va, self.alpha_f, gamma, th, q, q_ref, h, h_ref, V_ref, no, no, DT)
        da, dr = lateral(phi_ref, phi, p, r, r_ref, beta, self.k_lat, self.cmd0[:, 1], self.cmd0[:, 2])
        return np.stack([de, da, dr], 1), thr


class MPPIController:
    """MPPI with a world model (directory of a run). Batch: n aircraft x N candidates."""

    def __init__(self, run_dir, fl, a, dev):
        self.model, cfg, norm = load_model(run_dir)
        self.model.to(dev)
        self.H, self.dev = cfg["H"], dev
        if a.compile and dev.type == "cuda" and cfg.get("head") == "physics":
            try:
                self.model.decoder.compile_step()
            except Exception as e:                       # noqa: BLE001
                print(f"compile_step failed ({e}), running uncompiled", flush=True)
        self.mean = torch.tensor(norm.mean, device=dev)
        self.std = torch.tensor(norm.std, device=dev)
        self.n, self.N, self.T = fl.n, a.samples, a.horizon
        self.lo = torch.tensor(list(fl.p["act_min"]) + [0.0], device=dev, dtype=torch.float32)
        self.hi = torch.tensor(list(fl.p["act_max"]) + [1.0], device=dev, dtype=torch.float32)
        self.sigma = torch.tensor(a.sigma, device=dev, dtype=torch.float32)
        self.rho, self.lam, self.w_du = a.smooth, a.lam, a.w_du
        self.ac = torch.from_numpy(fl.params).to(dev).repeat_interleave(self.N, 0)
        self.gen = torch.Generator(device=dev).manual_seed(a.seed + 3)
        u0 = np.c_[fl.cmd0, fl.thr0].astype(np.float32)
        self.U = torch.from_numpy(u0).to(dev)[:, None].repeat(1, self.T, 1)        # nominal (n, T, 4)

    def noise(self):
        n, N, T = self.n, self.N, self.T
        e = torch.randn(n, N, T, N_CTRL, device=self.dev, generator=self.gen)
        out = [e[:, :, 0]]
        c = (1 - self.rho ** 2) ** 0.5
        for k in range(1, T):                                         # AR(1): smooth, stationary variance
            out.append(self.rho * out[-1] + c * e[:, :, k])
        eps = torch.stack(out, 2) * self.sigma
        eps[:, 0] = 0                                                 # the nominal itself is a candidate
        return eps

    @torch.no_grad()
    def __call__(self, s, h, h_ref, V_ref, phi_ref, hist, u_prev):
        """hist (n, H, 16) measured features (last row = now, its command slot overwritten here)."""
        n, N, T, H, dev = self.n, self.N, self.T, self.H, self.dev
        Uc = torch.minimum(torch.maximum(self.U[:, None] + self.noise(), self.lo), self.hi)     # (n, N, T, 4)
        past = torch.from_numpy(hist[:, -H:].astype(np.float32)).to(dev)
        seq = torch.zeros(n, N, H + T, past.shape[-1], device=dev)
        seq[:, :, :H] = past[:, None]
        seq[:, :, H:, N_STATE + N_CTRL:] = past[:, None, -1:, N_STATE + N_CTRL:]               # wind held
        seq[:, :, H - 1:H + T - 1, N_STATE:N_STATE + N_CTRL] = Uc
        seq[:, :, H + T - 1, N_STATE:N_STATE + N_CTRL] = Uc[:, :, -1]
        seqn = ((seq - self.mean) / self.std).reshape(n * N, H + T, -1)
        S = self.model.predict_states(seqn, H, True, self.ac)[0] * self.std[:N_STATE] + self.mean[:N_STATE]
        Va, al, be, p, q, r, ph, th, hd = S.reshape(n, N, T, N_STATE).unbind(-1)
        tt = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=dev)[:, None, None]
        hp = tt(h) + torch.cumsum(hd, -1) * DT
        href, vref, pref = tt(h_ref), tt(V_ref), tt(phi_ref)
        hd_ref = torch.clamp(0.25 * (href - hp), -2.5, 2.5)
        relu = torch.relu
        J = (((hp - href).clamp(-15, 15) / 5) ** 2 + (hd - hd_ref) ** 2 + (Va - vref) ** 2
             + ((ph - pref) / 0.1) ** 2 + (be / 0.05) ** 2
             + 50 * (relu(al - 0.19) / 0.02) ** 2 + 50 * relu(14.0 - Va) ** 2
             + 50 * (relu(ph.abs() - 0.9) / 0.05) ** 2 + 50 * (relu(th.abs() - 0.6) / 0.05) ** 2).mean(-1)
        up = torch.from_numpy(u_prev.astype(np.float32)).to(dev)[:, None, None]
        dU = torch.diff(torch.cat([up.expand(-1, N, -1, -1), Uc], 2), dim=2) / self.sigma
        J = J + self.w_du * (dU ** 2).mean((-1, -2))
        J = torch.nan_to_num(J, nan=1e9, posinf=1e9)
        Jmin = J.min(1, keepdim=True).values
        temp = self.lam * (J.median(1, keepdim=True).values - Jmin) + 1e-8
        w = torch.softmax(-(J - Jmin) / temp, 1)                                                 # (n, N)
        Unew = (w[:, :, None, None] * Uc).sum(1)
        self.U = torch.cat([Unew[:, 1:], Unew[:, -1:]], 1)
        u = Unew[:, 0].cpu().numpy().astype(np.float64)
        return u[:, :3], u[:, 3]


def run(name, a, cfg, std_state, H_hist, dev):
    fl = Flights(cfg, a.n_train, a.n_test, a.seed, a.duration, a.turbulence, a.gust, a.noise, std_state)
    warm_ctrl = TecsController(fl, cfg)
    ctrl = warm_ctrl if name == "tecs" else MPPIController(name, fl, a, dev)
    n, K = fl.n, fl.K
    hist = np.zeros((n, H_hist, 16))
    u_prev = np.c_[fl.cmd0, fl.thr0]
    keys = ["h", "h_ref", "Va", "V_ref", "phi", "phi_ref", "alpha", "beta", "de", "da", "dr", "thr"]
    tr = {k: np.full((n, K), np.nan, np.float32) for k in keys}
    alive_t = np.zeros((n, K), bool)
    t0 = time.time()
    for kk in range(fl.warm + K):
        k = kk - fl.warm
        s, s_meas, w, h = fl.measure()
        hist = np.roll(hist, -1, 1)
        hist[:, -1, :N_STATE], hist[:, -1, N_STATE + N_CTRL:] = s_meas, w
        hist[:, -1, N_STATE:N_STATE + N_CTRL] = u_prev                       # placeholder until chosen
        if k < 0:
            refs = (fl.h0, fl.V0, np.zeros(n))
            cmd, thr = warm_ctrl(s_meas, h, *refs)
        else:
            refs = (fl.h_ref[:, k], fl.V_ref[:, k], fl.phi_ref[:, k])
            if name == "tecs":
                cmd, thr = ctrl(s_meas, h, *refs)
            else:
                cmd, thr = ctrl(s_meas, h, *refs, hist, u_prev)
        cmd, thr = fl.step(cmd, thr)
        u_prev = np.c_[cmd, thr]
        hist[:, -1, N_STATE:N_STATE + N_CTRL] = u_prev
        if k >= 0:
            vals = dict(h=h, h_ref=refs[0], Va=s[:, 0], V_ref=refs[1], phi=s[:, 6], phi_ref=refs[2], alpha=s[:, 1],
                        beta=s[:, 2], de=cmd[:, 0], da=cmd[:, 1], dr=cmd[:, 2], thr=thr)
            for key in keys:
                tr[key][:, k] = vals[key]
            alive_t[:, k] = fl.alive
            if k % 200 == 0:
                print(f"  {name}: t = {k * DT:5.1f} s, alive {fl.alive.mean():.2f}, {time.time() - t0:.0f} s",
                      flush=True)
    return fl, tr, alive_t


def metrics(fl, tr, alive_t):
    out = {}
    for g, m in (("avions_entrainement", ~fl.is_test), ("avions_de_test", fl.is_test)):
        al = alive_t[m]
        def rms(x, y, scale=1.0):
            e = np.where(al, (tr[x][m] - tr[y][m]) * scale, np.nan)
            return float(np.nanmean(np.sqrt(np.nanmean(e ** 2, 1))))
        du = np.abs(np.diff(np.stack([tr[c][m] for c in ("de", "da", "dr", "thr")], -1), axis=1))
        out[g] = dict(n=int(m.sum()), survival=float(al[:, -1].mean()), time_alive_s=float(al.sum(1).mean() * DT),
                      rms_h_m=rms("h", "h_ref"), rms_V_ms=rms("Va", "V_ref"),
                      rms_phi_deg=rms("phi", "phi_ref", 180 / np.pi),
                      mean_abs_dcmd=float(np.nanmean(np.where(al[:, 1:, None], du, np.nan))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controllers", nargs="+", required=True, help="'tecs' and/or run directories")
    ap.add_argument("--config", default="configs/data_calm.yaml")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-train", type=int, default=20, help="aircraft from the training ranges")
    ap.add_argument("--n-test", type=int, default=20, help="extreme aircraft")
    ap.add_argument("--duration", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--turbulence", default="none", choices=LEVELS)
    ap.add_argument("--gust", type=float, default=0.0, help="max gust amplitude per axis (m/s)")
    ap.add_argument("--noise", type=float, default=0.0, help="sensor noise, in std of each state")
    ap.add_argument("--horizon", type=int, default=30, help="MPPI horizon (steps of 50 ms)")
    ap.add_argument("--samples", type=int, default=256)
    ap.add_argument("--sigma", type=float, nargs=4, default=[0.04, 0.04, 0.03, 0.08],
                    help="exploration noise: elevator, aileron, rudder (rad), throttle")
    ap.add_argument("--smooth", type=float, default=0.8, help="AR(1) correlation of the noise between steps")
    ap.add_argument("--lam", type=float, default=0.1, help="temperature, relative to median - min cost")
    ap.add_argument("--w-du", type=float, default=0.05, help="weight of the command-rate cost")
    ap.add_argument("--compile", type=int, default=1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    dev = torch.device(a.device)
    torch.set_grad_enabled(False)

    runs = [c for c in a.controllers if c != "tecs"]
    cfgs = [json.load(open(Path(r) / "config.json")) for r in runs]
    std_state = np.asarray(cfgs[0]["norm"]["std"][:N_STATE]) if cfgs else np.ones(N_STATE)
    H_hist = max([c["H"] for c in cfgs] + [1])

    results, traces = dict(args=vars(a)), {}
    for c in a.controllers:
        t0 = time.time()
        print(f"== {c}", flush=True)
        fl, tr, alive_t = run(c, a, cfg, std_state, H_hist, dev)
        results[c] = metrics(fl, tr, alive_t)
        results[c]["seconds"] = round(time.time() - t0, 1)
        print(json.dumps(results[c]), flush=True)
        tag = Path(c).name
        traces.update({f"{tag}__{k}": v for k, v in tr.items()})
        traces[f"{tag}__alive"] = alive_t
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out, "w"), indent=1)
    np.savez_compressed(out.with_name(out.stem + "_traces.npz"), is_test=fl.is_test, **traces)
    print("written", out, flush=True)


if __name__ == "__main__":
    main()
