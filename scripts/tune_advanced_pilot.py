"""Step responses of the data-generation autopilot (TECS + lateral pilot, wingjepa/sim/pilot.py) on the
advanced_plane, with the gains of a data config (default configs/data_advanced.yaml): the nominal aircraft and
four corners of the extreme randomization box. Same control loop as scripts/generate_flights.py (20 Hz commands,
RK4 0.01 s, angle of attack filtered for the TECS limiter), without envelope protection.

Scenarios: altitude step +20 m at 15 m/s; speed step 12 -> 18 m/s; 30 deg bank turn at 15 m/s; 2-1-1 elevator
excitation (3 deg, unit 0.3 s) with the autopilot frozen; engine-off glide (speed target 12 m/s).

Usage: python scripts/tune_advanced_pilot.py --out <directory> [--config configs/data_advanced.yaml]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.sim.advanced_plane import (AdvancedPlane, H, P, PHI, Q, R, TH, U, W,  # noqa: E402
                                         dcm_body_to_ned, load_config)
from wingjepa.sim.pilot import TECS, lateral  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DT, DT_SIM = 0.05, 0.01
NAMES = ["nominal", "corner A (heavy, weak, draggy, aft CG, slow)", "corner B (light, strong, clean, fwd CG, fast)",
         "corner C (random)", "corner D (random)"]


def corners(p, ranges):
    """Overrides: nominal + 4 corners of the extreme box (every factor at +-its limit r1)."""
    keys = ["mass", "inertia", "motor_k", "cg_shift", "servo_tau"] + list(ranges["aero"])
    lim = np.array([ranges[k][1] for k in keys[:5]] + [ranges["aero"][k][1] for k in ranges["aero"]])
    # A: heavy, high inertia, weak motor, aft CG, slow servos, every aero group low except the drag (high)
    sA = np.array([1, 1, -1, 1, 1] + [1 if k == "CD" else -1 for k in ranges["aero"]], float)
    rng = np.random.default_rng(11)
    S = np.stack([np.zeros_like(sA), sA, -sA, rng.choice([-1.0, 1.0], len(sA)), rng.choice([-1.0, 1.0], len(sA))])
    f = 1 + S * lim
    ia = 5
    return dict(mass=p["mass"] * f[:, 0], Ixx=p["Ixx"] * f[:, 1], Iyy=p["Iyy"] * f[:, 1], Izz=p["Izz"] * f[:, 1],
                motor_k=f[:, 2], cg_shift=f[:, 3] - 1, servo_tau=p["servo_tau"] * f[:, 4],
                aero_scale={k: f[:, ia + i] for i, k in enumerate(ranges["aero"])})


def fly(ac, cfg, V0, T, h_ref, V_ref, phi_ref, de_add=None, freeze=None, eoff=None):
    """Closed loop as generate_flights.simulate. Reference functions of t -> (n,) arrays. Returns traces."""
    pil, g = cfg["pilot"], ac.p["g"]
    n = ac.n
    Y, cmd0, thr0, ok = ac.trim(np.full(n, V0), h=200.0)
    assert ok.all(), "trim failed"
    h0 = Y[:, H].copy()
    tecs = TECS(n, pil["tecs"], thr_ff=thr0, theta_ff=Y[:, TH], de_ff=cmd0[:, 0], g=g)
    held = np.c_[cmd0, thr0]
    K = int(round(T / DT))
    rec = {k: np.zeros((n, K)) for k in ("t", "Va", "h", "alpha", "beta", "phi", "theta", "q", "de", "da", "dr",
                                         "thr", "gamma")}
    zero = np.zeros((n, 3))
    air_f = None
    t = 0.0
    for k in range(K):
        F = ac.forces(Y, held[:, 3], zero)
        Va, alpha, beta = F["Va"], F["alpha"], F["beta"]
        C = dcm_body_to_ned(Y[:, PHI], Y[:, TH], Y[:, 8])
        hdot = -np.einsum("nj,nj->n", C[:, 2, :], Y[:, U:W + 1])
        gamma = np.arcsin(np.clip(hdot / Va, -1, 1))
        phi, th = Y[:, PHI], Y[:, TH]
        psidot = g * np.tan(np.clip(phi, -1.3, 1.3)) / Va
        q_ref, r_ref = psidot * np.cos(th) * np.sin(phi), psidot * np.cos(th) * np.cos(phi)
        air = np.stack([Va, alpha, beta], 1)
        air_f = air.copy() if air_f is None else air_f + DT / (pil["protection"]["filter_tau_s"] + DT) * (air - air_f)
        fr = freeze(t) if freeze else np.zeros(n, bool)
        eo = eoff(t) if eoff else np.zeros(n, bool)
        de_c, thr_c = tecs(Va, air_f[:, 1], gamma, th, Y[:, Q], q_ref, Y[:, H], h0 + h_ref(t), V_ref(t), eo, fr, DT)
        da_c, dr_c = lateral(phi_ref(t), phi, Y[:, P], Y[:, R], r_ref, beta, pil["lateral"]["gains"],
                             cmd0[:, 1], cmd0[:, 2])
        held = np.where(fr[:, None], held, np.stack([de_c, da_c, dr_c, thr_c], 1))
        add = de_add(t) if de_add else 0.0
        cmd = np.clip(held[:, :3] + np.array([1.0, 0, 0]) * np.reshape(add, (-1, 1)), ac.p["act_min"], ac.p["act_max"])
        thr = np.where(eo, 0.0, np.clip(held[:, 3], 0, 1))
        for name, val in (("t", t), ("Va", Va), ("h", Y[:, H] - h0), ("alpha", alpha), ("beta", beta),
                          ("phi", phi), ("theta", th), ("q", Y[:, Q]), ("de", Y[:, 12]), ("da", Y[:, 13]),
                          ("dr", Y[:, 14]), ("thr", thr), ("gamma", gamma)):
            rec[name][:, k] = val
        for _ in range(int(round(DT / DT_SIM))):
            Y = ac.step(Y, cmd, thr, zero, zero, DT_SIM)
            t += DT_SIM
    assert np.isfinite(Y).all()
    return rec


def settle(t, x, target, band, t0):
    """Settling time after t0: last time |x - target| > band (0 if always inside after t0)."""
    out = np.abs(x - target) > band
    idx = np.where(out & (t >= t0))[0]
    return 0.0 if len(idx) == 0 else t[idx[-1]] - t0 + DT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default="configs/data_advanced.yaml")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(open(ROOT / a.config, encoding="utf-8"))
    p = load_config(ROOT / cfg["aircraft"]["config"])
    ov = corners(p, cfg["aircraft"]["ranges"])
    n = len(NAMES)
    ac = AdvancedPlane(p, n, overrides=ov)
    const = lambda v: (lambda t: np.full(n, float(v)))
    step = lambda t0, a0, a1: (lambda t: np.full(n, a1 if t >= t0 else a0, float))
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def plot(rec, panels, title, fname, refs=None):
        fig, axs = plt.subplots(len(panels), 1, figsize=(10, 2.1 * len(panels)), sharex=True)
        for ax, (key, scale, lab) in zip(axs, panels):
            for i in range(n):
                ax.plot(rec["t"][i], rec[key][i] * scale, lw=1.3 if i == 0 else 0.9, label=NAMES[i] if ax is axs[0] else None)
            if refs and key in refs:
                ax.plot(rec["t"][0], refs[key], "k:", lw=1)
            ax.set_ylabel(lab, fontsize=9)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7, ncol=2)
        axs[-1].set_xlabel("t (s)")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=120)
        plt.close(fig)

    d = 180 / np.pi
    T = 25.0
    # 1. altitude step +20 m
    r = fly(ac, cfg, 15.0, T, step(2, 0, 20), const(15.0), const(0.0))
    t = r["t"][0]
    say("Altitude step +20 m at 15 m/s (band +-1 m)")
    for i in range(n):
        say(f"  {NAMES[i]:48s} overshoot {max(r['h'][i].max() - 20, 0):5.2f} m, settling {settle(t, r['h'][i], 20, 1.0, 2):5.2f} s,"
            f" |dV| max {np.abs(r['Va'][i] - 15).max():4.2f} m/s, alpha max {r['alpha'][i].max() * d:5.2f} deg")
    plot(r, [("h", 1, "h - h0 (m)"), ("Va", 1, "Va (m/s)"), ("theta", d, "theta (deg)"), ("alpha", d, "alpha (deg)"),
             ("de", d, "de (deg)"), ("thr", 1, "throttle")], "altitude step +20 m (TECS)", "pilot_altitude_step.png",
         refs={"h": np.where(t >= 2, 20, 0)})
    # 2. speed step 12 -> 18 m/s
    r = fly(ac, cfg, 12.0, T, const(0.0), step(2, 12.0, 18.0), const(0.0))
    say("Speed step 12 -> 18 m/s (band +-0.3 m/s)")
    for i in range(n):
        say(f"  {NAMES[i]:48s} overshoot {max(r['Va'][i].max() - 18, 0):5.2f} m/s, settling "
            f"{settle(t, r['Va'][i], 18, 0.3, 2):5.2f} s, |dh| max {np.abs(r['h'][i]).max():5.2f} m")
    plot(r, [("Va", 1, "Va (m/s)"), ("h", 1, "h - h0 (m)"), ("theta", d, "theta (deg)"), ("de", d, "de (deg)"),
             ("thr", 1, "throttle")], "speed step 12 -> 18 m/s (TECS)", "pilot_speed_step.png",
         refs={"Va": np.where(t >= 2, 18, 12)})
    # 3. 30 deg bank turn
    r = fly(ac, cfg, 15.0, T, const(0.0), const(15.0), step(2, 0.0, np.radians(30)))
    say("Bank turn 30 deg at 15 m/s (band +-1.5 deg)")
    for i in range(n):
        ph = r["phi"][i] * d
        rise = t[np.argmax(ph >= 27)] - 2 if (ph >= 27).any() else np.nan
        say(f"  {NAMES[i]:48s} rise (90 %) {rise:4.2f} s, overshoot {max(ph.max() - 30, 0):4.2f} deg, settling "
            f"{settle(t, ph, 30, 1.5, 2):5.2f} s, |beta| max {np.abs(r['beta'][i]).max() * d:4.2f} deg, "
            f"h loss {-r['h'][i].min():4.2f} m, Va {r['Va'][i].min():5.2f}-{r['Va'][i].max():5.2f}")
    plot(r, [("phi", d, "phi (deg)"), ("beta", d, "beta (deg)"), ("h", 1, "h - h0 (m)"), ("Va", 1, "Va (m/s)"),
             ("da", d, "da (deg)"), ("dr", d, "dr (deg)")], "30 deg bank turn (lateral pilot + TECS)",
         "pilot_bank_turn.png", refs={"phi": np.where(t >= 2, 30, 0)})
    # 4. 2-1-1 elevator excitation, autopilot frozen
    u = 0.3
    amp = np.radians(3.0)

    def de_add(tt):
        if 2 <= tt < 2 + 2 * u:
            return amp
        if 2 + 2 * u <= tt < 2 + 3 * u:
            return -amp
        if 2 + 3 * u <= tt < 2 + 4 * u:
            return amp
        return 0.0
    fz = lambda tt: np.full(n, 2 <= tt < 2 + 4 * u + 2.5)
    r = fly(ac, cfg, 15.0, 15.0, const(0.0), const(15.0), const(0.0), de_add=de_add, freeze=fz)
    t2 = r["t"][0]
    say("2-1-1 elevator excitation (3 deg, unit 0.3 s) with the autopilot frozen 2-5.7 s, then TECS")
    for i in range(n):
        say(f"  {NAMES[i]:48s} alpha {r['alpha'][i].min() * d:5.2f} to {r['alpha'][i].max() * d:5.2f} deg, q "
            f"{r['q'][i].min() * d:6.1f} to {r['q'][i].max() * d:5.1f} deg/s, Va {r['Va'][i].min():5.2f}-"
            f"{r['Va'][i].max():5.2f}, h {r['h'][i].min():5.2f} to {r['h'][i].max():4.2f} m, back within 1 m after "
            f"{settle(t2, r['h'][i], 0, 1.0, 5.7):4.1f} s")
    plot(r, [("de", d, "de (deg)"), ("alpha", d, "alpha (deg)"), ("q", d, "q (deg/s)"), ("theta", d, "theta (deg)"),
             ("Va", 1, "Va (m/s)"), ("h", 1, "h - h0 (m)")], "2-1-1 elevator excitation, autopilot frozen 2-5.7 s",
         "pilot_211_elevator.png")
    # 5. engine-off glide
    eo = lambda tt: np.full(n, tt >= 2)
    r = fly(ac, cfg, 15.0, T, const(0.0), const(12.0), const(0.0), eoff=eo)
    say("Engine-off glide from 15 m/s (speed target 12 m/s, theta_glide_0 + speed loop)")
    last = t >= T - 5
    for i in range(n):
        Va_e = r["Va"][i, last].mean()
        sink = -np.gradient(r["h"][i], DT)[last].mean()
        say(f"  {NAMES[i]:48s} Va last 5 s {Va_e:5.2f} m/s (min over the glide {r['Va'][i, t > 2].min():5.2f}), "
            f"sink {sink:4.2f} m/s, L/D {Va_e / max(sink, 1e-6):4.1f}, alpha {r['alpha'][i, last].mean() * d:4.1f} "
            f"deg, theta {r['theta'][i, last].mean() * d:5.1f} deg")
    plot(r, [("Va", 1, "Va (m/s)"), ("h", 1, "h - h0 (m)"), ("theta", d, "theta (deg)"), ("alpha", d, "alpha (deg)"),
             ("de", d, "de (deg)")], "engine-off glide (speed target 12 m/s)", "pilot_glide.png")
    robustness(cfg, p, say)
    (out / "pilot_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def robustness(cfg, p, say, n_each=40):
    """Edge-of-envelope scenarios on random aircraft (training ranges and extreme box corners): count the flights
    that leave a sane envelope (|theta| > 45 deg, alpha outside [-15, 20] deg, Va < 6 m/s, |phi| > 60 deg)."""
    from wingjepa.sim.advanced_plane import sample_overrides
    rng = np.random.default_rng(7)
    r = cfg["aircraft"]["ranges"]
    o_train = sample_overrides(rng, n_each, p, r)
    keys = ["mass", "inertia", "motor_k", "cg_shift", "servo_tau"] + list(r["aero"])
    lim = np.array([r[k][1] for k in keys[:5]] + [r["aero"][k][1] for k in r["aero"]])
    f = 1 + rng.choice([-1.0, 1.0], (n_each, len(lim))) * lim
    o_ext = dict(mass=p["mass"] * f[:, 0], Ixx=p["Ixx"] * f[:, 1], Iyy=p["Iyy"] * f[:, 1], Izz=p["Izz"] * f[:, 1],
                 motor_k=f[:, 2], cg_shift=f[:, 3] - 1, servo_tau=p["servo_tau"] * f[:, 4],
                 aero_scale={k: f[:, 5 + i] for i, k in enumerate(r["aero"])})
    say()
    say(f"Robustness: {n_each} training aircraft + {n_each} extreme corners, flights leaving the sane envelope "
        f"(|theta| > 45, alpha outside [-15, 20] deg, Va < 6 m/s, |phi| > 60 deg), 20 s each")
    for label, o in (("training", o_train), ("extreme", o_ext)):
        ac = AdvancedPlane(p, n_each, overrides=o)
        n = n_each
        const = lambda v: (lambda t: np.full(n, float(v)))
        step = lambda t0, a0, a1: (lambda t: np.full(n, a1 if t >= t0 else a0, float))
        scen = {
            "alt +20 m at 10 m/s": dict(V0=10.0, h_ref=step(2, 0, 20), V_ref=const(10.0), phi_ref=const(0.0)),
            "alt -20 m at 22 m/s": dict(V0=22.0, h_ref=step(2, 0, -20), V_ref=const(22.0), phi_ref=const(0.0)),
            "speed 10 -> 22 m/s": dict(V0=10.0, h_ref=const(0.0), V_ref=step(2, 10.0, 22.0), phi_ref=const(0.0)),
            "speed 22 -> 10 m/s": dict(V0=22.0, h_ref=const(0.0), V_ref=step(2, 22.0, 10.0), phi_ref=const(0.0)),
            "bank 40 deg at 14 m/s": dict(V0=14.0, h_ref=const(0.0), V_ref=const(14.0),
                                          phi_ref=step(2, 0.0, np.radians(40))),
            "glide from 10 m/s": dict(V0=10.0, h_ref=const(0.0), V_ref=const(10.0), phi_ref=const(0.0),
                                      eoff=lambda t: np.full(n, t >= 2)),
        }
        for name, s in scen.items():
            rec = fly(ac, cfg, s["V0"], 20.0, s["h_ref"], s["V_ref"], s["phi_ref"], eoff=s.get("eoff"))
            bad = ((np.abs(rec["theta"]) > np.radians(45)).any(1) | (rec["alpha"] > np.radians(20)).any(1)
                   | (rec["alpha"] < np.radians(-15)).any(1) | (rec["Va"] < 6).any(1)
                   | (np.abs(rec["phi"]) > np.radians(60)).any(1))
            say(f"  {label:9s} {name:24s} out of envelope {bad.sum():2d}/{n}; alpha max "
                f"{np.degrees(rec['alpha'].max()):5.1f} deg, Va {rec['Va'].min():5.2f}-{rec['Va'].max():5.2f} m/s, "
                f"|phi| max {np.degrees(np.abs(rec['phi']).max()):4.1f} deg")


if __name__ == "__main__":
    main()
