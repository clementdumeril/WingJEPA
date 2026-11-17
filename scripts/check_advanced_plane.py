"""Plausibility checks of the advanced_plane port (wingjepa/sim/advanced_plane.py): trim table, linear modes at
15 m/s, static margin, 30 s open-loop response to an elevator doublet then a rudder doublet, and trim / stability of
aircraft at the corners of the extreme randomization box.

Usage: python scripts/check_advanced_plane.py --out <directory for the PNG and summary.txt>
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.sim.advanced_plane import (AERO_GROUPS, DA, DE, DR, H, OM, P, PHI, PSI, Q, R, TH, U, V, W, XN,  # noqa: E402
                                         YE, AdvancedPlane, linear_modes, load_config)

ROOT = Path(__file__).resolve().parents[1]


def describe(ev):
    """Text for eigenvalues: oscillatory pairs (wn, zeta, period) and real modes (time constant)."""
    out, seen = [], set()
    for lam in sorted(ev, key=lambda z: (abs(z.imag) < 1e-9, -abs(z))):
        if lam.imag < -1e-9:
            continue
        key = (round(lam.real, 6), round(abs(lam.imag), 6))
        if key in seen:
            continue
        seen.add(key)
        if abs(lam.imag) > 1e-9:
            wn = abs(lam)
            out.append(f"{lam.real:+.4f} +- {abs(lam.imag):.4f}j  wn {wn:.3f} rad/s  zeta {-lam.real / wn:.3f}  "
                       f"period {2 * np.pi / abs(lam.imag):.2f} s")
        else:
            tau = -1 / lam.real if abs(lam.real) > 1e-12 else np.inf
            out.append(f"{lam.real:+.4f}  time constant {tau:.2f} s" + ("  (unstable)" if lam.real > 0 else ""))
    return out


def corner_overrides(p, signs):
    """Overrides with every factor at its extreme limit r1, signs (n, n_factors) in {-1, +1}."""
    r = p["randomization"]
    names = ["mass", "Ixx", "Iyy", "Izz", "motor_k", "cg_shift", "servo_tau"] + list(r["aero"])
    lim = [r["mass"][1]] + [r["inertia"][1]] * 3 + [r["motor_k"][1], r["cg_shift"][1], r["servo_tau"][1]] \
        + [r["aero"][k][1] for k in r["aero"]]
    f = 1 + signs * np.array(lim)
    o = dict(mass=p["mass"] * f[:, 0], Ixx=p["Ixx"] * f[:, 1], Iyy=p["Iyy"] * f[:, 2], Izz=p["Izz"] * f[:, 3],
             motor_k=f[:, 4], cg_shift=f[:, 5] - 1, servo_tau=p["servo_tau"] * f[:, 6],
             aero_scale={k: f[:, 7 + i] for i, k in enumerate(r["aero"])})
    return o, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-corners", type=int, default=1000)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = load_config(ROOT / "configs/advanced_plane.yaml")
    lines = []
    say = lambda s="": (print(s), lines.append(s))

    # --- trim table ---------------------------------------------------------------------------------------------
    speeds = np.array([8.0, 10.0, 12.0, 15.0, 18.0, 22.0, 26.0, 30.0])
    ac = AdvancedPlane(p, len(speeds))
    Y, cmd, thr, ok = ac.trim(speeds, h=100.0)
    F = ac.forces(Y, thr, np.zeros((len(speeds), 3)))
    say("Trim, nominal aircraft, level flight (WingJEPA signs: de > 0 nose down, da > 0 roll left, dr > 0 yaw left)")
    say(f"{'V':>5s} {'ok':>3s} {'alpha':>7s} {'de':>7s} {'da':>7s} {'dr':>7s} {'phi':>7s} {'thr':>6s} {'omega':>7s} "
        f"{'T (N)':>6s} {'CL':>6s} {'L/D':>5s} {'sigma':>6s}  (deg)")
    for i, Vt in enumerate(speeds):
        say(f"{Vt:5.1f} {int(ok[i]):3d} {np.degrees(Y[i, TH]):7.3f} {np.degrees(cmd[i, 0]):7.3f} "
            f"{np.degrees(cmd[i, 1]):7.3f} {np.degrees(cmd[i, 2]):7.3f} {np.degrees(Y[i, PHI]):7.3f} {thr[i]:6.4f} "
            f"{Y[i, OM]:7.1f} {F['T'][i]:6.3f} {F['C'][i, 2]:6.3f} {F['lift'][i] / F['drag'][i]:5.1f} "
            f"{F['sigma'][i]:6.4f}")
    # stall speed: CL max of the blended curve
    al = np.linspace(0, 0.6, 6001)
    sig = (lambda a: (1 + np.exp(-p["M"] * (a - p["alpha_stall"])) + np.exp(p["M"] * (a + p["alpha_stall"])))
           / ((1 + np.exp(-p["M"] * (a - p["alpha_stall"]))) * (1 + np.exp(p["M"] * (a + p["alpha_stall"])))))(al)
    CL = (1 - sig) * (p["CL0"] + p["CLa"] * al) + sig * 2 * np.sin(al) ** 2 * np.cos(al)
    i = CL.argmax()
    Vs = np.sqrt(2 * p["mass"] * p["g"] / (p["rho"] * p["S"] * CL[i]))
    say(f"CL max {CL[i]:.3f} at alpha {np.degrees(al[i]):.1f} deg (no elevator, no q): 1 g stall speed ~{Vs:.1f} m/s")

    # --- linear modes at 15 m/s ---------------------------------------------------------------------------------
    ac1 = AdvancedPlane(p, 1)
    Y1, c1, t1, ok1 = ac1.trim(15.0, h=100.0)
    m = linear_modes(ac1, Y1, c1, t1)
    say()
    say("Linear modes at 15 m/s (surfaces and rotor speed held)")
    say("  longitudinal [u, w, q, theta]:")
    for s in describe(m["long"]):
        say("    " + s)
    say("  lateral [v, p, r, phi]:")
    for s in describe(m["lat"]):
        say("    " + s)
    coupled = np.sort_complex(m["full"])
    say("  coupled 8x8 eigenvalues: " + ", ".join(f"{z.real:+.3f}{z.imag:+.3f}j" for z in coupled))
    # static margin: dCm/dCL about the CG, from the aero force and moment with alpha perturbed at constant V
    eps = 1e-5
    Yp, Ym = Y1.copy(), Y1.copy()
    a0 = Y1[0, TH]
    for Yx, da_ in ((Yp, eps), (Ym, -eps)):
        Yx[0, U], Yx[0, W] = 15 * np.cos(a0 + da_), 15 * np.sin(a0 + da_)
    Fp, Fm = ac1.forces(Yp, t1, np.zeros((1, 3))), ac1.forces(Ym, t1, np.zeros((1, 3)))
    dCL = (Fp["C"][0, 2] - Fm["C"][0, 2]) / (2 * eps)
    dCm = (Fp["C"][0, 4] - Fm["C"][0, 4]) / (2 * eps)
    say(f"  CL_alpha {dCL:.3f} /rad, Cm_alpha (about the CG, cp lever arm included) {dCm:.3f} /rad: "
        f"static margin -Cm_a/CL_a = {-dCm / dCL:.3f} chord ({-dCm / dCL * p['c'] * 100:.1f} cm)")

    # --- 30 s open-loop response -----------------------------------------------------------------------------------
    dt, Tend = 0.01, 30.0
    N = int(round(Tend / dt))
    t = np.arange(N + 1) * dt
    de_add = np.where((t >= 2) & (t < 3), 1.0, np.where((t >= 3) & (t < 4), -1.0, 0.0)) * np.radians(2.0)
    dr_add = np.where((t >= 15) & (t < 16), 1.0, np.where((t >= 16) & (t < 17), -1.0, 0.0)) * np.radians(3.0)
    Ys = np.zeros((N + 1, 16))
    Ys[0] = Y1[0]
    zero = np.zeros((1, 3))
    for k in range(N):
        cmd_k = c1 + np.array([[de_add[k], 0.0, dr_add[k]]])
        Ys[k + 1] = ac1.step(Ys[k:k + 1], cmd_k, t1, zero, zero, dt)[0]
    Fs = ac1.forces(Ys, np.full(N + 1, t1[0]), np.zeros((N + 1, 3)))
    finite = np.isfinite(Ys).all()
    say()
    say(f"30 s open loop from trim at 15 m/s: elevator doublet +-2 deg (t = 2-4 s), rudder doublet +-3 deg (t = 15-17 s)")
    say(f"  finite: {finite}; Va {Fs['Va'].min():.2f}-{Fs['Va'].max():.2f} m/s; alpha "
        f"{np.degrees(Fs['alpha'].min()):.2f} to {np.degrees(Fs['alpha'].max()):.2f} deg; beta max "
        f"{np.degrees(np.abs(Fs['beta']).max()):.2f} deg; |phi| max {np.degrees(np.abs(Ys[:, PHI]).max()):.2f} deg; "
        f"h {Ys[:, H].min():.2f}-{Ys[:, H].max():.2f} m")
    # phugoid-like oscillation of the airspeed between the two doublets (t 4-15 s): zero crossings of Va - Va0
    sel = (t > 4) & (t < 15)
    dv = Fs["Va"][sel] - 15.0
    zc = np.where(np.diff(np.sign(dv)) != 0)[0]
    if len(zc) >= 3:
        say(f"  airspeed oscillation after the elevator doublet: half periods {np.round(np.diff(t[sel][zc]), 2)} s")
    ten = (t > 25)
    say(f"  last 5 s: |q| max {np.abs(Ys[ten, Q]).max():.4f} rad/s, |p| max {np.abs(Ys[ten, P]).max():.4f}, "
        f"|r| max {np.abs(Ys[ten, R]).max():.4f}, Va - 15 in [{(Fs['Va'][ten] - 15).min():.3f}, "
        f"{(Fs['Va'][ten] - 15).max():.3f}] m/s")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(4, 2, figsize=(12, 11), sharex=True)
    d = np.degrees
    plots = [
        (axs[0, 0], [(Fs["Va"], "Va (m/s)")], "airspeed"),
        (axs[0, 1], [(d(Fs["alpha"]), "alpha"), (d(Ys[:, TH]), "theta")], "angle of attack, pitch (deg)"),
        (axs[1, 0], [(d(Ys[:, Q]), "q")], "pitch rate (deg/s)"),
        (axs[1, 1], [(Ys[:, H] - Ys[0, H], "h - h0")], "altitude change (m)"),
        (axs[2, 0], [(d(Fs["beta"]), "beta"), (d(Ys[:, PHI]), "phi")], "sideslip, roll (deg)"),
        (axs[2, 1], [(d(Ys[:, P]), "p"), (d(Ys[:, R]), "r")], "roll and yaw rates (deg/s)"),
        (axs[3, 0], [(d(Ys[:, DE]), "de"), (d(Ys[:, DR]), "dr"), (d(Ys[:, DA]), "da")],
         "surfaces (deg; de > 0 nose down, dr > 0 yaw left)"),
        (axs[3, 1], [(d(np.unwrap(Ys[:, PSI]) - Ys[0, PSI]), "psi - psi0")], "heading change (deg)"),
    ]
    for ax, series, title in plots:
        for y, lab in series:
            ax.plot(t, y, label=lab, lw=1.2)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")
        for t0 in (2, 15):
            ax.axvspan(t0, t0 + 2, color="0.85", zorder=0)
    for ax in axs[-1]:
        ax.set_xlabel("t (s)")
    fig.suptitle("advanced_plane port: open loop from trim at 15 m/s, elevator doublet 2 deg, rudder doublet 3 deg",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "advanced_plane_doublets.png", dpi=130)
    plt.close(fig)

    # --- extreme randomization corners ----------------------------------------------------------------------------
    rng = np.random.default_rng(1)
    nf = 7 + len(p["randomization"]["aero"])
    nc = a.n_corners
    signs = rng.choice([-1.0, 1.0], (nc, nf))
    o, names = corner_overrides(p, signs)
    say()
    say(f"Extreme randomization: {nc} random corners of the box (every factor at +-its limit r1), "
        f"factors {names}")
    for Vt in (12.0, 15.0, 18.0):
        acc = AdvancedPlane(p, nc, overrides=o)
        Yc, cc, tc, okc = acc.trim(np.full(nc, Vt), h=100.0)
        say(f"  {Vt:4.1f} m/s: trim ok {okc.sum()}/{nc}; throttle {tc[okc].min():.3f}-{tc[okc].max():.3f}; "
            f"de {np.degrees(cc[okc, 0]).min():.2f} to {np.degrees(cc[okc, 0]).max():.2f} deg; alpha "
            f"{np.degrees(Yc[okc, TH]).min():.2f} to {np.degrees(Yc[okc, TH]).max():.2f} deg")
        if Vt == 15.0:
            worst = []
            for j in range(min(nc, 200)):
                sub = {k: (v[j:j + 1] if not isinstance(v, dict) else {g: x[j:j + 1] for g, x in v.items()})
                       for k, v in o.items()}
                a1 = AdvancedPlane(p, 1, overrides=sub)
                mm = linear_modes(a1, Yc[j:j + 1], cc[j:j + 1], tc[j:j + 1])
                worst.append((mm["long"].real.max(), mm["lat"].real.max()))
            worst = np.array(worst)
            say(f"  15 m/s, linear modes of the first 200 corners: max Re(longitudinal) {worst[:, 0].max():+.4f}, "
                f"max Re(lateral) {worst[:, 1].max():+.4f} 1/s "
                f"(lateral unstable in {(worst[:, 1] > 0).sum()}, longitudinal unstable in {(worst[:, 0] > 0).sum()})")
    (out / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwritten: {out / 'advanced_plane_doublets.png'}, {out / 'summary.txt'}")


if __name__ == "__main__":
    main()
