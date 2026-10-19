"""Validation de la traduction 6DoF du simulateur UMN (wingjepa/sim/ultrastick.py).

1. Ultra Stick 25e : notre simulation contre les trajectoires de référence de leur simulateur Simulink
   non linéaire (Simulation/NL_Sim/example1.m) :
     - doublet_fullmodel.mat : doublets de 1 deg sur la profondeur (0-10 s), la direction (10-20 s)
       puis les ailerons (20-30 s) ;
     - doublet_longmodel.mat : doublet de profondeur seul.
   Valide les équations du mouvement 6DoF, la propulsion (dont les moments de l'hélice) et les unités.
2. Ultra Stick 120 (FASER, tables NASA) : relecture des tables, équilibres de 14 à 30 m/s, réponses
   à des doublets de profondeur, d'ailerons et de direction, plané moteur coupé.

Usage : python scripts/validate_ultrastick.py --umn-dir <dossier UMNUAV_sim_v4> [--out runs/validation]
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.sim.ultrastick import (DA, DE, DR, H, N_STATE, OM, P, PHI, PSI, Q, R, TH, U, V, W, XN, YE,  # noqa: E402
                                     FaserTables, UltraStick, load_config)

D2R = np.pi / 180
REF_STATES = [("u", U), ("v", V), ("w", W), ("north", XN), ("east", YE), ("h", H), ("phi", PHI),
              ("theta", TH), ("psi", PSI), ("p", P), ("q", Q), ("r", R), ("omega", OM)]


def run_25e(umn, fname, blocks):
    """blocks : liste de 3 entrées (0 = rien, 'e' / 'a' / 'r' = doublet sur cette gouverne) de 10 s."""
    ref = loadmat(umn / "Simulation/NL_Sim/Verification" / fname, squeeze_me=True)
    tr = loadmat(umn / "Simulation/Libraries/UAV_trimcondition.mat", squeeze_me=True,
                 struct_as_record=False)["TrimCondition"]
    t_ref, x_ref = ref["tsim"], ref["xsim"]
    p = load_config("configs/ultrastick25e_validation.yaml")
    ac = UltraStick(p, 1)
    trim = np.array([tr.Inputs.elevator, tr.Inputs.aileron, tr.Inputs.rudder], float)
    thr = float(tr.Inputs.throttle)
    Y = np.zeros((1, N_STATE))
    for k, (_, i) in enumerate(REF_STATES):
        Y[0, i] = x_ref[0, k]
    Y[0, H] = -x_ref[0, 5]
    Y[0, DE:DR + 1] = trim

    f = 50                                            # même séquence que example1.m
    dub = np.r_[np.zeros(4 * f), np.ones(f), -np.ones(f), np.zeros(4 * f)] * D2R
    seq = np.zeros((3 * len(dub), 3))
    for b, which in enumerate(blocks):
        if which:
            seq[b * len(dub):(b + 1) * len(dub), "ear".index(which)] = dub
    seq += trim
    t_seq = np.arange(len(seq)) / f
    dt = 0.002
    ts = np.arange(0, t_ref[-1] + dt / 2, dt)
    hist = np.zeros((len(ts), N_STATE))
    z3 = np.zeros((1, 3))
    for i, t in enumerate(ts):
        hist[i] = Y[0]
        cmd = np.array([[np.interp(t + dt / 2, t_seq, seq[:, j]) for j in range(3)]])
        Y[:, DE:DR + 1] = cmd                      # pas de dynamique de gouverne dans UAV_NL
        Y = ac.step(Y, cmd, thr, z3, z3, dt, actuator=False)
    ours = {name: np.interp(t_ref, ts, hist[:, i]) for name, i in REF_STATES}
    refs = {name: x_ref[:, k] for k, (name, _) in enumerate(REF_STATES)}
    refs["h"] = -x_ref[:, 5]
    return t_ref, ours, refs


def report(label, ours, refs, keys):
    print(f"[25e] {label} : écart à la référence UMN (RMS / max | amplitude de la réponse)")
    for k in keys:
        e = ours[k] - refs[k]
        print(f"      {k:6s} RMS {np.sqrt(np.mean(e**2)):.4g}  max {np.abs(e).max():.4g}  | {np.ptp(refs[k]):.4g}")


def validate_25e(umn, out):
    t, o, r = run_25e(umn, "doublet_fullmodel.mat", ["e", "r", "a"])
    report("doublets profondeur, direction, ailerons (30 s)", o, r,
           ["u", "v", "w", "p", "q", "r", "phi", "theta", "psi", "h", "omega"])
    fig, ax = plt.subplots(6, 1, figsize=(9, 13), sharex=True)
    for a, k, s, lab in [(ax[0], "q", 180 / np.pi, "q [deg/s]"), (ax[1], "p", 180 / np.pi, "p [deg/s]"),
                         (ax[2], "r", 180 / np.pi, "r [deg/s]"), (ax[3], "phi", 180 / np.pi, "phi [deg]"),
                         (ax[4], "v", 1, "v [m/s]"), (ax[5], "h", 1, "h [m]")]:
        a.plot(t, r[k] * s, "k", lw=2, label="UMN Simulink (référence)")
        a.plot(t, o[k] * s, "r--", label="notre traduction")
        a.set_ylabel(lab); a.grid(True)
    ax[0].legend(); ax[-1].set_xlabel("t [s]")
    fig.suptitle("Ultra Stick 25e : doublets de 1 deg (profondeur 4-6 s, direction 14-16 s, ailerons 24-26 s)")
    fig.tight_layout(); fig.savefig(out / "ultrastick25e_vs_umn_6dof.png", dpi=90); plt.close(fig)
    _, o, r = run_25e(umn, "doublet_longmodel.mat", ["e", 0, 0])
    report("doublet de profondeur seul (30 s)", o, r, ["u", "w", "q", "theta", "h"])


def validate_faser(out):
    p = load_config("configs/ultrastick120.yaml")
    tab = FaserTables(p["aero_file"])
    m = loadmat(p["aero_file"], squeeze_me=True, struct_as_record=False)
    b = m["C6_bas"]
    A, B = np.meshgrid(np.asarray(b.alpha, float), np.asarray(b.beta, float), indexing="ij")
    err = np.abs(tab.base(np.stack([A.ravel(), B.ravel()], 1)) - b.data.reshape(-1, 6)).max()
    print(f"[FASER] relecture des tables de base aux points de grille : écart max {err:.2e}")

    V = np.arange(14.0, 31.0, 2.0)
    ac = UltraStick(p, len(V), tab)
    Y, cmd, thr, ok = ac.trim(V)
    F = ac.forces(Y, thr, np.zeros((len(V), 3)))
    print("[FASER] équilibres en palier, ailes presque à plat, dérapage nul (air calme, 100 m) :")
    print("      V     alpha   de      gaz    tr/min   P [W]   L/D   phi    da     dr   [deg]  ok")
    for i in range(len(V)):
        print(f"    {V[i]:5.1f}  {np.rad2deg(F['alpha'][i]):6.2f}  {np.rad2deg(cmd[i, 0]):6.2f}  {thr[i]:5.3f}"
              f"  {Y[i, OM] * 30 / np.pi:6.0f}  {F['P_motor'][i]:6.1f}  {F['lift'][i] / F['drag'][i]:4.1f}"
              f"  {np.rad2deg(Y[i, PHI]):5.2f}  {np.rad2deg(cmd[i, 1]):5.2f}  {np.rad2deg(cmd[i, 2]):5.2f}  {ok[i]}")

    ac1 = UltraStick(p, 1, tab)
    Y0, c0, thr0, ok0 = ac1.trim(np.array([20.0]))
    z3, dt = np.zeros((1, 3)), 0.01
    runs = {}
    for name, dur, axis, amp in [("profondeur", 20.0, 0, 3), ("ailerons", 20.0, 1, 5), ("direction", 20.0, 2, 5),
                                 ("plane", 60.0, None, 0)]:
        Y = Y0.copy(); hist = []
        for k in range(int(dur / dt)):
            t = k * dt
            cmd = c0.copy()
            if axis is not None:
                cmd[0, axis] += amp * D2R if 2 <= t < 3 else -amp * D2R if 3 <= t < 4 else 0.0
            th = np.zeros(1) if name == "plane" else thr0
            F = ac1.forces(Y, th, z3)
            hist.append([t, F["Va"][0], F["alpha"][0], F["beta"][0], Y[0, P], Y[0, Q], Y[0, R], Y[0, PHI],
                         Y[0, TH], Y[0, PSI], Y[0, H], F["lift"][0], F["drag"][0]])
            Y = ac1.step(Y, cmd, th, z3, z3, dt)
        runs[name] = np.array(hist)
    deg = np.rad2deg
    for name in ("profondeur", "ailerons", "direction"):
        d = runs[name]
        print(f"[FASER] doublet {name} : |p| max {deg(np.abs(d[:, 4])).max():.1f}, |q| max {deg(np.abs(d[:, 5])).max():.1f},"
              f" |r| max {deg(np.abs(d[:, 6])).max():.1f} deg/s ; |beta| max {deg(np.abs(d[:, 3])).max():.1f}, "
              f"|phi| fin {deg(abs(d[-1, 7])):.1f} deg ; Va {d[:, 1].min():.1f}-{d[:, 1].max():.1f} m/s ; "
              f"finies : {np.isfinite(d).all()}")
    g = runs["plane"]; last = g[-1000:]
    print(f"[FASER] plané moteur coupé, 50-60 s : Va {last[:, 1].mean():.1f} m/s, taux de chute "
          f"{-np.gradient(last[:, 10], dt).mean():.2f} m/s, L/D {np.mean(last[:, 11] / last[:, 12]):.1f}, "
          f"|phi| {deg(np.abs(last[:, 7])).max():.1f} deg")

    fig, ax = plt.subplots(3, 3, figsize=(14, 8), sharex=True)
    for j, name in enumerate(["profondeur", "ailerons", "direction"]):
        r = runs[name]
        ax[0, j].plot(r[:, 0], deg(r[:, 4]), label="p"); ax[0, j].plot(r[:, 0], deg(r[:, 5]), label="q")
        ax[0, j].plot(r[:, 0], deg(r[:, 6]), label="r"); ax[0, j].legend(); ax[0, j].set_title(f"doublet {name}")
        ax[1, j].plot(r[:, 0], deg(r[:, 7]), label="phi"); ax[1, j].plot(r[:, 0], deg(r[:, 8]), label="theta")
        ax[1, j].plot(r[:, 0], deg(r[:, 3]), label="beta"); ax[1, j].legend()
        ax[2, j].plot(r[:, 0], r[:, 1]); ax[2, j].set_ylabel("Va [m/s]"); ax[2, j].set_xlabel("t [s]")
    ax[0, 0].set_ylabel("deg/s"); ax[1, 0].set_ylabel("deg")
    fig.suptitle("Ultra Stick 120 (tables NASA, 6DoF), depuis l'équilibre à 20 m/s")
    fig.tight_layout(); fig.savefig(out / "ultrastick120_6dof_doublets.png", dpi=90); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umn-dir", required=True)
    ap.add_argument("--out", default="runs/validation")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    validate_25e(Path(a.umn_dir), out)
    validate_faser(out)
    print(f"figures : {out}")


if __name__ == "__main__":
    main()
