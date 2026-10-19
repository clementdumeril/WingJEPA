"""Validation de la traduction du simulateur UMN (wingjepa/sim/ultrastick.py).

1. Ultra Stick 25e : notre simulation contre la trajectoire de référence de leur simulateur Simulink
   non linéaire (doublet de profondeur de 1 deg depuis l'équilibre à 17 m/s, doublet_longmodel.mat).
   Valide les équations du mouvement, la propulsion et les unités (parties communes avec FASER).
2. Ultra Stick 120 (FASER, tables NASA) : lecture des tables, équilibres de 14 à 30 m/s, réponse à un
   doublet, plané moteur coupé.

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
from wingjepa.sim.ultrastick import (DE, H, OM, Q, TH, U, W, X, FaserTables, UltraStick,  # noqa: E402
                                     load_config)

D2R = np.pi / 180


def validate_25e(umn, out):
    ref = loadmat(umn / "Simulation/NL_Sim/Verification/doublet_longmodel.mat", squeeze_me=True)
    tr = loadmat(umn / "Simulation/Libraries/UAV_trimcondition.mat", squeeze_me=True,
                 struct_as_record=False)["TrimCondition"]
    t_ref, x_ref, y_ref = ref["tsim"], ref["xsim"], ref["ysim"]
    p = load_config("configs/ultrastick25e_validation.yaml")
    ac = UltraStick(p, 1)
    de_trim, thr = float(tr.Inputs.elevator), float(tr.Inputs.throttle)
    Y = np.array([[x_ref[0, 0], x_ref[0, 2], 0.0, x_ref[0, 7], 0.0, -x_ref[0, 5], de_trim, x_ref[0, 12]]])

    z = np.zeros(1)
    d0 = ac.derivatives(Y, Y[:, DE], thr, z, z, actuator=False)[0]
    print("[25e] résidu à l'équilibre UMN (du, dw, dq, domega) :",
          np.round(d0[[U, W, Q, OM]], 5))

    # entrée : même séquence que example1.m (50 Hz, interpolation linéaire comme l'entrée Simulink)
    f = 50
    dub = np.r_[np.zeros(4 * f), np.ones(f), -np.ones(f), np.zeros(4 * f)]
    u_seq = np.r_[dub, 0 * dub, 0 * dub] * D2R + de_trim
    t_seq = np.arange(len(u_seq)) / f
    dt, T = 0.002, t_ref[-1]
    ts = np.arange(0, T + dt / 2, dt)
    hist = np.zeros((len(ts), 8))
    for i, t in enumerate(ts):
        hist[i] = Y[0]
        de = np.interp(t + dt / 2, t_seq, u_seq)
        Y[:, DE] = de
        Y = ac.step(Y, np.array([de]), thr, z, z, z, z, dt, actuator=False)
    ours = {k: np.interp(t_ref, ts, hist[:, i]) for k, i in [("u", U), ("w", W), ("q", Q), ("theta", TH),
                                                            ("h", H), ("omega", OM)]}
    refs = {"u": x_ref[:, 0], "w": x_ref[:, 2], "q": x_ref[:, 10], "theta": x_ref[:, 7],
            "h": -x_ref[:, 5], "omega": x_ref[:, 12]}
    alpha_ours = np.arctan2(ours["w"], ours["u"])
    print("[25e] écart à la référence UMN sur 30 s (RMS / max | amplitude de la réponse) :")
    for k in refs:
        e = ours[k] - refs[k]
        amp = np.ptp(refs[k])
        print(f"      {k:6s} RMS {np.sqrt(np.mean(e**2)):.4g}  max {np.abs(e).max():.4g}  | {amp:.4g}")
    e = alpha_ours - y_ref[:, 2]
    print(f"      alpha  RMS {np.sqrt(np.mean(e**2)):.4g} rad, max {np.abs(e).max():.4g} | {np.ptp(y_ref[:, 2]):.4g}")

    fig, ax = plt.subplots(4, 1, figsize=(9, 9), sharex=True)
    for a, k, s, lab in [(ax[0], "u", 1, "u [m/s]"), (ax[1], "q", 180 / np.pi, "q [deg/s]"),
                         (ax[2], "theta", 180 / np.pi, "theta [deg]"), (ax[3], "h", 1, "h [m]")]:
        a.plot(t_ref, refs[k] * s, "k", lw=2, label="UMN Simulink (référence)")
        a.plot(t_ref, ours[k] * s, "r--", label="notre traduction")
        a.set_ylabel(lab); a.grid(True)
    ax[0].legend(); ax[-1].set_xlabel("t [s]")
    fig.suptitle("Ultra Stick 25e : doublet de profondeur 1 deg, équilibre 17 m/s")
    fig.tight_layout(); fig.savefig(out / "ultrastick25e_vs_umn.png", dpi=90); plt.close(fig)


def validate_faser(out):
    p = load_config("configs/ultrastick120.yaml")
    tab = FaserTables(p["aero_file"])
    # 1. les tables sont relues exactement aux points de grille
    m = loadmat(p["aero_file"], squeeze_me=True, struct_as_record=False)
    b = m["C6_bas"]; j0 = list(b.beta).index(0)
    err = np.abs(tab.base(np.asarray(b.alpha, float)[:, None]) - b.data[:, j0, :][:, [0, 2, 4]]).max()
    print(f"[FASER] relecture des tables de base aux points de grille : écart max {err:.2e}")

    # 2. équilibres en palier
    V = np.arange(14.0, 31.0, 2.0)
    ac = UltraStick(p, len(V), tab)
    Y, de, thr, ok = ac.trim(V)
    F = ac.forces(Y, thr, *(np.zeros(len(V)),) * 2)
    print("[FASER] équilibres en palier (air calme, 100 m) :")
    print("      V [m/s]  alpha [deg]  delta_e [deg]  gaz [-]  omega [tr/min]  P_moteur [W]  L/D   ok")
    for i in range(len(V)):
        print(f"      {V[i]:6.1f}  {np.rad2deg(F['alpha'][i]):10.2f}  {np.rad2deg(de[i]):12.2f}  "
              f"{thr[i]:7.3f}  {Y[i, OM] * 30 / np.pi:12.0f}  {F['P_motor'][i]:11.1f}  "
              f"{F['lift'][i] / F['drag'][i]:5.1f}  {ok[i]}")

    # 3. doublet de profondeur de 3 deg depuis l'équilibre à 20 m/s, puis 4. plané moteur coupé (40 s)
    ac1 = UltraStick(p, 1, tab)
    Y0, de0, thr0, ok0 = ac1.trim(np.array([20.0]))
    z, dt = np.zeros(1), 0.01
    runs = {}
    for name, dur in [("doublet", 20.0), ("plane", 60.0)]:
        Y = Y0.copy(); hist = []
        for k in range(int(dur / dt)):
            t = k * dt
            if name == "doublet":
                de = de0 + (3 * D2R if 2 <= t < 3 else -3 * D2R if 3 <= t < 4 else 0.0)
                th = thr0
            else:
                de, th = de0, np.zeros(1)
            F = ac1.forces(Y, th, z, z)
            hist.append([t, Y[0, U], Y[0, W], Y[0, Q], Y[0, TH], Y[0, H], Y[0, DE], Y[0, OM],
                         F["Va"][0], F["alpha"][0], F["lift"][0], F["drag"][0], F["T"][0]])
            Y = ac1.step(Y, np.asarray(de, float).reshape(1), th, z, z, z, z, dt)
        runs[name] = np.array(hist)
    d = runs["doublet"]
    print(f"[FASER] doublet 3 deg à 20 m/s : alpha {np.rad2deg(d[:, 9]).min():.1f} à "
          f"{np.rad2deg(d[:, 9]).max():.1f} deg, q max {np.rad2deg(np.abs(d[:, 3])).max():.1f} deg/s, "
          f"vitesse {d[:, 8].min():.1f}-{d[:, 8].max():.1f} m/s, valeurs finies : {np.isfinite(d).all()}")
    g = runs["plane"]; last = g[-1000:]
    sink = -np.gradient(last[:, 5], dt).mean()
    print(f"[FASER] plané moteur coupé (profondeur d'équilibre à 20 m/s), 50-60 s : Va {last[:, 8].mean():.1f} m/s, "
          f"taux de chute {sink:.2f} m/s, finesse L/D {np.mean(last[:, 10] / last[:, 11]):.1f}, "
          f"hélice {last[-1, 7] * 30 / np.pi:.0f} tr/min, poussée {last[-1, 12]:.2f} N")

    fig, ax = plt.subplots(3, 2, figsize=(12, 8), sharex="col")
    for j, name in enumerate(["doublet", "plane"]):
        r = runs[name]
        ax[0, j].plot(r[:, 0], r[:, 8]); ax[0, j].set_ylabel("Va [m/s]")
        ax[1, j].plot(r[:, 0], np.rad2deg(r[:, 9]), label="alpha"); ax[1, j].plot(r[:, 0], np.rad2deg(r[:, 4]), label="theta")
        ax[1, j].legend(); ax[1, j].set_ylabel("deg")
        ax[2, j].plot(r[:, 0], r[:, 5]); ax[2, j].set_ylabel("h [m]"); ax[2, j].set_xlabel("t [s]")
        ax[0, j].set_title({"doublet": "doublet de profondeur 3 deg", "plane": "moteur coupé"}[name])
    fig.suptitle("Ultra Stick 120 (tables NASA), depuis l'équilibre à 20 m/s")
    fig.tight_layout(); fig.savefig(out / "ultrastick120_doublet_glide.png", dpi=90); plt.close(fig)


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
