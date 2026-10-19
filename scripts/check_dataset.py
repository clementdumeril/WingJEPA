"""Contrôles de sanité d'un set de vols générés : texte (à copier-coller) + figures.

Usage : python scripts/check_dataset.py --dir <out_dir>/<name> --set smoke
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.sim.pilot import TYPE_NAMES  # noqa: E402

G = 9.80665   # comme configs/ultrastick120.yaml


def load(d, s):
    files = sorted(Path(d).glob(f"{s}_shard*.npz"))
    assert files, f"aucun fichier {s}_shard*.npz dans {d}"
    zs = [np.load(f, allow_pickle=False) for f in files]
    ch = list(zs[0]["channels"])
    data = np.concatenate([z["data"] for z in zs])
    meta = {k: np.concatenate([z[k] for z in zs]) for k in zs[0].files if k not in ("data", "channels")}
    return data, ch, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--set", required=True)
    a = ap.parse_args()
    D, ch, M = load(a.dir, a.set)
    c = {k: D[:, :, i] for i, k in enumerate(ch)}
    n, K = D.shape[:2]
    dt = float(c["t"][0, 1] - c["t"][0, 0])
    alive = c["alive"] > 0.5
    ok = alive.all(1)
    print(f"== {a.set} : {n} vols x {K} pas ({K * dt:.0f} s, dt = {dt} s)")

    # 1. validité
    print(f"[1] valeurs non finies : {int((~np.isfinite(D)).sum())} (dont dans la partie valide des vols : "
          f"{int((~np.isfinite(D[alive])).sum())})")
    print(f"[1] vols restés dans l'enveloppe : {ok.sum()}/{n}")
    for i, name in enumerate(TYPE_NAMES):
        m = M["type"] == i
        print(f"     {name:10s} : {ok[m].sum()}/{m.sum()}")
    dead = ~ok
    if dead.any():
        print(f"     sorties d'enveloppe à t = {np.round(M['t_dead'][dead][:10], 1)} s (10 premières)")

    # 2. plages parcourues (vols valides, échantillons vivants)
    v = alive & ok[:, None]
    def rng_(k, f=1.0):
        x = c[k][v] * f
        return f"{np.percentile(x, 1):7.2f} {np.percentile(x, 50):7.2f} {np.percentile(x, 99):7.2f}"
    print("[2] percentiles 1 / 50 / 99 :")
    for k, f, u in [("Va", 1, "m/s"), ("alpha", 57.3, "deg"), ("theta", 57.3, "deg"), ("q", 57.3, "deg/s"),
                    ("h", 1, "m"), ("delta_e", 57.3, "deg"), ("delta_t_cmd", 1, "-"), ("thrust", 1, "N"),
                    ("P_motor", 1, "W"), ("omega_rpm", 1, "rpm"), ("wind_x", 1, "m/s"), ("wind_d", 1, "m/s"),
                    ("turb_w", 1, "m/s")]:
        print(f"     {k:8s} [{u:5s}] {rng_(k, f)}")
    print(f"     part du temps moteur coupé : {c['engine_off'][v].mean():.2f} ; boucle ouverte : "
          f"{c['open_loop'][v].mean():.2f}")
    a0 = np.deg2rad(12.5)   # incidence du maximum de portance de l'Ultra Stick 120 (tables NASA)
    print(f"     échantillons alpha > {np.rad2deg(a0):.0f} deg : {(c['alpha'][v] > a0).mean():.4f} ; "
          f"vols concernés : {int(((c['alpha'] > a0) & v).any(1).sum())}")

    # 3. bilan d'énergie (exact pour ces équations) : dE/dt = (Fx_aero + T) u + Fz_aero w + My q,
    #    avec E = 1/2 m (u^2 + w^2) + 1/2 Iyy q^2 + m g h_air, h_air = h + w_d moyen * t
    np.seterr(all="ignore")   # les vols sortis de l'enveloppe sont exclus des statistiques ci-dessous
    m, Iyy = M["mass"][:, None], M["Iyy"][:, None]
    h_air = c["h"] + M["wind_steady_d"][:, None] * c["t"]
    E = 0.5 * m * (c["u"]**2 + c["w"]**2) + 0.5 * Iyy * c["q"]**2 + m * G * h_air
    P = (c["Fx_aero"] + c["thrust"]) * c["u"] + c["Fz_aero"] * c["w"] + c["My_aero"] * c["q"]
    dE = (E[:, 2:] - E[:, :-2]) / (2 * dt)
    vv = v[:, 2:] & v[:, :-2]
    r = dE[vv] - P[:, 1:-1][vv]
    print(f"[3] bilan d'énergie : RMS(dE/dt - P) = {np.sqrt(np.mean(r**2)):.2f} W pour RMS(P) = "
          f"{np.sqrt(np.mean(P[:, 1:-1][vv]**2)):.2f} W ; corrélation = "
          f"{np.corrcoef(dE[vv], P[:, 1:-1][vv])[0, 1]:.4f}")

    # 4. plané : moteur coupé depuis > 4 s, vent vertical faible
    eo = c["engine_off"] > 0.5
    run = np.zeros_like(eo, int)
    for k in range(1, K):
        run[:, k] = np.where(eo[:, k], run[:, k - 1] + 1, 0)
    gl = v & (run * dt > 4) & (np.abs(c["wind_d"]) < 0.3)
    if gl.any():
        hdot = np.gradient(c["h"], dt, axis=1)
        sink = -(hdot + c["wind_d"])[gl]                         # taux de chute par rapport à l'air
        LD = (c["lift"] / np.maximum(c["drag"], 1e-6))[gl]
        print(f"[4] plané ({gl.sum()} échantillons) : Va médiane {np.median(c['Va'][gl]):.1f} m/s, "
              f"taux de chute air médian {np.median(sink):.2f} m/s, finesse L/D médiane {np.median(LD):.1f}")
    else:
        print("[4] plané : aucun segment moteur coupé de plus de 4 s")

    # 5. signes : profondeur positive -> q diminue ; moteur -> poussée
    qd = np.gradient(c["q"], dt, axis=1)
    de = c["delta_e"] - c["delta_e"].mean(1, keepdims=True)
    print(f"[5] corr(delta_e, dq/dt) = {np.corrcoef(de[v], qd[v])[0, 1]:+.3f} (attendu < 0) ; "
          f"corr(delta_t_cmd, poussée) = {np.corrcoef(c['delta_t_cmd'][v], c['thrust'][v])[0, 1]:+.3f} (attendu > 0)")
    print(f"[6] énergie moteur moyenne par vol : {np.mean(c['E_motor'][ok, -1]) / 3600:.2f} Wh")

    # figures
    out = Path(a.dir) / "checks" / a.set
    out.mkdir(parents=True, exist_ok=True)
    picks = [int(np.flatnonzero(M["type"] == i)[0]) for i in range(4) if (M["type"] == i).any()]
    if eo.any(1).any():
        picks.append(int(np.flatnonzero(eo.any(1))[0]))
    for i in picks:
        fig, ax = plt.subplots(5, 1, figsize=(9, 10), sharex=True)
        t = c["t"][i]
        ax[0].plot(t, c["Va"][i]); ax[0].set_ylabel("Va [m/s]")
        ax[1].plot(t, c["h"][i]); ax[1].set_ylabel("h [m]")
        ax[2].plot(t, np.rad2deg(c["theta"][i]), label="theta"); ax[2].plot(t, np.rad2deg(c["alpha"][i]), label="alpha")
        ax[2].legend(); ax[2].set_ylabel("deg")
        ax[3].plot(t, np.rad2deg(c["delta_e"][i]), label="delta_e [deg]"); ax[3].plot(t, 10 * c["delta_t_cmd"][i], label="10 x delta_t")
        ax[3].fill_between(t, -5, 5, where=c["engine_off"][i] > 0.5, alpha=0.2, label="moteur coupé"); ax[3].legend()
        ax[4].plot(t, c["wind_x"][i], label="wind_x"); ax[4].plot(t, c["wind_d"][i], label="wind_d")
        ax[4].plot(t, c["turb_w"][i], label="turb_w", lw=0.6); ax[4].legend(); ax[4].set_xlabel("t [s]")
        fig.suptitle(f"vol {i} - {TYPE_NAMES[M['type'][i]]} - turbulence {M['turb_level'][i]}")
        fig.tight_layout(); fig.savefig(out / f"flight_{i:04d}.png", dpi=90); plt.close(fig)
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for axi, (k, f) in zip(ax.ravel(), [("Va", 1), ("alpha", 57.3), ("delta_e", 57.3), ("delta_t_cmd", 1),
                                        ("P_motor", 1), ("wind_x", 1)]):
        axi.hist(c[k][v] * f, 60); axi.set_title(k)
    fig.tight_layout(); fig.savefig(out / "histograms.png", dpi=90); plt.close(fig)
    print(f"figures : {out}")


if __name__ == "__main__":
    main()
