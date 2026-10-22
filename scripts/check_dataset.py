"""Sanity checks on a set of generated (6DoF) flights: text output (copy-paste friendly) + figures.

Usage: python scripts/check_dataset.py --dir <out_dir>/<name> --set smoke
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.sim.pilot import LAT_NAMES, TYPE_NAMES  # noqa: E402

G = 9.80665   # same as configs/ultrastick120.yaml
DEG = 180 / np.pi


def load(d, s):
    files = sorted(Path(d).glob(f"{s}_shard*.npz"))
    assert files, f"no file {s}_shard*.npz in {d}"
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
    np.seterr(all="ignore")   # flights that left the envelope are excluded from the statistics
    D, ch, M = load(a.dir, a.set)
    c = {k: D[:, :, i].astype(np.float64) for i, k in enumerate(ch)}
    n, K = D.shape[:2]
    dt = float(c["t"][0, 1] - c["t"][0, 0])
    alive = c["alive"] > 0.5
    ok = alive.all(1)
    print(f"== {a.set} : {n} flights x {K} steps ({K * dt:.0f} s, dt = {dt:.3f} s)")

    # 1. validity
    print(f"[1] non-finite values: {int((~np.isfinite(D)).sum())} (in the valid part: "
          f"{int((~np.isfinite(D[alive])).sum())})")
    print(f"[1] flights that stayed in the envelope to the end: {ok.sum()}/{n}")
    for names, key in ((TYPE_NAMES, "type"), (LAT_NAMES, "lat_mode")):
        print("     " + " ; ".join(f"{nm} {ok[M[key] == i].sum()}/{(M[key] == i).sum()}" for i, nm in enumerate(names)))
    if (~ok).any():
        print(f"     exits: median at t = {np.nanmedian(M['t_dead'][~ok]):.1f} s ; fraction of flight time lost "
              f"{1 - alive.mean():.3f}")
    print(f"     fraction of time under envelope protection: {c['protect'][alive].mean():.3f}")

    # 2. ranges covered (valid samples)
    v = alive
    print("[2] percentiles 1 / 50 / 99 (valid samples):")
    for k, f, u in [("Va", 1, "m/s"), ("alpha", DEG, "deg"), ("beta", DEG, "deg"), ("phi", DEG, "deg"),
                    ("theta", DEG, "deg"), ("p", DEG, "deg/s"), ("q", DEG, "deg/s"), ("r", DEG, "deg/s"),
                    ("h", 1, "m"), ("de", DEG, "deg"), ("da", DEG, "deg"), ("dr", DEG, "deg"),
                    ("dt_cmd", 1, "-"), ("P_motor", 1, "W"), ("turb_w", 1, "m/s")]:
        x = c[k][v] * f
        print(f"     {k:8s} [{u:5s}] {np.percentile(x, 1):8.2f} {np.percentile(x, 50):8.2f} {np.percentile(x, 99):8.2f}")
    print(f"     engine off: {c['engine_off'][v].mean():.2f} of the time ; open loop: {c['open_loop'][v].mean():.2f} ; "
          f"|phi| > 20 deg : {(np.abs(c['phi'][v]) > 20 / DEG).mean():.2f}")
    print(f"     alpha > 12 deg: {(c['alpha'][v] > 12 / DEG).mean():.4f} of the samples, "
          f"{int(((c['alpha'] > 12 / DEG) & v).any(1).sum())} flights")
    turn = np.unwrap(c["psi"], axis=1)
    print(f"     median net heading change per flight: {np.median(np.abs(turn[:, -1] - turn[:, 0])) * DEG:.0f} deg")

    # 3. energy balance (exact for these equations): dE/dt = F.v + M.omega
    #    E = 1/2 m |v|^2 + 1/2 w^T I w + m g h_air, h_air = h + mean w_d * t
    m = M["mass"][:, None]
    Ixx, Iyy, Izz, Ixz = (M[k][:, None] for k in ("Ixx", "Iyy", "Izz", "Ixz"))
    p_, q_, r_ = c["p"], c["q"], c["r"]
    Erot = 0.5 * (Ixx * p_**2 + Iyy * q_**2 + Izz * r_**2) - Ixz * p_ * r_
    h_air = c["h"] + M["wind_steady_d"][:, None] * c["t"]
    E = 0.5 * m * (c["u"]**2 + c["v"]**2 + c["w"]**2) + Erot + m * G * h_air
    Pw = ((c["Fx_aero"] + c["thrust"]) * c["u"] + c["Fy_aero"] * c["v"] + c["Fz_aero"] * c["w"]
          + (c["Mx_aero"] + c["Mx_prop"]) * p_ + (c["My_aero"] + c["My_prop"]) * q_ + (c["Mz_aero"] + c["Mz_prop"]) * r_)
    dE = (E[:, 2:] - E[:, :-2]) / (2 * dt)
    vv = v[:, 2:] & v[:, :-2]
    res = dE[vv] - Pw[:, 1:-1][vv]
    print(f"[3] energy balance: RMS(dE/dt - P) = {np.sqrt(np.mean(res**2)):.2f} W for RMS(P) = "
          f"{np.sqrt(np.mean(Pw[:, 1:-1][vv]**2)):.2f} W ; correlation = {np.corrcoef(dE[vv], Pw[:, 1:-1][vv])[0, 1]:.4f}")

    # 4. glide: engine off for > 4 s, wings nearly level, weak vertical wind
    eo = c["engine_off"] > 0.5
    run = np.zeros_like(eo, int)
    for k in range(1, K):
        run[:, k] = np.where(eo[:, k], run[:, k - 1] + 1, 0)
    gl = v & (run * dt > 4) & (np.abs(c["wind_d"]) < 0.3) & (np.abs(c["phi"]) < 10 / DEG)
    if gl.any():
        sink = -(np.gradient(c["h"], dt, axis=1) + c["wind_d"])[gl]
        print(f"[4] glide ({gl.sum()} samples): median Va {np.median(c['Va'][gl]):.1f} m/s, sink rate "
              f"{np.median(sink):.2f} m/s, median L/D {np.median((c['lift'] / np.maximum(c['drag'], 1e-6))[gl]):.1f}")

    # 5. control signs (control surface positions centred per flight)
    def corr(x, y):
        x = x - np.nanmean(np.where(v, x, np.nan), 1, keepdims=True)
        return np.corrcoef(x[v], y[v])[0, 1]
    dp, dq, dr_ = (np.gradient(c[k], dt, axis=1) for k in ("p", "q", "r"))
    print(f"[5] corr(elevator, dq/dt) {corr(c['de'], dq):+.3f} (expected < 0) ; corr(aileron, dp/dt) "
          f"{corr(c['da'], dp):+.3f} (< 0) ; corr(rudder, dr/dt) {corr(c['dr'], dr_):+.3f} (< 0) ; "
          f"corr(throttle, thrust) {np.corrcoef(c['dt_cmd'][v], c['thrust'][v])[0, 1]:+.3f} (> 0)")
    print(f"[6] mean motor energy per flight: {np.mean(c['E_motor'][ok, -1]) / 3600:.2f} Wh")

    # figures
    out = Path(a.dir) / "checks" / a.set
    out.mkdir(parents=True, exist_ok=True)
    picks = []
    for i in range(4):
        for j in range(3):
            idx = np.flatnonzero((M["type"] == i) & (M["lat_mode"] == j))
            if len(idx):
                picks.append(int(idx[0]))
    for i in picks[:8]:
        fig, ax = plt.subplots(5, 1, figsize=(9, 11), sharex=True)
        t = c["t"][i]
        ax[0].plot(t, c["Va"][i]); ax[0].set_ylabel("Va [m/s]")
        ax[1].plot(t, c["h"][i]); ax[1].set_ylabel("h [m]")
        for k in ("theta", "alpha", "phi", "beta"):
            ax[2].plot(t, c[k][i] * DEG, label=k)
        ax[2].plot(t, c["phi_ref"][i] * DEG, "k:", label="phi_ref"); ax[2].legend(ncol=5); ax[2].set_ylabel("deg")
        for k in ("de", "da", "dr"):
            ax[3].plot(t, c[k][i] * DEG, label=k)
        ax[3].plot(t, 10 * c["dt_cmd"][i], label="10 x throttle")
        ax[3].fill_between(t, -5, 5, where=c["engine_off"][i] > 0.5, alpha=0.2, label="engine off")
        ax[3].fill_between(t, -8, -5, where=c["protect"][i] > 0.5, color="r", alpha=0.3, label="protection")
        ax[3].legend(ncol=3); ax[3].set_ylabel("deg")
        for k in ("wind_n", "wind_e", "wind_d", "turb_w"):
            ax[4].plot(t, c[k][i], label=k, lw=0.8)
        ax[4].legend(ncol=4); ax[4].set_xlabel("t [s]")
        fig.suptitle(f"flight {i} - {TYPE_NAMES[M['type'][i]]} / {LAT_NAMES[M['lat_mode'][i]]} - "
                     f"turbulence {M['turb_level'][i]}")
        fig.tight_layout(); fig.savefig(out / f"flight_{i:04d}.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for i in picks[:8]:
        ax[0].plot(c["east"][i][alive[i]], c["north"][i][alive[i]], lw=0.8)
    ax[0].set_aspect("equal"); ax[0].set_xlabel("east [m]"); ax[0].set_ylabel("north [m]")
    ax[0].set_title("trajectories, top view")
    ax[1].hist(c["phi"][v] * DEG, 80); ax[1].set_title("bank angle phi [deg]")
    fig.tight_layout(); fig.savefig(out / "overview.png", dpi=150); plt.close(fig)
    print(f"figures: {out}")


if __name__ == "__main__":
    main()
