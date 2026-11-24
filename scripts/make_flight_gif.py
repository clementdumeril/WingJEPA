"""Animated GIF of closed-loop flights: the same aircraft and targets flown by several controllers.

Reads the `<out>_traces.npz` written by scripts/closed_loop.py. Left: 3D path with the aircraft drawn (enlarged) at its
attitude; right: altitude and airspeed against their targets. A frozen flight (envelope exit) is marked lost.

  python scripts/make_flight_gif.py runs/closed_loop/tsweep_traces.npz --aircraft 25 \
      --controllers a0T1_adv_s0 a0_adv_s0 --labels "MPPI, model trained on 1 step" "MPPI, model trained on 20 steps" \
      --out docs/figures/flight_mppi.gif
"""
import argparse
from pathlib import Path

import numpy as np

DT = 0.05
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, MUTED, GRID, REF = "#1f1f1e", "#6b6a64", "#e4e3de", "#9a9993"

# aircraft outline in body axes (x forward, y right, z down), in meters before scaling
WING = np.array([[0.3, 0, 0], [-0.1, 0.75, 0], [-0.25, 0.75, 0], [-0.15, 0, 0], [-0.25, -0.75, 0], [-0.1, -0.75, 0]])
BODY = np.array([[0.55, 0, 0], [-0.6, 0.05, 0], [-0.6, -0.05, 0]])
TAIL_H = np.array([[-0.45, 0, 0], [-0.6, 0.25, 0], [-0.65, 0.25, 0], [-0.65, -0.25, 0], [-0.6, -0.25, 0]])
TAIL_V = np.array([[-0.45, 0, 0], [-0.65, 0, -0.22], [-0.68, 0, -0.22], [-0.65, 0, 0]])


def dcm(phi, th, psi):
    """Body to NED."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    return np.array([[ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp],
                     [ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp],
                     [-st, sf * ct, cf * ct]])


def to_plot(ned):
    """NED -> plot axes (east, north, up)."""
    return np.stack([ned[..., 1], ned[..., 0], -ned[..., 2]], -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traces")
    ap.add_argument("--controllers", nargs="+", required=True, help="tags in the traces file (run directory names)")
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--aircraft", type=int, default=0, help="index of the aircraft in the batch")
    ap.add_argument("--t0", type=float, default=0.0)
    ap.add_argument("--t1", type=float, default=None)
    ap.add_argument("--every", type=int, default=4, help="one frame every N steps of 50 ms")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--scale", type=float, default=None, help="drawing scale of the aircraft (default: 10 %% of the view)")
    ap.add_argument("--dpi", type=int, default=80)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    d = np.load(a.traces)
    i = a.aircraft
    labels = a.labels or a.controllers
    K = d[f"{a.controllers[0]}__h"].shape[1]
    k0, k1 = int(a.t0 / DT), int((a.t1 or K * DT) / DT)
    ks = np.arange(k0, k1)
    tr = {c: {k: d[f"{c}__{k}"][i, ks] for k in ("north", "east", "h", "phi", "theta", "psi", "Va", "h_ref", "V_ref")}
          for c in a.controllers}
    alive = {c: d[f"{c}__alive"][i, ks] for c in a.controllers}
    t = ks * DT
    ref = tr[a.controllers[0]]
    group = "extreme aircraft" if d["is_test"][i] else "training-range aircraft"

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig = plt.figure(figsize=(9.6, 4.4))
    ax3 = fig.add_axes([0.0, 0.02, 0.55, 0.9], projection="3d")
    axh = fig.add_axes([0.63, 0.56, 0.35, 0.34])
    axv = fig.add_axes([0.63, 0.12, 0.35, 0.34])
    pts = np.concatenate([np.stack([tr[c]["east"], tr[c]["north"], tr[c]["h"]], 1) for c in a.controllers])
    pts = pts[np.isfinite(pts).all(1)]
    lo, hi = pts.min(0), pts.max(0)
    span = max(hi[0] - lo[0], hi[1] - lo[1], 50.0) / 2
    a.scale = a.scale or 0.2 * span
    span += a.scale
    mid = (lo + hi) / 2
    ax3.set_xlim(mid[0] - span, mid[0] + span)
    ax3.set_ylim(mid[1] - span, mid[1] + span)
    ax3.set_zlim(lo[2] - 10, hi[2] + 10)
    ax3.set_box_aspect((1, 1, 0.45), zoom=1.25)
    ax3.set_xlabel("east (m)", labelpad=-2)
    ax3.set_ylabel("north (m)", labelpad=-2)
    ax3.set_zlabel("altitude (m)", labelpad=-2)
    ax3.tick_params(labelsize=7, pad=-1)
    for pane in (ax3.xaxis.pane, ax3.yaxis.pane, ax3.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0))
        pane.set_edgecolor(GRID)
    ax3.view_init(elev=22, azim=-60)

    for ax, key, rkey, ylab in [(axh, "h", "h_ref", "altitude (m)"), (axv, "Va", "V_ref", "airspeed (m/s)")]:
        ax.plot(t, ref[rkey], color=REF, lw=1.5, ls="--", label="target")
        ax.set_xlim(t[0], t[-1])
        vals = np.concatenate([tr[c][key] for c in a.controllers] + [ref[rkey]])
        vals = vals[np.isfinite(vals)]
        pad = 0.08 * (vals.max() - vals.min() + 1)
        ax.set_ylim(vals.min() - pad, vals.max() + pad)
        ax.set_ylabel(ylab)
        ax.grid(color=GRID, lw=0.7)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axv.set_xlabel("time (s)")

    trails, glyphs, lines_h, lines_v, status = [], [], [], [], []
    for j, c in enumerate(a.controllers):
        col = COLORS[j % len(COLORS)]
        trails.append(ax3.plot([], [], [], color=col, lw=1.6)[0])
        g = Poly3DCollection([], facecolor=col, edgecolor=INK, linewidths=0.4, alpha=0.95)
        ax3.add_collection3d(g)
        glyphs.append(g)
        lines_h.append(axh.plot([], [], color=col, lw=1.8, label=labels[j])[0])
        lines_v.append(axv.plot([], [], color=col, lw=1.8)[0])
        status.append(fig.text(0.02, 0.86 - 0.05 * j, "", color=INK, fontsize=9))
    cur_h, cur_v = axh.axvline(t[0], color=MUTED, lw=0.8), axv.axvline(t[0], color=MUTED, lw=0.8)
    axh.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, 1.32), ncol=1)
    title = fig.text(0.02, 0.95, "", color=INK, fontsize=10.5)
    frames = list(range(0, len(ks), a.every)) + [len(ks) - 1] * a.fps

    def draw(f):
        title.set_text(f"Closed loop, {group}, t = {t[f]:4.1f} s (aircraft drawn {a.scale:.0f}x larger)")
        for j, c in enumerate(a.controllers):
            x = tr[c]
            ok = np.isfinite(x["h"][:f + 1])
            last = np.flatnonzero(ok)[-1] if ok.any() else 0
            trails[j].set_data_3d(x["east"][:f + 1][ok], x["north"][:f + 1][ok], x["h"][:f + 1][ok])
            C = dcm(x["phi"][last], x["theta"][last], x["psi"][last])
            pos = np.array([x["north"][last], x["east"][last], -x["h"][last]])
            glyphs[j].set_verts([to_plot(pos + a.scale * (C @ P.T).T) for P in (WING, BODY, TAIL_H, TAIL_V)])
            lines_h[j].set_data(t[:f + 1], x["h"][:f + 1])
            lines_v[j].set_data(t[:f + 1], x["Va"][:f + 1])
            lost = not alive[c][f]
            status[j].set_text(f"{labels[j]}: {'lost (left the flight envelope)' if lost else 'flying'}")
            status[j].set_color("#c0392b" if lost else INK)
        cur_h.set_xdata([t[f]])
        cur_v.set_xdata([t[f]])
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    anim.save(a.out, writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    print("written", a.out, f"{Path(a.out).stat().st_size / 1e6:.1f} MB, {len(frames)} frames")


if __name__ == "__main__":
    main()
