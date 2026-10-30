"""3D flight paths: where each world model thinks the aircraft goes, against the simulator (as in SkyJEPA).

Each model gets the same 2 s of measured history, then predicts alone for --horizon-s seconds (commands
known, wind held at its last estimate). Its predicted states are turned into a flight path exactly as
for the position error (scripts/position_error.py: heading from the predicted rates, air velocity rotated
to north-east plus the held wind, vertical speed integrated). One start is drawn at random (--seed) on a
training-range aircraft and one on an extreme test aircraft, not hand-picked.

Writes <out>.gif (paths drawn over time, slowly rotating view, under 5 MB), <out>.png (last frame) and
<out>.json (the paths, in meters from the start point, for other viewers).

Usage: python scripts/make_trajectory_3d.py --data-dir <data_dir> \
           --models runs/a0_calmT100_s0 runs/jepa_calmT100_s0 --labels "A0" "JEPA"
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from evaluate import DT, load_model, predict  # noqa: E402
from position_error import integrate, starts  # noqa: E402
from wingjepa.data.windows import aircraft_params, features, load_set  # noqa: E402

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # categorical slots, fixed order
PANELS = (("train", "Training-range aircraft"), ("extreme", "Extreme test aircraft"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--set", default="test_long")
    ap.add_argument("--horizon-s", type=float, default=20.0)
    ap.add_argument("--stride-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--title", default="Flight path rebuilt from open-loop predictions")
    ap.add_argument("--steps-per-frame", type=int, default=5)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--dpi", type=int, default=75)
    ap.add_argument("--out", default="assets/trajectory_3d")
    a = ap.parse_args()
    assert len(a.models) == len(a.labels) <= len(COLORS), "one label per model, at most 5 models"

    models = [load_model(d) for d in a.models]
    Hm = max(cfg["H"] for _, cfg, _ in models)
    L = int(a.horizon_s / DT)
    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, _ = features(data, ch)
    P = aircraft_params(meta)
    raw = {k: data[:, :, ch.index(k)].astype(np.float64) for k in ("psi", "north", "east", "h", "wind_n_est", "wind_e_est")}
    test = meta["is_test_aircraft"].astype(bool)
    rng = np.random.default_rng(a.seed)

    panels = []
    for kind, title in PANELS:
        keep = test if kind == "extreme" else ~test
        st = starts(alive & keep[:, None], Hm, L, int(a.stride_s / DT))
        f, s = st[rng.integers(len(st))]
        s0 = s + Hm - 1                                                    # last measured step
        pos = lambda i: np.stack([raw["east"][f, i], raw["north"][f, i], raw["h"][f, i]], -1)
        origin = pos(s0)
        hist = pos(np.arange(s, s0 + 1)) - origin
        truth = pos(np.arange(s0, s0 + L + 1)) - origin
        seq = X[f:f + 1, s:s + Hm + L]
        psi0, wind = raw["psi"][f:f + 1, s0], np.array([[raw["wind_n_est"][f, s0], raw["wind_e_est"][f, s0]]])
        pos0 = np.array([[raw["north"][f, s0], raw["east"][f, s0], raw["h"][f, s0]]])
        paths = []
        for model, cfg, norm in models:
            H = cfg["H"]
            uses_ac = bool(cfg.get("aircraft_input")) or cfg.get("head") == "physics"
            pred = predict(model, norm, seq[:, Hm - H:], H, True, P[f:f + 1] if uses_ac else None).astype(np.float64)
            neh = integrate(pred, psi0, pos0, wind)[0]                     # north, east, altitude
            paths.append(np.vstack([np.zeros(3), neh[:, [1, 0, 2]] - origin]))
        err = {lab: float(np.linalg.norm(p[-1] - truth[-1])) for lab, p in zip(a.labels, paths)}
        print(f"{title}: flight {f}, start {s * DT:.1f} s; position error at {a.horizon_s:.0f} s [m]: "
              + ", ".join(f"{k} {v:.0f}" for k, v in err.items()))
        panels.append(dict(title=title, flight=int(f), start_s=float(s * DT), hist=hist, truth=truth,
                           paths=paths, error_end_m=err))

    fig = plt.figure(figsize=(11, 5.2))
    fig.suptitle(f"{a.title}: 2 s of history (grey), then {a.horizon_s:.0f} s predicted with only the commands",
                 fontsize=10)
    axes, artists = [], []
    for i, pn in enumerate(panels):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        allp = np.vstack([pn["hist"], pn["truth"]] + pn["paths"])
        lo, hi = allp.min(0), allp.max(0)
        c, r = (lo[:2] + hi[:2]) / 2, (hi[:2] - lo[:2]).max() / 2 * 1.05 + 1   # same scale north and east
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        zpad = 0.1 * (hi[2] - lo[2]) + 1
        ax.set_zlim(lo[2] - zpad, hi[2] + zpad)
        ax.set_box_aspect((1, 1, 0.55))
        ax.set_xlabel("east [m]", fontsize=8)
        ax.set_ylabel("north [m]", fontsize=8)
        ax.set_zlabel("altitude change [m]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_title(pn["title"], fontsize=10)
        ax.plot(*pn["hist"].T, color="0.6", lw=2)
        lt, = ax.plot([], [], [], color="k", lw=2, label="simulator (truth)")
        ht, = ax.plot([], [], [], "o", color="k", ms=4)
        lines = [ax.plot([], [], [], color=COLORS[j], lw=1.6, label=lab)[0] for j, lab in enumerate(a.labels)]
        heads = [ax.plot([], [], [], "o", color=COLORS[j], ms=4)[0] for j in range(len(a.labels))]
        axes.append(ax)
        artists.append((pn, lt, ht, lines, heads))
    axes[0].legend(fontsize=8, loc="upper left")
    clock = fig.text(0.5, 0.03, "", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))

    n_frames = L // a.steps_per_frame + 1

    def draw(n):
        k = min(n * a.steps_per_frame, L) + 1
        for pn, lt, ht, lines, heads in artists:
            for line, head, p in [(lt, ht, pn["truth"])] + list(zip(lines, heads, pn["paths"])):
                line.set_data_3d(p[:k, 0], p[:k, 1], p[:k, 2])
                head.set_data_3d(p[k - 1:k, 0], p[k - 1:k, 1], p[k - 1:k, 2])
        for ax in axes:
            ax.view_init(elev=22, azim=-60 + 30 * n / max(n_frames - 1, 1))
        clock.set_text(f"t = {(k - 1) * DT:4.1f} s after the last measurement")
        return []

    frames = list(range(n_frames)) + [n_frames - 1] * int(1.5 * a.fps)     # hold the end for 1.5 s
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=frames).save(out.with_suffix(".gif"), writer=PillowWriter(fps=a.fps), dpi=a.dpi)
    draw(n_frames - 1)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    r1 = lambda x: np.round(x, 1).tolist()
    json.dump(dict(dt=DT, horizon_s=a.horizon_s, frame="east, north, altitude change [m] from the last measurement",
                   panels=[dict(title=pn["title"], flight=pn["flight"], start_s=pn["start_s"], history=r1(pn["hist"]),
                                truth=r1(pn["truth"]), error_end_m=pn["error_end_m"],
                                models={lab: r1(p) for lab, p in zip(a.labels, pn["paths"])}) for pn in panels]),
              open(out.with_suffix(".json"), "w"))
    size = out.with_suffix(".gif").stat().st_size / 1e6
    print(f"written {out.with_suffix('.gif')} ({size:.1f} MB), .png and .json"
          + ("  WARNING: over 5 MB, lower --dpi or raise --steps-per-frame" if size > 5 else ""))


if __name__ == "__main__":
    main()
