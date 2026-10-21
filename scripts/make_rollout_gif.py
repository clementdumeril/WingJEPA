"""Hero animation: open-loop predictions of several world models against the simulator, on one flight.

Each model gets the same 2 s of measured history, then predicts alone for --horizon-s seconds, receiving
only the commands actually applied and the last wind estimate (as in scripts/evaluate.py). The start
point is drawn at random (--seed) among the valid starts of the chosen group, not hand-picked.

Writes <out>.gif (about 800 px wide, 12 fps) and <out>.png (last frame).

Usage: python scripts/make_rollout_gif.py --data-dir <data_dir> \
           --models runs/a0_v2T100_s0 runs/c_direct_v2T100_ajuste_s0 --labels "A0" "C-direct, fitted prior"
"""
import argparse
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
from wingjepa.data.windows import STATE, aircraft_params, features, load_set  # noqa: E402
from wingjepa.models.jepa import NOMINAL_AIRCRAFT  # noqa: E402

PANELS = [("h", "altitude change [m]", 1.0), ("Va", "airspeed [m/s]", 1.0),
          ("phi", "bank angle [deg]", 180 / np.pi), ("theta", "pitch angle [deg]", 180 / np.pi)]


def starts(alive, n_past, n_future, stride):
    out = []
    for f in range(alive.shape[0]):
        for s in range(0, alive.shape[1] - n_past - n_future + 1, stride):
            if alive[f, s:s + n_past + n_future].all():
                out.append((f, s))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--set", default="test_long")
    ap.add_argument("--group", choices=["extreme", "train"], default="extreme")
    ap.add_argument("--horizon-s", type=float, default=20.0)
    ap.add_argument("--stride-s", type=float, default=10.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--index", type=int, default=None, help="start index instead of a random draw")
    ap.add_argument("--steps-per-frame", type=int, default=2)
    ap.add_argument("--fps", type=int, default=12)
    ap.add_argument("--out", default="assets/rollout_extreme_aircraft")
    a = ap.parse_args()
    assert len(a.models) == len(a.labels), "one label per model"

    models = [load_model(d) for d in a.models]
    Hm = max(cfg["H"] for _, cfg, _ in models)
    L = int(a.horizon_s / DT)
    data, ch, meta = load_set(a.data_dir, a.set)
    X, alive, h = features(data, ch)
    P = aircraft_params(meta)
    test = meta["is_test_aircraft"].astype(bool)
    keep = test if a.group == "extreme" else ~test
    cand = [(f, s) for f, s in starts(alive, Hm, L, int(a.stride_s / DT)) if keep[f]]
    i = a.index if a.index is not None else int(np.random.default_rng(a.seed).integers(len(cand)))
    f, s = cand[i]
    seq = X[f:f + 1, s:s + Hm + L]
    t = (np.arange(Hm + L) - Hm + 1) * DT                                  # 0 = last measured step
    print(f"flight {f}, start {s * DT:.1f} s (candidate {i} of {len(cand)})")

    hdot = STATE.index("hdot")
    truth = {k: seq[0, :, STATE.index(k)] * sc for k, _, sc in PANELS if k != "h"}
    truth["h"] = h[f, s:s + Hm + L] - h[f, s + Hm - 1]
    preds = []
    for model, cfg, norm in models:
        H = cfg["H"]
        uses_ac = bool(cfg.get("aircraft_input")) or cfg.get("head") == "physics"
        p = predict(model, norm, seq[:, Hm - H:], H, True, P[f:f + 1] if uses_ac else None)[0]
        d = {k: p[:, STATE.index(k)] * sc for k, _, sc in PANELS if k != "h"}
        d["h"] = np.cumsum(p[:, hdot]) * DT
        preds.append(d)

    fig, axes = plt.subplots(len(PANELS), 1, figsize=(8, 7.2), sharex=True)
    dev = P[f] / np.asarray(NOMINAL_AIRCRAFT) - 1
    kind = "Extreme test aircraft" if a.group == "extreme" else "Training-range aircraft"
    fig.suptitle(f"{kind} (mass {dev[0]:+.0%}, roll inertia {dev[1]:+.0%}): open-loop prediction\n"
                 "from 2 s of measured history, commands known, wind held at last estimate", fontsize=10)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    lines = []
    for ax, (k, ylab, _) in zip(axes, PANELS):
        ax.plot(t[:Hm], truth[k][:Hm], color="0.6", lw=1.5)
        lt, = ax.plot([], [], color="k", lw=1.8, label="simulator (truth)")
        lp = [ax.plot([], [], color=colors[j], lw=1.5, label=lab)[0] for j, lab in enumerate(a.labels)]
        lines.append((k, lt, lp))
        allv = np.concatenate([truth[k]] + [d[k] for d in preds])
        pad = 0.1 * (np.ptp(allv) + 1e-6)
        ax.set_ylim(allv.min() - pad, allv.max() + pad)
        ax.axvline(0, color="0.4", ls=":", lw=1)
        ax.set_ylabel(ylab, fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8, loc="upper left", ncol=len(a.labels) + 1)
    axes[-1].set_xlabel("time since prediction start [s]")
    axes[-1].set_xlim(t[0], t[-1])
    clock = axes[0].text(0.99, 0.05, "", transform=axes[0].transAxes, ha="right", fontsize=9)
    fig.tight_layout()

    n_frames = L // a.steps_per_frame + 1

    def draw(n):
        k_ = min(n * a.steps_per_frame, L)
        for k, lt, lp in lines:
            lt.set_data(t[Hm - 1:Hm + k_], truth[k][Hm - 1:Hm + k_])
            for d, line in zip(preds, lp):
                start = truth[k][Hm - 1]
                line.set_data(t[Hm - 1:Hm + k_], np.concatenate([[start if k != "h" else 0.0], d[k][:k_]]))
        clock.set_text(f"t = {k_ * DT:4.1f} s")
        return []

    frames = list(range(n_frames)) + [n_frames - 1] * (2 * a.fps)          # hold the end for 2 s
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    FuncAnimation(fig, draw, frames=frames).save(out.with_suffix(".gif"), writer=PillowWriter(fps=a.fps), dpi=100)
    draw(n_frames - 1)
    fig.savefig(out.with_suffix(".png"), dpi=150)
    print(f"written {out.with_suffix('.gif')} ({out.with_suffix('.gif').stat().st_size / 1e6:.1f} MB) and .png")


if __name__ == "__main__":
    main()
