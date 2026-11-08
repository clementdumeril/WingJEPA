"""Closed-loop figure: tracking error, survival and command activity of each controller vs sensor noise.

Reads runs/closed_loop/sweep_nz<noise>_s<seed>.json (scripts/closed_loop.py), averages over the model seeds
(bars = min-max) and over the two aircraft groups unless --group is given. A controller whose survival is
below --min-survival at a noise level is left out of the error panels there (its errors only cover the
time before the loss), but stays in the survival panel.

Usage: python scripts/plot_closed_loop.py --dir runs/closed_loop --out docs/figures/closed_loop_noise.pdf
"""
import argparse
import collections
import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LABELS = {   # order of the legend; MPPI with each world model, except TECS
    "tecs": "TECS autopilot (no model)",
    "c_direct_calm_sky_nul": "physics head, end to end",
    "c_jepa_calm_sky_nul": "JEPA frozen + physics probe",
    "a0_calm_sky": "A0 + MLP (black box)",
    "a0_calm_sky_noiseaug": "A0 + MLP, noise-augmented",
    "jepa_calm_sky": "JEPA frozen + MLP probe",
}
COLORS = {   # same colors as Fig. 2 (scripts/plot_paper.py)
    "tecs": "k", "c_direct_calm_sky_nul": "C6", "c_jepa_calm_sky_nul": "C4", "a0_calm_sky": "C1",
    "a0_calm_sky_noiseaug": "C8", "jepa_calm_sky": "C3",
}
GROUPS = ("avions_entrainement", "avions_de_test")
KEYS = ("rms_V_ms", "rms_h_m", "survival", "mean_abs_dcmd")


def load(d, group):
    """{controller: {noise: (n_seeds, 4) array of KEYS}}"""
    R = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in glob.glob(os.path.join(d, "sweep_nz*_s*.json")):
        nz = float(Path(f).stem.split("sweep_nz")[1].rsplit("_s", 1)[0])
        res = json.load(open(f))
        for c in (k for k in res if k != "args"):
            m = "tecs" if c == "tecs" else os.path.basename(c.rstrip("/")).rsplit("_s", 1)[0]
            gs = GROUPS if group == "both" else (group,)
            R[m][nz].append([np.mean([res[c][g][k] for g in gs]) for k in KEYS])
    return {m: {nz: np.array(v) for nz, v in sorted(d_.items())} for m, d_ in R.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="runs/closed_loop")
    ap.add_argument("--out", default="docs/figures/closed_loop_noise.pdf")
    ap.add_argument("--group", default="both", choices=("both",) + GROUPS)
    ap.add_argument("--min-survival", type=float, default=0.5)
    a = ap.parse_args()
    R = load(a.dir, a.group)
    order = [m for m in LABELS if m in R] + [m for m in R if m not in LABELS]
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.2})
    fig, axs = plt.subplots(1, 5, figsize=(7.0, 1.55), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.15]))
    panels = [(0, "airspeed RMS error [m/s]", False), (1, "altitude RMS error [m]", False),
              (2, "survival", True), (3, "command change per step", True)]
    for ax, (j, ylabel, keep_all) in zip(axs[:4], panels):
        for i, m in enumerate(order):
            nzs = [nz for nz in R[m] if keep_all or R[m][nz][:, 2].mean() >= a.min_survival]
            if not nzs:
                continue
            y = np.array([R[m][nz][:, j].mean() for nz in nzs])
            lo = y - np.array([R[m][nz][:, j].min() for nz in nzs])
            hi = np.array([R[m][nz][:, j].max() for nz in nzs]) - y
            ax.errorbar(nzs, y, yerr=[lo, hi], color=COLORS.get(m, f"C{i}"), marker="o", ms=2,
                        capsize=1.5, lw=1.0, elinewidth=0.6, ls="--" if m == "tecs" else "-",
                        label=LABELS.get(m, m))
        ax.set_xlabel("sensor noise [std]", labelpad=1)
        ax.set_ylabel(ylabel, labelpad=1)
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(length=2, pad=1)
    axs[3].set_yscale("log")
    axs[2].set_ylim(-0.05, 1.05)
    axs[4].axis("off")
    h, l = axs[2].get_legend_handles_labels()
    axs[4].legend(h, l, loc="center left", fontsize=6, frameon=False, handlelength=1.6, borderaxespad=0,
                  title="controller (MPPI with model)", title_fontsize=6)
    fig.tight_layout(pad=0.2, w_pad=0.5)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=300)
    for m in order:   # the numbers behind the figure
        for nz, v in R[m].items():
            print(f"{m:24s} noise {nz:4.2f}  {len(v)} seeds  " + "  ".join(
                f"{k} {v[:, j].mean():.4g}" for j, k in enumerate(KEYS)))
    print("written", a.out)


if __name__ == "__main__":
    main()
