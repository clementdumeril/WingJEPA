"""Figure 2 of the one-page paper, at its printed size (7 x 1.4 in, 7 pt): position error vs prediction time on
training-range and extreme aircraft, noise robustness, and one shared legend. Reads the same
<run>/eval_skyjepa_<set>.json files as scripts/plot_skyjepa.py (mean over seeds, bands = min/max).

Usage: python scripts/plot_paper.py --runs runs --out docs/figures/paper_fig2.pdf
"""
import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_skyjepa import LABELS, load  # noqa: E402

SHORT = {   # legend of the paper, same order as LABELS
    "a0_calm_sky_pred": "Predictive",
    "a0_calm_sky": "A0 + MLP",
    "c_jepa_calm_sky_a0": "A0 frozen + physics probe",
    "jepa_calm_sky": "JEPA frozen + MLP probe",
    "c_jepa_calm_sky_nul": "JEPA frozen + physics probe",
    "jepa_ft_calm_sky_ft": "JEPA fine-tuned",
    "c_direct_calm_sky_nul": "Physics head, end to end",
    "c_direct_calm_sky_porth": "Port-Hamiltonian, end to end",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--set", default="test_long")
    ap.add_argument("--out", default="docs/figures/paper_fig2.pdf")
    a = ap.parse_args()
    res = load(a.runs, a.set)
    order = [b for b in LABELS if b in res]
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6, "ytick.labelsize": 6,
                         "axes.linewidth": 0.6, "lines.linewidth": 1.2})
    fig, axs = plt.subplots(1, 4, figsize=(7.0, 1.45), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.78]))
    panels = [("avions_entrainement", "training-range aircraft"), ("avions_de_test", "extreme aircraft")]
    for ax, (g, title) in zip(axs[:2], panels):
        for i, b in enumerate(order):
            if g not in res[b]:
                continue
            y = np.stack(res[b][g]["position_rmse_m"])
            t = (np.arange(y.shape[1]) + 1) * res[b][g]["dt"]
            ax.plot(t, y.mean(0), color=f"C{i}", label=SHORT.get(b, b))
            if len(y) > 1:
                ax.fill_between(t, y.min(0), y.max(0), color=f"C{i}", alpha=0.18, lw=0)
        ax.set_title(title, fontsize=7, pad=2)
        ax.set_xlabel("prediction time [s]", labelpad=1)
    axs[0].set_ylabel("position RMSE [m]", labelpad=1)
    ax = axs[2]
    for i, b in enumerate(order):
        r = res[b].get("avions_entrainement")
        if r is None:
            continue
        y = np.array(r["noise"])
        ax.errorbar(r["sigmas"], y.mean(0), yerr=[y.mean(0) - y.min(0), y.max(0) - y.mean(0)], color=f"C{i}",
                    marker="o", ms=2, capsize=1.5, lw=1.0, elinewidth=0.6)
    ax.set_title("noise robustness, training-range", fontsize=7, pad=2)
    ax.set_xlabel("observation noise [std]", labelpad=1)
    ax.set_ylabel("median state RMSE, 1 s", labelpad=1)
    for ax in axs[:3]:
        ax.grid(alpha=0.3, lw=0.4)
        ax.tick_params(length=2, pad=1)
    axs[3].axis("off")
    h, l = axs[0].get_legend_handles_labels()
    axs[3].legend(h, l, loc="center left", fontsize=6.3, frameon=False, handlelength=1.4, borderaxespad=0)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out)
    fig.savefig(Path(a.out).with_suffix(".png"), dpi=300)
    print("written", a.out)


if __name__ == "__main__":
    main()
