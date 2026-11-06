"""Figures and Table III of SkyJEPA (arXiv 2606.23444), redone on the fixed-wing runs of the SkyJEPA settings,
from the <run>/eval_skyjepa_<set>.json files of scripts/eval_skyjepa.py (mean over seeds, whiskers = min/max).

  <out>/open_loop.png        position, velocity and attitude RMSE vs prediction time (their Fig. 8)
  <out>/rollout_error.png    compounding ratio vs prediction time (their Fig. 6), and its two terms: open-loop
                             and one-step (teacher-forced) state RMSE, log scale
  <out>/noise.png            median state RMSE over the first second vs observation noise (their Fig. 9)
  <out>/straightening.png    temporal straightening of the predicted latent trajectories (their Fig. 7c);
                             c_jepa runs left out, their latent is the one of their frozen trunk
  stdout                     Table III: position RMSE [m] and attitude error [deg] at 1 s and --at-s seconds

Usage: python scripts/plot_skyjepa.py --runs runs --out docs/figures/skyjepa
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# run name without the seed -> (label, their row in Table III), in plotting order
LABELS = {
    "a0_calm_sky_pred": ("Predictive: one-step state loss", "Predictive"),
    "a0_calm_sky": ("A0: state loss, 1 s rollouts", "Recon + Prober (closest)"),
    "a0_calm_sky_physreg": ("A0 + physics in the loss", "Pred + Phys Reg (closest)"),
    "c_jepa_calm_sky_a0": ("A0 trunk frozen + physics probe", "Recon + PI Prober (closest)"),
    "jepa_calm_sky": ("JEPA frozen + MLP probe", "Ours + Prober"),
    "c_jepa_calm_sky_nul": ("JEPA frozen + physics probe", "Ours + PI Prober"),
    "jepa_ft_calm_sky_ft": ("JEPA + fine-tuning", "(not in the paper)"),
    "c_direct_calm_sky_nul": ("Physics head trained end to end", "(not in the paper)"),
    "c_direct_calm_sky_porth": ("Port-Hamiltonian head trained end to end", "(not in the paper)"),
}
GROUPS = {"avions_entrainement": "training-range aircraft", "avions_de_test": "extreme aircraft"}


def load(runs, set_name):
    """{base: {group: {metric: array (seeds, L)}}} and scalar metrics."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for f in sorted(Path(runs).glob(f"*/eval_skyjepa_{set_name}.json")):
        base = re.sub(r"_s\d+$", "", f.parent.name)
        if base not in LABELS:
            continue
        for g, r in json.load(open(f)).items():
            for k in ("position_rmse_m", "velocity_rmse_ms", "attitude_rmse_deg", "compounding_ratio",
                      "error_growth", "state_rmse_norm", "state_rmse_norm_teacher_forced"):
                out[base][g][k].append(np.array(r[k], float))
            out[base][g]["straightening"].append(r["straightening"]["mean"])
            sig = sorted(r["noise_robustness"], key=float)
            out[base][g]["sigmas"] = [float(s) for s in sig]
            out[base][g]["noise"].append([r["noise_robustness"][s]["median"] for s in sig])
            out[base][g]["dt"] = r["dt"]
    return out


def curve(ax, t, ys, label):
    y = np.stack(ys)
    ax.plot(t, y.mean(0), label=label, lw=1.6)
    if len(y) > 1:
        ax.fill_between(t, y.min(0), y.max(0), alpha=0.18)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--set", default="test_long")
    ap.add_argument("--out", default="docs/figures/skyjepa")
    ap.add_argument("--at-s", type=float, default=5.0)
    a = ap.parse_args()
    res = load(a.runs, a.set)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    order = [b for b in LABELS if b in res]

    for g, gl in GROUPS.items():
        fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
        for b in order:
            if g not in res[b]:
                continue
            r = res[b][g]
            t = (np.arange(len(r["position_rmse_m"][0])) + 1) * r["dt"]
            for ax, k in zip(axs, ("position_rmse_m", "velocity_rmse_ms", "attitude_rmse_deg")):
                curve(ax, t, r[k], LABELS[b][0])
        for ax, yl in zip(axs, ("position RMSE [m]", "velocity RMSE [m/s]", "attitude error [deg]")):
            ax.set_xlabel("prediction time [s]")
            ax.set_ylabel(yl)
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.suptitle(f"Open-loop error vs prediction time, calm air, {gl}")
        fig.tight_layout()
        fig.savefig(out / f"open_loop_{g}.png", dpi=150)
        plt.close(fig)

        # right panel: the two terms of the ratio, so that a low CR from a poor one-step error is visible
        fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
        for i, b in enumerate(order):
            if g not in res[b]:
                continue
            r = res[b][g]
            t = (np.arange(len(r["compounding_ratio"][0])) + 1) * r["dt"]
            curve(axs[0], t, r["compounding_ratio"], LABELS[b][0])
            c = f"C{i}"
            axs[1].plot(t, np.mean(r["state_rmse_norm"], 0), color=c, lw=1.6, label=LABELS[b][0])
            axs[1].plot(t, np.mean(r["state_rmse_norm_teacher_forced"], 0), color=c, lw=1.2, ls="--")
        axs[0].axhline(1, color="k", lw=0.8, ls="--")
        axs[0].set_ylabel("compounding ratio CR$_k$")
        axs[1].set_yscale("log")
        axs[1].set_ylabel("normalized state RMSE (log)")
        axs[1].set_title("solid: open-loop rollout   dashed: one step from the true past", fontsize=8)
        for ax in axs:
            ax.set_xlabel("prediction time [s]")
            ax.grid(alpha=0.3)
        axs[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"rollout_error_{g}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(5.5, 3.6))
        for b in order:
            if g in res[b]:
                r = res[b][g]
                y = np.array(r["noise"])
                ax.errorbar(r["sigmas"], y.mean(0), yerr=[y.mean(0) - y.min(0), y.max(0) - y.mean(0)],
                            marker="o", ms=3, capsize=2, label=LABELS[b][0])
        ax.set_xlabel("observation noise on the past states [std]")
        ax.set_ylabel("median state RMSE over 1 s (normalized)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"noise_{g}.png", dpi=150)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6, 3.4))
        names = [b for b in order if g in res[b] and not b.startswith("c_jepa")]   # c_jepa: latent of its trunk
        vals = [np.array(res[b][g]["straightening"]) for b in names]
        ax.barh(range(len(names)), [v.mean() for v in vals],
                xerr=[[v.mean() - v.min() for v in vals], [v.max() - v.mean() for v in vals]], capsize=2)
        ax.set_yticks(range(len(names)), [LABELS[b][0] for b in names], fontsize=7)
        ax.set_xlabel("temporal straightening (mean cosine of latent velocities)")
        ax.axvline(0, color="k", lw=0.8)
        fig.tight_layout()
        fig.savefig(out / f"straightening_{g}.png", dpi=150)
        plt.close(fig)

    for g, gl in GROUPS.items():
        print(f"\nTable III, {gl} (mean over seeds, min-max):")
        print(f"| Model | Their row | Position 1 s (m) | Position {a.at_s:g} s (m) | Attitude 1 s (deg) "
              f"| Attitude {a.at_s:g} s (deg) | CR at {a.at_s:g} s | Seeds |")
        print("|---|---|---|---|---|---|---|---|")
        for b in order:
            if g not in res[b]:
                continue
            r = res[b][g]
            k1, kL = int(round(1 / r["dt"])) - 1, int(round(a.at_s / r["dt"])) - 1
            cell = lambda k, i: (lambda v: f"{v.mean():.2f} ({v.min():.2f}-{v.max():.2f})")(
                np.array([y[i] for y in r[k]]))
            print(f"| {LABELS[b][0]} | {LABELS[b][1]} | {cell('position_rmse_m', k1)} | {cell('position_rmse_m', kL)} "
                  f"| {cell('attitude_rmse_deg', k1)} | {cell('attitude_rmse_deg', kL)} "
                  f"| {cell('compounding_ratio', kL)} | {len(r['position_rmse_m'])} |")


if __name__ == "__main__":
    main()
