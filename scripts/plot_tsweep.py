"""Unroll-length sweep: one-step error and 3 s position error against the training unroll length T.

Runs `<family>T<t>_adv_s<seed>` (scripts/train_a.py --T t --T-max t, then scripts/eval_skyjepa.py), three seeds, test
aircraft of data_advanced. T = 1 of the latent model is `pred_adv`, T = 20 is the default run of each family.

  python scripts/plot_tsweep.py --collect runs     # on the cluster: eval_skyjepa_test_long.json -> results/tsweep.csv
  python scripts/plot_tsweep.py                    # results/tsweep.csv -> docs/figures/tsweep.png
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILIES = {
    "mlp_direct": {1: "mlpT1", 2: "mlpT2", 5: "mlpT5", 10: "mlpT10", 20: "mlp", 40: "mlpT40"},
    "a0": {1: "pred", 2: "a0T2", 5: "a0T5", 10: "a0T10", 20: "a0", 40: "a0T40"},
    "jepa": {1: "jepaT1", 5: "jepaT5", 10: "jepaT10", 20: "jepa"},
}
LABEL = {"mlp_direct": "MLP on states", "a0": "latent model, prediction loss", "jepa": "latent model, JEPA loss"}
STYLE = {"mlp_direct": ("#2a78d6", "o"), "a0": ("#eb6834", "s"), "jepa": ("#1baf7a", "^")}
COLS = ["family", "T", "seed", "one_step", "err_1s", "pos_1p5s_m", "pos_3s_m"]


def collect(runs, out):
    rows = []
    for fam, names in FAMILIES.items():
        for T, name in names.items():
            for seed in (0, 1, 2):
                f = Path(runs) / f"{name}_adv_s{seed}" / "eval_skyjepa_test_long.json"
                if not f.exists():
                    continue
                r = json.load(open(f))["avions_de_test"]
                rows.append([fam, T, seed, r["state_rmse_norm_teacher_forced"][0], r["state_rmse_norm"][19],
                             r["position_rmse_m"][29], r["position_rmse_m"][59]])
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(COLS)
        w.writerows([[x if isinstance(x, (str, int)) else f"{x:.4f}" for x in row] for row in rows])
    print(f"{len(rows)} runs -> {out}")


def plot(csv_path, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = {}
    for r in csv.DictReader(open(csv_path)):
        data.setdefault(r["family"], {}).setdefault(int(r["T"]), []).append(
            (float(r["one_step"]), float(r["pos_3s_m"])))
    ink, muted, grid = "#1f1f1e", "#6b6a64", "#e4e3de"
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": muted, "axes.labelcolor": ink, "xtick.color": muted,
                         "ytick.color": muted, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), constrained_layout=True)
    for i, (ax, title, ylab) in enumerate([(axes[0], "(a) One-step error", "normalized state RMSE, one step"),
                                           (axes[1], "(b) Error after 3 s of rollout", "position RMSE at 3 s (m)")]):
        for fam in FAMILIES:
            Ts = sorted(data.get(fam, {}))
            if not Ts:
                continue
            v = [np.array([x[i] for x in data[fam][T]]) for T in Ts]
            m, s = np.array([x.mean() for x in v]), np.array([x.std() for x in v])
            c, mk = STYLE[fam]
            ax.plot(Ts, m, color=c, lw=2, marker=mk, ms=7, mec="white", mew=1.2, label=LABEL[fam], zorder=3)
            if i == 1:
                ax.fill_between(Ts, m - s, m + s, color=c, alpha=0.18, lw=0, zorder=2)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 40])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "40"])
        ax.minorticks_off()
        ax.set_xlabel("training unroll length T (steps of 50 ms)")
        ax.set_ylabel(ylab)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", color=grid, lw=0.8, zorder=0)
        ax.set_title(title, loc="left", color=ink, fontsize=10.5)
    axes[0].legend(frameon=False, fontsize=8.5, loc="center right")
    fig.savefig(out, dpi=170)
    print("written", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", default=None, help="runs directory to read the evaluations from")
    ap.add_argument("--csv", default=str(ROOT / "results" / "tsweep.csv"))
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures" / "tsweep.png"))
    a = ap.parse_args()
    if a.collect:
        collect(a.collect, a.csv)
    else:
        plot(a.csv, a.out)


if __name__ == "__main__":
    main()
