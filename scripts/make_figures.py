"""README figures from the evaluation files collected in results/ (scripts/collect_results.py).

  assets/useful_horizon.png       median useful horizon per model, long flights, training vs extreme
                                  aircraft (bar = mean over seeds, whiskers = min/max over seeds)
  assets/altitude_error.png       RMS altitude error vs prediction time, with the two trivial baselines

Usage: python scripts/make_figures.py [--results results] [--out assets]
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

# run name without the seed suffix -> label, in plotting order (runs not listed are ignored)
LABELS = {
    "a0_v2T100": "A0: direct state loss",
    "jepa_v2T100": "A: JEPA",
    "jepa_v2T100_ema": "A: JEPA, slow target encoder",
    "jepa_ft_v2T100": "A: JEPA pre-training + fine-tuning",
    "jepa_ft_v2T100_ema": "A: JEPA pre-training (slow target) + fine-tuning",
    "c_direct_v2T100_ajuste": "C-direct, fitted prior",
    "c_direct_v2T100_generique": "C-direct, generic prior",
    "c_direct_v2T100_faux": "C-direct, wrong prior",
    "c_direct_v2T100_nul": "C0-direct, no prior",
    "c_jepa_v2T100_ajuste": "C-JEPA, fitted prior",
    "c_jepa_v2T100_generique": "C-JEPA, generic prior",
    "c_jepa_v2T100_faux": "C-JEPA, wrong prior",
    "c_jepa_v2T100_nul": "C0-JEPA, no prior",
}
GROUPS = {"avions_entrainement": "Training-range aircraft", "avions_de_test": "Extreme test aircraft"}
TIMES = ["1s", "5s", "20s", "60s"]


def load(results):
    """{label: [eval_test_long of each seed]}"""
    runs = defaultdict(list)
    for d in sorted(Path(results).iterdir()):
        f = d / "eval_test_long.json"
        key = re.sub(r"_s\d+$", "", d.name)
        if f.exists() and key in LABELS:
            runs[LABELS[key]].append(json.load(open(f)))
    return {lab: runs[lab] for lab in LABELS.values() if lab in runs}


def stat(evals, group, who, field):
    v = []
    for e in evals:
        r = e[group]["par_turbulence"]["toutes"][who]
        v.append(r["horizon_utile_s"]["mediane"] if field == "horizon" else r["h"][field])
    return np.mean(v), np.min(v), np.max(v)


def useful_horizon(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 0.45 * len(runs) + 1.8), sharey=True)
    labels = list(runs)
    y = np.arange(len(labels))[::-1]
    for ax, (g, title) in zip(axes, GROUPS.items()):
        m = np.array([stat(runs[l], g, "modele_vent_estime", "horizon") for l in labels])
        ax.barh(y, m[:, 0], xerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], color="tab:blue", alpha=0.8, capsize=3)
        first = runs[labels[0]]
        for who, style, name in (("reference", "--", "constant state"),
                                 ("reference_altitude_constante", ":", "constant altitude")):
            ax.axvline(stat(first, g, who, "horizon")[0], color="k", ls=style, label=f"baseline: {name}")
        ax.set_title(title)
        ax.set_xlabel("median useful horizon [s]")
        ax.grid(axis="x", alpha=0.3)
    axes[0].set_yticks(y, labels)
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Open-loop prediction from 2 s of history (60 s long flights)")
    fig.tight_layout()
    fig.savefig(out / "useful_horizon.png", dpi=150)
    plt.close(fig)


def altitude_error(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    t = [int(s[:-1]) for s in TIMES]
    for ax, (g, title) in zip(axes, GROUPS.items()):
        for l, ev in runs.items():
            ax.plot(t, [stat(ev, g, "modele_vent_estime", s)[0] for s in TIMES], "o-", label=l, lw=1.2, ms=3)
        first = next(iter(runs.values()))
        for who, style, name in (("reference", "--", "baseline: constant state"),
                                 ("reference_altitude_constante", ":", "baseline: constant altitude")):
            ax.plot(t, [stat(first, g, who, s)[0] for s in TIMES], "k" + style, label=name)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(t, [str(x) for x in t])
        ax.set_xlabel("prediction time [s]")
        ax.set_title(title)
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("RMS altitude error [m]")
    axes[1].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "altitude_error.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="assets")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(a.results)
    if not runs:
        raise SystemExit(f"no evaluated run of {list(LABELS)} in {a.results}")
    print("models:", {l: len(v) for l, v in runs.items()}, "(number of seeds)")
    useful_horizon(runs, out)
    altitude_error(runs, out)
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
