"""README figures from the evaluation files collected in results/ (scripts/collect_results.py).

  assets/useful_horizon.png       median useful horizon per model, long flights, training vs extreme
                                  aircraft (bar = mean over seeds, whiskers = min/max over seeds)
  assets/altitude_error.png       RMS altitude error vs prediction time, with the two trivial baselines
  assets/calm_vs_turbulence.png   median useful horizon of the main models, turbulent vs calm air (PAIRS)
  assets/position_error.png       RMS 3D position error at 5 s (the SkyJEPA metric), turbulent vs calm air
  assets/probe.png                how well the present state can be read back from the encoder latent, A0 vs
                                  JEPA (probe.json, scripts/diagnose.py)

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
# (turbulent run, calm-air run, label): the models of the calm-air comparison
PAIRS = [
    ("a0_v2T100", "a0_calmT100", "A0: direct state loss"),
    ("jepa_v2T100", "jepa_calmT100", "A: JEPA, frozen latent"),
    ("jepa_ft_v2T100", "jepa_ft_calmT100", "A: JEPA pre-training + fine-tuning"),
    ("joint_v2T100", "joint_calmT100", "A: JEPA + state loss (joint)"),
    ("c_jepa_v2T100_ajuste", "c_jepa_calmT100_ajuste", "C-JEPA: physics head on the JEPA trunk"),
    ("c_direct_v2T100_ajuste", "c_direct_calmT100_ajuste", "C-direct: physics head, free correction"),
    ("c_direct_v2T100_ajuste_param", "c_direct_calmT100_ajuste_param", "C-direct: physics head, constrained"),
]
CONDITIONS = (("turbulent", "#9ec5f4"), ("calm air", "#2a78d6"))         # same hue, light -> dark
PROBE_RUNS = (("a0_v2T100_s0", "A0, turbulent"), ("jepa_v2T100_s0", "JEPA, turbulent"),
              ("a0_calmT100_s0", "A0, calm air"), ("jepa_calmT100_s0", "JEPA, calm air"))
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


def load_runs(results, fname):
    """{run name without the seed: [content of fname for each seed]}, for every run that has the file."""
    runs = defaultdict(list)
    for d in sorted(Path(results).iterdir()):
        if (d / fname).exists():
            runs[re.sub(r"_s\d+$", "", d.name)].append(json.load(open(d / fname)))
    return runs


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


def paired_bars(values, xlabel, title, out, baselines=None):
    """values: {group: [(label, [(mean, lo, hi) or None per condition])]} -> horizontal grouped bars."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 0.62 * len(PAIRS) + 1.6), sharey=True)
    labels = [p[2] for p in PAIRS]
    y = np.arange(len(labels))[::-1]
    h = 0.38
    for ax, (g, gtitle) in zip(axes, GROUPS.items()):
        for c, (cname, color) in enumerate(CONDITIONS):
            yc = y + (0.5 - c) * h
            for yi, (_, vals) in zip(yc, values[g]):
                v = vals[c]
                if v is None:
                    continue
                ax.barh(yi, v[0], height=h - 0.04, color=color, label=cname)     # legend: deduplicated below
                ax.errorbar(v[0], yi, xerr=[[v[0] - v[1]], [v[2] - v[0]]], color="0.25", lw=1, capsize=2)
        for cname, (val, style) in (baselines or {}).get(g, {}).items():
            ax.axvline(val, color="0.2", ls=style, lw=1, label=f"baseline, {cname}")
        ax.set_title(gtitle)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_yticks(y, labels)
    hd, lb = axes[0].get_legend_handles_labels()
    uniq = dict(zip(lb, hd))                                             # one entry per condition and baseline
    axes[1].legend(uniq.values(), uniq.keys(), fontsize=8, loc="lower right")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def calm_vs_turbulence(results, out):
    runs = load_runs(results, "eval_test_long.json")
    vals = {g: [] for g in GROUPS}
    for turb, calm, label in PAIRS:
        for g in GROUPS:
            vals[g].append((label, [stat(runs[k], g, "modele_vent_estime", "horizon") if runs.get(k) else None
                                    for k in (turb, calm)]))
    base = {g: {} for g in GROUPS}
    for (cname, _), k in zip(CONDITIONS, ("a0_v2T100", "a0_calmT100")):
        for g in GROUPS:
            if runs.get(k):
                base[g][cname] = (stat(runs[k], g, "reference", "horizon")[0], "--" if cname == "turbulent" else ":")
    paired_bars(vals, "median useful horizon [s]",
                "Open-loop prediction from 2 s of history: turbulent vs calm air (same flights)",
                out / "calm_vs_turbulence.png", base)


def position_error(results, out):
    runs = load_runs(results, "eval_position_test_long.json")
    pos = lambda evs, g, who: (np.mean([e[g][who]["5s"]["rms"] for e in evs]), min(e[g][who]["5s"]["rms"] for e in evs),
                               max(e[g][who]["5s"]["rms"] for e in evs))
    vals = {g: [] for g in GROUPS}
    for turb, calm, label in PAIRS:
        for g in GROUPS:
            vals[g].append((label, [pos(runs[k], g, "model") if runs.get(k) else None for k in (turb, calm)]))
    base = {g: {} for g in GROUPS}
    for (cname, _), k in zip(CONDITIONS, ("a0_v2T100", "a0_calmT100")):
        for g in GROUPS:
            if runs.get(k):
                base[g][cname] = (pos(runs[k], g, "reference")[0], "--" if cname == "turbulent" else ":")
    paired_bars(vals, "RMS 3D position error at 5 s [m] (lower is better)",
                "Position rebuilt from the predicted states (SkyJEPA metric)", out / "position_error.png", base)


def probe(results, out):
    found = [(Path(results) / r / "probe.json", lab) for r, lab in PROBE_RUNS]
    found = [(json.load(open(f))["probe"], lab) for f, lab in found if f.exists()]
    if not found:
        return False
    names = list(found[0][0])
    x = np.arange(len(names))
    w = 0.8 / len(found)
    colors = ["#9ec5f4", "#f3b89d", "#2a78d6", "#eb6834"]                 # A0 blue, JEPA orange; light = turbulent
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for i, (pr, lab) in enumerate(found):
        ax.bar(x + (i - (len(found) - 1) / 2) * w, [pr[k] for k in names], width=w - 0.02, label=lab,
               color=colors[i % len(colors)])
    ax.set_xticks(x, names)
    ax.set_ylabel("normalized error (0 = perfect, 1 = lost)")
    ax.set_title("Can the present state be read back from the encoder latent? (small MLP probe, same size as the decoder)")
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "probe.png", dpi=150)
    plt.close(fig)
    return True


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
    calm_vs_turbulence(a.results, out)
    position_error(a.results, out)
    print("probe figure:", "written" if probe(a.results, out) else "skipped (no probe.json)")
    print(f"figures written to {out}")


if __name__ == "__main__":
    main()
