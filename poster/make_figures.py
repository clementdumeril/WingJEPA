"""Data figures for the WingJEPA poster (48 x 36 in).

Reads results/<run>_s<seed>/eval_skyjepa_test_long.json and poster/data/savio_results.json,
writes vector PDF + 200-dpi PNG of every figure into poster/figures/. Each figure is drawn at
its printed size (one 8.6-in poster column), so the font sizes below are the printed sizes.
Run from anywhere: python poster/make_figures.py
"""
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "poster" / "figures"
SEEDS = (0, 1, 2)
DT = 0.05
SIGMAS = ["0", "0.05", "0.1", "0.2", "0.5"]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
LOSS_FILL, LOSS_INK = "#f6d3d3", "#a32020"
NUM = 20  # value labels and table cells (tick-label size); all other text is 22 pt

# One identity per model, shared by every figure. "wrap" is the same name broken over two lines.
M = {
    "cjepa":    dict(label="JEPA + physics", color="#E07B1A", marker="o"),
    "jepa":     dict(label="JEPA + MLP", color="#E3BE85", marker="s"),
    "a0":       dict(label="Direct + MLP", color="#2E6DB4", marker="D"),
    "cjepa_a0": dict(label="Direct frozen + physics", wrap="Direct frozen\n+ physics", color="#86B8E8", marker="v"),
    "cdirect":  dict(label="Direct + physics (end to end)", wrap="Direct + physics\n(end to end)",
                     color="#2E8B57", marker="^"),
    "mlp":      dict(label="Plain MLP", color="#333333", marker="P"),
    "tecs":     dict(label="TECS autopilot", color="#8A8A8A", marker="X"),
}
NOISE_SUFFIX = "+ noise"  # Gaussian noise on the past states in training (sigma ~ U[0, 0.5]), clean targets


def wrap(k):
    return M[k].get("wrap", M[k]["label"])


RUN = {"a0": "a0_calm_sky", "jepa": "jepa_calm_sky", "cjepa": "c_jepa_calm_sky_nul",
       "cjepa_a0": "c_jepa_calm_sky_a0", "cdirect": "c_direct_calm_sky_nul", "a0_noise": "a0_calm_sky_noiseaug",
       "pred": "a0_calm_sky_pred", "physreg": "a0_calm_sky_physreg"}
CL_KEY = {"tecs": "tecs", "mlp": "mlp_direct_calm_sky_s0", "a0": "a0_calm_sky_s0",
          "cdirect": "c_direct_calm_sky_nul_s0", "cjepa": "c_jepa_calm_sky_nul_s0",
          "cjepa_ctx8": "c_jepa_calm_sky_ctx8_nul_s0", "cjepa_a0": "c_jepa_calm_sky_a0_s0"}

plt.rcParams.update({
    "font.family": "Arial", "font.size": 22, "axes.labelsize": 22, "axes.titlesize": 22,
    "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 22,
    "lines.linewidth": 3, "lines.markersize": 10, "axes.linewidth": 1.2,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42, "legend.frameon": False, "hatch.linewidth": 1.5,
    "legend.handlelength": 1.8, "legend.handletextpad": 0.5, "legend.labelspacing": 0.35,
    "legend.borderaxespad": 0.2, "legend.borderpad": 0.1,
    "figure.constrained_layout.h_pad": 0.03, "figure.constrained_layout.w_pad": 0.03,
})

# ---------------------------------------------------------------- data
SAVIO = json.loads((ROOT / "poster" / "data" / "savio_results.json").read_text())
GROUPS = {"train": "avions_entrainement", "test": "avions_de_test"}


def load_eval(run):
    """Per-seed eval dicts of one run."""
    return [json.loads((ROOT / "results" / f"{run}_s{s}" / "eval_skyjepa_test_long.json").read_text())
            for s in SEEDS]


EVAL = {k: load_eval(r) for k, r in RUN.items()}
MLP_OL = SAVIO["open_loop_5s"]["mlp_direct_calm_sky"]


def pos_curves(key, group):
    """Position RMSE curves, shape (seeds, 100)."""
    return np.array([e[GROUPS[group]]["position_rmse_m"] for e in EVAL[key]])


def pos5(key, group="train"):
    return pos_curves(key, group)[:, -1].mean()


def noise_curve(key, group="train"):
    return np.array([[e[GROUPS[group]]["noise_robustness"][s]["median"] for s in SIGMAS]
                     for e in EVAL[key]]).mean(0)


def save(fig, stem):
    for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
        fig.savefig(OUT / f"{stem}.{ext}", **kw)
    plt.close(fig)
    print("wrote", stem)


# ---------------------------------------------------------------- 1. open loop
def fig_openloop():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.1), sharey=True, layout="constrained")
    t = (np.arange(100) + 1) * DT
    lines = ("a0", "jepa", "cjepa", "cjepa_a0", "cdirect")
    for ax, (grp, title) in zip(axes, [("train", "Training range"), ("test", "Extreme aircraft")]):
        for k in lines:
            c = pos_curves(k, grp)
            ax.fill_between(t, c.min(0), c.max(0), color=M[k]["color"], alpha=0.18, lw=0)
            ax.plot(t, c.mean(0), color=M[k]["color"], marker=M[k]["marker"], markevery=[19, 39, 59, 79, 99],
                    mec="white", mew=1.5, solid_capstyle="round")
        for tt, field in ((1, "pos_1s"), (5, "pos_5s")):  # plain MLP: 3-seed mean, min-max bar
            v = np.array(MLP_OL[grp][field])
            ax.errorbar(tt + 0.15, v.mean(), yerr=[[v.mean() - v.min()], [v.max() - v.mean()]],
                        color=M["mlp"]["color"], marker="P", ms=12, mec="white", mew=1.5, capsize=5, lw=2.5,
                        zorder=5)
        ax.set_title(title, pad=6)
        ax.set_xlim(0, 5.4)
        ax.set_xticks(range(6))
        ax.set_xlabel("Time [s]")
    axes[0].set_ylim(0, 26)
    axes[0].set_ylabel("Position RMSE [m]")
    handles = [Line2D([], [], color=M[k]["color"], marker=M[k]["marker"], mec="white", label=wrap(k))
               for k in lines]
    handles.append(Line2D([], [], color=M["mlp"]["color"], marker="P", ls="none", ms=12, label="Plain MLP"))
    fig.legend(handles=handles, loc="outside right center")
    save(fig, "fig_openloop")


# ---------------------------------------------------------------- 2. SkyJEPA gap
def fig_skyjepa_gap():
    """Both panels normalized by the one-step 'Predictive' baseline (ours: a0_calm_sky_pred)."""
    sky = SAVIO["skyjepa_table3"]
    base_sky, base_ours = sky["Predictive"][0], pos5("pred")
    mlp5 = np.mean(MLP_OL["train"]["pos_5s"])
    # (key, row label, colour, two-line label?)
    rows = [("pred", "Predictive (one-step)", "#4A5A70", False),
            ("physreg", "Predictive +\nphysics in loss", "#8E9BB0", True),
            ("a0", "Direct + MLP\n(multi-step)", M["a0"]["color"], True),
            ("recon", "Recon. + physics", "#B5B5B5", False),
            ("jepa", M["jepa"]["label"], M["jepa"]["color"], False),
            ("cjepa", M["cjepa"]["label"], M["cjepa"]["color"], False),
            ("cdirect", wrap("cdirect"), M["cdirect"]["color"], True),
            ("mlp", M["mlp"]["label"], M["mlp"]["color"], False)]
    left = {"pred": 1.0, "physreg": sky["Predictive + Physics Reg."][0] / base_sky,
            "recon": sky["Reconstruction + PI Prober"][0] / base_sky,
            "jepa": sky["JEPA + Prober"][0] / base_sky, "cjepa": sky["JEPA + PI Prober"][0] / base_sky}
    right = {k: pos5(k) / base_ours for k in ("pred", "physreg", "a0", "jepa", "cjepa", "cdirect")}
    right["mlp"] = mlp5 / base_ours
    # two-line rows get a taller slot so their labels do not touch
    pitch = np.array([1.4 if r[3] else 1.0 for r in rows])
    y = -(np.cumsum(pitch) - pitch / 2)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.8), sharey=True, layout="constrained")
    xmax = 1.45
    for ax, vals, title, base, missing in zip(
            axes, (left, right), ("SkyJEPA, quadrotor", "Ours, fixed wing"),
            (f"{base_sky:.2f} m", f"{base_ours:.1f} m"),
            ({"a0": "not reported", "cdirect": None, "mlp": None}, {"recon": "not run"})):
        for yi, (k, _, col, _) in zip(y, rows):
            if k in vals:
                ax.barh(yi, vals[k], height=0.7, color=col, lw=0)
                bold = ax is axes[1] and k in ("cdirect", "mlp")
                ax.text(vals[k] + 0.03, yi, f"{vals[k]:.2f}", va="center", fontsize=NUM, zorder=4,
                        fontweight="bold" if bold else "normal", bbox=dict(fc="white", ec="none", pad=0.5))
                if k == "pred":
                    ax.text(0.5, yi, f"= {base}", va="center", ha="center", fontsize=NUM, color="white")
            elif missing.get(k) is None:  # control SkyJEPA did not run
                ax.add_patch(Rectangle((0, yi - 0.35), xmax - 0.03, 0.7, fill=False, hatch="//",
                                       ec=col, lw=2, alpha=0.6))
                ax.text(xmax / 2, yi, "not tested", ha="center", va="center", fontsize=NUM,
                        bbox=dict(fc="white", ec="none", pad=1))
            else:
                ax.text(0.03, yi, missing[k], va="center", fontsize=NUM, color=MUTED)
        ax.axvline(1.0, color=INK2, lw=1.5, zorder=0)
        ax.set_xlim(0, xmax)
        ax.set_ylim(y[-1] - 0.6, y[0] + 0.6)
        ax.set_xticks([0, 0.5, 1.0])
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, pad=4)
    axes[0].set_yticks(y, [r[1] for r in rows])
    for tl in axes[0].get_yticklabels():
        tl.set_linespacing(0.95)
    fig.supxlabel("Position error / Predictive (lower is better)", fontsize=22)
    save(fig, "fig_skyjepa_gap")
    return left, right, base_ours, mlp5


# ---------------------------------------------------------------- 3. probes
def fig_probe():
    pr = SAVIO["probe_r2"]
    groups = [("After 1\nGRU step", "a0_z_t+1", "jepa_z_t+1"),
              ("Free context\nlatent c", "a0_ctx8_c", "jepa_ctx8_c"),
              ("Encoder reads\ncommands", "a0_acth1_z_t", "jepa_acth1_z_t")]
    ink = {"a0": "#1f4e85", "cjepa": "#a8560c"}
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.95), sharey=True, layout="constrained")
    y = []
    for g in range(3):
        y += [g * 2.6, g * 2.6 + 1]
    y = np.array(y)
    keys = [k for _, a, j in groups for k in (a, j)]
    fams = ["a0", "cjepa"] * 3
    raw = pr["raw_window_states_commands"]
    for ax, p, title in zip(axes, ("motor", "cg"), ("Motor power", "CG shift")):
        for yi, key, fam in zip(y, keys, fams):
            v = pr[key][p]
            ax.barh(yi, max(v, 0.004), height=0.8, color=M[fam]["color"], lw=0)
            tx = max(v, 0) + 0.02
            if raw[p] - 0.04 < tx < raw[p] + 0.03:  # label would sit on the dashed line: start after it
                tx = raw[p] + 0.02
            ax.text(tx, yi, f"{v:.2f}", va="center", fontsize=NUM, color=INK2, zorder=4,
                    bbox=dict(fc="white", ec="none", pad=0.5))
        ax.axvline(raw[p], color=INK, lw=2.5, ls="--", zorder=1)
        ax.set_xlim(-0.05, 1.18)
        ax.set_xticks([0, 0.5, 1])
        ax.set_ylim(y[-1] + 0.6, y[0] - 0.6)
        ax.axvline(0, color="#c3c2b7", lw=1.2)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_title(title, pad=4)
    axes[0].set_yticks(y, ["Direct", "JEPA"] * 3)
    for tl, fam in zip(axes[0].get_yticklabels(), fams):
        tl.set_color(ink[fam])
    for name, a, _ in groups:  # representation name to the left of each Direct/JEPA pair
        yc = y[keys.index(a)] + 0.5
        axes[0].annotate(name, (0, yc), xycoords=("axes fraction", "data"), xytext=(-82, 0),
                         textcoords="offset points", ha="right", va="center", fontsize=NUM, linespacing=0.95)
    handle = Line2D([], [], color=INK, lw=2.5, ls="--", label="raw 0.55 s window (training-range aircraft)")
    fig.legend(handles=[handle], loc="outside lower center", title=r"$R^2$ of an MLP reader, extreme aircraft",
               title_fontsize=22)
    save(fig, "fig_probe")


# ---------------------------------------------------------------- 4. noise, open loop
def fig_noise_openloop():
    fig, ax = plt.subplots(figsize=(8.6, 3.85), layout="constrained")
    x = np.arange(len(SIGMAS))
    nol = SAVIO["noise_open_loop"]
    mlp = np.mean([nol["mlp_direct_calm_sky_s0"]["train"], nol["mlp_direct_calm_sky_s1"]["train"]], axis=0)
    # plain MLP drawn first so the nearly equal JEPA + physics curve stays visible on top
    series = [("mlp", mlp, "-")] + [(k, noise_curve(k), "-") for k in ("a0", "jepa", "cjepa", "cdirect")]
    series += [("a0", noise_curve("a0_noise"), "--")]
    handles = []
    for k, v, ls in series:
        lab = wrap(k) if ls == "-" else f"{M[k]['label']}\n{NOISE_SUFFIX}"
        if k == "mlp":
            lab = "Plain MLP\n(2 seeds)"
        handles.append(ax.plot(x, v, color=M[k]["color"], marker=M[k]["marker"], ls=ls, mec="white", mew=1.5,
                               label=lab)[0])
    ax.set_xticks(x, SIGMAS)
    ax.set_xlabel(r"Noise $\sigma$ on past states [std]")
    ax.set_ylabel("1-s state error\n(normalized median)")
    ax.set_ylim(0, 0.82)
    ax.set_xlim(-0.2, len(SIGMAS) - 0.8)
    handles = handles[1:5] + handles[:1] + handles[5:]
    fig.legend(handles=handles, loc="outside right center", labelspacing=0.3)
    save(fig, "fig_noise_openloop")
    return {k + ("_noise" if ls == "--" else ""): v for k, v, ls in series}


# ---------------------------------------------------------------- 5. closed loop
CONDS = [("gauss_0.5", "Gauss.\n0.5"), ("gauss_1.0", "Gauss.\n1.0"), ("colored_0.5", "colored\n0.5"),
         ("bias_0.5", "bias\n0.5"), ("dropout_0.3", "dropout\n0.3"), ("turb_moderate", "turb.\nmod."),
         ("turb_severe", "turb.\nsevere")]


def rel_airspeed(cond, k):
    """% airspeed RMS vs JEPA + physics; uses the paired difference dV when it was logged."""
    ctl = SAVIO["closed_loop"][cond]["controllers"]
    ref = ctl[CL_KEY["cjepa"]]["V"]
    c = ctl[CL_KEY[k]]
    val = 100 * c["dV"] / ref if "dV" in c else 100 * (c["V"] / ref - 1)
    ci = [100 * d / ref for d in c["dV_ci"]] if "dV_ci" in c else None
    return val, ci


def fig_closedloop():
    keys = ("cdirect", "mlp", "cjepa_a0")
    fig, ax = plt.subplots(figsize=(8.6, 6.0), layout="constrained")
    w = 0.27
    out = {}
    for gi, (x0, x1, name) in enumerate(((-0.5, 3.5, "Sensor noise\n(fake motion)"),
                                         (3.5, 6.5, "Real motion /\ndata gaps"))):
        ax.axvspan(x0, x1, color="#f3f2ee" if gi == 0 else "#e9eef5", zorder=0, lw=0)
        ax.text((x0 + x1) / 2, 57, name, ha="center", va="top", fontweight="bold", color=INK2, linespacing=1.0)
    for i, (cond, _) in enumerate(CONDS):
        for j, k in enumerate(keys):
            v, ci = rel_airspeed(cond, k)
            out[(cond, k)] = (v, ci)
            xx = i + (j - 1) * w
            ax.bar(xx, v, width=w - 0.03, color=M[k]["color"], lw=0, zorder=2)
            if ci:
                ax.errorbar(xx, v, yerr=[[v - ci[0]], [ci[1] - v]], color=INK, lw=2, capsize=4, zorder=3)
            if abs(v) > 15:
                ax.text(xx, (ci[1] if ci else v) + 0.8, f"+{v:.0f}%", ha="center", va="bottom", fontsize=NUM)
    ax.axhline(0, color=INK, lw=2.5, zorder=4)
    ax.text(4.6, 3, "↑ JEPA better", ha="center", va="bottom", color=INK2, style="italic")
    ax.text(1.5, -3, "↓ JEPA worse", ha="center", va="top", color=INK2, style="italic")
    ax.set_xticks(range(len(CONDS)), [c[1] for c in CONDS])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-13, 57)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Airspeed RMS vs\nJEPA + physics [%]")
    handles = [Patch(fc=M[k]["color"], label=wrap(k)) for k in keys]
    leg = fig.legend(handles=handles, loc="outside lower center", ncol=3, columnspacing=1.0,
                     title="Trained on clean data, one training seed per model;\n200 aircraft; env. seed 1. "
                           "Error bars: paired bootstrap 95% CI.",
                     title_fontsize=22)
    leg.get_title().set_color(INK2)
    save(fig, "fig_closedloop")
    return out


# ---------------------------------------------------------------- 7. equal noise training
def fig_equal_noise():
    """Closed loop with every model trained with the same Gaussian noise on the past states
    (sigma drawn uniformly in [0, 0.5] per window)."""
    eq = SAVIO["closed_loop"]["equal_noise_training"]
    conds = [("gauss_1.0", "Gauss.\n1.0"), ("colored_0.5", "colored\n0.5"), ("bias_0.5", "bias\n0.5"),
             ("dropout_0.3", "dropout\n0.3"), ("turb_severe", "turb.\nsevere")]
    bars = [("mlp_direct_calm_sky_noiseaug_s0", "mlp", "Plain MLP " + NOISE_SUFFIX),
            ("c_direct_calm_sky_nul_noiseaug_s0", "cdirect", "Direct + physics " + NOISE_SUFFIX),
            ("c_jepa_calm_sky_dn_nul_s0", "cjepa", "JEPA + physics " + NOISE_SUFFIX)]
    xs = np.array([0, 1.85, 2.85, 3.85, 4.85])  # wider slot for the condition whose kind was seen in training
    w = 0.3
    fig, ax = plt.subplots(figsize=(8.6, 4.6), layout="constrained")
    for x0, x1, name, col in ((-1.1, 1.1, "kind seen, level\n2x training max", "#e9eef5"),
                              (1.1, 5.47, "never seen in training", "#f3f2ee")):
        ax.axvspan(x0, x1, color=col, zorder=0, lw=0)
        ax.text((x0 + x1) / 2, 2.88, name, ha="center", va="top", fontweight="bold", color=INK2, linespacing=1.0)
    out = {}
    for x, (cond, _) in zip(xs, conds):
        ctl = eq[cond]["controllers"]
        for j, (key, k, _) in enumerate(bars):
            v, xx = ctl[key]["V"], x + (j - 1) * w
            out[(cond, key)] = v
            ax.bar(xx, v, width=w - 0.04, color=M[k]["color"], lw=0, zorder=2)
            ink = INK if k == "cjepa" else "white"
            ax.text(xx, 0.06, f"{v:.2f}", rotation=90, ha="center", va="bottom", fontsize=NUM, color=ink, zorder=3)
            surv = ctl[key]["survival"]
            if surv < 200:
                ax.text(xx, max(v, ctl["c_jepa_calm_sky_nul_s0"]["V"]) + 0.07, f"{surv}/200", ha="center",
                        va="bottom", fontsize=NUM, color=LOSS_INK, zorder=5,
                        bbox=dict(fc="white", ec="none", pad=0.5, alpha=0.8))
        clean = ctl["c_jepa_calm_sky_nul_s0"]["V"]
        out[(cond, "c_jepa_calm_sky_nul_s0")] = clean
        ax.plot(x + w, clean, marker="D", ms=11, mfc="white", mec=INK, mew=2, ls="none", zorder=4)
    ax.set_xticks(xs, [c[1] for c in conds])
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-1.1, 5.47)
    ax.set_ylim(0, 2.92)
    ax.set_yticks([0, 0.5, 1, 1.5, 2])
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Airspeed\nRMS [m/s]")
    handles = [Patch(fc=M[k]["color"], label=lab) for _, k, lab in (bars[0], bars[2], bars[1])]
    handles.append(Line2D([], [], marker="D", ms=11, mfc="white", mec=INK, mew=2, ls="none",
                          label="JEPA + physics,\nclean training"))
    leg = fig.legend(handles=handles, loc="outside lower center", ncol=2, columnspacing=1.2,
                     title="One training seed per model; 200 aircraft; env. seed 1.", title_fontsize=22)
    leg.get_title().set_color(INK2)
    save(fig, "fig_equal_noise")
    return out


# ---------------------------------------------------------------- 6. survival
def fig_survival():
    conds = [("gauss_0.5", "Gauss.\n0.5"), ("dropout_0.3", "drop.\n0.3"), ("gauss_1.0", "Gauss.\n1.0"),
             ("colored_0.5", "col.\n0.5"), ("bias_0.5", "bias\n0.5"), ("turb_moderate", "turb.\nmod."),
             ("turb_severe", "turb.\nsev.")]
    rows = [("tecs", M["tecs"]["label"]), ("mlp", M["mlp"]["label"]), ("a0", M["a0"]["label"]),
            ("cdirect", wrap("cdirect")), ("cjepa", M["cjepa"]["label"]),
            ("cjepa_ctx8", "JEPA + physics,\ncontext c"), ("cjepa_a0", wrap("cjepa_a0"))]
    fig, ax = plt.subplots(figsize=(8.6, 4.0), layout="constrained")
    for r, (k, _) in enumerate(rows):
        for c, (cond, _) in enumerate(conds):
            s = SAVIO["closed_loop"][cond]["controllers"].get(CL_KEY[k], {}).get("survival")
            loss = s is not None and s < 200
            ax.add_patch(Rectangle((c + 0.04, r + 0.05), 0.92, 0.9, lw=0,
                                   fc=LOSS_FILL if loss else ("#f3f2ee" if s else "white")))
            ax.text(c + 0.5, r + 0.5, "–" if s is None else str(s), ha="center", va="center", fontsize=NUM,
                    fontweight="bold" if loss else "normal",
                    color=LOSS_INK if loss else (MUTED if s is None else INK2))
    ax.set_xlim(0, len(conds))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks(np.arange(len(conds)) + 0.5, [c[1] for c in conds])
    ax.xaxis.tick_top()
    ax.set_yticks(np.arange(len(rows)) + 0.5, [r[1] for r in rows], color=INK)
    for tl in ax.get_yticklabels():
        tl.set_linespacing(0.9)
    ax.tick_params(length=0, pad=4)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.annotate("Survivors\nout of 200", (0, 1), xycoords="axes fraction", xytext=(-8, 6),
                textcoords="offset points", ha="right", va="bottom", color=INK2, linespacing=1.0)
    save(fig, "fig_survival")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_openloop()
    left, right, base, mlp5 = fig_skyjepa_gap()
    print("gap: SkyJEPA", {k: round(v, 3) for k, v in left.items()}, "| ours",
          {k: round(float(v), 3) for k, v in right.items()}, "| baseline %.2f m, MLP 5 s %.2f m" % (base, mlp5))
    fig_probe()
    fig_noise_openloop()
    fig_closedloop()
    fig_survival()
    print("equal noise V:", fig_equal_noise())
