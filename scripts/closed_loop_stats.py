"""Paired statistics of closed-loop runs (scripts/closed_loop.py), pooled over several runs of the same
controllers (for instance several environment seeds --seed, i.e. different aircraft, references and noise).

Every controller flies the same aircraft with the same noise in a run, so comparisons are paired per aircraft:
  - survival per aircraft group with its exact 95 % interval (Clopper-Pearson);
  - exact McNemar test of each controller against --ref on survival (discordant aircraft only);
  - mean time alive;
  - tracking RMS (airspeed, altitude, bank) over the aircraft that EVERY controller keeps alive for the whole
    flight, so that a controller is not rewarded for losing its hardest aircraft (survivor bias), with the
    paired difference to --ref and a bootstrap 95 % interval over aircraft.

Usage:
  python scripts/closed_loop_stats.py --runs runs/closed_loop_s*/sweep_nz0.5_s0.json --ref tecs \
      [--out runs/closed_loop_stats_nz0.5.json]
Each run needs its <run>_traces.npz next to it. Controllers are named by their run directory (tecs: "tecs").
"""
import argparse
import json
from math import comb
from pathlib import Path

import numpy as np

DT = 0.05


def binom_cdf(k, n, p):
    return sum(comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def clopper_pearson(k, n, alpha=0.05):
    """Exact interval of a proportion k / n (bisection on the binomial tails)."""
    def solve(f, lo=0.0, hi=1.0):
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if f(mid) else (lo, mid)
        return 0.5 * (lo + hi)
    low = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p) < alpha / 2)
    high = 1.0 if k == n else solve(lambda p: binom_cdf(k, n, p) > alpha / 2)
    return low, high


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p-value from the discordant counts b (only A survives) and c (only B)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * binom_cdf(min(b, c), n, 0.5))


def load(runs):
    """{controller: dict of per-aircraft arrays}, is_test (n,), pooled over the runs (same controllers in each)."""
    ctrls, groups = None, []
    per = {}
    for r in runs:
        r = Path(r)
        res = json.load(open(r))
        names = [Path(c).name for c in res["args"]["controllers"]]
        if ctrls is None:
            ctrls = names
        elif names != ctrls:
            raise SystemExit(f"{r}: controllers {names} differ from {ctrls}")
        tr = np.load(r.with_name(r.stem + "_traces.npz"))
        groups.append(tr["is_test"])
        for c in ctrls:
            al = tr[f"{c}__alive"]
            d = per.setdefault(c, {k: [] for k in ("alive_end", "t_alive", "eV", "eh", "ephi", "full")})
            d["alive_end"].append(al[:, -1])
            d["t_alive"].append(al.sum(1) * DT)
            d["full"].append(al.all(1))
            for k, (x, y, s) in dict(eV=("Va", "V_ref", 1.0), eh=("h", "h_ref", 1.0),
                                     ephi=("phi", "phi_ref", 180 / np.pi)).items():
                e = np.where(al, (tr[f"{c}__{x}"] - tr[f"{c}__{y}"]) * s, np.nan)
                d[k].append(np.sqrt(np.nanmean(e ** 2, 1)))
    per = {c: {k: np.concatenate(v) for k, v in d.items()} for c, d in per.items()}
    return ctrls, per, np.concatenate(groups).astype(bool)


def bootstrap_ci(x, n_boot=2000, seed=0):
    if len(x) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="closed_loop.py output JSON files")
    ap.add_argument("--ref", default="tecs", help="reference controller of the paired comparisons")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    ctrls, per, is_test = load(a.runs)
    if a.ref not in per:
        raise SystemExit(f"--ref {a.ref} not among {ctrls}")
    out = dict(runs=a.runs, ref=a.ref, groups={})
    for g, m in (("avions_entrainement", ~is_test), ("avions_de_test", is_test), ("tous", np.ones_like(is_test))):
        common = np.all([per[c]["full"][m] for c in ctrls], 0)          # alive for the whole flight with all
        res = dict(n=int(m.sum()), n_common_survivors=int(common.sum()), controllers={})
        print(f"\n== {g}: {m.sum()} aircraft, {common.sum()} kept alive by every controller")
        ref = per[a.ref]
        for c in ctrls:
            d = per[c]
            k, n = int(d["alive_end"][m].sum()), int(m.sum())
            lo, hi = clopper_pearson(k, n)
            b = int((d["alive_end"][m] & ~ref["alive_end"][m]).sum())
            cc = int((~d["alive_end"][m] & ref["alive_end"][m]).sum())
            row = dict(survival=k / n, survival_ci95=[lo, hi], lost=n - k, time_alive_s=float(d["t_alive"][m].mean()),
                       mcnemar_vs_ref=dict(only_this_survives=b, only_ref_survives=cc, p=mcnemar_exact(b, cc)))
            for e in ("eV", "eh", "ephi"):
                x = d[e][m][common]
                diff = x - ref[e][m][common]
                row[f"rms_{e[1:]}_common"] = float(np.mean(x)) if len(x) else float("nan")
                row[f"rms_{e[1:]}_minus_ref"] = float(np.mean(diff)) if len(diff) else float("nan")
                row[f"rms_{e[1:]}_minus_ref_ci95"] = bootstrap_ci(diff)
            res["controllers"][c] = row
            print(f"   {c:34s} survival {k:3d}/{n:<3d} [{lo:.2f}, {hi:.2f}]  McNemar p {row['mcnemar_vs_ref']['p']:.3f}"
                  f"  V {row['rms_V_common']:.2f} ({row['rms_V_minus_ref']:+.2f})  h {row['rms_h_common']:.2f}"
                  f"  phi {row['rms_phi_common']:.2f}")
        out["groups"][g] = res
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=1)
        print("written", a.out)


if __name__ == "__main__":
    main()
