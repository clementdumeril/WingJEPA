"""Choice of the MPPI settings from a small tuning grid (outputs of scripts/closed_loop.py), frozen afterwards for
every model of the closed-loop comparison.

Each tuning file holds the same controllers flown with one setting. Score of a setting, averaged over its
controllers and both aircraft groups: survival first (higher is better), then the tracking cost
rms_h / 5 m + rms_V / 1 m/s + rms_phi / 5 deg (lower is better). Tune on one flight seed, then compare the models on
another (closed_loop.py --seed), so that the settings are not fitted to the test flights.

Usage: python scripts/pick_mppi.py runs/closed_loop/tune/*.json --out runs/closed_loop/mppi_best.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

KEYS = ("lam", "sigma", "smooth", "w_du")


def score(res):
    surv, cost = [], []
    for c, r in res.items():
        if c in ("args", "tecs"):
            continue
        for g in ("avions_entrainement", "avions_de_test"):
            m = r[g]
            surv.append(m["survival"])
            cost.append(m["rms_h_m"] / 5 + m["rms_V_ms"] + m["rms_phi_deg"] / 5)
    return float(np.mean(surv)), float(np.nanmean(cost)) if np.isfinite(cost).any() else float("inf")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = []
    for f in a.files:
        res = json.load(open(f))
        s, c = score(res)
        rows.append((-s, c, f, {k: res["args"][k] for k in KEYS}))
        print(f"{Path(f).name:40s} survival {s:.3f}  cost {c:.3f}  {rows[-1][3]}")
    rows.sort(key=lambda r: (r[0], r[1]))
    best = rows[0]
    json.dump(dict(best[3], source=best[2], survival=-best[0], cost=best[1]), open(a.out, "w"), indent=1)
    print(f"best: {best[2]} -> {a.out}")


if __name__ == "__main__":
    main()
