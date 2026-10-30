"""Copy the small outputs of finished runs into results/ (tracked by git) and build a summary table.

For each run directory under runs/ that has been evaluated, results/<run>/ receives config.json,
log.json, eval_train.json and eval_test_long.json (a few hundred kB in total; model weights stay on the
cluster). results/summary.csv has one row per run with the numbers quoted in the README.

Usage: python scripts/collect_results.py [--runs "runs/*_s[0-9]*"] [--out results]
"""
import argparse
import csv
import glob
import json
import shutil
from pathlib import Path

GROUPS = {"test_flights": ("train", "vols_de_test"),
          "train_aircraft": ("test_long", "avions_entrainement"),
          "extreme_aircraft": ("test_long", "avions_de_test")}
WHO = {"model": "modele_vent_estime", "ref_constant_state": "reference",
       "ref_constant_altitude": "reference_altitude_constante"}


def relative(path):
    """/.../WingJEPA/data/data_v1 -> data/data_v1"""
    return path.split("/WingJEPA/", 1)[-1]


def summarize(run):
    cfg = json.load(open(run / "config.json"))
    prior = (cfg.get("prior") or {}).get("name", "")
    row = dict(run=run.name, mode=cfg["mode"], seed=cfg["seed"], history_steps=cfg["H"],
               aircraft_input=int(bool(cfg.get("aircraft_input"))), head=cfg.get("head", "mlp"), prior=prior,
               correction=cfg.get("correction", "free") if cfg.get("head") == "physics" else "",
               params=sum(cfg["n_params"].values()), best_val_state=round(cfg.get("best_val_state", float("nan")), 4))
    for g, (s, key) in GROUPS.items():
        f = run / f"eval_{s}.json"
        res = json.load(open(f))[key]["par_turbulence"]["toutes"] if f.exists() else {}
        tmax = "60s" if s == "test_long" else "20s"
        for w, wk in WHO.items():
            if wk not in res:
                continue
            row[f"{w}_horizon_s_{g}"] = round(res[wk]["horizon_utile_s"]["mediane"], 2)
            row[f"{w}_alt_rms_{tmax}_m_{g}"] = round(res[wk]["h"].get(tmax, float("nan")), 1)
    pos = run / "eval_position_test_long.json"
    if pos.exists():
        ep = json.load(open(pos))
        for g, (_, key) in GROUPS.items():
            for t in ("5s", "10s"):
                if key in ep and t in ep[key]["model"]:
                    row[f"model_pos_rms_{t}_m_{g}"] = round(ep[key]["model"][t]["rms"], 1)
                    row[f"ref_constant_state_pos_rms_{t}_m_{g}"] = round(ep[key]["reference"][t]["rms"], 1)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/*_s[0-9]*")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    out = Path(a.out)
    rows = []
    for d in sorted(glob.glob(a.runs)):
        run = Path(d)
        if not (run / "config.json").exists() or not (run / "eval_test_long.json").exists():
            print(f"skipped (not finished or not evaluated): {run}")
            continue
        dst = out / run.name
        dst.mkdir(parents=True, exist_ok=True)
        for f in ("log.json", "eval_train.json", "eval_test_long.json", "eval_position_test_long.json", "probe.json"):
            if (run / f).exists():
                shutil.copy(run / f, dst / f)
        cfg = json.load(open(run / "config.json"))
        cfg["data_dir"] = relative(cfg["data_dir"])                    # no cluster paths in the repository
        json.dump(cfg, open(dst / "config.json", "w"), indent=1)
        rows.append(summarize(run))
        print(f"collected {run.name}")
    if not rows:
        return
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} runs -> {out / 'summary.csv'}")


if __name__ == "__main__":
    main()
