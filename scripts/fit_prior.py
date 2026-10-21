"""Builds the four priors of the physics head of model C (docs/model_design.md, section 11).

  ajuste    : least squares on the aero coefficients (CD, CY, CL, Cl, Cm, Cn) and the thrust recorded
              by the simulator, training flights only. The model (regressors: TERMS in
              wingjepa/models/physics.py) is simpler than the NASA tables (no propeller slipstream effect,
              no stall): even when fitted, the prior is not perfect (R2 printed).
  generique : published coefficients of the Ultra Stick 25e (little brother, configs/ultrastick25e_validation.yaml),
              drag linearized in alpha; thrust = the fitted one (the motor is known from its test bench).
  faux      : fitted, each coefficient multiplied by 1 +/- 30 % (sign drawn at random, fixed seed).
  nul       : everything zero (C0: only gravity and the equations of motion remain).
Outputs: <out>/<name>.json, to be passed to train_a.py --prior.

Usage: python scripts/fit_prior.py --data-dir <out_dir>/data_v1
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wingjepa.data.windows import load_set  # noqa: E402
from wingjepa.models.physics import COEFS, TERMS, regressors  # noqa: E402
from wingjepa.sim.ultrastick import isa_density, load_config  # noqa: E402


def fit(data, ch, mask, p):
    need = ["alpha", "beta", "de", "da", "dr", "p", "q", "r", "Va", "dt_cmd", "h", "thrust"] + COEFS[:-1]
    c = {k: data[:, :, ch.index(k)][mask].astype(np.float64) for k in need}
    V2 = 2 * np.maximum(c["Va"], 3.0)
    R = regressors(c["alpha"], c["beta"], c["de"], c["da"], c["dr"], c["p"] * p["b"] / V2,
                   c["q"] * p["c"] / V2, c["r"] * p["b"] / V2, c["dt_cmd"], c["Va"])
    target = {k: c[k] for k in COEFS[:-1]}
    target["T"] = c["thrust"]
    coef, r2 = {}, {}
    for k in COEFS:
        A = np.stack(R[k], 1)
        sol = np.linalg.lstsq(A, target[k], rcond=None)[0]
        res = target[k] - A @ sol
        coef[k] = sol.tolist()
        r2[k] = float(1 - res.var() / target[k].var())
    return coef, r2, float(isa_density(c["h"].mean()))


def generic_25e(g, thrust):
    CL, CD, Cm, CY, Cl, Cn = g["CL"], g["CDw"], g["Cm"], g["CYw"], g["Cl"], g["Cn"]
    k = 1 / (np.pi * g["b"] ** 2 / g["S"] * CD["osw"])
    d = CL["zero"] - CL["minD"]
    lat = lambda t: [t["beta"], t["dail"], t["drud"], t["p"], t["r"]]
    return {
        "CL": [CL["zero"], CL["alpha"], 0.0, CL["delev"], CL["q"]],
        # CD = min + (CL - minD)^2 / (pi AR e) + |...|, with CL = zero + alpha * a (cross terms neglected)
        "CD": [CD["min"] + k * d * d, 2 * k * d * CL["alpha"], k * CL["alpha"] ** 2, 0.0,
               CD["delev"], CD["dail"], CD["drud"]],
        "CY": lat(CY), "Cl": lat(Cl), "Cn": lat(Cn),
        "Cm": [Cm["zero"], Cm["alpha"], 0.0, Cm["delev"], Cm["q"]],
        "T": list(thrust),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--set", default="train")
    ap.add_argument("--config", default="configs/ultrastick120.yaml")
    ap.add_argument("--generic", default="configs/ultrastick25e_validation.yaml")
    ap.add_argument("--out", default=None, help="default: <data-dir>/priors")
    ap.add_argument("--wrong-frac", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    p = load_config(a.config)
    out = Path(a.out or Path(a.data_dir) / "priors")
    out.mkdir(parents=True, exist_ok=True)

    data, ch, meta = load_set(a.data_dir, a.set)
    alive = data[:, :, ch.index("alive")] > 0.5
    mask = alive & (meta["split"] == "train")[:, None]
    print(f"fitting on {mask.sum()} steps of {(meta['split'] == 'train').sum()} training flights")
    coef, r2, rho = fit(data, ch, mask, p)

    rng = np.random.default_rng(a.seed)
    priors = {
        "ajuste": coef,
        "generique": generic_25e(load_config(a.generic), coef["T"]),
        "faux": {k: (np.asarray(v) * (1 + a.wrong_frac * rng.choice([-1, 1], len(v)))).tolist()
                 for k, v in coef.items()},
        "nul": {k: [0.0] * len(v) for k, v in coef.items()},
    }
    for name, cf in priors.items():
        d = dict(name=name, coef=cf, terms=TERMS, rho=rho, S=p["S"], b=p["b"], c=p["c"], g=p["g"])
        if name == "ajuste":
            d["r2"] = r2
        json.dump(d, open(out / f"{name}.json", "w"), indent=1)
    print(f"air density used: {rho:.4f} kg/m3")
    print("R2 of the fitted prior (share of the variance explained by the simplified model):")
    for k in COEFS:
        print(f"   {k:3s} R2 = {r2[k]:.3f}")
    print("coefficients (fitted | generic 25e):")
    for k in COEFS:
        for t, x, y in zip(TERMS[k], priors["ajuste"][k], priors["generique"][k]):
            print(f"   {k:3s} {t:7s} {x:+9.4f} | {y:+9.4f}")
    print(f"priors written to {out}")


if __name__ == "__main__":
    main()
