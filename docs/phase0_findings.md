# Phase 0: results of the checks (Oct 15, 2026)

## JSBSim

- `pip install jsbsim` (v1.3.2) works on Windows, and the aircraft data come with it.
- Speed: ~92,000 steps/s on one core (1/120 s step), or about 770 s of simulated flight per second.
  20,000 flights of 30 s take about 13 min on one core.
- Engine off: `propulsion/set-running = 0` works (c172p with the engine off glides at a ratio of about 9, which is realistic).
- Bundled gliders (`scripts/phase0_glider_sweep.py`, steady glide, averaged over 40 s):

| Aircraft | Verdict | Details |
|---|---|---|
| sgs126 (Schweizer 1-26, ~200 kg) | usable | consistent: more nose-up elevator → slower; glide ratio 21–26 at 16–31 m/s; no spurious oscillation |
| minisgs (~5 kg) | not usable | inconsistent glide ratios and angles of attack, oscillations |
| SGS | not usable | absurd glide ratios (up to 65), negative sink rate |
| sgs233 | not usable | crashes / zero states |

Proposed choice: sgs126 as the reference glider, with a randomized family around it, and powered aircraft with the
engine off (c172p, pa28, J3Cub…) as test aircraft with a different structure. One limitation is that sgs126 is a
full-size glider, not a UAV, but the method doesn't depend on that.

## NTNU Skywalker X8 flight dataset (doi:10.18710/NNHVBP)

- 106 maneuvers (53 with a clean wing, 53 with an iced wing), including longitudinal 3-2-1-1 inputs and doublets.
- About 20 Hz with 0.04–0.06 s of jitter, so the data need resampling on `TIME`. Flown at ~100 m and ~18 m/s.
- Useful longitudinal signals: `Va_EKF_CG`, `alpha` (EKF estimate, there is no vane), `theta`, `q`,
  `height`, `u/w`, `elevator` (command), `throttle`, `RPS`, accelerations.
- Parameters provided in `clean_params.json`: mass 3.73 kg, span 2.1 m, chord 0.357 m, S = 0.75 m²,
  inertia (Jy = 0.14 kg·m²), propeller model (CT, CQ), control surface calibration.
- The authors' intended split: IDs 1–3 for training, IDs ≥ 4 for validation.
- Don't use the precomputed coefficients (`CX`…`CL`): they depend on the authors' propeller model.
- License: CC0 on DataverseNO, but the ReadMe says CC-BY 4.0, so the dataset should be cited either way.

## Open decisions for phase 1

- Longitudinal state: (V_a, γ, θ, q, h) or (u, w, θ, q, h). Inputs: elevator, possibly throttle.
- Randomization ranges for the sgs126 family and the test aircraft.
- The three prior levels (fitted / generic / wrong).
- Horizons: 1 s (20 steps) in training, up to 20 s in evaluation.
