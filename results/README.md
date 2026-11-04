# Detailed results

Full tables and findings behind the top-level README. Every number comes from `summary.csv` (one row per run, built
by `scripts/collect_results.py`) or from the per-run files in `<run>/`; the probe numbers come from
`<run>/probe.json`.

## Definitions

- Useful horizon: the first time the open-loop prediction is off by more than 5 m in altitude, 1 m/s in airspeed or
  10 deg in bank or pitch. Reported as the median over the flights of a run.
- Position error: the RMS 3D distance after 5 s, rebuilt from the predicted states (the metric SkyJEPA reports).
- Every model gets 2 s of history (40 steps at 20 Hz) plus the known mass and inertia, then predicts alone from the
  commands. Training and test use the same flights, once in calm air (`configs/data_calm.yaml`) and once with
  turbulence and gusts (`configs/data_v1.yaml`).
- Test sets: held-out 30 s flights of training-range aircraft; 120 s flights of training-range aircraft ("long
  flights"); 120 s flights of aircraft whose mass, inertia, aerodynamics and motor are outside the training range
  ("extreme aircraft").
- Mean over three seeds unless marked.

## Run names

`<model>_<dataset>T100_<prior>[_param]_s<seed>`. Model: `a0` (direct state loss), `jepa`, `jepa_ft` (pre-training
then fine-tuning), `joint`, `c_jepa`, `c_direct`. Dataset: `calm` (calm air), `v2` (turbulence). Prior of the physics
head: `ajuste` = fitted, `generique` = generic (published Ultra Stick 25e), `faux` = fitted with every coefficient
30 % off, `nul` = none. `_param`: constrained head. `_ema`: slow target encoder. Runs without `T100` (v1 inputs) are
the first experiments, kept for the record.

## Calm air

| Model | Held-out flights | Long flights, training-range aircraft | Long flights, extreme aircraft | Position error at 5 s (training-range / extreme) |
|---|---|---|---|---|
| Baseline: constant state | 2.15 s | 2.15 s | 2.10 s | 25.4 / 30.8 m |
| A0: direct state loss | 9.56 s | 8.89 s | 2.88 s | 5.5 / 15.4 m |
| A: JEPA, decoder on the frozen latent | 1.93 s | 1.83 s | 0.05 s | 15.8 / 30.9 m |
| A: JEPA pre-training, then fine-tuning (1 seed) | 5.85 s | 5.90 s | 2.30 s | 7.2 / 15.5 m |
| A: JEPA + state loss, trained jointly (1 seed) | 5.15 s | 5.00 s | 1.65 s | |
| C-JEPA: physics head on the frozen JEPA trunk | 4.32 s | 4.07 s | 1.57 s | 9.8 / 27.2 m |
| C-direct: physics head, free correction | 11.43 s | 10.15 s | 2.78 s | 5.4 / 16.4 m |
| C-direct: physics head, constrained | 3.37 s | 3.14 s | 1.50 s | |

## Turbulence

| Model | Held-out flights | Long flights, training-range aircraft | Long flights, extreme aircraft | Position error at 5 s (training-range) |
|---|---|---|---|---|
| Baseline: constant state | 1.50 s | 1.60 s | 1.55 s | 34.4 m |
| A0: direct state loss | 3.58 s | 3.73 s | 1.88 s | 12.8 m |
| A: JEPA, decoder on the frozen latent | 1.23 s | 1.01 s | 0.55 s | 16.8 m |
| A: JEPA pre-training, then fine-tuning (1 seed) | 3.10 s | 3.25 s | 1.65 s | 13.4 m |
| C-JEPA: physics head on the frozen JEPA trunk | 2.46 s | 2.55 s | 1.32 s | 14.6 m |
| C-direct: physics head, free correction | 3.57 s | 4.08 s | 1.92 s | 13.2 m |
| C-direct: physics head, constrained | 2.11 s | 2.05 s | 1.20 s | |

Prior variants of the physics head (turbulence, one seed each) are in `../assets/useful_horizon.png`; the altitude
error over time is in `../assets/altitude_error.png`. No model beats "keep the altitude constant" at 60 s.

## Findings

- Physics helps most as a flexible scaffold. In calm air, C-direct beats A0 by 20 % on held-out flights (11.43
  against 9.56 s) and by 14 % on the long flights of training-range aircraft (10.15 against 8.89 s); the seed ranges
  do not overlap. The position error at 5 s is the same (5.4 against 5.5 m), and on extreme aircraft there is no gain
  (2.78 against 2.88 s). In turbulence the gain mostly disappears (3.57 against 3.58 s on held-out flights).
- The JEPA latent is the bottleneck, and the cause is the objective, not turbulence. A probe trained to read the
  present state back from the latent loses the pitch rate (normalized error 0.82) and the sideslip (0.66) in calm air
  (`jepa_calmT100_s0/probe.json`), more than in turbulence (0.38 and 0.61, `jepa_v2T100_s0/probe.json`). From the
  latent of A0 every quantity reads back to 0.04 or better (`a0_calmT100_s0/probe.json`). JEPA predicts its own
  latent, and dropping fast, hard-to-predict variables makes that easier. Every way of using JEPA stays below A0:
  fine-tuning reaches 5.85 s, and adding the JEPA loss to the state loss costs 44 % of A0's horizon on long flights
  (8.89 to 5.00 s).
- The physics head improves JEPA's position error at 5 s by 38 % in calm air (15.8 to 9.8 m) and by 13 % in
  turbulence (16.8 to 14.6 m). SkyJEPA reports 74 % on its quadrotor task.
- A rigid physics head is much worse. Restricting the network to changing the 38 aerodynamic coefficients of the
  prior, once per flight, drops the horizon to 3.37 s in calm air and 2.11 s in turbulence. With a prior whose
  coefficients are all 30 % off it gets 3.12 s (1 seed). It identifies only the lift coefficient: in calm air,
  across extreme test aircraft, the learned lift change follows the true one (Spearman 0.55 to 0.63 over the 3
  seeds, 0.44 to 0.61 on training-range aircraft), while the six other groups (drag, side force, roll, pitch and yaw
  moments, thrust) stay within ±0.13 (`c_direct_calmT100_ajuste_param_s*/eval_test_long.json`, key
  `identification`). Two likely reasons: the prior's linear structure is imperfect (its least-squares fit, printed by
  `scripts/fit_prior.py`, does not reach R² = 1), so no set of coefficients is exactly right; and 2 s of states
  without the command history does not pin them down.
- With the free correction, the prior does not matter (turbulence, one seed per prior): with a wrong prior or none,
  the head does as well as with the fitted one. The learned correction absorbs whatever the prior gets wrong.

## What is next

The main difference left with SkyJEPA is the input: its world model encodes the history of the actions with a second
encoder, these models see only the states. The command-history encoder is implemented (`--action-history`,
`cluster/smoke_action.sbatch`) and has no results yet.
