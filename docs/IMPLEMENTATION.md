# Implementation details

Detail moved out of the top-level README. Results: [RESULTS.md](RESULTS.md) (current setting) and
[`../results/README.md`](../results/README.md) (earlier setting).

## Models

Every recipe uses the same network. JEPA trains the encoder and predictor with a JEPA loss (latent prediction +
SIGReg), then a probe on the frozen latent; A0 trains the same network on the state error; the physics head carries
the body-axis state with the equations of motion and learns the aerodynamic coefficients from the latent. Code:
`wingjepa/models/jepa.py`, `wingjepa/models/physics.py`.

Current setting (SkyJEPA's), about 11k parameters (A0 10,713; C 11,542; D 15,052):

- Input: 0.55 s of states (11 steps x 9 at 20 Hz: Va, alpha, beta, p, q, r, phi, theta, vertical speed). The MLP
  models get no aircraft parameters; the physics heads use the true mass and inertia.
- Encoder: causal TCN, dilations 1, 2, 4, latent of size 24. The 4 commands go through a small action TCN; the 3-axis
  wind estimate is held at its last value.
- Predictor: GRU, trained on 1 s open-loop rollouts, evaluated on 5 s.
- Recipes: JEPA (frozen, MLP or physics probe), A0, B (A0 plus a penalty on the kinematic residuals of the predicted
  states: attitude kinematics, vertical speed), one-step state loss, A0 on noisy inputs.
- Heads: MLP; physics head C (Newton-Euler, RK4, 6 learned body-axis coefficients per step, no aerodynamic prior);
  port-Hamiltonian head D ((J - R) e + G u, J skew, R positive semi-definite, |J|, |G|, |R| <= 10).

Earlier setting, about 22k parameters: 2 s of states (40 steps) plus the mass and inertia (5 numbers) as input;
dilations 1, 2, 4, 8, 16 (63-step receptive field), latent of size 32; rollouts of 1 to 5 s in training, up to 60 s
in evaluation; physics head with a fitted aerodynamic prior plus 6 learned corrections, or (`--correction param`)
the 38 prior coefficients changed once per flight.

## Components

| Component | Path | What it is |
| --- | --- | --- |
| Truth simulator | `wingjepa/sim/ultrastick.py`, `wind.py` | 6-DoF port of UMNUAV_sim_v4: RK4 at 100 Hz, actuators with rate limits, propeller and motor, 1-cos gusts, Dryden turbulence (MIL-F-8785C), filtered wind estimate |
| Flight generator | `scripts/generate_flights.py`, `wingjepa/sim/pilot.py` | TECS speed and altitude autopilot, bank and coordination loops, excitations, envelope protection; 2,000 flights of 30 s and 200 of 120 s, half of these on extreme aircraft |
| World models | `wingjepa/models/jepa.py`, `scripts/train_a.py` | TCN encoder, GRU predictor, MLP decoder; training modes `jepa`, `jepa_ft`, `a0`, `joint`, `c_direct`, `c_jepa`; restart from checkpoint |
| Physics head | `wingjepa/models/physics.py` | Newton-Euler with the true mass and inertia, prior forces plus learned corrections |
| Priors | `scripts/fit_prior.py` | fitted by least squares, generic (published Ultra Stick 25e), wrong (30 % off), none |
| Evaluation | `scripts/eval_skyjepa.py` | current setting: 5 s open-loop errors on both aircraft sets, noise sweep |
| Closed loop | `scripts/closed_loop.py` | MPPI with each model against the TECS autopilot, sensor-noise sweep |
| Evaluation, earlier setting | `scripts/evaluate.py`, `position_error.py`, `diagnose.py` | open-loop rollouts up to 60 s, useful horizon, position error, two trivial baselines, true-wind ablation, latent probe |
| Results and figures | `scripts/collect_results.py`, `results_table.py`, `make_figures.py`, `make_trajectory_3d.py` | `results/summary.csv`, the tables and the figures in `docs/figures/` |
| Cluster jobs | `cluster/` | data generation, training, smoke tests, figure export |

## Checks

Unit tests show the physics head reproduces the simulator's equations of motion to a relative 1e-6 when given the
same aerodynamic model, and that the position rebuilt from the states is exact on a line and a turn. The port is
compared with the UMN Simulink reference trajectories ([figure](figures/validation/ultrastick25e_vs_umn_6dof.png),
[Ultra Stick 120 doublets](figures/validation/ultrastick120_6dof_doublets.png)). Porting surfaced problems in the
original model, which I fixed in `wingjepa/sim/ultrastick.py`: a corrupted cell of the NASA tables (a lift increment
of 4.352), the side force applied with the wrong sign, the FASER propulsion coded with the smaller 25e motor, and the
propeller-wash term disabled.

## Training runs

Each run writes `config.json`, `log.json`, `eval_train.json` and `eval_test_long.json`, copied to `results/<run>/` by
`scripts/collect_results.py`. From `log.json`: A0 trains in 1.7 h on 4 CPU threads, C-direct in 7.7 h on 2 threads
with `--compile 1` (the RK4 is inside the training loop). Model weights stay on the cluster. The calm-air runs use the
same commands as the turbulent ones with `configs/data_calm.yaml`.

## Figures

All in [`figures/`](figures/). Current setting: [`paper_fig2.png`](figures/paper_fig2.png)
(`scripts/plot_paper.py`) and [`closed_loop_noise.png`](figures/closed_loop_noise.png) (`scripts/plot_closed_loop.py`).

Earlier setting, redrawn from the committed results by
`python scripts/make_figures.py --results results --out docs/figures`:

- [`calm_vs_turbulence.png`](figures/calm_vs_turbulence.png): median useful horizon, calm vs turbulent air.
- [`useful_horizon.png`](figures/useful_horizon.png): every model and prior variant, long flights.
- [`position_error.png`](figures/position_error.png): RMS 3D position error at 5 s.
- [`altitude_error.png`](figures/altitude_error.png): altitude error vs prediction time.
- [`probe.png`](figures/probe.png): how well the present state reads back from the encoder latent.

Need the data and the trained models (`cluster/make_assets.sbatch`):
[`rollout_extreme_aircraft.gif`](figures/rollout_extreme_aircraft.gif) (states over 20 s on an extreme aircraft),
[`trajectory_3d_calm.gif`](figures/trajectory_3d_calm.gif) and the interactive
[`trajectory_3d_calm.html`](figures/trajectory_3d_calm.html) (open locally).
The overview figure `figures/overview.svg` (earlier setting) is rendered to PNG with headless Chrome at 2x.

## Credits

- Simulator model and Ultra Stick 120 / 25e data: University of Minnesota UAV Research Group, UMNUAV_sim_v4 (BSD
  license), with wind-tunnel tables from NASA Langley, downloaded by `scripts/fetch_umn.sh`.
- SkyJEPA: Rao et al., arXiv 2606.23444 (architecture and the idea of a physics probe).
- LeJEPA: Balestriero and LeCun, 2025 (SIGReg regularizer).
- APHYNITY: Yin et al., ICLR 2021 (keeping the learned correction to a physical model small).
- TECS: Lambregts, 1983; R. Beard and T. McLain, *Small Unmanned Aircraft*, 2012.
