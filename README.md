# WingJEPA: where should physics enter a latent world model of a fixed-wing UAV?

Latent world models that predict, open loop, how an Ultra Stick 120 responds to its four controls and to the wind. The same network trained directly on the state beats every JEPA variant I tried, on two metrics: over three seeds it stays within 5 m, 1 m/s and 10 deg of the simulator for a median 3.7 s, against 1.0 s for JEPA, 2.6 s for JEPA with a physics head (the SkyJEPA recipe) and 1.6 s for "nothing changes". On aircraft outside the training range every model falls back to about 2 s.

![Open-loop predictions of three world models against the simulator on an extreme test aircraft](assets/rollout_extreme_aircraft.gif)

*One start drawn at random on an extreme test aircraft. Each model sees 2 s of measured states, then only the commands applied.*

Independent project for *Physics-Inspired Machine Learning* (UC Berkeley, Fall 2026), started October 2026 and in progress. It builds on SkyJEPA (Rao et al., arXiv 2606.23444), which I am not affiliated with. The questions: does the SkyJEPA recipe (a JEPA world model read out by a physics head) carry over to a fixed-wing aircraft flying in turbulence, does it beat the same network trained directly, and where should physics enter when the physical model is imperfect?

| Model | Where physics enters |
|---|---|
| A0 / A | nowhere: direct state loss (A0) or JEPA latent prediction (A) |
| B | in the loss (soft physics penalty) |
| C0 | capacity control: model C with the prior set to zero |
| C | in the architecture: a physics head integrates Newton-Euler with a prior plus learned forces |
| D | in the architecture, with energy structure (port-Hamiltonian head) |

## Results

Median useful horizon: the first time the prediction is off by more than 5 m in altitude, 1 m/s in airspeed or 10 deg in bank or pitch. Position error: the 3D distance after 5 s, rebuilt from the predicted states, the metric SkyJEPA reports. All models get 2 s of history plus the known mass and inertia. Mean over three seeds (one for the fine-tuning variant); per-run numbers and ranges are in [results/summary.csv](results/summary.csv).

| Model | Held-out flights | Long flights, training-range aircraft | Long flights, **extreme aircraft** | Position error at 5 s |
|---|---|---|---|---|
| Baseline: constant state | 1.50 s | 1.60 s | 1.55 s | 34.4 m |
| A0: direct state loss | 3.58 s | 3.73 s | 1.88 s | **12.8 m** |
| A: JEPA, decoder on the frozen latent | 1.23 s | 1.01 s | 0.55 s | 16.8 m |
| A: JEPA pre-training, then fine-tuning | 3.10 s | 3.25 s | 1.65 s | 13.4 m |
| C-JEPA: physics head on the frozen JEPA trunk | 2.46 s | 2.55 s | 1.32 s | 14.6 m |
| C-direct: physics head trained end to end | 3.57 s | **4.08 s** | 1.92 s | 13.2 m |

![Median useful horizon per model on training-range and extreme aircraft](assets/useful_horizon.png)

What I learned:
- **Nothing built on JEPA beats the same network trained directly.** Frozen JEPA, JEPA with a physics head and JEPA as pre-training all stay below it, and the position error gives the same order as the useful horizon. Seed ranges do not overlap.
- The physics head does improve JEPA, as SkyJEPA reports, but much less: the position error at 5 s drops by 13 % here (16.8 to 14.6 m) against 74 % in their paper (5.56 to 1.43 m). On extreme aircraft it makes JEPA worse (21.6 to 26.6 m).
- Trained end to end, the physics head gives a small but consistent gain on the long flights of training-range aircraft (4.00 to 4.25 s against 3.65 to 3.85 s for A0), nothing on the other two sets, and more altitude drift on extreme aircraft at 60 s (406 to 419 m against 197 to 388 m; [altitude error vs time](assets/altitude_error.png)).
- The prior does not matter (one seed per prior): with a wrong prior or none at all, the head does as well as with the fitted one. The learned correction absorbs whatever the prior gets wrong. It does recover the missing moments (R2 0.45 to 0.78 with the fitted and wrong priors), but not the side force (R2 below 0) and barely the axial force (R2 at most 0.29).
- My working explanation for the gap with SkyJEPA is turbulence: a probe on the JEPA latent shows it drops sideslip, which is mostly turbulence here. A slow (EMA) target encoder made it worse (0.45 s). No model beats "keep the altitude constant" at 60 s (36 m).

## What I built

- Simulator (`wingjepa/sim/`): a Python 6-DoF port of the University of Minnesota UAV Lab Simulink model (UMNUAV_sim_v4) with the NASA Langley wind-tunnel tables of the Ultra Stick 120. Vectorized RK4 at 100 Hz, first-order actuators with rate limits, propeller and motor model, 1-cos gusts and Dryden turbulence (MIL-F-8785C), and a filtered wind estimate.
- Flight generator (`scripts/generate_flights.py`, `wingjepa/sim/pilot.py`): TECS autopilot for speed and altitude, bank and coordination loops, elevator/aileron/rudder excitation, random commands and engine-off segments, and an envelope protection that takes over near stall or the ground. Dataset: 2,000 flights of 30 s and 200 flights of 120 s, half of them on aircraft whose mass, inertia, aerodynamics and motor are outside the training range.
- World models (`wingjepa/models/jepa.py`): causal TCN encoder over 2 s of states (5 dilated layers, receptive field 63 steps) plus mass and inertia, a GRU predictor driven by the commands and the held wind estimate, and a head. About 22k parameters. Training modes in `scripts/train_a.py`: JEPA (latent prediction + SIGReg, optional slow target encoder), JEPA pre-training followed by fine-tuning, direct state loss, joint, and the two physics-head variants.
- Physics head (`wingjepa/models/physics.py`): replaces the MLP decoder. It carries the body-axis state and steps the Newton-Euler equations with the true mass and inertia; forces and moments are a prior plus 6 coefficient corrections computed from the latent.
- Priors (`scripts/fit_prior.py`): fitted by least squares on the simulator (R2 0.60 to 0.93 per coefficient), generic (published Ultra Stick 25e coefficients), wrong (fitted, each coefficient off by 30 %), and none.
- Evaluation (`scripts/evaluate.py`, `scripts/position_error.py`, `scripts/diagnose.py`): open-loop rollouts up to 60 s, useful horizon, position error rebuilt from the predicted states, two trivial baselines, results by turbulence level and with the true wind, and for model C a check of whether the learned correction matches the physics the prior is missing.

## How it works

```
2 s of measured states + mass, inertia --TCN--> z0 --GRU (commands, wind estimate)--> z1, z2, ...
model A:  state_k = MLP(z_k)                      (re-guessed at every step)
model C:  x_{k+1} = RK4[ Newton-Euler(x_k, F_prior(x_k, u_k) + dF(z_k, x_k, u_k); m, I) ]
```

In model C the physical state is carried by the equations from one step to the next, and the network only adds corrective forces. The correction starts at zero, so an untrained head is exactly the prior, and a small penalty keeps it small (as in APHYNITY). C-direct trains everything on the state error; C-JEPA freezes a JEPA trunk and trains only the head, like the physics probe of SkyJEPA.

## How I checked it

- The simulator port matches the UMN reference trajectories to about 0.1 to 1 % of the response (`scripts/validate_ultrastick.py`).
  ![Ultra Stick 25e doublets: UMN Simulink reference vs this port](assets/validation/ultrastick25e_vs_umn_6dof.png)
- Porting the model surfaced four problems in the original, which I fixed and documented: a corrupted cell in the NASA tables (lift coefficient 4.35 instead of about -0.05), the side force applied with the wrong sign, the FASER propulsion coded with the smaller 25e motor, and the propeller-wash term disabled. The side-force sign was checked against a real Baldr flight log: the measured slope is -0.0054 to -0.0059 per deg against -0.0068 in the table.
- Unit tests (`python -m pytest -q`): the physics head reproduces the simulator's equations of motion to 1e-6 when given the same aerodynamic model, and the position rebuilt from predicted states is exact on a straight line with wind and on a full turn.
- Dataset checks (`scripts/check_dataset.py`): energy balance closes with correlation 0.998, the response to each of the four controls has the right sign, no non-finite values.
- Every model is compared with two baselines that need no learning, and with the true wind instead of the estimate.
- New code paths run first on a short job (`cluster/smoke_c.sbatch`) before the long runs. That is how a NaN in the physics head was caught and fixed (pitch passing 90 deg inside an RK4 stage).

## Limitations

- Three seeds for the main models, one for the prior variants and the JEPA variants.
- My JEPA implementation follows the paper, not released code; a tuning I did not try could change its numbers.
- Simulation only. The comparison with real flight logs is planned, not done.
- The models ignore the wind input: giving them the true wind does not change their predictions.
- The flights come from my autopilot, which shapes the states the models see.
- The 1 m/s airspeed threshold is close to the airspeed fluctuations caused by turbulence.

## Status

- [x] 6-DoF simulator, validation, dataset
- [x] Models A and A0, diagnostics
- [x] Model C, physics head: 4 priors, direct and on a JEPA trunk
- [x] Three seeds for A0, JEPA and both physics-head variants; JEPA pre-training and slow-target variants
- [x] Position error in meters, for comparison with SkyJEPA
- [ ] Same comparison in calm air, to test the turbulence explanation
- [ ] A tighter correction (a few physical parameters instead of free forces), so the prior has to carry the physics
- [ ] Models B and D
- [ ] Check on real Ultra Stick 120 flight logs (UMN)
- [ ] Report

## Reproduce

```bash
python -m venv ~/.venvs/wingjepa && source ~/.venvs/wingjepa/bin/activate
pip install -e ".[dev]"
bash scripts/fetch_umn.sh                      # NASA tables, not redistributed here
```

On Savio (see [cluster/README.md](cluster/README.md)), from the repository root:

```bash
sbatch --array=0-19 cluster/gen_data.sbatch configs/data_v1.yaml train 20
sbatch --array=0-3  cluster/gen_data.sbatch configs/data_v1.yaml test_long 4
sbatch cluster/smoke_c.sbatch                 # also writes the four priors
sbatch cluster/train_a.sbatch a0 0 v2T100 --T 20 --T-max 100
sbatch --time=24:00:00 cluster/train_a.sbatch c_direct 0 v2T100_ajuste --T 20 --T-max 100 --prior data/data_v1/priors/ajuste.json
sbatch cluster/make_assets.sbatch             # results/ and the figures of this page
```

On 16 CPU cores, A0 trains and evaluates in 1.3 h and JEPA in 8.5 h; physics-head models take 8 to 18 h because of the RK4 inside the training loop. `cluster/make_assets.sbatch` runs the unit tests, then collects the results and draws the figures of this page.

## Credits

- Simulator model and Ultra Stick 120 / 25e data: University of Minnesota UAV Research Group, UMNUAV_sim_v4 (BSD license), with wind-tunnel tables from NASA Langley. The tables are downloaded by `scripts/fetch_umn.sh` and not included here.
- Baldr flight data: University of Minnesota UAV Research Group.
- SkyJEPA, Rao et al., arXiv 2606.23444 (architecture and the idea of a physics probe).
- LeJEPA, Balestriero and LeCun, 2025 (SIGReg regularizer).
- APHYNITY, Yin et al., ICLR 2021 (keeping the learned correction to a physical model small).
- TECS, Lambregts, 1983; Beard and McLain, *Small Unmanned Aircraft*, 2012.
