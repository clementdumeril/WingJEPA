# Archived: the Ultra Stick 120 study

The first study of WingJEPA (October to mid-November 2026) flew a simulated Ultra Stick 120, a Python port of the
University of Minnesota 6-DoF model with the NASA wind-tunnel tables. Its results are the
[poster](../../poster/poster.pdf), [docs/RESULTS.md](../../docs/RESULTS.md) and the run files in
[results/](../../results/README.md). The main code has since moved to PX4's `advanced_plane`; this folder keeps what
only served the Ultra Stick.

## Contents

| Path | What it is |
|---|---|
| `ultrastick.py` | the Ultra Stick 120 simulator (NASA tables, propeller and motor, actuators, trim), the linear Ultra Stick 25e model used for validation, and the per-flight randomization of the Ultra Stick datasets |
| `configs/` | aircraft configs (`ultrastick120.yaml`, `ultrastick25e_validation.yaml`) and the data configs `data_v1.yaml` (turbulence), `data_calm.yaml`, `data_calm_fixed.yaml` |
| `scripts/fetch_umn.sh` | downloads the NASA tables (`FASER_Aero.mat`) into `third_party/umn/` |
| `scripts/validate_ultrastick.py`, `phase0_*.py` | simulator validation against the UMN reference, first JSBSim checks |
| `scripts/fit_prior.py` | fitted, generic, wrong and null priors of the physics head |
| `scripts/evaluate.py`, `position_error.py`, `diagnose.py`, `probe_aircraft.py` | earlier evaluations: useful horizon up to 60 s, position error, latent probe, aircraft probe |
| `scripts/plot_*.py`, `make_figures.py`, `results_table.py`, `collect_results.py`, `make_rollout_gif.py`, `make_trajectory_3d.py`, `trajectory_viewer.html` | figures, tables and animations of the Ultra Stick runs |
| `cluster/` | the Slurm scripts of the Ultra Stick runs, benchmarks and smoke tests |
| `tests/test_ultrastick.py` | physics head against the Ultra Stick equations of motion, trim and RK4 step with the NASA tables |

## Reproducing the poster

The tag `ultrastick-final` (commit `1ac620a`) is the code as it was when the poster and `docs/RESULTS.md` were made,
with every option those runs used. Some of them were removed from the main code: the constrained and port-Hamiltonian
physics heads, the oracle and nominal aircraft parameters, the command-history encoders, the JEPA target window, the
EMA target encoder, the context latent and the identification JEPA, the `joint` and `jepa_ft` modes and the residual
head. Main refuses to load a run that used one of them (`model_from_config` names the tag), so for those runs, and for
the cluster scripts in `cluster/`, work from the tag:

```bash
git checkout ultrastick-final
```

## Running the archived code on main

Everything runs from the repository root. The simulator and the data pipeline work as they did; the training,
evaluation and closed-loop scripts are the main ones (`scripts/train_a.py`, `eval_skyjepa.py`, `closed_loop.py`),
which still fly the Ultra Stick when the data config points to it.

```bash
bash archive/ultrastick/scripts/fetch_umn.sh          # NASA tables into third_party/umn/ (not redistributed)
python -c "
import numpy as np
from archive.ultrastick.ultrastick import FaserTables, UltraStick, load_config
p = load_config('archive/ultrastick/configs/ultrastick120.yaml')
ac = UltraStick(p, 1, FaserTables(p['aero_file']))
Y, cmd, thr, ok = ac.trim(np.array([20.0]), h=200.0)
print('trimmed:', ok, '| u, v, w after one RK4 step:', ac.step(Y, cmd, thr, np.zeros((1, 3)), np.zeros((1, 3)), 0.01)[0, :3])"
python -m pytest -q archive/ultrastick/tests
python scripts/generate_flights.py --config archive/ultrastick/configs/data_calm.yaml --set train --shard 0 --n-shards 20
python archive/ultrastick/scripts/fit_prior.py --data-dir data/data_calm
python scripts/train_a.py --data-dir data/data_calm --out runs/c_direct_calm_s0 --mode c_direct \
    --prior data/data_calm/priors/nul.json --H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 \
    --action-encoder 1
python archive/ultrastick/scripts/evaluate.py --model-dir runs/c_direct_calm_s0 --data-dir data/data_calm --set test_long
python scripts/closed_loop.py --config archive/ultrastick/configs/data_calm.yaml --controllers tecs runs/c_direct_calm_s0 \
    --out runs/closed_loop/calm.json
```
