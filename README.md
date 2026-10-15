# WingJEPA

**Where should physics enter a latent world model?** JEPA-style dynamics models for fixed-wing flight under model misspecification.

Independent project for *Physics-Inspired Machine Learning* (UC Berkeley, Fall 2026), inspired by SkyJEPA (Rao et al., arXiv 2606.23444). Not affiliated with the SkyJEPA authors.

## Question

Does embedding an imperfect physical model into a latent dynamics architecture improve generalization compared with enforcing the same physics through the training objective?

| Model | Where physics enters |
|---|---|
| A | nowhere (unconstrained predictor) |
| B | loss (soft physics regularization) |
| C0 | capacity control (C with the physics integrator replaced by an equal-size network) |
| C | architecture (physics-based latent decoder + learned residual) |
| D | architecture + energy structure (port-Hamiltonian decoder) |

Truth data: JSBSim (nonlinear aerodynamics). Prior: simplified linear longitudinal aerodynamics at three levels of misspecification. Real-flight check: NTNU Skywalker X8 dataset (doi:10.18710/NNHVBP).

## Layout

```
configs/      experiment configuration files
wingjepa/     package: sim/ physics/ data/ models/ train/ eval/
scripts/      entry points (data generation, training, evaluation, phase-0 checks)
cluster/      Savio (BRC) setup and Slurm job templates
docs/         conventions, derivations, findings
tests/        unit tests
```

## Setup (local, Windows)

The environment and the data live outside OneDrive.

```bash
python -m venv C:/Users/admin/venvs/wingjepa
C:/Users/admin/venvs/wingjepa/Scripts/python -m pip install -e ".[dev]"
```

Cluster setup: see [cluster/README.md](cluster/README.md).

## Status

- [x] Phase 0: JSBSim checks, X8 dataset check, repository skeleton, cluster templates — see [docs/phase0_findings.md](docs/phase0_findings.md)
- [ ] Phase 1: longitudinal data generation
- [ ] Phase 2: models A, B, C0, C, D
- [ ] Phase 3: experiments on Savio
- [ ] Phase 4: analysis and X8 real-flight check
- [ ] Phase 5: report and presentation
