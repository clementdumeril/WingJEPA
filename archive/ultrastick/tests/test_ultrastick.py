"""Checks of the archived Ultra Stick model. Run from the repository root: python -m pytest -q archive/ultrastick/tests
The RK4 step with the NASA tables needs FASER_Aero.mat (bash archive/ultrastick/scripts/fetch_umn.sh); another copy can
be given with the environment variable FASER_AERO."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "archive/ultrastick/scripts")]
from archive.ultrastick.ultrastick import (DA, DE, DR, H, OM, P, PHI, PSI, Q, R, TH, U, V, W,  # noqa: E402
                                           FaserTables, UltraStick, isa_density, load_config)
from fit_prior import generic_25e  # noqa: E402
from wingjepa.models.physics import PhysicsHead  # noqa: E402

CONFIGS = ROOT / "archive/ultrastick/configs"
FASER = Path(os.environ.get("FASER_AERO", ROOT / "third_party/umn/FASER_Aero.mat"))


def test_physics_head_matches_simulator():
    """With the Ultra Stick 25e linear model as prior, the physics head must reproduce the simulator's
    equations of motion. Thrust and propeller moments are not in the prior; they are passed as the
    correction term. Elevator and pitch rate are zero, where the linearized drag of the prior is exact."""
    p = load_config(CONFIGS / "ultrastick25e_validation.yaml")
    n = 6
    rng = np.random.default_rng(0)
    Y = np.zeros((n, 16))
    Y[:, U] = rng.uniform(14, 24, n)
    Y[:, V] = rng.uniform(-1.5, 1.5, n)
    Y[:, W] = rng.uniform(-1, 3, n)
    Y[:, P], Y[:, R] = rng.uniform(-0.5, 0.5, n), rng.uniform(-0.3, 0.3, n)
    Y[:, PHI], Y[:, TH], Y[:, PSI] = rng.uniform(-0.6, 0.6, n), rng.uniform(-0.3, 0.3, n), 0.3
    Y[:, H] = 100.0
    Y[:, DA], Y[:, DR] = rng.uniform(-0.1, 0.1, n), rng.uniform(-0.1, 0.1, n)
    Y[:, OM] = 700.0
    cmd, thr = Y[:, DE:DR + 1].copy(), np.full(n, 0.6)
    ac = UltraStick(p, n)
    zero = np.zeros((n, 3))
    dY = ac.derivatives(Y, cmd, thr, zero, zero)
    F = ac.forces(Y, thr, zero)

    prior = dict(coef=generic_25e(p, [0.0] * 6), rho=float(isa_density(100.0)), S=p["S"], b=p["b"], c=p["c"], g=p["g"])
    head = PhysicsHead(prior, np.zeros(16), np.ones(16)).double()
    qS = 0.5 * prior["rho"] * F["Va"] ** 2 * p["S"]
    dC = np.stack([F["T"] / qS, 0 * qS, 0 * qS, F["M_prop"][:, 0] / (qS * p["b"]),
                   F["M_prop"][:, 1] / (qS * p["c"]), F["M_prop"][:, 2] / (qS * p["b"])], 1)
    x = torch.tensor(Y[:, [U, V, W, P, Q, R, PHI, TH]])
    ctrl = torch.tensor(np.c_[cmd, thr])
    acp = torch.tensor([[p["mass"], p["Ixx"], p["Iyy"], p["Izz"], p["Ixz"]]] * n, dtype=torch.float64)
    got = head.derivatives(x, ctrl, torch.tensor(dC), acp).numpy()
    want = dY[:, [U, V, W, P, Q, R, PHI, TH]]
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-8)


@pytest.mark.skipif(not FASER.exists(), reason=f"no NASA tables at {FASER} (archive/ultrastick/scripts/fetch_umn.sh)")
def test_trim_and_rk4_step_with_the_nasa_tables():
    """Ultra Stick 120 with the FASER tables: trimmed level flight at 18 m/s stays level over one 10 ms RK4 step."""
    p = load_config(CONFIGS / "ultrastick120.yaml")
    ac = UltraStick(p, 2, FaserTables(FASER))
    Y, cmd, thr, ok = ac.trim(np.array([17.0, 20.0]), h=200.0)
    assert ok.all()
    zero = np.zeros((2, 3))
    Yn = ac.step(Y, cmd, thr, zero, zero, 0.01)
    assert np.isfinite(Yn).all()
    np.testing.assert_allclose(Yn[:, [U, W, Q, TH, H]], Y[:, [U, W, Q, TH, H]], atol=1e-4)
