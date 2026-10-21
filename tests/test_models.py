"""Unit tests for the world models. Run from the repository root: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from fit_prior import generic_25e  # noqa: E402
from wingjepa.data.windows import N_STATE, Normalizer  # noqa: E402
from wingjepa.models.jepa import TCNEncoder, model_from_config  # noqa: E402
from wingjepa.models.physics import PhysicsHead  # noqa: E402
from wingjepa.sim.ultrastick import (DA, DE, DR, H, OM, P, PHI, PSI, Q, R, TH, U, V, W,  # noqa: E402
                                     UltraStick, isa_density, load_config)


def test_physics_head_matches_simulator():
    """With the Ultra Stick 25e linear model as prior, the physics head must reproduce the simulator's
    equations of motion. Thrust and propeller moments are not in the prior; they are passed as the
    correction term. Elevator and pitch rate are zero, where the linearized drag of the prior is exact."""
    p = load_config(ROOT / "configs/ultrastick25e_validation.yaml")
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


def test_state_conversion_round_trip():
    head = PhysicsHead(dict(coef={k: [0.0] * m for k, m in zip(["CD", "CY", "CL", "Cl", "Cm", "Cn", "T"],
                                                               [7, 5, 5, 5, 5, 5, 6])},
                            rho=1.2, S=0.77, b=1.9, c=0.43, g=9.81), np.zeros(16), np.ones(16))
    s = torch.tensor([[19.0, 0.05, -0.02, 0.1, -0.05, 0.02, 0.3, 0.08, 0.0]])
    x = head.from_features(s)
    back = head.to_features(x, torch.zeros(1))
    np.testing.assert_allclose(back[:, :8].numpy(), s[:, :8].numpy(), atol=1e-5)


def test_tcn_receptive_field_and_causality():
    torch.manual_seed(0)
    enc = TCNEncoder(n_in=N_STATE, channels=(16, 16, 32, 32, 32), dilations=(1, 2, 4, 8, 16), z_dim=8)
    assert enc.receptive_field == 63
    x = torch.randn(1, 80, N_STATE)
    z = enc(x)
    far, near = x.clone(), x.clone()
    far[:, 80 - 64] += 1.0          # just outside the receptive field
    near[:, 80 - 63] += 1.0         # oldest step inside it
    assert torch.allclose(enc(far), z)
    assert not torch.allclose(enc(near), z)


def test_v1_run_config_still_loads():
    norm = Normalizer(np.zeros(16), np.ones(16)).to_dict()
    m = model_from_config(dict(z_dim=32, norm=norm, H=10, seed=0))
    assert m.n_ctx == 0 and m.head == "mlp" and m.encoder.receptive_field == 15
    seq = torch.randn(4, 30, 16)
    pred, extra = m.predict_states(seq, 10)
    assert pred.shape == (4, 20, N_STATE) and extra is None
