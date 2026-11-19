"""Unit tests for the world models. Run from the repository root: python -m pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pytest
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


def test_action_encoder_feeds_the_predictor():
    """With the action encoder (SkyJEPA), the latent of step t+k depends on the commands applied up to step
    t+k-1 but not on later ones; the state latent and the JEPA targets do not read the commands."""
    torch.manual_seed(0)
    norm = Normalizer(np.zeros(16), np.ones(16)).to_dict()
    m = model_from_config(dict(z_dim=24, norm=norm, H=11, channels="8,8,16", dilations="1,2,4",
                               action_encoder=1, act_channels="4,4,8", seed=0))
    H, T, seq = 11, 20, torch.randn(2, 31, 16)
    zp = m.predict_latents(seq, H)
    assert zp.shape == (2, T, 24)
    late = seq.clone()
    late[:, H - 1 + 5, N_STATE:N_STATE + 4] += 1.0                     # command applied from t+5 to t+6
    zl = m.predict_latents(late, H)
    torch.testing.assert_close(zl[:, :5], zp[:, :5])
    assert not torch.allclose(zl[:, 5], zp[:, 5])
    past = seq.clone()
    past[:, H - 5, N_STATE:N_STATE + 4] += 1.0                         # command before t: seen through Enc_phi
    assert not torch.allclose(m.predict_latents(past, H)[:, 0], zp[:, 0])
    cmds = seq.clone()
    cmds[:, :, N_STATE:N_STATE + 4] += 1.0
    torch.testing.assert_close(m.encode(cmds[:, :H]), m.encode(seq[:, :H]))
    torch.testing.assert_close(m.target_latents(cmds, H), m.target_latents(seq, H))


def test_mlp_direct_shapes_causality_and_size():
    """Plain MLP baseline: rebuilt from its config, (B, T, 9) predictions that start from the last state, the
    prediction of step t+k+1 depends on the commands applied up to step t+k only and never on the future
    states; parameter count (13 H + 7 + 1) h1 + (h1 + 1) h2 + 9 (h2 + 1)."""
    torch.manual_seed(0)
    norm = Normalizer(np.zeros(16), np.ones(16)).to_dict()
    m = model_from_config(dict(mode="mlp_direct", head="mlp", z_dim=24, norm=norm, H=11, mlp_hidden="50,50",
                               aircraft_input=0, seed=0))
    H, T, h1, h2 = 11, 20, 50, 50
    n = m.n_params()["mlp"]
    print(f"mlp_direct parameters: {n}")
    assert not m.latent and n == (13 * H + 7 + 1) * h1 + (h1 + 1) * h2 + N_STATE * (h2 + 1) == 10559
    seq = torch.randn(2, H + T, 16)
    pred, extra = m.predict_states(seq, H)
    assert pred.shape == (2, T, N_STATE) and extra is None
    torch.testing.assert_close(pred, seq[:, H - 1:H, :N_STATE].expand(-1, T, -1))     # untrained: constant state
    with torch.no_grad():
        torch.nn.init.normal_(m.net[-1].weight, std=0.1)
    pred, _ = m.predict_states(seq, H)
    late = seq.clone()
    late[:, H - 1 + 5, N_STATE:N_STATE + 4] += 1.0                     # command applied from t+5 to t+6
    pl = m.predict_states(late, H)[0]
    torch.testing.assert_close(pl[:, :5], pred[:, :5])
    assert not torch.allclose(pl[:, 5], pred[:, 5])
    past = seq.clone()
    past[:, 0, N_STATE:N_STATE + 4] += 1.0                             # oldest command of the window
    assert not torch.allclose(m.predict_states(past, H)[0][:, 0], pred[:, 0])
    fut = seq.clone()
    fut[:, H:, :N_STATE] = 0.0                                         # future states are never read
    fut[:, H:, N_STATE + 4:] += 1.0                                    # nor the future wind, held by default
    torch.testing.assert_close(m.predict_states(fut, H)[0], pred)
    wind = m.predict_states(fut, H, hold_wind=False)[0]
    torch.testing.assert_close(wind[:, 0], pred[:, 0])
    assert not torch.allclose(wind[:, 1], pred[:, 1])


@pytest.mark.parametrize("option", [dict(action_history=2), dict(context_dim=8), dict(head="residual"),
                                    dict(head="physics", correction="param"), dict(head="physics", oracle_params=1)])
def test_runs_with_removed_options_are_refused(option):
    """A run of an earlier study whose architecture is no longer in the code fails with a message naming the tag,
    instead of loading wrong weights; the same keys at their "off" value (written by every run) load."""
    norm = Normalizer(np.zeros(16), np.ones(16)).to_dict()
    prior = dict(coef={k: [0.0] * m for k, m in zip(["CD", "CY", "CL", "Cl", "Cm", "Cn", "T"], [7, 5, 5, 5, 5, 5, 6])},
                 rho=1.2, S=0.77, b=1.9, c=0.43, g=9.81)
    cfg = dict(z_dim=24, norm=norm, H=11, channels="8,8,16", dilations="1,2,4", seed=0, prior=prior,
               action_history=0, context_dim=0, correction="free", oracle_params=0, target_window=0, ema=0.0)
    model_from_config(dict(cfg, head="physics" if option.get("head") == "physics" else "mlp"))
    with pytest.raises(ValueError, match="ultrastick-final"):
        model_from_config(dict(cfg, **option))
