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
from wingjepa.models.physics import COEFS, TERMS, PhysicsHead  # noqa: E402
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


def test_action_history_encoder_sees_only_the_commands_already_applied():
    """With --action-history the latent depends on the command of step H-2 (it produced the state of step
    H-1) but not on the one of step H-1 (applied after the last state, given to the predictor instead)."""
    torch.manual_seed(0)
    norm = Normalizer(np.zeros(16), np.ones(16)).to_dict()
    m = model_from_config(dict(z_dim=32, norm=norm, H=40, channels="16,16,32,32,32", dilations="1,2,4,8,16",
                               aircraft_input=1, action_history=1, seed=0))
    H, ac = 40, torch.full((2, 5), 1.0) * m.ac_nominal
    seq = torch.randn(2, H + 20, 16)
    z = m.encode(seq[:, :H], ac)
    last, before = seq.clone(), seq.clone()
    last[:, H - 1, N_STATE:N_STATE + 4] += 1.0
    before[:, H - 2, N_STATE:N_STATE + 4] += 1.0
    assert torch.allclose(m.encode(last[:, :H], ac), z)
    assert not torch.allclose(m.encode(before[:, :H], ac), z)
    wind = seq.clone()
    wind[:, :H, N_STATE + 4:] += 1.0                                   # the wind is not read
    assert torch.allclose(m.encode(wind[:, :H], ac), z)
    pred, _ = m.predict_states(seq, H, ac=ac)
    assert pred.shape == (2, 20, N_STATE) and m.target_latents(seq, H, ac).shape == (2, 20, 32)
    cmds = seq.clone()
    cmds[:, :, N_STATE:N_STATE + 4] += 1.0                             # the JEPA targets do not see the commands
    assert torch.allclose(m.target_latents(cmds, H, ac), m.target_latents(seq, H, ac))


def test_target_window():
    """With target_window 10, the target of step t+k depends only on the last 10 steps up to t+k; by default
    (target_window 0) the targets are the encoder on the full windows of H steps, as before."""
    torch.manual_seed(0)
    norm = Normalizer(np.zeros(16), np.ones(16)).to_dict()
    cfg = dict(z_dim=32, norm=norm, H=40, channels="16,16,32,32,32", dilations="1,2,4,8,16", seed=0)
    H, T, seq = 40, 20, torch.randn(2, 60, 16)
    full, short = model_from_config(cfg), model_from_config(dict(cfg, target_window=10))
    short.load_state_dict(full.state_dict())
    win = seq[:, :, :N_STATE].unfold(1, H, 1)[:, 1:]                   # the windows used before the option
    want = full.encoder(win.reshape(2 * T, N_STATE, H).transpose(1, 2)).reshape(2, T, -1)
    torch.testing.assert_close(full.target_latents(seq, H), want)
    zt = short.target_latents(seq, H)
    assert zt.shape == (2, T, 32)
    old = seq.clone()
    old[:, H + 4 - 10] += 1.0                                          # just before the window of the target t+5
    assert torch.allclose(short.target_latents(old, H)[:, 4], zt[:, 4])
    assert not torch.allclose(full.target_latents(old, H)[:, 4], full.target_latents(seq, H)[:, 4])


def test_action_encoder_feeds_the_predictor():
    """With the action encoder (SkyJEPA), the latent of step t+k depends on the commands applied up to step
    t+k-1 but not on later ones; the context latent and the JEPA targets do not read the commands."""
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


def _prior_25e():
    p = load_config(ROOT / "configs/ultrastick25e_validation.yaml")
    return dict(coef=generic_25e(p, [0.0] * 6), rho=float(isa_density(100.0)), S=p["S"], b=p["b"], c=p["c"], g=p["g"])


def test_constrained_head_changes_one_coefficient_group():
    """A per-flight change of the Cl group by +20 % scales the rolling moment by 1.2 (at zero incidence,
    where stability and body axes coincide) and leaves the other five components as they are."""
    head = PhysicsHead(_prior_25e(), np.zeros(16), np.ones(16), correction="param").double()
    torch.manual_seed(0)
    n = 5
    alpha = torch.zeros(n, dtype=torch.float64)
    beta, da, dr, p, r = (0.1 * torch.randn(n, dtype=torch.float64) for _ in range(5))
    args = (alpha, beta, 0 * alpha, da, dr, p, 0 * alpha, r, 0.5 + 0 * alpha, 18 + 0 * alpha)
    c, dcoef, i = head.coefficients(), torch.zeros(n, len(head.coef_scale), dtype=torch.float64), 0
    for k in COEFS:
        if k == "Cl":
            dcoef[:, i:i + 5] = 0.2 * c[i:i + 5] / head.coef_scale[i:i + 5]
        i += len(TERMS[k])
    base, changed = head.prior_body(*args), head.prior_body(*args, dcoef=dcoef)
    np.testing.assert_allclose(changed[:, 3].numpy(), 1.2 * base[:, 3].numpy(), rtol=1e-10)
    keep = [0, 1, 2, 4, 5]
    np.testing.assert_allclose(changed[:, keep].numpy(), base[:, keep].numpy(), rtol=1e-12)


def test_constrained_head_starts_as_the_prior():
    """Untrained, the free and the constrained heads both predict exactly the prior's trajectory; the
    constrained head returns one set of 38 coefficient changes per prediction."""
    torch.manual_seed(0)
    B, T = 3, 8
    s0 = torch.tensor([[18.0, 0.05, 0.01, 0.05, -0.02, 0.03, 0.2, 0.06, 0.0]]).repeat(B, 1)
    U = torch.cat([0.02 * torch.randn(B, T, 3), torch.full((B, T, 1), 0.5), torch.zeros(B, T, 3)], -1)
    ac = torch.tensor([[6.8, 0.67, 0.89, 1.58, -0.76]]).repeat(B, 1)
    zs = torch.randn(B, T, 32)
    out = {}
    for mode in ("free", "param"):
        head = PhysicsHead(_prior_25e(), np.zeros(16), np.ones(16), correction=mode)
        out[mode] = head(zs, s0, U, ac)
    torch.testing.assert_close(out["free"][0], out["param"][0])
    assert out["free"][1].shape == (B, T, 6) and out["param"][1].shape == (B, 1, 38)


def test_port_hamiltonian_head_cannot_create_energy():
    """porth: for any network output, J is skew and R positive semi-definite, so the power of the learned
    aerodynamics without commands, qS Va e_hat^T (J - R) e_hat, is <= 0; with commands it is at most the input
    power e_hat^T G u. Untrained, the head runs, returns (B, T, 60) raw outputs and stays close to the prior."""
    torch.manual_seed(0)
    head = PhysicsHead(_prior_25e(), np.zeros(16), np.ones(16), correction="porth").double()
    B = 64
    M = 3 * torch.randn(B, 60, dtype=torch.float64)
    J, R, G = head.porth_matrices(M)
    torch.testing.assert_close(J, -J.transpose(1, 2))
    assert torch.linalg.eigvalsh(R).min() > -1e-10
    Va, alpha, beta = 15 + 10 * torch.rand(B, dtype=torch.float64), 0.2 * torch.randn(B, dtype=torch.float64), \
        0.1 * torch.randn(B, dtype=torch.float64)
    p, q, r = (torch.randn(B, dtype=torch.float64) for _ in range(3))
    cb = torch.cos(beta)
    e = torch.stack([torch.cos(alpha) * cb, torch.sin(beta), torch.sin(alpha) * cb,
                     p * head.b / Va, q * head.c / Va, r * head.b / Va], -1)
    zero = torch.zeros(B, 4, dtype=torch.float64)
    assert ((head.porth_body(M, Va, alpha, beta, p, q, r, zero) * e).sum(-1) <= 1e-10).all()
    u = torch.randn(B, 4, dtype=torch.float64)
    power = (head.porth_body(M, Va, alpha, beta, p, q, r, u) * e).sum(-1)
    assert (power <= (e[:, :, None] * G * u[:, None, :]).sum((1, 2)) + 1e-10).all()

    Bt, T = 3, 8
    s0 = torch.tensor([[18.0, 0.05, 0.01, 0.05, -0.02, 0.03, 0.2, 0.06, 0.0]]).repeat(Bt, 1)
    U = torch.cat([0.02 * torch.randn(Bt, T, 3), torch.full((Bt, T, 1), 0.5), torch.zeros(Bt, T, 3)], -1)
    ac = torch.tensor([[6.8, 0.67, 0.89, 1.58, -0.76]]).repeat(Bt, 1)
    zs = torch.randn(Bt, T, 32)
    pred, raw = PhysicsHead(_prior_25e(), np.zeros(16), np.ones(16), correction="porth")(zs, s0, U, ac)
    ref, _ = PhysicsHead(_prior_25e(), np.zeros(16), np.ones(16), correction="free")(zs, s0, U, ac)
    assert raw.shape == (Bt, T, 60) and torch.isfinite(pred).all()
    assert (pred - ref).abs().max() < 0.05                     # only the small initial damping differs


def test_constrained_head_bound():
    """With --param-bound B, even a huge network output changes no coefficient by more than B * scale, and
    a small one is left as without bound."""
    head = PhysicsHead(_prior_25e(), np.zeros(16), np.ones(16), correction="param", param_bound=1.0)
    z = torch.randn(4, 32)
    with torch.no_grad():
        head.net[-1].bias.fill_(1e3)
        assert head.correction(z)[0].abs().max() <= 1.0
        head.net[-1].bias.fill_(1e-3)
        torch.testing.assert_close(head.correction(z)[0], torch.full((4, 38), 1e-4), rtol=1e-4, atol=0)
