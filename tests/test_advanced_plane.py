"""Tests of the advanced_plane port (wingjepa/sim/advanced_plane.py). Run from the repository root:
python -m pytest -q tests/test_advanced_plane.py"""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools" / "compare_advanced_cpp"))
from wingjepa.sim.advanced_plane import (DA, DE, DR, H, OM, P, PHI, PSI, Q, R, TH, U, V, W,  # noqa: E402
                                         AERO_GROUPS, AdvancedPlane, control_matrix, load_config,
                                         sample_overrides, stall_blend)

CFG = load_config(ROOT / "configs/advanced_plane.yaml")
ZERO = np.zeros((1, 3))


def level_state(Va=15.0, alpha=0.0, beta=0.0, omega=400.0):
    Y = np.zeros((1, 16))
    Y[0, U:W + 1] = Va * np.array([np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)])
    Y[0, H], Y[0, OM] = 100.0, omega
    return Y


def test_interface_keys():
    """forces() has every key UltraStick returns (scripts/generate_flights.py reads them)."""
    F = AdvancedPlane(CFG, 1).forces(level_state(), np.array([0.3]), ZERO)
    for k in ("F_aero", "M_aero", "T", "M_prop", "Q_prop", "Q_motor", "P_motor", "P_shaft", "Va", "alpha", "beta",
              "J", "C", "lift", "drag", "side"):
        assert k in F, k
    assert F["C"].shape == (1, 6) and F["F_aero"].shape == (1, 3)


def test_lift_drag_at_zero_alpha():
    """alpha = beta = 0, no rates, surfaces neutral. The stall blend is not exactly 0 at alpha = 0 (M = 15,
    alpha_stall = 19.4 deg: sigma = 1.22 %), so lift = qbar S (1 - sigma) CL0, not qbar S CL0."""
    Va = 15.0
    ac = AdvancedPlane(CFG, 1)
    F = ac.forces(level_state(Va), np.array([0.3]), ZERO)
    qS = 0.5 * CFG["rho"] * Va**2 * CFG["S"]
    e = np.exp(CFG["M"] * CFG["alpha_stall"])
    sigma = (1 + 2 * e) / (1 + e) ** 2
    assert sigma == pytest.approx(0.01224, abs=1e-4)
    CL = (1 - sigma) * CFG["CL0"]
    CD = (1 - sigma) * (CFG["CD0"] + CL**2 / (np.pi * CFG["AR"] * CFG["eff"]))
    assert F["lift"][0] == pytest.approx(qS * CL, rel=1e-12)
    assert F["drag"][0] == pytest.approx(qS * CD, rel=1e-12)
    assert F["side"][0] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(F["F_aero"][0], [-qS * CD, 0.0, -qS * CL], rtol=1e-12, atol=1e-12)
    # pitching moment about the CG: Cm0 about the centre of pressure + the lift at cp (12 cm aft): cp * lift
    assert F["M_aero"][0, 1] == pytest.approx(qS * CFG["c"] * CFG["Cm0"] + CFG["cp"] * qS * CL, rel=1e-12)
    # the hand formula of the task, without the blend, is 1.2 % off
    assert abs(F["lift"][0] / (qS * CFG["CL0"]) - 1) == pytest.approx(sigma, rel=1e-9)


def test_stall_blend():
    a_s, M = CFG["alpha_stall"], CFG["M"]
    assert stall_blend(np.array([a_s, -a_s]), M, a_s) == pytest.approx([0.5, 0.5], abs=1e-4)
    assert stall_blend(np.array([1.0, -1.0, 3.0]), M, a_s) == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)
    assert stall_blend(np.array([0.0]), M, a_s)[0] < 0.013
    # coefficients about the centre of pressure (cp = 0): flat plate past the stall, Cm frozen at its stall value
    ac = AdvancedPlane(CFG, 3, overrides={"cp": np.zeros(3)})
    alphas = np.array([0.8, 0.5, 0.6])
    Y = np.concatenate([level_state(15.0, a) for a in alphas])
    F = ac.forces(Y, np.full(3, 0.3), np.zeros((3, 3)))
    sig = stall_blend(alphas, M, a_s)
    CL_expected = (1 - sig) * (CFG["CL0"] + CFG["CLa"] * alphas) + sig * 2 * np.sin(alphas) ** 2 * np.cos(alphas)
    np.testing.assert_allclose(F["C"][:, 2], CL_expected, rtol=1e-12)
    Cm_stall = CFG["Cm0"] + CFG["Cma"] * a_s
    np.testing.assert_allclose(F["C"][1:, 4], [Cm_stall, Cm_stall], rtol=1e-12)


def test_thrust_and_rotor_lag():
    k = CFG["motor_constant"]
    ac = AdvancedPlane(CFG, 2, overrides={"motor_k": np.array([1.0, 1.2])})
    Y = np.concatenate([level_state(omega=800.0), level_state(omega=800.0)])
    F = ac.forces(Y, np.zeros(2), np.zeros((2, 3)))
    np.testing.assert_allclose(F["T"], [k * 800**2, 1.2 * k * 800**2], rtol=1e-14)
    np.testing.assert_allclose(F["M_prop"][:, 0], CFG["moment_constant"] * F["T"], rtol=1e-14)
    assert k * 1600**2 == pytest.approx(21.88, abs=0.01)                 # thrust_max of the C++
    # omega -> 10 + 1590 throttle: 12.5 ms up, 25 ms down
    thr = np.array([1.0, 0.0])
    d = ac.derivatives(Y, np.zeros((2, 3)), thr, np.zeros((2, 3)), np.zeros((2, 3)))
    np.testing.assert_allclose(d[:, OM], [(1600 - 800) / 0.0125, (10 - 800) / 0.025], rtol=1e-14)


def test_control_signs_and_values():
    """WingJEPA (UMN) conventions: de > 0 nose down, da > 0 roll left, dr > 0 yaw left (and side force right).
    C++ / Gazebo joints: elevator = -de (Cm_ctrl > 0: nose up), aileron = +da, rudder = -dr."""
    ac = AdvancedPlane(CFG, 4, overrides={"cp": np.zeros(4)})
    Y = np.concatenate([level_state() for _ in range(4)])
    Y[1, DE], Y[2, DA], Y[3, DR] = 0.1, 0.1, 0.1
    F = ac.forces(Y, np.full(4, 0.3), np.zeros((4, 3)))
    dM = F["M_aero"][1:] - F["M_aero"][0]
    assert dM[0, 1] < 0                                        # elevator TED: nose down
    assert dM[1, 0] < 0                                        # aileron: roll left
    assert dM[2, 2] < 0                                        # rudder TEL: yaw left
    assert F["F_aero"][3, 1] - F["F_aero"][0, 1] > 0           # rudder TEL: tail pushed right
    G = control_matrix(CFG)
    r2d = 180 / np.pi
    assert G[4, 0] == pytest.approx(-0.025798 * r2d)           # Cm per rad of de
    assert G[3, 1] == pytest.approx(-2 * 0.003331 * r2d)       # Cl per rad of da (both elevons)
    assert G[5, 2] == pytest.approx(-0.001613 * r2d)           # Cn per rad of dr
    assert G[2, 1] == 0 and G[4, 1] == 0                       # elevons in opposition: no lift, no pitch
    # with cp = 0 and alpha = 0 the pitching moment is exactly linear in the elevator coefficient
    qSc = 0.5 * CFG["rho"] * 15.0**2 * CFG["S"] * CFG["c"]
    assert dM[0, 1] == pytest.approx(qSc * G[4, 0] * 0.1, rel=1e-6)


def test_trim_15ms_matches_px4_throttle():
    """C++ (cpp/README.md): trims at 15 m/s with 24.8 % throttle; PX4 FW_THR_TRIM for this airframe is 25 %."""
    ac = AdvancedPlane(CFG, 1)
    Y, cmd, thr, ok = ac.trim(15.0, h=100.0)
    assert ok[0]
    assert thr[0] == pytest.approx(0.248, abs=0.003)
    assert np.degrees(Y[0, TH]) == pytest.approx(0.97, abs=0.05)     # alpha, level flight
    assert np.degrees(cmd[0, 0]) == pytest.approx(-1.90, abs=0.05)   # elevator trailing edge up
    assert abs(Y[0, PHI]) < 1e-3 and np.all(np.abs(cmd[0, 1:]) < np.radians(0.1))
    # the trim is an equilibrium: one second of simulation keeps it
    Yk = Y.copy()
    for _ in range(100):
        Yk = ac.step(Yk, cmd, thr, ZERO, ZERO, 0.01)
    np.testing.assert_allclose(Yk[0, [U, V, W, P, Q, R, PHI, TH]], Y[0, [U, V, W, P, Q, R, PHI, TH]], atol=1e-6)


def test_matches_cpp_transliteration():
    """Forces, moments and derivatives equal the numpy transliteration of dynamics.h after the frame conversion
    (tools/compare_advanced_cpp/compare.py; the real C++ is compared on Savio)."""
    import compare as cmp
    rng = np.random.default_rng(3)
    par = cmp.cpp_params()
    n = 60
    for i in range(n):
        Y = level_state(rng.uniform(5, 30), rng.uniform(-0.8, 1.0), rng.uniform(-0.3, 0.3), rng.uniform(10, 1600))
        Y[0, P:R + 1] = rng.uniform(-2, 2, 3)
        Y[0, PHI], Y[0, TH], Y[0, PSI] = rng.uniform(-1, 1), rng.uniform(-0.6, 0.6), rng.uniform(-3, 3)
        Y[0, DE:DR + 1] = rng.uniform(-0.78, 0.78, 3)
        wind = rng.uniform(-5, 5, 3)
        cmd, thr = rng.uniform(-0.78, 0.78, (1, 3)), rng.uniform(0, 1, 1)
        x, wind_w = cmp.from_wingjepa(Y[0], wind)
        c = np.empty(4)
        c[cmp.AIL], c[cmp.ELE], c[cmp.RUD], c[cmp.THR] = cmd[0, 1], -cmd[0, 0], -cmd[0, 2], thr[0]
        pw = {**par, "wind": wind_w}
        f, m = cmp.cpp_aero(pw, x)
        dx = cmp.cpp_derivative(pw, x, c)
        ac = AdvancedPlane(CFG, 1)
        F = ac.forces(Y, thr, ZERO)
        np.testing.assert_allclose(cmp.D @ (F["F_aero"] + F["F_prop"])[0], f, rtol=1e-10, atol=1e-11)
        np.testing.assert_allclose(cmp.D @ (F["M_aero"] + F["M_prop"])[0], m, rtol=1e-10, atol=1e-11)
        d = ac.derivatives(Y, cmd, thr, ZERO, wind[None])[0]
        ref = cmp.derivative_to_wingjepa(x, dx, wind_w)
        np.testing.assert_allclose(d[[U, V, W, P, Q, R, 9, 10, 11, DE, DA, DR, OM]], ref, rtol=1e-10, atol=1e-9)


def test_sample_overrides_ranges():
    rng = np.random.default_rng(0)
    n = 400
    is_test = np.r_[np.zeros(200, bool), np.ones(200, bool)]
    o = sample_overrides(rng, n, CFG, is_test=is_test)
    r = CFG["randomization"]
    f = o["mass"] / CFG["mass"] - 1
    assert np.all(np.abs(f[:200]) <= r["mass"][0])
    assert np.all((np.abs(f[200:]) >= r["mass"][0]) & (np.abs(f[200:]) <= r["mass"][1]))
    assert set(o["aero_scale"]) <= set(AERO_GROUPS)
    ac = AdvancedPlane(CFG, n, overrides=o)
    assert ac.cp.shape == (n,) and np.all(ac.servo_tau > 0)


def test_extreme_corners_trim():
    """Aircraft with every factor at +-its extreme limit trim at 12 and 18 m/s (the randomization keeps the
    aircraft flyable)."""
    r = CFG["randomization"]
    rng = np.random.default_rng(5)
    n = 64
    lim = np.array([r["mass"][1]] + [r["inertia"][1]] * 3 + [r["motor_k"][1], r["cg_shift"][1], r["servo_tau"][1]]
                   + [r["aero"][k][1] for k in r["aero"]])
    f = 1 + rng.choice([-1.0, 1.0], (n, len(lim))) * lim
    o = dict(mass=CFG["mass"] * f[:, 0], Ixx=CFG["Ixx"] * f[:, 1], Iyy=CFG["Iyy"] * f[:, 2], Izz=CFG["Izz"] * f[:, 3],
             motor_k=f[:, 4], cg_shift=f[:, 5] - 1, servo_tau=CFG["servo_tau"] * f[:, 6],
             aero_scale={k: f[:, 7 + i] for i, k in enumerate(r["aero"])})
    for Vt in (12.0, 18.0):
        Y, cmd, thr, ok = AdvancedPlane(CFG, n, overrides=o).trim(np.full(n, Vt))
        assert ok.all(), Vt
        assert np.all((thr > 0.05) & (thr < 0.6))


def test_generator_switch_end_to_end():
    """scripts/generate_flights.py flies the advanced_plane when the data config points to advanced_plane.yaml."""
    import yaml
    sys.path.insert(0, str(ROOT / "scripts"))
    import os
    from generate_flights import CHANNELS, make_aircraft, simulate
    cfg = yaml.safe_load(open(ROOT / "configs/data_advanced.yaml", encoding="utf-8"))
    rng = np.random.default_rng(0)
    p = load_config(ROOT / cfg["aircraft"]["config"])
    ac, ov = make_aircraft(rng, 3, p, cfg["aircraft"]["ranges"], np.zeros(3, bool))
    assert isinstance(ac, AdvancedPlane) and {"Ixz", "power_scale", "servo_tau"} <= set(ov)
    cwd = os.getcwd()
    os.chdir(ROOT)                                  # the data config names the aircraft config relative to the root
    try:
        rec, meta = simulate(np.random.default_rng(1), 4, 3.0, 2, cfg)
    finally:
        os.chdir(cwd)
    assert rec.shape == (4, 60, len(CHANNELS)) and np.isfinite(rec).all()
    assert meta["trim_ok"].all() and "aero_scale_de" in meta and "servo_tau" in meta
    assert (rec[:, -1, CHANNELS.index("alive")] > 0.5).all()
