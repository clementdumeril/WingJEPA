"""Position rebuilt from predicted states (scripts/eval_skyjepa.py: heading from the predicted rates, air velocity
rotated to north-east plus the held wind, then integrated)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_skyjepa import kinematics  # noqa: E402
from wingjepa.data.windows import STATE  # noqa: E402
from wingjepa.eval.io import DT  # noqa: E402


def level_flight(L, **kw):
    s = np.zeros((1, L, len(STATE)))
    vals = dict(Va=20.0, **kw)
    for k, v in vals.items():
        s[:, :, STATE.index(k)] = v
    return s


def integrate(states, psi0, wind_ne):
    """Positions north, east, altitude (N, L, 3) from the start, as eval_skyjepa.main does."""
    return np.cumsum(kinematics(states, psi0, wind_ne)[1], 1) * DT


def test_straight_flight_east_with_wind():
    pos = integrate(level_flight(20), np.array([np.pi / 2]), np.array([[3.0, 0.0]]))
    np.testing.assert_allclose(pos[0, -1], [3.0, 20.0, 0.0], atol=1e-9)    # 1 s: 20 m east + 3 m of wind north


def test_coordinated_turn_closes_a_circle():
    # yaw rate r at zero bank and pitch: heading turns at r, a full turn returns to the start
    r = 2 * np.pi / 10.0
    pos = integrate(level_flight(200, r=r), np.array([0.0]), np.zeros((1, 2)))
    assert np.linalg.norm(pos[0, -1, :2]) < 0.05 * 20.0 * 10.0 / np.pi
    psi = kinematics(level_flight(200, r=r), np.array([0.0]), np.zeros((1, 2)))[0]
    np.testing.assert_allclose(psi[0, -1], 2 * np.pi, rtol=1e-9)
