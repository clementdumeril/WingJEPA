"""Cross-check of wingjepa/sim/advanced_plane.py against the Fixed-Wing C++ advanced_plane (dynamics.h).

  python tools/compare_advanced_cpp/compare.py make  --out states.csv [--n 400 --seed 0]
  python tools/compare_advanced_cpp/compare.py check --states states.csv [--cpp cpp_out.csv]

make: random states and commands (wide: post-stall angles of attack, large rates and sideslip, wind), written in the
C++ conventions (body FLU, z-up world, quaternion, ground-relative world velocity) for compare_advanced.cpp.
check: converts every row to WingJEPA conventions (body FRD, NED, Euler angles, velocity relative to the mean wind,
UMN surface signs) and evaluates AdvancedPlane.forces / derivatives / step, then compares with
  - a line-by-line numpy transliteration of the C++ (cpp_* functions below, constants typed from dynamics.h, not
    read from the YAML): runs anywhere, checks the port and the frame conversion;
  - the real C++ output (--cpp), when compare_advanced has been run (Savio): also checks the transliteration.
Exit status 1 if an error exceeds its tolerance.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from wingjepa.sim.advanced_plane import (DA, DE, DR, H, OM, P, PHI, PSI, Q, R, TH, U, V, W, XN, YE,  # noqa: E402
                                         AdvancedPlane, dcm_body_to_ned, load_config)

INPUT = ["id", "px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz",
         "a_ail", "a_ele", "a_rud", "a_thr", "c_ail", "c_ele", "c_rud", "c_thr", "wind_x", "wind_y", "wind_z", "dt"]
BODY = ["px", "py", "pz", "qw", "qx", "qy", "qz", "vx", "vy", "vz", "wx", "wy", "wz", "a_ail", "a_ele", "a_rud", "a_thr"]
AIL, ELE, RUD, THR = range(4)
D = np.diag([1.0, -1.0, -1.0])                         # FLU <-> FRD and z-up world <-> NED (rotation by pi about x)

# ----- numpy transliteration of dynamics.h (advanced_plane, advanced_aerodynamics, derivative, rk4) -----------------
PI = np.pi
CL0, CLa, CD0, AR, eff = 0.15188, 5.015, 0.029, 6.5, 0.97
Cem0, Cema, CYb, Cellb, Cenb = 0.075, -0.463966, -0.258244, -0.039250, 0.100826
CDp, CYp, CLp, Cellp, Cemp, Cenp = 0.0, 0.065861, 0.0, -0.487407, 0.0, -0.040416
CDq, CYq, CLq, Cellq, Cemq, Cenq = 0.055166, 0.0, 7.971792, 0.0, -12.140140, 0.0
CDr, CYr, CLr, Cellr, Cemr, Cenr = 0.0, 0.230299, 0.0, 0.078165, 0.0, -0.089947
alpha_stall, Cema_stall = 0.3391428111, 0.0
M_blend, CD_fp_k1, CD_fp_k2 = 15.0, -0.224, -0.115
cp = -0.12
area, mac, density = 0.34, 0.22, 1.2041
CD_ctrl = [-0.000059, -0.000059, -0.000274, 0.0]
CY_ctrl = [0.000171, -0.000171, 0.0, -0.003913]
CL_ctrl = [-0.011940, -0.011940, -0.010696, 0.0]
Cell_ctrl = [-0.003331, 0.003331, 0.0, -0.000257]
Cem_ctrl = [0.001498, 0.001498, 0.025798, 0.0]
Cen_ctrl = [-0.000057, 0.000057, 0.0, 0.001613]
rotor_x, rotor_speed_min, rotor_speed_max, motor_constant = 0.3, 10.0, 1600.0, 8.54858e-06
moment_constant, rotor_drag, rolling_moment = 0.01, 8.06428e-05, 1e-06


def cpp_params():
    rotor = 0.005 * rotor_x * rotor_x
    J = np.diag([0.197563 + 9.75e-07, 0.1458929 + 0.000166704 + rotor, 0.1477 + 0.000167604 + rotor])
    surface = 45 * PI / 180
    return dict(mass=1.005, gravity=np.array([0, 0, -9.8]), J=J, J_inv=np.linalg.inv(J), air_density=density,
                wing_span=np.sqrt(area * AR), thrust_max=motor_constant * rotor_speed_max**2,
                tau=[0.1, 0.1, 0.1, 0.0125], tau_down=0.025,
                limits=[(-surface, surface)] * 3 + [(0.0, 1.0)], wind=np.zeros(3))


def quaternion_multiply(a, b):
    return np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                     a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                     a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                     a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


def rotation_matrix(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def euler_quaternion(roll, pitch, yaw):
    qz = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    qy = [np.cos(pitch / 2), 0, -np.sin(pitch / 2), 0]
    qx = [np.cos(roll / 2), np.sin(roll / 2), 0, 0]
    return quaternion_multiply(quaternion_multiply(qz, qy), qx)


def unpack(x):
    return x[0:3], x[3:7], x[7:10], x[10:13], x[13:17]


def cpp_aero(par, x):
    """advanced_aerodynamics: force and moment in body FLU."""
    pos, quat, vel, w, act = unpack(x)
    Rm = rotation_matrix(quat)
    body = Rm.T @ (vel - par["wind"])
    u, v, wz = body[0], -(body[1] + w[2] * cp), -(body[2] - w[1] * cp)
    f, m = np.zeros(3), np.zeros(3)
    V = np.sqrt(u * u + wz * wz)
    if V > 1e-9:
        S, b, c = area, par["wing_span"], mac
        alpha, beta = np.arctan2(wz, u), np.arctan2(v, u)
        sa, ca = np.sin(alpha), np.cos(alpha)
        Pp, Qq, Rr = w[0], -w[1], -w[2]
        dyn, hrv = 0.5 * par["air_density"] * V * V, 0.5 * par["air_density"] * V
        up, down = np.exp(-M_blend * (alpha - alpha_stall)), np.exp(M_blend * (alpha + alpha_stall))
        sigma = (1 + up + down) / ((1 + up) * (1 + down))
        deg = 180 / PI
        angle = [act[AIL] * deg, -act[AIL] * deg, act[ELE] * deg, act[RUD] * deg]
        CL_c = sum(angle[i] * CL_ctrl[i] for i in range(4))
        CD_c = sum(angle[i] * CD_ctrl[i] for i in range(4))
        CY_c = sum(angle[i] * CY_ctrl[i] for i in range(4))
        Cell_c = sum(angle[i] * Cell_ctrl[i] for i in range(4))
        Cem_c = sum(angle[i] * Cem_ctrl[i] for i in range(4))
        Cen_c = sum(angle[i] * Cen_ctrl[i] for i in range(4))
        sign = 1.0 if alpha >= 0 else -1.0
        CL = (1 - sigma) * (CL0 + CLa * alpha) + sigma * (2 * sign * sa * sa * ca) + CL_c
        lift = (CL * dyn + CLp * (Pp * b / 2) * hrv + CLq * (Qq * c / 2) * hrv + CLr * (Rr * b / 2) * hrv) * S
        CD_fp = 2 / (1 + np.exp(CD_fp_k1 + CD_fp_k2 * max(AR, 1 / AR)))
        CD = ((1 - sigma) * (CD0 + CL * CL / (PI * AR * eff))
              + sigma * abs(CD_fp * (0.5 - 0.5 * np.cos(2 * alpha))) + CD_c)
        drag = (CD * dyn + CDp * (Pp * b / 2) * hrv + CDq * (Qq * c / 2) * hrv + CDr * (Rr * b / 2) * hrv) * S
        CY = CYb * beta + CY_c
        side = (CY * dyn + CYp * (Pp * b / 2) * hrv + CYq * (Qq * c / 2) * hrv + CYr * (Rr * b / 2) * hrv) * S
        if alpha > alpha_stall:
            Cem = Cem0 + Cema * alpha_stall + Cema_stall * (alpha - alpha_stall)
        elif alpha < -alpha_stall:
            Cem = Cem0 - Cema * alpha_stall + Cema_stall * (alpha + alpha_stall)
        else:
            Cem = Cem0 + Cema * alpha
        Cem += Cem_c
        Cell, Cen = Cellb * beta + Cell_c, Cenb * beta + Cen_c
        f[0] = lift * sa - drag * ca
        f[1] = side
        f[2] = -lift * ca - drag * sa
        m[0] = (Cell * dyn + Cellp * (Pp * b / 2) * hrv + Cellq * (Qq * c / 2) * hrv + Cellr * (Rr * b / 2) * hrv) * S * b
        m[1] = (Cem * dyn + Cemp * (Pp * b / 2) * hrv + Cemq * (Qq * c / 2) * hrv + Cemr * (Rr * b / 2) * hrv) * S * c
        m[2] = (Cen * dyn + Cenp * (Pp * b / 2) * hrv + Cenq * (Qq * c / 2) * hrv + Cenr * (Rr * b / 2) * hrv) * S * b
        m[1] += -cp * f[2]
        m[2] += cp * f[1]
    force = np.array([f[0], -f[1], -f[2]])
    moment = np.array([m[0], -m[1], -m[2]])
    omega = rotor_speed_min + (rotor_speed_max - rotor_speed_min) * act[THR]
    thrust = par["thrust_max"] * (omega / rotor_speed_max) * (omega / rotor_speed_max)
    force[0] += thrust
    moment[0] += moment_constant * thrust
    across = [0.0, body[1] + w[2] * rotor_x, body[2] - w[1] * rotor_x]
    rotor_force = [0.0, -omega * rotor_drag * across[1], -omega * rotor_drag * across[2]]
    for i in (1, 2):
        force[i] += rotor_force[i]
        moment[i] += -omega * rolling_moment * across[i]
    moment[1] += -rotor_x * rotor_force[2]
    moment[2] += rotor_x * rotor_force[1]
    return force, moment


def cpp_derivative(par, x, command):
    pos, quat, vel, w, act = unpack(x)
    force, moment = cpp_aero(par, x)
    Rm = rotation_matrix(quat)
    dx = np.empty(17)
    dx[0:3] = vel
    dx[7:10] = Rm @ force / par["mass"] + par["gravity"]
    dx[3:7] = 0.5 * quaternion_multiply(quat, [0, w[0], w[1], w[2]])
    Jw = par["J"] @ w
    rhs = moment - np.cross(w, Jw)
    dx[10:13] = par["J_inv"] @ rhs
    for i in range(4):
        tau = par["tau"][i]
        if i == THR and par["tau_down"] > 0 and command[i] < act[i]:
            tau = par["tau_down"]
        dx[13 + i] = (command[i] - act[i]) / tau
    return dx


def cpp_step(par, x, command, dt):
    k1 = cpp_derivative(par, x, command)
    k2 = cpp_derivative(par, x + dt / 2 * k1, command)
    k3 = cpp_derivative(par, x + dt / 2 * k2, command)
    k4 = cpp_derivative(par, x + dt * k3, command)
    out = x + dt / 6 * k1
    out = out + dt / 3 * k2
    out = out + dt / 3 * k3
    out = out + dt / 6 * k4
    out[3:7] /= np.linalg.norm(out[3:7])
    for i in range(4):
        out[13 + i] = np.clip(out[13 + i], *par["limits"][i])
    return out


# ----- conversions -----------------------------------------------------------------------------------------------
def to_wingjepa(x, wind_w):
    """C++ body (17,) and world wind -> WingJEPA Y (16,), wind_ned (3,)."""
    pos, quat, vel, w, act = unpack(x)
    Rm = rotation_matrix(quat / np.linalg.norm(quat))
    C = D @ Rm @ D                                     # FRD -> NED
    Y = np.zeros(16)
    Y[U:W + 1] = D @ (Rm.T @ (vel - wind_w))           # air-relative (mean wind), FRD
    Y[P:R + 1] = D @ w
    Y[TH] = -np.arcsin(np.clip(C[2, 0], -1, 1))
    Y[PHI] = np.arctan2(C[2, 1], C[2, 2])
    Y[PSI] = np.arctan2(C[1, 0], C[0, 0])
    Y[XN], Y[YE], Y[H] = pos[0], -pos[1], pos[2]
    Y[DE], Y[DA], Y[DR] = -act[ELE], act[AIL], -act[RUD]
    Y[OM] = rotor_speed_min + (rotor_speed_max - rotor_speed_min) * act[THR]
    return Y, D @ wind_w


def command_to_wingjepa(c):
    return np.array([-c[ELE], c[AIL], -c[RUD]]), c[THR]


def from_wingjepa(Y, wind_ned):
    """WingJEPA Y (16,) and NED wind -> C++ body (17,), world wind."""
    q = euler_quaternion(Y[PHI], Y[TH], -Y[PSI])      # C++ yaw is positive nose left
    Rm = rotation_matrix(q)
    C = dcm_body_to_ned(Y[[PHI]], Y[[TH]], Y[[PSI]])[0]
    assert np.allclose(D @ Rm @ D, C, atol=1e-12), "frame conversion: D R(q) D != dcm_body_to_ned"
    wind_w = D @ wind_ned
    x = np.zeros(17)
    x[0:3] = [Y[XN], -Y[YE], Y[H]]
    x[3:7] = q
    x[7:10] = Rm @ (D @ Y[U:W + 1]) + wind_w
    x[10:13] = D @ Y[P:R + 1]
    x[13 + AIL], x[13 + ELE], x[13 + RUD] = Y[DA], -Y[DE], -Y[DR]
    x[13 + THR] = (Y[OM] - rotor_speed_min) / (rotor_speed_max - rotor_speed_min)
    return x, wind_w


def derivative_to_wingjepa(x, dx, wind_w):
    """C++ dx -> d[u v w p q r north east h de da dr omega] (13,) in WingJEPA conventions (Euler rates omitted)."""
    pos, quat, vel, w, act = unpack(x)
    Rm = rotation_matrix(quat / np.linalg.norm(quat))
    vb = Rm.T @ (vel - wind_w)
    dvb = -np.cross(w, vb) + Rm.T @ dx[7:10]         # d/dt R^T (v - wind), dR/dt = R [w]x
    out = np.empty(13)
    out[0:3] = D @ dvb
    out[3:6] = D @ dx[10:13]
    out[6:9] = [dx[0], -dx[1], dx[2]]
    out[9], out[10], out[11] = -dx[13 + ELE], dx[13 + AIL], -dx[13 + RUD]
    out[12] = (rotor_speed_max - rotor_speed_min) * dx[13 + THR]
    return out


# ----- make --------------------------------------------------------------------------------------------------------
def make(n, seed, out):
    rng = np.random.default_rng(seed)
    p = load_config(ROOT / "configs/advanced_plane.yaml")
    lim = p["act_max"][0]
    rows = []
    for i in range(n):
        Va = rng.uniform(4, 30)
        if i % 4 == 0:
            alpha = rng.uniform(-0.9, 1.2)               # deep stall / post-stall, both signs
        else:
            alpha = rng.uniform(-0.25, 0.4)              # flight envelope and around the stall
        beta = rng.uniform(-0.35, 0.35)
        Y = np.zeros(16)
        Y[U:W + 1] = Va * np.array([np.cos(alpha) * np.cos(beta), np.sin(beta), np.sin(alpha) * np.cos(beta)])
        Y[P:R + 1] = rng.uniform(-2, 2, 3) * rng.choice([0.1, 1.0])
        Y[PHI], Y[TH], Y[PSI] = rng.uniform(-1.2, 1.2), rng.uniform(-0.7, 0.7), rng.uniform(-np.pi, np.pi)
        Y[XN], Y[YE], Y[H] = rng.uniform(-500, 500), rng.uniform(-500, 500), rng.uniform(30, 300)
        Y[DE:DR + 1] = rng.uniform(-lim, lim, 3)
        Y[OM] = rng.uniform(10, 1600)
        wind = rng.uniform(-6, 6, 3) * np.array([1, 1, 0.3]) if i % 2 else np.zeros(3)
        if i == 0:                                       # zero airspeed: no aerodynamic force
            Y[U:W + 1] = 0.0
        cmd = rng.uniform(-lim, lim, 3)
        thr = rng.uniform(0, 1)
        dt = 0.02 if i % 5 == 0 else 0.01
        x, wind_w = from_wingjepa(Y, wind)
        c = np.empty(4)
        c[AIL], c[ELE], c[RUD], c[THR] = cmd[1], -cmd[0], -cmd[2], thr
        rows.append(np.r_[i, x, c, wind_w, dt])
    np.savetxt(out, np.array(rows), delimiter=",", header=",".join(INPUT), comments="", fmt="%.17g")
    print(f"wrote {n} rows to {out}")


# ----- check -------------------------------------------------------------------------------------------------------
def errors(name, a, ref, floor, tol, report):
    a, ref = np.asarray(a), np.asarray(ref)
    err = np.abs(a - ref)
    rel = err / np.maximum(np.abs(ref), floor)
    ok = rel.max() <= tol
    report.append((name, err.max(), rel.max(), tol, ok))
    return ok


def check(states, cpp_out):
    tab = np.genfromtxt(states, delimiter=",", names=True)
    data = np.array([[row[k] for k in INPUT] for row in tab])
    n = len(data)
    p = load_config(ROOT / "configs/advanced_plane.yaml")
    par = cpp_params()
    X, CMD, WIND, DT = data[:, 1:18], data[:, 18:22], data[:, 22:25], data[:, 25]

    # WingJEPA side, one batch
    Y = np.zeros((n, 16))
    wind_ned, cmd, thr = np.zeros((n, 3)), np.zeros((n, 3)), np.zeros(n)
    for i in range(n):
        Y[i], wind_ned[i] = to_wingjepa(X[i], WIND[i])
        cmd[i], thr[i] = command_to_wingjepa(CMD[i])
        # round trip of the conversion
        xb, wb = from_wingjepa(Y[i], wind_ned[i])
        sgn = np.sign(xb[3]) * np.sign(X[i, 3]) if X[i, 3] != 0 else 1.0
        assert np.allclose(np.r_[xb[:3], sgn * xb[3:7], xb[7:]], X[i], atol=1e-9), f"round trip, row {i}"
    ac = AdvancedPlane(p, n)
    zero = np.zeros((n, 3))
    F = ac.forces(Y, thr, zero)
    force_py = (F["F_aero"] + F["F_prop"]) @ D        # FRD -> FLU (D symmetric)
    moment_py = (F["M_aero"] + F["M_prop"]) @ D
    dY = ac.derivatives(Y, cmd, thr, zero, wind_ned)
    d_py = dY[:, [U, V, W, P, Q, R, XN, YE, H, DE, DA, DR, OM]]
    groups = {}
    for dt in np.unique(DT):
        sel = DT == dt
        sub = AdvancedPlane(p, int(sel.sum()))
        groups[dt] = (sel, sub.step(Y[sel], cmd[sel], thr[sel], zero[sel], wind_ned[sel], dt))
    Yn_py = np.zeros_like(Y)
    for sel, Yn in groups.values():
        Yn_py[sel] = Yn

    def reference(force, moment, dx, nxt):
        d_ref = np.array([derivative_to_wingjepa(X[i], dx[i], WIND[i]) for i in range(n)])
        Yn_ref = np.array([to_wingjepa(nxt[i], WIND[i])[0] for i in range(n)])
        return force, moment, d_ref, Yn_ref

    def compare(label, ref, report):
        force, moment, d_ref, Yn_ref = ref
        ok = errors(f"{label}: force (N)", force_py, force, 1e-3, 1e-9, report)
        ok &= errors(f"{label}: moment (N m)", moment_py, moment, 1e-3, 1e-9, report)
        ok &= errors(f"{label}: d(u,v,w)", d_py[:, 0:3], d_ref[:, 0:3], 1e-2, 1e-9, report)
        ok &= errors(f"{label}: d(p,q,r)", d_py[:, 3:6], d_ref[:, 3:6], 1e-2, 1e-9, report)
        ok &= errors(f"{label}: d(north,east,h)", d_py[:, 6:9], d_ref[:, 6:9], 1e-2, 1e-9, report)
        ok &= errors(f"{label}: d(surfaces, omega)", d_py[:, 9:13], d_ref[:, 9:13], 1e-2, 1e-9, report)
        dn = Yn_py - Yn_ref
        dn[:, PSI] = (dn[:, PSI] + np.pi) % (2 * np.pi) - np.pi
        step_change = Yn_ref - Y
        step_change[:, PSI] = (step_change[:, PSI] + np.pi) % (2 * np.pi) - np.pi
        # One RK4 step in Euler angles / body velocities vs. quaternion / world velocity: the two discretizations
        # of the same ODE differ by O(dt^5), checked by convergence() (the pass criterion); here the size of that
        # difference is only reported, relative to the change of the group over the step (vector norms).
        # Surfaces and rotor speed are integrated identically in both: tight tolerance.
        for name, idx, floor, tol in (("u,v,w", [U, V, W], 0.1, None), ("p,q,r", [P, Q, R], 0.1, None),
                                      ("phi,theta,psi", [PHI, TH, PSI], 0.01, None),
                                      ("north,east,h", [XN, YE, H], 0.1, None),
                                      ("surfaces", [DE, DA, DR], 1.0, 1e-9), ("omega", [OM], 100.0, 1e-9)):
            err = np.linalg.norm(dn[:, idx], axis=1)
            rel = err / np.maximum(np.linalg.norm(step_change[:, idx], axis=1), floor * DT)
            good = tol is None or rel.max() <= tol
            report.append((f"{label}: next {name} (rel. to the step change)", err.max(), rel.max(), tol, good))
            ok &= good
        return ok

    def convergence(report):
        """Discrepancy Python vs. transliteration after one step of dt, dt/2, dt/4: RK4 in two coordinate sets
        agree to O(dt^5), so it must shrink about 32x per halving (>= 16 required, rows above round-off)."""
        e = []
        for f in (1.0, 0.5, 0.25):
            ei = np.zeros(n)
            for i in range(n):
                dt = DT[i] * f
                y = AdvancedPlane(p, 1).step(Y[i:i + 1], cmd[i:i + 1], thr[i:i + 1], zero[:1], wind_ned[i:i + 1], dt)[0]
                yr = to_wingjepa(cpp_step({**par, "wind": WIND[i]}, X[i], CMD[i], dt), WIND[i])[0]
                d = y - yr
                d[PSI] = (d[PSI] + np.pi) % (2 * np.pi) - np.pi
                ei[i] = np.abs(d[:PSI + 1]).max()
            e.append(ei)
        e = np.array(e)
        sel = e[2] > 1e-10
        r1, r2 = e[0, sel] / e[1, sel], e[1, sel] / e[2, sel]
        good = bool(sel.sum() == 0 or min(r1.min(), r2.min()) >= 16)
        report.append((f"RK4 discrepancy ratio dt/(dt/2), (dt/2)/(dt/4): min {r1.min():.1f}, {r2.min():.1f}; "
                       f"median {np.median(r1):.1f}, {np.median(r2):.1f} ({sel.sum()} rows; columns: max at dt, at dt/4)",
                       e[0].max(), e[2].max(), 16, good))
        return good

    report = []
    nf, nm, nd, nn = [], [], [], []
    for i in range(n):
        f, m = cpp_aero({**par, "wind": WIND[i]}, X[i])
        nf.append(f), nm.append(m)
        nd.append(cpp_derivative({**par, "wind": WIND[i]}, X[i], CMD[i]))
        nn.append(cpp_step({**par, "wind": WIND[i]}, X[i], CMD[i], DT[i]))
    ref_np = reference(np.array(nf), np.array(nm), np.array(nd), np.array(nn))
    ok = compare("AdvancedPlane vs numpy C++ transliteration", ref_np, report)
    ok &= convergence(report)
    if cpp_out:
        out = np.genfromtxt(cpp_out, delimiter=",", names=True)
        ids = out["id"].astype(int)
        assert np.array_equal(ids, data[:, 0].astype(int)), "row ids differ between states and C++ output"
        force_c = np.stack([out[k] for k in ("Fx", "Fy", "Fz")], 1)
        moment_c = np.stack([out[k] for k in ("Mx", "My", "Mz")], 1)
        dx_c = np.stack([out["d_" + k] for k in BODY], 1)
        nx_c = np.stack([out["n_" + k] for k in BODY], 1)
        ref_c = reference(force_c, moment_c, dx_c, nx_c)
        ok &= compare("AdvancedPlane vs C++", ref_c, report)
        ok &= errors("numpy transliteration vs C++: force", np.array(nf), force_c, 1e-3, 1e-9, report)
        ok &= errors("numpy transliteration vs C++: moment", np.array(nm), moment_c, 1e-3, 1e-9, report)
        ok &= errors("numpy transliteration vs C++: dx", np.array(nd), dx_c, 1e-2, 1e-9, report)
        ok &= errors("numpy transliteration vs C++: next", np.array(nn), nx_c, 1.0, 1e-9, report)
    print(f"{n} states; post-stall (|alpha_cp| > alpha_stall): {(np.abs(F['alpha_cp']) > 0.339).sum()}, "
          f"with wind: {(np.abs(WIND).sum(1) > 0).sum()}, rotor slowing down: {(ac.omega_command(thr) < Y[:, OM]).sum()}")
    print(f"{'quantity':58s} {'max abs err':>12s} {'max rel err':>12s} {'tol':>8s}")
    for name, ea, er, tol, good in report:
        tol_s = "info" if tol is None else f"{tol:8.0e}"
        print(f"{name:58s} {ea:12.3e} {er:12.3e} {tol_s:>8s} {'' if tol is None else 'ok' if good else 'FAIL'}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return ok


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    m = sp.add_parser("make")
    m.add_argument("--out", default="states.csv")
    m.add_argument("--n", type=int, default=400)
    m.add_argument("--seed", type=int, default=0)
    c = sp.add_parser("check")
    c.add_argument("--states", default="states.csv")
    c.add_argument("--cpp", default=None, help="output of compare_advanced (optional)")
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.n, a.seed, a.out)
    else:
        sys.exit(0 if check(a.states, a.cpp) else 1)


if __name__ == "__main__":
    main()
