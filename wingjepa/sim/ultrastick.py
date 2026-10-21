"""Ground-truth 6-DoF simulator of the Ultra Stick 120 (FASER), vectorized over a batch of flights.

Python translation of UMNUAV_sim_v4 (University of Minnesota UAV Lab, BSD, Simulink 2011) :
  - aero (FASER_Aero_Lib.mdl): NASA Langley wind-tunnel tables (FASER_Aero.mat), linear
    interpolation with linear extrapolation (PreLookup / Interpolation_n-D blocks); coefficients
    [CD, CY, CL, Cl, Cm, Cn] in stability axes:
      base(alpha, beta) + elevator(alpha, de, J) -> CD, CL, Cm + rudder(alpha, beta, dr, J) -> CY, Cn
      + ailerons(alpha, beta, da) -> Cl + power(alpha, beta, J) -> all 6
      + damping: CLq, Cmq (q c/2V); CYr, Clr, Cnr (r b/2V); Clp, Cnp (p b/2V);
    then stability axes -> body axes ("Aerodynamic Forces and Moments" block);
  - propulsion (Propulsion_Lib.mdl): motor power = throttle polynomial, torque = P / omega,
    propeller C_T(J), C_P(J), J = V pi / (omega R); moments: [motor torque, Jmp omega r, -Jmp omega q];
  - control surfaces (Actuator_Lib.mdl): first order w/(s+w), rate and deflection limited;
  - motion: 6DoF rigid body, Euler angles, flat Earth (Aerospace Blockset 6DoF block);
    mean wind added to the inertial velocities, unsteady wind to the body velocities (as in UMN).
The linear model of the Ultra Stick 25e (Ultrastick_Aero_Lib.mdl) is included for validation.
Differences from the UMN simulator: docs/phase1_decisions.md.

State Y (n, 16): u v w | p q r | phi theta psi | north east h | de da dr (positions) | omega (propeller)
Inputs: de, da, dr (rad) and throttle (0-1). UMN conventions: elevator +TED (nose down),
ailerons da = (da_r - da_l)/2 +TED, rudder +TEL.
"""
import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator

(U, V, W, P, Q, R, PHI, TH, PSI, XN, YE, H, DE, DA, DR, OM) = range(16)
N_STATE = 16
D2R, R2D = np.pi / 180, 180 / np.pi


def isa_density(h):
    """1976 standard atmosphere, troposphere."""
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dcm_body_to_ned(phi, th, psi):
    """Matrices (n, 3, 3): v_NED = C @ v_body."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    C = np.empty(phi.shape + (3, 3))
    C[:, 0, 0], C[:, 0, 1], C[:, 0, 2] = ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp
    C[:, 1, 0], C[:, 1, 1], C[:, 1, 2] = ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp
    C[:, 2, 0], C[:, 2, 1], C[:, 2, 2] = -st, sf * ct, cf * ct
    return C


class FaserTables:
    """FASER NASA tables (FASER_Aero.mat)."""

    def __init__(self, mat_path):
        from scipy.io import loadmat
        m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        lin = dict(method="linear", bounds_error=False, fill_value=None)   # linear extrapolation
        ax = lambda s, f: np.asarray(getattr(s, f), float)
        b = m["C6_bas"]
        self.base = RegularGridInterpolator((ax(b, "alpha"), ax(b, "beta")), b.data, **lin)       # 6
        e = m["dC3_ele"]
        ele = np.array(e.data, float)
        # Corrupted cell (the only one, found by comparison with its neighbors): alpha = -5 deg,
        # elevator = -10 deg, J = 0.389 -> dCD, dCL, dCm = -0.805, 4.352, -3.083 while the
        # neighbors are ~0.02, -0.05, 0.13. UMN never reached it (J fixed at 0.7). It is replaced
        # by the linear extrapolation in alpha from alpha = 0 and 5 deg (same elevator, same J).
        if abs(ele[0, 0, 1, 1] - 4.352) < 1e-3:
            ele[0, 0, 1] = 2 * ele[1, 0, 1] - ele[2, 0, 1]
        self.ele = RegularGridInterpolator((ax(e, "alpha"), ax(e, "ele"), ax(e, "J")), ele, **lin)  # CD CL Cm
        r = m["dC2_rud"]
        self.rud = RegularGridInterpolator((ax(r, "alpha"), ax(r, "beta"), ax(r, "rud"), ax(r, "J")),
                                           r.data, **lin)                                         # CY Cn
        a = m["dC1_ail"]
        self.ail = RegularGridInterpolator((ax(a, "alpha"), ax(a, "beta"), ax(a, "ail")), a.data, **lin)  # Cl
        pw = m["dC6_pow"]
        self.pow = RegularGridInterpolator((ax(pw, "alpha"), ax(pw, "beta"), ax(pw, "J")), pw.data, **lin)
        # J saturated to the table ranges (Saturation blocks of FASER_Aero_Lib.mdl)
        self.J_ele = (ax(e, "J").min(), ax(e, "J").max())
        self.J_rud = (ax(r, "J").min(), ax(r, "J").max())
        self.J_pow = (ax(pw, "J").min() + 2.2e-16, ax(pw, "J").max() + 2.2e-16)
        d = m["dampder"]
        self.damp = {k: float(getattr(d, k)) for k in ("CLq", "Cmq", "CYr", "Clp", "Clr", "Cnp", "Cnr")}

    def coeffs(self, alpha, beta, de, da, dr, phat, qhat, rhat, J):
        """[CD, CY, CL, Cl, Cm, Cn] (n, 6), stability axes. Angles in rad."""
        a, b = alpha * R2D, beta * R2D
        C = self.base(np.stack([a, b], 1)) + self.pow(np.stack([a, b, np.clip(J, *self.J_pow)], 1))
        e = self.ele(np.stack([a, de * R2D, np.clip(J, *self.J_ele)], 1))
        C[:, [0, 2, 4]] += e
        C[:, [1, 5]] += self.rud(np.stack([a, b, dr * R2D, np.clip(J, *self.J_rud)], 1))
        C[:, 3] += self.ail(np.stack([a, b, da * R2D], 1))
        d = self.damp
        C[:, 1] += d["CYr"] * rhat
        C[:, 2] += d["CLq"] * qhat
        C[:, 3] += d["Clp"] * phat + d["Clr"] * rhat
        C[:, 4] += d["Cmq"] * qhat
        C[:, 5] += d["Cnp"] * phat + d["Cnr"] * rhat
        return C


def linear_25e_coeffs(p, alpha, beta, de, da, dr, phat, qhat, rhat):
    """Linear model of the Ultra Stick 25e (Ultrastick_Aero_Lib.mdl), flaps at zero."""
    cl, cd, cm, cy, cr, cn = p["CL"], p["CDw"], p["Cm"], p["CYw"], p["Cl"], p["Cn"]
    CL = cl["zero"] + cl["alpha"] * alpha + cl["delev"] * de + cl["q"] * qhat
    AR = p["b"] ** 2 / p["S"]
    CD = (cd["min"] + (CL - cl["minD"]) ** 2 / (np.pi * AR * cd["osw"]) + np.abs(cd["delev"] * de)
          + np.abs(cd["dail"] * da) + np.abs(cd["drud"] * dr))
    # Sign: the Simulink "Aerodynamic Forces and Moments" block inverts CY (F_y = -qS CY) and
    # Ultrastick_Aero_Lib compensates with a gain of -1; overall, the lateral force is the physical value
    # below (checked against doublet_fullmodel.mat: lateral errors divided by 25-30).
    CY = cy["beta"] * beta + cy["dail"] * da + cy["drud"] * dr + cy["p"] * phat + cy["r"] * rhat
    Cl = cr["beta"] * beta + cr["dail"] * da + cr["drud"] * dr + cr["p"] * phat + cr["r"] * rhat
    Cm = cm["zero"] + cm["alpha"] * alpha + cm["delev"] * de + cm["q"] * qhat
    Cn = cn["beta"] * beta + cn["dail"] * da + cn["drud"] * dr + cn["p"] * phat + cn["r"] * rhat
    return np.stack([CD, CY, CL, Cl, Cm, Cn], 1)


class UltraStick:
    """Batch of n aircraft. p: YAML config; overrides: dict of per-flight (n,) arrays."""

    def __init__(self, p, n, tables=None, overrides=None):
        self.p, self.n, self.tables = p, n, tables
        o = overrides or {}
        bc = lambda k, d: np.broadcast_to(o.get(k, d), (n,)).astype(float)
        self.mass = bc("mass", p["mass"])
        self.Ixx, self.Iyy, self.Izz = bc("Ixx", p["Ixx"]), bc("Iyy", p["Iyy"]), bc("Izz", p["Izz"])
        self.Ixz = bc("Ixz", p["Ixz"])
        self.power_scale = bc("power_scale", 1.0)
        self.cg_shift = bc("cg_shift", 0.0)          # fraction of chord, positive aft
        sc = o.get("aero_scale", {})
        self.scale = np.stack([np.broadcast_to(sc.get(k, 1.0), (n,))          # domain randomization
                               for k in ("CD", "CY", "CL", "Cl", "Cm", "Cn")], 1).astype(float)
        # inertia matrix [[Ixx 0 -Ixz] [0 Iyy 0] [-Ixz 0 Izz]] (UMN convention) and its inverse
        I = np.zeros((n, 3, 3))
        I[:, 0, 0], I[:, 1, 1], I[:, 2, 2] = self.Ixx, self.Iyy, self.Izz
        I[:, 0, 2] = I[:, 2, 0] = -self.Ixz
        self.I, self.Iinv = I, np.linalg.inv(I)

    # --- forces and moments --------------------------------------------------------------------
    def forces(self, Y, thr, gust_b):
        """gust_b (n, 3): unsteady wind (gusts + turbulence) in body axes."""
        p = self.p
        vr = Y[:, U:W + 1] - gust_b
        Va = np.maximum(np.linalg.norm(vr, axis=1), 0.5)
        alpha = np.arctan2(vr[:, 2], vr[:, 0])
        beta = np.arcsin(np.clip(vr[:, 1] / Va, -1, 1))
        rho = isa_density(Y[:, H])
        qbar = 0.5 * rho * Va**2
        V2 = 2 * np.maximum(Va, 0.001)
        phat, qhat, rhat = Y[:, P] * p["b"] / V2, Y[:, Q] * p["c"] / V2, Y[:, R] * p["b"] / V2

        omega = np.clip(Y[:, OM], p["omega_min"], p["omega_max"])
        Rp = p["prop_radius"]
        J = Va * np.pi / (omega * Rp)
        J_aero = J if p["j_mode"] == "actual" else np.ones_like(J)
        if p["aero_model"] == "faser_tables":
            C = self.tables.coeffs(alpha, beta, Y[:, DE], Y[:, DA], Y[:, DR], phat, qhat, rhat, J_aero)
        else:
            C = linear_25e_coeffs(p, alpha, beta, Y[:, DE], Y[:, DA], Y[:, DR], phat, qhat, rhat)
        C = C * self.scale
        ca, sa = np.cos(alpha), np.sin(alpha)
        CD, CY, CL, Cl, Cm, Cn = C.T
        Cm = Cm + self.cg_shift * (CL * ca + CD * sa)        # aft CG: pitch-up
        qS = qbar * p["S"]
        # Physical CY (lateral force opposite to sideslip). The NASA tables follow this convention
        # (CY < 0, Cn > 0, Cl < 0 for beta > 0). The UMN simulator in FASER mode passed CY to the
        # Simulink block that inverts it, without compensation: lateral force of the wrong sign (fixed here).
        Fs = np.stack([-CD, CY, -CL], 1) * qS[:, None]                      # stability axes
        Ms = np.stack([p["b"] * Cl, p["c"] * Cm, p["b"] * Cn], 1) * qS[:, None]
        F_aero = np.stack([ca * Fs[:, 0] - sa * Fs[:, 2], Fs[:, 1], sa * Fs[:, 0] + ca * Fs[:, 2]], 1)
        M_aero = np.stack([ca * Ms[:, 0] - sa * Ms[:, 2], Ms[:, 1], sa * Ms[:, 0] + ca * Ms[:, 2]], 1)

        T = np.polyval(p["prop_CT"], J) * rho * omega**2 * Rp**4 * 4 / np.pi**2
        Q_prop = np.polyval(p["prop_CP"], J) * rho * omega**2 * Rp**5 * 4 / np.pi**3
        P_motor = np.clip(np.polyval(p["motor_power"], np.clip(thr, 0, 1)) * self.power_scale,
                          0.0, p["motor_power_max"])
        Q_motor = P_motor / omega
        Jmp = p["Jmp"]
        M_prop = np.stack([Q_motor, Jmp * omega * Y[:, R], -Jmp * omega * Y[:, Q]], 1)
        return dict(F_aero=F_aero, M_aero=M_aero, T=T, M_prop=M_prop, Q_prop=Q_prop, Q_motor=Q_motor,
                    P_motor=P_motor, P_shaft=Q_prop * omega, Va=Va, alpha=alpha, beta=beta, J=J, C=C,
                    lift=qS * CL, drag=qS * CD, side=qS * CY)

    # --- dynamics -------------------------------------------------------------------------------
    def derivatives(self, Y, cmd, thr, gust_b, wind_ned, actuator=True):
        """cmd (n, 3): de, da, dr; wind_ned (n, 3): mean wind, which only displaces the aircraft."""
        p = self.p
        F = self.forces(Y, thr, gust_b)
        g, m = p["g"], self.mass
        u, v, w = Y[:, U], Y[:, V], Y[:, W]
        pr, qr, rr = Y[:, P], Y[:, Q], Y[:, R]
        phi, th, psi = Y[:, PHI], Y[:, TH], Y[:, PSI]
        sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(th), np.cos(th)
        Fx = F["F_aero"][:, 0] + F["T"]
        Fy, Fz = F["F_aero"][:, 1], F["F_aero"][:, 2]
        dY = np.empty_like(Y)
        dY[:, U] = rr * v - qr * w - g * st + Fx / m
        dY[:, V] = pr * w - rr * u + g * ct * sf + Fy / m
        dY[:, W] = qr * u - pr * v + g * ct * cf + Fz / m
        om = Y[:, P:R + 1]
        M = F["M_aero"] + F["M_prop"]
        Iw = np.einsum("nij,nj->ni", self.I, om)
        dY[:, P:R + 1] = np.einsum("nij,nj->ni", self.Iinv, M - np.cross(om, Iw))
        tt = np.tan(th)
        dY[:, PHI] = pr + (qr * sf + rr * cf) * tt
        dY[:, TH] = qr * cf - rr * sf
        dY[:, PSI] = (qr * sf + rr * cf) / ct
        Ve = np.einsum("nij,nj->ni", dcm_body_to_ned(phi, th, psi), Y[:, U:W + 1]) + wind_ned
        dY[:, XN], dY[:, YE], dY[:, H] = Ve[:, 0], Ve[:, 1], -Ve[:, 2]
        if actuator:
            wbw = 2 * np.pi * p["act_bw_hz"]
            dY[:, DE:DR + 1] = np.clip(wbw * (cmd - Y[:, DE:DR + 1]), -p["act_rate_max"], p["act_rate_max"])
        else:
            dY[:, DE:DR + 1] = 0.0
        dY[:, OM] = (F["Q_motor"] - F["Q_prop"]) / p["Jmp"]
        return dY

    def step(self, Y, cmd, thr, gust_b, wind_ned, dt, actuator=True):
        """One RK4 step (inputs held constant over the step)."""
        p = self.p
        cmd = np.clip(cmd, p["act_min"], p["act_max"])
        f = lambda y: self.derivatives(y, cmd, thr, gust_b, wind_ned, actuator)
        k1 = f(Y)
        k2 = f(Y + 0.5 * dt * k1)
        k3 = f(Y + 0.5 * dt * k2)
        k4 = f(Y + dt * k3)
        Yn = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Yn[:, DE:DR + 1] = np.clip(Yn[:, DE:DR + 1], p["act_min"], p["act_max"])
        Yn[:, OM] = np.clip(Yn[:, OM], p["omega_min"], p["omega_max"])
        return Yn

    # --- trim -----------------------------------------------------------------------------------
    def trim(self, airspeed, gamma=0.0, h=100.0, psi=0.0, iters=60):
        """Steady straight flight, calm air, zero sideslip (like the UMN trim).
        Unknowns: alpha, de, throttle, omega, phi, da, dr; equations: du, dw, dq, domega, dv, dp, dr = 0.
        Vectorized Levenberg-Marquardt; ok = False if the aircraft cannot be trimmed."""
        n = self.n
        Vt, h, psi = (np.broadcast_to(x, (n,)).astype(float) for x in (airspeed, h, psi))
        z = np.zeros(n)
        x = np.stack([np.full(n, 0.05), z, np.full(n, 0.5), np.full(n, 800.0), z, z, z], 1)
        scale = np.array([1, 1, 10, 0.01, 1, 10, 10])   # equation scaling

        def residual(x):
            a, de, thr, om, phi, da, dr = x.T
            Y = np.zeros((n, N_STATE))
            Y[:, U], Y[:, W] = Vt * np.cos(a), Vt * np.sin(a)
            Y[:, PHI], Y[:, TH], Y[:, PSI], Y[:, H] = phi, a + gamma, psi, h
            Y[:, DE], Y[:, DA], Y[:, DR], Y[:, OM] = de, da, dr, om
            d = self.derivatives(Y, Y[:, DE:DR + 1], thr, np.zeros((n, 3)), np.zeros((n, 3)), actuator=False)
            return d[:, [U, W, Q, OM, V, P, R]] * scale, Y

        eps = np.array([1e-5, 1e-5, 1e-5, 1e-2, 1e-5, 1e-5, 1e-5])
        lo = np.array([-0.05, -0.05, -0.1, -100, -0.05, -0.05, -0.05])
        for _ in range(iters):
            r, _ = residual(x)
            Jm = np.empty((n, 7, 7))
            for k in range(7):
                step = np.full(n, eps[k])
                if k == 2:                          # throttle at the upper stop: backward difference
                    step = np.where(x[:, 2] + eps[2] > 1.0, -eps[2], eps[2])
                xk = x.copy()
                xk[:, k] += step
                Jm[:, :, k] = (residual(xk)[0] - r) / step[:, None]
            JT = np.swapaxes(Jm, 1, 2)
            JTJ = JT @ Jm
            A = JTJ + 1e-6 * np.eye(7) * np.diagonal(JTJ, axis1=1, axis2=2)[:, None, :] + 1e-12 * np.eye(7)
            dx = np.linalg.solve(A, -(JT @ r[..., None]))[..., 0]
            x = x + np.clip(dx, lo, -lo)
            x[:, 2] = np.clip(x[:, 2], 0, 1)
        r, Y = residual(x)
        ok = np.abs(r).max(1) < 1e-6
        return Y, x[:, [1, 5, 6]], x[:, 2], ok
