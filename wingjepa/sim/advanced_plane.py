"""Ground-truth 6-DoF simulator of PX4's advanced_plane, as Gazebo simulates it, vectorized over a batch of flights.

Python port of Fixed-Wing/cpp/include/fixed_wing/dynamics.h (namespace advanced, advanced_plane(),
advanced_aerodynamics(), derivative(), rk4()), with the interface and state layout of UltraStick so that it is a
drop-in replacement in the generator, the trim and the tests:
  - aero: gz-sim AdvancedLiftDrag with the coefficients of models/advanced_plane/model.sdf: attached-flow polar
    blended into a flat plate past the stall (sigma), rate terms in 0.5 rho V, control derivatives per joint degree;
    evaluated with the air velocity at the centre of pressure (cp, 0, 0), speed and angle of attack in the
    lift-drag plane only (V = sqrt(u^2 + w^2), beta = atan2(v, u)), force applied at the centre of pressure;
  - propeller: MulticopterMotorModel (velocity motor): thrust k omega^2 along +x, reaction torque
    moment_constant * thrust about +x, rotor drag and rolling moment against the air velocity across the axis,
    the rotor 0.3 m ahead of the centre of mass; omega follows 10 + 1590 throttle with a first-order lag of
    12.5 ms up / 25 ms down;
  - servos: first order, 0.1 s, +-45 deg, no rate limit;
  - motion: 6-DoF rigid body, Euler angles, flat Earth, constant air density 1.2041 (Gazebo has no atmosphere
    model), g = 9.8 (Gazebo's default world).

State Y (n, 16): u v w | p q r | phi theta psi | north east h | de da dr (positions) | omega (rotor, rad/s)
Inputs: de, da, dr (rad) and throttle (0-1), as UltraStick.

Frames and signs (the C++ uses body FLU = x forward, y left, z up and a z-up world; here body FRD and NED):
  FLU -> FRD and z-up world -> NED are both the rotation by pi about x, diag(1, -1, -1), a proper rotation, so
  vectors AND pseudo-vectors (angular velocity, moments) map the same way: (x, y, z) -> (x, -y, -z).
  - world: north = x_w, east = -y_w, h = z_w; wind (NED) = (wx, -wy, -wz);
  - body velocity, rates, forces, moments: second and third components change sign; the inertia tensor is
    diagonal (the xz product would change sign twice: unchanged);
  - Euler angles: C++ euler_quaternion(roll, pitch, yaw) = qz(yaw) qy(-pitch) qx(roll) gives phi = roll,
    theta = pitch, psi = -yaw (the C++ yaw is positive nose left);
  - the C++ already evaluates the AdvancedLiftDrag in FRD internally (u, v = -(...), w = -(...), P, Q = -w_y,
    R = -w_z) and converts back with force[1] = -f[1], force[2] = -f[2]: those two flips cancel here;
  - velocities: the C++ integrates the ground velocity in the world frame; here u, v, w are relative to the mean
    wind (which only displaces the aircraft) and gust_b is subtracted for the aerodynamics, as in UltraStick.
    For a constant wind both give the same motion.
Control surfaces, WingJEPA / UMN conventions (as UltraStick and pilot.py, so the same autopilot gains apply):
  - de > 0: elevator trailing edge down, nose down. C++ elevator = -de (its Cm_ctrl > 0: positive = nose up);
  - da > 0: rolls left. C++ aileron = +da (servo_0 = +da, servo_1 = -da; its Cl per aileron degree is
    -0.006662: positive already rolls left). The elevons only act as ailerons (their CL and Cm cancel);
  - dr > 0: trailing edge left, yaws left. C++ rudder = -dr (its Cn_ctrl > 0: positive = yaw right).
  JOINTS below holds this map.

Not in the C++ model (zeros or Gazebo-consistent placeholders in forces(), see its docstring): advance ratio J,
electrical motor power.
"""
import numpy as np

from wingjepa.sim.ultrastick import (DA, DE, DR, H, N_STATE, OM, P, PHI, PSI, Q, R, TH, U, V, W, XN, YE,  # noqa: F401
                                     dcm_body_to_ned, isa_density, load_config)

D2R, R2D = np.pi / 180, 180 / np.pi
COEFS = ("CD", "CY", "CL", "Cl", "Cm", "Cn")          # order of forces()["C"] and of the control matrix rows
# joint angle (rad) of [servo_0 left elevon, servo_1 right elevon, servo_2 elevator, servo_3 rudder]
# = JOINTS @ [de, da, dr] (WingJEPA conventions -> C++ / Gazebo joint conventions)
JOINTS = np.array([[0.0, 1.0, 0.0],
                   [0.0, -1.0, 0.0],
                   [-1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])
# randomized coefficient groups (aero_scale keys): which nominal coefficients each factor multiplies
AERO_GROUPS = {
    "CL": ("CL0", "CLa", "CLq"),
    "CD": ("CD0", "induced", "CDq"),                   # induced: the factor 1 / (pi AR eff)
    "CY": ("CYb", "CYp", "CYr"),
    "Cl": ("Clb", "Clp", "Clr"),
    "Cm": ("Cm0", "Cma"),
    "Cmq": ("Cmq",),
    "Cn": ("Cnb", "Cnp", "Cnr"),
    "de": ("column de",), "da": ("column da",), "dr": ("column dr",),   # control effectiveness
}
RATE = ("CDp", "CYp", "CLp", "Clp", "Cmp", "Cnp", "CDq", "CYq", "CLq", "Clq", "Cmq", "Cnq",
        "CDr", "CYr", "CLr", "Clr", "Cmr", "Cnr")


def control_matrix(p):
    """(6, 3): increments of [CD, CY, CL, Cl, Cm, Cn] per rad of [de, da, dr] (WingJEPA conventions)."""
    ctrl = np.array([p["CD_ctrl"], p["CY_ctrl"], p["CL_ctrl"], p["Cl_ctrl"], p["Cm_ctrl"], p["Cn_ctrl"]], float)
    return ctrl @ JOINTS * R2D                          # per degree of joint -> per rad of WingJEPA input


def stall_blend(alpha, M, alpha_stall):
    """AdvancedLiftDrag sigma: 0 in attached flow, 1 past the stall (about 0.5 at +-alpha_stall)."""
    with np.errstate(over="ignore"):
        up = np.exp(-M * (alpha - alpha_stall))
        down = np.exp(M * (alpha + alpha_stall))
        s = (1 + up + down) / ((1 + up) * (1 + down))
    # far past the stall one exponential overflows (inf / inf): the limit is 1
    return np.where(np.isfinite(s), s, 1.0)


def _factor(rng, m, r, test):
    """Training: 1 + U(-r0, r0); test aircraft: 1 +- U(r0, r1) (as scripts/generate_flights.py)."""
    if not test:
        return 1 + rng.uniform(-r[0], r[0], m)
    return 1 + rng.choice([-1, 1], m) * rng.uniform(r[0], r[1], m)


def sample_overrides(rng, n, p, ranges=None, is_test=None):
    """Per-flight domain randomization of the advanced_plane (multiplicative factors; offset for the CG).
    ranges: the 'randomization' section of configs/advanced_plane.yaml (default: p['randomization']);
    is_test (n,) bool: test aircraft drawn outside the training range, in [r0, r1]."""
    ranges = ranges or p["randomization"]
    is_test = np.zeros(n, bool) if is_test is None else np.asarray(is_test, bool)
    out = {k: np.empty(n) for k in ("mass", "fIxx", "fIyy", "fIzz", "motor_k", "cg_shift", "servo_tau")}
    aero = {k: np.empty(n) for k in ranges["aero"]}
    for test in (False, True):
        sel = is_test == test
        m = int(sel.sum())
        out["mass"][sel] = p["mass"] * _factor(rng, m, ranges["mass"], test)
        for k in ("fIxx", "fIyy", "fIzz"):
            out[k][sel] = _factor(rng, m, ranges["inertia"], test)
        out["motor_k"][sel] = _factor(rng, m, ranges["motor_k"], test)
        out["cg_shift"][sel] = _factor(rng, m, ranges["cg_shift"], test) - 1
        out["servo_tau"][sel] = p["servo_tau"] * _factor(rng, m, ranges["servo_tau"], test)
        for k, r in ranges["aero"].items():
            aero[k][sel] = _factor(rng, m, r, test)
    # Ixz (zero: diagonal tensor) and power_scale (= motor_k) are written too, so that the generator's metadata and
    # the readers of the UltraStick keys (wingjepa/data/windows.py aircraft_params, oracle columns) work unchanged
    return dict(mass=out["mass"], Ixx=p["Ixx"] * out["fIxx"], Iyy=p["Iyy"] * out["fIyy"],
                Izz=p["Izz"] * out["fIzz"], Ixz=np.full(n, float(p.get("Ixz", 0.0))), motor_k=out["motor_k"],
                power_scale=out["motor_k"], cg_shift=out["cg_shift"], servo_tau=out["servo_tau"], aero_scale=aero)


class AdvancedPlane:
    """Batch of n advanced_planes. p: YAML config (configs/advanced_plane.yaml); tables: unused (UltraStick
    interface); overrides: dict of per-flight (n,) arrays:
      mass, Ixx, Iyy, Izz, Ixz (kg, kg m2, absolute values);
      motor_k: factor on the motor constant (thrust and its torque); power_scale is accepted as an alias;
      cp: centre-of-pressure position (m, body x, absolute) or cg_shift: CG shift as a fraction of the chord,
          positive aft (cp = cp0 + cg_shift c, the analog of UltraStick's cg_shift);
      servo_tau: servo time constant (s, absolute);
      aero_scale: dict of factors per group of AERO_GROUPS (CL, CD, CY, Cl, Cm, Cmq, Cn, de, da, dr)."""

    def __init__(self, p, n, tables=None, overrides=None):
        self.p, self.n = p, n
        o = overrides or {}
        bc = lambda k, d: np.broadcast_to(o.get(k, d), (n,)).astype(float)
        self.mass = bc("mass", p["mass"])
        self.Ixx, self.Iyy, self.Izz = bc("Ixx", p["Ixx"]), bc("Iyy", p["Iyy"]), bc("Izz", p["Izz"])
        self.Ixz = bc("Ixz", p.get("Ixz", 0.0))
        I = np.zeros((n, 3, 3))
        I[:, 0, 0], I[:, 1, 1], I[:, 2, 2] = self.Ixx, self.Iyy, self.Izz
        I[:, 0, 2] = I[:, 2, 0] = -self.Ixz
        self.I, self.Iinv = I, np.linalg.inv(I)

        self.S, self.c, self.AR = float(p["S"]), float(p["c"]), float(p["AR"])
        self.b = float(np.sqrt(self.S * self.AR))       # as the C++: wing_span = sqrt(area AR)
        self.motor_k = p["motor_constant"] * bc("motor_k", o.get("power_scale", 1.0))
        self.cg_shift = bc("cg_shift", 0.0)
        self.cp = bc("cp", p["cp"] + self.cg_shift * self.c) if "cp" in o else p["cp"] + self.cg_shift * self.c
        self.servo_tau = bc("servo_tau", p["servo_tau"])

        sc = o.get("aero_scale", {})
        unknown = set(sc) - set(AERO_GROUPS)
        if unknown:
            raise KeyError(f"unknown aero_scale groups {sorted(unknown)}; known: {list(AERO_GROUPS)}")
        self.scale = {g: np.broadcast_to(sc.get(g, 1.0), (n,)).astype(float) for g in AERO_GROUPS}
        k = {name: np.full(n, float(p[name])) for name in
             ("CL0", "CLa", "CD0", "Cm0", "Cma", "CYb", "Clb", "Cnb") + RATE}
        k["induced"] = np.full(n, 1 / (np.pi * self.AR * p["eff"]))
        for g, names in AERO_GROUPS.items():
            for name in names:
                if name in k:
                    k[name] = k[name] * self.scale[g]
        self.k = k
        G = control_matrix(p)                            # (6, 3)
        cols = np.stack([self.scale["de"], self.scale["da"], self.scale["dr"]], 1)   # (n, 3)
        self.G = G[None] * cols[:, None, :]              # (n, 6, 3)
        self.CD_fp = 2 / (1 + np.exp(p["CD_fp_k1"] + p["CD_fp_k2"] * max(self.AR, 1 / self.AR)))

    def rho(self, h):
        if self.p.get("density_model", "constant") == "isa":
            return isa_density(h)
        return np.full(np.shape(h), float(self.p["rho"]))

    # --- forces and moments --------------------------------------------------------------------
    def forces(self, Y, thr, gust_b):
        """Forces and moments in body FRD axes about the centre of mass. gust_b (n, 3): unsteady wind in body axes.
        thr is not used (the thrust follows the lagged rotor speed Y[:, OM]); kept for the UltraStick interface.

        Keys (same as UltraStick, plus F_prop and the AdvancedLiftDrag internals):
          F_aero, M_aero (n, 3): AdvancedLiftDrag force and moment (moment about the CG, the lever arm of the
              centre of pressure included);
          T: thrust along +x; F_prop (n, 3) = [T, rotor drag y, rotor drag z]; M_prop (n, 3): reaction torque
              moment_constant T about +x, rolling moment and rotor-drag lever arm;
          Q_prop = Q_motor = moment_constant T (N m: velocity-controlled motor, the motor torque balances the
              propeller's), P_shaft = P_motor = Q_prop omega. Gazebo-consistent but not a physical power (its
              moment_constant gives a propulsive efficiency above 1); no electrical model;
          Va, alpha, beta: air data at the CG (beta = asin(v / Va)), Va floored at 0.5 m/s as UltraStick;
          V_ld, alpha_cp, beta_cp: speed in the lift-drag plane and angles at the centre of pressure, as
              AdvancedLiftDrag uses them;
          lift, drag, side (N, including the rate terms); C (n, 6) = [CD, CY, CL, Cl, Cm, Cn] = forces / (qbar S),
              moments about the CG / (qbar S [b, c, b]), qbar = 0.5 rho V_ld^2 (zeros when V_ld = 0);
          sigma: stall blend; J: zeros (MulticopterMotorModel has no advance ratio)."""
        k, n = self.k, Y.shape[0]
        vr = Y[:, U:W + 1] - gust_b                    # air-relative velocity at the CG, FRD
        pr, qr, rr = Y[:, P], Y[:, Q], Y[:, R]
        Va = np.linalg.norm(vr, axis=1)
        Va_out = np.maximum(Va, 0.5)
        alpha_cg = np.arctan2(vr[:, 2], vr[:, 0])
        beta_cg = np.arcsin(np.clip(vr[:, 1] / Va_out, -1, 1))
        rho = self.rho(Y[:, H])

        # AdvancedLiftDrag at the centre of pressure r = (cp, 0, 0): v + omega x r = (u, v + r cp, w - q cp)
        cp = self.cp
        u, v, w = vr[:, 0], vr[:, 1] + rr * cp, vr[:, 2] - qr * cp
        V = np.sqrt(u**2 + w**2)                        # the sideslip component does not count
        live = V > 1e-9
        alpha, beta = np.arctan2(w, u), np.arctan2(v, u)
        sa, ca = np.sin(alpha), np.cos(alpha)
        dyn, hrv = 0.5 * rho * V**2, 0.5 * rho * V
        sig = stall_blend(alpha, self.p["M"], self.p["alpha_stall"])
        dC = np.einsum("nij,nj->ni", self.G, Y[:, DE:DR + 1])    # control increments [CD, CY, CL, Cl, Cm, Cn]
        b, c, S = self.b, self.c, self.S
        ph, qh, rh = pr * b / 2 * hrv, qr * c / 2 * hrv, rr * b / 2 * hrv   # rate terms, times 0.5 rho V
        rate = lambda C: k[C + "p"] * ph + k[C + "q"] * qh + k[C + "r"] * rh

        sign = np.where(alpha >= 0, 1.0, -1.0)
        CL = (1 - sig) * (k["CL0"] + k["CLa"] * alpha) + sig * (2 * sign * sa**2 * ca) + dC[:, 2]
        lift = (CL * dyn + rate("CL")) * S
        CD = ((1 - sig) * (k["CD0"] + CL**2 * k["induced"])
              + sig * np.abs(self.CD_fp * (0.5 - 0.5 * np.cos(2 * alpha))) + dC[:, 0])
        drag = (CD * dyn + rate("CD")) * S
        CY = k["CYb"] * beta + dC[:, 1]
        side = (CY * dyn + rate("CY")) * S
        a_s, Cma_s = self.p["alpha_stall"], self.p["Cma_stall"]
        Cm = np.where(alpha > a_s, k["Cm0"] + k["Cma"] * a_s + Cma_s * (alpha - a_s),
                      np.where(alpha < -a_s, k["Cm0"] - k["Cma"] * a_s + Cma_s * (alpha + a_s),
                               k["Cm0"] + k["Cma"] * alpha)) + dC[:, 4]
        Cl = k["Clb"] * beta + dC[:, 3]
        Cn = k["Cnb"] * beta + dC[:, 5]
        # lift along -z of the stability axes, drag along -x (same expressions as UltraStick)
        F_aero = np.stack([lift * sa - drag * ca, side, -lift * ca - drag * sa], 1)
        M_aero = np.stack([(Cl * dyn + rate("Cl")) * S * b,
                           (Cm * dyn + rate("Cm")) * S * c,
                           (Cn * dyn + rate("Cn")) * S * b], 1)
        # force applied at the centre of pressure: + r x F = (0, -cp Fz, cp Fy)
        M_aero[:, 1] += -cp * F_aero[:, 2]
        M_aero[:, 2] += cp * F_aero[:, 1]
        F_aero = np.where(live[:, None], F_aero, 0.0)
        M_aero = np.where(live[:, None], M_aero, 0.0)

        # propeller (MulticopterMotorModel), rotor at (rotor_x, 0, 0)
        pp = self.p
        omega = Y[:, OM]
        T = self.motor_k * omega**2
        rx = pp["rotor_x"]
        ay, az = vr[:, 1] + rr * rx, vr[:, 2] - qr * rx          # air velocity of the rotor across its axis
        Fry, Frz = -omega * pp["rotor_drag"] * ay, -omega * pp["rotor_drag"] * az
        F_prop = np.stack([T, Fry, Frz], 1)
        Q_prop = pp["moment_constant"] * T
        M_prop = np.stack([Q_prop,
                           -omega * pp["rolling_moment"] * ay - rx * Frz,
                           -omega * pp["rolling_moment"] * az + rx * Fry], 1)

        qS = np.where(live, dyn * S, np.inf)
        C = np.stack([drag / qS, side / qS, lift / qS, M_aero[:, 0] / (qS * b), M_aero[:, 1] / (qS * c),
                      M_aero[:, 2] / (qS * b)], 1)
        lift, drag, side = (np.where(live, x, 0.0) for x in (lift, drag, side))
        return dict(F_aero=F_aero, M_aero=M_aero, T=T, F_prop=F_prop, M_prop=M_prop, Q_prop=Q_prop,
                    Q_motor=Q_prop, P_motor=Q_prop * omega, P_shaft=Q_prop * omega, Va=Va_out, alpha=alpha_cg,
                    beta=beta_cg, V_ld=V, alpha_cp=alpha, beta_cp=beta, sigma=sig, J=np.zeros(n), C=C,
                    lift=lift, drag=drag, side=side)

    # --- dynamics -------------------------------------------------------------------------------
    def omega_command(self, thr):
        p = self.p
        return p["omega_min"] + (p["omega_max"] - p["omega_min"]) * np.clip(thr, 0.0, 1.0)

    def derivatives(self, Y, cmd, thr, gust_b, wind_ned, actuator=True):
        """cmd (n, 3): de, da, dr; thr (n,): throttle; wind_ned (n, 3): mean wind, which only displaces the aircraft."""
        p = self.p
        F = self.forces(Y, thr, gust_b)
        g, m = p["g"], self.mass
        u, v, w = Y[:, U], Y[:, V], Y[:, W]
        pr, qr, rr = Y[:, P], Y[:, Q], Y[:, R]
        phi, th, psi = Y[:, PHI], Y[:, TH], Y[:, PSI]
        sf, cf, st, ct = np.sin(phi), np.cos(phi), np.sin(th), np.cos(th)
        Fb = F["F_aero"] + F["F_prop"]
        dY = np.empty_like(Y)
        dY[:, U] = rr * v - qr * w - g * st + Fb[:, 0] / m
        dY[:, V] = pr * w - rr * u + g * ct * sf + Fb[:, 1] / m
        dY[:, W] = qr * u - pr * v + g * ct * cf + Fb[:, 2] / m
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
            dY[:, DE:DR + 1] = (cmd - Y[:, DE:DR + 1]) / self.servo_tau[:, None]
        else:
            dY[:, DE:DR + 1] = 0.0
        # rotor speed: first order toward 10 + 1590 throttle, faster up than down (as the C++: down when the
        # command is below the state)
        om_cmd = self.omega_command(thr)
        tau = np.where(om_cmd < Y[:, OM], p["motor_tau_down"], p["motor_tau_up"])
        dY[:, OM] = (om_cmd - Y[:, OM]) / tau
        return dY

    def step(self, Y, cmd, thr, gust_b, wind_ned, dt, actuator=True):
        """One RK4 step (inputs held constant over the step); surfaces and rotor speed clipped after it, as the
        C++ post_integration."""
        p = self.p
        cmd = np.clip(cmd, p["act_min"], p["act_max"])
        thr = np.clip(thr, 0.0, 1.0)
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
        """Steady straight flight, calm air, zero sideslip (same as UltraStick.trim).
        Unknowns: alpha, de, throttle, omega, phi, da, dr; equations: du, dw, dq, domega, dv, dp, dr = 0.
        Vectorized Levenberg-Marquardt. Returns Y (n, 16), [de, da, dr] (n, 3), throttle (n,), ok (n,)."""
        n = self.n
        Vt, h, psi = (np.broadcast_to(x, (n,)).astype(float) for x in (airspeed, h, psi))
        z = np.zeros(n)
        x = np.stack([np.full(n, 0.05), z, np.full(n, 0.25), np.full(n, 400.0), z, z, z], 1)
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
            for j in range(7):
                step = np.full(n, eps[j])
                if j == 2:                          # throttle at the upper stop: backward difference
                    step = np.where(x[:, 2] + eps[2] > 1.0, -eps[2], eps[2])
                xk = x.copy()
                xk[:, j] += step
                Jm[:, :, j] = (residual(xk)[0] - r) / step[:, None]
            JT = np.swapaxes(Jm, 1, 2)
            JTJ = JT @ Jm
            A = JTJ + 1e-6 * np.eye(7) * np.diagonal(JTJ, axis1=1, axis2=2)[:, None, :] + 1e-12 * np.eye(7)
            dx = np.linalg.solve(A, -(JT @ r[..., None]))[..., 0]
            x = x + np.clip(dx, lo, -lo)
            x[:, 2] = np.clip(x[:, 2], 0, 1)
        r, Y = residual(x)
        ok = np.abs(r).max(1) < 1e-6
        ok &= np.all(np.abs(x[:, [1, 5, 6]]) <= np.asarray(self.p["act_max"]), axis=1)   # within the servo throws
        return Y, x[:, [1, 5, 6]], x[:, 2], ok


def linear_modes(ac, Y, cmd, thr, eps=1e-6):
    """Eigenvalues of the rigid-body motion linearized about Y (surfaces and rotor speed held: they are inputs here).
    Returns dict(long=eigvals of [u, w, q, theta], lat=eigvals of [v, p, r, phi], A=9x9 Jacobian on u..psi)
    for the first flight of the batch."""
    idx = [U, V, W, P, Q, R, PHI, TH, PSI]
    Y0 = Y[:1].copy()
    zero = np.zeros((1, 3))
    f0 = ac.derivatives(Y0, cmd[:1], thr[:1], zero, zero, actuator=False)
    A = np.empty((9, 9))
    for j, sj in enumerate(idx):
        Yp, Ym = Y0.copy(), Y0.copy()
        Yp[0, sj] += eps
        Ym[0, sj] -= eps
        fp = ac.derivatives(Yp, cmd[:1], thr[:1], zero, zero, actuator=False)
        fm = ac.derivatives(Ym, cmd[:1], thr[:1], zero, zero, actuator=False)
        A[:, j] = (fp[0, idx] - fm[0, idx]) / (2 * eps)
    del f0
    lon = [0, 2, 4, 7]                                  # u, w, q, theta
    lat = [1, 3, 5, 6]                                  # v, p, r, phi
    return dict(A=A, long=np.linalg.eigvals(A[np.ix_(lon, lon)]), lat=np.linalg.eigvals(A[np.ix_(lat, lat)]),
                full=np.linalg.eigvals(A[:8, :8]))
