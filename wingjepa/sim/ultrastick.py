"""Simulateur de vérité 6 degrés de liberté de l'Ultra Stick 120 (FASER), vectorisé sur un lot de vols.

Traduction en Python de UMNUAV_sim_v4 (University of Minnesota UAV Lab, BSD, Simulink 2011) :
  - aéro (FASER_Aero_Lib.mdl) : tables de soufflerie NASA Langley (FASER_Aero.mat), interpolation
    linéaire avec extrapolation linéaire (blocs PreLookup / Interpolation_n-D) ; coefficients
    [CD, CY, CL, Cl, Cm, Cn] en axes de stabilité :
      base(alpha, beta) + profondeur(alpha, de, J) -> CD, CL, Cm + direction(alpha, beta, dr, J) -> CY, Cn
      + ailerons(alpha, beta, da) -> Cl + puissance(alpha, beta, J) -> les 6
      + amortissement : CLq, Cmq (q c/2V) ; CYr, Clr, Cnr (r b/2V) ; Clp, Cnp (p b/2V) ;
    puis passage axes de stabilité -> axes avion (bloc « Aerodynamic Forces and Moments ») ;
  - propulsion (Propulsion_Lib.mdl) : puissance moteur = polynôme du gaz, couple = P / omega,
    hélice C_T(J), C_P(J), J = V pi / (omega R) ; moments : [couple moteur, Jmp omega r, -Jmp omega q] ;
  - gouvernes (Actuator_Lib.mdl) : 1er ordre w/(s+w), vitesse et débattement limités ;
  - mouvement : corps rigide 6DoF, angles d'Euler, Terre plate (bloc 6DoF de l'Aerospace Blockset) ;
    vent moyen ajouté aux vitesses inertielles, vent instationnaire aux vitesses avion (comme UMN).
Le modèle linéaire de l'Ultra Stick 25e (Ultrastick_Aero_Lib.mdl) est inclus pour la validation.
Écarts au simulateur UMN : docs/phase1_decisions.md.

État Y (n, 16) : u v w | p q r | phi theta psi | north east h | de da dr (positions) | omega (hélice)
Commandes : de, da, dr (rad) et gaz (0-1). Conventions UMN : profondeur +TED (piqueur),
ailerons da = (da_d - da_g)/2 +TED, direction +TEL.
"""
import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator

(U, V, W, P, Q, R, PHI, TH, PSI, XN, YE, H, DE, DA, DR, OM) = range(16)
N_STATE = 16
D2R, R2D = np.pi / 180, 180 / np.pi


def isa_density(h):
    """Atmosphère standard 1976, troposphère."""
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559


def load_config(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def dcm_body_to_ned(phi, th, psi):
    """Matrices (n, 3, 3) : v_NED = C @ v_avion."""
    cf, sf, ct, st, cp, sp = np.cos(phi), np.sin(phi), np.cos(th), np.sin(th), np.cos(psi), np.sin(psi)
    C = np.empty(phi.shape + (3, 3))
    C[:, 0, 0], C[:, 0, 1], C[:, 0, 2] = ct * cp, sf * st * cp - cf * sp, cf * st * cp + sf * sp
    C[:, 1, 0], C[:, 1, 1], C[:, 1, 2] = ct * sp, sf * st * sp + cf * cp, cf * st * sp - sf * cp
    C[:, 2, 0], C[:, 2, 1], C[:, 2, 2] = -st, sf * ct, cf * ct
    return C


class FaserTables:
    """Tables NASA de FASER (FASER_Aero.mat)."""

    def __init__(self, mat_path):
        from scipy.io import loadmat
        m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        lin = dict(method="linear", bounds_error=False, fill_value=None)   # extrapolation linéaire
        ax = lambda s, f: np.asarray(getattr(s, f), float)
        b = m["C6_bas"]
        self.base = RegularGridInterpolator((ax(b, "alpha"), ax(b, "beta")), b.data, **lin)       # 6
        e = m["dC3_ele"]
        ele = np.array(e.data, float)
        # Cellule corrompue (la seule, détectée par comparaison aux voisines) : alpha = -5 deg,
        # profondeur = -10 deg, J = 0,389 -> dCD, dCL, dCm = -0.805, 4.352, -3.083 alors que les
        # voisines valent ~0.02, -0.05, 0.13. UMN ne l'atteignait jamais (J fixé à 0,7). On la remplace
        # par l'extrapolation linéaire en alpha depuis alpha = 0 et 5 deg (même profondeur, même J).
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
        # saturation de J aux plages des tables (blocs Saturation de FASER_Aero_Lib.mdl)
        self.J_ele = (ax(e, "J").min(), ax(e, "J").max())
        self.J_rud = (ax(r, "J").min(), ax(r, "J").max())
        self.J_pow = (ax(pw, "J").min() + 2.2e-16, ax(pw, "J").max() + 2.2e-16)
        d = m["dampder"]
        self.damp = {k: float(getattr(d, k)) for k in ("CLq", "Cmq", "CYr", "Clp", "Clr", "Cnp", "Cnr")}

    def coeffs(self, alpha, beta, de, da, dr, phat, qhat, rhat, J):
        """[CD, CY, CL, Cl, Cm, Cn] (n, 6), axes de stabilité. Angles en rad."""
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
    """Modèle linéaire de l'Ultra Stick 25e (Ultrastick_Aero_Lib.mdl), volets à zéro."""
    cl, cd, cm, cy, cr, cn = p["CL"], p["CDw"], p["Cm"], p["CYw"], p["Cl"], p["Cn"]
    CL = cl["zero"] + cl["alpha"] * alpha + cl["delev"] * de + cl["q"] * qhat
    AR = p["b"] ** 2 / p["S"]
    CD = (cd["min"] + (CL - cl["minD"]) ** 2 / (np.pi * AR * cd["osw"]) + np.abs(cd["delev"] * de)
          + np.abs(cd["dail"] * da) + np.abs(cd["drud"] * dr))
    # Signe : le bloc Simulink « Aerodynamic Forces and Moments » inverse CY (F_y = -qS CY) et
    # Ultrastick_Aero_Lib compense par un gain -1 ; au total, la force latérale est la valeur physique
    # ci-dessous (vérifié contre doublet_fullmodel.mat : erreurs latérales divisées par 25-30).
    CY = cy["beta"] * beta + cy["dail"] * da + cy["drud"] * dr + cy["p"] * phat + cy["r"] * rhat
    Cl = cr["beta"] * beta + cr["dail"] * da + cr["drud"] * dr + cr["p"] * phat + cr["r"] * rhat
    Cm = cm["zero"] + cm["alpha"] * alpha + cm["delev"] * de + cm["q"] * qhat
    Cn = cn["beta"] * beta + cn["dail"] * da + cn["drud"] * dr + cn["p"] * phat + cn["r"] * rhat
    return np.stack([CD, CY, CL, Cl, Cm, Cn], 1)


class UltraStick:
    """Lot de n avions. p : config YAML ; overrides : dict de tableaux (n,) par vol."""

    def __init__(self, p, n, tables=None, overrides=None):
        self.p, self.n, self.tables = p, n, tables
        o = overrides or {}
        bc = lambda k, d: np.broadcast_to(o.get(k, d), (n,)).astype(float)
        self.mass = bc("mass", p["mass"])
        self.Ixx, self.Iyy, self.Izz = bc("Ixx", p["Ixx"]), bc("Iyy", p["Iyy"]), bc("Izz", p["Izz"])
        self.Ixz = bc("Ixz", p["Ixz"])
        self.power_scale = bc("power_scale", 1.0)
        self.cg_shift = bc("cg_shift", 0.0)          # fraction de corde, positif vers l'arrière
        sc = o.get("aero_scale", {})
        self.scale = np.stack([np.broadcast_to(sc.get(k, 1.0), (n,))          # randomisation de domaine
                               for k in ("CD", "CY", "CL", "Cl", "Cm", "Cn")], 1).astype(float)
        # matrice d'inertie [[Ixx 0 -Ixz] [0 Iyy 0] [-Ixz 0 Izz]] (convention UMN) et son inverse
        I = np.zeros((n, 3, 3))
        I[:, 0, 0], I[:, 1, 1], I[:, 2, 2] = self.Ixx, self.Iyy, self.Izz
        I[:, 0, 2] = I[:, 2, 0] = -self.Ixz
        self.I, self.Iinv = I, np.linalg.inv(I)

    # --- forces et moments ---------------------------------------------------------------------
    def forces(self, Y, thr, gust_b):
        """gust_b (n, 3) : vent instationnaire (rafales + turbulence) en axes avion."""
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
        Cm = Cm + self.cg_shift * (CL * ca + CD * sa)        # centrage vers l'arrière : cabreur
        qS = qbar * p["S"]
        # CY physique (force latérale opposée au dérapage). Les tables NASA suivent cette convention
        # (CY < 0, Cn > 0, Cl < 0 pour beta > 0). Le simulateur UMN en mode FASER passait CY au bloc
        # Simulink qui l'inverse, sans compensation : force latérale de mauvais signe (corrigé ici).
        Fs = np.stack([-CD, CY, -CL], 1) * qS[:, None]                      # axes de stabilité
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

    # --- dynamique ------------------------------------------------------------------------------
    def derivatives(self, Y, cmd, thr, gust_b, wind_ned, actuator=True):
        """cmd (n, 3) : de, da, dr ; wind_ned (n, 3) : vent moyen, qui ne fait que déplacer l'avion."""
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
        """Un pas RK4 (entrées constantes sur le pas)."""
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

    # --- équilibre ------------------------------------------------------------------------------
    def trim(self, airspeed, gamma=0.0, h=100.0, psi=0.0, iters=60):
        """Vol rectiligne stabilisé, air calme, dérapage nul (comme le trim UMN).
        Inconnues : alpha, de, gaz, omega, phi, da, dr ; équations : du, dw, dq, domega, dv, dp, dr = 0.
        Levenberg-Marquardt vectorisé ; ok = False si l'avion ne peut pas être équilibré."""
        n = self.n
        Vt, h, psi = (np.broadcast_to(x, (n,)).astype(float) for x in (airspeed, h, psi))
        z = np.zeros(n)
        x = np.stack([np.full(n, 0.05), z, np.full(n, 0.5), np.full(n, 800.0), z, z, z], 1)
        scale = np.array([1, 1, 10, 0.01, 1, 10, 10])   # mise à l'échelle des équations

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
                if k == 2:                          # gaz en butée haute : différence vers le bas
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
