"""Simulateur de vérité longitudinal de l'Ultra Stick 120 (FASER), vectorisé sur un lot de vols.

Traduction de UMNUAV_sim_v4 (University of Minnesota UAV Lab, BSD) restreinte au plan longitudinal :
  - aéro : tables de soufflerie NASA Langley (FASER_Aero.mat), interpolation linéaire avec
    extrapolation linéaire, comme les blocs PreLookup / Interpolation_n-D de FASER_Aero_Lib.mdl ;
    coefficients [CD, CY, CL, Cl, Cm, Cn] en axes de stabilité, convertis en axes avion ;
    total = base(alpha, beta) + profondeur(alpha, delta_e, J) + puissance(alpha, beta, J) + amortissement ;
  - propulsion (Propulsion_Lib.mdl) : puissance moteur = polynôme du gaz, couple = P / omega,
    hélice C_T(J), C_P(J), J = V pi / (omega R), omega intégré avec l'inertie Jmp ;
  - gouverne (Actuator_Lib.mdl) : 1er ordre w/(s+w) limité en vitesse et en débattement ;
  - mouvement : corps rigide, Terre plate (comme le 6DoF de l'Aerospace Blockset), réduit à u, w, q, theta.
Le modèle linéaire de l'Ultra Stick 25e (Ultrastick_Aero_Lib.mdl) est inclus pour la validation.

État Y (n, 8) : u, w, q, theta, x, h, delta_e (position), omega (hélice, rad/s)
"""
import numpy as np
import yaml
from scipy.interpolate import RegularGridInterpolator

U, W, Q, TH, X, H, DE, OM = range(8)
N_STATE = 8
D2R, R2D = np.pi / 180, 180 / np.pi


def isa_density(h):
    """Atmosphère standard 1976, troposphère."""
    T = 288.15 - 0.0065 * h
    return 1.225 * (T / 288.15) ** 4.2559


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


class FaserTables:
    """Tables NASA de FASER restreintes à beta = 0 (vol longitudinal)."""

    def __init__(self, mat_path):
        from scipy.io import loadmat
        m = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        b = m["C6_bas"]
        j0 = list(np.asarray(b.beta)).index(0)
        a = np.asarray(b.alpha, float)
        lin = dict(method="linear", bounds_error=False, fill_value=None)   # extrapolation linéaire
        self.base = RegularGridInterpolator((a,), b.data[:, j0, :][:, [0, 2, 4]], **lin)   # CD, CL, Cm
        e = m["dC3_ele"]
        ele_data = np.array(e.data, float)
        # Cellule corrompue (la seule, détectée par comparaison aux voisines) : alpha = -5 deg,
        # profondeur = -10 deg, J = 0,389 -> dCD, dCL, dCm = -0.805, 4.352, -3.083 alors que les
        # voisines valent ~0.02, -0.05, 0.13. UMN ne l'atteignait jamais (J fixé à 0,7). On la remplace
        # par l'extrapolation linéaire en alpha depuis alpha = 0 et 5 deg (même profondeur, même J).
        bad = (0, 0, 1)
        if abs(ele_data[bad][1] - 4.352) < 1e-3:
            ele_data[bad] = 2 * ele_data[1, 0, 1] - ele_data[2, 0, 1]
        self.J_ele = (float(np.min(e.J)), float(np.max(e.J)))
        self.ele = RegularGridInterpolator((np.asarray(e.alpha, float), np.asarray(e.ele, float),
                                            np.asarray(e.J, float)), ele_data, **lin)     # dCD, dCL, dCm
        p = m["dC6_pow"]
        jp = list(np.asarray(p.beta)).index(0)
        self.J_pow = (float(np.min(p.J)), float(np.max(p.J)))
        self.pow = RegularGridInterpolator((np.asarray(p.alpha, float), np.asarray(p.J, float)),
                                           p.data[:, jp, :, :][:, :, [0, 2, 4]], **lin)
        d = m["dampder"]
        self.CLq, self.Cmq = float(d.CLq), float(d.Cmq)
        self.alpha_grid = a

    def coeffs(self, alpha, de, qhat, J, scale=None):
        """CD, CL, Cm (axes de stabilité). alpha, de en rad ; J saturé aux plages des tables."""
        ad, ed = alpha * R2D, de * R2D
        Je = np.clip(J, *self.J_ele)
        Jp = np.clip(J, self.J_pow[0] + 2.2e-16, self.J_pow[1] + 2.2e-16)
        c = (self.base(ad[:, None]) + self.ele(np.stack([ad, ed, Je], 1))
             + self.pow(np.stack([ad, Jp], 1)))
        CD, CL, Cm = c[:, 0], c[:, 1] + self.CLq * qhat, c[:, 2] + self.Cmq * qhat
        if scale is not None:   # randomisation de domaine : facteurs multiplicatifs par vol
            CD, CL, Cm = CD * scale["CD"], CL * scale["CL"], Cm * scale["Cm"]
        return CD, CL, Cm


def linear_25e_coeffs(p, alpha, de, qhat):
    """Modèle linéaire de l'Ultra Stick 25e (Ultrastick_Aero_Lib.mdl), volets et latéral à zéro."""
    cl, cd, cm = p["CL"], p["CDw"], p["Cm"]
    CL = cl["zero"] + cl["alpha"] * alpha + cl["delev"] * de + cl["q"] * qhat
    AR = p["b"] ** 2 / p["S"]
    CD = cd["min"] + (CL - cl["minD"]) ** 2 / (np.pi * AR * cd["osw"]) + np.abs(cd["delev"] * de)
    Cm = cm["zero"] + cm["alpha"] * alpha + cm["delev"] * de + cm["q"] * qhat
    return CD, CL, Cm


class UltraStick:
    """Lot de n avions. p : config YAML ; overrides : dict de tableaux (n,) par vol (masse, Iyy, ...)."""

    def __init__(self, p, n, tables=None, overrides=None):
        self.p, self.n, self.tables = p, n, tables
        o = overrides or {}
        self.mass = np.broadcast_to(o.get("mass", p["mass"]), (n,)).astype(float)
        self.Iyy = np.broadcast_to(o.get("Iyy", p["Iyy"]), (n,)).astype(float)
        self.power_scale = np.broadcast_to(o.get("power_scale", 1.0), (n,)).astype(float)
        self.cg_shift = np.broadcast_to(o.get("cg_shift", 0.0), (n,)).astype(float)   # fraction de corde
        self.scale = o.get("aero_scale")   # dict CD / CL / Cm -> (n,)

    # --- forces -------------------------------------------------------------------------------
    def forces(self, Y, thr, ug, wg):
        """Comme UMNUAV_sim_v4 : (u, w) sont relatifs à la masse d'air moyenne ; seul le vent
        instationnaire (rafales + turbulence, axes avion) entre dans l'aérodynamique."""
        p = self.p
        ur, wr = Y[:, U] - ug, Y[:, W] - wg
        Va = np.maximum(np.hypot(ur, wr), 0.5)
        alpha = np.arctan2(wr, ur)
        rho = isa_density(Y[:, H])
        qbar_S = 0.5 * rho * Va**2 * p["S"]
        qhat = Y[:, Q] * p["c"] / (2 * np.maximum(Va, 0.001))

        omega = np.clip(Y[:, OM], p["omega_min"], p["omega_max"])
        R = p["prop_radius"]
        J = Va * np.pi / (omega * R)
        J_aero = J if p["j_mode"] == "actual" else np.ones_like(J)

        if p["aero_model"] == "faser_tables":
            CD, CL, Cm = self.tables.coeffs(alpha, Y[:, DE], qhat, J_aero, self.scale)
        else:
            CD, CL, Cm = linear_25e_coeffs(p, alpha, Y[:, DE], qhat)
        ca, sa = np.cos(alpha), np.sin(alpha)
        Cm = Cm + self.cg_shift * (CL * ca + CD * sa)       # centrage vers l'arrière : cabreur
        Fx = qbar_S * (-CD * ca + CL * sa)
        Fz = qbar_S * (-CD * sa - CL * ca)
        My = qbar_S * p["c"] * Cm

        T = np.polyval(p["prop_CT"], J) * rho * omega**2 * R**4 * 4 / np.pi**2
        Q_prop = np.polyval(p["prop_CP"], J) * rho * omega**2 * R**5 * 4 / np.pi**3
        P_motor = np.clip(np.polyval(p["motor_power"], np.clip(thr, 0, 1)) * self.power_scale,
                          0.0, p["motor_power_max"])
        Q_motor = P_motor / omega
        return dict(Fx=Fx, Fz=Fz, My=My, T=T, Q_prop=Q_prop, Q_motor=Q_motor, P_motor=P_motor,
                    P_shaft=Q_prop * omega, Va=Va, alpha=alpha, J=J, CD=CD, CL=CL, Cm=Cm,
                    lift=qbar_S * CL, drag=qbar_S * CD)

    # --- dynamique ----------------------------------------------------------------------------
    def derivatives(self, Y, de_cmd, thr, ug, wg, wx=0.0, wd=0.0, actuator=True):
        """wx, wd : vent moyen (inertiel), qui ne fait que déplacer l'avion."""
        p = self.p
        F = self.forces(Y, thr, ug, wg)
        u, w, q, th = Y[:, U], Y[:, W], Y[:, Q], Y[:, TH]
        g, m = p["g"], self.mass
        dY = np.empty_like(Y)
        dY[:, U] = -q * w - g * np.sin(th) + (F["Fx"] + F["T"]) / m
        dY[:, W] = q * u + g * np.cos(th) + F["Fz"] / m
        dY[:, Q] = F["My"] / self.Iyy
        dY[:, TH] = q
        dY[:, X] = u * np.cos(th) + w * np.sin(th) + wx      # position sol : + vent moyen
        dY[:, H] = u * np.sin(th) - w * np.cos(th) - wd
        if actuator:
            wbw = 2 * np.pi * p["elev_bw_hz"]
            dY[:, DE] = np.clip(wbw * (de_cmd - Y[:, DE]), -p["elev_rate_max"], p["elev_rate_max"])
        else:
            dY[:, DE] = 0.0
        dY[:, OM] = (F["Q_motor"] - F["Q_prop"]) / p["Jmp"]
        return dY

    def step(self, Y, de_cmd, thr, ug, wg, wx, wd, dt, actuator=True):
        """Un pas RK4 (entrées constantes sur le pas)."""
        p = self.p
        de_cmd = np.clip(de_cmd, p["elev_min"], p["elev_max"])
        f = lambda y: self.derivatives(y, de_cmd, thr, ug, wg, wx, wd, actuator)
        k1 = f(Y)
        k2 = f(Y + 0.5 * dt * k1)
        k3 = f(Y + 0.5 * dt * k2)
        k4 = f(Y + dt * k3)
        Yn = Y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        Yn[:, DE] = np.clip(Yn[:, DE], p["elev_min"], p["elev_max"])
        Yn[:, OM] = np.clip(Yn[:, OM], p["omega_min"], p["omega_max"])
        return Yn

    # --- équilibre ----------------------------------------------------------------------------
    def trim(self, V, gamma=0.0, h=100.0, iters=50):
        """Vol rectiligne stabilisé à la vitesse air V (n,) et pente gamma, air calme.
        Inconnues : alpha, delta_e, gaz, omega (Newton à différences finies, vectorisé)."""
        n = self.n
        V = np.broadcast_to(V, (n,)).astype(float)
        h = np.broadcast_to(h, (n,)).astype(float)
        z = np.zeros(n)
        x = np.stack([np.full(n, 0.05), np.zeros(n), np.full(n, 0.5), np.full(n, 800.0)], 1)

        def residual(x):
            a, de, thr, om = x.T
            th = a + gamma
            Y = np.stack([V * np.cos(a), V * np.sin(a), z, th, z, h, de, om], 1)
            d = self.derivatives(Y, de, thr, z, z, actuator=False)
            return np.stack([d[:, U], d[:, W], d[:, Q] * 10, d[:, OM] / 100], 1), Y

        for _ in range(iters):
            r, _ = residual(x)
            Jm = np.empty((n, 4, 4))
            eps = np.array([1e-5, 1e-5, 1e-5, 1e-2])
            for k in range(4):
                step = np.full(n, eps[k])
                if k == 2:                          # gaz en butée haute : différence vers le bas
                    step = np.where(x[:, 2] + eps[2] > 1.0, -eps[2], eps[2])
                xk = x.copy()
                xk[:, k] += step
                Jm[:, :, k] = (residual(xk)[0] - r) / step[:, None]
            # Levenberg-Marquardt : jamais singulier (un avion impossible à équilibrer reste ok = False)
            JT = np.swapaxes(Jm, 1, 2)
            JTJ = JT @ Jm
            A = JTJ + 1e-6 * np.eye(4) * np.diagonal(JTJ, axis1=1, axis2=2)[:, None, :] + 1e-12 * np.eye(4)
            dx = np.linalg.solve(A, -(JT @ r[..., None]))[..., 0]
            x = x + np.clip(dx, [-0.05, -0.05, -0.1, -100], [0.05, 0.05, 0.1, 100])
            x[:, 2] = np.clip(x[:, 2], 0, 1)
        r, Y = residual(x)
        ok = np.abs(r).max(1) < 1e-6
        return Y, x[:, 1], x[:, 2], ok
