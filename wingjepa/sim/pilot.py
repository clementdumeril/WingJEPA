"""Génération des commandes : autopilote TECS + quatre types de vols (phase1_decisions.md).

Tout est pré-tiré par vol sous forme de profils (n, K) sur la grille de commande (20 Hz) :
consignes h_ref / V_ref, perturbations additives sur la profondeur et le moteur, fenêtres
« boucle ouverte » (autopilote figé) et fenêtres « moteur coupé ».

Types : 0 = stabilisé, 1 = excitation (2-1-1, 3-2-1-1, doublet), 2 = consignes aléatoires suivies
par le TECS, 3 = commandes aléatoires lissées.
"""
import numpy as np

STABLE, EXCITATION, REFERENCES, RANDOM = range(4)
TYPE_NAMES = ["stabilise", "excitation", "consignes", "aleatoire"]

MULTISTEP = {"2-1-1": [(2, 1), (1, -1), (1, 1)],
             "3-2-1-1": [(3, 1), (2, -1), (1, 1), (1, -1)],
             "doublet": [(1, 1), (1, -1)]}


def _windows(rng, K, dt, rate_per_min, dur_range):
    """Masque booléen (K,) de fenêtres aléatoires (processus de Poisson)."""
    m = np.zeros(K, bool)
    if rate_per_min <= 0:
        return m
    t = rng.exponential(60 / rate_per_min)
    T = K * dt
    while t < T:
        d = rng.uniform(*dur_range)
        m[int(t / dt):int(min(t + d, T) / dt)] = True
        t += d + rng.exponential(60 / rate_per_min)
    return m


def make_profiles(rng, n, K, dt, cfg, h0, V0):
    c = cfg
    types = rng.choice(4, size=n, p=np.array(c["type_probs"]) / np.sum(c["type_probs"]))
    h_ref = np.repeat(h0[:, None], K, 1).astype(float)
    V_ref = np.repeat(V0[:, None], K, 1).astype(float)
    de_add, dt_add = np.zeros((n, K)), np.zeros((n, K))
    open_loop, engine_off = np.zeros((n, K), bool), np.zeros((n, K), bool)

    for i in range(n):
        ty = types[i]
        if ty == REFERENCES:
            k = 0
            while k < K:
                L = int(rng.uniform(*c["ref_hold_s"]) / dt)
                h_ref[i, k:] = h0[i] + rng.uniform(-c["ref_dh_max"], c["ref_dh_max"])
                V_ref[i, k:] = rng.uniform(*c["V_range"])
                k += L
        elif ty == EXCITATION:
            k = int(rng.uniform(1, 4) / dt)
            while k < K:
                name = rng.choice(list(MULTISTEP))
                width = rng.uniform(*c["exc_width_s"])
                axis = rng.choice(["e", "t", "et"])
                ae = np.deg2rad(rng.uniform(*c["exc_elev_deg"])) * rng.choice([-1, 1])
                at = rng.uniform(*c["exc_thr"]) * rng.choice([-1, 1])
                k0 = k
                for units, s in MULTISTEP[name]:
                    k1 = min(k0 + int(units * width / dt), K)
                    if "e" in axis:
                        de_add[i, k0:k1] = s * ae
                    if "t" in axis:
                        dt_add[i, k0:k1] = s * at
                    k0 = k1
                k_end = min(k0 + int(c["exc_free_s"] / dt), K)
                open_loop[i, k:k_end] = True          # autopilote figé pendant la manœuvre et après
                k = k_end + int(rng.uniform(*c["exc_gap_s"]) / dt)
        elif ty == RANDOM:
            tau = rng.uniform(*c["rand_tau_s"])
            a = np.exp(-dt / tau)
            se = np.deg2rad(rng.uniform(*c["rand_elev_deg"]))
            st = rng.uniform(*c["rand_thr"])
            xe = xt = 0.0
            for k in range(K):
                xe = a * xe + se * np.sqrt(1 - a**2) * rng.standard_normal()
                xt = a * xt + st * np.sqrt(1 - a**2) * rng.standard_normal()
                de_add[i, k], dt_add[i, k] = xe, xt
            open_loop[i] = _windows(rng, K, dt, c["rand_openloop_per_min"], c["rand_openloop_s"])
        if rng.random() < c["engine_off_prob"]:
            engine_off[i] = _windows(rng, K, dt, c["engine_off_per_min"], c["engine_off_s"])
    return dict(types=types, h_ref=h_ref, V_ref=V_ref, de_add=de_add, dt_add=dt_add,
                open_loop=open_loop, engine_off=engine_off)


class TECS:
    """Total Energy Control System simplifié (vectorisé).

    Moteur  -> énergie totale        : e_T = (gamma_ref - gamma) + (Vdot_ref - Vdot)/g
    Assiette -> répartition          : e_D = (gamma_ref - gamma) - (Vdot_ref - Vdot)/g
    Boucle interne d'assiette        : delta_e = -k_th (theta_cmd - theta) + k_q q
    Moteur coupé : l'assiette tient la vitesse (priorité vitesse), comme en plané.
    Anticipations par vol (thr_ff, theta_ff, de_ff) : valeurs d'équilibre au départ ; les intégrateurs
    corrigent le reste.
    """

    def __init__(self, n, gains, thr_ff=None, theta_ff=None, de_ff=None, g=9.81):
        self.k, self.g = gains, g
        self.thr_ff = np.full(n, gains["thr_0"]) if thr_ff is None else np.asarray(thr_ff, float)
        self.theta_ff = np.full(n, gains["theta_0"]) if theta_ff is None else np.asarray(theta_ff, float)
        self.de_ff = np.zeros(n) if de_ff is None else np.asarray(de_ff, float)
        self.I_T, self.I_D, self.I_G = np.zeros(n), np.zeros(n), np.zeros(n)
        self.Vdot = np.zeros(n)
        self.Va_prev = None

    def __call__(self, Va, gamma, theta, q, h, h_ref, V_ref, engine_off, freeze, dt):
        k, g = self.k, self.g
        if self.Va_prev is None:
            self.Va_prev = Va.copy()
        a = dt / (k["vdot_tau"] + dt)
        self.Vdot += a * ((Va - self.Va_prev) / dt - self.Vdot)
        self.Va_prev = Va.copy()

        gam_ref = np.clip(k["k_h"] * (h_ref - h), -k["gamma_max"], k["gamma_max"])
        vd_ref = np.clip(k["k_V"] * (V_ref - Va), -k["vdot_max"], k["vdot_max"])
        e_T = (gam_ref - gamma) + (vd_ref - self.Vdot) / g
        e_D = (gam_ref - gamma) - (vd_ref - self.Vdot) / g
        e_G = Va - V_ref                                    # plané : trop vite -> cabrer

        live = ~freeze
        on, off = live & ~engine_off, live & engine_off
        self.I_T = np.where(on, np.clip(self.I_T + k["k_TI"] * e_T * dt, -k["I_T_max"], k["I_T_max"]), self.I_T)
        self.I_D = np.where(on, np.clip(self.I_D + k["k_DI"] * e_D * dt, -k["I_D_max"], k["I_D_max"]), self.I_D)
        self.I_G = np.where(off, np.clip(self.I_G + k["k_GI"] * e_G * dt, -k["I_D_max"], k["I_D_max"]), self.I_G)

        thr = np.clip(self.thr_ff + k["k_TP"] * e_T + self.I_T, 0.0, 1.0)
        th_cmd = np.where(engine_off,
                          k["theta_glide_0"] + k["k_GP"] * e_G + self.I_G,
                          self.theta_ff + k["k_DP"] * e_D + self.I_D)
        th_cmd = np.clip(th_cmd, -k["theta_cmd_max"], k["theta_cmd_max"])
        de = self.de_ff - k["k_th"] * (th_cmd - theta) + k["k_q"] * q
        return de, np.where(engine_off, 0.0, thr)
