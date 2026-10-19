"""Génération des commandes (3D) : autopilote TECS + pilote latéral + protection d'enveloppe.

Tout est pré-tiré par vol sous forme de profils (n, K) sur la grille de commande (20 Hz) :
  - longitudinal : consignes h_ref / V_ref, perturbations additives profondeur / gaz,
    fenêtres « boucle ouverte » (autopilote figé) et « moteur coupé » ;
  - latéral : consigne d'inclinaison phi_ref (ligne droite, virages et cercles, virages en S),
    perturbations additives ailerons / direction.
Types de vols : 0 = stabilisé, 1 = excitation (2-1-1, 3-2-1-1, doublets, sur les 4 commandes),
2 = consignes aléatoires suivies par le TECS, 3 = commandes aléatoires lissées.
La protection d'enveloppe (dans le générateur) annule manœuvres et perturbations près du décrochage.
"""
import numpy as np

STABLE, EXCITATION, REFERENCES, RANDOM = range(4)
TYPE_NAMES = ["stabilise", "excitation", "consignes", "aleatoire"]
LAT_NAMES = ["droit", "virages", "virages_S"]

MULTISTEP = {"2-1-1": [(2, 1), (1, -1), (1, 1)],
             "3-2-1-1": [(3, 1), (2, -1), (1, 1), (1, -1)],
             "doublet": [(1, 1), (1, -1)]}
EXC_AXES = ["e", "t", "et", "a", "r", "ar"]   # profondeur, gaz, ailerons, direction


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


def _ou(rng, K, dt, tau, sigma):
    a = np.exp(-dt / tau)
    x, out = 0.0, np.empty(K)
    for k in range(K):
        x = a * x + sigma * np.sqrt(1 - a**2) * rng.standard_normal()
        out[k] = x
    return out


def make_profiles(rng, n, K, dt, cfg, h0, V0):
    c, lc = cfg, cfg["lateral"]
    types = rng.choice(4, size=n, p=np.array(c["type_probs"]) / np.sum(c["type_probs"]))
    lat = rng.choice(3, size=n, p=np.array(lc["mode_probs"]) / np.sum(lc["mode_probs"]))
    h_ref = np.repeat(h0[:, None], K, 1).astype(float)
    V_ref = np.repeat(V0[:, None], K, 1).astype(float)
    phi_ref = np.zeros((n, K))
    add = {k: np.zeros((n, K)) for k in ("de", "dt", "da", "dr")}
    open_loop, engine_off = np.zeros((n, K), bool), np.zeros((n, K), bool)
    tgrid = np.arange(K) * dt

    for i in range(n):
        # --- latéral : consigne d'inclinaison
        if lat[i] == 1:                                   # virages et cercles, alternés avec des lignes droites
            k = 0
            while k < K:
                L = int(rng.uniform(*lc["turn_hold_s"]) / dt)
                if rng.random() < 0.6:
                    phi_ref[i, k:k + L] = np.deg2rad(rng.uniform(*lc["bank_deg"])) * rng.choice([-1, 1])
                k += L
        elif lat[i] == 2:                                 # virages en S
            amp = np.deg2rad(rng.uniform(*lc["s_bank_deg"]))
            per = rng.uniform(*lc["s_period_s"])
            phi_ref[i] = amp * np.sin(2 * np.pi * tgrid / per + rng.uniform(0, 2 * np.pi))
        # --- longitudinal et perturbations selon le type
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
            amp = {"e": np.deg2rad(rng.uniform(*c["exc_elev_deg"])), "t": rng.uniform(*c["exc_thr"]),
                   "a": np.deg2rad(rng.uniform(*c["exc_ail_deg"])), "r": np.deg2rad(rng.uniform(*c["exc_rud_deg"]))}
            while k < K:
                name = rng.choice(list(MULTISTEP))
                width = rng.uniform(*c["exc_width_s"])
                axes = rng.choice(EXC_AXES)
                sgn = {a: rng.choice([-1, 1]) for a in "etar"}
                k0 = k
                for units, s in MULTISTEP[name]:
                    k1 = min(k0 + int(units * width / dt), K)
                    for a in axes:
                        add[{"e": "de", "t": "dt", "a": "da", "r": "dr"}[a]][i, k0:k1] = s * sgn[a] * amp[a]
                    k0 = k1
                k_end = min(k0 + int(c["exc_free_s"] / dt), K)
                open_loop[i, k:k_end] = True              # autopilote figé pendant la manœuvre et après
                k = k_end + int(rng.uniform(*c["exc_gap_s"]) / dt)
        elif ty == RANDOM:
            tau = rng.uniform(*c["rand_tau_s"])
            add["de"][i] = _ou(rng, K, dt, tau, np.deg2rad(rng.uniform(*c["rand_elev_deg"])))
            add["dt"][i] = _ou(rng, K, dt, tau, rng.uniform(*c["rand_thr"]))
            add["da"][i] = _ou(rng, K, dt, tau, np.deg2rad(rng.uniform(*c["rand_ail_deg"])))
            add["dr"][i] = _ou(rng, K, dt, tau, np.deg2rad(rng.uniform(*c["rand_rud_deg"])))
            open_loop[i] = _windows(rng, K, dt, c["rand_openloop_per_min"], c["rand_openloop_s"])
        if rng.random() < c["engine_off_prob"]:
            engine_off[i] = _windows(rng, K, dt, c["engine_off_per_min"], c["engine_off_s"])
    return dict(types=types, lat=lat, h_ref=h_ref, V_ref=V_ref, phi_ref=phi_ref, open_loop=open_loop,
                engine_off=engine_off, **{f"{k}_add": v for k, v in add.items()})


class TECS:
    """Total Energy Control System simplifié (vectorisé).

    Moteur  -> énergie totale        : e_T = (gamma_ref - gamma) + (Vdot_ref - Vdot)/g
    Assiette -> répartition          : e_D = (gamma_ref - gamma) - (Vdot_ref - Vdot)/g
    Boucle interne d'assiette        : de = de_ff - k_th (theta_cmd - theta) + k_q (q - q_ref)
      (q_ref : vitesse de tangage d'un virage coordonné)
    Moteur coupé : l'assiette tient la vitesse (priorité vitesse), comme en plané.
    Limiteur d'incidence (comme un autopilote) : hors de [alpha_floor, alpha_ceil], la profondeur ramène
    l'incidence dans le domaine (évite ressources et piqués trop brutaux).
    Anticipations par vol (thr_ff, theta_ff, de_ff) : valeurs d'équilibre au départ.
    """

    def __init__(self, n, gains, thr_ff=None, theta_ff=None, de_ff=None, g=9.81):
        self.k, self.g = gains, g
        self.thr_ff = np.full(n, gains["thr_0"]) if thr_ff is None else np.asarray(thr_ff, float)
        self.theta_ff = np.full(n, gains["theta_0"]) if theta_ff is None else np.asarray(theta_ff, float)
        self.de_ff = np.zeros(n) if de_ff is None else np.asarray(de_ff, float)
        self.I_T, self.I_D, self.I_G = np.zeros(n), np.zeros(n), np.zeros(n)
        self.Vdot = np.zeros(n)
        self.Va_prev = None

    def __call__(self, Va, alpha, gamma, theta, q, q_ref, h, h_ref, V_ref, engine_off, freeze, dt):
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
        th_cmd = np.where(engine_off, k["theta_glide_0"] + k["k_GP"] * e_G + self.I_G,
                          self.theta_ff + k["k_DP"] * e_D + self.I_D)
        th_cmd = np.clip(th_cmd, -k["theta_cmd_max"], k["theta_cmd_max"])
        de = self.de_ff - k["k_th"] * (th_cmd - theta) + k["k_q"] * (q - q_ref)
        a_lo, a_hi = np.deg2rad(k["alpha_floor_deg"]), np.deg2rad(k["alpha_ceil_deg"])
        de = de + k["k_alpha"] * (np.minimum(alpha - a_lo, 0.0) + np.maximum(alpha - a_hi, 0.0))
        return de, np.where(engine_off, 0.0, thr)


def lateral(phi_ref, phi, p, r, r_ref, beta, gains, da_ff, dr_ff):
    """Pilote latéral : inclinaison par les ailerons, virage coordonné par la direction.
    Conventions UMN : da > 0 -> roulis à gauche ; dr > 0 (TEL) -> lacet à gauche."""
    k = gains
    da = da_ff - k["k_phi"] * (phi_ref - phi) + k["k_p"] * p
    dr = dr_ff + k["k_r"] * (r - r_ref) - k["k_beta"] * beta
    return da, dr
