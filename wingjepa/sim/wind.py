"""Vent par vol : vent moyen + rafales (1 - cos) + turbulence de Dryden, et vent estimé.

Comme UMNUAV_sim_v4 : le vent moyen ne fait que déplacer l'avion ; rafales et turbulence sont
« instationnaires » et entrent dans l'aérodynamique (axes avion).
Conventions (docs/conventions.md) : w_x horizontal dans l'axe de vol (positif = vent arrière),
w_d vertical (positif vers le bas).

Dryden basse altitude, MIL-F-8785C (formules en pieds), comme le bloc de l'Aerospace Blockset utilisé
par UMN : L_w = h, L_u = h / (0.177 + 0.000823 h)^1.2, sigma_w = 0.1 W20,
sigma_u = sigma_w / (0.177 + 0.000823 h)^0.4. u_g : filtre du 1er ordre (discrétisation exacte) ;
w_g : (1 + sqrt(3) T s) / (1 + T s)^2, T = L_w / Va (Euler-Maruyama, variance stationnaire sigma_w^2).
"""
import numpy as np

FT, KT = 0.3048, 0.514444
W20 = {"none": 0.0, "light": 15 * KT, "moderate": 30 * KT, "severe": 45 * KT}


class Wind:
    def __init__(self, rng, n, cfg, duration):
        self.rng, self.n = rng, n
        self.steady_x = rng.uniform(-cfg["steady_horizontal_max"], cfg["steady_horizontal_max"], n)
        self.steady_d = rng.uniform(-cfg["steady_vertical_max"], cfg["steady_vertical_max"], n)
        levels = list(cfg["turbulence_probs"])
        probs = np.array([cfg["turbulence_probs"][k] for k in levels], dtype=float)
        self.turb_level = np.array(levels)[rng.choice(len(levels), size=n, p=probs / probs.sum())]
        # rafales : (vol, début, durée, amplitude horizontale, amplitude verticale)
        rate = cfg["gust_rate_per_min"] / 60.0
        gusts = []
        for i in range(n):
            t = rng.exponential(1 / rate) if rate > 0 else np.inf
            while t < duration:
                dur = rng.uniform(*cfg["gust_duration_s"])
                gusts.append((i, t, dur, *rng.uniform(-1, 1, 2) * cfg["gust_amplitude_max"]))
                t += dur + rng.exponential(1 / rate)
        self.gusts = np.array(gusts).reshape(-1, 5)
        # vent estimé : passe-bas du vrai vent (sans turbulence) + biais lent, comme un estimateur embarqué
        self.est_tau = cfg["estimate_tau_s"]
        self.bias_sigma, self.bias_T = cfg["estimate_bias_sigma"], cfg["estimate_bias_T_s"]
        self.est_x, self.est_d = self.steady_x.copy(), self.steady_d.copy()
        self.bias_x = rng.normal(0, self.bias_sigma, n)
        self.bias_d = rng.normal(0, self.bias_sigma / 3, n)
        self.w20 = np.array([W20[level] for level in self.turb_level])
        self.u_g, self.w1, self.w2 = np.zeros(n), np.zeros(n), np.zeros(n)

    def inertial(self, t):
        """Vent moyen + rafales au temps t, (w_x, w_d) de forme (n,)."""
        wx, wd = self.steady_x.copy(), self.steady_d.copy()
        if len(self.gusts):
            i, t0, dur, ax, ad = self.gusts.T
            on = (t >= t0) & (t < t0 + dur)
            if on.any():
                s = 0.5 * (1 - np.cos(2 * np.pi * (t - t0[on]) / dur[on]))
                np.add.at(wx, i[on].astype(int), ax[on] * s)
                np.add.at(wd, i[on].astype(int), ad[on] * s)
        return wx, wd

    def step_turbulence(self, Va, h, dt):
        """Avance la turbulence d'un pas ; renvoie (u_g, w_g) en axes avion."""
        Va = np.maximum(Va, 5.0)
        h_ft = np.clip(h, 10.0, 300.0) / FT
        k = 0.177 + 0.000823 * h_ft
        L_u, L_w = h_ft / k**1.2 * FT, h_ft * FT
        sig_w = 0.1 * self.w20
        sig_u = sig_w / k**0.4
        a = np.exp(-Va * dt / L_u)
        self.u_g = a * self.u_g + sig_u * np.sqrt(1 - a**2) * self.rng.standard_normal(self.n)
        T = L_w / Va
        eta = sig_w * np.sqrt(T / dt) * self.rng.standard_normal(self.n)
        w1_dot, w2_dot = self.w2, (-self.w1 - 2 * T * self.w2 + eta) / T**2
        self.w1, self.w2 = self.w1 + dt * w1_dot, self.w2 + dt * w2_dot
        return self.u_g.copy(), self.w1 + np.sqrt(3) * T * self.w2

    def step_estimate(self, wx_true, wd_true, dt):
        b = np.exp(-dt / self.bias_T)
        s = np.sqrt(1 - b**2)
        self.bias_x = b * self.bias_x + self.bias_sigma * s * self.rng.standard_normal(self.n)
        self.bias_d = b * self.bias_d + self.bias_sigma / 3 * s * self.rng.standard_normal(self.n)
        g = dt / (self.est_tau + dt)
        self.est_x += g * (wx_true - self.est_x)
        self.est_d += g * (wd_true - self.est_d)
        return self.est_x + self.bias_x, self.est_d + self.bias_d
