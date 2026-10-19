"""Vent par vol (3D) : vent moyen + rafales (1 - cos) + turbulence de Dryden, et vent estimé.

Comme UMNUAV_sim_v4 : le vent moyen ne fait que déplacer l'avion ; rafales et turbulence sont
« instationnaires » et entrent dans l'aérodynamique (axes avion).
Repère inertiel NED : nord, est, bas (vent descendant positif).

Dryden basse altitude, MIL-F-8785C (formules en pieds), comme le bloc de l'Aerospace Blockset utilisé
par UMN : L_w = h, L_u = L_v = h / (0.177 + 0.000823 h)^1.2, sigma_w = 0.1 W20,
sigma_u = sigma_v = sigma_w / (0.177 + 0.000823 h)^0.4.
u : filtre du 1er ordre (discrétisation exacte) ; v, w : (1 + sqrt(3) T s) / (1 + T s)^2, T = L / Va
(Euler-Maruyama, variance stationnaire sigma^2). Turbulence angulaire non modélisée.
"""
import numpy as np

FT, KT = 0.3048, 0.514444
W20 = {"none": 0.0, "light": 15 * KT, "moderate": 30 * KT, "severe": 45 * KT}


class Wind:
    def __init__(self, rng, n, cfg, duration):
        self.rng, self.n = rng, n
        mag = rng.uniform(0, cfg["steady_horizontal_max"], n)
        direction = rng.uniform(0, 2 * np.pi, n)
        self.steady = np.stack([mag * np.cos(direction), mag * np.sin(direction),
                                rng.uniform(-cfg["steady_vertical_max"], cfg["steady_vertical_max"], n)], 1)
        levels = list(cfg["turbulence_probs"])
        probs = np.array([cfg["turbulence_probs"][k] for k in levels], dtype=float)
        self.turb_level = np.array(levels)[rng.choice(len(levels), size=n, p=probs / probs.sum())]
        self.w20 = np.array([W20[level] for level in self.turb_level])
        # rafales : (vol, début, durée, amplitude N, E, D)
        rate = cfg["gust_rate_per_min"] / 60.0
        gusts = []
        for i in range(n):
            t = rng.exponential(1 / rate) if rate > 0 else np.inf
            while t < duration:
                dur = rng.uniform(*cfg["gust_duration_s"])
                gusts.append((i, t, dur, *rng.uniform(-1, 1, 3) * cfg["gust_amplitude_max"]))
                t += dur + rng.exponential(1 / rate)
        self.gusts = np.array(gusts).reshape(-1, 6)
        # états des filtres de Dryden (u ; v et w à deux états)
        self.u_g = np.zeros(n)
        self.v1, self.v2, self.w1, self.w2 = (np.zeros(n) for _ in range(4))
        # vent estimé : passe-bas du vrai vent (sans turbulence) + biais lent, comme un estimateur embarqué
        self.est_tau = cfg["estimate_tau_s"]
        self.bias_sigma, self.bias_T = cfg["estimate_bias_sigma"], cfg["estimate_bias_T_s"]
        self.est = self.steady.copy()
        self.bias = rng.normal(0, 1, (n, 3)) * self.bias_sigma * np.array([1, 1, 1 / 3])

    def gust(self, t):
        """Rafales au temps t, repère NED (n, 3)."""
        g = np.zeros((self.n, 3))
        if len(self.gusts):
            i, t0, dur = self.gusts[:, 0], self.gusts[:, 1], self.gusts[:, 2]
            on = (t >= t0) & (t < t0 + dur)
            if on.any():
                s = 0.5 * (1 - np.cos(2 * np.pi * (t - t0[on]) / dur[on]))
                np.add.at(g, i[on].astype(int), self.gusts[on, 3:] * s[:, None])
        return g

    def _second_order(self, x1, x2, sig, T, dt):
        eta = sig * np.sqrt(T / dt) * self.rng.standard_normal(self.n)
        x1n, x2n = x1 + dt * x2, x2 + dt * (-x1 - 2 * T * x2 + eta) / T**2
        return x1n, x2n, x1n + np.sqrt(3) * T * x2n

    def step_turbulence(self, Va, h, dt):
        """Avance la turbulence d'un pas ; renvoie (n, 3) en axes avion."""
        Va = np.maximum(Va, 5.0)
        h_ft = np.clip(h, 10.0, 300.0) / FT
        k = 0.177 + 0.000823 * h_ft
        L_u, L_w = h_ft / k**1.2 * FT, h_ft * FT
        sig_w = 0.1 * self.w20
        sig_u = sig_w / k**0.4
        a = np.exp(-Va * dt / L_u)
        self.u_g = a * self.u_g + sig_u * np.sqrt(1 - a**2) * self.rng.standard_normal(self.n)
        self.v1, self.v2, v_g = self._second_order(self.v1, self.v2, sig_u, L_u / Va, dt)
        self.w1, self.w2, w_g = self._second_order(self.w1, self.w2, sig_w, L_w / Va, dt)
        return np.stack([self.u_g, v_g, w_g], 1)

    def step_estimate(self, wind_true_ned, dt):
        """Vent estimé (n, 3) : vrai vent (moyen + rafales) filtré + biais lent (Ornstein-Uhlenbeck)."""
        b = np.exp(-dt / self.bias_T)
        sig = self.bias_sigma * np.array([1, 1, 1 / 3])
        self.bias = b * self.bias + sig * np.sqrt(1 - b**2) * self.rng.standard_normal((self.n, 3))
        self.est += dt / (self.est_tau + dt) * (wind_true_ned - self.est)
        return self.est + self.bias
