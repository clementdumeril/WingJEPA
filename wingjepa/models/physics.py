"""Tête physique du modèle C (docs/model_design.md, § 11).

Elle remplace le décodeur MLP. Au lieu de relire l'état dans chaque latent prédit, elle PORTE un état
physique x_k = (u, v, w, p, q, r, phi, theta) et l'avance avec les équations de Newton en 3D :

  forces et moments = a priori (coefficients aéro + poussée, fonction de x_k et des commandes)
                      + correction apprise (6 coefficients, fonction du latent z_k, de x_k, des commandes)
  x_{k+1} = RK4 des équations du mouvement sur 0,05 s, avec la VRAIE masse et les VRAIES inerties.

(u, v, w) : vitesse par rapport à la masse d'air moyenne, en axes avion. Avec un vent moyen constant, ses
équations sont celles du vol sans vent (même convention que le simulateur) ; rafales et turbulence sont
inconnues de la tête, la correction peut en tenir compte via le latent.
La correction est exprimée en coefficients sans dimension en axes avion (CX, CY, CZ, Cl, Cm, Cn),
multipliés par la pression dynamique : c'est la forme naturelle d'une erreur d'aéro (avion plus traînant,
gouverne plus efficace...). Elle est initialisée à zéro : au départ, la tête est exactement l'a priori.
Gouvernes : on utilise les commandes (actionneurs du 1er ordre de 0,02 s, plus rapides que le pas).

L'a priori est un modèle linéaire en paramètres, coefficients en axes de stabilité comme les tables NASA
(régresseurs : TERMS ; ajusté, générique 25e, faux ou nul : scripts/fit_prior.py).
"""
import math

import torch
import torch.nn as nn

from wingjepa.data.windows import N_CTRL, N_STATE, N_WIND

COEFS = ["CD", "CY", "CL", "Cl", "Cm", "Cn", "T"]
# noms des régresseurs, pour les fichiers d'a priori lisibles
TERMS = {
    "CL": ["1", "alpha", "alpha2", "de", "qhat"],
    "CD": ["1", "alpha", "alpha2", "beta2", "|de|", "|da|", "|dr|"],
    "CY": ["beta", "da", "dr", "phat", "rhat"],
    "Cl": ["beta", "da", "dr", "phat", "rhat"],
    "Cm": ["1", "alpha", "alpha2", "de", "qhat"],
    "Cn": ["beta", "da", "dr", "phat", "rhat"],
    "T": ["1", "dt", "dt2", "dt*Va", "Va", "Va2"],                    # poussée [N]
}


def regressors(alpha, beta, de, da, dr, phat, qhat, rhat, thr, Va):
    """Colonnes des régresseurs par coefficient (marche avec numpy et torch)."""
    one = alpha * 0 + 1
    lat = [beta, da, dr, phat, rhat]
    return {
        "CL": [one, alpha, alpha * alpha, de, qhat],
        "CD": [one, alpha, alpha * alpha, beta * beta, abs(de), abs(da), abs(dr)],
        "CY": lat, "Cl": lat, "Cn": lat,
        "Cm": [one, alpha, alpha * alpha, de, qhat],
        "T": [one, thr, thr * thr, thr * Va, Va, Va * Va],
    }


class PhysicsHead(nn.Module):
    """prior : dict (scripts/fit_prior.py) avec coef {nom: liste}, rho, S, b, c, g.
    norm_mean, norm_std : normalisation des 16 grandeurs (pour lire les entrées et rendre des sorties
    normalisées, comme le décodeur MLP)."""

    def __init__(self, prior, norm_mean, norm_std, z_dim=32, hidden=64, out_scale=0.1, n_sub=1, dt=0.05):
        super().__init__()
        for k in COEFS:
            self.register_buffer(f"c_{k}", torch.tensor(prior["coef"][k], dtype=torch.float32))
        self.rho, self.S, self.b, self.c, self.g = (float(prior[k]) for k in ("rho", "S", "b", "c", "g"))
        self.register_buffer("mean", torch.as_tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(norm_std, dtype=torch.float32))
        # bornes de sécurité de l'état porté (évite les NaN si une prédiction part loin)
        self.register_buffer("lo", torch.tensor([0.5, -40, -40, -8, -8, -8, -math.pi, -1.4]))
        self.register_buffer("hi", torch.tensor([80.0, 40, 40, 8, 8, 8, math.pi, 1.4]))
        self.out_scale, self.n_sub, self.dt = out_scale, n_sub, dt
        self.net = nn.Sequential(nn.Linear(z_dim + N_STATE + N_CTRL + N_WIND, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # ---------- a priori
    def prior_body(self, alpha, beta, de, da, dr, p, q, r, thr, Va):
        """Coefficients a priori en axes avion (B, 6) : CX, CY, CZ (poussée incluse), Cl, Cm, Cn."""
        V2 = 2 * Va
        R = regressors(alpha, beta, de, da, dr, p * self.b / V2, q * self.c / V2, r * self.b / V2, thr, Va)
        C = {k: sum(cf * x for cf, x in zip(getattr(self, f"c_{k}"), R[k])) for k in COEFS}
        qS = 0.5 * self.rho * Va * Va * self.S
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        # axes de stabilité -> axes avion (comme UltraStick.forces)
        CX = -ca * C["CD"] + sa * C["CL"] + C["T"] / qS
        CZ = -sa * C["CD"] - ca * C["CL"]
        Cl = ca * C["Cl"] - sa * C["Cn"]
        Cn = sa * C["Cl"] + ca * C["Cn"]
        return torch.stack([CX, C["CY"], CZ, Cl, C["Cm"], Cn], -1)

    # ---------- état
    @staticmethod
    def air_data(x):
        u, v, w = x[:, 0], x[:, 1], x[:, 2]
        Va = torch.sqrt(u * u + v * v + w * w).clamp_min(3.0)
        # beta = asin(v / Va), écrit avec atan2 : même valeur, mais dérivée finie partout
        return Va, torch.atan2(w, u), torch.atan2(v, torch.sqrt(u * u + w * w + 1e-6))

    def from_features(self, s):
        """État physique (B, 8) depuis les 9 grandeurs NON normalisées."""
        Va, al, be = s[:, 0], s[:, 1], s[:, 2]
        uvw = torch.stack([Va * torch.cos(al) * torch.cos(be), Va * torch.sin(be), Va * torch.sin(al) * torch.cos(be)], -1)
        return torch.cat([uvw, s[:, 3:8]], -1)

    def to_features(self, x, wind_down):
        """9 grandeurs (B, 9) NON normalisées ; vitesse verticale = air + vent vertical (vers le bas)."""
        Va, al, be = self.air_data(x)
        u, v, w, phi, th = x[:, 0], x[:, 1], x[:, 2], x[:, 6], x[:, 7]
        hdot = u * torch.sin(th) - v * torch.sin(phi) * torch.cos(th) - w * torch.cos(phi) * torch.cos(th) - wind_down
        return torch.cat([torch.stack([Va, al, be], -1), x[:, 3:8], hdot[:, None]], -1)

    # ---------- dynamique
    def derivatives(self, x, ctrl, dC, ac):
        """x (B, 8), ctrl (B, 4) physiques, dC (B, 6) correction, ac (B, 5) masse et inerties."""
        # bornes aussi aux étapes intermédiaires du RK4 (sinon theta peut y franchir 90 deg : tan infinie)
        x = torch.maximum(torch.minimum(x, self.hi), self.lo)
        u, v, w, p, q, r, phi, th = x.unbind(-1)
        Va, alpha, beta = self.air_data(x)
        C = self.prior_body(alpha, beta, ctrl[:, 0], ctrl[:, 1], ctrl[:, 2], p, q, r, ctrl[:, 3], Va) + dC
        qS = 0.5 * self.rho * Va * Va * self.S
        m, Ixx, Iyy, Izz, Ixz = ac.unbind(-1)
        g = self.g
        sf, cf, st, ct = torch.sin(phi), torch.cos(phi), torch.sin(th), torch.cos(th)
        du = r * v - q * w - g * st + C[:, 0] * qS / m
        dv = p * w - r * u + g * ct * sf + C[:, 1] * qS / m
        dw = q * u - p * v + g * ct * cf + C[:, 2] * qS / m
        # moments ; inertie [[Ixx 0 -Ixz] [0 Iyy 0] [-Ixz 0 Izz]] (convention UMN, comme le simulateur)
        Mx, My, Mz = C[:, 3] * qS * self.b, C[:, 4] * qS * self.c, C[:, 5] * qS * self.b
        hx, hy, hz = Ixx * p - Ixz * r, Iyy * q, Izz * r - Ixz * p          # I omega
        ex = Mx - (q * hz - r * hy)                                          # M - omega x (I omega)
        ey = My - (r * hx - p * hz)
        ez = Mz - (p * hy - q * hx)
        G = Ixx * Izz - Ixz * Ixz
        dp, dq, dr = (Izz * ex + Ixz * ez) / G, ey / Iyy, (Ixz * ex + Ixx * ez) / G
        dphi = p + (q * sf + r * cf) * torch.tan(th)
        dth = q * cf - r * sf
        return torch.stack([du, dv, dw, dp, dq, dr, dphi, dth], -1)

    def step(self, x, ctrl, dC, ac):
        h = self.dt / self.n_sub
        for _ in range(self.n_sub):
            k1 = self.derivatives(x, ctrl, dC, ac)
            k2 = self.derivatives(x + 0.5 * h * k1, ctrl, dC, ac)
            k3 = self.derivatives(x + 0.5 * h * k2, ctrl, dC, ac)
            k4 = self.derivatives(x + h * k3, ctrl, dC, ac)
            x = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
            x = torch.maximum(torch.minimum(x, self.hi), self.lo)
        return x

    def correction(self, z, s_n, u_n):
        """Correction (B, 6) et sortie brute du réseau (pour la pénalité de taille)."""
        raw = self.net(torch.cat([z, s_n, u_n], -1))
        return raw * self.out_scale, raw

    def forward(self, zs, s0, U, ac):
        """zs (B, T, D) : latents z_0..z_{T-1} (début de chaque pas) ; s0 (B, 9) : dernier état observé,
        normalisé ; U (B, T, 7) : entrées futures normalisées ; ac (B, 5).
        -> états prédits normalisés (B, T, 9), sorties brutes de la correction (B, T, 6)."""
        mean_s, std_s = self.mean[:N_STATE], self.std[:N_STATE]
        mean_u, std_u = self.mean[N_STATE:], self.std[N_STATE:]
        Uph = U * std_u + mean_u
        x = self.from_features(s0 * std_s + mean_s)
        s_n, out, raws = s0, [], []
        for k in range(U.shape[1]):
            dC, raw = self.correction(zs[:, k], s_n, U[:, k])
            x = self.step(x, Uph[:, k, :N_CTRL], dC, ac)
            s_n = (self.to_features(x, Uph[:, k, N_CTRL + 2]) - mean_s) / std_s
            out.append(s_n)
            raws.append(raw)
        return torch.stack(out, 1), torch.stack(raws, 1)
