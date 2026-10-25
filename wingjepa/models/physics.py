"""Physics head of model C (docs/model_design.md, section 11).

It replaces the MLP decoder. Instead of reading the state back out of each predicted latent, it CARRIES a
physical state x_k = (u, v, w, p, q, r, phi, theta) and advances it with the 3D Newton equations:

  forces and moments = prior (aero coefficients + thrust, function of x_k and the controls)
                       + learned correction (6 coefficients, function of the latent z_k, x_k, the controls)
  x_{k+1} = RK4 of the equations of motion over 0.05 s, with the TRUE mass and the TRUE inertias.

(u, v, w): velocity relative to the mean air mass, in body axes. With a constant mean wind, its
equations are those of flight without wind (same convention as the simulator); gusts and turbulence are
unknown to the head, the correction can account for them through the latent.
The correction is expressed as dimensionless coefficients in body axes (CX, CY, CZ, Cl, Cm, Cn),
multiplied by the dynamic pressure: this is the natural form of an aero error (more draggy aircraft,
more effective control surface...). It is initialized to zero: at the start, the head is exactly the prior.
Control surfaces: the commands are used (first-order actuators with a 0.02 s time constant, faster than the step).

The prior is a model linear in its parameters, with coefficients in stability axes like the NASA tables
(regressors: TERMS; fitted, generic 25th, wrong or null: scripts/fit_prior.py).
"""
import math

import torch
import torch.nn as nn

from wingjepa.data.windows import N_CTRL, N_STATE, N_WIND

COEFS = ["CD", "CY", "CL", "Cl", "Cm", "Cn", "T"]
# regressor names, for readable prior files
TERMS = {
    "CL": ["1", "alpha", "alpha2", "de", "qhat"],
    "CD": ["1", "alpha", "alpha2", "beta2", "|de|", "|da|", "|dr|"],
    "CY": ["beta", "da", "dr", "phat", "rhat"],
    "Cl": ["beta", "da", "dr", "phat", "rhat"],
    "Cm": ["1", "alpha", "alpha2", "de", "qhat"],
    "Cn": ["beta", "da", "dr", "phat", "rhat"],
    "T": ["1", "dt", "dt2", "dt*Va", "Va", "Va2"],                    # thrust [N]
}


def regressors(alpha, beta, de, da, dr, phat, qhat, rhat, thr, Va):
    """Regressor columns for each coefficient (works with numpy and torch)."""
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
    """prior: dict (scripts/fit_prior.py) with coef {name: list}, rho, S, b, c, g.
    norm_mean, norm_std: normalization of the 16 quantities (to read the inputs and return normalized
    outputs, like the MLP decoder)."""

    def __init__(self, prior, norm_mean, norm_std, z_dim=32, hidden=64, out_scale=0.1, n_sub=1, dt=0.05):
        super().__init__()
        for k in COEFS:
            self.register_buffer(f"c_{k}", torch.tensor(prior["coef"][k], dtype=torch.float32))
        self.rho, self.S, self.b, self.c, self.g = (float(prior[k]) for k in ("rho", "S", "b", "c", "g"))
        self.register_buffer("mean", torch.as_tensor(norm_mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(norm_std, dtype=torch.float32))
        # safety bounds on the carried state (avoids NaN if a prediction drifts far)
        self.register_buffer("lo", torch.tensor([0.5, -40, -40, -8, -8, -8, -math.pi, -1.4]))
        self.register_buffer("hi", torch.tensor([80.0, 40, 40, 8, 8, 8, math.pi, 1.4]))
        self.out_scale, self.n_sub, self.dt = out_scale, n_sub, dt
        self._step = self.step                                   # replaced by a compiled version (compile_step())
        self.net = nn.Sequential(nn.Linear(z_dim + N_STATE + N_CTRL + N_WIND, hidden), nn.GELU(),
                                 nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 6))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    # ---------- prior
    def prior_body(self, alpha, beta, de, da, dr, p, q, r, thr, Va):
        """Prior coefficients in body axes (B, 6): CX, CY, CZ (thrust included), Cl, Cm, Cn."""
        V2 = 2 * Va
        R = regressors(alpha, beta, de, da, dr, p * self.b / V2, q * self.c / V2, r * self.b / V2, thr, Va)
        C = {k: sum(cf * x for cf, x in zip(getattr(self, f"c_{k}"), R[k])) for k in COEFS}
        qS = 0.5 * self.rho * Va * Va * self.S
        ca, sa = torch.cos(alpha), torch.sin(alpha)
        # stability axes -> body axes (as in UltraStick.forces)
        CX = -ca * C["CD"] + sa * C["CL"] + C["T"] / qS
        CZ = -sa * C["CD"] - ca * C["CL"]
        Cl = ca * C["Cl"] - sa * C["Cn"]
        Cn = sa * C["Cl"] + ca * C["Cn"]
        return torch.stack([CX, C["CY"], CZ, Cl, C["Cm"], Cn], -1)

    # ---------- state
    @staticmethod
    def air_data(x):
        u, v, w = x[:, 0], x[:, 1], x[:, 2]
        Va = torch.sqrt(u * u + v * v + w * w).clamp_min(3.0)
        # beta = asin(v / Va), written with atan2: same value, but finite derivative everywhere
        return Va, torch.atan2(w, u), torch.atan2(v, torch.sqrt(u * u + w * w + 1e-6))

    def from_features(self, s):
        """Physical state (B, 8) from the 9 UNnormalized quantities."""
        Va, al, be = s[:, 0], s[:, 1], s[:, 2]
        uvw = torch.stack([Va * torch.cos(al) * torch.cos(be), Va * torch.sin(be), Va * torch.sin(al) * torch.cos(be)], -1)
        return torch.cat([uvw, s[:, 3:8]], -1)

    def to_features(self, x, wind_down):
        """9 UNnormalized quantities (B, 9); vertical speed = air + vertical wind (downward)."""
        Va, al, be = self.air_data(x)
        u, v, w, phi, th = x[:, 0], x[:, 1], x[:, 2], x[:, 6], x[:, 7]
        hdot = u * torch.sin(th) - v * torch.sin(phi) * torch.cos(th) - w * torch.cos(phi) * torch.cos(th) - wind_down
        return torch.cat([torch.stack([Va, al, be], -1), x[:, 3:8], hdot[:, None]], -1)

    # ---------- dynamics
    def derivatives(self, x, ctrl, dC, ac):
        """x (B, 8), ctrl (B, 4) physical, dC (B, 6) correction, ac (B, 5) mass and inertias."""
        # bounds also at the intermediate RK4 stages (otherwise theta can cross 90 deg there: infinite tan)
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
        # moments; inertia [[Ixx 0 -Ixz] [0 Iyy 0] [-Ixz 0 Izz]] (UMN convention, as in the simulator)
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
        """Correction (B, 6) and raw network output (for the size penalty)."""
        raw = self.net(torch.cat([z, s_n, u_n], -1))
        return raw * self.out_scale, raw

    def compile_step(self):
        """Compile one integration step (RK4 stages, prior, Newton-Euler) into fused kernels with
        torch.compile. The step is otherwise a few hundred tiny tensor operations, whose fixed cost
        dominates the training time. Only the step is compiled, not the loop over the horizon, so the
        curriculum (horizon growing during training) does not trigger recompilations."""
        self._step = torch.compile(self.step, dynamic=False)
        return self

    def forward(self, zs, s0, U, ac):
        """zs (B, T, D): latents z_0..z_{T-1} (start of each step); s0 (B, 9): last observed state,
        normalized; U (B, T, 7): normalized future inputs; ac (B, 5).
        -> normalized predicted states (B, T, 9), raw outputs of the correction (B, T, 6)."""
        mean_s, std_s = self.mean[:N_STATE], self.std[:N_STATE]
        mean_u, std_u = self.mean[N_STATE:], self.std[N_STATE:]
        Uph = U * std_u + mean_u
        x = self.from_features(s0 * std_s + mean_s)
        s_n, out, raws = s0, [], []
        for k in range(U.shape[1]):
            dC, raw = self.correction(zs[:, k], s_n, U[:, k])
            x = self._step(x, Uph[:, k, :N_CTRL], dC, ac)
            s_n = (self.to_features(x, Uph[:, k, N_CTRL + 2]) - mean_s) / std_s
            out.append(s_n)
            raws.append(raw)
        return torch.stack(out, 1), torch.stack(raws, 1)
