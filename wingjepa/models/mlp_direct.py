"""Plain MLP dynamics model (Nagabandi et al. 2018), the simplest neural baseline: no encoder, no latent.

  input of step k : last H measured / predicted states (H x 9), the H commands that led to them (H x 4),
                    command and wind applied from step k to k+1 (7), aircraft parameters if aircraft_input (5)
  output          : state increment, in normalized units: s_{k+1} = s_k + MLP(...)
The standardization is affine per quantity, so a normalized increment is the physical increment divided by
the std of the quantity, and the prediction decodes like those of the other models (Normalizer.state_inverse).
Rollout: autoregressive over the T future steps, each predicted state is appended to the sliding window; the
commands of the window are the commands actually applied (past) or given (future), as for the other models.
Parameters: with n_in = 13 H + 7 (+ 5 with aircraft_input) and hidden widths h_1..h_n,
  sum_l (n_{l-1} + 1) n_l with n_0 = n_in, n_{n+1} = 9; H = 11 and 50,50: 10559.
Same interface as WorldModel for the evaluation and control scripts (predict_states, future_inputs, n_params,
n_ctx); latent = False tells the scripts that there is no latent to probe or straighten.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wingjepa.data.windows import N_CTRL, N_STATE, N_WIND
from wingjepa.models.jepa import NOMINAL_AIRCRAFT, WorldModel


class MLPDirect(nn.Module):
    latent = False

    def __init__(self, H, hidden=(50, 50), n_ctx=0):
        super().__init__()
        self.H, self.n_ctx, self.head = H, n_ctx, "mlp_direct"
        self.receptive_field = H
        n = H * (N_STATE + N_CTRL) + N_CTRL + N_WIND + n_ctx
        layers = []
        for h in hidden:
            layers += [nn.Linear(n, h), nn.GELU()]
            n = h
        layers.append(nn.Linear(n, N_STATE))
        self.net = nn.Sequential(*layers)
        # zero increment at the start: the untrained model is the "constant state" reference
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("ac_nominal", torch.tensor(NOMINAL_AIRCRAFT), persistent=False)

    @classmethod
    def from_config(cls, cfg):
        hidden = tuple(int(x) for x in str(cfg.get("mlp_hidden") or "50,50").split(","))
        return cls(cfg["H"], hidden, n_ctx=5 if cfg.get("aircraft_input") else 0)

    future_inputs = staticmethod(WorldModel.future_inputs)

    def ctx(self, ac):
        """Aircraft parameters (B, 5) -> input: relative deviation from nominal, / 0.3 (as WorldModel.ctx)."""
        if not self.n_ctx:
            return None
        return (ac[:, :5] / self.ac_nominal - 1) / 0.3

    def step(self, states, cmds, u, c=None):
        """states (B, H, 9), cmds (B, H, 4) commands that led to them, u (B, 7) command and wind applied
        next -> next normalized state (B, 9)."""
        x = [states.flatten(1), cmds.flatten(1), u] + ([] if c is None else [c])
        return states[:, -1] + self.net(torch.cat(x, -1))

    def predict_states(self, seq, H, hold_wind=True, ac=None):
        """seq : (B, H+T, 16) normalized -> predicted normalized states (B, T, 9), None (no physics term).
        Only the states of the H past steps are read; the command of step k is applied from k to k+1."""
        assert H == self.H, f"model built for H = {self.H}, called with H = {H}"
        U = self.future_inputs(seq, H, hold_wind)                     # (B, T, 7)
        # commands that led to each state: the one of step i-1 for state i (zero before the window)
        cmd = F.pad(seq[:, :-1, N_STATE:N_STATE + N_CTRL], (0, 0, 1, 0))
        c = self.ctx(ac)
        win, out = seq[:, :H, :N_STATE], []
        for k in range(U.shape[1]):
            s = self.step(win, cmd[:, k:k + H], U[:, k], c)
            out.append(s)
            win = torch.cat([win[:, 1:], s[:, None]], 1)
        return torch.stack(out, 1), None

    def n_params(self):
        return {"mlp": sum(p.numel() for p in self.parameters())}
