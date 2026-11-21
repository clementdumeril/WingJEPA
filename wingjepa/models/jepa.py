"""JEPA-type latent world model (docs/model_design.md): TCN encoder, GRU predictor, decoder.

  past: states (B, H, 9) --encoder--> z (B, D)
  (the encoder sees neither the controls nor the wind, as in SkyJEPA, whose state encoder reads the states only
  and whose action encoder meets them in the predictor (eq. 7-8))
  z_k, input_k (controls + wind held constant) --predictor--> z_{k+1}
  z_k --decoder--> state (B, 9)
The decoder is the "head": black box (model A, MLP) or physics head (model C, models/physics.py).

v2 (docs/model_design.md, section 11): 2 s of past (H = 40) and an encoder that sees all of it (dilations
1, 2, 4, 8, 16: receptive field 63 steps); known aircraft mass and inertias, added to the past summary
before the last encoder layer (n_ctx = 5). v1: H = 10, dilations 1, 2, 4, no aircraft input.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wingjepa.data.windows import N_CTRL, N_STATE, N_WIND


class CausalBlock(nn.Module):
    """Dilated causal 1D convolution + activation + residual connection."""

    def __init__(self, c_in, c_out, k, d):
        super().__init__()
        self.pad = (k - 1) * d                      # left padding only: never the future
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                           # x : (B, C, H)
        y = F.gelu(self.conv(F.pad(x, (self.pad, 0))))
        return y + self.skip(x)


def causal_tcn(c, channels, k, dilations):
    blocks = []
    for ch, d in zip(channels, dilations):
        blocks.append(CausalBlock(c, ch, k, d))
        c = ch
    return nn.Sequential(*blocks)


class TCNEncoder(nn.Module):
    """Causal TCN over the past states; its last step, joined by the aircraft parameters if n_ctx > 0, gives z."""

    def __init__(self, n_in=16, channels=(16, 16, 32), k=3, dilations=(1, 2, 4), z_dim=32, n_ctx=0):
        super().__init__()
        self.n_in = n_in
        self.tcn = causal_tcn(n_in, channels, k, dilations)
        self.out = nn.Linear(channels[-1] + n_ctx, z_dim)
        self.receptive_field = 1 + (k - 1) * sum(dilations)

    def forward(self, past, ctx=None):              # past : (B, H, n_in) ; ctx : (B, n_ctx), constant over the flight
        h = self.tcn(past[:, :, :self.n_in].transpose(1, 2))[:, :, -1]          # last step
        if ctx is not None:
            h = torch.cat([h, ctx], -1)
        return self.out(h)


class GRUPredictor(nn.Module):
    """act_channels: action encoder of SkyJEPA (Enc_phi, eq. 7-8 of the paper), a causal TCN over the
    commands; at each step the GRU receives the encoding of the command window ending at that step
    instead of the raw command. Part of the predictor, so that it is trained, frozen and reused with it."""

    def __init__(self, n_in=N_CTRL + N_WIND, z_dim=32, act_channels=None, k=3, dilations=(1, 2, 4)):
        super().__init__()
        self.act = causal_tcn(N_CTRL, act_channels, k, dilations) if act_channels else None
        if act_channels:
            n_in += act_channels[-1] - N_CTRL
        self.cell = nn.GRUCell(n_in, z_dim)

    def inputs(self, seq, H, U):
        """GRU inputs of steps t..t+T-1. U: raw future inputs (B, T, 7); with the action encoder, the
        command part becomes Enc_phi of the commands applied up to each step (B, T, C + N_WIND)."""
        if self.act is None:
            return U
        cmd = seq[:, :-1, N_STATE:N_STATE + N_CTRL].transpose(1, 2)    # command of step k applied from k to k+1
        za = self.act(cmd)[:, :, H - 1:].transpose(1, 2)               # windows ending at t..t+T-1
        return torch.cat([za, U[:, :, N_CTRL:]], -1)

    def forward(self, z, u):                        # z : (B, D), u : (B, 7)
        return self.cell(u, z)

    def rollout(self, z, U):                        # U : (B, T, 7) -> (B, T, D) = z_{t+1..t+T}
        out = []
        for k in range(U.shape[1]):
            z = self.cell(U[:, k], z)
            out.append(z)
        return torch.stack(out, 1)


class MLPDecoder(nn.Module):
    def __init__(self, z_dim=32, hidden=64, n_out=N_STATE):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(z_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, n_out))

    def forward(self, z):
        return self.net(z)


def sigreg(z, n_proj=256, n_knots=17, t_max=5.0):
    """SIGReg (LeJEPA, Balestriero & LeCun 2025): pushes the distribution of the z in a batch towards an
    isotropic Gaussian. Random projections; for each one, Epps-Pulley statistic: gap between the
    empirical characteristic function and that of N(0, 1), weighted by exp(-t^2/2)."""
    N, D = z.shape
    A = torch.randn(D, n_proj, device=z.device)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(-t_max, t_max, n_knots, device=z.device)
    w = torch.exp(-0.5 * t**2)
    x = (z @ A).unsqueeze(-1) * t                   # (N, M, K)
    ecf_re, ecf_im = torch.cos(x).mean(0), torch.sin(x).mean(0)   # (M, K)
    err = ((ecf_re - w) ** 2 + ecf_im**2) * w
    return torch.trapezoid(err, t, dim=-1).mean() * N


# nominal aircraft of --aircraft-input 1 (the Ultra Stick 120, archive/ultrastick/configs/ultrastick120.yaml): the
# parameters are given as relative deviations from these values
NOMINAL_AIRCRAFT = (6.804, 0.672485702, 0.889416574, 1.57817209, -0.759258051)


class WorldModel(nn.Module):
    """head = "mlp" (black-box decoder) or "physics" (physics head, prior and norm required).
    act_channels: the predictor receives the commands through an action encoder (SkyJEPA's Enc_phi).
    n_priv > 0: asymmetric JEPA, a separate target encoder reads the clean future windows and n_priv privileged
    simulator parameters (training only: in use, only the encoder, the predictor and the head run)."""

    def __init__(self, z_dim=32, tcn_channels=(16, 16, 32), k=3, dilations=(1, 2, 4), dec_hidden=64,
                 n_ctx=0, head="mlp", prior=None, norm=None, out_scale=0.1, n_sub=1, act_channels=None, n_priv=0):
        super().__init__()
        self.encoder = TCNEncoder(N_STATE, tcn_channels, k, dilations, z_dim, n_ctx)
        self.target_encoder = TCNEncoder(N_STATE, tcn_channels, k, dilations, z_dim, n_priv) if n_priv else None
        self.predictor = GRUPredictor(N_CTRL + N_WIND, z_dim, act_channels, k, dilations)
        self.n_ctx, self.head = n_ctx, head
        self.register_buffer("ac_nominal", torch.tensor(NOMINAL_AIRCRAFT), persistent=False)
        if head == "mlp":
            self.decoder = MLPDecoder(z_dim, dec_hidden)
        elif head == "physics":
            from wingjepa.models.physics import PhysicsHead
            self.decoder = PhysicsHead(prior, norm.mean, norm.std, z_dim, dec_hidden, out_scale, n_sub)
        else:
            raise ValueError(f"unknown head {head!r}: mlp or physics")

    def ctx(self, ac):
        """Aircraft parameters (B, 5) -> encoder input: relative deviation from nominal, / 0.3."""
        if not self.n_ctx:
            return None
        return (ac[:, :5] / self.ac_nominal - 1) / 0.3

    def encode(self, past, ac=None):
        """past : (B, H, 16) normalized window (states, commands, wind) -> z (B, D)."""
        return self.encoder(past[:, :, :N_STATE], self.ctx(ac))

    @staticmethod
    def future_inputs(seq, H, hold_wind=True):
        """Predictor inputs for the future steps: controls actually applied, and estimated wind
        HELD CONSTANT at its last known value (in use, the future wind is not known).
        hold_wind=False: the wind of each step is given ("known wind" evaluation)."""
        ctrl = seq[:, H - 1:-1, N_STATE:N_STATE + N_CTRL]            # control applied from k to k+1
        if hold_wind:
            wind = seq[:, H - 1:H, N_STATE + N_CTRL:].expand(-1, ctrl.shape[1], -1)
        else:
            wind = seq[:, H - 1:-1, N_STATE + N_CTRL:]
        return torch.cat([ctrl, wind], -1)

    def latents(self, seq, H, hold_wind=True, ac=None):
        """seq : (B, H+T, 16) -> z_0 (B, D), predicted latents z_1..z_T (B, T, D), future inputs (B, T, 7)."""
        U = self.future_inputs(seq, H, hold_wind)
        z0 = self.encode(seq[:, :H], ac)
        return z0, self.predictor.rollout(z0, self.predictor.inputs(seq, H, U)), U

    def predict_latents(self, seq, H, hold_wind=True, ac=None):
        """seq : (B, H+T, 16) -> predicted latents (B, T, D) for steps t+1..t+T."""
        return self.latents(seq, H, hold_wind, ac)[1]

    def head_states(self, z0, zs, U, seq, H, ac=None):
        """Normalized predicted states (B, T, 9) and, for the physics head, raw outputs of the correction."""
        if self.head == "mlp":
            return self.decoder(zs), None
        zk = torch.cat([z0[:, None], zs[:, :-1]], 1)                 # latent at the start of each step
        return self.decoder(zk, seq[:, H - 1, :N_STATE], U, ac)

    def predict_states(self, seq, H, hold_wind=True, ac=None):
        z0, zs, U = self.latents(seq, H, hold_wind, ac)
        return self.head_states(z0, zs, U, seq, H, ac)

    def target_latents(self, seq, H, ac=None, priv=None):
        """Encoder applied to the true state windows of H steps ending at t+1..t+T -> (B, T, D).
        Asymmetric JEPA (target_encoder): the target encoder instead, with the normalized privileged parameters
        priv (B, n_priv) as its context."""
        win = seq[:, 1:, :N_STATE].unfold(1, H, 1)                    # (B, T, 9, H), ending at t+1..t+T
        B, T = win.shape[:2]
        enc, c = (self.encoder, self.ctx(ac)) if self.target_encoder is None else (self.target_encoder, priv)
        c = None if c is None else c.repeat_interleave(T, 0)
        z = enc(win.reshape(B * T, N_STATE, H).transpose(1, 2), c)
        return z.reshape(B, T, -1)

    def n_params(self):
        parts = [("encodeur", self.encoder), ("predicteur", self.predictor), ("decodeur", self.decoder)]
        if self.target_encoder is not None:                          # training only, not part of the flown model
            parts.append(("encodeur_cible", self.target_encoder))
        return {k: sum(p.numel() for p in m.parameters()) for k, m in parts}


# architecture options of earlier studies, removed from this code (they are in the tag ultrastick-final):
# config.json key -> the value that means "off"; the last two only change the physics head. A run that used one of
# them cannot be rebuilt here. (Training-only options, such as the EMA target encoder, do not matter for loading.)
REMOVED_OPTIONS = dict(action_history=0, context_dim=0, correction="free", oracle_params=0)


def model_from_config(cfg, prior=None):
    """Rebuild the model of a run (config.json). v1 runs lack the v2 keys: v1 values are used."""
    keys = list(REMOVED_OPTIONS)[:2] + (list(REMOVED_OPTIONS)[2:] if cfg.get("head") == "physics" else [])
    used = [f"{k}={cfg[k]}" for k in keys if cfg.get(k) not in (None, REMOVED_OPTIONS[k])]
    if cfg.get("head") == "residual":
        used.append("head=residual")
    if used:
        raise ValueError(f"this run uses {', '.join(used)}, removed from this version of the code: "
                         f"check out the tag ultrastick-final to load it")
    if cfg.get("mode") == "mlp_direct":                              # plain MLP baseline, no latent
        from wingjepa.models.mlp_direct import MLPDirect
        return MLPDirect.from_config(cfg)
    from wingjepa.data.windows import Normalizer
    ints = lambda v, d: tuple(int(x) for x in str(v).split(",")) if v is not None else d
    return WorldModel(z_dim=cfg["z_dim"], tcn_channels=ints(cfg.get("channels"), (16, 16, 32)),
                      dilations=ints(cfg.get("dilations"), (1, 2, 4)),
                      n_ctx=5 if cfg.get("aircraft_input") else 0, head=cfg.get("head", "mlp"),
                      prior=prior or cfg.get("prior"), norm=Normalizer.from_dict(cfg["norm"]),
                      out_scale=cfg.get("out_scale", 0.1), n_sub=cfg.get("n_sub", 1),
                      act_channels=ints(cfg.get("act_channels"), None) if cfg.get("action_encoder") else None,
                      n_priv=int(cfg.get("n_priv") or 0) if cfg.get("target_priv") else 0)
