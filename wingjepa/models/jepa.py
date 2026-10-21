"""World model latent de type JEPA (docs/model_design.md) : encodeur TCN, prédicteur GRU, décodeur.

  passé : états seuls (B, H, 9) --encodeur--> z (B, D)
  (comme SkyJEPA : l'encodeur ne voit pas les commandes ni le vent, sinon le latent apprend surtout ces
  signaux lents et faciles à prédire au lieu de la dynamique de l'avion — vérifié sur un petit essai)
  z_k, entrée_k (commandes + vent tenu constant) --prédicteur--> z_{k+1}
  z_k --décodeur--> état (B, 9)
Le décodeur est la « tête » : boîte noire (modèle A, MLP) ou tête physique (modèle C, models/physics.py).

v2 (docs/model_design.md, § 11) : 2 s de passé (H = 40) et un encodeur qui les voit en entier (dilatations
1, 2, 4, 8, 16 : champ réceptif 63 pas) ; masse et inerties de l'avion, connues, ajoutées au résumé du
passé avant la dernière couche de l'encodeur (n_ctx = 5). v1 : H = 10, dilatations 1, 2, 4, sans avion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from wingjepa.data.windows import N_CTRL, N_STATE, N_WIND


class CausalBlock(nn.Module):
    """Convolution 1D causale dilatée + activation + connexion résiduelle."""

    def __init__(self, c_in, c_out, k, d):
        super().__init__()
        self.pad = (k - 1) * d                      # rembourrage à gauche seulement : jamais le futur
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=d)
        self.skip = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):                           # x : (B, C, H)
        y = F.gelu(self.conv(F.pad(x, (self.pad, 0))))
        return y + self.skip(x)


class TCNEncoder(nn.Module):
    def __init__(self, n_in=16, channels=(16, 16, 32), k=3, dilations=(1, 2, 4), z_dim=32, n_ctx=0):
        super().__init__()
        blocks, c = [], n_in
        for ch, d in zip(channels, dilations):
            blocks.append(CausalBlock(c, ch, k, d))
            c = ch
        self.tcn = nn.Sequential(*blocks)
        self.out = nn.Linear(c + n_ctx, z_dim)
        self.receptive_field = 1 + (k - 1) * sum(dilations)

    def forward(self, past, ctx=None):              # past : (B, H, n_in) ; ctx : (B, n_ctx), constant sur le vol
        h = self.tcn(past.transpose(1, 2))[:, :, -1]               # dernier pas
        if ctx is not None:
            h = torch.cat([h, ctx], -1)
        return self.out(h)


class GRUPredictor(nn.Module):
    def __init__(self, n_in=N_CTRL + N_WIND, z_dim=32):
        super().__init__()
        self.cell = nn.GRUCell(n_in, z_dim)

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
    """SIGReg (LeJEPA, Balestriero & LeCun 2025) : pousse la distribution des z d'un lot vers une
    gaussienne isotrope. Projections aléatoires ; pour chacune, statistique d'Epps-Pulley : écart
    entre la fonction caractéristique empirique et celle de N(0, 1), pondérée par exp(-t^2/2)."""
    N, D = z.shape
    A = torch.randn(D, n_proj, device=z.device)
    A = A / A.norm(dim=0, keepdim=True)
    t = torch.linspace(-t_max, t_max, n_knots, device=z.device)
    w = torch.exp(-0.5 * t**2)
    x = (z @ A).unsqueeze(-1) * t                   # (N, M, K)
    ecf_re, ecf_im = torch.cos(x).mean(0), torch.sin(x).mean(0)   # (M, K)
    err = ((ecf_re - w) ** 2 + ecf_im**2) * w
    return torch.trapezoid(err, t, dim=-1).mean() * N


# avion nominal (configs/ultrastick120.yaml) : les paramètres sont donnés en écart relatif à ces valeurs
NOMINAL_AIRCRAFT = (6.804, 0.672485702, 0.889416574, 1.57817209, -0.759258051)


class WorldModel(nn.Module):
    """head = "mlp" (décodeur boîte noire) ou "physics" (tête physique, prior et norm requis)."""

    def __init__(self, z_dim=32, tcn_channels=(16, 16, 32), k=3, dilations=(1, 2, 4), dec_hidden=64,
                 n_ctx=0, head="mlp", prior=None, norm=None, out_scale=0.1, n_sub=1):
        super().__init__()
        self.encoder = TCNEncoder(N_STATE, tcn_channels, k, dilations, z_dim, n_ctx)
        self.predictor = GRUPredictor(N_CTRL + N_WIND, z_dim)
        self.n_ctx, self.head = n_ctx, head
        self.register_buffer("ac_nominal", torch.tensor(NOMINAL_AIRCRAFT), persistent=False)
        if head == "mlp":
            self.decoder = MLPDecoder(z_dim, dec_hidden)
        else:
            from wingjepa.models.physics import PhysicsHead
            self.decoder = PhysicsHead(prior, norm.mean, norm.std, z_dim, dec_hidden, out_scale, n_sub)

    def ctx(self, ac):
        """Paramètres de l'avion (B, 5) -> entrée de l'encodeur : écart relatif au nominal, / 0,3."""
        if not self.n_ctx:
            return None
        return (ac / self.ac_nominal - 1) / 0.3

    @staticmethod
    def future_inputs(seq, H, hold_wind=True):
        """Entrées du prédicteur pour les pas futurs : commandes réellement appliquées, et vent estimé
        TENU CONSTANT à sa dernière valeur connue (en usage, on ne connaît pas le vent futur).
        hold_wind=False : on donne le vent de chaque pas (évaluation « vent connu »)."""
        ctrl = seq[:, H - 1:-1, N_STATE:N_STATE + N_CTRL]            # commande appliquée de k à k+1
        if hold_wind:
            wind = seq[:, H - 1:H, N_STATE + N_CTRL:].expand(-1, ctrl.shape[1], -1)
        else:
            wind = seq[:, H - 1:-1, N_STATE + N_CTRL:]
        return torch.cat([ctrl, wind], -1)

    def latents(self, seq, H, hold_wind=True, ac=None):
        """seq : (B, H+T, 16) -> z_0 (B, D), latents prédits z_1..z_T (B, T, D), entrées futures (B, T, 7)."""
        U = self.future_inputs(seq, H, hold_wind)
        z0 = self.encoder(seq[:, :H, :N_STATE], self.ctx(ac))
        return z0, self.predictor.rollout(z0, U), U

    def predict_latents(self, seq, H, hold_wind=True, ac=None):
        """seq : (B, H+T, 16) -> latents prédits (B, T, D) pour les pas t+1..t+T."""
        return self.latents(seq, H, hold_wind, ac)[1]

    def head_states(self, z0, zs, U, seq, H, ac=None):
        """États prédits normalisés (B, T, 9) et, pour la tête physique, sorties brutes de la correction."""
        if self.head == "mlp":
            return self.decoder(zs), None
        zk = torch.cat([z0[:, None], zs[:, :-1]], 1)                 # latent au début de chaque pas
        return self.decoder(zk, seq[:, H - 1, :N_STATE], U, ac)

    def predict_states(self, seq, H, hold_wind=True, ac=None):
        z0, zs, U = self.latents(seq, H, hold_wind, ac)
        return self.head_states(z0, zs, U, seq, H, ac)

    def target_latents(self, seq, H, ac=None):
        """Encodeur appliqué aux vraies fenêtres finissant en t+1..t+T -> (B, T, D)."""
        win = seq[:, :, :N_STATE].unfold(1, H, 1)[:, 1:]             # (B, T, 9, H)
        B, T = win.shape[:2]
        c = self.ctx(ac)
        c = None if c is None else c.repeat_interleave(T, 0)
        z = self.encoder(win.reshape(B * T, win.shape[2], H).transpose(1, 2), c)
        return z.reshape(B, T, -1)

    def n_params(self):
        return {k: sum(p.numel() for p in m.parameters())
                for k, m in [("encodeur", self.encoder), ("predicteur", self.predictor), ("decodeur", self.decoder)]}


def model_from_config(cfg, prior=None):
    """Reconstruit le modèle d'un run (config.json). Les runs v1 n'ont pas les clés v2 : valeurs v1."""
    from wingjepa.data.windows import Normalizer
    ints = lambda v, d: tuple(int(x) for x in str(v).split(",")) if v is not None else d
    head = cfg.get("head", "mlp")
    return WorldModel(z_dim=cfg["z_dim"], tcn_channels=ints(cfg.get("channels"), (16, 16, 32)),
                      dilations=ints(cfg.get("dilations"), (1, 2, 4)),
                      n_ctx=5 if cfg.get("aircraft_input") else 0, head=head,
                      prior=prior or cfg.get("prior"), norm=Normalizer.from_dict(cfg["norm"]),
                      out_scale=cfg.get("out_scale", 0.1), n_sub=cfg.get("n_sub", 1))
