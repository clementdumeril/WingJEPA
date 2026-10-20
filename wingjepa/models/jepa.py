"""World model latent de type JEPA (docs/model_design.md) : encodeur TCN, prédicteur GRU, décodeur.

  passé : états seuls (B, H, 9) --encodeur--> z (B, D)
  (comme SkyJEPA : l'encodeur ne voit pas les commandes ni le vent, sinon le latent apprend surtout ces
  signaux lents et faciles à prédire au lieu de la dynamique de l'avion — vérifié sur un petit essai)
  z_k, entrée_k (commandes + vent tenu constant) --prédicteur--> z_{k+1}
  z_k --décodeur--> état (B, 9)
Le décodeur est la « tête » : boîte noire (modèle A) ici ; les modèles B, C, D changent la perte ou la tête.
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
    def __init__(self, n_in=16, channels=(16, 16, 32), k=3, dilations=(1, 2, 4), z_dim=32):
        super().__init__()
        blocks, c = [], n_in
        for ch, d in zip(channels, dilations):
            blocks.append(CausalBlock(c, ch, k, d))
            c = ch
        self.tcn = nn.Sequential(*blocks)
        self.out = nn.Linear(c, z_dim)
        self.receptive_field = 1 + (k - 1) * sum(dilations)

    def forward(self, past):                        # past : (B, H, n_in)
        h = self.tcn(past.transpose(1, 2))          # (B, C, H)
        return self.out(h[:, :, -1])                # dernier pas -> z


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


class WorldModel(nn.Module):
    def __init__(self, z_dim=32, tcn_channels=(16, 16, 32), k=3, dilations=(1, 2, 4), dec_hidden=64):
        super().__init__()
        self.encoder = TCNEncoder(N_STATE, tcn_channels, k, dilations, z_dim)
        self.predictor = GRUPredictor(N_CTRL + N_WIND, z_dim)
        self.decoder = MLPDecoder(z_dim, dec_hidden)

    @staticmethod
    def future_inputs(seq, H):
        """Entrées du prédicteur pour les pas futurs : commandes réellement appliquées, vent estimé
        TENU CONSTANT à sa dernière valeur connue (en usage, on ne connaît pas le vent futur)."""
        ctrl = seq[:, H - 1:-1, N_STATE:N_STATE + N_CTRL]            # commande appliquée de k à k+1
        wind = seq[:, H - 1:H, N_STATE + N_CTRL:].expand(-1, ctrl.shape[1], -1)
        return torch.cat([ctrl, wind], -1)

    def predict_latents(self, seq, H):
        """seq : (B, H+T, 16) -> latents prédits (B, T, D) pour les pas t+1..t+T."""
        z0 = self.encoder(seq[:, :H, :N_STATE])
        return self.predictor.rollout(z0, self.future_inputs(seq, H))

    def target_latents(self, seq, H):
        """Encodeur appliqué aux vraies fenêtres finissant en t+1..t+T -> (B, T, D)."""
        win = seq[:, :, :N_STATE].unfold(1, H, 1)[:, 1:]             # (B, T, 9, H)
        B, T = win.shape[:2]
        z = self.encoder(win.reshape(B * T, win.shape[2], H).transpose(1, 2))
        return z.reshape(B, T, -1)

    def n_params(self):
        return {k: sum(p.numel() for p in m.parameters())
                for k, m in [("encodeur", self.encoder), ("predicteur", self.predictor), ("decodeur", self.decoder)]}
