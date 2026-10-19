# Conventions (brouillon, à valider avant d'écrire le générateur)

Règle : **tout le code interne est en SI et en radians**. Les degrés ne servent qu'à l'affichage et aux
entrées des tables NASA (converties dans `wingjepa/sim/ultrastick.py`).

## Repères

| Repère | Définition |
|---|---|
| Inertiel | **NED** : x Nord, y Est, z vers le bas. Altitude $h = -p_D$ |
| Avion | **FRD** : x vers le nez, y aile droite, z vers le bas |
| Longitudinal | on ne garde que le plan vertical de la trajectoire : axes x et z du repère avion |

## Variables et signes

| Symbole | Nom | Unité | Signe positif |
|---|---|---|---|
| $u, w$ | vitesse par rapport au **sol**, axes avion | m/s | $u$ vers l'avant, $w$ vers le bas |
| $q$ | vitesse de tangage | rad/s | nez qui monte |
| $\theta$ | assiette | rad | nez au-dessus de l'horizon |
| $h$ | altitude | m | vers le haut |
| $u_r, w_r$ | vitesse par rapport à **l'air** : $(u, w) - $ vent exprimé en axes avion | m/s | idem $u, w$ |
| $V_a$ | vitesse air $\sqrt{u_r^2 + w_r^2}$ | m/s | — |
| $\alpha$ | incidence $\operatorname{atan2}(w_r, u_r)$ | rad | vent relatif venant du dessous |
| $\gamma_a$ | pente air $\theta - \alpha$ | rad | montée |
| $e$ | énergie spécifique $h + V_a^2 / 2g$ | m | — |
| $\delta_e$ | profondeur | rad | **bord de fuite vers le bas** (moment piqueur, $C_{m_{\delta_e}} < 0$), convention Beard & McLain |
| $\delta_t$ | moteur | sans unité, $[0, 1]$ | 0 = coupé |
| $w_N, w_D$ | vent, repère inertiel (horizontal dans l'axe de vol, vertical) | m/s | $w_D > 0$ = **descendant** ; ascendance = $w_D < 0$ |

Ultra Stick 120 (convention UMN) : profondeur positive = bord de fuite vers le bas (piqueur) ;
ailerons da = (da_droit − da_gauche)/2, positif = roulis à gauche ; direction positive = bord de fuite
à gauche (lacet à gauche) ; débattements ±25°. Angles d'Euler phi (roulis, aile droite basse positive),
theta, psi (cap). Dérapage beta = asin(v / Va), positif = vent relatif venant de la droite.

## Temps

- Simulation : RK4 à 100 Hz.
- Dataset et world model : **20 Hz** ($\Delta t = 0{,}05$ s), échantillonnage sans moyennage.
- Commandes : constantes par morceaux sur chaque pas de 0,05 s.

## Format des données (proposition)

- Un fichier `.npz` par lot de vols + un `meta.json` par lot.
- Par vol, séries temporelles alignées (longueur N) :
  - état 6DoF : `u, v, w` (axes avion, par rapport à la masse d'air moyenne), `p, q, r`, `phi, theta, psi`,
    `north, east, h` ; air : `Va, alpha, beta, gamma_a, e_spec` ;
  - commandes : `de_cmd, da_cmd, dr_cmd, dt_cmd` ; positions des gouvernes `de, da, dr` ; hélice `omega_rpm, J` ;
  - vent : `wind_n, wind_e, wind_d` (moyen + rafales, NED), `turb_u, turb_v, turb_w` (turbulence, axes avion),
    `wind_n_est, wind_e_est, wind_d_est` (estimé) ;
  - vérité : forces et moments aéro `Fx/Fy/Fz_aero, Mx/My/Mz_aero`, moments de l'hélice `Mx/My/Mz_prop`,
    `lift, drag, side, CD, CY, CL, Cl, Cm, Cn, thrust`, puissance `P_motor, P_shaft`, énergie `E_motor` ;
  - pilote : `phi_ref` ; indicateurs `engine_off, open_loop, protect, alive`.
  Liste exacte : `CHANNELS` dans `scripts/generate_flights.py`.
- Par vol, métadonnées : paramètres de l'avion tirés (masse, inertie, centrage, coefficients),
  paramètres du vent et de la turbulence, type de manœuvre, graine aléatoire.
- Découpage entraînement / validation / test **par vol et par avion**, jamais par fenêtre, pour éviter les fuites.
