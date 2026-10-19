# Conception du world model (brouillon, à figer en Phase 2)

Question du projet : **où faire entrer la physique dans un world model latent**, et que se passe-t-il
quand cette physique est fausse ?

Les valeurs numériques ci-dessous sont des **points de départ** repris de SkyJEPA (arXiv 2606.23444),
à ajuster. Les points marqués « à vérifier » ne sont pas encore tranchés.

---

## 1. Vocabulaire

| Terme | Ce que c'est ici |
|---|---|
| **Simulateur (Ultra Stick 120, tables NASA)** | joue le rôle de **la réalité** : on n'y touche pas, il produit les données et la vérité (états, forces) |
| **World model** | le réseau qu'on entraîne : passé + commandes futures + vent → trajectoire future |
| **JEPA** | la **méthode d'entraînement** : prédire dans un espace latent, pas directement l'état |
| **Prior** | des équations d'avion **simplifiées**, placées **dans le world model** (versions B, C, D) |
| **Contrôleur** | hors du world model : il essaie des séquences de commandes *dans* le world model et choisit (MPPI) |

Pendant l'entraînement, les poids changent ; pendant la prédiction, ils sont **figés** (pas d'apprentissage en ligne).

---

## 2. Quelle JEPA

Une **JEPA conditionnée par les commandes, sur des états** (comme SkyJEPA et PLDM, arXiv 2502.14819),
avec la régularisation anti-effondrement **SIGReg** de LeJEPA. Pas d'images dans le cœur du projet
(voir § 8).

---

## 3. Entrées, sorties, notations

Pas de temps $\Delta t = 0{,}05$ s (20 Hz). Les données sont en **6 degrés de liberté** (virages compris) ;
les modèles commencent par la dynamique **longitudinale** (ci-dessous), puis passent au 6DoF sur les mêmes données.

- **État physique** $x = (u, w, q, \theta, h)$ : vitesses dans le repère avion (sol), vitesse de tangage,
  assiette, altitude.
- **Grandeurs dérivées** : vitesse air $V_a$, incidence $\alpha$, pente $\gamma$, énergie spécifique
  $e = h + V_a^2 / 2g$.
- **Commandes** $a = (\delta_e, \delta_t)$ : profondeur, moteur.
- **Vent** $w_{\text{vent}}$ (horizontal dans l'axe + vertical), donné en entrée. Deux réglages
  d'évaluation : **vent exact** (mesure l'erreur du modèle seule) et **vent estimé** (réaliste).
- **Passé** : fenêtre de $H = 10$ pas (0,5 s) d'états, commandes et vent.
- **Altitude** : l'encodeur ne reçoit **pas** l'altitude absolue (la dynamique n'en dépend presque pas) ;
  elle est reconstruite par intégration (C, D) ou prédite en variation (A, B).

Sortie du world model : la trajectoire $\hat x_{t+1}, \dots, \hat x_{t+T}$ et l'énergie $\hat e$, pour une
séquence de commandes donnée. $T = 20$ (1 s) à l'entraînement, éventuellement allongé progressivement
jusqu'à 5 s ; jusqu'à 60 s en évaluation.

---

## 4. Tronc commun (identique pour toutes les versions)

```
 passé 0,5 s (état, commandes, vent)
              │
         ENCODEUR E   (petit réseau temporel, TCN)
              │
             z_t      (vecteur latent, ~16–24 dimensions, non précisé dans SkyJEPA)
              │
   ┌──► PRÉDICTEUR P (GRU) ◄── commande a_k + vent du pas k
   │          │
   └──────  ẑ_{k+1}         (boucle : le modèle ne revoit jamais la réalité)
              │
         TÊTE D       ──► état physique x̂_k  (c'est la tête qui change entre A, B, C, D)
```

**Loss commune**, sur un rollout de $T$ pas :

$$
\mathcal L = \underbrace{\sum_{k=1}^{T} \lVert \hat z_{t+k} - E(\text{fenêtre réelle finissant à } t+k) \rVert^2}_{\text{JEPA}}
+ \lambda_{\text{SIG}}\,\text{SIGReg}(z)
+ \lambda_x \underbrace{\sum_{k=1}^{T} \lVert D(\hat z_{t+k}) - x_{t+k} \rVert^2}_{\text{état décodé}}
\;(+\ \text{termes propres à B, C, D})
$$

- **Ce que fait SkyJEPA** (vérifié, arXiv 2606.23444v2, § IV–V) : cible = **même encodeur**, pas d'EMA
  (LeJEPA : SIGReg suffit contre l'effondrement, sans stop-gradient) ; loss JEPA = L2 au carré moyennée
  sur le rollout + $\lambda_{\text{SIG}} = 0{,}02$ × SIGReg sur les latents prédits ; **aucun terme
  physique ni de reconstruction dans l'étape JEPA**. La sonde physique est entraînée **ensuite**, sur
  le modèle **figé** (stop-gradient sur les latents). ~9 k paramètres au total ; TCN [8,8,16] (états),
  [4,4,8] (actions), GRU 24, H = 10, T = 20, Adam, lot 2048.
- **Décision à prendre (Phase 2)** : entraîner la tête **après coup sur un latent figé** (comme SkyJEPA :
  la physique ne façonne pas le latent) ou **conjointement** ($\lambda_x > 0$ ci-dessus : la physique
  façonne le latent). C'est une dimension naturelle de la question « où entre la physique » ; on peut
  tester les deux pour C.
- **Budget de paramètres identique** pour toutes les versions (ordre de grandeur de SkyJEPA, ~10 k),
  même optimiseur, mêmes données, même nombre de pas d'entraînement, 3 graines.

---

## 5. Les versions comparées

### A — boîte noire
Tête $D$ = petit MLP : $\hat x_k = \text{MLP}(\hat z_k)$. Aucune physique.

### B — la physique « juge » (dans la loss)
Même réseau que A. On ajoute des **pénalités** quand les prédictions violent les équations du § 6 :

$$
\mathcal L_B = \lambda_{\text{phys}} \sum_k \Big(
\big\lVert \tfrac{\hat h_{k+1}-\hat h_k}{\Delta t} - (\hat u \sin\hat\theta - \hat w \cos\hat\theta) \big\rVert^2
+ \big\lVert \tfrac{\hat\theta_{k+1}-\hat\theta_k}{\Delta t} - \hat q \big\rVert^2
+ \big\lVert \tfrac{\hat u_{k+1}-\hat u_k}{\Delta t} - f_u^{\text{prior}}(\hat x_k, a_k) \big\rVert^2 + \dots \Big)
$$

Le réseau peut enfreindre ces règles s'il y gagne sur le reste de la loss.

### C — la physique « construit » (dans l'architecture)
À chaque pas, la tête :
1. calcule les **forces du prior** (§ 6) à partir de l'état prédit $\hat x_k$, des commandes et du vent ;
2. ajoute une **correction apprise** lue dans le latent : $\Delta F_x, \Delta F_z, \Delta M = \text{MLP}(\hat z_k, \hat x_k, a_k)$ ;
3. **fait avancer l'état avec les équations de Newton** (§ 6, intégrateur à sous-pas).

L'état physique est donc porté par l'intégrateur tout au long du rollout ; le réseau ne peut pas sortir
de la physique, il apprend seulement ce que le prior décrit mal (décrochage, erreurs de coefficients, hélice…).
Petite pénalité $\lambda_{\text{res}} \lVert \Delta F \rVert^2$ pour que le prior fasse le maximum
(idée d'APHYNITY). Équivalent avion de la « sonde physique » de SkyJEPA.

### C0 — contrôle d'équité
Même architecture et même nombre de paramètres que C, **sans prior** (forces du prior mises à zéro, la
correction apprend tout). Si C bat C0, c'est la physique qui aide, pas la forme du réseau.

### D — la physique par l'énergie
Comme C, mais l'état avance sous une forme qui **garantit le bilan d'énergie** (§ 7) :
pertes apprises **forcées positives**, apport du moteur et échange avec le vent explicites.
Forme exacte (port-hamiltonienne) à dériver en Phase 2 — réf. Fahmi & Woolsey.

### Optionnel — baseline sans JEPA
Même réseau que A, entraîné **uniquement** sur l'état (pas de loss latente), pour vérifier que la JEPA
apporte quelque chose (la baseline « prédictive » de SkyJEPA).

### Où entre la qualité du prior
B, C et D utilisent les coefficients du prior à **trois niveaux** :
- **ajusté** : coefficients identifiés sur l'avion de base, presque justes ;
- **générique** : coefficients « de manuel » pour un petit UAV, moyennement faux ;
- **faussé** : coefficients volontairement décalés d'environ 30 %.

A et C0 n'ont pas de prior, ils ne sont pas concernés.

---

## 6. Le prior longitudinal (Beard & McLain, ch. 4)

Vitesse air dans le repère avion : $u_r = u - u_{\text{vent}}$, $w_r = w - w_{\text{vent}}$,
$V_a = \sqrt{u_r^2 + w_r^2}$, $\alpha = \operatorname{atan2}(w_r, u_r)$, $\bar q = \tfrac12 \rho V_a^2$.

Coefficients aéro (linéaires en $\alpha$, polaire quadratique pour la traînée) :

$$
C_L = C_{L_0} + C_{L_\alpha}\alpha + C_{L_q}\tfrac{c}{2V_a}q + C_{L_{\delta_e}}\delta_e,
\qquad
C_D = C_{D_0} + K\,C_L^2,
\qquad
C_m = C_{m_0} + C_{m_\alpha}\alpha + C_{m_q}\tfrac{c}{2V_a}q + C_{m_{\delta_e}}\delta_e
$$

Forces et moment (repère avion) :

$$
F_x = \bar q S\,(-C_D\cos\alpha + C_L\sin\alpha) + T, \qquad
F_z = \bar q S\,(-C_D\sin\alpha - C_L\cos\alpha), \qquad
M = \bar q S c\, C_m
$$

Poussée : $T = f_T(\delta_t, V_a)$, modèle d'hélice simple (forme exacte à fixer avec l'avion de base ;
le jeu X8 fournit un modèle $C_T(J)$).

Équations du mouvement :

$$
\dot u = -q w - g\sin\theta + F_x/m, \quad
\dot w = q u + g\cos\theta + F_z/m, \quad
\dot q = M / J_y, \quad
\dot\theta = q, \quad
\dot h = u\sin\theta - w\cos\theta
$$

Ce que le prior **ignore** (et que le simulateur contient) : décrochage et non-linéarités en $\alpha$,
effets de vitesse sur les coefficients, hélice réelle, retards des gouvernes et du moteur, etc.
C'est ce que la correction apprise doit capturer — et ce qu'on peut vérifier, puisque le simulateur
enregistre les **vraies forces**.

---

## 7. Bilan d'énergie (pour D et pour l'évaluation)

Énergie mécanique (repère sol) $E = \tfrac12 m (u^2 + w^2) + \tfrac12 J_y q^2 + m g h$ :

$$
\dot E = \underbrace{T\,u}_{\text{moteur}} \;-\; \underbrace{D\,V_a}_{\text{pertes}} \;+\; \underbrace{F_{\text{aéro}} \cdot w_{\text{vent}}}_{\text{échange avec le vent}} \;+\; M q
$$

En air calme, le dernier terme lié au vent disparaît. Ce bilan sert aussi de **contrôle de sanité** du
dataset (on le reboucle avec les vraies forces enregistrées).

---

## 8. Évaluation

- **Horizon utile** : temps avant que l'erreur dépasse **5 m en altitude ou 1 m/s en vitesse air**
  (seuils à confirmer). C'est le chiffre phare.
- Erreurs à 1, 5, 20 et 60 s ; erreur sur l'énergie.
- Sur des **vols jamais vus** et des **avions de test** hors des plages d'entraînement.
- Deux réglages de vent : exact / estimé.
- Facteurs : version (A, B, C0, C, D) × qualité du prior (3) × taille des données × graines (3).

---

## 9. Extensions possibles (hors cœur)

- **Images en entrée** : le simulateur ne fait pas de rendu, mais les images peuvent être **générées après
  coup à partir des trajectoires enregistrées** (rendu synthétique de l'horizon et du sol, ou FlightGear). Seul l'encodeur change ; question : la tête physique aide-t-elle encore quand l'état
  doit être extrait des pixels ?
- Vrais vols d'un autre avion (Skywalker X8, NTNU), dynamique 6-DoF, adaptation en ligne au vent.
