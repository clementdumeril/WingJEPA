# Phase 0 — résultats des vérifications (15 oct. 2026)

## JSBSim

- `pip install jsbsim` (v1.3.2) fonctionne sous Windows ; données des avions incluses.
- **Vitesse** : ~92 000 pas/s sur un cœur (pas de 1/120 s) ≈ **770 s de vol simulées par seconde**.
  20 000 vols de 30 s ≈ 13 min sur un cœur.
- **Moteur coupé** : `propulsion/set-running = 0` fonctionne (c172p moteur coupé : finesse ≈ 9, réaliste).
- **Planeurs fournis** (`scripts/phase0_glider_sweep.py`, vol plané stabilisé, moyenne sur 40 s) :

| Avion | Verdict | Détail |
|---|---|---|
| **sgs126** (Schweizer 1-26, ~200 kg) | **utilisable** | comportement cohérent : profondeur cabrée → plus lent ; finesse 21–26 à 16–31 m/s ; pas d'oscillation parasite |
| minisgs (~5 kg) | inutilisable | finesses et incidences incohérentes, oscillations |
| SGS | inutilisable | finesses absurdes (jusqu'à 65), taux de chute négatif |
| sgs233 | inutilisable | crashs / états nuls |

→ **Choix proposé** : sgs126 comme planeur de référence (famille randomisée autour de lui) ;
avions motorisés moteur coupé (c172p, pa28, J3Cub…) comme **avions de test** de structure différente.
Limite : sgs126 est un planeur grandeur nature, pas un UAV ; la méthode n'en dépend pas.

## Jeu de données réel NTNU Skywalker X8 (doi:10.18710/NNHVBP)

- 106 manœuvres (53 aile propre, 53 givrée), dont longitudinales **3-2-1-1** et **doublets**.
- **~20 Hz** (gigue 0,04–0,06 s → rééchantillonner sur `TIME`), vol à ~100 m et ~18 m/s.
- Signaux utiles (longitudinal) : `Va_EKF_CG`, `alpha` (estimé par EKF, pas de girouette), `theta`, `q`,
  `height`, `u/w`, `elevator` (commande), `throttle`, `RPS`, accélérations.
- Paramètres fournis (`clean_params.json`) : masse 3,73 kg, envergure 2,1 m, corde 0,357 m, S = 0,75 m²,
  inertie (Jy = 0,14 kg·m²), modèle d'hélice (CT, CQ), calibrage des gouvernes.
- Découpage prévu par les auteurs : IDs 1–3 = entraînement, IDs ≥ 4 = validation.
- Ne pas utiliser les coefficients pré-calculés (`CX`…`CL`) : ils dépendent du modèle d'hélice des auteurs.
- Licence : CC0 sur DataverseNO, mais le ReadMe indique CC-BY 4.0 → **citer le jeu de données** dans tous les cas.

## Décisions restant à prendre (phase 1)

- État longitudinal : (V_a, γ, θ, q, h) ou (u, w, θ, q, h) ; commandes : profondeur, (moteur).
- Plages de randomisation de la famille sgs126 et avions de test.
- Les trois niveaux de prior (ajusté / générique / faussé).
- Horizons : 1 s (20 pas) à l'entraînement, jusqu'à 20 s à l'évaluation.
