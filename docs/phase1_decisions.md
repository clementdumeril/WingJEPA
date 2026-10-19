# Phase 1 — décisions pour le dataset `data_v1` (19 oct. 2026)

Objectif : un world model qui sait prédire, le plus loin possible, comment un petit avion **motorisé**
réagit à ses commandes (moteur, profondeur) et au vent, avec son **énergie**.
Conception du modèle : [model_design.md](model_design.md). Conventions : [conventions.md](conventions.md).

## L'avion : Ultra Stick 120 (University of Minnesota UAV Lab)

Seul avion trouvé qui a **à la fois** un modèle de simulation crédible écrit par d'autres **et** de vrais
vols publics (recherche du 16 oct. ; écartés : X8 « maison » = vérité écrite par nous et linéaire ;
JSBSim + Rascal = pas de vrais vols ; X8 réel = extraits de 10 s en croisière seulement).

| | Simulation | Vrais vols |
|---|---|---|
| Source | UMNUAV_sim_v4 (Simulink, BSD), avion FASER | archive UMN, avions FASER et **Baldr** (même cellule) |
| Contenu | tables de soufflerie **NASA Langley** : incidence −5 à 40°, décrochage vers 12,5°, effet de la profondeur et du souffle de l'hélice ; hélice et moteur mesurés | Baldr (2017) : 100 Hz, gaz variable, **incidence mesurée** (sondes en bout d'aile), position des gouvernes, décrochages, 4 configurations (nominale, centrage arrière, inertie de roulis, les deux) |
| Limites | pas de batterie modélisée | pas de batterie ni de vent fournis (vent recalculable) ; vrilles hors périmètre |

## Simulateur de vérité (`wingjepa/sim/ultrastick.py`)

- **Traduction en Python du simulateur UMN**, restreinte au longitudinal, sans modifier les tables :
  aéro (tables NASA, interpolation et extrapolation linéaires comme Simulink), propulsion (puissance
  moteur = polynôme du gaz, hélice C_T(J), C_P(J), vitesse de l'hélice intégrée), gouverne (1er ordre
  8 Hz, limitée en vitesse et en débattement), corps rigide. Paramètres : `configs/ultrastick120.yaml`.
- **Validation** (`scripts/validate_ultrastick.py`) : sur l'Ultra Stick 25e, notre traduction reproduit la
  trajectoire de référence de leur simulateur Simulink (doublet de profondeur) : erreur RMS sur q de
  0,0003 rad/s pour une réponse de 0,22, sur l'incidence de 5·10⁻⁵ rad. Mêmes équations de mouvement et
  de propulsion que FASER.
- **Écarts au simulateur UMN, documentés** :
  1. **effet du souffle de l'hélice actif** : UMN fixait J = 1 (saturé à 0,7, point où les tables de
     puissance valent zéro) ; on utilise le vrai J ;
  2. **une cellule corrompue** des tables (incidence −5°, profondeur −10°, J = 0,389 : dCL = 4,35 au lieu
     de ~−0,05) remplacée par l'extrapolation de ses voisines ; UMN ne l'atteignait jamais ;
  3. **propulsion de FASER** : la bibliothèque Simulink code en dur le moteur et l'hélice du petit 25e ; on
     suit les valeurs que le fichier de configuration de FASER déclare (hélice de rayon 0,1778 m,
     puissance ×6) — sans cela l'avion ne peut pas tenir le palier.
- **Vent** (comme UMN) : le vent moyen ne fait que déplacer l'avion ; rafales (1 − cos) et turbulence de
  Dryden (MIL-F-8785C) entrent dans l'aérodynamique.
- Le **prior** du world model est bien plus simple que cette vérité (aéro linéaire, polaire
  $C_{D_0} + K C_L^2$, poussée simplifiée, sans décrochage ni souffle d'hélice).

## Ce qu'on fait varier

**D'un vol à l'autre — l'avion** (tirages uniformes) :

| Paramètre | Entraînement | Avions de test |
|---|---|---|
| Masse (6,8 kg) | ±20 % | ±20–35 % |
| Inertie de tangage | ±30 % | ±30–45 % |
| Centrage | ±3 % de corde | ±3–5 % |
| Coefficients CD, CL, Cm (facteurs indépendants) | ±20 % | ±20–30 % |
| Puissance moteur | ±30 % | ±30–45 % |

**Pendant un vol — les commandes**, quatre types mélangés : vols stabilisés ; manœuvres d'excitation
(2-1-1, 3-2-1-1, doublets) sur la profondeur **et** le moteur ; consignes aléatoires d'altitude et de
vitesse suivies par un **autopilote TECS** ; commandes aléatoires lissées. Des segments **moteur coupé**
(plané). Vitesses de consigne 15–28 m/s (décrochage vers 12 m/s), passages près du décrochage.

**Le vent** : moyen jusqu'à 10 m/s (horizontal) et 1,5 m/s (vertical), rafales jusqu'à 3 m/s, turbulence
(aucune, légère, modérée, forte). Le world model reçoit le vent **estimé** ; le vrai vent est enregistré.

## Volume et format

- **Entraînement / validation** : 2 000 vols de 30 s à 20 Hz, découpés **par vol** (80 / 10 / 10).
- **Test long horizon** : 200 vols de 120 s, moitié avions d'entraînement, moitié avions de test.
- On augmente ensuite selon les résultats (la taille des données est un axe d'étude).
- Variables : voir [conventions.md](conventions.md).

## Premier essai local (19 oct., 40 vols)

35/40 vols restent dans l'enveloppe (les sorties : décrochages pendant des manœuvres) ; bilan d'énergie
rebouclé (corrélation 0,9985) ; signes des commandes corrects ; 5 vols passent au-delà de 12° d'incidence ;
~1 min de calcul pour 40 vols. À surveiller sur le lot Savio : la part du temps moteur coupé et
l'amplitude des oscillations d'assiette (±25° dans certains planés).

## Exécution

Génération sur **Savio** : `cluster/smoke_test.sbatch` (40 vols + contrôles), puis tableau de jobs
`cluster/gen_data.sbatch` (un paquet de vols par job, une graine par job).

## Encore ouvert

- Seuils de l'horizon utile (proposé : 5 m d'altitude ou 1 m/s de vitesse air).
- Réglages fins de l'autopilote et proportions des types de commandes, après le lot Savio.
- Pipeline des vrais vols Baldr (mêmes variables que la simulation).
