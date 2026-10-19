# Phase 1 — décisions pour le dataset `data_v1` (19 oct. 2026, 6 degrés de liberté)

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

- **Traduction en Python du simulateur UMN, en 6 degrés de liberté**, sans modifier les tables : aéro
  (tables NASA [CD, CY, CL, Cl, Cm, Cn] en fonction de l'incidence, du dérapage, des trois gouvernes et du
  souffle de l'hélice, interpolation et extrapolation linéaires comme Simulink), propulsion (puissance
  moteur = polynôme du gaz, hélice C_T(J), C_P(J), vitesse de l'hélice intégrée, couple moteur et effets
  gyroscopiques), gouvernes (1er ordre 8 Hz, limitées en vitesse et en débattement), corps rigide 6DoF
  (angles d'Euler, Terre plate). Paramètres : `configs/ultrastick120.yaml`.
- **Validation** (`scripts/validate_ultrastick.py`) contre les trajectoires de référence de leur
  simulateur Simulink (Ultra Stick 25e, mêmes équations de mouvement et de propulsion que FASER) :
  doublets de profondeur, de direction et d'ailerons ; erreurs RMS : p 0,0007 rad/s, q 0,001 rad/s,
  r 0,0013 rad/s pour des réponses de 0,2 à 0,45 rad/s ; dérapage v 0,004 m/s pour 2,4 m/s.
- **Écarts au simulateur UMN, documentés** :
  1. **effet du souffle de l'hélice actif** : UMN fixait J = 1 (saturé à 0,7, point où les tables de
     puissance valent zéro) ; on utilise le vrai J ;
  2. **une cellule corrompue** des tables (incidence −5°, profondeur −10°, J = 0,389 : dCL = 4,35 au lieu
     de ~−0,05) remplacée par l'extrapolation de ses voisines ; UMN ne l'atteignait jamais ;
  3. **propulsion de FASER** : la bibliothèque Simulink code en dur le moteur et l'hélice du petit 25e ;
     on suit les valeurs déclarées pour FASER (hélice de rayon 0,1778 m, puissance ×6) — sans cela
     l'avion ne peut pas tenir le palier ;
  4. **signe de la force latérale** : le bloc Simulink des forces aérodynamiques inverse CY ; le modèle du
     25e compense (gain −1, vérifié : sans lui, les erreurs latérales sont 25 à 30 fois plus grandes),
     le mode FASER non, ce qui donnait une force latérale déstabilisante. Les tables NASA suivent la
     convention physique (CY < 0, Cn > 0, Cl < 0 pour un dérapage positif) : on l'utilise.
- **À vérifier sur les vrais vols** : l'inertie croisée de FASER (Ixz = −0,76 pour Ixx = 0,67) est
  inhabituellement grande ; elle couple fortement roulis et lacet. Gardée telle quelle.
- **Vent** (comme UMN) : le vent moyen (direction quelconque) ne fait que déplacer l'avion ; rafales
  (1 − cos) et turbulence de Dryden (MIL-F-8785C, u, v, w) entrent dans l'aérodynamique.
- Le **prior** du world model est bien plus simple que cette vérité (dérivées de stabilité linéaires de
  type Beard & McLain, sans décrochage ni souffle d'hélice).

## Ce qu'on fait varier

**D'un vol à l'autre — l'avion** (tirages uniformes) :

| Paramètre | Entraînement | Avions de test |
|---|---|---|
| Masse (6,8 kg) | ±20 % | ±20–35 % |
| Inerties Ixx, Iyy, Izz (Ixz suit) | ±30 % | ±30–45 % |
| Centrage | ±3 % de corde | ±3–5 % |
| Coefficients CD, CY, CL, Cl, Cm, Cn (facteurs indépendants) | ±20 % | ±20–30 % |
| Puissance moteur | ±30 % | ±30–45 % |

**Pendant un vol — les commandes** (profondeur, ailerons, direction, gaz) :
- longitudinal, quatre types : vols stabilisés ; manœuvres d'excitation (2-1-1, 3-2-1-1, doublets) sur
  les quatre commandes ; consignes aléatoires d'altitude et de vitesse suivies par un **autopilote TECS** ;
  commandes aléatoires lissées ; segments **moteur coupé** (plané) ;
- latéral, trois modes : ligne droite ; virages et cercles (inclinaison 10–40°) ; virages en S ;
  un pilote latéral tient l'inclinaison (ailerons) et coordonne le virage (direction) ; l'inclinaison
  est limitée à basse vitesse ;
- **protection d'enveloppe** (comme un autopilote) : à partir des données air réelles lissées sur 0,5 s,
  si la vitesse sort de [13, 33] m/s, l'incidence de [−5°, 12°], l'inclinaison dépasse 60°, l'assiette
  45°, le dérapage 20° ou l'altitude descend sous 60 m, la manœuvre en cours est annulée et l'autopilote
  reprend la main 3 s (ailes à plat, ≥ 20 m/s). En boucle fermée, un **limiteur d'incidence** (−4° à 11°)
  agit sur la profondeur. La protection change le pilote, pas la physique : ses commandes sont
  enregistrées comme les autres. Elle évite les pertes de contrôle hors du domaine de validité des tables
  et donne des vols longs complets, nécessaires pour mesurer l'horizon de prédiction.
- **Sortie d'enveloppe** (vol arrêté, figé sur son dernier état valide) : vitesse hors [5, 45] m/s,
  incidence hors [−15°, 35°] (sous −5°, prolongement linéaire des tables), dérapage > 25°,
  inclinaison > 75°, assiette > 80°, altitude < 20 m.

**Le vent** : moyen jusqu'à 10 m/s (direction quelconque) et 1,5 m/s vertical, rafales jusqu'à 3 m/s par
axe, turbulence (aucune, légère, modérée, forte). Le world model reçoit le vent **estimé** ; le vrai vent
est enregistré.

## Volume et format

- **Entraînement / validation** : 2 000 vols de 30 s à 20 Hz, découpés **par vol** (80 / 10 / 10).
- **Test long horizon** : 200 vols de 120 s, moitié avions d'entraînement, moitié avions de test.
- On augmente ensuite selon les résultats (la taille des données est un axe d'étude).
- Variables : voir [conventions.md](conventions.md).

## Essais (local, 6DoF)

- 40 vols de 30 s : **40/40 complets**, protection active 5,5 % du temps, 8 vols au-delà de 12°
  d'incidence, bilan d'énergie 6DoF rebouclé (corrélation 0,997), signes des quatre commandes corrects.
- 24 vols de 120 s (moitié avions de test) : **23/24 complets** (66 % avant la protection), protection 7,5 %
  du temps, changement de cap médian 210° par vol.
- Réglages issus de ces essais : excitations latérales réduites, gains du pilote latéral, protection
  sur données air lissées (sans quoi les rafales donnaient de fausses alarmes), limite basse d'incidence
  à −15° (les pointes de turbulence coupaient des vols normaux).

## Exécution

Génération sur **Savio** : `cluster/smoke_test.sbatch` (40 vols + contrôles), puis tableau de jobs
`cluster/gen_data.sbatch` (un paquet de vols par job, une graine par job).

## Encore ouvert

- Seuils de l'horizon utile (proposé : 5 m d'altitude ou 1 m/s de vitesse air).
- Réglages fins de l'autopilote et proportions des types de commandes, après le lot Savio.
- Pipeline des vrais vols Baldr (mêmes variables que la simulation).
