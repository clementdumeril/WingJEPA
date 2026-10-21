# Journal du projet WingJEPA

Récapitulatifs successifs : ce qui a été fait, testé, trouvé, et vers quoi on va.
Détails techniques : `phase1_decisions.md` (simulateur, données), `model_design.md` (modèles),
`conventions.md` (variables, repères).

---

## 1. Objectif

Entraîner un **world model** : un réseau qui prédit comment un avion va bouger dans les secondes qui
viennent, en fonction des commandes. Question du projet : **où mettre la physique** dans ce réseau
(nulle part, dans la perte, dans l'architecture), surtout quand l'avion est différent de ceux vus à
l'entraînement.

## 2. Historique des choix (14-20 oct. 2026)

- **Phase 0** : vérifications JSBSim, jeu de données réel du Skywalker X8, squelette du dépôt, Savio.
- **Choix de l'avion**, après plusieurs pistes :
  - X8 dans un simulateur écrit par nous : écarté (vérité écrite par nous, modèle publié linéaire) ;
  - JSBSim + Rascal (avion du SITL PX4) : écarté (aucun vrai vol de cet avion) ;
  - **Ultra Stick 120** (University of Minnesota) : seul avion trouvé avec **à la fois** un modèle de
    simulation crédible écrit par d'autres (tables de soufflerie NASA Langley) et de **vrais vols publics**
    (avions FASER et Baldr). Retenu.
- **Simulateur** : traduction en Python, en 3D (6 degrés de liberté), du simulateur Simulink de l'UMN.
  Validé contre leurs trajectoires de référence (erreurs ~0,1 à 1 % de la réponse).
- **Problèmes trouvés dans l'original et corrigés** (documentés) :
  1. une cellule corrompue des tables NASA (portance 4,35 au lieu de ~-0,05) ;
  2. la force latérale de mauvais signe en mode FASER — confirmé sur un vrai vol Baldr (la force
     s'oppose au dérapage, pente mesurée -0,0054 à -0,0059 /deg contre -0,0068 pour la table NASA) ;
  3. la propulsion de FASER codée avec le moteur du petit 25e ;
  4. l'effet du souffle de l'hélice désactivé (J figé) : réactivé.
- **Génération des vols** : autopilote TECS, pilote latéral (virages, cercles, virages en S), manœuvres
  d'excitation, commandes aléatoires, moteur coupé, vent + rafales + turbulence de Dryden.
  Ajouts après essais : **protection d'enveloppe** (l'autopilote reprend la main près du décrochage,
  en piqué, en ressource, près du sol), limiteur d'incidence, données air lissées pour éviter les
  fausses alarmes, limite basse d'incidence à -15°.

## 3. Le jeu de données `data_v1` (Savio)

| Set | Contenu | Usage |
|---|---|---|
| **train** | 2 000 vols de 30 s, découpés 80 % / 10 % / 10 % | entraînement / choix du modèle / test |
| **test_long** | 200 vols de 120 s : moitié avions « comme à l'entraînement », moitié **avions de test extrêmes** (masse, inertie, aéro hors des plages d'entraînement) | long terme et généralisation |

Contrôles : 0 valeur aberrante, 98,8 % des vols courts et 93,5 % des vols longs complets, bilan
d'énergie rebouclé (corrélation 0,998), signes des quatre commandes corrects, 205 vols au-delà de 12°
d'incidence (zone de décrochage), inclinaisons jusqu'à ±38°.

## 4. Les modèles (modèle A et référence A0, 20 oct. 2026)

Même réseau pour tous (~15 500 paramètres) : **encodeur** TCN (résume les 0,5 s passées, états seuls) ->
**prédicteur** GRU (avance ce résumé pas à pas avec les commandes et le vent estimé) -> **décodeur** MLP
(traduit en grandeurs physiques). Seule change la façon de l'entraîner :

| Modèle | Entraînement | Horizon d'entraînement |
|---|---|---|
| A0, 1 s | prédit directement l'état physique (pas de JEPA) | 1 s |
| JEPA, 1 s | JEPA (prédit dans le latent), décodeur appris après, latent figé | 1 s |
| A0, 5 s | comme A0 | de 1 à 5 s progressivement |
| JEPA, 5 s | comme JEPA | de 1 à 5 s progressivement |
| Joint, 5 s | JEPA + erreur d'état ensemble, décodeur non figé | de 1 à 5 s progressivement |

Corrections en cours de route : l'encodeur JEPA voyait d'abord commandes et vent et son latent apprenait
ces signaux faciles plutôt que la dynamique (encodeur sur les états seuls depuis) ; SIGReg estimé sur un
échantillon (coût de calcul) ; évaluation séparée par niveau de turbulence et variante « vrai vent ».

## 5. Comment on mesure

On prend un instant d'un vol jamais vu, on donne au modèle **0,5 s de passé réel**, puis il prédit
**seul**, pas à pas (0,05 s), en ne recevant que les commandes réellement appliquées et le vent estimé ;
il ne revoit jamais la réalité. À chaque pas on compare à la vérité.

- **Horizon utile** = premier instant où il décroche : erreur d'altitude > 5 m, de vitesse > 1 m/s,
  ou d'inclinaison / d'assiette > 10°. On répète depuis des centaines de départs (toutes les 2,5 s ;
  386 prédictions sur les vols de test) et on prend la **médiane**. « 3,70 s » = dans la moitié des cas,
  le modèle reste précis au moins 3,7 s sans correction.
- **Erreur d'altitude à 60 s** : erreur typique (RMS) après 60 s de prédiction seule : **dérive à long terme**.
- **Référence** : « rien ne change » (l'avion garde son état actuel, vitesse verticale comprise).

## 6. Résultats (une seule graine par modèle)

Horizon utile médian (plus = mieux) :

| | Référence | A0 1 s | JEPA 1 s | **A0 5 s** | JEPA 5 s | Joint 5 s |
|---|---|---|---|---|---|---|
| Vols de test (tous) | 1,70 | 2,58 | 1,75 | **3,70** | 1,40 | 2,25 |
| Vols de test, **air calme** | 2,17 | 2,98 | 2,45 | **5,98** | 3,28 | 3,62 |
| Vols longs, avions d'entraînement | 1,50 | 2,40 | 1,85 | **3,55** | 1,23 | 2,00 |
| Vols longs, **avions de test extrêmes** | 1,55 | 1,75 | 1,20 | **1,80** | 0,75 | 1,10 |

Erreur d'altitude à 60 s (moins = mieux) :

| | Référence | A0 1 s | JEPA 1 s | A0 5 s | **JEPA 5 s** | Joint 5 s |
|---|---|---|---|---|---|---|
| Avions d'entraînement | 145 m | 88 m | 185 m | 150 m | **70 m** | 87 m |
| Avions de test extrêmes | 185 m | 192 m | 342 m | 300 m | **163 m** | 233 m |

## 7. Ce qu'on a appris

1. **Entraîner sur 5 s au lieu de 1 s aide beaucoup** (A0 : de 2,6 à 3,7 s, jusqu'à 6 s en air calme).
2. **Le meilleur à court terme est A0** (sans JEPA) — l'inverse de SkyJEPA.
3. **JEPA est le meilleur sur l'altitude à long terme**, mais mauvais à court terme.
4. **Aucun modèle ne généralise aux avions extrêmes** : à peine mieux que « rien ne change ». C'est là
   que la physique devrait aider (cœur de la question du projet).
5. **Connaître le vrai vent ne change rien** : les modèles semblent ignorer l'entrée vent.
6. **Prudence** : une seule graine par modèle ; il en faut 3 avant de conclure.

## 8. Questions ouvertes et suite (20 oct. 2026)

- Pourquoi A0 est meilleur à court terme que JEPA ? Pourquoi JEPA est meilleur sur l'altitude à long
  terme ? Pourquoi aucun modèle ne généralise aux avions extrêmes ? -> script de diagnostic
  (`scripts/diagnose.py`).
- 3 graines pour A0 5 s et JEPA 5 s.
- Concevoir le modèle C (décodeur physique : équations du mouvement + correction apprise).
- Vérifier pourquoi le vent est ignoré.

## 9. Diagnostic des modèles (21 oct. 2026, `scripts/diagnose.py`, une graine)

**Q1. Pourquoi A0 est meilleur à court terme que JEPA ?**
- Sonde (état présent relu dans le latent z(t), 0 = parfait, 1 = rien) : A0 <= 0,02 partout ; JEPA
  bon sur Va, phi, theta, vitesse verticale (0,01-0,02), moyen sur alpha, p, q (0,05-0,09), r (0,13),
  **mauvais sur le dérapage beta (0,46)** ; joint entre les deux (beta 0,28).
- Dès 0,25 s, l'erreur de JEPA est 2 à 4 fois celle d'A0 (ex. phi 0,21 contre 0,05), bien au-dessus de
  ce que la sonde perd : la perte vient surtout du **prédicteur**, entraîné à bien prédire le latent, pas
  l'état.
- Explication : (1) JEPA optimise une erreur dans le latent, A0 optimise directement l'erreur sur l'état
  qu'on mesure ; (2) JEPA garde ce qui est prévisible et jette le reste : le dérapage, très agité par la
  turbulence, est en partie jeté.

**Q2. Pourquoi JEPA est « meilleur » sur l'altitude à long terme ?**
- Nouvelle référence « altitude constante » : **36 m à 60 s, meilleure que tous les modèles** (A0 150 m,
  JEPA 70 m, joint 87 m). Les vols restent autour de leurs consignes d'altitude (pilote automatique), donc
  l'altitude revient vers sa moyenne. **Aucun modèle ne bat cette référence triviale à 60 s** : le long
  terme n'est pas résolu.
- JEPA n'est pas plus « prudent » (dispersion de la vitesse verticale prédite / vraie : 0,92), il est
  surtout **moins biaisé** (biais -0,26 m/s à 60 s contre -2,47 m/s pour A0, qui dérive vers le bas au-delà
  des 5 s vus à l'entraînement).

**Q3. Pourquoi aucun modèle ne généralise aux avions extrêmes ?**
- Corrélations entre horizon utile et écart de chaque paramètre : toutes faibles (|rho| <= 0,13) ; aucun
  paramètre n'explique seul l'échec (un peu : inertie de roulis, puissance, CL, Cn).
- Erreurs à 1 s, avions d'entraînement -> de test : c'est surtout le **tangage** qui casse (A0 : q 3,0 ->
  5,5 deg/s, theta 1,8 -> 3,9 deg, vitesse verticale 0,7 -> 1,2 m/s).
- Explication : **le modèle ne sait pas sur quel avion il vole**. 0,5 s d'historique est trop court pour
  « deviner » la masse, l'inertie ou l'aéro. Conséquence pour le modèle C : des équations physiques avec
  des paramètres nominaux ne connaîtront pas non plus l'avion extrême ; il faudra lui donner l'information
  (historique plus long pour l'inférer, ou paramètres connus comme la masse en entrée).

**À faire** : ajouter la référence « altitude constante » à l'évaluation ; trancher, pour le modèle C,
comment le modèle connaît l'avion (historique plus long, paramètres en entrée) ; 3 graines en cours.

## 10. Modèle C : décisions et plan (21 oct. 2026)

Décisions (détail : `model_design.md` § 11) :
1. **Le modèle doit savoir sur quel avion il vole** : 2 s de passé (encodeur à 5 couches pour les voir) et
   masse + inerties en entrée, **pour tous les modèles** (A0-v2, A-v2 réentraînés).
2. **Deux variantes de C** : C-direct (tout appris ensemble, contre A0-v2) et C-JEPA (tronc JEPA figé +
   tête physique, contre A-v2, comme SkyJEPA).
3. **Équations de Newton complètes en 3D** dans la tête.
4. **Trois niveaux d'a priori + contrôle** : ajusté, générique (Ultra Stick 25e), faux (+/- 30 %), nul (C0).
5. **Correction = 6 forces et moments** ajoutés avant Newton (vraie masse, vraies inerties), pénalité de
   taille.

Plan (une graine, puis 3) : smoke test (`cluster/smoke_c.sbatch`, construit aussi les a priori), puis
A0-v2, A-v2, C-direct x 4 a priori, et C-JEPA x 4 a priori une fois A-v2 fini. Le code n'a pas été exécuté
sur le PC : tout est vérifié sur Savio.
