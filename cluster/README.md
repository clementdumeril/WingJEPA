# Savio (BRC) — mise en route

Référence officielle : https://docs-research-it.berkeley.edu/services/high-performance-computing/user-guide/

> La connexion demande **ton PIN + le code Google Authenticator**. Ces étapes sont à faire **par toi** ;
> rien dans ce dépôt ne stocke ni ne saisit d'identifiants.
> Les points marqués « à vérifier » viennent de connaissances générales sur Savio, pas de la page de connexion.

## 1. Première connexion (une fois)

1. Avoir configuré le jeton OTP (Google Authenticator) — voir *Setting Up One-Time Passwords* dans la doc.
2. Depuis un terminal :
   ```bash
   ssh <ton_login_savio>@hpc.brc.berkeley.edu
   # Password: PIN suivi immédiatement du code à 6 chiffres
   ```
3. Une fois connecté, repérer **ton compte Slurm** (commence par `fc_` ou `co_`) et les partitions accessibles :
   ```bash
   sacctmgr -p show associations user=$USER     # à vérifier : liste comptes / partitions / QoS
   sinfo -s                                     # partitions disponibles
   ```
   → Noter `ACCOUNT` (ex. `fc_xxx`) : il remplace `<ACCOUNT>` dans les scripts `.sbatch`.

## 2. Connexion sans mot de passe (optionnel, certificat de 12 h)

Sous Windows, la doc recommande WSL (Ubuntu). Dans un terminal WSL :
```bash
git clone https://github.com/lbnl-science-it/lrc-scripts.git
cd lrc-scripts
./request_cert.sh -p brc          # demande login, PIN, puis OTP
```
Puis ajouter à `~/.ssh/config` (côté Windows, chemin vers le certificat dans WSL) :
```
Host savio
    User <ton_login_savio>
    HostName hpc.brc.berkeley.edu
    IdentityFile //wsl$/Ubuntu/home/<user_wsl>/.ssh/ssh_certs/brc_cert
    IdentitiesOnly yes
```
Ensuite : `ssh savio`. Le certificat expire après 12 h (le recréer avec `request_cert.sh`).

## 3. Installer l'environnement sur Savio (une fois)

```bash
ssh savio
cd /global/scratch/users/$USER            # à vérifier : espace scratch (grand, non sauvegardé)
git clone https://github.com/clementdumeril/WingJEPA.git
cd WingJEPA
bash cluster/setup_env.sh                  # crée l'environnement Python dans scratch
```

## 4. Lancer des calculs

**Ne jamais lancer de calcul sur le nœud de connexion** : toujours passer par Slurm.

```bash
sbatch cluster/job_cpu.sbatch scripts/phase0_jsbsim_checks.py     # un job CPU de test
sbatch --array=0-134 cluster/job_array.sbatch configs/sweep.yaml  # la grille d'expériences (phase 3)
squeue -u $USER                                                    # suivre ses jobs
```

Nos calculs (JSBSim, modèles ~100 k paramètres) sont **surtout CPU** : les partitions HTC
(`savio3_htc`, `savio4_htc`, allocation par cœur) sont adaptées. Le GPU n'est pas indispensable.

## 5. Récupérer les résultats

Utiliser le nœud de transfert `dtn.brc.berkeley.edu` (pas le nœud de connexion) :
```bash
scp -r <ton_login_savio>@dtn.brc.berkeley.edu:/global/scratch/users/<ton_login_savio>/WingJEPA/runs ./runs
```

## Notre configuration (vérifiée le 15 oct. 2026)

- Login : `clementdumeril` — compte Slurm : **`ic_piml`** (instructional computing du cours).
- CPU : `savio3_htc`, `savio4_htc` (QoS `savio_normal`, `savio_debug`) → **défaut des scripts : `savio4_htc`**.
- GPU (si besoin) : `savio3_gpu` (QoS `a40_gpu3_ica`, `v100_gpu3_ica`, `gtx2080_gpu3_ica`),
  `savio4_gpu` (QoS `a5k_gpu4_ica`).
- Espace de travail : `/global/scratch/users/clementdumeril/WingJEPA`.
