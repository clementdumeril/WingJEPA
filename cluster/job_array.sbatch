#!/bin/bash
# MODÈLE DE JOB, PAS ENCORE UTILISABLE : scripts/train.py et configs/sweep.yaml n'existent pas encore.
# Principe : un job par configuration d'expérience, choisie par l'index SLURM_ARRAY_TASK_ID.
# La grille (modèles, priors, tailles de données, graines) sera décidée en phase 2-3.
# Usage futur : sbatch --array=0-<N-1> cluster/job_array.sbatch configs/sweep.yaml
#SBATCH --job-name=wingjepa-sweep
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --output=runs/slurm/%x-%A_%a.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
python scripts/train.py --sweep "$1" --index "$SLURM_ARRAY_TASK_ID"
