#!/bin/bash
# JOB TEMPLATE, NOT USABLE YET: scripts/train.py and configs/sweep.yaml do not exist yet.
# Principle: one job per experiment configuration, selected by the SLURM_ARRAY_TASK_ID index.
# The grid (models, priors, data sizes, seeds) will be decided in phase 2-3.
# Future usage: sbatch --array=0-<N-1> cluster/job_array.sbatch configs/sweep.yaml
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
