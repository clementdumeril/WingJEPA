#!/bin/bash
# Réévalue des modèles déjà entraînés (sans réentraîner).
# Usage : sbatch cluster/eval_a.sbatch runs/jepa_s0 runs/a0_s0
#SBATCH --job-name=wingjepa-eval
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
for M in "$@"; do
  echo "################ $M"
  python scripts/evaluate.py --model-dir "$M" --data-dir "$DATA" --set train
  python scripts/evaluate.py --model-dir "$M" --data-dir "$DATA" --set test_long
done
