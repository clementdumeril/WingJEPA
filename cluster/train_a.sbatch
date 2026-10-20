#!/bin/bash
# Entraînement + évaluation du modèle A (jepa) ou de la référence A0 (a0).
# Usage : sbatch cluster/train_a.sbatch jepa        (ou a0)  [graine, défaut 0]
# Sorties : runs/<mode>_s<graine>/ (model.pt, config.json, log.json, eval_*.json)
#SBATCH --job-name=wingjepa-train
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
MODE=${1:-jepa}
SEED=${2:-0}
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
OUT=runs/${MODE}_s${SEED}
python scripts/train_a.py --data-dir "$DATA" --set train --out "$OUT" --mode "$MODE" --seed "$SEED"
python scripts/evaluate.py --model-dir "$OUT" --data-dir "$DATA" --set train
python scripts/evaluate.py --model-dir "$OUT" --data-dir "$DATA" --set test_long
