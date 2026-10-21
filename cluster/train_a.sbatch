#!/bin/bash
# Entraînement + évaluation d'un world model (modèle A : jepa, joint ; référence : a0).
# Usage : sbatch cluster/train_a.sbatch <mode> <graine> <étiquette> [options de train_a.py]
#   ex. : sbatch cluster/train_a.sbatch a0 0 T100 --T 20 --T-max 100
# Sorties : runs/<mode>_<étiquette>_s<graine>/ (model.pt, config.json, log.json, eval_*.json)
#SBATCH --job-name=wingjepa-train
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
MODE=${1:-jepa}
SEED=${2:-0}
TAG=${3:-base}
shift $(( $# < 3 ? $# : 3 ))
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
OUT=runs/${MODE}_${TAG}_s${SEED}
echo "mode=$MODE graine=$SEED étiquette=$TAG options=$* -> $OUT"
python scripts/train_a.py --data-dir "$DATA" --set train --out "$OUT" --mode "$MODE" --seed "$SEED" "$@"
python scripts/evaluate.py --model-dir "$OUT" --data-dir "$DATA" --set train
python scripts/evaluate.py --model-dir "$OUT" --data-dir "$DATA" --set test_long
