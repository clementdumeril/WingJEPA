#!/bin/bash
# Training + evaluation of a world model (model A: jepa, joint; reference: a0).
# Usage: sbatch cluster/train_a.sbatch <mode> <seed> <tag> [train_a.py options]
#   e.g.: sbatch cluster/train_a.sbatch a0 0 T100 --T 20 --T-max 100
# Outputs: runs/<mode>_<tag>_s<seed>/ (model.pt, config.json, log.json, eval_*.json)
# CPUs: 16 by default; override with sbatch -c N (the thread count follows SLURM_CPUS_PER_TASK).
#SBATCH --job-name=wingjepa-train
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source cluster/compiler.sh
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
MODE=${1:-jepa}
SEED=${2:-0}
TAG=${3:-base}
shift $(( $# < 3 ? $# : 3 ))
DATA=${DATA:-/global/scratch/users/$USER/WingJEPA/data/data_v1}     # another dataset: sbatch --export=ALL,DATA=<dir> ...
OUT=runs/${MODE}_${TAG}_s${SEED}
echo "mode=$MODE seed=$SEED tag=$TAG options=$* -> $OUT"
python scripts/train_a.py --data-dir "$DATA" --set train --out "$OUT" --mode "$MODE" --seed "$SEED" "$@"
python scripts/evaluate.py --model-dir "$OUT" --data-dir "$DATA" --set train
python scripts/evaluate.py --model-dir "$OUT" --data-dir "$DATA" --set test_long
