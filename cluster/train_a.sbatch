#!/bin/bash
# Training + evaluation of a world model of the SkyJEPA suite (jepa, a0, c_direct, c_jepa, mlp_direct).
# Usage: sbatch cluster/train_a.sbatch <mode> <seed> <tag> [train_a.py options]
#   e.g.: S="--H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 --action-encoder 1 --T 20"
#         sbatch cluster/train_a.sbatch a0 0 adv $S
# Outputs: runs/<mode>_<tag>_s<seed>/ (model.pt, config.json, log.json, eval_skyjepa_test_long.json)
# CPUs: 16 by default; override with sbatch -c N (the thread count follows SLURM_CPUS_PER_TASK).
# Stopped by the time limit: resubmit the same command, the run continues from runs/<...>/resume.pt.
# On a GPU (GPU environment from cluster/setup_gpu_env.sh):
#   sbatch --partition=savio3_gpu --qos=a40_gpu3_ica --gres=gpu:A40:1 -c 8 \
#     --export=ALL,VENV=/global/scratch/users/$USER/venvs/wingjepa-gpu cluster/train_a.sbatch jepa 0 T100 ... --device cuda
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
source ${VENV:-/global/scratch/users/$USER/venvs/wingjepa}/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
MODE=${1:-jepa}
SEED=${2:-0}
TAG=${3:-base}
shift $(( $# < 3 ? $# : 3 ))
DATA=${DATA:-/global/scratch/users/$USER/WingJEPA/data/data_advanced}   # another dataset: sbatch --export=ALL,DATA=<dir> ...
OUT=runs/${MODE}_${TAG}_s${SEED}
echo "mode=$MODE seed=$SEED tag=$TAG options=$* -> $OUT"
python scripts/train_a.py --data-dir "$DATA" --set train --out "$OUT" --mode "$MODE" --seed "$SEED" "$@"
python scripts/eval_skyjepa.py --data-dir "$DATA" --models "$OUT"
