#!/bin/bash
# Quick check (debug queue) of the JEPA variants before long runs: a few steps of jepa with a slow target
# encoder (--ema) and of jepa_ft (JEPA pre-training + fine-tuning), then a short evaluation.
# Nothing is kept (runs/smoke_jepa/). Usage: sbatch cluster/smoke_jepa.sbatch
#SBATCH --job-name=wingjepa-smoke-jepa
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_debug
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
S=runs/smoke_jepa
FAST="--steps1 30 --steps2 30 --batch 64 --eval-every 10 --val-windows 256 --warmup 5 --T 10 --T-max 20"

python scripts/train_a.py --data-dir "$DATA" --out $S/jepa_ema --mode jepa --ema 0.996 $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa_ft_ema --mode jepa_ft --ema 0.996 $FAST
python scripts/evaluate.py --model-dir $S/jepa_ft_ema --data-dir "$DATA" --set test_long --horizon-s 5 --stride-s 30
echo "SMOKE JEPA OK"
