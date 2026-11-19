#!/bin/bash
# Quick check (debug queue) of the constrained physics head (--correction param) before the long runs:
#   1. unit tests (tests/), including the two of the constrained head;
#   2. a few compiled steps of c_direct and c_jepa with --correction param (fitted prior);
#   3. short evaluation, with the attribution and the identification (learned multiplier vs true factor).
# Nothing is kept from these runs (runs/smoke_param/). Usage: sbatch cluster/smoke_param.sbatch
#SBATCH --job-name=wingjepa-smoke-param
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_debug
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source cluster/compiler.sh
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_calm
PRIOR=data/data_v1/priors/ajuste.json
S=runs/smoke_param
FAST="--steps1 30 --steps2 30 --batch 64 --eval-every 10 --val-windows 256 --warmup 5 --T 10 --T-max 20 --compile 1"

echo "=== 1. unit tests"
python -m pytest -q tests
echo "=== 2. very short training runs"
python scripts/train_a.py --data-dir "$DATA" --out $S/cd_param --mode c_direct --prior $PRIOR --correction param $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/cj_param --mode c_jepa --prior $PRIOR --correction param \
  --trunk runs/jepa_calmT100_s0 $FAST
echo "=== 3. short evaluation"
python scripts/evaluate.py --model-dir $S/cd_param --data-dir "$DATA" --set test_long --horizon-s 5 --stride-s 30
echo "SMOKE PARAM OK"
