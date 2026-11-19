#!/bin/bash
# Quick check (debug queue) of everything model C adds, before the long training runs:
#   1. builds the four priors (data/data_v1/priors/*.json, reused afterwards for real);
#   2. a few steps of each mode (a0 and jepa v2, c_direct with fitted and null prior, c_jepa);
#   3. short evaluation of a model C (with the attribution); reload of an old v1 run.
# Nothing is kept from these training runs (runs/smoke_c/). Usage: sbatch cluster/smoke_c.sbatch
#SBATCH --job-name=wingjepa-smoke-c
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
S=runs/smoke_c
FAST="--steps1 30 --steps2 30 --batch 64 --eval-every 10 --val-windows 256 --warmup 5 --T 10 --T-max 20"

echo "=== 1. priors"
python scripts/fit_prior.py --data-dir "$DATA"
echo "=== 2. very short training runs"
python scripts/train_a.py --data-dir "$DATA" --out $S/a0 --mode a0 $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa --mode jepa $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/cd_ajuste --mode c_direct --prior "$DATA/priors/ajuste.json" $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/cd_nul --mode c_direct --prior "$DATA/priors/nul.json" --lambda-res 0 $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/cj_generique --mode c_jepa --prior "$DATA/priors/generique.json" --trunk $S/jepa $FAST
echo "=== 3. short evaluation"
python scripts/evaluate.py --model-dir $S/cd_ajuste --data-dir "$DATA" --set test_long --horizon-s 5 --stride-s 30
python scripts/evaluate.py --model-dir $S/a0 --data-dir "$DATA" --set test_long --horizon-s 5 --stride-s 30
if [ -d runs/a0_T100_s0 ]; then
  python -c "import sys; sys.path[:0] = ['scripts', '.']; from evaluate import load_model; m, c, n = load_model('runs/a0_T100_s0'); print('old v1 run reloaded, H =', c['H'])"
fi
echo "SMOKE C OK"
