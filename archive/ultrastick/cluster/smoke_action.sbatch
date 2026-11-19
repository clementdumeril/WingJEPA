#!/bin/bash
# Quick check (debug queue) of the encoder of the command history (--action-history 1) before the long runs:
#   1. unit tests (tests/), including the one of the command-history encoder;
#   2. a few steps of jepa, then c_jepa on that trunk (takes the option from it), and of a0;
#   3. short evaluation and latent probe (scripts/diagnose.py).
# Nothing is kept (runs/smoke_action/). Usage: sbatch cluster/smoke_action.sbatch
#SBATCH --job-name=wingjepa-smoke-action
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
S=runs/smoke_action
FAST="--steps1 30 --steps2 30 --batch 64 --eval-every 10 --val-windows 256 --warmup 5 --T 10 --T-max 20"

echo "=== 1. unit tests"
python -m pytest -q tests
echo "=== 2. very short training runs"
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa --mode jepa --action-history 1 $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/c_jepa --mode c_jepa --prior $PRIOR --trunk $S/jepa $FAST
grep -q '"action_history": 1' $S/c_jepa/config.json && echo "c_jepa took the command-history encoder from its trunk"
python scripts/train_a.py --data-dir "$DATA" --out $S/a0 --mode a0 --action-history 1 $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa_tw --mode jepa --target-window 10 $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa_ae --mode jepa --action-encoder 1 --H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/c_jepa_ae --mode c_jepa --prior $PRIOR --trunk $S/jepa_ae $FAST
grep -q '"action_encoder": 1' $S/c_jepa_ae/config.json && echo "c_jepa took the action encoder from its trunk"
echo "=== 3. short evaluation and probe"
python scripts/evaluate.py --model-dir $S/c_jepa --data-dir "$DATA" --set test_long --horizon-s 5 --stride-s 30
python scripts/diagnose.py --data-dir "$DATA" --models $S/jepa
echo "SMOKE ACTION OK"
