#!/bin/bash
# Quick check (debug queue) of the pieces added for the training plan, before the long runs:
#   1. unit tests (tests/);
#   2. a few steps of: a0 with the joint command encoder (--action-history 2), the plain MLP baseline
#      (--mode mlp_direct), the residual control head (c_direct --head-kind residual), a denoising JEPA
#      (jepa --input-noise 0.5) and c_jepa on it, the context latent (jepa and a0 --context-dim 8, c_jepa on it), the identification JEPA (--id-jepa);
#   3. eval_skyjepa with every noise kind and with the nominal mass and inertia, the aircraft probe, a tiny
#      closed loop with colored noise and nominal parameters, and its paired statistics.
# Nothing is kept (runs/smoke_plan/). Usage: sbatch cluster/smoke_plan.sbatch
#SBATCH --job-name=wingjepa-smoke-plan
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
PRIOR=data/data_v1/priors/nul.json
S=runs/smoke_plan
rm -rf $S
ARCH="--H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 --action-encoder 1"
FAST="--steps1 30 --steps2 30 --batch 64 --eval-every 10 --val-windows 256 --warmup 5 --T 10 --T-max 20"

echo "=== 1. unit tests"
python -m pytest -q tests
echo "=== 2. very short training runs"
python scripts/train_a.py --data-dir "$DATA" --out $S/a0_ah2 --mode a0 --action-history 2 $ARCH $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/mlp --mode mlp_direct $ARCH $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/res --mode c_direct --head-kind residual $ARCH $FAST
grep -q '"head": "residual"' $S/res/config.json && echo "c_direct built the residual head"
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa_dn --mode jepa --input-noise 0.5 $ARCH $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/c_jepa_dn --mode c_jepa --prior $PRIOR --trunk $S/jepa_dn $ARCH $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa_ctx --mode jepa --context-dim 8 $ARCH $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/a0_ctx --mode a0 --context-dim 8 $ARCH $FAST
python scripts/train_a.py --data-dir "$DATA" --out $S/c_jepa_ctx --mode c_jepa --prior $PRIOR --trunk $S/jepa_ctx $ARCH $FAST
grep -q '"context_dim": 8' $S/c_jepa_ctx/config.json && echo "c_jepa took the context latent of its trunk"
python scripts/train_a.py --data-dir "$DATA" --out $S/jepa_id --mode jepa --context-dim 8 --id-jepa 1 $ARCH $FAST
grep -q '"id_inv"' $S/jepa_id/log.json && echo "identification JEPA loss logged"
echo "=== 3. evaluations"
M="$S/a0_ah2 $S/mlp $S/res $S/c_jepa_dn $S/jepa_ctx $S/c_jepa_ctx"
python scripts/eval_skyjepa.py --data-dir "$DATA" --models $M --noise-kinds gauss bias colored dropout --max-rollouts 64
python scripts/eval_skyjepa.py --data-dir "$DATA" --models $S/c_jepa_dn --nominal-params 1 --max-rollouts 64
ls $S/c_jepa_dn/eval_skyjepa_test_long_nominal.json
python scripts/probe_aircraft.py --data-dir "$DATA" --models $S/a0_ah2 $S/jepa_dn $S/jepa_ctx $S/a0_ctx $S/jepa_id --hist-s 0.55 --n-fit 2000 --steps 50 \
    --out $S/probe
for k in colored dropout; do
    python scripts/closed_loop.py --controllers tecs $S/mlp $S/c_jepa_dn $S/c_jepa_ctx --out $S/cl_$k.json --n-train 2 --n-test 2 \
        --duration 3 --samples 16 --compile 0 --noise 0.3 --noise-kind $k --nominal-params 1 --device cpu
done
python scripts/closed_loop_stats.py --runs $S/cl_colored.json $S/cl_dropout.json --ref tecs
echo "SMOKE PLAN OK"
