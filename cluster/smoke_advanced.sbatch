#!/bin/bash
# Quick check (debug queue) of the SkyJEPA test suite on the advanced_plane data, before the long runs: a few steps
# of each recipe of SkyJEPA's Table III (one-step Predictive, Predictive + physics in the loss, reconstruction-type
# direct latent + MLP prober, its frozen trunk + physics prober, JEPA + MLP prober, JEPA frozen + physics prober)
# and of the two extra controls (physics head end to end, plain MLP), then the evaluation and a tiny closed loop.
# Nothing is kept (runs/smoke_advanced/). Usage: sbatch cluster/smoke_advanced.sbatch
#SBATCH --job-name=wingjepa-smoke-adv
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
DATA=/global/scratch/users/$USER/WingJEPA/data/data_advanced
PRIOR=configs/priors/advanced_plane_nul.json
S=runs/smoke_advanced
rm -rf $S
ARCH="--H 11 --channels 8,8,16 --dilations 1,2,4 --z-dim 24 --aircraft-input 0 --action-encoder 1"
FAST="--steps1 30 --steps2 30 --batch 64 --eval-every 10 --val-windows 256 --warmup 5 --T 10 --T-max 20"
T="python scripts/train_a.py --data-dir $DATA"

echo "=== 1. unit tests of the aircraft"
python -m pytest -q tests/test_advanced_plane.py
echo "=== 2. very short training runs"
$T --out $S/pred --mode a0 $ARCH $FAST --T 1 --T-max 1
$T --out $S/pred_phys --mode a0 $ARCH $FAST --T 1 --T-max 1 --lambda-phys 1
$T --out $S/a0 --mode a0 $ARCH $FAST
$T --out $S/a0_pi --mode c_jepa --prior $PRIOR --trunk $S/a0 $ARCH $FAST
$T --out $S/jepa --mode jepa $ARCH $FAST
$T --out $S/jepa_pi --mode c_jepa --prior $PRIOR --trunk $S/jepa $ARCH $FAST
$T --out $S/cdirect --mode c_direct --prior $PRIOR $ARCH $FAST
$T --out $S/mlp --mode mlp_direct $ARCH $FAST
echo "=== 3. evaluation (SkyJEPA metrics) and a tiny closed loop"
python scripts/eval_skyjepa.py --data-dir "$DATA" --models $S/pred $S/pred_phys $S/a0 $S/a0_pi $S/jepa $S/jepa_pi \
    $S/cdirect $S/mlp --max-rollouts 64
python scripts/closed_loop.py --config configs/data_advanced.yaml --controllers tecs $S/a0 $S/jepa_pi \
    --out $S/cl.json --n-train 2 --n-test 2 --duration 3 --samples 16 --horizon 20 --compile 0 --noise 0.1 --device cpu
echo "SMOKE ADVANCED OK"
