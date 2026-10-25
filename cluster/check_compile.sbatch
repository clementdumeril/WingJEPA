#!/bin/bash
# Checks the compiled physics head under a growing horizon (curriculum 1 s -> 5 s within 81 steps, a new
# horizon at almost every step): it must keep the compiled speed, i.e. not recompile at every horizon and
# fall back to the plain step. Fails if the last 40 steps average more than 0.9 s (compiled: ~0.45 s at
# the 5 s horizon, plain: ~1.4 to 2.2 s). Usage: K=$(sbatch --parsable cluster/check_compile.sbatch)
#SBATCH --job-name=wingjepa-check-compile
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=2
#SBATCH --time=00:45:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source cluster/compiler.sh
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
python scripts/train_a.py --data-dir "$DATA" --out runs/bench/check_compile --mode c_direct \
    --prior "$DATA/priors/ajuste.json" --compile 1 --steps1 0 --steps2 81 --eval-every 40 \
    --val-windows 256 --warmup 5 --T 20 --T-max 100 --curriculum-frac 1.0 \
  | python -c "import sys, json
recs = [json.loads(l) for l in sys.stdin if l.startswith('{\"stage\"')]
rate = (recs[-1]['sec'] - recs[1]['sec']) / (recs[-1]['step'] - recs[1]['step'])
print(f'growing horizon (T {recs[1][\"T\"]} -> {recs[-1][\"T\"]}): {rate:.3f} s per step')
sys.exit(0 if rate < 0.9 else 'too slow: the compiled step is recompiling or falling back')"
echo "CHECK COMPILE OK"
