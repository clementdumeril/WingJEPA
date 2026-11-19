#!/bin/bash
# Checks the compiled physics head under a growing horizon (curriculum 1 s -> 5 s within 81 steps, a new
# horizon at almost every step), against the plain step on the same node:
#   - counts the recompilations and prints the first reasons (TORCH_LOGS=recompiles);
#   - fails unless the compiled run is at least 1.7 times faster than the plain one.
# Usage: K=$(sbatch --parsable cluster/check_compile.sbatch)
#SBATCH --job-name=wingjepa-check-compile
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm runs/bench
source cluster/compiler.sh
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
rate() {   # seconds per step between the evaluations at steps 40 and 80
  python -c "import sys, json
recs = [json.loads(l) for l in open(sys.argv[1]) if l.startswith('{\"stage\"')]
print((recs[-1]['sec'] - recs[1]['sec']) / (recs[-1]['step'] - recs[1]['step']))" "$1"
}
for c in 0 1; do
  TORCH_LOGS=recompiles python scripts/train_a.py --data-dir "$DATA" --out runs/bench/check_compile$c --mode c_direct \
      --prior "$DATA/priors/ajuste.json" --compile $c --steps1 0 --steps2 81 --eval-every 40 \
      --val-windows 256 --warmup 5 --T 20 --T-max 100 --curriculum-frac 1.0 \
      > runs/bench/check_compile$c.out 2> runs/bench/check_compile$c.err
done
P=$(rate runs/bench/check_compile0.out); C=$(rate runs/bench/check_compile1.out)
N=$(grep -c "Recompiling" runs/bench/check_compile1.err || true)
echo "growing horizon, plain: $P s per step | compiled: $C s per step | recompilations: $N"
echo "first recompilation reasons:"
grep -A 3 "Recompiling" runs/bench/check_compile1.err | head -16 || true
python -c "import sys; r = $P / $C; print(f'speed-up x{r:.2f}'); sys.exit(0 if r > 1.7 else 'compiled step not fast enough')"
echo "CHECK COMPILE OK"
