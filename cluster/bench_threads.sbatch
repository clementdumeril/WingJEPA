#!/bin/bash
# How fast does one training step run with 1, 2, 4, 8, 16 and 32 threads? The models are small (22k
# parameters), so extra threads may not help; the answer sets how many CPUs each run should request, so
# that nodes are filled with several runs instead of one run idling on many cores.
# Also prints the node size and the partitions the account can use (GPU access).
# Usage: sbatch cluster/bench_threads.sbatch
#SBATCH --job-name=wingjepa-bench
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
B=runs/bench
echo "=== node: $(nproc) usable cores here; partition nodes (cores, memory MB):"
sinfo -p savio4_htc -h -o "%c %m" | sort -u | head -3
echo "=== partitions of account ic_piml for $USER:"
sacctmgr -n -p show associations user=$USER format=account,partition,qos 2>/dev/null | head -20 || true
STEPS="--steps1 0 --steps2 41 --eval-every 40 --val-windows 256 --warmup 5 --T 100"
for mode in a0 c_direct; do
  extra=""; [ "$mode" = c_direct ] && extra="--prior $DATA/priors/ajuste.json"
  for th in 1 2 4 8 16 32; do
    export OMP_NUM_THREADS=$th
    python scripts/train_a.py --data-dir "$DATA" --out $B/${mode}_t$th --mode $mode --threads $th $STEPS $extra \
      | python -c "import sys, json
recs = [json.loads(l) for l in sys.stdin if l.startswith('{\"stage\"')]
print(f'$mode threads $th: {(recs[-1][\"sec\"] - recs[0][\"sec\"]) / (recs[-1][\"step\"] - recs[0][\"step\"]):.3f} s per step (batch 1024, 5 s horizon)')"
  done
done
echo "BENCH OK"
