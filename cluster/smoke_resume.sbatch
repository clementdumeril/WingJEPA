#!/bin/bash
# Checks the restart of train_a.py (resume.pt): a short run is killed each time it has printed two more
# evaluations, and resubmitted until it finishes, then compared with the same run done in one go. The
# validation losses must be identical at every evaluation (same steps, same random draws), and resume.pt
# must be gone at the end. Killing on progress, not after a fixed time, keeps the check independent of
# how long the start-up (data loading) takes on the node.
# Done for jepa (two stages: the restart can fall in either) and c_direct (physics head).
# Usage: sbatch cluster/smoke_resume.sbatch
#SBATCH --job-name=wingjepa-smoke-resume
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm runs/bench
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
COMMON="--data-dir $DATA --batch 128 --val-windows 256 --eval-every 25 --warmup 10 --T 20 --T-max 40 --checkpoint-min 0.1"

launch() {   # one launch of the interrupted run; returns 124 if it was killed
  local before now pid
  before=$(grep -c '"stage"' $CUT.out || true)
  python scripts/train_a.py $COMMON --out $CUT --mode $mode $extra >> $CUT.out 2>&1 &
  pid=$!
  while kill -0 $pid 2>/dev/null; do
    sleep 1
    now=$(grep -c '"stage"' $CUT.out || true)
    if [ $((now - before)) -ge 2 ]; then
      kill $pid; wait $pid 2>/dev/null || true
      return 124
    fi
  done
  wait $pid
}

for mode in jepa c_direct; do
  if [ "$mode" = jepa ]; then extra="--steps1 150 --steps2 150"
  else extra="--steps1 40 --steps2 40 --prior $DATA/priors/ajuste.json"; fi
  REF=runs/bench/resume_ref_$mode; CUT=runs/bench/resume_cut_$mode
  rm -rf $REF $CUT $REF.out $CUT.out
  touch $CUT.out
  SECONDS=0
  python scripts/train_a.py $COMMON --out $REF --mode $mode $extra > $REF.out
  echo "$mode: run in one go took $SECONDS s"
  n=0
  while true; do
    set +e; launch; rc=$?; set -e
    [ $rc -eq 0 ] && break
    [ $rc -ne 124 ] && { echo "$mode: train_a.py failed (exit $rc)"; tail -20 $CUT.out; exit 1; }
    n=$((n + 1)); [ $n -ge 40 ] && { echo "$mode: still not finished after $n restarts"; exit 1; }
  done
  r=$(grep -c '^resuming' $CUT.out || true)
  echo "$mode: killed $n times, resumed $r times from resume.pt"
  [ "$r" -ge 1 ] || { echo "$mode: no restart happened, the check proves nothing"; exit 1; }
  python - $REF $CUT <<'PY'
import json, sys
from pathlib import Path
ref, cut = (json.load(open(Path(d) / "log.json")) for d in sys.argv[1:])
key = lambda r: (r["stage"], r["step"])
a = {key(r): r["val"] for r in ref}
b = {key(r): r["val"] for r in cut}
assert a.keys() == b.keys(), f"different evaluations: {sorted(a.keys() ^ b.keys())[:5]}"
diff = max(abs(a[k] - b[k]) / max(abs(a[k]), 1e-12) for k in a)
print(f"  {len(a)} evaluations, largest relative difference {diff:.1e}")
assert diff < 1e-5, "the resumed run differs from the run done in one go"
assert not (Path(sys.argv[2]) / "resume.pt").exists(), "resume.pt left behind"
PY
done
echo "SMOKE RESUME OK"
