#!/bin/bash
# Is the compiled physics-head step (train_a.py --compile 1) correct, and how much faster is it?
#   1. same outputs and gradients as the plain step, on random states (tolerance 1e-4);
#   2. seconds per training step of c_direct at a 5 s horizon, plain vs compiled, 1 to 8 threads
#      (the first step, which includes the compilation, is reported apart).
# Usage: sbatch cluster/bench_compile.sbatch
#SBATCH --job-name=wingjepa-bench-compile
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=8
#SBATCH --time=01:30:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
PRIOR=$DATA/priors/ajuste.json
python -c "import torch, sys; print('python', sys.version.split()[0], '| torch', torch.__version__)"
gcc --version | head -1 || echo "no gcc in PATH"

echo "=== 1. compiled step = plain step"
python - "$PRIOR" <<'PY'
import json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from wingjepa.models.physics import PhysicsHead
torch.manual_seed(0)
prior = json.load(open(sys.argv[1]))
plain = PhysicsHead(prior, np.zeros(16), np.ones(16))
comp = PhysicsHead(prior, np.zeros(16), np.ones(16)).compile_step()
B = 1024
x = torch.cat([torch.rand(B, 1) * 10 + 15, torch.randn(B, 2), torch.randn(B, 3) * 0.3, torch.randn(B, 2) * 0.3], 1)
ctrl = torch.cat([torch.randn(B, 3) * 0.1, torch.rand(B, 1)], 1)
dC = (torch.randn(B, 6) * 0.01).requires_grad_()
ac = torch.tensor([[6.8, 0.67, 0.89, 1.58, -0.76]]).repeat(B, 1)
out = {}
for name, head in (("plain", plain), ("compiled", comp)):
    d = dC.detach().clone().requires_grad_()
    y = head._step(x, ctrl, d, ac)
    y.sum().backward()
    out[name] = (y.detach(), d.grad)
dy = (out["plain"][0] - out["compiled"][0]).abs().max().item()
dg = (out["plain"][1] - out["compiled"][1]).abs().max().item()
print(f"max difference: state {dy:.2e}, gradient {dg:.2e}")
assert dy < 1e-4 and dg < 1e-3, "compiled step differs from the plain step"
print("same outputs and gradients")
PY

echo "=== 2. speed (c_direct, batch 1024, 5 s horizon)"
STEPS="--steps1 0 --steps2 41 --eval-every 40 --val-windows 256 --warmup 5 --T 100"
for cfg in "0 2" "1 1" "1 2" "1 4" "1 8"; do
  set -- $cfg
  export OMP_NUM_THREADS=$2
  python scripts/train_a.py --data-dir "$DATA" --out runs/bench/compile$1_t$2 --mode c_direct --prior "$PRIOR" \
      --threads $2 --compile $1 $STEPS 2>/dev/null \
    | python -c "import sys, json
recs = [json.loads(l) for l in sys.stdin if l.startswith('{\"stage\"')]
rate = (recs[-1]['sec'] - recs[0]['sec']) / (recs[-1]['step'] - recs[0]['step'])
print(f'compile=$1 threads=$2: {rate:.3f} s per step (first step incl. compilation: {recs[0][\"sec\"]:.0f} s)')"
done
echo "BENCH COMPILE OK"
