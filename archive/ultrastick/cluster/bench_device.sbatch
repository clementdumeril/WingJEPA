#!/bin/bash
# CPU or GPU for each model? On one GPU node (A40 + its 8 cores):
#   1. the GPU computes the same thing as the CPU (states, JEPA targets, physics head), up to the TF32
#      rounding of the convolutions;
#   2. seconds per training step at a 5 s horizon, batch 1024: jepa (stage 1), a0, c_direct plain and
#      compiled, on the GPU, and jepa on the 8 CPU cores for comparison;
#   3. the full-run time these rates imply (40,000 steps), next to what the CPU runs actually took.
# Needs the GPU environment (bash cluster/setup_gpu_env.sh, once).
# Usage: sbatch cluster/bench_device.sbatch
#   on an A5000 instead: sbatch --partition=savio4_gpu --qos=a5k_gpu4_ica --gres=gpu:A5000:1 -c 4 cluster/bench_device.sbatch
#SBATCH --job-name=wingjepa-bench-device
#SBATCH --account=ic_piml
#SBATCH --partition=savio3_gpu
#SBATCH --qos=a40_gpu3_ica
#SBATCH --gres=gpu:A40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm runs/bench
source cluster/compiler.sh
source /global/scratch/users/$USER/venvs/wingjepa-gpu/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
PRIOR=$DATA/priors/ajuste.json
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -c "import torch; print('torch', torch.__version__, '| CUDA', torch.version.cuda, '| GPU visible:', torch.cuda.is_available())"

echo "=== 1. GPU = CPU"
python - "$PRIOR" <<'PY'
import json, sys
import numpy as np, torch
sys.path.insert(0, ".")
from wingjepa.data.windows import Normalizer
from wingjepa.models.jepa import WorldModel
torch.manual_seed(0)
prior = json.load(open(sys.argv[1]))
B, H, T = 256, 40, 100
seq = torch.randn(B, H + T, 16) * 0.5              # normalized: plausible flight states once denormalized
ac = torch.tensor([[6.8, 0.67, 0.89, 1.58, -0.76]]).repeat(B, 1)
norm = Normalizer.from_dict(json.load(open("runs/a0_v2T100_s0/config.json"))["norm"])
for head in ("mlp", "physics"):
    m = WorldModel(32, (16, 16, 32, 32, 32), dilations=(1, 2, 4, 8, 16), n_ctx=5, head=head, prior=prior, norm=norm)
    out = {}
    for dev in ("cpu", "cuda"):
        md = m.to(dev)
        s, a_ = seq.to(dev), ac.to(dev)
        pred = md.predict_states(s, H, True, a_)[0]
        zt = md.target_latents(s, H, a_)
        loss = pred.pow(2).mean() + zt.pow(2).mean()
        md.zero_grad()
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in md.parameters() if p.grad is not None])
        out[dev] = [x.detach().cpu() for x in (pred, zt, g)]
    rel = [((c - g).abs().max() / c.abs().max()).item() for c, g in zip(out["cpu"], out["cuda"])]
    print(f"{head}: largest relative difference states {rel[0]:.1e}, JEPA targets {rel[1]:.1e}, gradients {rel[2]:.1e}")
    assert max(rel) < 1e-2, "GPU and CPU disagree"
print("same outputs on GPU and CPU")
PY

echo "=== 2. speed (batch 1024, 5 s horizon)"
STEPS="--val-windows 256 --eval-every 40 --warmup 5 --T 100 --checkpoint-min 0"
declare -A RATE
bench() {   # name, then train_a.py options
  local name=$1; shift
  RATE[$name]=$(python scripts/train_a.py --data-dir "$DATA" --out runs/bench/device_$name "$@" $STEPS \
    | python -c "import sys, json
recs = [json.loads(l) for l in sys.stdin if l.startswith('{\"stage\"')]
print(f'{(recs[-1][\"sec\"] - recs[0][\"sec\"]) / (recs[-1][\"step\"] - recs[0][\"step\"]):.3f}')")
  echo "$name: ${RATE[$name]} s per step"
}
bench jepa_gpu       --mode jepa --steps1 41 --steps2 0 --device cuda
bench jepa_cpu8      --mode jepa --steps1 41 --steps2 0 --device cpu
bench a0_gpu         --mode a0 --steps1 0 --steps2 41 --device cuda
bench c_direct_gpu   --mode c_direct --steps1 0 --steps2 41 --prior "$PRIOR" --device cuda
bench c_compiled_gpu --mode c_direct --steps1 0 --steps2 41 --prior "$PRIOR" --device cuda --compile 1 \
  || echo "c_compiled_gpu: torch.compile failed on the GPU (see above); the plain GPU rate still holds"

echo "=== 3. full runs (20,000 + 20,000 steps; upper bound: the curriculum starts at 1 s, not 5 s)"
python - ${RATE[jepa_gpu]} ${RATE[jepa_cpu8]} ${RATE[a0_gpu]} ${RATE[c_compiled_gpu]:-nan} <<'PY'
import json, sys
jg, jc, ag, cg = map(float, sys.argv[1:])
h = lambda s: f"{s * 20000 / 3600:.1f} h"
print(f"GPU: jepa ~{h(jg + ag)} (stage 1 + decoder), a0 ~{h(2 * ag)}, c_direct compiled ~{h(2 * cg)}")
print(f"8 CPU cores of this node: jepa stage 1 alone ~{h(jc)}")
for run in ("jepa_v2T100_s0", "a0_v2T100_s0", "c_direct_calmT100_ajuste_s0"):
    try:
        log = json.load(open(f"runs/{run}/log.json"))
        total = sum(max(r["sec"] for r in log if r["stage"] == st) for st in {r["stage"] for r in log})
        print(f"actual CPU run {run}: {total / 3600:.1f} h")
    except FileNotFoundError:
        pass
PY
echo "BENCH DEVICE OK"
