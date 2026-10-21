#!/bin/bash
# Builds everything the README shows, from finished runs:
#   results/<run>/ and results/summary.csv   (scripts/collect_results.py)
#   assets/useful_horizon.png, assets/altitude_error.png   (scripts/make_figures.py)
#   assets/rollout_extreme_aircraft.gif / .png   (scripts/make_rollout_gif.py)
#   assets/validation/*.png: simulator vs UMN reference (scripts/validate_ultrastick.py; needs the full
#   UMN archive, downloaded here if the node has internet access, skipped otherwise)
# The unit tests (tests/) run first.
# Usage: sbatch cluster/make_assets.sbatch
#SBATCH --job-name=wingjepa-assets
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1

python -m pytest -q tests
python scripts/collect_results.py
python scripts/make_figures.py
python scripts/make_rollout_gif.py --data-dir "$DATA" \
  --models runs/a0_v2T100_s0 runs/jepa_v2T100_s0 runs/c_direct_v2T100_ajuste_s0 \
  --labels "A0 (direct)" "A (JEPA)" "C (physics head, fitted prior)"

UMN=third_party/umn_full
if [ ! -d "$UMN/UMNUAV_sim_v4" ]; then
  mkdir -p "$UMN"
  if curl -fsSL -o "$UMN/sim.zip" "https://dept.aem.umn.edu/~uav/UMNUAV_sim_v4.zip"; then
    unzip -q -o "$UMN/sim.zip" -d "$UMN" && rm "$UMN/sim.zip"
  else
    echo "no internet access from this node: simulator validation figures skipped"
  fi
fi
if [ -d "$UMN/UMNUAV_sim_v4" ]; then
  python scripts/validate_ultrastick.py --umn-dir "$UMN/UMNUAV_sim_v4" --out assets/validation
fi
ls -la assets assets/validation 2>/dev/null || true
echo "ASSETS OK"
