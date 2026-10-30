#!/bin/bash
# Builds everything the README shows, from finished runs, into export/ (ignored by git, so that pulls on
# the cluster never clash with the committed copies), then packs it as export.tgz:
#   export/results/<run>/ and summary.csv   (scripts/collect_results.py)
#   export/assets/useful_horizon.png, altitude_error.png, calm_vs_turbulence.png, position_error.png,
#   probe.png   (scripts/make_figures.py)
#   export/assets/rollout_extreme_aircraft.gif / .png   (scripts/make_rollout_gif.py)
#   export/assets/trajectory_3d_calm.gif / .png / .json   (scripts/make_trajectory_3d.py, calm air)
#   export/assets/validation/*.png: simulator vs UMN reference (scripts/validate_ultrastick.py; needs the
#   full UMN archive, downloaded here if the node has internet access, skipped otherwise)
# Unpack export.tgz at the repository root of the local clone to update results/ and assets/.
# The unit tests (tests/) run first.
# Usage: sbatch cluster/make_assets.sbatch
#SBATCH --job-name=wingjepa-assets
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
DATA=/global/scratch/users/$USER/WingJEPA/data/data_v1
CALM=/global/scratch/users/$USER/WingJEPA/data/data_calm

python -m pytest -q tests
E=export
python scripts/collect_results.py --out $E/results
python scripts/make_figures.py --results $E/results --out $E/assets
python scripts/make_rollout_gif.py --data-dir "$DATA" --out $E/assets/rollout_extreme_aircraft \
  --models runs/a0_v2T100_s0 runs/jepa_v2T100_s0 runs/c_direct_v2T100_ajuste_s0 \
  --labels "A0 (direct)" "A (JEPA)" "C (physics head, fitted prior)"
python scripts/make_trajectory_3d.py --data-dir "$CALM" --out $E/assets/trajectory_3d_calm --title "Calm air" \
  --models runs/a0_calmT100_s0 runs/jepa_calmT100_s0 runs/c_jepa_calmT100_ajuste_s0 runs/c_direct_calmT100_ajuste_s0 \
  --labels "A0 (direct)" "A (JEPA)" "C-JEPA (physics head on JEPA)" "C-direct (physics head)"

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
  python scripts/validate_ultrastick.py --umn-dir "$UMN/UMNUAV_sim_v4" --out $E/assets/validation
fi
tar czf export.tgz -C $E results assets
ls -la $E/assets export.tgz
echo "ASSETS OK"
