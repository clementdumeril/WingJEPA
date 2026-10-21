#!/bin/bash
# Creates the WingJEPA Python environment on Savio, in the scratch space.
# Run from the root of the repo cloned on Savio: bash cluster/setup_env.sh
set -euo pipefail

ENV_DIR=/global/scratch/users/$USER/venvs/wingjepa   # to check: Savio scratch path

module load python 2>/dev/null || module load anaconda3   # to check: Python module name (`module avail python`)

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -c "import jsbsim, torch; print('jsbsim', jsbsim.__version__, '| torch', torch.__version__)"
mkdir -p runs/slurm   # Slurm opens the output file before the script runs: the directory must exist
bash scripts/fetch_umn.sh   # NASA tables of the Ultra Stick 120 (UMN simulator), from the login node
echo "Environment ready: source $ENV_DIR/bin/activate"
