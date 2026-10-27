#!/bin/bash
# Creates a second Python environment with a CUDA build of torch, for training on Savio's GPUs
# (train_a.py --device cuda). The main environment (setup_env.sh) keeps the CPU build.
# Same torch version as the main environment, so CPU and GPU runs differ only by the device.
# Run once, from the root of the repo on a login node: bash cluster/setup_gpu_env.sh
set -euo pipefail

CPU_ENV=/global/scratch/users/$USER/venvs/wingjepa
ENV_DIR=/global/scratch/users/$USER/venvs/wingjepa-gpu

module load python 2>/dev/null || module load anaconda3
V=$("$CPU_ENV/bin/python" -c "import torch; print(torch.__version__.split('+')[0])")
echo "torch version of the main environment: $V"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
# the default Linux wheels on PyPI are the CUDA builds; no cache: the home quota is small
python -m pip install --no-cache-dir "torch==$V"
python -m pip install --no-cache-dir -e ".[dev]"
python -c "import torch; print('torch', torch.__version__, '| CUDA build', torch.version.cuda)"
echo "GPU environment ready: VENV=$ENV_DIR (no GPU on the login node, so CUDA is only checked by bench_device.sbatch)"
