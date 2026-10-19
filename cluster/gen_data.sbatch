#!/bin/bash
# Génération des vols : un shard par tâche du tableau.
# Usage : sbatch --array=0-19 cluster/gen_data.sbatch configs/data_v1.yaml train 20
#         sbatch --array=0-3  cluster/gen_data.sbatch configs/data_v1.yaml test_long 4
# Puis le contrôle : sbatch cluster/job_cpu.sbatch scripts/check_dataset.py --dir <out_dir>/<name> --set train
#SBATCH --job-name=wingjepa-gen
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_normal
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=runs/slurm/%x-%A_%a.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=1
python scripts/generate_flights.py --config "$1" --set "$2" --shard "$SLURM_ARRAY_TASK_ID" --n-shards "$3"
