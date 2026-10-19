#!/bin/bash
# Test rapide : 40 vols de 30 s (2 shards), puis les contrôles. Tout le résultat est dans le log.
# Usage : sbatch cluster/smoke_test.sbatch
#SBATCH --job-name=wingjepa-smoke
#SBATCH --account=ic_piml
#SBATCH --partition=savio4_htc
#SBATCH --qos=savio_debug
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --output=runs/slurm/%x-%j.out

set -euo pipefail
mkdir -p runs/slurm
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
export OMP_NUM_THREADS=1
test -f third_party/umn/FASER_Aero.mat || { echo "Lancer d'abord : bash scripts/fetch_umn.sh"; exit 1; }
OUT=/global/scratch/users/$USER/WingJEPA/data
python scripts/generate_flights.py --config configs/data_v1.yaml --set smoke --shard 0 --n-shards 2
python scripts/generate_flights.py --config configs/data_v1.yaml --set smoke --shard 1 --n-shards 2
python scripts/check_dataset.py --dir "$OUT/data_v1" --set smoke
