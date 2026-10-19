#!/bin/bash
# Crée l'environnement Python de WingJEPA sur Savio, dans l'espace scratch.
# À lancer depuis la racine du dépôt cloné sur Savio : bash cluster/setup_env.sh
set -euo pipefail

ENV_DIR=/global/scratch/users/$USER/venvs/wingjepa   # à vérifier : chemin scratch Savio

module load python 2>/dev/null || module load anaconda3   # à vérifier : nom du module Python (`module avail python`)

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
python -c "import jsbsim, torch; print('jsbsim', jsbsim.__version__, '| torch', torch.__version__)"
mkdir -p runs/slurm   # Slurm ouvre le fichier de sortie avant le script : le dossier doit exister
bash scripts/fetch_umn.sh   # tables NASA de l'Ultra Stick 120 (simulateur UMN), depuis le nœud de connexion
echo "Environnement prêt : source $ENV_DIR/bin/activate"
