#!/bin/bash
# Collects results and redraws the figures into export/ (ignored by git), then packs export.tgz.
# Quick version of make_assets.sbatch without the GIF and the simulator validation; runs on a login node.
# Usage (repository root, environment activated): bash cluster/export.sh
set -euo pipefail
python scripts/collect_results.py --out export/results > /dev/null
python scripts/make_figures.py --results export/results --out export/assets
tar czf export.tgz -C export results assets
echo "export.tgz: $(tar tzf export.tgz | grep -c eval_position) position files, $(ls export/results | wc -l) entries"
