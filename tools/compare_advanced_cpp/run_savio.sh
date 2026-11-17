#!/bin/bash
# Builds compare_advanced against the Fixed-Wing headers, runs it on states.csv and compares with AdvancedPlane.
# Usage, from the WingJEPA repository root on a Savio login node:
#   bash tools/compare_advanced_cpp/run_savio.sh /path/to/Fixed-Wing
# A separate bash process, so that the "set -euo pipefail" left by cluster/compiler.sh cannot close the login shell.
set -eo pipefail
FW=${1:?usage: bash tools/compare_advanced_cpp/run_savio.sh /path/to/Fixed-Wing}
HERE=tools/compare_advanced_cpp
test -f "$FW/cpp/include/fixed_wing/dynamics.h" || { echo "no $FW/cpp/include/fixed_wing/dynamics.h"; exit 1; }
source cluster/compiler.sh
g++ -O2 -std=c++17 -DFW_AIRPLANE=1 -I "$FW/cpp/include" -I "$HERE/stub" "$HERE/compare_advanced.cpp" -o "$HERE/compare_advanced"
source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
test -f "$HERE/states.csv" || python "$HERE/compare.py" make --out "$HERE/states.csv"
"$HERE/compare_advanced" "$HERE/states.csv" "$HERE/cpp_out.csv"
python "$HERE/compare.py" check --states "$HERE/states.csv" --cpp "$HERE/cpp_out.csv"
