#!/bin/bash
# Downloads the UMN UAV Lab simulator (UMNUAV_sim_v4, BSD) and extracts the NASA aero tables of
# the Ultra Stick 120 (FASER_Aero.mat) into third_party/umn/ (excluded from git). Once, from the repo root.
set -euo pipefail
DEST=third_party/umn
mkdir -p "$DEST"
TMP=$(mktemp -d)
curl -fsSL -o "$TMP/UMNUAV_sim_v4.zip" "https://dept.aem.umn.edu/~uav/UMNUAV_sim_v4.zip"
unzip -q -o -j "$TMP/UMNUAV_sim_v4.zip" "UMNUAV_sim_v4/Simulation/Libraries/FASER_Aero.mat" -d "$DEST"
unzip -q -o -j "$TMP/UMNUAV_sim_v4.zip" "UMNUAV_sim_v4/Software/FlightCode/LICENSE.txt" -d "$DEST"
rm -rf "$TMP"
ls -l "$DEST"
