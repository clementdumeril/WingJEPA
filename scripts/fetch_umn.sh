#!/bin/bash
# Télécharge le simulateur UMN UAV Lab (UMNUAV_sim_v4, BSD) et en extrait les tables aéro NASA de
# l'Ultra Stick 120 (FASER_Aero.mat) dans third_party/umn/ (exclu de git). Une fois, depuis la racine.
set -euo pipefail
DEST=third_party/umn
mkdir -p "$DEST"
TMP=$(mktemp -d)
curl -fsSL -o "$TMP/UMNUAV_sim_v4.zip" "https://dept.aem.umn.edu/~uav/UMNUAV_sim_v4.zip"
unzip -q -o -j "$TMP/UMNUAV_sim_v4.zip" "UMNUAV_sim_v4/Simulation/Libraries/FASER_Aero.mat" -d "$DEST"
unzip -q -o -j "$TMP/UMNUAV_sim_v4.zip" "UMNUAV_sim_v4/Software/FlightCode/LICENSE.txt" -d "$DEST"
rm -rf "$TMP"
ls -l "$DEST"
