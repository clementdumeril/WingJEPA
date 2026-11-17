Cross-check of wingjepa/sim/advanced_plane.py against the Fixed-Wing C++ model (cpp/include/fixed_wing/dynamics.h)
====================================================================================================================

Files
  compare.py              make: random states/commands in the C++ conventions -> states.csv
                          check: converts each row to WingJEPA conventions, runs AdvancedPlane.forces/derivatives/step
                          and compares with (1) a numpy line-by-line transliteration of the C++ (always) and
                          (2) the real C++ output (--cpp). Exit status 1 on failure.
  compare_advanced.cpp    reads states.csv, calls advanced_plane(p), advanced_aerodynamics, derivative, rk4 +
                          post_integration, writes cpp_out.csv (force, moment in FLU; dx; next state).
  stub/rl_tools/...       empty stand-in for RLtools' environments.h (fixed_wing.h only derives a tag class from it),
                          so that no RLtools checkout is needed.
  states.csv              400 rows generated locally (seed 0): 102 past the stall, 200 with wind, dt 0.01 / 0.02.
  run_savio.sh            all of the below in one separate bash process.

C++ input columns: id, px py pz (z-up world), qw qx qy qz (body FLU -> world), vx vy vz (world, ground-relative),
wx wy wz (FLU), a_ail a_ele a_rud a_thr (actual: rad, throttle 0-1), c_ail c_ele c_rud c_thr (commands), wind_x
wind_y wind_z (world), dt.

Pass criteria: forces, moments and all derivatives agree to 1e-9 (relative); surfaces and rotor speed after one step
to 1e-9; the rigid-body state after one RK4 step is integrated in different coordinates (Euler/body velocity vs
quaternion/world velocity), so the two only agree to O(dt^5): the check requires the discrepancy to shrink at least
16x per halving of dt (observed ~31x) and prints its size; the numpy transliteration must match the C++ on
everything to 1e-9 (relative; floors of 1e-3 N, 1e-2, 1), next state included.

Locally (Windows, no compiler):
  cd WingJEPA
  python tools/compare_advanced_cpp/compare.py make  --out tools/compare_advanced_cpp/states.csv
  python tools/compare_advanced_cpp/compare.py check --states tools/compare_advanced_cpp/states.csv

On Savio (login node is enough, < 1 min). The Fixed-Wing repository must be on Savio first, e.g. from the PC:
  scp -r C:/Users/admin/OneDrive/Desktop/Work/Capstones/Fixed-Wing/cpp/include savio:/global/scratch/users/<savio_login>/Fixed-Wing/cpp/
(only cpp/include is needed) or a git clone of it. Then, on Savio:

  cd /global/scratch/users/$USER/WingJEPA          # with this folder pulled / copied
  bash tools/compare_advanced_cpp/run_savio.sh /global/scratch/users/$USER/Fixed-Wing

which does exactly:
  bash -c 'source cluster/compiler.sh; \
    g++ -O2 -std=c++17 -DFW_AIRPLANE=1 -I "$FW/cpp/include" -I tools/compare_advanced_cpp/stub \
        tools/compare_advanced_cpp/compare_advanced.cpp -o tools/compare_advanced_cpp/compare_advanced'
  tools/compare_advanced_cpp/compare_advanced tools/compare_advanced_cpp/states.csv tools/compare_advanced_cpp/cpp_out.csv
  source /global/scratch/users/$USER/venvs/wingjepa/bin/activate
  python tools/compare_advanced_cpp/compare.py check --states tools/compare_advanced_cpp/states.csv \
         --cpp tools/compare_advanced_cpp/cpp_out.csv

(cluster/compiler.sh ends with "set -euo pipefail": source it inside bash -c or the script, not in the login shell.)
Expected: "ALL OK", with the C++ lines at round-off level (~1e-12 or below).
