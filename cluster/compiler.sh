# Sourced by job scripts: load the newest gcc module, since torch.compile builds C++20 code and the
# system gcc (8.5 on Savio's Rocky 8) does not accept -std=c++20. Harmless when nothing is compiled.
# Strict mode is relaxed meanwhile (module does not support it) and restored afterwards: a problem here
# must never stop a training job.
set +euo pipefail
GCC_MOD=$(module -t avail gcc 2>&1 | grep -E '^gcc/[0-9]' | sort -V | tail -1)
[ -n "$GCC_MOD" ] && module load "$GCC_MOD" > /dev/null 2>&1
echo "C++ compiler: $(g++ --version 2>/dev/null | head -1)"
set -euo pipefail
