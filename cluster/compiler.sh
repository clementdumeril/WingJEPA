# Sourced by job scripts: load a gcc module recent enough for torch.compile, which builds C++20 code
# (the system gcc 8.5 on Savio's Rocky 8 rejects -std=c++20). gcc 11 is preferred: Savio's Python is built
# with it, so the compiled kernels link against the libstdc++ already loaded in the process (code built
# by gcc 13 needs GLIBCXX_3.4.32, which that libstdc++ lacks). Harmless when nothing is compiled.
# Strict mode is relaxed meanwhile (module does not support it) and restored afterwards: a problem here
# must never stop a training job.
set +euo pipefail
AVAIL=$(module -t avail gcc 2>&1 | grep -E '^gcc/[0-9]' | sort -V)
GCC_MOD=$(echo "$AVAIL" | grep -E '^gcc/11' | tail -1)
[ -z "$GCC_MOD" ] && GCC_MOD=$(echo "$AVAIL" | tail -1)
[ -n "$GCC_MOD" ] && module load "$GCC_MOD" > /dev/null 2>&1
echo "C++ compiler: $(g++ --version 2>/dev/null | head -1)"
set -euo pipefail
