# Sourced by job scripts: load the newest gcc module, since torch.compile builds C++20 code and the
# system gcc (8.5 on Savio's Rocky 8) does not accept -std=c++20. Harmless when nothing is compiled.
GCC_MOD=$(module -t avail gcc 2>&1 | grep -E '^gcc/[0-9]' | sort -V | tail -1)
if [ -n "$GCC_MOD" ]; then
  module load "$GCC_MOD" 2>/dev/null || true
fi
echo "C++ compiler: $(g++ --version | head -1)"
