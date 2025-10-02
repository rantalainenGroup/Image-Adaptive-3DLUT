#!/usr/bin/env bash
# fail fast + clear pipes
set -euo pipefail

# CUDA from conda env
export CUDA_HOME="${CUDA_HOME:-${CONDA_PREFIX:-}}"
export PATH="$CUDA_HOME/bin:$PATH"

# Base include/lib
export CPATH="$CUDA_HOME/include:${CPATH:-}"
export LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"

# Torch runtime libs (fixes 'libc10.so' import errors)
TORCH_LIB="$(python -c 'import os,torch;print(os.path.join(os.path.dirname(torch.__file__),"lib"))')"
export LD_LIBRARY_PATH="$TORCH_LIB:$LD_LIBRARY_PATH"

# Your GPU: RTX 2080 Ti (Turing, SM 7.5)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"

echo "CUDA_HOME = $CUDA_HOME"
command -v nvcc >/dev/null || { echo "nvcc not found on PATH"; exit 1; }
nvcc --version || true

# Quick header sanity (helps catch cusparse/cuda_runtime missing)
for hdr in cuda_runtime.h cusparse.h; do
  if ! echo "$CPATH" | tr ':' '\n' | xargs -I{} bash -lc 'test -f "{}/'"$hdr"'"' ; then
    echo " Header $hdr not found in CPATH. Current CPATH:"
    echo "    $CPATH"
  fi
done

# --- Clean & build in place (single job for readable errors) ---
python setup.py clean
MAX_JOBS="${MAX_JOBS:-1}" python setup.py build_ext --inplace -v

echo " Build finished. .so is in trilinear_cpp/"