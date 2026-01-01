#!/usr/bin/env bash
set -euo pipefail
out=bench/results_openblas_rayon_gemm1.csv
mkdir -p bench
TILE=32
echo "openblas,rayon,tile,size,tiled_blas,blas" > "$out"
for ob in 1 2 4 8; do
  for rt in 1 2 4 8; do
    echo "Running OPENBLAS_NUM_THREADS=$ob RAYON_NUM_THREADS=$rt"
    OPENBLAS_NUM_THREADS=$ob OMP_NUM_THREADS=$ob MKL_NUM_THREADS=$ob RAYON_NUM_THREADS=$rt GEMM_THREADS=1 \
      cargo run -p gemm-example -- 64x64x64,128x128x128,256x256x128 32 3 2>&1 | \
    awk -v ob="$ob" -v rt="$rt" -v tile="$TILE" '
      /^--- Running SGEMM test for size/ {size=$7}
      /^  Tiled-BLAS:/ {tiled=$(NF-1)}
      /^  BLAS   :/ {blas=$(NF-1); print ob","rt","tile","size","tiled","blas}
    '
  done
done >> "$out"

echo "Wrote $out"