# gemm-example

This small Rust example computes a single-precision GEMM (SGEMM) two ways:

- A simple naive Rust triple-loop implementation (reference)
- Calling a BLAS library's `cblas_sgemm` at runtime (the code prefers `libmkl_rt.so`)

The program compares the two results and prints the maximum absolute difference.

Usage

- Build:

  cargo build -p gemm-example --release

- Run (without MKL present):

  cargo run -p gemm-example --release

  The program will run the Rust reference implementation and skip the BLAS run with a warning.

- Run (with MKL):

  If you have Intel MKL installed, ensure `libmkl_rt.so` is available on `LD_LIBRARY_PATH` or via the system linker path. For example:

  ```bash
  export MKLROOT=/opt/intel/mkl
  export LD_LIBRARY_PATH="$MKLROOT/lib/intel64:$LD_LIBRARY_PATH"
  cargo run -p gemm-example --release
  ```

  The program will load `libmkl_rt.so`, run `cblas_sgemm`, and compare results.

Notes

- The example uses dynamic loading (libloading) to avoid hard-linking against MKL at build time, so it compiles even if MKL isn't installed.
- Numerical differences between implementations are expected; the default tolerance is 1e-3 for single precision.

CI

- The repository includes a GitHub Actions workflow (`.github/workflows/gemm.yml`) that will install OpenBLAS on the runner only if it's not already present and then build and run the `gemm-example` to validate the implementation automatically.

Multiple sizes

- By default the example runs a small set of sizes (`64x64x64`, `128x128x128`, `256x256x128`).
- To run custom sizes, set the `GEMM_SIZES` environment variable or pass a single CLI argument with a comma-separated list of sizes in `MxNxK` form. Examples:

```bash
# env var
GEMM_SIZES="32x32x32,64x64x32" cargo run -p gemm-example --release

# CLI arg
cargo run -p gemm-example --release -- "32x32x32,64x64x32"
```

Tiled implementation

- A blocked/tiled SGEMM (`tiled_sgemm`) was added and is validated alongside the naive implementation. The tile size is configurable via the `GEMM_TILE` env var or the second CLI arg (default: 32).

```bash
# Use tile size 16 via env var
GEMM_TILE=16 cargo run -p gemm-example --release

# Or pass tile as a second CLI arg
cargo run -p gemm-example --release -- "64x64x64" 16
```

Parallel + SIMD implementation

- A parallel + SIMD implementation (`tiled_sgemm_parallel_simd`) was added. It uses `rayon` to parallelize over row tiles and the `wide` crate (`f32x8`) to vectorize inner loops.
- Control the number of threads with `GEMM_THREADS` env var or pass as the fourth CLI arg. When `GEMM_THREADS` is set the program will also set the common BLAS threading env vars (`OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `OMP_NUM_THREADS`) to the same value and attempt to configure Rayon global thread pool.

```bash
# Use 8 threads and tile 32
GEMM_THREADS=8 GEMM_TILE=32 cargo run -p gemm-example --release

# Or pass via CLI args: sizes, tile, repeats, threads
cargo run -p gemm-example --release -- "64x64x64" 32 3 8
```

- The run will check naive, tiled, and parallel+SIMD implementations against BLAS and report timings/GFLOPS for all.

Timings

- By default each implementation is timed over `3` runs and the average time and GFLOPS are reported.
- Configure repeats via `GEMM_REPEATS` env var or as the third CLI arg: `GEMM_REPEATS=5 cargo run -p gemm-example --release`.
