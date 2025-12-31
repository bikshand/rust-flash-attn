# rust-flash-attn
FlashAttention in Rust

## Examples

This repository contains a small example that demonstrates validating a Rust GEMM implementation against a BLAS `cblas_sgemm` (e.g., MKL):

- `gemm-example/` - a small binary that computes SGEMM in Rust and compares against a runtime-loaded BLAS (prefers `libmkl_rt.so`).

See `gemm-example/README.md` for instructions on building and running the example.

## Flash Attention in Rust

TODO
