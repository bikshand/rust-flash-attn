use libloading::Library;
use rayon::prelude::*;
use rand::Rng;
use std::error::Error;
use std::time::Instant;

// -------------------------------
// Shape abstraction
// -------------------------------
pub trait Shape {
    const M: usize;
    const N: usize;
    const K: usize;
}
pub struct ShapeMNK<const M: usize, const N: usize, const K: usize>;
impl<const M: usize, const N: usize, const K: usize> Shape for ShapeMNK<M, N, K> {
    const M: usize = M;
    const N: usize = N;
    const K: usize = K;
}

// -------------------------------
// Tile abstraction
// -------------------------------
pub struct Tile<const TM: usize, const TN: usize, const TK: usize>;
pub trait TilePolicy {
    const TM: usize;
    const TN: usize;
    const TK: usize;
}
impl<const TM: usize, const TN: usize, const TK: usize> TilePolicy for Tile<TM, TN, TK> {
    const TM: usize = TM;
    const TN: usize = TN;
    const TK: usize = TK;
}

// -------------------------------
// BLAS loader
// -------------------------------
fn try_load_blas() -> Result<Library, Box<dyn Error>> {
    for name in ["libmkl_rt.so", "libopenblas.so", "libblas.so"] {
        if let Ok(lib) = unsafe { Library::new(name) } {
            println!("Loaded BLAS library: {}", name);
            return Ok(lib);
        }
    }
    Err("Could not find BLAS library".into())
}

// -------------------------------
// Call cblas_sgemm
// -------------------------------
unsafe fn call_cblas_sgemm(
    lib: &Library,
    m: i32,
    n: i32,
    k: i32,
    alpha: f32,
    a: *const f32,
    lda: i32,
    b: *const f32,
    ldb: i32,
    beta: f32,
    c: *mut f32,
    ldc: i32,
) -> Result<(), Box<dyn Error>> {
    type CblasFn =
        unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f32, *const f32, i32, *const f32, i32, f32, *mut f32, i32);
    let f: libloading::Symbol<CblasFn> = lib.get(b"cblas_sgemm\0")?;
    f(101, 111, 111, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
    Ok(())
}

// -------------------------------
// Tiled GEMM with safe parallel slices
// -------------------------------
fn tiled_gemm_blas_strided<S: Shape>(
    lib: &Library,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    alpha: f32,
    beta: f32,
    tm: usize,
    tn: usize,
    tk: usize,
) -> Result<(), Box<dyn Error>> {
    let m = S::M;
    let n = S::N;
    let k = S::K;

    // Partition C into non-overlapping slices safely
    let mut c_rem = &mut c[..];
    let mut c_chunks: Vec<&mut [f32]> = Vec::new();
    while !c_rem.is_empty() {
        let row_tile = tm.min(c_rem.len() / n);
        let (c_sub, rest) = c_rem.split_at_mut(row_tile * n);
        c_chunks.push(c_sub);
        c_rem = rest;
    }

    // Parallel loop over row tiles
    c_chunks.into_par_iter().enumerate().for_each(|(tile_idx, c_sub)| {
        let i0 = tile_idx * tm;
        for j0 in (0..n).step_by(tn) {
            for kk in (0..k).step_by(tk) {
                let m_tile = c_sub.len() / n;
                let n_tile = (j0 + tn).min(n) - j0;
                let k_tile = (kk + tk).min(k) - kk;

                let a_ptr = unsafe { a.as_ptr().add(i0 * k + kk) };
                let b_ptr = unsafe { b.as_ptr().add(kk * n + j0) };
                let c_ptr = unsafe { c_sub.as_mut_ptr().add(j0) };

                unsafe {
                    call_cblas_sgemm(
                        lib,
                        m_tile as i32,
                        n_tile as i32,
                        k_tile as i32,
                        alpha,
                        a_ptr,
                        k as i32,
                        b_ptr,
                        n as i32,
                        if kk == 0 { beta } else { 1.0 },
                        c_ptr,
                        n as i32,
                    )
                    .unwrap();
                }
            }
        }
    });

    Ok(())
}

// -------------------------------
// Max difference
// -------------------------------
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

// -------------------------------
// Parse CSV env variable
// -------------------------------
fn parse_csv(var: &str) -> Vec<usize> {
    std::env::var(var)
        .ok()
        .map(|s| s.split(',').filter_map(|v| v.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![32, 64, 128, 256])
}

// -------------------------------
// Main
// -------------------------------
fn main() -> Result<(), Box<dyn Error>> {
    const M: usize = 8192;
    const N: usize = 8192;
    const K: usize = 256;

    let lib = try_load_blas()?;
    let mut rng = rand::thread_rng();

    let mut a = vec![0f32; M * K];
    let mut b = vec![0f32; K * N];
    let mut c_blas = vec![0f32; M * N];
    let mut c_tiled = vec![0f32; M * N];

    for v in &mut a {
        *v = rng.gen_range(-1.0..1.0);
    }
    for v in &mut b {
        *v = rng.gen_range(-1.0..1.0);
    }

    // Tiles
    let allowed_tm = parse_csv("TILE_M");
    let allowed_tn = parse_csv("TILE_N");
    let allowed_tk = parse_csv("TILE_K");
    println!("Allowed TM={:?} TN={:?} TK={:?}", allowed_tm, allowed_tn, allowed_tk);

    // Threads
    let allowed_blas_threads = parse_csv("BLAS_THREADS");
    let allowed_rayon_threads = parse_csv("RAYON_THREADS");
    println!(
        "Allowed BLAS_THREADS={:?} RAYON_THREADS={:?}",
        allowed_blas_threads, allowed_rayon_threads
    );

    // --- Run baseline BLAS and tiled GEMM ---
    for &bt in &allowed_blas_threads {
        for &rt in &allowed_rayon_threads {
            println!("Running BLAS GEMM: BLAS_THREADS={} RAYON_THREADS={}", bt, rt);
            std::env::set_var("OPENBLAS_NUM_THREADS", bt.to_string());
            std::env::set_var("RAYON_NUM_THREADS", rt.to_string());
            let _ = rayon::ThreadPoolBuilder::new().num_threads(rt).build_global();

            // Clear output matrices
            for v in &mut c_blas { *v = 0.0; }

            // --- BLAS GEMM ---
            let start_blas = Instant::now();
            unsafe {
                call_cblas_sgemm(
                    &lib,
                    M as i32,
                    N as i32,
                    K as i32,
                    1.0,
                    a.as_ptr(),
                    K as i32,
                    b.as_ptr(),
                    N as i32,
                    0.0,
                    c_blas.as_mut_ptr(),
                    N as i32,
                )?;
            }
            let dur_blas = start_blas.elapsed();
            println!("BLAS GEMM: {:>6.3} s\n", dur_blas.as_secs_f64());

            // --- Sweep tiled GEMM ---
            for &tm in &allowed_tm {
                for &tn in &allowed_tn {
                    for &tk in &allowed_tk {
                        if tm > M || tn > N || tk > K { continue; }

                        // Clear tiled output
                        for v in &mut c_tiled { *v = 0.0; }

                        println!(
                            "Tiled GEMM TM={} TN={} TK={} BLAS_THREADS={} RAYON_THREADS={}",
                            tm, tn, tk, bt, rt
                        );

                        let start = Instant::now();
                        tiled_gemm_blas_strided::<ShapeMNK<M, N, K>>(
                            &lib, &a, &b, &mut c_tiled, 1.0, 0.0, tm, tn, tk,
                        )?;
                        let dur_tiled = start.elapsed();

                        let diff = max_abs_diff(&c_blas, &c_tiled);
                        println!(
                            "  {:>6.3} s, max diff = {:e}\n",
                            dur_tiled.as_secs_f64(),
                            diff
                        );
                    }
                }
            }
        }
    }

    Ok(())
}

