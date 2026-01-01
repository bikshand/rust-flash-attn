use libloading::Library;
use rand::Rng;
use std::error::Error;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_NO_TRANS: i32 = 111;

fn naive_sgemm(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32]) {
    // A: m x k, B: k x n, C: m x n
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = alpha * sum + beta * c[i * n + j];
        }
    }
}

fn tiled_sgemm(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32], tile: usize) {
    // Blocked (tiled) matrix multiply. c will be updated as: c = alpha*A*B + beta*c
    // We'll scale c by beta first (if needed) then add alpha * A*B in blocks (scalar implementation).
    if beta != 1.0 {
        for v in c.iter_mut() {
            *v *= beta;
        }
    }

    let tile = std::cmp::max(1, tile);

    for ii in (0..m).step_by(tile) {
        let i_end = (ii + tile).min(m);
        for kk in (0..k).step_by(tile) {
            let k_end = (kk + tile).min(k);
            for jj in (0..n).step_by(tile) {
                let j_end = (jj + tile).min(n);
                for i in ii..i_end {
                    for p in kk..k_end {
                        let a_ip = a[i * k + p];
                        for j in jj..j_end {
                            c[i * n + j] += alpha * a_ip * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
}

/// Parallel + SIMD tiled SGEMM using `rayon` for parallelism and `wide` for f32x8 vector ops.
fn tiled_sgemm_parallel_simd(m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32], tile: usize, threads: Option<usize>) {
    // scale c by beta (if needed)
    if beta != 1.0 {
        for v in c.iter_mut() {
            *v *= beta;
        }
    }

    let tile = std::cmp::max(1, tile);

    // Configure rayon thread pool for the operation
    let pool = if let Some(t) = threads {
        Some(rayon::ThreadPoolBuilder::new().num_threads(t).build().expect("Failed to build thread pool"))
    } else {
        None
    };

    // Vector width: 8 floats (f32x8)
    use wide::f32x8;

    let c_ptr_addr = c.as_mut_ptr() as usize;

    let do_work = || {
        use rayon::prelude::*;
        let row_tiles: Vec<usize> = (0..m).step_by(tile).collect();
        row_tiles.into_par_iter().for_each(|ii| {
            let c_ptr = c_ptr_addr as *mut f32;
            let i_end = (ii + tile).min(m);
            for kk in (0..k).step_by(tile) {
                let k_end = (kk + tile).min(k);
                for jj in (0..n).step_by(tile) {
                    let j_end = (jj + tile).min(n);
                    for i in ii..i_end {
                        for p in kk..k_end {
                            let a_ip = a[i * k + p];
                            let base_b = p * n;
                            let base_c = i * n;

                            // Vectorized inner loop (process 8 floats at a time)
                            let mut j = jj;
                            while j + 8 <= j_end {
                                // safe copy to temporary arrays then convert to SIMD vectors
                                let mut tmpb: [f32; 8] = [0.0; 8];
                                tmpb.copy_from_slice(&b[base_b + j..base_b + j + 8]);
                                let bvec = f32x8::from(tmpb);

                                // load c via raw pointer into temp
                                let mut tmpc: [f32; 8] = [0.0; 8];
                                unsafe {
                                    std::ptr::copy_nonoverlapping(c_ptr.add(base_c + j), tmpc.as_mut_ptr(), 8);
                                }
                                let mut cvec = f32x8::from(tmpc);

                                cvec = cvec + (bvec * f32x8::splat(alpha * a_ip));

                                let out = cvec.to_array();
                                unsafe {
                                    std::ptr::copy_nonoverlapping(out.as_ptr(), c_ptr.add(base_c + j), 8);
                                }
                                j += 8;
                            }
                            // Remainder
                            for j in j..j_end {
                                unsafe {
                                    let p = c_ptr.add(base_c + j);
                                    *p = *p + alpha * a_ip * b[base_b + j];
                                }
                            }
                        }
                    }
                }
            }
        });
    };

    if let Some(p) = pool {
        p.install(do_work);
    } else {
        do_work();
    }
}

fn try_load_mkl() -> Result<Library, Box<dyn Error>> {
    // Try common library names; prefer MKL runtime
    let names = [
        "libmkl_rt.so",
        "libopenblas.so",
        "libopenblas.so.0",
        "libblas.so",
        "libblas.so.3",
    ];
    for name in names {
        // `Library::new` is `unsafe` in recent libloading versions
        if let Ok(lib) = unsafe { Library::new(name) } {
            println!("Loaded BLAS library: {}", name);
            return Ok(lib);
        }
    }
    Err("Could not find libmkl_rt.so/libopenblas.so/libblas.so on LD_LIBRARY_PATH".into())
}

fn call_cblas_sgemm(lib: &Library, m: i32, n: i32, k: i32, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32]) -> Result<(), Box<dyn Error>> {
    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f32, *const f32, i32, *const f32, i32, f32, *mut f32, i32)> = lib.get(b"cblas_sgemm\0")?;
        func(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m,
            n,
            k,
            alpha,
            a.as_ptr(),
            k,
            b.as_ptr(),
            n,
            beta,
            c.as_mut_ptr(),
            n,
        );
    }
    Ok(())
}

/// Call cblas_sgemm allowing explicit leading dimensions and pointers to tiles.
fn call_cblas_sgemm_strided(lib: &Library, m: i32, n: i32, k: i32, alpha: f32, a_ptr: *const f32, lda: i32, b_ptr: *const f32, ldb: i32, beta: f32, c_ptr: *mut f32, ldc: i32) -> Result<(), Box<dyn Error>> {
    unsafe {
        let func: libloading::Symbol<unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f32, *const f32, i32, *const f32, i32, f32, *mut f32, i32)> = lib.get(b"cblas_sgemm\0")?;
        func(
            CBLAS_ROW_MAJOR,
            CBLAS_NO_TRANS,
            CBLAS_NO_TRANS,
            m,
            n,
            k,
            alpha,
            a_ptr,
            lda,
            b_ptr,
            ldb,
            beta,
            c_ptr,
            ldc,
        );
    }
    Ok(())
}

/// Tiled SGEMM where each tile multiply is delegated to BLAS. The outer loops are in Rust.
fn tiled_sgemm_blas_tiles(lib: &Library, m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32], tile: usize, threads: Option<usize>) -> Result<(), Box<dyn Error>> {
    // Scale C by beta first (if needed)
    if beta != 1.0 {
        for v in c.iter_mut() {
            *v *= beta;
        }
    }

    let tile = std::cmp::max(1, tile);

    // Resolve the cblas function pointer once so we can call it safely from threads
    type CblasFn = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f32, *const f32, i32, *const f32, i32, f32, *mut f32, i32);
    let func: libloading::Symbol<CblasFn> = unsafe { lib.get(b"cblas_sgemm\0")? };
    let fptr: CblasFn = *func;

    let row_tiles: Vec<usize> = (0..m).step_by(tile).collect();

    if let Some(t) = threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().expect("Failed to build thread pool");
        pool.scope(|s| {
            // Partition C into non-overlapping sub-slices and spawn tasks for each
        let mut c_rem = &mut c[..];
        for &ii in &row_tiles {
            // compute row tile bounds
            let i_end = (ii + tile).min(m);
            let rows = i_end - ii;
            let (c_sub, rest) = c_rem.split_at_mut(rows * n);
            c_rem = rest;
            let ii_local = ii;
            // spawn a task that owns the sub-slice by mutable borrow (scope allows non-'static borrows)
            s.spawn(move |_| {
                for kk in (0..k).step_by(tile) {
                    let k_end = (kk + tile).min(k);
                    let k_tile = (k_end - kk) as i32;
                    for jj in (0..n).step_by(tile) {
                        let j_end = (jj + tile).min(n);
                        let n_tile = (j_end - jj) as i32;
                        unsafe {
                            let a_ptr = a.as_ptr().add(ii_local * k + kk);
                            let b_ptr = b.as_ptr().add(kk * n + jj);
                            // c_sub has rows for ii..i_end with full row stride 'n'
                            let c_ptr = c_sub.as_mut_ptr().add(jj);
                            fptr(
                                CBLAS_ROW_MAJOR,
                                CBLAS_NO_TRANS,
                                CBLAS_NO_TRANS,
                                rows as i32,
                                n_tile,
                                k_tile,
                                alpha,
                                a_ptr,
                                k as i32,
                                b_ptr,
                                n as i32,
                                1.0f32,
                                c_ptr,
                                n as i32,
                            );
                        }
                    }
                }
            });
        }
        });
    } else {
        // Use global Rayon thread pool (if configured via RAYON_NUM_THREADS) for parallelism across row tiles
        rayon::scope(|s| {
            // Partition C into non-overlapping sub-slices and spawn scoped tasks that will execute on the global pool
            let mut c_rem = &mut c[..];
            for &ii in &row_tiles {
                let i_end = (ii + tile).min(m);
                let rows = i_end - ii;
                let (c_sub, rest) = c_rem.split_at_mut(rows * n);
                c_rem = rest;
                let ii_local = ii;
                s.spawn(move |_| {
                    for kk in (0..k).step_by(tile) {
                        let k_end = (kk + tile).min(k);
                        let k_tile = (k_end - kk) as i32;
                        for jj in (0..n).step_by(tile) {
                            let j_end = (jj + tile).min(n);
                            let n_tile = (j_end - jj) as i32;
                            unsafe {
                                let a_ptr = a.as_ptr().add(ii_local * k + kk);
                                let b_ptr = b.as_ptr().add(kk * n + jj);
                                // c_sub has rows for ii_local..i_end with full row stride 'n'
                                let c_ptr = c_sub.as_mut_ptr().add(jj);
                                fptr(
                                    CBLAS_ROW_MAJOR,
                                    CBLAS_NO_TRANS,
                                    CBLAS_NO_TRANS,
                                    rows as i32,
                                    n_tile,
                                    k_tile,
                                    alpha,
                                    a_ptr,
                                    k as i32,
                                    b_ptr,
                                    n as i32,
                                    1.0f32,
                                    c_ptr,
                                    n as i32,
                                );
                            }
                        }
                    }
                });
            }
        });
    }

    Ok(())
}

/// Tiled SGEMM where each tile is copied into contiguous buffers (packed), passed to BLAS,
/// and the resulting tile is copied back to the destination matrix.
fn tiled_sgemm_blas_tiles_copy(lib: &Library, m: usize, n: usize, k: usize, alpha: f32, a: &[f32], b: &[f32], beta: f32, c: &mut [f32], tile: usize, threads: Option<usize>) -> Result<(), Box<dyn Error>> {
    // Scale C by beta first (if needed)
    if beta != 1.0 {
        for v in c.iter_mut() {
            *v *= beta;
        }
    }

    let tile = std::cmp::max(1, tile);

    // Resolve BLAS function once for threaded calls
    type CblasFn = unsafe extern "C" fn(i32, i32, i32, i32, i32, i32, f32, *const f32, i32, *const f32, i32, f32, *mut f32, i32);
    let func: libloading::Symbol<CblasFn> = unsafe { lib.get(b"cblas_sgemm\0")? };
    let fptr: CblasFn = *func;

    let row_tiles: Vec<usize> = (0..m).step_by(tile).collect();

    if let Some(t) = threads {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().expect("Failed to build thread pool");
        pool.scope(|s| {
            let mut c_rem = &mut c[..];
            for &ii in &row_tiles {
                let i_end = (ii + tile).min(m);
                let rows = i_end - ii;
                let (c_sub, rest) = c_rem.split_at_mut(rows * n);
                c_rem = rest;
                let ii_local = ii;
                s.spawn(move |_| {
                    for kk in (0..k).step_by(tile) {
                        let k_end = (kk + tile).min(k);
                        let k_tile = k_end - kk;
                        for jj in (0..n).step_by(tile) {
                            let j_end = (jj + tile).min(n);
                            let n_tile = j_end - jj;

                            // Allocate packed buffers per-task/tile
                            let mut a_pack = vec![0f32; rows * k_tile];
                            let mut b_pack = vec![0f32; k_tile * n_tile];
                            let mut c_pack = vec![0f32; rows * n_tile];

                            // Pack A
                            for i in 0..rows {
                                for p in 0..k_tile {
                                    a_pack[i * k_tile + p] = a[(ii_local + i) * k + (kk + p)];
                                }
                            }
                            // Pack B
                            for p in 0..k_tile {
                                for j in 0..n_tile {
                                    b_pack[p * n_tile + j] = b[(kk + p) * n + (jj + j)];
                                }
                            }
                            // Init C pack from current C slice
                            for i in 0..rows {
                                for j in 0..n_tile {
                                    c_pack[i * n_tile + j] = c_sub[i * n + (jj + j)];
                                }
                            }

                            // Call BLAS on packed small tiles
                            unsafe {
                                fptr(
                                    CBLAS_ROW_MAJOR,
                                    CBLAS_NO_TRANS,
                                    CBLAS_NO_TRANS,
                                    rows as i32,
                                    n_tile as i32,
                                    k_tile as i32,
                                    alpha,
                                    a_pack.as_ptr(),
                                    k_tile as i32,
                                    b_pack.as_ptr(),
                                    n_tile as i32,
                                    1.0f32,
                                    c_pack.as_mut_ptr(),
                                    n_tile as i32,
                                );
                            }

                            // Copy back
                            for i in 0..rows {
                                for j in 0..n_tile {
                                    c_sub[i * n + (jj + j)] = c_pack[i * n_tile + j];
                                }
                            }
                        }
                    }
                });
            }
        });
    } else {
        // Single-threaded fallback (same as previous behavior)
        for ii in (0..m).step_by(tile) {
            let i_end = (ii + tile).min(m);
            let m_tile = i_end - ii;
            for kk in (0..k).step_by(tile) {
                let k_end = (kk + tile).min(k);
                let k_tile = k_end - kk;
                for jj in (0..n).step_by(tile) {
                    let j_end = (jj + tile).min(n);
                    let n_tile = j_end - jj;

                    // Allocate packed buffers
                    let mut a_pack = vec![0f32; m_tile * k_tile];
                    let mut b_pack = vec![0f32; k_tile * n_tile];
                    let mut c_pack = vec![0f32; m_tile * n_tile];

                    // Pack A tile (row-major: m_tile rows, k_tile cols)
                    for i in 0..m_tile {
                        for p in 0..k_tile {
                            a_pack[i * k_tile + p] = a[(ii + i) * k + (kk + p)];
                        }
                    }

                    // Pack B tile (row-major representation: k_tile rows, n_tile cols)
                    for p in 0..k_tile {
                        for j in 0..n_tile {
                            b_pack[p * n_tile + j] = b[(kk + p) * n + (jj + j)];
                        }
                    }

                    // Initialize C pack from existing C (already scaled by beta)
                    for i in 0..m_tile {
                        for j in 0..n_tile {
                            c_pack[i * n_tile + j] = c[(ii + i) * n + (jj + j)];
                        }
                    }

                    // Call BLAS on the small packed tiles. Leading dims are k_tile, n_tile, n_tile respectively.
                    call_cblas_sgemm_strided(
                        lib,
                        m_tile as i32,
                        n_tile as i32,
                        k_tile as i32,
                        alpha,
                        a_pack.as_ptr(),
                        k_tile as i32,
                        b_pack.as_ptr(),
                        n_tile as i32,
                        1.0f32,
                        c_pack.as_mut_ptr(),
                        n_tile as i32,
                    )?;

                    // Copy result back to C
                    for i in 0..m_tile {
                        for j in 0..n_tile {
                            c[(ii + i) * n + (jj + j)] = c_pack[i * n_tile + j];
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

fn max_abs_diff(x: &[f32], y: &[f32]) -> f32 {
    x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max)
}

fn parse_sizes(s: &str) -> Result<Vec<(usize, usize, usize)>, Box<dyn Error>> {
    let mut out = Vec::new();
    for part in s.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let nums: Vec<&str> = p.split('x').collect();
        if nums.len() != 3 {
            return Err(format!("Invalid size '{}', expected MxNxK", p).into());
        }
        let m = nums[0].parse()?;
        let n = nums[1].parse()?;
        let k = nums[2].parse()?;
        out.push((m, n, k));
    }
    Ok(out)
}

fn parse_tile(s: &str) -> Result<usize, Box<dyn Error>> {
    Ok(s.parse::<usize>()?)
}

fn main() -> Result<(), Box<dyn Error>> {
    // Defaults: a few small-to-medium sizes to exercise different shapes
    let default_sizes = vec![(64usize, 64usize, 64usize), (128, 128, 128), (256, 256, 128)];

    // Sizes can be provided via env var `GEMM_SIZES` or as the first CLI arg.
    // Format: 64x64x64,128x128x128
    let sizes = if let Ok(s) = std::env::var("GEMM_SIZES") {
        parse_sizes(&s)?
    } else if let Some(arg) = std::env::args().nth(1) {
        parse_sizes(&arg)?
    } else {
        default_sizes
    };

    // Tile size can be provided via `GEMM_TILE` env var or second CLI arg; default to 32.
    let tile = if let Ok(t) = std::env::var("GEMM_TILE") {
        parse_tile(&t)?
    } else if let Some(arg2) = std::env::args().nth(2) {
        parse_tile(&arg2)?
    } else {
        32usize
    };

    // Threads: GEMM_THREADS env var or fourth CLI arg; parsed once for the program
    let threads: Option<usize> = if let Ok(t) = std::env::var("GEMM_THREADS") {
        t.parse::<usize>().ok()
    } else if let Some(arg3) = std::env::args().nth(4) {
        arg3.parse::<usize>().ok()
    } else {
        None
    };

    // If threads is set, configure Rayon global pool. Only force BLAS internals to 1 when GEMM_THREADS > 1
    if let Some(t) = threads {
        if t > 1 {
            // For multi-threaded tile-level execution, avoid oversubscription by forcing BLAS internals to 1
            std::env::set_var("OPENBLAS_NUM_THREADS", "1");
            std::env::set_var("OMP_NUM_THREADS", "1");
            std::env::set_var("MKL_NUM_THREADS", "1");
            // Set Rayon threads explicitly for multi-threaded case
            std::env::set_var("RAYON_NUM_THREADS", t.to_string());
            // try to set global rayon pool; ignore error if already set
            let _ = rayon::ThreadPoolBuilder::new().num_threads(t).build_global();
            println!("Set GEMM thread count to {} (Rayon) and forced BLAS internals to 1 to avoid oversubscription", t);
        } else {
            // For GEMM_THREADS == 1, leave BLAS and Rayon env vars untouched so they can be varied externally
            println!("Set GEMM thread count to 1 (no changes to BLAS/Rayon env vars)",);
        }
    }

    // Print effective thread configuration
    let gemm_threads_val = threads.map(|t| t.to_string()).unwrap_or_else(|| "unset".to_string());
    let openblas_val = std::env::var("OPENBLAS_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    let rayon_val = std::env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "unset".to_string());
    println!("GEMM_THREADS = {}, OPENBLAS_NUM_THREADS = {}, RAYON_NUM_THREADS = {}", gemm_threads_val, openblas_val, rayon_val);

    let tol = 1e-3f32;

    // Try to load MKL / BLAS library
    let lib = match try_load_mkl() {
        Ok(l) => l,
        Err(e) => {
            eprintln!("Warning: {}. Skipping BLAS validation. The Rust reference results are available.", e);
            println!("If you want to validate against a BLAS library, set LD_LIBRARY_PATH to include libmkl_rt.so or libopenblas.so and re-run.");
            return Ok(());
        }
    };

    let mut any_fail = false;

    for (m, n, k) in sizes {
        println!("--- Running SGEMM test for size {}x{}x{} (tile = {}) ---", m, n, k, tile);

        let mut rng = rand::thread_rng();

        let mut a = vec![0f32; m * k];
        let mut b = vec![0f32; k * n];
        let mut c_init = vec![0f32; m * n];

        for v in &mut a {
            *v = rng.gen_range(-1.0f32..1.0f32);
        }
        for v in &mut b {
            *v = rng.gen_range(-1.0f32..1.0f32);
        }
        for v in &mut c_init {
            *v = rng.gen_range(-1.0f32..1.0f32);
        }

        let alpha = 1.0f32;
        let beta = 1.0f32;

        // Naive
        let mut c_naive = c_init.clone();
        naive_sgemm(m, n, k, alpha, &a, &b, beta, &mut c_naive);

        // Tiled (scalar)
        let mut c_tiled = c_init.clone();
        tiled_sgemm(m, n, k, alpha, &a, &b, beta, &mut c_tiled, tile);

        // Parallel + SIMD tiled
        let mut c_par_simd = c_init.clone();
        // threads controlled via GEMM_THREADS env or fourth CLI arg
        let threads = if let Ok(t) = std::env::var("GEMM_THREADS") {
            t.parse::<usize>().ok()
        } else if let Some(arg3) = std::env::args().nth(4) {
            arg3.parse::<usize>().ok()
        } else {
            None
        };
        // If GEMM_THREADS == 1, treat it as "no explicit per-function thread count" so global RAYON_NUM_THREADS can be varied.
        let threads_for_calls = match threads {
            Some(1) => None,
            other => other,
        };
        tiled_sgemm_parallel_simd(m, n, k, alpha, &a, &b, beta, &mut c_par_simd, tile, threads_for_calls);

        // BLAS
        let mut c_blas = c_init.clone();
        call_cblas_sgemm(&lib, m as i32, n as i32, k as i32, alpha, &a, &b, beta, &mut c_blas)?;

        // Tiled BLAS: outer loops in Rust, inner tiles delegated to BLAS
        let mut c_tiled_blas = c_init.clone();
        tiled_sgemm_blas_tiles(&lib, m, n, k, alpha, &a, &b, beta, &mut c_tiled_blas, tile, threads_for_calls)?;

        // Tiled BLAS (pack/copy): copy each tile to contiguous buffers, call BLAS, copy back
        let mut c_tiled_blas_copy = c_init.clone();
        tiled_sgemm_blas_tiles_copy(&lib, m, n, k, alpha, &a, &b, beta, &mut c_tiled_blas_copy, tile, threads_for_calls)?;

        let diff_naive_blas = max_abs_diff(&c_naive, &c_blas);
        let diff_tiled_blas = max_abs_diff(&c_tiled, &c_blas);
        let diff_par_blas = max_abs_diff(&c_par_simd, &c_blas);
        let diff_tiledblas_blas = max_abs_diff(&c_tiled_blas, &c_blas);
        let diff_tiledblascopy_blas = max_abs_diff(&c_tiled_blas_copy, &c_blas);
        let diff_naive_tiled = max_abs_diff(&c_naive, &c_tiled);
        let diff_naive_par = max_abs_diff(&c_naive, &c_par_simd);

        println!("Naive vs BLAS max abs diff: {} (tol = {})", diff_naive_blas, tol);
        println!("Tiled vs BLAS max abs diff: {} (tol = {})", diff_tiled_blas, tol);
        println!("Tiled-BLAS-Copy vs BLAS max abs diff: {} (tol = {})", diff_tiledblascopy_blas, tol);
        println!("Parallel+SIMD vs BLAS max abs diff: {} (tol = {})", diff_par_blas, tol);
        println!("Naive vs Tiled max abs diff: {}", diff_naive_tiled);
        println!("Naive vs Parallel+SIMD max abs diff: {}", diff_naive_par);

        let ok_naive = diff_naive_blas <= tol;
        let ok_tiled = diff_tiled_blas <= tol;
        let ok_tiled_blas = diff_tiledblas_blas <= tol;
        let ok_tiled_blas_copy = diff_tiledblascopy_blas <= tol;
        let ok_par = diff_par_blas <= tol;

        if ok_naive {
            println!("✅ size {}x{}x{} (naive): OK", m, n, k);
        } else {
            println!("❌ size {}x{}x{} (naive): FAILED (maxdiff > {})", m, n, k, tol);
            any_fail = true;
        }

        if ok_tiled {
            println!("✅ size {}x{}x{} (tiled tile={}): OK", m, n, k, tile);
        } else {
            println!("❌ size {}x{}x{} (tiled tile={}): FAILED (maxdiff > {})", m, n, k, tile, tol);
            any_fail = true;
        }

        if ok_tiled_blas {
            println!("✅ size {}x{}x{} (tiled-BLAS tile={}): OK", m, n, k, tile);
        } else {
            println!("❌ size {}x{}x{} (tiled-BLAS tile={}): FAILED (maxdiff > {})", m, n, k, tile, tol);
            any_fail = true;
        }

        if ok_tiled_blas_copy {
            println!("✅ size {}x{}x{} (tiled-BLAS-Copy tile={}): OK", m, n, k, tile);
        } else {
            println!("❌ size {}x{}x{} (tiled-BLAS-Copy tile={}): FAILED (maxdiff > {})", m, n, k, tile, tol);
            any_fail = true;
        }

        if ok_par {
            println!("✅ size {}x{}x{} (parallel+simd tile={}): OK", m, n, k, tile);
        } else {
            println!("❌ size {}x{}x{} (parallel+simd tile={}): FAILED (maxdiff > {})", m, n, k, tile, tol);
            any_fail = true;
        }

        // small sanity check: naive and tiled should match closely
        if diff_naive_tiled > 1e-6 {
            println!("⚠️ Note: naive and tiled implementations differ by {} (this may be due to floating point ordering).", diff_naive_tiled);
        }
        if diff_naive_par > 1e-6 {
            println!("⚠️ Note: naive and parallel+SIMD implementations differ by {} (this may be due to floating point ordering).", diff_naive_par);
        }

        // Timing: configurable repeats via GEMM_REPEATS or third CLI arg (default 3)
        let repeats: usize = if let Ok(r) = std::env::var("GEMM_REPEATS") {
            r.parse().unwrap_or(3)
        } else if let Some(rarg) = std::env::args().nth(3) {
            rarg.parse().unwrap_or(3)
        } else {
            3usize
        };

        use std::time::Instant;

        let flops = 2.0f64 * (m as f64) * (n as f64) * (k as f64);

        // Time naive
        let mut t_naive = 0f64;
        for _ in 0..repeats {
            let mut c_tmp = c_init.clone();
            let start = Instant::now();
            naive_sgemm(m, n, k, alpha, &a, &b, beta, &mut c_tmp);
            t_naive += start.elapsed().as_secs_f64();
        }
        let avg_naive = t_naive / (repeats as f64);
        let gflops_naive = flops / (avg_naive * 1e9);

        // Time tiled
        let mut t_tiled = 0f64;
        for _ in 0..repeats {
            let mut c_tmp = c_init.clone();
            let start = Instant::now();
            tiled_sgemm(m, n, k, alpha, &a, &b, beta, &mut c_tmp, tile);
            t_tiled += start.elapsed().as_secs_f64();
        }
        let avg_tiled = t_tiled / (repeats as f64);
        let gflops_tiled = flops / (avg_tiled * 1e9);

        // Time parallel+SIMD
        let mut t_par = 0f64;
        for _ in 0..repeats {
            let mut c_tmp = c_init.clone();
            let start = Instant::now();
            tiled_sgemm_parallel_simd(m, n, k, alpha, &a, &b, beta, &mut c_tmp, tile, threads_for_calls);
            t_par += start.elapsed().as_secs_f64();
        }
        let avg_par = t_par / (repeats as f64);
        let gflops_par = flops / (avg_par * 1e9);

        // Time tiled BLAS
        let mut t_tiled_blas = 0f64;
        for _ in 0..repeats {
            let mut c_tmp = c_init.clone();
            let start = Instant::now();
            tiled_sgemm_blas_tiles(&lib, m, n, k, alpha, &a, &b, beta, &mut c_tmp, tile, threads_for_calls)?;
            t_tiled_blas += start.elapsed().as_secs_f64();
        }
        let avg_tiled_blas = t_tiled_blas / (repeats as f64);
        let gflops_tiled_blas = flops / (avg_tiled_blas * 1e9);

        // Time tiled BLAS (pack/copy)
        let mut t_tiled_blas_copy = 0f64;
        for _ in 0..repeats {
            let mut c_tmp = c_init.clone();
            let start = Instant::now();
            tiled_sgemm_blas_tiles_copy(&lib, m, n, k, alpha, &a, &b, beta, &mut c_tmp, tile, threads_for_calls)?;
            t_tiled_blas_copy += start.elapsed().as_secs_f64();
        }
        let avg_tiled_blas_copy = t_tiled_blas_copy / (repeats as f64);
        let gflops_tiled_blas_copy = flops / (avg_tiled_blas_copy * 1e9);

        // Time BLAS
        let mut t_blas = 0f64;
        for _ in 0..repeats {
            let mut c_tmp = c_init.clone();
            let start = Instant::now();
            call_cblas_sgemm(&lib, m as i32, n as i32, k as i32, alpha, &a, &b, beta, &mut c_tmp)?;
            t_blas += start.elapsed().as_secs_f64();
        }
        let avg_blas = t_blas / (repeats as f64);
        let gflops_blas = flops / (avg_blas * 1e9);

        println!("Timings (avg over {} runs):", repeats);
        println!("  Naive  : {:>9.6} s  {:>8.3} GFLOPS", avg_naive, gflops_naive);
        println!("  Tiled  : {:>9.6} s  {:>8.3} GFLOPS", avg_tiled, gflops_tiled);
        println!("  Par+SIMD: {:>8.6} s  {:>8.3} GFLOPS", avg_par, gflops_par);
        println!("  Tiled-BLAS: {:>9.6} s  {:>8.3} GFLOPS", avg_tiled_blas, gflops_tiled_blas);
        println!("  Tiled-BLAS-Copy: {:>9.6} s  {:>8.3} GFLOPS", avg_tiled_blas_copy, gflops_tiled_blas_copy);
        println!("  BLAS   : {:>9.6} s  {:>8.3} GFLOPS", avg_blas, gflops_blas);

        println!("");
    }

    if any_fail {
        Err("One or more SGEMM validations failed".into())
    } else {
        println!("All validations passed ✅");
        Ok(())
    }
}
