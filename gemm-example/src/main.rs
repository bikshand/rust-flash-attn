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

// Removed scalar tiled and parallel+SIMD tiled implementations to keep only naive, BLAS and tiled_sgemm_blas_tiles.

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

        // BLAS
        let mut c_blas = c_init.clone();
        call_cblas_sgemm(&lib, m as i32, n as i32, k as i32, alpha, &a, &b, beta, &mut c_blas)?;

        // Tiled BLAS: outer loops in Rust, inner tiles delegated to BLAS
        let mut c_tiled_blas = c_init.clone();
        let threads_for_calls = match threads {
            Some(1) => None,
            other => other,
        };
        tiled_sgemm_blas_tiles(&lib, m, n, k, alpha, &a, &b, beta, &mut c_tiled_blas, tile, threads_for_calls)?;

        let diff_naive_blas = max_abs_diff(&c_naive, &c_blas);
        let diff_tiledblas_blas = max_abs_diff(&c_tiled_blas, &c_blas);

        println!("Naive vs BLAS max abs diff: {} (tol = {})", diff_naive_blas, tol);
        println!("Tiled-BLAS vs BLAS max abs diff: {} (tol = {})", diff_tiledblas_blas, tol);

        let ok_naive = diff_naive_blas <= tol;
        let ok_tiled_blas = diff_tiledblas_blas <= tol;

        if ok_naive {
            println!("✅ size {}x{}x{} (naive): OK", m, n, k);
        } else {
            println!("❌ size {}x{}x{} (naive): FAILED (maxdiff > {})", m, n, k, tol);
            any_fail = true;
        }

        if ok_tiled_blas {
            println!("✅ size {}x{}x{} (tiled-BLAS tile={}): OK", m, n, k, tile);
        } else {
            println!("❌ size {}x{}x{} (tiled-BLAS tile={}): FAILED (maxdiff > {})", m, n, k, tile, tol);
            any_fail = true;
        }

        // small sanity check: naive and tiled-BLAS should match closely
        let diff_naive_tiledblas = max_abs_diff(&c_naive, &c_tiled_blas);
        if diff_naive_tiledblas > 1e-6 {
            println!("⚠️ Note: naive and tiled-BLAS implementations differ by {} (this may be due to floating point ordering).", diff_naive_tiledblas);
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
        println!("  Tiled-BLAS: {:>9.6} s  {:>8.3} GFLOPS", avg_tiled_blas, gflops_tiled_blas);
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
