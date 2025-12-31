use libloading::Library;
use rand::Rng;
use std::error::Error;

const CBLAS_ROW_MAJOR: i32 = 101;
const CBLAS_COL_MAJOR: i32 = 102;
const CBLAS_NO_TRANS: i32 = 111;
const CBLAS_TRANS: i32 = 112;
const CBLAS_CONJ_TRANS: i32 = 113;

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

    let mut do_work = || {
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

    // If threads is set, configure BLAS thread envvars and Rayon global pool
    if let Some(t) = threads {
        std::env::set_var("OPENBLAS_NUM_THREADS", t.to_string());
        std::env::set_var("OMP_NUM_THREADS", t.to_string());
        std::env::set_var("MKL_NUM_THREADS", t.to_string());
        std::env::set_var("RAYON_NUM_THREADS", t.to_string());
        // try to set global rayon pool; ignore error if already set
        let _ = rayon::ThreadPoolBuilder::new().num_threads(t).build_global();
        println!("Set thread count to {} (OPENBLAS_NUM_THREADS, MKL_NUM_THREADS, OMP_NUM_THREADS, RAYON_NUM_THREADS)", t);
    }

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
        tiled_sgemm_parallel_simd(m, n, k, alpha, &a, &b, beta, &mut c_par_simd, tile, threads);

        // BLAS
        let mut c_blas = c_init.clone();
        call_cblas_sgemm(&lib, m as i32, n as i32, k as i32, alpha, &a, &b, beta, &mut c_blas)?;

        let diff_naive_blas = max_abs_diff(&c_naive, &c_blas);
        let diff_tiled_blas = max_abs_diff(&c_tiled, &c_blas);
        let diff_par_blas = max_abs_diff(&c_par_simd, &c_blas);
        let diff_naive_tiled = max_abs_diff(&c_naive, &c_tiled);
        let diff_naive_par = max_abs_diff(&c_naive, &c_par_simd);

        println!("Naive vs BLAS max abs diff: {} (tol = {})", diff_naive_blas, tol);
        println!("Tiled vs BLAS max abs diff: {} (tol = {})", diff_tiled_blas, tol);
        println!("Parallel+SIMD vs BLAS max abs diff: {} (tol = {})", diff_par_blas, tol);
        println!("Naive vs Tiled max abs diff: {}", diff_naive_tiled);
        println!("Naive vs Parallel+SIMD max abs diff: {}", diff_naive_par);

        let ok_naive = diff_naive_blas <= tol;
        let ok_tiled = diff_tiled_blas <= tol;
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
            tiled_sgemm_parallel_simd(m, n, k, alpha, &a, &b, beta, &mut c_tmp, tile, threads);
            t_par += start.elapsed().as_secs_f64();
        }
        let avg_par = t_par / (repeats as f64);
        let gflops_par = flops / (avg_par * 1e9);

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
