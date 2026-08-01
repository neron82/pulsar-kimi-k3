fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return; // kernels are CUDA/Linux; other hosts get an empty crate
    }
    println!("cargo:rerun-if-changed=cuda");
    // One fatbin for every NVIDIA generation the kernels can serve. The
    // floor is dp4a = sm_61 (Pascal / GTX 10-series); nothing newer is
    // required (no tensor cores, no async-copy, static <=48KB shared).
    //
    //   sm_61  SASS  GTX 10-series           + compute_61 PTX: JIT floor
    //   sm_75  SASS  GTX 16 / RTX 20-series    for anything unlisted
    //   sm_86  SASS  RTX 30-series             (sm_70 Volta, sm_80 A100,
    //   sm_89  SASS  RTX 40-series              Hopper, ...)
    //   compute_89 PTX: JIT for sm_90+ (Blackwell RTX 50 etc.) with the
    //   newest ISA the toolkit knows, instead of the sm_61 floor.
    //
    // PULSAR_CUDA_ARCH overrides (e.g. "89" for a fast dev build, or
    // "89,120" once the toolkit codegens Blackwell SASS natively).
    // CUDA 13 dropped sm_61, so the default is filtered against the actual
    // nvcc toolkit while older CUDA toolkits can still retain Pascal.
    // 80 must stay in the list: the int8 mma prefill GEMM gates on
    // cc >= 8 at runtime, so every >= 8.0 device needs a fatbin entry
    // compiled with __CUDA_ARCH__ >= 800 (A100 falling back to the
    // compute_61 floor PTX would silently run the empty stub)
    let archs = std::env::var("PULSAR_CUDA_ARCH")
        .map(|value| value.split(',').map(str::to_owned).collect())
        .unwrap_or_else(|_| default_archs());
    let mut build = cc::Build::new();
    // CPU-Q8 uses ordinary IEEE f32 division, sqrt, and non-contracted
    // scalar arithmetic at its quantization boundaries.  Fast-math changes
    // scales and nonlinear controls enough to alter packed activations and
    // recurrent state, so keep production CUDA arithmetic precise.
    build
        .cuda(true)
        .flag("-O3")
        .flag("--fmad=false")
        .flag("--prec-div=true")
        .flag("--prec-sqrt=true");
    // nvcc rejects host compilers newer than its toolkit supports (e.g.
    // CUDA 12.0 caps at gcc 12 while distro c++ is gcc 13). Probe a tiny
    // compile with candidate ccbins and take the first one nvcc accepts.
    if let Some(ccbin) = pick_ccbin() {
        build.flag(&format!("-ccbin={ccbin}"));
    }
    let list: Vec<&str> = archs
        .iter()
        .map(String::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect();
    for (i, a) in list.iter().enumerate() {
        let first = i == 0;
        let last = i + 1 == list.len();
        // lowest arch also embeds its PTX (universal JIT floor); highest
        // embeds its PTX too (best ISA for future GPUs); middles are SASS-only
        let code = if first || last {
            format!("arch=compute_{a},code=[sm_{a},compute_{a}]")
        } else {
            format!("arch=compute_{a},code=sm_{a}")
        };
        build.flag("-gencode").flag(&code);
    }
    build
        .file("cuda/pulsar_kernels.cu")
        .compile("pulsar_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}

fn default_archs() -> Vec<String> {
    let requested = ["61", "75", "80", "86", "89"];
    let output = std::process::Command::new("nvcc")
        .arg("--list-gpu-arch")
        .output();
    let supported = match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .lines()
            .filter_map(|line| line.trim().strip_prefix("compute_"))
            .map(str::to_owned)
            .collect::<std::collections::HashSet<_>>(),
        _ => return requested.iter().map(|arch| (*arch).into()).collect(),
    };

    let filtered = requested
        .iter()
        .filter(|arch| supported.contains(**arch))
        .map(|arch| (*arch).into())
        .collect::<Vec<String>>();
    if filtered.is_empty() {
        requested.iter().map(|arch| (*arch).into()).collect()
    } else {
        filtered
    }
}

fn pick_ccbin() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(env) = std::env::var("NVCC_CCBIN") {
        return Some(env); // explicit override, no probe
    }
    candidates.push("c++".into());
    for v in ["14", "13", "12", "11", "10"] {
        candidates.push(format!("g++-{v}"));
    }
    let out = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".into());
    let probe = format!("{out}/ccbin_probe.cu");
    std::fs::write(&probe, "int main(){return 0;}\n").ok()?;
    for cand in candidates {
        let ok = std::process::Command::new("nvcc")
            .args([&format!("-ccbin={cand}"), "-c", &probe, "-o"])
            .arg(format!("{out}/ccbin_probe.o"))
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(cand);
        }
    }
    None // let nvcc use its default and report its own error
}
