//! Expert-fetch benchmark. Modes:
//!   fetch-bench plan     <model.gguf> <count> <seed> <plan-out>
//!       Parse the gguf, sample `count` random expert slabs, write the plan.
//!   fetch-bench run      <model.gguf> <plan-file> <qd>
//!       Execute the plan with io_uring + O_DIRECT, print stats.
//!   fetch-bench pipeline <model.gguf> <plan-file> <qd> <max-slots>
//!       Execute the plan through the bounded pipeline, print per-stage stats.
//! The C reference (bench/expert_fetch_bench.c) consumes the same plan file
//! so both implementations perform byte-identical I/O.

use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(String::as_str) {
        Some("plan") if args.len() == 6 => plan(&args[2], &args[3], &args[4], &args[5]),
        Some("run") if args.len() == 5 => run(&args[2], &args[3], &args[4]),
        Some("pipeline") if args.len() == 6 => pipeline_run(&args[2], &args[3], &args[4], &args[5]),
        _ => {
            eprintln!("usage: fetch-bench plan     <model.gguf> <count> <seed> <plan-out>");
            eprintln!("       fetch-bench run      <model.gguf> <plan-file> <qd>");
            eprintln!("       fetch-bench pipeline <model.gguf> <plan-file> <qd> <max-slots>");
            exit(2);
        }
    }
}

fn plan(model: &str, count: &str, seed: &str, out: &str) {
    let count: usize = count.parse().expect("count");
    let mut state: u64 = seed.parse().expect("seed");
    let head = read_head(model, 32 << 20);
    let g = gguf::Gguf::parse(&head).expect("gguf parse");
    let model_len = std::fs::metadata(model).expect("stat").len();
    let all = stream::expert_reads(&g, model_len).expect("expert reads");
    eprintln!(
        "universe: {} expert slabs across {} exps tensors",
        all.len(),
        all.len() / g.arch_meta("expert_count").and_then(gguf::Value::as_u64).unwrap() as usize
    );
    // xorshift64: deterministic, identical sampling for any future re-run
    let mut picks = Vec::with_capacity(count);
    for _ in 0..count {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        picks.push(all[(state % all.len() as u64) as usize]);
    }
    std::fs::write(out, stream::plan_to_string(&picks)).expect("write plan");
    let total: u64 = picks.iter().map(|r| r.len).sum();
    eprintln!("plan: {} reads, {:.2} GiB payload -> {}", picks.len(), total as f64 / (1u64 << 30) as f64, out);
}

#[cfg(target_os = "linux")]
fn run(model: &str, plan_file: &str, qd: &str) {
    use std::os::unix::fs::OpenOptionsExt;
    let qd: usize = qd.parse().expect("qd");
    let reads = stream::plan_from_str(
        &std::fs::read_to_string(plan_file).expect("read plan"),
    )
    .expect("parse plan");
    let file = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(model)
        .expect("open O_DIRECT");
    let stats = stream::uring::run_plan(&file, &reads, qd, 4096).expect("run");
    println!(
        "rust: {} reads, payload {:.2} GiB, disk {:.2} GiB, {:.3} s, {:.2} GB/s payload, {:.2} GB/s disk, checksum {:02x}",
        stats.reads,
        stats.bytes_payload as f64 / (1u64 << 30) as f64,
        stats.bytes_disk as f64 / (1u64 << 30) as f64,
        stats.secs,
        stats.bytes_payload as f64 / stats.secs / 1e9,
        stats.bytes_disk as f64 / stats.secs / 1e9,
        stats.checksum,
    );
}

#[cfg(target_os = "linux")]
fn pipeline_run(model: &str, plan_file: &str, qd: &str, max_slots: &str) {
    use stream::pipeline::{Pipeline, PipelineConfig};
    let qd: usize = qd.parse().expect("qd");
    let max_slots: usize = max_slots.parse().expect("max_slots");
    let reads = stream::plan_from_str(
        &std::fs::read_to_string(plan_file).expect("read plan"),
    )
    .expect("parse plan");

    let mut pl = Pipeline::open(
        &std::path::Path::new(model),
        PipelineConfig { qd, max_slots, buf_alloc: None },
    )
    .expect("open pipeline");

    let t0 = std::time::Instant::now();
    let batch = pl.submit(&reads).expect("submit");
    let results: Vec<_> = batch.collect::<std::io::Result<Vec<_>>>().expect("collect");
    let secs = t0.elapsed().as_secs_f64();

    let s = pl.stats();
    let total_payload: u64 = results.iter().map(|r| r.payload().len() as u64).sum();
    println!(
        "pipeline: {} reads, payload {:.2} GiB, disk {:.2} GiB, {:.3} s, {:.2} GB/s payload, {:.2} GB/s disk",
        s.reads,
        total_payload as f64 / (1u64 << 30) as f64,
        s.bytes_disk as f64 / (1u64 << 30) as f64,
        secs,
        total_payload as f64 / secs / 1e9,
        s.bytes_disk as f64 / secs / 1e9,
    );
    println!(
        "  per-stage: submit {:.1} ms, complete {:.1} ms, yield {:.1} ms",
        s.t_submit_ns as f64 / 1e6,
        s.t_complete_ns as f64 / 1e6,
        s.t_yield_ns as f64 / 1e6,
    );
}

#[cfg(not(target_os = "linux"))]
fn run(_: &str, _: &str, _: &str) {
    eprintln!("fetch-bench run is Linux-only (io_uring)");
    exit(1);
}

#[cfg(not(target_os = "linux"))]
fn pipeline_run(_: &str, _: &str, _: &str, _: &str) {
    eprintln!("fetch-bench pipeline is Linux-only (io_uring)");
    exit(1);
}

fn read_head(path: &str, n: usize) -> Vec<u8> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).expect("open model");
    let mut buf = vec![0u8; n];
    let mut got = 0;
    while got < n {
        match f.read(&mut buf[got..]) {
            Ok(0) => break,
            Ok(k) => got += k,
            Err(e) => panic!("read: {e}"),
        }
    }
    buf.truncate(got);
    buf
}
