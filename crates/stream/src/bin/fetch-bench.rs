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
    let shards = open_shards(model);
    let (g, model_len) = merged_header(&shards);
    let all = stream::expert_reads(&g, model_len).expect("expert reads");
    eprintln!(
        "universe: {} expert slabs across {} exps tensors",
        all.len(),
        all.len()
            / g.arch_meta("expert_count")
                .and_then(gguf::Value::as_u64)
                .unwrap() as usize
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
    eprintln!(
        "plan: {} reads, {:.2} GiB payload -> {}",
        picks.len(),
        total as f64 / (1u64 << 30) as f64,
        out
    );
}

#[cfg(target_os = "linux")]
fn run(model: &str, plan_file: &str, qd: &str) {
    let qd: usize = qd.parse().expect("qd");
    let reads = stream::plan_from_str(&std::fs::read_to_string(plan_file).expect("read plan"))
        .expect("parse plan");
    let shards = open_shards(model);
    let mut fetcher = stream::fetch::Fetcher::open_split(&shards, qd, None).expect("open O_DIRECT");
    let t0 = std::time::Instant::now();
    let slabs = fetcher.fetch(&reads).expect("run");
    let secs = t0.elapsed().as_secs_f64();
    let payload: u64 = slabs.iter().map(|s| s.payload().len() as u64).sum();
    let disk: u64 = reads
        .iter()
        .map(|r| {
            let aligned = r.offset & !4095;
            (r.offset - aligned + r.len).next_multiple_of(4096)
        })
        .sum();
    let checksum = slabs.iter().fold(0u8, |sum, slab| {
        sum ^ slab.payload()[slab.payload().len() / 2]
    });
    println!(
        "rust: {} reads, payload {:.2} GiB, disk {:.2} GiB, {:.3} s, {:.2} GB/s payload, {:.2} GB/s disk, checksum {:02x}",
        reads.len(),
        payload as f64 / (1u64 << 30) as f64,
        disk as f64 / (1u64 << 30) as f64,
        secs,
        payload as f64 / secs / 1e9,
        disk as f64 / secs / 1e9,
        checksum,
    );
}

#[cfg(target_os = "linux")]
fn pipeline_run(model: &str, plan_file: &str, qd: &str, max_slots: &str) {
    use stream::pipeline::{Pipeline, PipelineConfig};
    let qd: usize = qd.parse().expect("qd");
    let max_slots: usize = max_slots.parse().expect("max_slots");
    let reads = stream::plan_from_str(&std::fs::read_to_string(plan_file).expect("read plan"))
        .expect("parse plan");

    let shards = open_shards(model);
    let mut pl = Pipeline::open_split(
        &shards,
        PipelineConfig {
            qd,
            max_slots,
            buf_alloc: None,
        },
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

fn open_shards(model: &str) -> Vec<(u64, std::path::PathBuf)> {
    let path = std::path::Path::new(model);
    let paths = gguf::split_shards(path).unwrap_or_else(|| vec![path.to_path_buf()]);
    let mut base = 0u64;
    let mut shards = Vec::with_capacity(paths.len());
    for path in paths {
        let len = std::fs::metadata(&path).expect("stat shard").len();
        shards.push((base, path));
        base += len;
    }
    shards
}

fn merged_header(shards: &[(u64, std::path::PathBuf)]) -> (gguf::Gguf, u64) {
    let mut headers = Vec::with_capacity(shards.len());
    let mut model_len = 0u64;
    for (_, path) in shards {
        headers.push(gguf::Gguf::parse(&read_head_path(path, 32 << 20)).expect("gguf parse"));
        model_len += std::fs::metadata(path).expect("stat shard").len();
    }
    let bases: Vec<u64> = shards.iter().map(|(base, _)| *base).collect();
    (gguf::Gguf::merge_split(headers, &bases), model_len)
}

fn read_head_path(path: &std::path::Path, n: usize) -> Vec<u8> {
    read_head(path.to_str().expect("utf8 model path"), n)
}
