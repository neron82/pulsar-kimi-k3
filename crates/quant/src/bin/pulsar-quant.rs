//! pulsar-quant: rewrite a BF16/F16/F32 gguf (single or -00001-of-N split)
//! into a recipe-quantized single gguf.
//!
//!   pulsar-quant -i model-BF16-00001-of-00003.gguf -o out.gguf \
//!       --map "_exps.=q2_k" --default q8_0
//!
//! Recipe rules: `--map pat=type` (repeatable, comma-separable) matches by
//! SUBSTRING against the tensor name, first match wins; `--default` covers
//! unmatched 2D+ tensors. 1D tensors (norms, biases) always stay f32.
//! K-quant targets need row width % 256 == 0; offenders fall back to q8_0
//! (width % 32) or f16, with a warning - same guardrails llama.cpp applies.
//!
//! Shards are processed ONE AT A TIME (stage 3), so a split BF16 source
//! bigger than the disk streams through: `--fetch-cmd 'CMD {}'` runs when
//! a shard file is missing ({} = shard path), `--delete-shards` removes
//! each shard once its tensors are written. The output header is patched
//! in last (reserved space + exact-size pulsar.pad metadata key) because
//! the full tensor table isn't known until every shard header was seen;
//! `--header-reserve MB` overrides the auto estimate if it ever errors.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;

use gguf::{Gguf, TensorType, Value};

fn parse_type(s: &str) -> Result<TensorType, String> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "q8_0" => TensorType::Q8_0,
        "q2_k" => TensorType::Q2K,
        "q3_k" => TensorType::Q3K,
        "q4_k" => TensorType::Q4K,
        "q5_k" => TensorType::Q5K,
        "q6_k" => TensorType::Q6K,
        "iq2_xxs" => TensorType::IQ2XXS,
        "f16" => TensorType::F16,
        "f32" => TensorType::F32,
        other => {
            return Err(format!(
                "unknown target type {other} (stage 1: q8_0 q2_k..q6_k f16 f32)"
            ))
        }
    })
}

fn parse_header(path: &std::path::Path) -> Result<Gguf, String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut n = 32 << 20;
    loop {
        let mut head = vec![0u8; n];
        let got = {
            let mut read = 0;
            while read < head.len() {
                match f.read(&mut head[read..]) {
                    Ok(0) => break,
                    Ok(k) => read += k,
                    Err(e) => return Err(e.to_string()),
                }
            }
            read
        };
        match Gguf::parse(&head[..got]) {
            Ok(g) => return Ok(g),
            Err(gguf::Error::Truncated { .. }) if got == n => {
                n *= 2;
                f.seek(SeekFrom::Start(0)).map_err(|e| e.to_string())?;
            }
            Err(e) => return Err(format!("{}: {e:?}", path.display())),
        }
    }
}

/// Make sure a shard file exists, running the fetch command if not.
fn ensure_shard(path: &std::path::Path, fetch_cmd: Option<&str>) -> Result<(), String> {
    if path.exists() {
        return Ok(());
    }
    let cmd =
        fetch_cmd.ok_or_else(|| format!("{} missing and no --fetch-cmd given", path.display()))?;
    let cmd = cmd.replace("{}", &path.display().to_string());
    eprintln!("pulsar-quant: fetching {} ({cmd})", path.display());
    let st = std::process::Command::new("sh")
        .arg("-c")
        .arg(&cmd)
        .status()
        .map_err(|e| format!("fetch: {e}"))?;
    if !st.success() {
        return Err(format!(
            "fetch command failed ({st}) for {}",
            path.display()
        ));
    }
    if !path.exists() {
        return Err(format!(
            "fetch command succeeded but {} still missing",
            path.display()
        ));
    }
    Ok(())
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn value_type_id(v: &Value) -> u32 {
    match v {
        Value::U8(_) => 0,
        Value::I8(_) => 1,
        Value::U16(_) => 2,
        Value::I16(_) => 3,
        Value::U32(_) => 4,
        Value::I32(_) => 5,
        Value::F32(_) => 6,
        Value::Bool(_) => 7,
        Value::String(_) => 8,
        Value::Array(_) => 9,
        Value::U64(_) => 10,
        Value::I64(_) => 11,
        Value::F64(_) => 12,
    }
}

fn write_value_payload(out: &mut Vec<u8>, v: &Value) -> Result<(), String> {
    match v {
        Value::U8(x) => out.push(*x),
        Value::I8(x) => out.push(*x as u8),
        Value::U16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I16(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Bool(x) => out.push(*x as u8),
        Value::String(s) => write_string(out, s),
        Value::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::I64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::F64(x) => out.extend_from_slice(&x.to_le_bytes()),
        Value::Array(items) => {
            let elem_ty = items.first().map(value_type_id).unwrap_or(4);
            if items.iter().any(|it| value_type_id(it) != elem_ty) {
                return Err("heterogeneous metadata array".into());
            }
            out.extend_from_slice(&elem_ty.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for it in items {
                write_value_payload(out, it)?;
            }
        }
    }
    Ok(())
}

struct OutTensor {
    name: String,
    dims: Vec<u64>,
    ty: TensorType,
    out_off: u64, // relative to output data section
}

fn main() {
    if let Err(e) = run() {
        eprintln!("pulsar-quant: {e}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut input = None;
    let mut output = None;
    let mut maps: Vec<(String, TensorType)> = Vec::new();
    let mut default_ty = TensorType::Q8_0;
    let mut imatrix_path: Option<String> = None;
    let mut fetch_cmd: Option<String> = None;
    let mut delete_shards = false;
    let mut header_reserve_mb: Option<u64> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        let mut need = |what: &str| args.next().ok_or(format!("{what} needs a value"));
        match a.as_str() {
            "-i" => input = Some(need("-i")?),
            "-o" => output = Some(need("-o")?),
            "--map" => {
                for part in need("--map")?.split(',') {
                    let (pat, ty) = part
                        .split_once('=')
                        .ok_or(format!("bad --map entry {part}"))?;
                    maps.push((pat.to_string(), parse_type(ty)?));
                }
            }
            "--default" => default_ty = parse_type(&need("--default")?)?,
            "--imatrix" => imatrix_path = Some(need("--imatrix")?),
            "--fetch-cmd" => fetch_cmd = Some(need("--fetch-cmd")?),
            "--delete-shards" => delete_shards = true,
            "--header-reserve" => {
                header_reserve_mb = Some(
                    need("--header-reserve")?
                        .parse::<u64>()
                        .map_err(|e| e.to_string())?,
                )
            }
            other => return Err(format!("unknown arg {other}")),
        }
    }
    let input = std::path::PathBuf::from(input.ok_or("-i required")?);
    let output = std::path::PathBuf::from(output.ok_or("-o required")?);
    let imatrix = match &imatrix_path {
        Some(p) => Some(quant::iq::read_imatrix(std::path::Path::new(p))?),
        None => None,
    };

    // ---- shard NAME list; shards may not exist yet (--fetch-cmd streams them)
    let shard_paths = gguf::split_shard_names(&input).unwrap_or_else(|| vec![input.clone()]);
    let split = shard_paths.len() > 1;

    // shard 1 carries the model metadata and sizes the header reserve
    ensure_shard(&shard_paths[0], fetch_cmd.as_deref())?;
    let g0 = parse_header(&shard_paths[0])?;
    let align = g0.alignment.max(32);
    let n_total = match g0.metadata.get("split.tensors.count") {
        Some(Value::I32(n)) => *n as u64,
        Some(Value::U32(n)) => *n as u64,
        Some(Value::I64(n)) => *n as u64,
        Some(Value::U64(n)) => *n,
        _ => g0.tensors.len() as u64,
    };
    // shard-1 header already holds the full metadata (tokenizer et al);
    // add table room for the other shards' tensor entries
    let reserve = header_reserve_mb
        .map(|m| m << 20)
        .unwrap_or(g0.data_offset + n_total * 160 + (1 << 20));
    let data_start = reserve.next_multiple_of(align);
    eprintln!(
        "pulsar-quant: arch {}, {} shard(s), ~{} tensors, header reserve {:.1} MB",
        g0.architecture().unwrap_or("?"),
        shard_paths.len(),
        n_total,
        data_start as f64 / 1e6
    );

    // ---- decide output types
    let pick = |name: &str, dims: &[u64]| -> TensorType {
        if dims.len() < 2 {
            return TensorType::F32;
        }
        let want = maps
            .iter()
            .find(|(pat, _)| name.contains(pat.as_str()))
            .map(|&(_, ty)| ty)
            .unwrap_or(default_ty);
        let row = dims[0];
        let ok = match want {
            TensorType::Q2K
            | TensorType::Q3K
            | TensorType::Q4K
            | TensorType::Q5K
            | TensorType::Q6K
            | TensorType::IQ2XXS => row % 256 == 0,
            TensorType::Q8_0 => row % 32 == 0,
            _ => true,
        };
        if ok {
            want
        } else if row % 32 == 0 {
            eprintln!("pulsar-quant: {name} width {row} not /256, falling back to q8_0");
            TensorType::Q8_0
        } else {
            eprintln!("pulsar-quant: {name} width {row} not /32, falling back to f16");
            TensorType::F16
        }
    };

    // ---- output: data first at the reserve, header patched in at the end
    let f = File::create(&output).map_err(|e| e.to_string())?;
    let mut w = BufWriter::with_capacity(8 << 20, f);
    w.seek(SeekFrom::Start(data_start))
        .map_err(|e| e.to_string())?;

    let nthread = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8);
    let t0 = std::time::Instant::now();
    let mut out_tensors: Vec<OutTensor> = Vec::with_capacity(n_total as usize);
    let mut out_off = 0u64;
    let mut written = 0u64;
    let mut by_type: HashMap<String, u64> = HashMap::new();
    let mut meta: Vec<(String, Value)> = Vec::new();
    let mut g0 = Some(g0);

    for (si, sp) in shard_paths.iter().enumerate() {
        ensure_shard(sp, fetch_cmd.as_deref())?;
        let g = match g0.take() {
            Some(g) => g,
            None => parse_header(sp)?,
        };
        if si == 0 {
            meta = g
                .metadata
                .iter()
                .filter(|(k, _)| !k.starts_with("split."))
                .map(|(k, v)| (k.clone(), v.clone()))
                .collect();
            meta.sort_by(|a, b| a.0.cmp(&b.0));
        }
        let file = File::open(sp).map_err(|e| format!("{}: {e}", sp.display()))?;
        eprintln!(
            "pulsar-quant: shard {}/{}: {} tensors",
            si + 1,
            shard_paths.len(),
            g.tensors.len()
        );

        for t in &g.tensors {
            match t.ty {
                TensorType::F32 | TensorType::F16 | TensorType::BF16 => {}
                other => {
                    return Err(format!(
                        "{}: source type {other:?} is not a float type",
                        t.name
                    ))
                }
            }
            let mut ty = pick(&t.name, &t.dims);
            if ty == TensorType::IQ2XXS {
                let row = t.dims[0];
                let n_exp = t.dims.get(2).copied().unwrap_or(1);
                let ok = imatrix
                    .as_ref()
                    .and_then(|m| m.get(&t.name))
                    .is_some_and(|e| e.len() as u64 == row || e.len() as u64 == row * n_exp);
                if !ok {
                    eprintln!(
                        "pulsar-quant: {} has no usable imatrix entry, falling back to q2_k",
                        t.name
                    );
                    ty = TensorType::Q2K;
                }
            }
            let row = *t.dims.first().unwrap_or(&1) as usize;
            let rows = t.dims.iter().skip(1).product::<u64>().max(1) as usize;
            let src_row_bytes = t.ty.row_bytes(row as u64).unwrap() as usize;
            let out_row_bytes = ty.row_bytes(row as u64).ok_or("row_bytes")? as usize;

            // read whole source tensor (largest single tensors in BF16 are
            // a few GB; fine on a 30GB box)
            let mut src = vec![0u8; src_row_bytes * rows];
            file.read_exact_at(&mut src, g.data_offset + t.offset)
                .map_err(|e| format!("{}: read {e}", t.name))?;

            let chunk_rows = rows.div_ceil(nthread);
            let mut parts: Vec<Vec<u8>> = Vec::with_capacity(nthread);
            std::thread::scope(|s| -> Result<(), String> {
                let mut handles = Vec::new();
                for c in 0..nthread {
                    let lo = c * chunk_rows;
                    if lo >= rows {
                        break;
                    }
                    let hi = ((c + 1) * chunk_rows).min(rows);
                    let src = &src[lo * src_row_bytes..hi * src_row_bytes];
                    let (src_ty, out_ty) = (t.ty, ty);
                    let entry = imatrix.as_ref().and_then(|m| m.get(&t.name));
                    let ne1 = t.dims.get(1).copied().unwrap_or(1) as usize;
                    handles.push(s.spawn(move || -> Result<Vec<u8>, String> {
                        let mut buf = Vec::with_capacity((hi - lo) * out_row_bytes);
                        let mut f32row = Vec::with_capacity(row);
                        for (k, r) in src.chunks_exact(src_row_bytes).enumerate() {
                            quant::row_to_f32(src_ty, r, &mut f32row)?;
                            let qw = entry.map(|e| {
                                if e.len() == row {
                                    &e[..]
                                } else {
                                    // per-expert imatrix: ne0 * n_expert values
                                    let expert = (lo + k) / ne1.max(1);
                                    &e[expert * row..(expert + 1) * row]
                                }
                            });
                            quant::quantize_row(out_ty, &f32row, qw, &mut buf)?;
                        }
                        Ok(buf)
                    }));
                }
                for h in handles {
                    parts.push(h.join().map_err(|_| "encode thread panicked")??);
                }
                Ok(())
            })?;

            let mut nbytes = 0u64;
            for p in &parts {
                w.write_all(p).map_err(|e| e.to_string())?;
                nbytes += p.len() as u64;
            }
            let end = out_off + nbytes;
            let next = end.next_multiple_of(align);
            w.write_all(&vec![0u8; (next - end) as usize])
                .map_err(|e| e.to_string())?;
            out_tensors.push(OutTensor {
                name: t.name.clone(),
                dims: t.dims.clone(),
                ty,
                out_off,
            });
            out_off = next;
            written += nbytes;
            *by_type.entry(format!("{:?}", ty)).or_default() += nbytes;
            if out_tensors.len() % 50 == 1 {
                eprintln!(
                    "pulsar-quant: [{}/~{}] {} ({:.1}GB written, {:.0}s)",
                    out_tensors.len(),
                    n_total,
                    t.name,
                    written as f64 / 1e9,
                    t0.elapsed().as_secs_f32()
                );
            }
        }
        if delete_shards && split {
            std::fs::remove_file(sp).map_err(|e| e.to_string())?;
            eprintln!("pulsar-quant: deleted {}", sp.display());
        }
    }
    w.flush().map_err(|e| e.to_string())?;
    let f = w.into_inner().map_err(|e| e.to_string())?;

    // ---- header, sized to end EXACTLY at data_start via a pad key
    let mut head = Vec::with_capacity(data_start as usize);
    head.extend_from_slice(&gguf::GGUF_MAGIC.to_le_bytes());
    head.extend_from_slice(&3u32.to_le_bytes());
    head.extend_from_slice(&(out_tensors.len() as u64).to_le_bytes());
    head.extend_from_slice(&((meta.len() + 1) as u64).to_le_bytes()); // +1: pulsar.pad
    for (k, v) in &meta {
        write_string(&mut head, k);
        head.extend_from_slice(&value_type_id(v).to_le_bytes());
        write_value_payload(&mut head, v)?;
    }
    let mut table = Vec::new();
    for t in &out_tensors {
        write_string(&mut table, &t.name);
        table.extend_from_slice(&(t.dims.len() as u32).to_le_bytes());
        for d in &t.dims {
            table.extend_from_slice(&d.to_le_bytes());
        }
        table.extend_from_slice(&t.ty.to_id().to_le_bytes());
        table.extend_from_slice(&t.out_off.to_le_bytes());
    }
    const PAD_KEY: &str = "pulsar.pad";
    let overhead = 8 + PAD_KEY.len() as u64 + 4 + 8; // key + type id + payload len
    let used = head.len() as u64 + overhead + table.len() as u64;
    let pad = data_start.checked_sub(used).ok_or_else(|| {
        format!(
            "header ({used}B) exceeds reserve ({data_start}B); rerun with --header-reserve {}",
            (used >> 20) + 8
        )
    })?;
    write_string(&mut head, PAD_KEY);
    head.extend_from_slice(&8u32.to_le_bytes()); // string
    head.extend_from_slice(&pad.to_le_bytes());
    head.resize(head.len() + pad as usize, b' ');
    head.extend_from_slice(&table);
    assert_eq!(head.len() as u64, data_start);
    f.write_all_at(&head, 0).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    drop(f);

    // self-check: the output must parse and carry every tensor
    let check = parse_header(&output)?;
    if check.tensors.len() != out_tensors.len() {
        return Err(format!(
            "self-check: output has {} tensors, expected {}",
            check.tensors.len(),
            out_tensors.len()
        ));
    }

    let mut summary: Vec<_> = by_type.into_iter().collect();
    summary.sort_by(|a, b| b.1.cmp(&a.1));
    for (ty, b) in summary {
        eprintln!("pulsar-quant: {ty}: {:.2} GB", b as f64 / 1e9);
    }
    eprintln!(
        "pulsar-quant: wrote {} ({:.2} GB) in {:.0}s",
        output.display(),
        (data_start + out_off) as f64 / 1e9,
        t0.elapsed().as_secs_f32()
    );
    Ok(())
}
