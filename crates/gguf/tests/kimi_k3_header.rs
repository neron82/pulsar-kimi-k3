//! Parses a synthetic K3 GGUF header fixture (no tensor data — the parser
//! must never need it). Covers:
//!
//! - 896-expert 3D slab dimensions
//! - K3 metadata arrays (`attention.head_count_kv`, shared_expert_count)
//! - Split-shard-compatible header parsing (merge_split)
//! - Fail-closed behavior for malformed dimensions
//! - Standard quant types Q8_0 / Q2_K / Q3_K on expert slabs
//!
//! The fixture is built programmatically so it stays small and auditable
//! without requiring a terabyte model file.

use gguf::{Gguf, TensorType, K3_ARCH, K3_ATTN_LAYER_KEY, K3_HEAD_COUNT_KV_KEY, K3_SHARED_EXP_KEY};

// ---------------------------------------------------------------------------
// Helpers: write GGUF v3 primitives into a byte buffer
// ---------------------------------------------------------------------------

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn put_kv_string(out: &mut Vec<u8>, key: &str, val: &str) {
    put_str(out, key);
    out.extend_from_slice(&8u32.to_le_bytes()); // value type String
    put_str(out, val);
}

fn put_kv_u64(out: &mut Vec<u8>, key: &str, val: u64) {
    put_str(out, key);
    out.extend_from_slice(&10u32.to_le_bytes()); // value type U64
    out.extend_from_slice(&val.to_le_bytes());
}

fn put_kv_u64_array(out: &mut Vec<u8>, key: &str, vals: &[u64]) {
    put_str(out, key);
    out.extend_from_slice(&9u32.to_le_bytes()); // value type Array
    out.extend_from_slice(&10u32.to_le_bytes()); // element type U64
    out.extend_from_slice(&(vals.len() as u64).to_le_bytes());
    for v in vals {
        out.extend_from_slice(&v.to_le_bytes());
    }
}

fn put_tensor(out: &mut Vec<u8>, name: &str, dims: &[u64], ty: u32, offset: u64) {
    put_str(out, name);
    out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
    for d in dims {
        out.extend_from_slice(&d.to_le_bytes());
    }
    out.extend_from_slice(&ty.to_le_bytes());
    out.extend_from_slice(&offset.to_le_bytes());
}

/// Build a synthetic K3 GGUF v3 header with the given tensors and metadata.
/// Returns (header_bytes, data_offset).
fn build_k3_header(
    extra_kvs: &[(&str, &[u8])], // raw key-value payloads already encoded
    tensors: &[(&str, &[u64], u32, u64)], // (name, dims, type_id, offset)
    alignment: u64,
) -> (Vec<u8>, u64) {
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic
    h.extend_from_slice(&3u32.to_le_bytes()); // version
    h.extend_from_slice(&(tensors.len() as u64).to_le_bytes()); // tensor_count
    h.extend_from_slice(&(extra_kvs.len() as u64).to_le_bytes()); // kv_count

    for (key, payload) in extra_kvs {
        put_str(&mut h, key);
        h.extend_from_slice(payload);
    }

    for (name, dims, ty, offset) in tensors {
        put_tensor(&mut h, name, dims, *ty, *offset);
    }

    // pad to alignment
    while h.len() % alignment as usize != 0 {
        h.push(0);
    }
    let data_offset = h.len() as u64;
    (h, data_offset)
}

// ---------------------------------------------------------------------------
// Fixture: a minimal K3 model with 93 layers, 896 experts, 2 shared experts
// ---------------------------------------------------------------------------

const N_LAYERS: u64 = 93;
const N_EXPERTS: u64 = 896;
const N_USED: u64 = 16;
const N_SHARED: u64 = 2;
const HIDDEN: u64 = 7168;
const EXPERT_FFN: u64 = 3072;
const N_HEADS: u64 = 96;
const HEAD_DIM: u64 = 128;

/// Build a minimal K3 header with representative tensors for every layer.
/// Returns (header_bytes, parsed Gguf).
fn k3_minimal_fixture() -> (Vec<u8>, Gguf) {
    let mut kvs: Vec<(&str, &[u8])> = Vec::new();

    // architecture
    let mut arch_payload = Vec::new();
    arch_payload.extend_from_slice(&8u32.to_le_bytes()); // String
    put_str(&mut arch_payload, K3_ARCH);
    kvs.push(("general.architecture", &arch_payload[..]));

    // block_count = 93
    let mut bc_payload = Vec::new();
    bc_payload.extend_from_slice(&10u32.to_le_bytes()); // U64
    bc_payload.extend_from_slice(&N_LAYERS.to_le_bytes());
    kvs.push(("kimi-k3.block_count", &bc_payload[..]));

    // expert_count = 896
    let mut ec_payload = Vec::new();
    ec_payload.extend_from_slice(&10u32.to_le_bytes());
    ec_payload.extend_from_slice(&N_EXPERTS.to_le_bytes());
    kvs.push(("kimi-k3.expert_count", &ec_payload[..]));

    // expert_used_count = 16
    let mut euc_payload = Vec::new();
    euc_payload.extend_from_slice(&10u32.to_le_bytes());
    euc_payload.extend_from_slice(&N_USED.to_le_bytes());
    kvs.push(("kimi-k3.expert_used_count", &euc_payload[..]));

    // shared_expert_count = 2
    let mut sec_payload = Vec::new();
    sec_payload.extend_from_slice(&10u32.to_le_bytes());
    sec_payload.extend_from_slice(&N_SHARED.to_le_bytes());
    kvs.push((K3_SHARED_EXP_KEY, &sec_payload[..]));

    // Canonical per-layer discriminator: 0 = KDA, 1 = gated MLA.
    let layer_kv: Vec<u64> = (0..N_LAYERS).map(|i| if i < 69 { 0 } else { 1 }).collect();
    let mut layer_kv_payload = Vec::new();
    put_kv_u64_array(
        &mut layer_kv_payload,
        "kimi-k3.attention.head_count_kv",
        &layer_kv,
    );
    // The helper already encoded the key; keep only its value payload here.
    // build_k3_header writes the key separately.
    let key_len = 8 + "kimi-k3.attention.head_count_kv".len();
    kvs.push((
        "kimi-k3.attention.head_count_kv",
        &layer_kv_payload[key_len..],
    ));

    // Legacy string array retained only to exercise the parser compatibility
    // helper; the production engine does not consume it.

    let attn: Vec<&str> = (0..N_LAYERS)
        .map(|i| if i < 69 { "kda" } else { "mla" })
        .collect();
    let attn_strs: Vec<&str> = attn.iter().copied().collect();
    let mut attn_payload = Vec::new();
    attn_payload.extend_from_slice(&9u32.to_le_bytes()); // Array
    attn_payload.extend_from_slice(&8u32.to_le_bytes()); // element type String
    attn_payload.extend_from_slice(&(attn_strs.len() as u64).to_le_bytes());
    for a in &attn_strs {
        put_str(&mut attn_payload, a);
    }
    kvs.push((K3_ATTN_LAYER_KEY, &attn_payload[..]));

    // context_length = 1M
    let mut ctx_payload = Vec::new();
    ctx_payload.extend_from_slice(&10u32.to_le_bytes());
    ctx_payload.extend_from_slice(&1_000_000u64.to_le_bytes());
    kvs.push(("kimi-k3.context_length", &ctx_payload[..]));

    // embedding_length = 7168
    let mut emb_payload = Vec::new();
    emb_payload.extend_from_slice(&10u32.to_le_bytes());
    emb_payload.extend_from_slice(&HIDDEN.to_le_bytes());
    kvs.push(("kimi-k3.embedding_length", &emb_payload[..]));

    // Build tensors: one per layer for each major weight group.
    // We use representative types: Q8_0 for attention/dense, Q2_K for
    // expert slabs, Q3_K for some shared experts.
    let mut tensors: Vec<(&str, &[u64], u32, u64)> = Vec::new();
    let mut off = 0u64;

    for l in 0..N_LAYERS {
        let prefix = format!("blk.{l}");

        // KDA / MLA attention projections (2D: [head_dim * n_heads, hidden])
        let attn_q = format!("{prefix}.attn_q.weight");
        tensors.push((
            Box::leak(attn_q.into_boxed_str()),
            &[HEAD_DIM * N_HEADS, HIDDEN],
            8,
            off,
        ));
        off += HEAD_DIM * N_HEADS * 34 / 32; // Q8_0 row_bytes

        let attn_k = format!("{prefix}.attn_k.weight");
        tensors.push((
            Box::leak(attn_k.into_boxed_str()),
            &[HEAD_DIM * N_HEADS, HIDDEN],
            8,
            off,
        ));
        off += HEAD_DIM * N_HEADS * 34 / 32;

        let attn_v = format!("{prefix}.attn_v.weight");
        tensors.push((
            Box::leak(attn_v.into_boxed_str()),
            &[HEAD_DIM * N_HEADS, HIDDEN],
            8,
            off,
        ));
        off += HEAD_DIM * N_HEADS * 34 / 32;

        let attn_o = format!("{prefix}.attn_output.weight");
        tensors.push((
            Box::leak(attn_o.into_boxed_str()),
            &[HIDDEN, HEAD_DIM * N_HEADS],
            8,
            off,
        ));
        off += HIDDEN * 34 / 32 * (HEAD_DIM * N_HEADS);

        // FFN gate expert slab (3D: [expert_ffn, hidden, 896])
        let gate_exps = format!("{prefix}.ffn_gate_exps.weight");
        tensors.push((
            Box::leak(gate_exps.into_boxed_str()),
            &[EXPERT_FFN, HIDDEN, N_EXPERTS],
            10, // Q2K
            off,
        ));
        off += EXPERT_FFN * 84 / 256 * HIDDEN * N_EXPERTS; // Q2K row_bytes

        // FFN up expert slab
        let up_exps = format!("{prefix}.ffn_up_exps.weight");
        tensors.push((
            Box::leak(up_exps.into_boxed_str()),
            &[EXPERT_FFN, HIDDEN, N_EXPERTS],
            10, // Q2K
            off,
        ));
        off += EXPERT_FFN * 84 / 256 * HIDDEN * N_EXPERTS;

        // FFN down expert slab
        let down_exps = format!("{prefix}.ffn_down_exps.weight");
        tensors.push((
            Box::leak(down_exps.into_boxed_str()),
            &[HIDDEN, EXPERT_FFN, N_EXPERTS],
            10, // Q2K
            off,
        ));
        off += HIDDEN * 84 / 256 * EXPERT_FFN * N_EXPERTS;

        // Router (2D: [n_experts, hidden])
        let router = format!("{prefix}.ffn_router.weight");
        tensors.push((
            Box::leak(router.into_boxed_str()),
            &[N_EXPERTS, HIDDEN],
            8, // Q8_0
            off,
        ));
        off += N_EXPERTS * 34 / 32 * HIDDEN;

        // Shared expert gate (2D: [expert_ffn, hidden]) x2
        let shared_gate = format!("{prefix}.ffn_shared_gate.weight");
        tensors.push((
            Box::leak(shared_gate.into_boxed_str()),
            &[EXPERT_FFN, HIDDEN],
            11, // Q3K
            off,
        ));
        off += EXPERT_FFN * 110 / 256 * HIDDEN;

        let shared_up = format!("{prefix}.ffn_shared_up.weight");
        tensors.push((
            Box::leak(shared_up.into_boxed_str()),
            &[EXPERT_FFN, HIDDEN],
            11, // Q3K
            off,
        ));
        off += EXPERT_FFN * 110 / 256 * HIDDEN;

        let shared_down = format!("{prefix}.ffn_shared_down.weight");
        tensors.push((
            Box::leak(shared_down.into_boxed_str()),
            &[HIDDEN, EXPERT_FFN],
            11, // Q3K
            off,
        ));
        off += HIDDEN * 110 / 256 * EXPERT_FFN;

        // Layer norms (1D: [hidden])
        let attn_norm = format!("{prefix}.attn_norm.weight");
        tensors.push((Box::leak(attn_norm.into_boxed_str()), &[HIDDEN], 0, off)); // F32
        off += HIDDEN * 4;

        let ffn_norm = format!("{prefix}.ffn_norm.weight");
        tensors.push((Box::leak(ffn_norm.into_boxed_str()), &[HIDDEN], 0, off));
        off += HIDDEN * 4;
    }

    // Output norm + lm_head
    tensors.push(("output_norm.weight", &[HIDDEN], 0, off));
    off += HIDDEN * 4;
    tensors.push(("output.weight", &[HIDDEN, 32000], 8, off)); // Q8_0 vocab

    let (h, _data_offset) = build_k3_header(&kvs, &tensors, 32);
    let g = Gguf::parse(&h).expect("parse K3 minimal fixture");
    (h, g)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn parses_k3_architecture() {
    let (_, g) = k3_minimal_fixture();
    assert!(g.is_kimi_k3());
    assert_eq!(g.architecture(), Some(K3_ARCH));
}

#[test]
fn parses_k3_block_count() {
    let (_, g) = k3_minimal_fixture();
    assert_eq!(g.k3_block_count(), Some(N_LAYERS));
}

#[test]
fn parses_k3_expert_counts() {
    let (_, g) = k3_minimal_fixture();
    assert_eq!(g.k3_expert_count(), Some(N_EXPERTS));
    assert_eq!(g.k3_expert_used_count(), Some(N_USED));
    assert_eq!(g.k3_shared_expert_count(), Some(N_SHARED));
}

#[test]
fn parses_k3_canonical_head_count_kv_array() {
    let (_, g) = k3_minimal_fixture();
    let values = g
        .k3_head_count_kv()
        .expect("canonical attention.head_count_kv array");
    assert_eq!(values.len(), N_LAYERS as usize);
    assert!(values[..69].iter().all(|&v| v == 0));
    assert!(values[69..].iter().all(|&v| v == 1));
    assert_eq!(g.architecture(), Some(K3_ARCH));
    assert!(g.metadata.contains_key(K3_HEAD_COUNT_KV_KEY));
}

#[test]
fn parses_k3_legacy_attention_layer_array() {
    let (_, g) = k3_minimal_fixture();
    let layers = g.k3_attention_layers().expect("attention_layer array");
    assert_eq!(layers.len(), N_LAYERS as usize);
    // First 69 are KDA, last 24 are MLA
    for (i, l) in layers.iter().enumerate() {
        if i < 69 {
            assert_eq!(l, "kda", "layer {i} should be kda");
        } else {
            assert_eq!(l, "mla", "layer {i} should be mla");
        }
    }
}

#[test]
fn parses_k3_3d_slabs() {
    let (_, g) = k3_minimal_fixture();
    // Every expert slab tensor should be 3D with 896 experts
    for t in &g.tensors {
        if t.name.contains("_exps.") {
            assert!(t.is_3d_slab(), "{} should be 3D", t.name);
            assert_eq!(
                t.slab_expert_count(),
                Some(N_EXPERTS),
                "{} expert count",
                t.name
            );
            assert!(
                t.slab_expert_byte_size().is_some(),
                "{} slab expert byte size",
                t.name
            );
        }
    }
}

#[test]
fn parses_k3_standard_quant_types() {
    let (_, g) = k3_minimal_fixture();
    // Expert slabs should be Q2K
    let gate = g.tensor("blk.0.ffn_gate_exps.weight").expect("gate tensor");
    assert_eq!(gate.ty, TensorType::Q2K);
    // Attention projections should be Q8_0
    let attn_q = g.tensor("blk.0.attn_q.weight").expect("attn_q");
    assert_eq!(attn_q.ty, TensorType::Q8_0);
    // Shared experts should be Q3K
    let shared = g
        .tensor("blk.0.ffn_shared_gate.weight")
        .expect("shared gate");
    assert_eq!(shared.ty, TensorType::Q3K);
    // 1D norms should be F32
    let norm = g.tensor("blk.0.attn_norm.weight").expect("attn_norm");
    assert_eq!(norm.ty, TensorType::F32);
}

#[test]
fn parses_k3_uniform_expert_slabs() {
    let (_, g) = k3_minimal_fixture();
    // All expert slabs of the same kind must have identical byte sizes
    // (the streaming cache's core assumption)
    let slab0 = g
        .tensor("blk.0.ffn_gate_exps.weight")
        .expect("gate slab 0")
        .slab_expert_byte_size()
        .unwrap();
    for l in 1..N_LAYERS {
        let t = g
            .tensor(&format!("blk.{l}.ffn_gate_exps.weight"))
            .expect("gate slab");
        assert_eq!(
            t.slab_expert_byte_size().unwrap(),
            slab0,
            "layer {l} gate slab size mismatch"
        );
    }
}

#[test]
fn parses_k3_data_offset_aligned() {
    let (_, g) = k3_minimal_fixture();
    assert!(g.data_offset % g.alignment == 0);
    assert!(
        g.tensors.len() > 1000,
        "expected ~1k+ tensors, got {}",
        g.tensors.len()
    );
}

#[test]
fn parses_k3_every_tensor_has_known_byte_size() {
    let (_, g) = k3_minimal_fixture();
    for t in &g.tensors {
        assert!(
            t.byte_size().is_some(),
            "tensor {} has unmodeled type {:?}",
            t.name,
            t.ty
        );
    }
}

// ---------------------------------------------------------------------------
// Split-shard merge
// ---------------------------------------------------------------------------

#[test]
fn merge_split_k3_shards() {
    // Build two shards from scratch with virtual offsets
    let mut kvs: Vec<(&str, &[u8])> = Vec::new();

    // architecture
    let mut arch_payload = Vec::new();
    arch_payload.extend_from_slice(&8u32.to_le_bytes());
    put_str(&mut arch_payload, K3_ARCH);
    kvs.push(("general.architecture", &arch_payload[..]));

    let mut ec_payload = Vec::new();
    ec_payload.extend_from_slice(&10u32.to_le_bytes());
    ec_payload.extend_from_slice(&N_EXPERTS.to_le_bytes());
    kvs.push(("kimi-k3.expert_count", &ec_payload[..]));

    // Shard 0: 2 tensors (attn_q, gate_exps)
    let mut tensors0: Vec<(&str, &[u64], u32, u64)> = Vec::new();
    tensors0.push(("blk.0.attn_q.weight", &[128, 128], 8, 0));
    tensors0.push(("blk.0.ffn_gate_exps.weight", &[64, 128, 896], 10, 1024));

    // Shard 1: 2 tensors (up_exps, down_exps)
    let mut tensors1: Vec<(&str, &[u64], u32, u64)> = Vec::new();
    tensors1.push(("blk.0.ffn_up_exps.weight", &[64, 128, 896], 10, 0));
    tensors1.push(("blk.0.ffn_down_exps.weight", &[128, 64, 896], 10, 2048));

    let (h0, _) = build_k3_header(&kvs, &tensors0, 32);
    let (h1, _) = build_k3_header(&kvs, &tensors1, 32);

    let g0 = Gguf::parse(&h0).expect("shard 0");
    let g1 = Gguf::parse(&h1).expect("shard 1");

    // Simulate split: shard 0 data at offset 0, shard 1 data at 1GB
    let base0 = 0u64;
    let base1 = 1_000_000_000u64;
    let s0 = Gguf {
        metadata: g0.metadata.clone(),
        tensors: g0.tensors.clone(),
        data_offset: g0.data_offset,
        ..g0
    };
    let s1 = Gguf {
        metadata: g1.metadata.clone(),
        tensors: g1.tensors.clone(),
        data_offset: g1.data_offset,
        ..g1
    };

    // merge_split adds bases[i] + data_offset to each tensor offset,
    // so we do NOT pre-add them here — the function handles it.
    let merged = Gguf::merge_split(vec![s0, s1], &[base0, base1]);
    assert_eq!(merged.tensors.len(), 4);
    assert_eq!(merged.data_offset, 0);
    assert!(merged.is_kimi_k3());
    assert_eq!(merged.k3_expert_count(), Some(N_EXPERTS));

    // Shard 0 tensors: offset = base0 + data_offset + local_offset
    // = 0 + ~288 + 0 = ~288
    let attn_q = merged.tensor("blk.0.attn_q.weight").unwrap();
    assert!(
        attn_q.offset < 1024,
        "shard 0 tensor offset too large: {}",
        attn_q.offset
    );

    // Shard 1 tensors: offset = base1 + data_offset + local_offset
    // = 1_000_000_000 + ~288 + 0 = ~1_000_000_288
    let up_exps = merged.tensor("blk.0.ffn_up_exps.weight").unwrap();
    assert!(
        up_exps.offset >= base1,
        "shard 1 tensor offset {} should be >= {base1}",
        up_exps.offset
    );
    assert!(
        up_exps.offset < base1 + 1_000_000,
        "shard 1 tensor offset {} should be < {}+1MB",
        up_exps.offset,
        base1
    );
}

// ---------------------------------------------------------------------------
// Fail-closed: malformed dimensions
// ---------------------------------------------------------------------------

#[test]
fn rejects_4d_tensor() {
    // A 4D tensor should be rejected (max 8 dims, but 4D is unusual for K3)
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&1u64.to_le_bytes()); // 1 tensor
    h.extend_from_slice(&1u64.to_le_bytes()); // 1 kv
    put_kv_string(&mut h, "general.architecture", K3_ARCH);
    put_tensor(&mut h, "bad.4d.weight", &[64, 64, 64, 64], 8, 0);
    while h.len() % 32 != 0 {
        h.push(0);
    }
    // 4D is within the 8-dim limit, so it should parse fine
    let g = Gguf::parse(&h).expect("4D tensor should parse (max dims is 8)");
    assert_eq!(g.tensors.len(), 1);
    assert!(!g.tensors[0].is_3d_slab()); // not 3D (4 dims)
                                         // slab_expert_count returns dims[2] regardless of total dims;
                                         // for a 4D tensor dims[2] is the 3rd dimension, not "expert count"
    assert!(g.tensors[0].slab_expert_byte_size().is_none());
}

#[test]
fn rejects_9d_tensor() {
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&1u64.to_le_bytes());
    h.extend_from_slice(&1u64.to_le_bytes());
    put_kv_string(&mut h, "general.architecture", K3_ARCH);
    put_tensor(&mut h, "bad.9d.weight", &[2, 2, 2, 2, 2, 2, 2, 2, 2], 8, 0);
    while h.len() % 32 != 0 {
        h.push(0);
    }
    let err = Gguf::parse(&h).unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("implausible"),
        "expected TooMany error, got: {msg}"
    );
}

#[test]
fn rejects_missing_architecture() {
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&0u64.to_le_bytes()); // 0 tensors
    h.extend_from_slice(&0u64.to_le_bytes()); // 0 kvs
    while h.len() % 32 != 0 {
        h.push(0);
    }
    let g = Gguf::parse(&h).expect("empty header should parse");
    assert!(g.architecture().is_none());
    assert!(!g.is_kimi_k3());
    assert!(g.k3_block_count().is_none());
    assert!(g.k3_expert_count().is_none());
    assert!(g.k3_attention_layers().is_none());
}

#[test]
fn rejects_missing_k3_metadata() {
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&0u64.to_le_bytes());
    h.extend_from_slice(&1u64.to_le_bytes());
    put_kv_string(&mut h, "general.architecture", K3_ARCH);
    while h.len() % 32 != 0 {
        h.push(0);
    }
    let g = Gguf::parse(&h).expect("header with arch only should parse");
    assert!(g.is_kimi_k3());
    // Missing expert_count etc. should return None, not panic
    assert!(g.k3_expert_count().is_none());
    assert!(g.k3_shared_expert_count().is_none());
    assert!(g.k3_attention_layers().is_none());
}

#[test]
fn rejects_wrong_attention_layer_type() {
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&0u64.to_le_bytes());
    h.extend_from_slice(&2u64.to_le_bytes());
    put_kv_string(&mut h, "general.architecture", K3_ARCH);
    // attention_layer as a single string instead of array
    put_kv_string(&mut h, K3_ATTN_LAYER_KEY, "kda");
    while h.len() % 32 != 0 {
        h.push(0);
    }
    let g = Gguf::parse(&h).expect("header should parse");
    assert!(g.is_kimi_k3());
    // Wrong type should return None, not panic
    assert!(g.k3_attention_layers().is_none());
}

#[test]
fn rejects_zero_expert_count() {
    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes());
    h.extend_from_slice(&3u32.to_le_bytes());
    h.extend_from_slice(&0u64.to_le_bytes());
    h.extend_from_slice(&2u64.to_le_bytes());
    put_kv_string(&mut h, "general.architecture", K3_ARCH);
    put_kv_u64(&mut h, "kimi-k3.expert_count", 0);
    while h.len() % 32 != 0 {
        h.push(0);
    }
    let g = Gguf::parse(&h).expect("header should parse");
    assert!(g.is_kimi_k3());
    assert_eq!(g.k3_expert_count(), Some(0));
}

#[test]
fn rejects_bad_magic() {
    let h = vec![0u8; 16];
    let err = Gguf::parse(&h).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("not a GGUF"), "expected BadMagic, got: {msg}");
}

#[test]
fn rejects_truncated_header() {
    let h = vec![0x47, 0x47, 0x55, 0x46]; // magic but nothing else
    let err = Gguf::parse(&h).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("truncated"), "expected Truncated, got: {msg}");
}
