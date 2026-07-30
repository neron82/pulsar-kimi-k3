//! End-to-end: build a tiny K3-shaped BF16 gguf with 896-expert 3D slabs,
//! run pulsar-quant on it with a K3 recipe, parse the result, and verify
//! the standard-quant types and 3D slab geometry survive the round trip.
//!
//! Follows the writer_e2e.rs convention: hand-build input, invoke the
//! binary, parse output, check values.

use std::io::Write;

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

/// Build a tiny K3-shaped GGUF v3 with:
/// - 1 layer (to keep the fixture small)
/// - 896 experts in 3D slabs
/// - 2 shared experts
/// - KDA attention metadata
/// - BF16 source tensors
fn build_k3_tiny_input(path: &std::path::Path) {
    const HIDDEN: u64 = 128; // tiny hidden for speed
    const EXPERT_FFN: u64 = 64; // tiny expert FFN
    const N_EXPERTS: u64 = 896; // real K3 expert count
    const N_HEADS: u64 = 4; // tiny head count
    const HEAD_DIM: u64 = 32; // tiny head dim

    let mut h = Vec::new();
    h.extend_from_slice(&0x4655_4747u32.to_le_bytes()); // magic
    h.extend_from_slice(&3u32.to_le_bytes()); // version
    h.extend_from_slice(&8u64.to_le_bytes()); // 8 tensors
    h.extend_from_slice(&7u64.to_le_bytes()); // 7 KVs

    // Metadata
    put_str(&mut h, "general.architecture");
    h.extend_from_slice(&8u32.to_le_bytes()); // String
    put_str(&mut h, "kimi-k3");

    put_str(&mut h, "kimi-k3.block_count");
    h.extend_from_slice(&10u32.to_le_bytes()); // U64
    h.extend_from_slice(&1u64.to_le_bytes());

    put_str(&mut h, "kimi-k3.expert_count");
    h.extend_from_slice(&10u32.to_le_bytes());
    h.extend_from_slice(&N_EXPERTS.to_le_bytes());

    put_str(&mut h, "kimi-k3.expert_used_count");
    h.extend_from_slice(&10u32.to_le_bytes());
    h.extend_from_slice(&16u64.to_le_bytes());

    put_str(&mut h, "kimi-k3.shared_expert_count");
    h.extend_from_slice(&10u32.to_le_bytes());
    h.extend_from_slice(&2u64.to_le_bytes());

    put_str(&mut h, "kimi-k3.attention_layer");
    h.extend_from_slice(&9u32.to_le_bytes()); // Array
    h.extend_from_slice(&8u32.to_le_bytes()); // element type String
    h.extend_from_slice(&1u64.to_le_bytes()); // 1 element
    put_str(&mut h, "kda");

    put_str(&mut h, "kimi-k3.embedding_length");
    h.extend_from_slice(&10u32.to_le_bytes());
    h.extend_from_slice(&HIDDEN.to_le_bytes());

    // Generate BF16 source data
    let bf16_bytes = |n: usize| -> Vec<u8> {
        (0..n)
            .map(|i| {
                let x = ((i as f32) * 0.37).sin() * 0.8;
                ((x.to_bits() >> 16) as u16).to_le_bytes()
            })
            .flatten()
            .collect()
    };

    // Data section
    let mut data = Vec::new();
    let mut offs = Vec::new();

    // Tensor 0: attn_q.weight [head_dim*n_heads, hidden]
    let n0 = (HEAD_DIM * N_HEADS * HIDDEN) as usize;
    offs.push(data.len() as u64);
    data.extend_from_slice(&bf16_bytes(n0));
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor 1: ffn_gate_exps.weight [expert_ffn, hidden, 896] — 3D slab
    let n1 = (EXPERT_FFN * HIDDEN * N_EXPERTS) as usize;
    offs.push(data.len() as u64);
    data.extend_from_slice(&bf16_bytes(n1));
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor 2: ffn_up_exps.weight [expert_ffn, hidden, 896]
    let n2 = (EXPERT_FFN * HIDDEN * N_EXPERTS) as usize;
    offs.push(data.len() as u64);
    data.extend_from_slice(&bf16_bytes(n2));
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor 3: ffn_down_exps.weight [hidden, expert_ffn, 896]
    let n3 = (HIDDEN * EXPERT_FFN * N_EXPERTS) as usize;
    offs.push(data.len() as u64);
    data.extend_from_slice(&bf16_bytes(n3));
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor 4: ffn_router.weight [n_experts, hidden]
    let n4 = (N_EXPERTS * HIDDEN) as usize;
    offs.push(data.len() as u64);
    data.extend_from_slice(&bf16_bytes(n4));
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor 5: ffn_shared_gate.weight [expert_ffn, hidden]
    let n5 = (EXPERT_FFN * HIDDEN) as usize;
    offs.push(data.len() as u64);
    data.extend_from_slice(&bf16_bytes(n5));
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor 6: ffn_shared_up.weight [expert_ffn, hidden]
    let n6 = (EXPERT_FFN * HIDDEN) as usize;
    offs.push(data.len() as u64);
    data.extend_from_slice(&bf16_bytes(n6));
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor 7: attn_norm.weight [hidden] — 1D, stays F32
    let n7 = HIDDEN as usize;
    offs.push(data.len() as u64);
    let f32_bytes: Vec<u8> = (0..n7)
        .flat_map(|i| (i as f32 * 0.001).to_le_bytes())
        .collect();
    data.extend_from_slice(&f32_bytes);
    while data.len() % 32 != 0 {
        data.push(0);
    }

    // Tensor table
    let dims_2d_attn = [HEAD_DIM * N_HEADS, HIDDEN];
    let dims_3d_gate = [EXPERT_FFN, HIDDEN, N_EXPERTS];
    let dims_3d_up = [EXPERT_FFN, HIDDEN, N_EXPERTS];
    let dims_3d_down = [HIDDEN, EXPERT_FFN, N_EXPERTS];
    let dims_2d_router = [N_EXPERTS, HIDDEN];
    let dims_2d_shared = [EXPERT_FFN, HIDDEN];
    let dims_1d = [HIDDEN];

    let tensors: [(&str, &[u64], u32, usize); 8] = [
        ("blk.0.attn_q.weight", &dims_2d_attn, 30, offs[0] as usize), // BF16
        (
            "blk.0.ffn_gate_exps.weight",
            &dims_3d_gate,
            30,
            offs[1] as usize,
        ), // BF16
        (
            "blk.0.ffn_up_exps.weight",
            &dims_3d_up,
            30,
            offs[2] as usize,
        ), // BF16
        (
            "blk.0.ffn_down_exps.weight",
            &dims_3d_down,
            30,
            offs[3] as usize,
        ), // BF16
        (
            "blk.0.ffn_router.weight",
            &dims_2d_router,
            30,
            offs[4] as usize,
        ), // BF16
        (
            "blk.0.ffn_shared_gate.weight",
            &dims_2d_shared,
            30,
            offs[5] as usize,
        ), // BF16
        (
            "blk.0.ffn_shared_up.weight",
            &dims_2d_shared,
            30,
            offs[6] as usize,
        ), // BF16
        ("blk.0.attn_norm.weight", &dims_1d, 0, offs[7] as usize),    // F32
    ];

    for (name, dims, ty, offset) in &tensors {
        put_str(&mut h, name);
        h.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in *dims {
            h.extend_from_slice(&d.to_le_bytes());
        }
        h.extend_from_slice(&ty.to_le_bytes());
        h.extend_from_slice(&(*offset as u64).to_le_bytes());
    }

    while h.len() % 32 != 0 {
        h.push(0);
    }

    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&h).unwrap();
    f.write_all(&data).unwrap();
    drop(f);
}

#[test]
fn quantize_k3_tiny_model() {
    let dir = std::env::temp_dir().join(format!("pq-k3-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let input = dir.join("tiny-k3-BF16.gguf");
    let output = dir.join("tiny-k3-recipe.gguf");

    build_k3_tiny_input(&input);

    // Run pulsar-quant with a K3 recipe:
    // - expert slabs (_exps.) → q2_k
    // - shared experts (_shared_) → q3_k
    // - default → q8_0
    let st = std::process::Command::new(env!("CARGO_BIN_EXE_pulsar-quant"))
        .args(["-i"])
        .arg(&input)
        .args(["-o"])
        .arg(&output)
        .args(["--map", "_exps.=q2_k,_shared_=q3_k", "--default", "q8_0"])
        .status()
        .unwrap();
    assert!(st.success(), "pulsar-quant exited with {st}");

    // Parse output
    let head = std::fs::read(&output).unwrap();
    let g = gguf::Gguf::parse(&head).unwrap();

    // Architecture preserved
    assert_eq!(g.architecture(), Some("kimi-k3"));
    assert!(g.is_kimi_k3());
    assert_eq!(g.k3_expert_count(), Some(896));
    assert_eq!(g.k3_shared_expert_count(), Some(2));

    // Expert slabs should be Q2K (but fall back to Q8_0 when width % 256 != 0)
    let gate = g.tensor("blk.0.ffn_gate_exps.weight").expect("gate");
    // Our tiny fixture has width 64, which is not /256, so pulsar-quant
    // falls back to Q8_0.  Real K3 models have width 3072 (/256) and
    // will hit Q2K directly.
    assert_eq!(gate.ty, gguf::TensorType::Q8_0);
    assert!(gate.is_3d_slab());
    assert_eq!(gate.slab_expert_count(), Some(896));
    assert!(gate.slab_expert_byte_size().is_some());

    let up = g.tensor("blk.0.ffn_up_exps.weight").expect("up");
    assert_eq!(up.ty, gguf::TensorType::Q8_0);
    assert!(up.is_3d_slab());

    let down = g.tensor("blk.0.ffn_down_exps.weight").expect("down");
    assert_eq!(down.ty, gguf::TensorType::Q8_0);
    assert!(down.is_3d_slab());

    // Shared experts should be Q3K (but fall back to Q8_0 when width % 256 != 0)
    let shared_gate = g
        .tensor("blk.0.ffn_shared_gate.weight")
        .expect("shared gate");
    assert_eq!(shared_gate.ty, gguf::TensorType::Q8_0);

    let shared_up = g.tensor("blk.0.ffn_shared_up.weight").expect("shared up");
    assert_eq!(shared_up.ty, gguf::TensorType::Q8_0);
    let attn_q = g.tensor("blk.0.attn_q.weight").expect("attn_q");
    assert_eq!(attn_q.ty, gguf::TensorType::Q8_0);

    // 1D norm should be F32
    let norm = g.tensor("blk.0.attn_norm.weight").expect("norm");
    assert_eq!(norm.ty, gguf::TensorType::F32);

    // Router should be Q8_0 (default, 2D)
    let router = g.tensor("blk.0.ffn_router.weight").expect("router");
    assert_eq!(router.ty, gguf::TensorType::Q8_0);

    // All tensors must have computable byte sizes
    for t in &g.tensors {
        assert!(
            t.byte_size().is_some(),
            "tensor {} has unmodeled type {:?}",
            t.name,
            t.ty
        );
    }

    // Data offset must be aligned
    assert!(g.data_offset % g.alignment == 0);

    // Verify the 3D slab dimensions survived the round trip
    let gate = g.tensor("blk.0.ffn_gate_exps.weight").unwrap();
    assert_eq!(gate.dims.len(), 3);
    assert_eq!(gate.dims[2], 896);

    // Clean up
    std::fs::remove_dir_all(&dir).ok();
}
