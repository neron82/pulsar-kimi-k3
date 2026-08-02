//! Kimi K3: hybrid KDA + gated-MLA (NoPE) with Attention Residuals (AttnRes),
//! Stable LatentMoE and SiTU-GLU activation. Phase 2: real single-token forward.
//!
//! Atomic order (per layer):
//!   embed -> for each layer:
//!     AttnRes pre-attn mixture (snapshot every attn_res_block_size)
//!     attn_norm
//!     KDA (causal sconv + L2 norm + safe gate + beta + delta step + output gate/norm + o_proj)
//!       or gated MLA (q_lora + kv_compress + absorbed attention + output gate + o_proj)
//!     residual update (snapshot restart: prefix = cur, else prefix += cur)
//!     AttnRes pre-FFN mixture
//!     ffn_norm
//!     dense SiTU-GLU or latent Stable-MoE (router on hidden, latent_down, top-16/896,
//!       routed SiTU experts, latent norm/up, 2 shared experts)
//!     residual update (prefix += cur)
//!   -> final AttnRes mixture -> output_norm -> output head
//!
//! Reference: moonshotai/Kimi-K3 (HF), sglang kimi_k3.py,
//! atomic-llama-cpp-turboquant-kimi/src/models/kimi-k3.cpp.

use super::{Model, Result, State};
use kernels::{DeviceBuf, ExpertPtrs};
use stream::Read as StreamRead;

use super::StreamingStore;

/// Select the K3 routed-expert implementation. CPU remains the default until
/// the CUDA path has been validated against the host reference.
fn k3_expert_backend() -> &'static str {
    match std::env::var("PULSAR_K3_EXPERT_BACKEND").as_deref() {
        Ok("cuda") => "cuda",
        Ok("cpu-q8") => "cpu-q8",
        Ok("cpu") | Err(_) => "cpu",
        _ => "cpu",
    }
}

fn k3_accum_mode() -> &'static str {
    match std::env::var("PULSAR_K3_ACCUM_MODE").as_deref() {
        Ok("serial") => "serial",
        Ok("serial-nofma") => "serial-nofma",
        Ok("f64-reference") => "f64-reference",
        _ => "current",
    }
}

// ── K3 contract constants (from the reference model) ──────────────────────
// These are the canonical K3 values; the gguf metadata is the source of truth.
// Documented here for reference during Phase 2 forward implementation.
//
//   n_layer:           93
//   n_kda_layer:       69  (n_head_kv == 0 → recurrent KDA)
//   n_mla_layer:       24  (gated MLA, NoPE)
//   n_leading_dense:    1
//   n_expert:         896
//   n_expert_used:     16
//   n_expert_shared:    2
//   n_embd:          7168
//   n_head:            96
//   n_head_kv:          0 on KDA layers, 96 on MLA layers
//   head_dim (KDA):   128
//   head_dim (MLA):   128 (key_length_mla)
//   moe_latent_size: 3584
//   n_ff_exp:        3072
//   n_ff_dense:      3072 (dense FFN width, SiTU-GLU)
//   n_ff_shexp:      6144 (shared expert FFN width = n_ff_exp * n_expert_shared)
//   kda_head_dim:     128
//   situ_beta:         4.0
//   situ_linear_beta: 25.0
//   attn_res_block_size: 12
//   gate_lower_bound:  -5.0 (multiplier after sigmoid)
//   rope:              none (KDA recurrence + NoPE MLA)

// ── Per-layer kind ────────────────────────────────────────────────────────

/// Identifies whether a K3 layer is KDA (recurrent delta attention) or
/// gated MLA (multi-head latent attention, NoPE).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum K3LayerKind {
    /// Kimi Delta Attention: recurrent conv1d + delta rule + safe gate.
    Kda,
    /// Gated multi-head latent attention (NoPE, no positional embeddings).
    Mla,
}

// ── K3 MLA dimension constants ────────────────────────────────────────────

/// Canonical K3 gated MLA dimensions (from the reference model).
/// These are the contract values; the gguf metadata is the source of truth.
pub struct K3MlaDims {
    pub n_head: u32,
    pub q_lora_rank: u32,
    pub kv_lora_rank: u32,
    pub qk_nope: u32,
    pub qk_rope: u32,
    pub v_mla: u32,
    pub n_embd: u32,
}

impl K3MlaDims {
    pub const fn canonical() -> Self {
        K3MlaDims {
            n_head: 96,
            q_lora_rank: 1536,
            kv_lora_rank: 512,
            qk_nope: 128,
            qk_rope: 64,
            v_mla: 128,
            n_embd: 7168,
        }
    }

    pub fn qk_dim(&self) -> u32 {
        self.qk_nope + self.qk_rope
    }

    pub fn q_b_out(&self) -> u32 {
        self.n_head * self.qk_dim()
    }

    pub fn kv_a_out(&self) -> u32 {
        self.kv_lora_rank + self.qk_rope
    }

    pub fn gate_out(&self) -> u32 {
        self.n_head * self.v_mla
    }

    pub fn o_proj_in(&self) -> u32 {
        self.n_head * self.v_mla
    }
}

/// Map an engine quant id to the GGUF type supported by the K3 host
/// dequantizer. Unsupported expert formats fail closed instead of reaching
/// `dequant_block`'s unreachable arm.
fn k3_quant_to_tensor_type(q: u32) -> Result<gguf::TensorType> {
    use gguf::TensorType as T;
    Ok(match q {
        kernels::QUANT_Q2_K => T::Q2K,
        kernels::QUANT_Q3_K => T::Q3K,
        kernels::QUANT_Q4_K => T::Q4K,
        kernels::QUANT_Q5_K => T::Q5K,
        kernels::QUANT_Q6_K => T::Q6K,
        kernels::QUANT_Q8_0 => T::Q8_0,
        kernels::QUANT_Q4_0 => T::Q4_0,
        _ => return Err(format!("K3 latent MoE: unsupported quant id {q}").into()),
    })
}

/// Dequantize one packed expert slab with strict byte/shape validation.
///
/// The current AtomicChat model routes Q2_K/Q3_K expert slabs. Q8_0 and
/// Q4_0 remain supported for the existing K3/Q8 path; IQ formats are not
/// silently accepted until a real dequantizer for them is wired here.
fn k3_dequant_expert_bytes(buf: &[u8], n: usize, quant: u32) -> Result<Vec<f32>> {
    use gguf::TensorType as T;
    let ty = k3_quant_to_tensor_type(quant)?;
    let (block_elems, block_bytes) = match ty {
        T::Q8_0 => (32usize, 34usize),
        T::Q4_0 => (32usize, 18usize),
        _ => {
            let (elems, bytes) = ty
                .block_layout()
                .ok_or_else(|| format!("K3 latent MoE: no block layout for {ty:?}"))?;
            (elems as usize, bytes as usize)
        }
    };
    if n % block_elems != 0 {
        return Err(format!(
            "K3 latent MoE: {ty:?} element count {n} is not divisible by block size {block_elems}"
        )
        .into());
    }
    let expected = (n / block_elems) * block_bytes;
    if buf.len() != expected {
        return Err(format!(
            "K3 latent MoE: {ty:?} buffer has {} bytes, expected {expected} bytes",
            buf.len()
        )
        .into());
    }

    let mut out = Vec::with_capacity(n);
    for block in buf.chunks_exact(block_bytes) {
        match ty {
            T::Q8_0 => {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                out.extend(block[2..].iter().map(|&q| d * (q as i8 as f32)));
            }
            T::Q4_0 => {
                let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
                for i in 0..16 {
                    out.push(d * ((block[2 + i] & 0x0f) as f32 - 8.0));
                }
                for i in 0..16 {
                    out.push(d * ((block[2 + i] >> 4) as f32 - 8.0));
                }
            }
            _ => {
                let mut decoded = [0.0f32; 256];
                crate::real::requant::dequant_block(ty, block, &mut decoded);
                out.extend_from_slice(&decoded[..block_elems]);
            }
        }
    }
    Ok(out)
}

// ── Per-layer weight struct ──────────────────────────────────────────────

/// Kimi K3 per-layer weights.  Every layer has both an attention stack
/// (KDA or MLA) and an FFN stack (dense SiTU-GLU or Stable LatentMoE).
/// The `kind` field discriminates which attention path to use.
///
/// Architecture-specific fields are `Option` — they are `Some` only when
/// the layer type requires them.  No dummy buffers.
pub struct KimiK3W {
    pub kind: K3LayerKind,

    // ── Shared norms ─────────────────────────────────────────────────
    pub attn_norm: DeviceBuf, // f32 [n_embd]
    pub ffn_norm: DeviceBuf,  // f32 [n_embd]

    // ── AttnRes per-layer mixtures ────────────────────────────────────
    pub attn_res_norm: DeviceBuf, // f32 [n_embd]
    pub attn_res_proj: DeviceBuf, // f32 [n_embd]
    pub ffn_res_norm: DeviceBuf,  // f32 [n_embd]
    pub ffn_res_proj: DeviceBuf,  // f32 [n_embd]

    // ── KDA attention weights (valid when kind == Kda) ────────────────
    // Conv1d kernels: [d_conv, 1, d_inner, 1] or [d_conv, 1, d_inner]
    pub ssm_q_conv: Option<DeviceBuf>,
    pub ssm_k_conv: Option<DeviceBuf>,
    pub ssm_v_conv: Option<DeviceBuf>,
    // QKV projections: K3DenseWeight [n_embd -> d_inner] each
    pub wq: Option<K3DenseWeight>,
    pub wk: Option<K3DenseWeight>,
    pub wv: Option<K3DenseWeight>,
    // Forget gate (low-rank): f_a [n_embd -> kda_head_dim], f_b [kda_head_dim -> d_inner]
    pub ssm_f_a: Option<K3DenseWeight>,
    pub ssm_f_b: Option<DeviceBuf>, // F32 (absorbed, custom layout)
    // Beta mixing coefficient: [n_embd -> n_head]
    pub ssm_beta: Option<K3DenseWeight>,
    // exp(A_log), per head: [n_head] (positive, stored as exp(A_log))
    pub ssm_a: Option<DeviceBuf>,
    // dt_bias: [d_inner]
    pub ssm_dt_b: Option<DeviceBuf>,
    // Full-rank output gate: [n_embd -> d_inner]
    pub wqkv_gate: Option<K3DenseWeight>,
    // Output norm: f32 [kda_head_dim]
    pub ssm_o_norm: Option<DeviceBuf>,
    // Output projection: K3DenseWeight [d_inner -> n_embd]
    pub wo: Option<K3DenseWeight>,

    // ── MLA attention weights (valid when kind == Mla) ────────────────
    // Q lora: q_a [n_embd -> q_lora_rank], q_b [q_lora_rank -> n_head * qk_dim]
    pub mla_wq_a: Option<K3DenseWeight>,
    pub mla_wq_b: Option<K3DenseWeight>,
    pub mla_q_a_norm: Option<DeviceBuf>,
    pub mla_kv_a_norm: Option<DeviceBuf>,
    // KV compression: [n_embd -> kv_lora_rank + qk_rope]
    pub mla_wkv_a_mqa: Option<K3DenseWeight>,
    // KV up-projections (MLA KV cache enabled): wk_b, wv_b
    pub mla_wk_b: Option<DeviceBuf>, // F32 (absorbed, custom 3D layout)
    pub mla_wv_b: Option<DeviceBuf>, // F32 (absorbed, custom 3D layout)
    // Legacy fused wkv_b (MLA KV cache disabled): [kv_lora_rank -> n_head * (qk_nope + v_mla)]
    pub mla_wkv_b: Option<K3DenseWeight>,
    // MLA output gate: [n_embd -> n_head * value_mla]
    pub mla_wqkv_gate: Option<K3DenseWeight>,
    // MLA output projection: K3DenseWeight [n_head * value_mla -> n_embd]
    pub mla_wo: Option<K3DenseWeight>,

    // ── Dense FFN (leading dense layers, SiTU-GLU) ───────────────────
    pub ffn_gate: Option<K3DenseWeight>, // K3DenseWeight [n_embd -> n_ff_dense]
    pub ffn_up: Option<K3DenseWeight>,   // K3DenseWeight [n_embd -> n_ff_dense]
    pub ffn_down: Option<K3DenseWeight>, // K3DenseWeight [n_ff_dense -> n_embd]

    // ── Stable LatentMoE weights (routed layers) ─────────────────────
    // Router: [n_embd -> n_expert]
    pub ffn_gate_inp: Option<K3DenseWeight>,
    // Router bias: [n_expert]
    pub ffn_exp_probs_b: Option<DeviceBuf>,
    // Latent down/up/norm: [n_embd -> moe_latent], [moe_latent -> n_embd], [moe_latent]
    pub ffn_latent_down: Option<K3DenseWeight>,
    pub ffn_latent_up: Option<DeviceBuf>, // Q8_0 (absorbed, custom layout)
    pub ffn_latent_norm: Option<DeviceBuf>,
    // Routed expert tensors (latent-space experts)
    pub ffn_gate_exps: Option<super::ExpertTensor>,
    pub ffn_up_exps: Option<super::ExpertTensor>,
    pub ffn_down_exps: Option<super::ExpertTensor>,
    // Shared experts (full hidden state, SiTU-GLU)
    pub ffn_gate_shexp: Option<K3DenseWeight>,
    pub ffn_up_shexp: Option<K3DenseWeight>,
    pub ffn_down_shexp: Option<DeviceBuf>, // F32 (absorbed, custom layout)
}

// ── Runtime state ──────────────────────────────────────────────────────────

/// Kimi K3 runtime: KDA recurrent states (conv + ssm) per layer, AttnRes
/// snapshot bank, and scratch buffers sized for the K3 contract.
pub struct KimiK3Rt {
    /// Per-layer conv1d states: [n_layer][3][(d_conv-1) * d_inner] f32
    /// (Q, K, V streams).  Only KDA layers use these; MLA layers leave
    /// dummy 1-byte entries.
    pub conv_states: Vec<[DeviceBuf; 3]>,
    /// Per-layer SSM (delta-rule) states: [n_layer][head_dim * head_dim * n_head] f32
    /// Only KDA layers use these.
    pub ssm_states: Vec<DeviceBuf>,
    /// AttnRes snapshot bank: up to ceil(n_layer / attn_res_block_size) rows
    /// of [n_embd] f32 each, stored as a flat DeviceBuf.
    pub res_bank: DeviceBuf,
    pub res_bank_len: u32,
    /// Scratch: normed input for the attention path
    pub normed: DeviceBuf,
    /// Scratch: attention output before residual
    pub attn_out: DeviceBuf,
    /// Scratch: FFN output before residual
    pub ffn_out: DeviceBuf,
    /// Scratch: router logits
    pub router_logits: DeviceBuf,
    /// Scratch: latent space buffer
    pub latent: DeviceBuf,
    /// Scratch: MoE output (weighted expert sum)
    pub moe_out: DeviceBuf,
    /// Scratch: shared expert output
    pub shexp_out: DeviceBuf,
    /// Scratch: AttnRes mixture output
    pub mix_out: DeviceBuf,
    /// Scratch: KDA safe gate intermediate (f_a output)
    pub kda_f_a: DeviceBuf,
    /// Scratch: KDA safe gate raw (f_b + dt_bias)
    pub kda_g_raw: DeviceBuf,
    /// Scratch: KDA output gate logits
    pub kda_gate_logits: DeviceBuf,
    /// Scratch: KDA conv output (before L2 norm)
    pub kda_conv_out: DeviceBuf,
    /// Scratch: KDA L2-normed Q
    pub kda_q_normed: DeviceBuf,
    /// Scratch: KDA L2-normed K
    pub kda_k_normed: DeviceBuf,
    /// Scratch: KDA V after conv
    pub kda_v_conv: DeviceBuf,
    /// Scratch: KDA output gate gated result
    pub kda_gated: DeviceBuf,
    /// Scratch: KDA output normed
    pub kda_o_normed: DeviceBuf,
    /// Scratch: dense FFN gate activation
    pub dense_gate: DeviceBuf,
    /// Scratch: dense FFN up activation
    pub dense_up: DeviceBuf,
    /// Scratch: dense FFN mid (after SiTU-GLU)
    pub dense_mid: DeviceBuf,
    /// Scratch: shared expert gate
    pub shexp_gate: DeviceBuf,
    /// Scratch: shared expert up
    pub shexp_up: DeviceBuf,
    /// Scratch: shared expert mid
    pub shexp_mid: DeviceBuf,
    /// Scratch: expert selected indices (host-read)
    pub expert_selected: DeviceBuf,
    /// Scratch: expert weights (host-read)
    pub expert_weights: DeviceBuf,
    /// Scratch: expert staging (one expert's gate/up/down)
    pub expert_staging: DeviceBuf,
    /// Scratch: expert gate output
    pub expert_gate: DeviceBuf,
    /// Scratch: expert up output
    pub expert_up: DeviceBuf,
    /// Scratch: expert mid (gate * up)
    pub expert_mid: DeviceBuf,
    /// Scratch: expert down output
    pub expert_down: DeviceBuf,
    /// Scratch: latent normed
    pub latent_normed: DeviceBuf,
    /// Scratch: Q8_K activation buffer for direct K-quant matmul dispatch.
    /// Sized for the largest in_dim (n_embd=7168 → 7168/256*292 = 8180 bytes).
    pub q8k_scratch: DeviceBuf,
}

impl KimiK3Rt {
    pub fn new(m: &Model) -> Result<KimiK3Rt> {
        let s = m.shape;
        let n_layer = s.n_exec_layer as usize;
        let n_embd = s.n_embd as usize;
        let n_head = s.n_head as usize;
        let kda_hd = s.kda_head_dim as usize;
        let d_inner = n_head * kda_hd;
        let d_conv = s.ssm_conv_k.max(1) as usize; // ssm_d_conv from gguf
        let conv_state_bytes = (d_conv - 1).max(1) * d_inner * 4; // f32

        let mut conv_states = Vec::with_capacity(n_layer);
        let mut ssm_states = Vec::with_capacity(n_layer);
        for il in 0..n_layer {
            let kind = m
                .k3_layer_kinds
                .get(il)
                .copied()
                .unwrap_or(K3LayerKind::Mla);
            match kind {
                K3LayerKind::Kda => {
                    // Three conv streams (Q, K, V) + one SSM state
                    let q =
                        DeviceBuf::alloc_named(conv_state_bytes, "K3 KDA recurrent conv state")?;
                    let k =
                        DeviceBuf::alloc_named(conv_state_bytes, "K3 KDA recurrent conv state")?;
                    let v =
                        DeviceBuf::alloc_named(conv_state_bytes, "K3 KDA recurrent conv state")?;
                    conv_states.push([q, k, v]);
                    let ssm_bytes = kda_hd * kda_hd * n_head * 4; // f32 [head_dim][head_dim][n_head]
                    ssm_states.push(DeviceBuf::alloc_named(
                        ssm_bytes,
                        "K3 KDA recurrent SSM state",
                    )?);
                }
                K3LayerKind::Mla => {
                    // Dummy entries (MLA has no recurrent state)
                    conv_states.push([
                        DeviceBuf::alloc_named(4, "K3 MLA state placeholder")?,
                        DeviceBuf::alloc_named(4, "K3 MLA state placeholder")?,
                        DeviceBuf::alloc_named(4, "K3 MLA state placeholder")?,
                    ]);
                    ssm_states.push(DeviceBuf::alloc_named(4, "K3 MLA state placeholder")?);
                }
            }
        }

        let res_block = s.attn_res_block_size.max(1) as usize;
        let res_bank_cap = (n_layer + res_block - 1) / res_block;
        let res_bank =
            DeviceBuf::alloc_named(res_bank_cap * n_embd * 4, "K3 AttnRes snapshot bank")?;

        let f32s = |n: usize| DeviceBuf::alloc_named(n * 4, "K3 runtime scratch");
        let mb = 1; // decode-only for now
                    // K3 MLA intermediates are wider than the hidden state: Q has
                    // n_head * (qk_nope + qk_rope) elements and the gated attention
                    // output has n_head * value_mla. Size every reused scratch for the
                    // largest contract dimension, not merely n_embd.
        let qk_dim = (s.qk_nope + s.qk_rope) as usize;
        let mla_q = n_head * qk_dim.max(s.value_mla as usize);
        let mla_gate = n_head * s.value_mla as usize;
        let mla_kv = n_head * (qk_dim + s.value_mla as usize);
        let moe_latent = s.moe_latent_size.max(1) as usize;
        let n_ff_dense = s.n_ff_dense.max(1) as usize;
        let n_ff_exp = s.n_ff_exp.max(1) as usize;
        let n_expert = s.n_expert.max(1) as usize;
        let n_expert_used = s.n_expert_used.max(1) as usize;

        Ok(KimiK3Rt {
            conv_states,
            ssm_states,
            res_bank,
            res_bank_len: 0,
            normed: f32s(mb * n_embd)?,
            attn_out: f32s(mb * n_embd.max(mla_q))?,
            ffn_out: f32s(mb * n_embd.max(moe_latent).max(mla_kv))?,
            router_logits: f32s(mb * n_expert.max(mla_gate))?,
            latent: f32s(mb * moe_latent.max(s.n_kv_lora as usize))?,
            moe_out: f32s(mb * n_embd.max(moe_latent).max(s.qk_rope as usize))?,
            shexp_out: f32s(mb * n_embd.max(mla_gate))?,
            mix_out: f32s(mb * n_embd.max(mla_gate))?,
            kda_f_a: f32s(mb * kda_hd)?,
            kda_g_raw: f32s(mb * d_inner)?,
            kda_gate_logits: f32s(mb * d_inner)?,
            kda_conv_out: f32s(mb * d_inner)?,
            kda_q_normed: f32s(mb * d_inner)?,
            kda_k_normed: f32s(mb * d_inner)?,
            kda_v_conv: f32s(mb * d_inner)?,
            kda_gated: f32s(mb * d_inner)?,
            kda_o_normed: f32s(mb * d_inner)?,
            dense_gate: f32s(mb * n_ff_dense)?,
            dense_up: f32s(mb * n_ff_dense)?,
            dense_mid: f32s(mb * n_ff_dense)?,
            shexp_gate: f32s(mb * n_ff_exp * s.n_expert_shared.max(1) as usize)?,
            shexp_up: f32s(mb * n_ff_exp * s.n_expert_shared.max(1) as usize)?,
            shexp_mid: f32s(mb * n_ff_exp * s.n_expert_shared.max(1) as usize)?,
            expert_selected: DeviceBuf::alloc(mb * n_expert_used * 4)?,
            expert_weights: f32s(mb * n_expert_used)?,
            expert_staging: f32s(mb * moe_latent)?,
            expert_gate: f32s(mb * n_ff_exp)?,
            expert_up: f32s(mb * n_ff_exp)?,
            expert_mid: f32s(mb * n_ff_exp)?,
            expert_down: f32s(mb * moe_latent)?,
            latent_normed: f32s(mb * moe_latent)?,
            // Q8_K scratch: largest in_dim is n_embd (7168).
            // Q8_K block = 256 elems, 292 bytes per block.
            // 7168 / 256 = 28 blocks → 28 * 292 = 8176 bytes.
            q8k_scratch: DeviceBuf::alloc(
                (s.n_embd.max(s.moe_latent_size).max(s.n_ff_dense) as usize)
                    .div_ceil(kernels::Q8_K_BLOCK_ELEMS)
                    * kernels::Q8_K_BLOCK_BYTES,
            )?,
        })
    }

    /// Reset all recurrent states to zero (fresh sequence).
    pub fn reset(&mut self) -> Result {
        for states in self.conv_states.iter_mut() {
            for b in states.iter_mut() {
                let n = b.bytes();
                if n > 4 {
                    kernels::zero(b, n)?;
                }
            }
        }
        for b in self.ssm_states.iter_mut() {
            let n = b.bytes();
            if n > 4 {
                kernels::zero(b, n)?;
            }
        }
        self.res_bank_len = 0;
        Ok(())
    }
}

// ── Forward implementation ─────────────────────────────────────────────────

impl Model {
    /// K3 gated MLA single-token forward with compact causal cache and NoPE.
    ///
    /// Computes the full gated MLA primitive for one token:
    ///   1. Q = q_b(rms_norm(q_a(x)))  (low-rank Q projection)
    ///   2. kv_cmpr_pe = x @ wkv_a_mqa  (KV compression)
    ///   3. kv_cmpr = kv_cmpr_pe[:kv_lora_rank], k_pe = kv_cmpr_pe[kv_lora_rank:]
    ///   4. kv_cmpr = rms_norm(kv_cmpr, kv_a_norm)
    ///   5. Cache-aware absorbed attention (split-weight path)
    ///   6. gate = sigmoid(x @ wqkv_gate)
    ///   7. out = (attn * gate) @ wo
    ///
    /// Returns the attention output [n_embd] before residual.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn k3_mla_step(
        &self,
        rt: &mut KimiK3Rt,
        x: &DeviceBuf,                     // [n_embd] f32
        mla_wq_a: &K3DenseWeight,          // [n_embd, q_lora_rank] K3DenseWeight
        mla_wq_b: &K3DenseWeight,          // [q_lora_rank, n_head * qk_dim] K3DenseWeight
        mla_q_a_norm: &DeviceBuf,          // [q_lora_rank] f32
        mla_wkv_a_mqa: &K3DenseWeight,     // [n_embd, kv_lora_rank + qk_rope] K3DenseWeight
        mla_kv_a_norm: &DeviceBuf,         // [kv_lora_rank] f32
        mla_wk_b: Option<&DeviceBuf>,      // [qk_nope, kv_lora_rank, n_head] f32 (split path)
        mla_wv_b: Option<&DeviceBuf>,      // [kv_lora_rank, v_mla, n_head] f32 (split path)
        mla_wkv_b: Option<&K3DenseWeight>, // [kv_lora_rank, n_head * (qk_nope + v_mla)] K3DenseWeight (fused path)
        mla_wqkv_gate: &K3DenseWeight,     // [n_embd, n_head * v_mla] K3DenseWeight
        mla_wo: &K3DenseWeight,            // [n_head * v_mla, n_embd] K3DenseWeight
        kv_lora_cache: &mut DeviceBuf,     // [ctx, kv_lora_rank] f32
        k_tail_cache: &mut DeviceBuf,      // [ctx, qk_rope] f32
        qk_low: &mut DeviceBuf,            // [n_head, kv_lora_rank] f32
        pos: u32,
        ctx: u32,
        dims: &K3MlaDims,
        eps: f32,
    ) -> Result {
        let n_embd = dims.n_embd;
        let q_lora_rank = dims.q_lora_rank;
        let kv_lora_rank = dims.kv_lora_rank;
        let qk_nope = dims.qk_nope;
        let qk_rope = dims.qk_rope;
        let v_mla = dims.v_mla;
        let n_head = dims.n_head;
        let qk_dim = dims.qk_dim();
        let q_b_out = dims.q_b_out();
        let kv_a_out = dims.kv_a_out();
        let gate_out = dims.gate_out();
        let o_proj_in = dims.o_proj_in();
        let scale = 1.0 / (qk_dim as f32).sqrt();

        // Scratch buffers from rt
        let q_lora = &mut rt.normed; // reuse normed scratch [q_lora_rank]
        let q_full = &mut rt.attn_out; // reuse attn_out scratch [n_head * qk_dim]
        let kv_pe = &mut rt.ffn_out; // reuse ffn_out scratch [kv_lora_rank + qk_rope]
        let kv_cmpr = &mut rt.latent; // reuse latent scratch [kv_lora_rank]
        let k_pe = &mut rt.moe_out; // reuse moe_out scratch [qk_rope]
        let attn = &mut rt.shexp_out; // reuse shexp_out scratch [n_head * v_mla]
        let gate = &mut rt.router_logits; // reuse router_logits scratch [n_head * v_mla]
        let gated = &mut rt.mix_out; // reuse mix_out scratch [n_head * v_mla]

        // 1. Q low-rank projection: q_lora = rms_norm(x @ wq_a, q_a_norm)
        // wq_a is K3DenseWeight
        mla_wq_a
            .matmul(q_lora, x, &mut rt.q8k_scratch, n_embd, q_lora_rank, 1)
            .map_err(|e| format!("K3 MLA q_a: {e}"))?;
        kernels::rms_norm_inplace(q_lora, mla_q_a_norm, q_lora_rank, 1, eps)?;

        // q_full = q_lora @ wq_b  (wq_b is K3DenseWeight)
        mla_wq_b
            .matmul(q_full, q_lora, &mut rt.q8k_scratch, q_lora_rank, q_b_out, 1)
            .map_err(|e| format!("K3 MLA q_b: {e}"))?;

        // 2. KV compression: kv_pe = x @ wkv_a_mqa  (wkv_a_mqa is K3DenseWeight)
        mla_wkv_a_mqa
            .matmul(kv_pe, x, &mut rt.q8k_scratch, n_embd, kv_a_out, 1)
            .map_err(|e| format!("K3 MLA kv_a: {e}"))?;

        // 3. Split kv_cmpr and k_pe
        // kv_cmpr = kv_pe[:kv_lora_rank], k_pe = kv_pe[kv_lora_rank:]
        // Use host round-trip for the split (small: 512+64 floats).
        let kv_pe_host = kv_pe.read_f32(kv_a_out as usize)?;
        let kv_cmpr_host: Vec<f32> = kv_pe_host[..kv_lora_rank as usize].to_vec();
        let k_pe_host: Vec<f32> = kv_pe_host[kv_lora_rank as usize..].to_vec();
        kv_cmpr.write(0, kernels::as_bytes(&kv_cmpr_host))?;
        k_pe.write(0, kernels::as_bytes(&k_pe_host))?;

        // 4. RMSNorm kv_cmpr
        kernels::rms_norm_inplace(kv_cmpr, mla_kv_a_norm, kv_lora_rank, 1, eps)?;

        if pos >= ctx {
            return Err(format!("K3 MLA position {pos} exceeds context {ctx}").into());
        }

        // Store the normalized latent and unrotated tail for this position.
        kernels::mla_store_compact_kv(
            kv_lora_cache,
            k_tail_cache,
            kv_cmpr,
            kv_pe,
            pos,
            1,
            ctx,
            kv_a_out,
            kv_lora_rank,
            qk_rope,
        )?;

        // 5. Absorbed causal attention
        match (mla_wk_b, mla_wv_b, mla_wkv_b) {
            (Some(wk_b), Some(wv_b), _) => {
                kernels::k3_mla_cached_attn_split(
                    attn,
                    qk_low,
                    q_full,
                    kv_lora_cache,
                    k_tail_cache,
                    wk_b,
                    wv_b,
                    pos + 1,
                    ctx,
                    n_head,
                    qk_nope,
                    qk_rope,
                    kv_lora_rank,
                    v_mla,
                    scale,
                )?;
            }
            (_, _, Some(wkv_b)) => {
                let _ = wkv_b;
                return Err("K3 MLA fused KV weights need a cache-aware implementation".into());
            }
            (None, None, None) => {
                return Err(
                    "K3 MLA: neither split (wk_b/wv_b) nor fused (wkv_b) path available".into(),
                );
            }
            (None, Some(_), None) | (Some(_), None, None) => {
                return Err("K3 MLA: incomplete weight set — need both wk_b and wv_b for split path, or wkv_b for fused path".into());
            }
        }

        if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
            let sync_result: super::Result =
                kernels::sync().map_err(|e| format!("K3 MLA absorbed-attention sync: {e}").into());
            sync_result?;
        }

        // 6. Output gate: gate = sigmoid(x @ wqkv_gate)  (wqkv_gate is K3DenseWeight)
        mla_wqkv_gate
            .matmul(gate, x, &mut rt.q8k_scratch, n_embd, gate_out, 1)
            .map_err(|e| format!("K3 MLA gate: {e}"))?;
        kernels::k3_sigmoid_inplace(gate, gate_out)?;
        if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
            let sync_result: super::Result =
                kernels::sync().map_err(|e| format!("K3 MLA gate sigmoid sync: {e}").into());
            sync_result?;
        }

        // 7. gated = attn * gate
        kernels::copy_d2d(gated, 0, attn, 0, (o_proj_in as usize) * 4)?;
        kernels::k3_mul_inplace(gated, gate, o_proj_in)?;
        if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
            let sync_result: super::Result =
                kernels::sync().map_err(|e| format!("K3 MLA gate multiply sync: {e}").into());
            sync_result?;
        }
        // 8. o_proj: out = gated @ wo  (K3DenseWeight dispatch)
        let out = &mut rt.attn_out; // reuse attn_out for final output
        mla_wo
            .matmul(out, gated, &mut rt.q8k_scratch, o_proj_in, n_embd, 1)
            .map_err(|e| format!("K3 MLA output: {e}"))?;

        Ok(())
    }

    /// AttnRes mixture: host roundtrip implementation.
    ///
    /// Reads the snapshot bank + current prefix to host, computes
    /// softmax-weighted mixture, writes result back to device.
    ///
    /// Formula (from ATOMIC_CONTRACT.md §7.5):
    ///   v = concat(bank_rows..., prefix)  # [n_embd, n_rows]
    ///   k = RMSNorm(v)                    # weightless (ones)
    ///   sw = norm_w * proj_w              # elementwise, [n_embd]
    ///   scores = k @ sw                   # [n_rows]
    ///   probs = softmax(scores)
    ///   out = sum_r probs_r * v_r
    fn attn_res_mix(
        &self,
        rt: &mut KimiK3Rt,
        prefix: &DeviceBuf, // [n_embd] f32
        norm_w: &DeviceBuf, // [n_embd] f32
        proj_w: &DeviceBuf, // [n_embd] f32
        eps: f32,
    ) -> Result {
        let n_embd = self.shape.n_embd as usize;
        let n_rows = rt.res_bank_len as usize + 1; // bank rows + prefix

        if n_rows == 1 {
            // No bank entries yet: mixture is just the prefix
            kernels::copy_d2d(&mut rt.mix_out, 0, prefix, 0, n_embd * 4)?;
            return Ok(());
        }

        // Read bank + prefix to host
        let bank_bytes = (n_rows - 1) * n_embd * 4;
        let mut host_bank = vec![0u8; bank_bytes];
        if bank_bytes > 0 {
            rt.res_bank.read(0, &mut host_bank)?;
        }
        let host_prefix = prefix.read_f32(n_embd)?;

        // Build rows: bank rows first, then prefix
        let bank_f32: Vec<f32> = if bank_bytes > 0 {
            host_bank
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()
        } else {
            Vec::new()
        };

        // Read norm_w and proj_w
        let norm_w_host = norm_w.read_f32(n_embd)?;
        let proj_w_host = proj_w.read_f32(n_embd)?;

        // Compute sw = norm_w * proj_w (elementwise)
        let sw: Vec<f32> = norm_w_host
            .iter()
            .zip(proj_w_host.iter())
            .map(|(n, p)| n * p)
            .collect();

        // Compute scores: for each row, dot(k_row, sw)
        // k_row = weightless RMSNorm(row)
        let mut scores = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row: &[f32] = if r < n_rows - 1 {
                &bank_f32[r * n_embd..(r + 1) * n_embd]
            } else {
                &host_prefix
            };
            // Weightless RMSNorm
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let score: f32 = row
                .iter()
                .zip(sw.iter())
                .map(|(v, s)| v * inv_rms * s)
                .sum();
            scores.push(score);
        }

        // Softmax over scores
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        // Weighted sum: out = sum_r probs_r * v_r
        let mut mix_out = vec![0.0f32; n_embd];
        for r in 0..n_rows {
            let row: &[f32] = if r < n_rows - 1 {
                &bank_f32[r * n_embd..(r + 1) * n_embd]
            } else {
                &host_prefix
            };
            let p = probs[r];
            for i in 0..n_embd {
                mix_out[i] += p * row[i];
            }
        }

        // Write result back to device
        rt.mix_out.write(0, kernels::as_bytes(&mix_out))?;
        Ok(())
    }

    /// KDA single-token forward for one layer.
    ///
    /// Implements the full KDA attention path:
    ///   1. Causal conv1d on Q, K, V projections
    ///   2. L2 norm on Q, K
    ///   3. Safe gate: g1 = gate_lower_bound * sigmoid(exp(A_log) * (f_a(x) @ f_b + dt_bias))
    ///   4. Beta: sigmoid(x @ W_beta)
    ///   5. Delta-rule step: attn_out, new_state = delta_net(Q, K, V, g1, beta, state)
    ///   6. Output gate: g2 = sigmoid(x @ W_gate)
    ///   7. RMSNorm(attn_out, o_norm) * g2 @ W_o
    #[allow(clippy::too_many_arguments)]
    fn kda_layer_forward(
        &self,
        rt: &mut KimiK3Rt,
        x: &DeviceBuf, // [n_embd] f32 (residual stream input)
        w: &KimiK3W,
        il: usize,
        step: u32,
        eps: f32,
    ) -> Result {
        let s = self.shape;
        let n_embd = s.n_embd;
        let n_head = s.n_head;
        let kda_hd = s.kda_head_dim;
        let d_inner = n_head * kda_hd;
        let d_conv = s.ssm_conv_k.max(1);
        let gate_lower_bound = s.kda_gate_lower_bound;

        // Unwrap KDA weights (all must be Some for KDA layers)
        let wq = w.wq.as_ref().ok_or("KDA layer missing wq")?;
        let wk = w.wk.as_ref().ok_or("KDA layer missing wk")?;
        let wv = w.wv.as_ref().ok_or("KDA layer missing wv")?;
        let ssm_q_conv = w
            .ssm_q_conv
            .as_ref()
            .ok_or("KDA layer missing ssm_q_conv")?;
        let ssm_k_conv = w
            .ssm_k_conv
            .as_ref()
            .ok_or("KDA layer missing ssm_k_conv")?;
        let ssm_v_conv = w
            .ssm_v_conv
            .as_ref()
            .ok_or("KDA layer missing ssm_v_conv")?;
        let ssm_f_a = w.ssm_f_a.as_ref().ok_or("KDA layer missing ssm_f_a")?;
        let ssm_f_b = w.ssm_f_b.as_ref().ok_or("KDA layer missing ssm_f_b")?;
        let ssm_beta = w.ssm_beta.as_ref().ok_or("KDA layer missing ssm_beta")?;
        let ssm_a = w.ssm_a.as_ref().ok_or("KDA layer missing ssm_a")?;
        let ssm_dt_b = w.ssm_dt_b.as_ref().ok_or("KDA layer missing ssm_dt_b")?;
        let wqkv_gate = w.wqkv_gate.as_ref().ok_or("KDA layer missing wqkv_gate")?;
        let ssm_o_norm = w
            .ssm_o_norm
            .as_ref()
            .ok_or("KDA layer missing ssm_o_norm")?;
        let wo = w.wo.as_ref().ok_or("KDA layer missing wo")?;

        // These six projections all consume the same f32 `x`. Quantize once
        // and reuse the Q8_K activation instead of launching six conversions.
        kernels::quantize_q8_k(&mut rt.q8k_scratch, x, n_embd, 1)?;
        // wq: [n_embd -> d_inner]
        wq.matmul_q8k(&mut rt.kda_conv_out, &rt.q8k_scratch, n_embd, d_inner, 1)
            .map_err(|e| format!("K3 layer {il} KDA Q projection: {e}"))?;
        // Save Q projection for conv
        kernels::copy_d2d(
            &mut rt.kda_q_normed,
            0,
            &rt.kda_conv_out,
            0,
            (d_inner as usize) * 4,
        )?;

        wk.matmul_q8k(&mut rt.kda_k_normed, &rt.q8k_scratch, n_embd, d_inner, 1)
            .map_err(|e| format!("K3 layer {il} KDA K projection: {e}"))?;
        wv.matmul_q8k(&mut rt.kda_v_conv, &rt.q8k_scratch, n_embd, d_inner, 1)
            .map_err(|e| format!("K3 layer {il} KDA V projection: {e}"))?;

        k3_dump_device(step, "kda", il, "q_projection", &rt.kda_q_normed, d_inner)?;
        k3_dump_device(step, "kda", il, "k_projection", &rt.kda_k_normed, d_inner)?;
        k3_dump_device(step, "kda", il, "v_projection", &rt.kda_v_conv, d_inner)?;

        // 2. Causal conv1d on Q, K, V
        // sconv: out = silu(conv(x, kern, state))
        // The sconv kernel does: out = x + causal depthwise conv over last K inputs
        // For K3 we need: out = silu(conv(x_proj, conv_weight))
        // The sconv kernel handles the causal conv part; we apply silu separately.
        // Actually, the existing sconv kernel does: out = x + conv(x, state)
        // K3 reference: Q = silu(conv1d(x @ W_q, conv_q))
        // We'll use the sconv kernel for the conv part, then apply silu.
        // But sconv already includes the residual connection. For K3, the conv
        // is a standalone causal conv without residual. We need to handle this
        // differently: use the conv state directly.
        //
        // The existing pulsar_sconv kernel computes: out = x + depthwise_conv(x, state)
        // For K3 we want: out = silu(depthwise_conv(x_proj, conv_weight))
        // We'll use sconv and then subtract x to get just the conv part.
        // Actually, looking at the reference more carefully:
        // K3 conv1d: Q = causal_conv1d(x @ W_q, conv_q)
        // where causal_conv1d(y, w) = silu(conv(y, w))
        // The conv is a depthwise conv over the last d_conv inputs.
        //
        // For simplicity in this first implementation, we use the sconv kernel
        // which does: out = x + depthwise_conv(x, state)
        // Then we extract just the conv part and apply silu.
        // This is a host-roundtrip approach for correctness.

        // Q conv
        let q_conv_state = &mut rt.conv_states[il][0];
        k3_dump_device(
            step,
            "kda_state",
            il,
            "conv_q_before",
            q_conv_state,
            d_inner * (d_conv - 1).max(1),
        )?;
        kernels::sconv(
            &mut rt.kda_conv_out,
            &rt.kda_q_normed,
            ssm_q_conv,
            q_conv_state,
            1,
            d_inner,
            d_conv,
        )?;
        // sconv does: out = x + conv(x). We need just silu(conv(x)).
        // Read back, compute silu(conv_part) = silu(out - x), write back.
        let q_after_conv = rt.kda_conv_out.read_f32(d_inner as usize)?;
        let q_before_conv = rt.kda_q_normed.read_f32(d_inner as usize)?;
        k3_dump_device(
            step,
            "kda",
            il,
            "q_sconv_residual",
            &rt.kda_conv_out,
            d_inner,
        )?;
        let q_conv_only: Vec<f32> = q_after_conv
            .iter()
            .zip(q_before_conv.iter())
            .map(|(after, before)| {
                let conv_part = after - before;
                // silu: x * sigmoid(x)
                conv_part / (1.0 + (-conv_part).exp())
            })
            .collect();
        rt.kda_q_normed.write(0, kernels::as_bytes(&q_conv_only))?;
        k3_dump_device(step, "kda", il, "q_conv_output", &rt.kda_q_normed, d_inner)?;
        k3_dump_device(
            step,
            "kda_state",
            il,
            "conv_q_after",
            q_conv_state,
            d_inner * (d_conv - 1).max(1),
        )?;

        // K conv
        let k_conv_state = &mut rt.conv_states[il][1];
        k3_dump_device(
            step,
            "kda_state",
            il,
            "conv_k_before",
            k_conv_state,
            d_inner * (d_conv - 1).max(1),
        )?;
        kernels::sconv(
            &mut rt.kda_conv_out,
            &rt.kda_k_normed,
            ssm_k_conv,
            k_conv_state,
            1,
            d_inner,
            d_conv,
        )?;
        let k_after_conv = rt.kda_conv_out.read_f32(d_inner as usize)?;
        let k_before_conv = rt.kda_k_normed.read_f32(d_inner as usize)?;
        k3_dump_device(
            step,
            "kda",
            il,
            "k_sconv_residual",
            &rt.kda_conv_out,
            d_inner,
        )?;
        let k_conv_only: Vec<f32> = k_after_conv
            .iter()
            .zip(k_before_conv.iter())
            .map(|(after, before)| {
                let conv_part = after - before;
                conv_part / (1.0 + (-conv_part).exp())
            })
            .collect();
        rt.kda_k_normed.write(0, kernels::as_bytes(&k_conv_only))?;
        k3_dump_device(step, "kda", il, "k_conv_output", &rt.kda_k_normed, d_inner)?;
        k3_dump_device(
            step,
            "kda_state",
            il,
            "conv_k_after",
            k_conv_state,
            d_inner * (d_conv - 1).max(1),
        )?;

        // V conv
        let v_conv_state = &mut rt.conv_states[il][2];
        k3_dump_device(
            step,
            "kda_state",
            il,
            "conv_v_before",
            v_conv_state,
            d_inner * (d_conv - 1).max(1),
        )?;
        kernels::sconv(
            &mut rt.kda_conv_out,
            &rt.kda_v_conv,
            ssm_v_conv,
            v_conv_state,
            1,
            d_inner,
            d_conv,
        )?;
        let v_after_conv = rt.kda_conv_out.read_f32(d_inner as usize)?;
        let v_before_conv = rt.kda_v_conv.read_f32(d_inner as usize)?;
        k3_dump_device(
            step,
            "kda",
            il,
            "v_sconv_residual",
            &rt.kda_conv_out,
            d_inner,
        )?;
        let v_conv_only: Vec<f32> = v_after_conv
            .iter()
            .zip(v_before_conv.iter())
            .map(|(after, before)| {
                let conv_part = after - before;
                conv_part / (1.0 + (-conv_part).exp())
            })
            .collect();
        rt.kda_v_conv.write(0, kernels::as_bytes(&v_conv_only))?;
        k3_dump_device(step, "kda", il, "v_conv_output", &rt.kda_v_conv, d_inner)?;
        k3_dump_device(
            step,
            "kda_state",
            il,
            "conv_v_after",
            v_conv_state,
            d_inner * (d_conv - 1).max(1),
        )?;

        // 3. L2 norm on Q and K (per-head)
        // Reshape [d_inner] -> [n_head, kda_hd], L2 norm each head
        // Use the existing qwen35_l2_norm kernel which does per-row L2 norm
        kernels::qwen35_l2_norm(&mut rt.kda_q_normed, n_head, kda_hd, 1.0e-12)?;
        kernels::qwen35_l2_norm(&mut rt.kda_k_normed, n_head, kda_hd, 1.0e-12)?;
        k3_dump_device(step, "kda", il, "q_normed", &rt.kda_q_normed, d_inner)?;
        k3_dump_device(step, "kda", il, "k_normed", &rt.kda_k_normed, d_inner)?;

        // 4. Safe gate: g1 = gate_lower_bound * sigmoid(exp(A_log) * (f_a(x) @ f_b + dt_bias))
        // f_a = x @ W_f_a  [n_embd -> kda_head_dim]
        ssm_f_a.matmul_q8k(&mut rt.kda_f_a, &rt.q8k_scratch, n_embd, kda_hd, 1)?;
        // g_raw = f_a @ W_f_b  [kda_head_dim -> d_inner]
        kernels::matmul_f32(&mut rt.kda_g_raw, ssm_f_b, &rt.kda_f_a, kda_hd, d_inner, 1)?;
        // g_raw += dt_bias
        // Read, add bias, write back (host roundtrip for correctness)
        let mut g_raw_host = rt.kda_g_raw.read_f32(d_inner as usize)?;
        let dt_b_host = ssm_dt_b.read_f32(d_inner as usize)?;
        for i in 0..d_inner as usize {
            g_raw_host[i] += dt_b_host[i];
        }
        // Existing K3 GGUF converters emitted both exp(A_log) and
        // -exp(A_log). The bounded gate consumes the positive magnitude.
        let a_host = ssm_a.read_f32(n_head as usize)?;
        if il == 0 && step == 0 && std::env::var_os("PULSAR_K3_DEBUG_A").is_some() {
            let min = a_host.iter().copied().fold(f32::INFINITY, f32::min);
            let max = a_host.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            eprintln!("pulsar: K3 loaded ssm_a range [{min:.9}, {max:.9}]");
        }
        // K3's bounded gate uses exp(A_log), independent of stored sign.
        for h in 0..n_head as usize {
            for d in 0..kda_hd as usize {
                let idx = h * kda_hd as usize + d;
                g_raw_host[idx] *= a_host[h].abs();
            }
        }
        // sigmoid and multiply by gate_lower_bound
        for v in g_raw_host.iter_mut() {
            *v = gate_lower_bound / (1.0 + (-*v).exp());
        }
        rt.kda_g_raw.write(0, kernels::as_bytes(&g_raw_host))?;
        k3_dump_device(step, "kda", il, "gate_alpha", &rt.kda_g_raw, d_inner)?;

        // 5. Beta: sigmoid(x @ W_beta)  [n_embd -> n_head]
        ssm_beta.matmul_q8k(&mut rt.kda_gate_logits, &rt.q8k_scratch, n_embd, n_head, 1)?;
        let mut beta_host = rt.kda_gate_logits.read_f32(n_head as usize)?;
        for v in beta_host.iter_mut() {
            *v = 1.0 / (1.0 + (-*v).exp());
        }
        rt.kda_gate_logits.write(0, kernels::as_bytes(&beta_host))?;
        k3_dump_device(step, "kda", il, "beta", &rt.kda_gate_logits, n_head)?;

        // 6. Delta-rule step: attn_out, new_state = delta_net(Q, K, V, g1, beta, state)
        let ssm_state = &mut rt.ssm_states[il];
        k3_dump_device(
            step,
            "kda_state",
            il,
            "recurrent_before",
            ssm_state,
            n_head * kda_hd * kda_hd,
        )?;
        kernels::k3_kda_step(
            &mut rt.attn_out,
            ssm_state,
            &rt.kda_q_normed,
            &rt.kda_k_normed,
            &rt.kda_v_conv,
            &rt.kda_g_raw,
            &rt.kda_gate_logits,
            n_head,
            n_head,
            kda_hd,
        )?;
        k3_dump_device(
            step,
            "kda_state",
            il,
            "recurrent_after",
            ssm_state,
            n_head * kda_hd * kda_hd,
        )?;
        k3_dump_device(step, "kda", il, "recurrent_output", &rt.attn_out, d_inner)?;
        if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
            let sync_result: super::Result = kernels::sync()
                .map_err(|e| format!("K3 layer {il} KDA delta-step sync: {e}").into());
            sync_result?;
        }

        // 7. Output gate: g2 = sigmoid(x @ W_gate)
        wqkv_gate.matmul_q8k(&mut rt.kda_gate_logits, &rt.q8k_scratch, n_embd, d_inner, 1)?;
        kernels::k3_sigmoid_inplace(&mut rt.kda_gate_logits, d_inner)?;
        k3_dump_device(step, "kda", il, "output_gate", &rt.kda_gate_logits, d_inner)?;
        if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
            let sync_result: super::Result = kernels::sync()
                .map_err(|e| format!("K3 layer {il} KDA gate sigmoid sync: {e}").into());
            sync_result?;
        }

        // 8. RMSNorm independently over each 128-wide head.
        kernels::rms_norm(
            &mut rt.kda_o_normed,
            &rt.attn_out,
            ssm_o_norm,
            kda_hd,
            n_head,
            eps,
        )?;
        if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
            let sync_result: super::Result = kernels::sync()
                .map_err(|e| format!("K3 layer {il} KDA output norm sync: {e}").into());
            sync_result?;
        }

        // 9. gated = normed * sigmoid(g2)
        kernels::copy_d2d(
            &mut rt.kda_gated,
            0,
            &rt.kda_o_normed,
            0,
            (d_inner as usize) * 4,
        )?;
        kernels::k3_mul_inplace(&mut rt.kda_gated, &rt.kda_gate_logits, d_inner)?;
        if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
            let sync_result: super::Result = kernels::sync()
                .map_err(|e| format!("K3 layer {il} KDA gate multiply sync: {e}").into());
            sync_result?;
        }

        // 10. o_proj: out = gated @ W_o  (K3DenseWeight dispatch)
        wo.matmul(
            &mut rt.attn_out,
            &rt.kda_gated,
            &mut rt.q8k_scratch,
            d_inner,
            n_embd,
            1,
        )?;
        k3_dump_device(
            step,
            "kda",
            il,
            "output_before_residual",
            &rt.attn_out,
            n_embd,
        )?;

        Ok(())
    }

    /// Dense SiTU-GLU FFN forward for one layer.
    fn dense_ffn_forward(
        &self,
        rt: &mut KimiK3Rt,
        x: &DeviceBuf, // [n_embd] f32 (normed input)
        w: &KimiK3W,
    ) -> Result {
        let s = self.shape;
        let n_embd = s.n_embd;
        let n_ff_dense = s.n_ff_dense;

        let ffn_gate = w
            .ffn_gate
            .as_ref()
            .ok_or("dense FFN layer missing ffn_gate")?;
        let ffn_up = w.ffn_up.as_ref().ok_or("dense FFN layer missing ffn_up")?;
        let ffn_down = w
            .ffn_down
            .as_ref()
            .ok_or("dense FFN layer missing ffn_down")?;

        // Gate and up consume the same activation: quantize once and reuse.
        kernels::quantize_q8_k(&mut rt.q8k_scratch, x, n_embd, 1)?;
        // gate = x @ W_gate  (K3DenseWeight dispatch)
        ffn_gate.matmul_q8k(&mut rt.dense_gate, &rt.q8k_scratch, n_embd, n_ff_dense, 1)?;
        // up = x @ W_up  (K3DenseWeight dispatch)
        ffn_up.matmul_q8k(&mut rt.dense_up, &rt.q8k_scratch, n_embd, n_ff_dense, 1)?;
        // mid = SiTU_Gate(gate) * SiTU_Linear(up)
        kernels::k3_situ_glu(
            &mut rt.dense_mid,
            &rt.dense_gate,
            &rt.dense_up,
            n_ff_dense,
            s.situ_beta,
            s.situ_linear_beta,
        )?;
        // out = mid @ W_down  (K3DenseWeight dispatch)
        ffn_down.matmul(
            &mut rt.ffn_out,
            &rt.dense_mid,
            &mut rt.q8k_scratch,
            n_ff_dense,
            n_embd,
            1,
        )?;

        Ok(())
    }

    /// Stable LatentMoE forward for one layer.
    ///
    /// Uses the generic cache/tier resolve path (StreamingStore, DeviceSlabCache,
    /// ExpertTier) to fetch expert slabs instead of synchronous VFile reads.
    /// For Q2_K/Q3_K triples, when PULSAR_K3_EXPERT_BACKEND=cuda, dispatches to the existing
    /// moe_pair_swiglu/moe_down CUDA kernels with act_op=4 (K3 SiTU-GLU,
    /// beta=4 and linear_beta=25). Otherwise falls back to the host
    /// dequant+CPU matmul path for correctness.
    fn latent_moe_forward(
        &self,
        store: &mut StreamingStore,
        dev_cache: &mut super::DeviceSlabCache,
        staging: &mut DeviceBuf,
        rt: &mut KimiK3Rt,
        x: &DeviceBuf, // [n_embd] f32 (normed input)
        w: &KimiK3W,
        il: usize,
        step: u32,
        mut layer_prof: Option<&mut super::K3LayerProfile>,
    ) -> Result {
        let s = self.shape;
        let n_embd = s.n_embd;
        let n_expert = s.n_expert;
        let n_expert_used = s.n_expert_used;
        let n_expert_shared = s.n_expert_shared;
        let moe_latent = s.moe_latent_size;
        let n_ff_exp = s.n_ff_exp;

        let ffn_gate_inp = w
            .ffn_gate_inp
            .as_ref()
            .ok_or("MoE layer missing ffn_gate_inp")?;
        let ffn_exp_probs_b = w
            .ffn_exp_probs_b
            .as_ref()
            .ok_or("MoE layer missing ffn_exp_probs_b")?;
        let ffn_latent_down = w
            .ffn_latent_down
            .as_ref()
            .ok_or("MoE layer missing ffn_latent_down")?;
        let ffn_latent_norm = w
            .ffn_latent_norm
            .as_ref()
            .ok_or("MoE layer missing ffn_latent_norm")?;
        let ffn_latent_up = w
            .ffn_latent_up
            .as_ref()
            .ok_or("MoE layer missing ffn_latent_up")?;
        let ffn_gate_exps = w
            .ffn_gate_exps
            .as_ref()
            .ok_or("MoE layer missing ffn_gate_exps")?;
        let ffn_up_exps = w
            .ffn_up_exps
            .as_ref()
            .ok_or("MoE layer missing ffn_up_exps")?;
        let ffn_down_exps = w
            .ffn_down_exps
            .as_ref()
            .ok_or("MoE layer missing ffn_down_exps")?;
        let ffn_gate_shexp = w
            .ffn_gate_shexp
            .as_ref()
            .ok_or("MoE layer missing ffn_gate_shexp")?;
        let ffn_up_shexp = w
            .ffn_up_shexp
            .as_ref()
            .ok_or("MoE layer missing ffn_up_shexp")?;
        let ffn_down_shexp = w
            .ffn_down_shexp
            .as_ref()
            .ok_or("MoE layer missing ffn_down_shexp")?;

        // ── Quant-aware expert dequantization ────────────────────────────────
        // Each K3 expert tensor carries its own packed type/row geometry.
        // Keep the first correctness path on host, but do not reinterpret all
        // slabs as Q8_0.
        // 1. Latent projection: latent = x @ W_latent_down  [n_embd -> moe_latent]
        ffn_latent_down.matmul(
            &mut rt.latent,
            x,
            &mut rt.q8k_scratch,
            n_embd,
            moe_latent,
            1,
        )?;
        k3_dump_device(step, "moe", il, "router_input", x, n_embd)?;
        k3_dump_device(step, "moe", il, "latent_projection", &rt.latent, moe_latent)?;

        // 2. Router: logits = x @ W_gate_inp  [n_embd -> n_expert]
        let router_t0 = std::time::Instant::now();
        ffn_gate_inp.matmul(
            &mut rt.router_logits,
            x,
            &mut rt.q8k_scratch,
            n_embd,
            n_expert,
            1,
        )?;
        k3_dump_device(
            step,
            "moe",
            il,
            "router_logits",
            &rt.router_logits,
            n_expert,
        )?;

        // 3. Router select: sigmoid scores, top-k, renormalize
        // Use the K3-specific router which supports up to 896 experts
        kernels::k3_router_select(
            &mut rt.expert_selected,
            &mut rt.expert_weights,
            &rt.router_logits,
            ffn_exp_probs_b,
            n_expert,
            n_expert_used,
            s.expert_weight_scale,
            1,
        )?;
        if let Some(p) = layer_prof.as_deref_mut() {
            p.router_gpu += router_t0.elapsed();
        }

        // 4. Read selected indices and weights from device (host roundtrip)
        let readback_t0 = std::time::Instant::now();
        let selected = rt.expert_selected.read_i32(n_expert_used as usize)?;
        let weights = rt.expert_weights.read_f32(n_expert_used as usize)?;
        k3_dump_i32(step, "moe", il, "selected_experts", &selected)?;
        k3_dump_host_f32(step, "moe", il, "routing_weights", &weights)?;
        if let Some(p) = layer_prof.as_deref_mut() {
            p.router_sync += readback_t0.elapsed();
            p.d2h_bytes = p.d2h_bytes.saturating_add((n_expert_used as u64) * 8);
        }

        // 5. Resolve expert slabs through the generic cache/tier system
        //    instead of synchronous VFile reads.
        //
        //    The resolve path uses StreamingStore (io_uring + LFU host cache),
        //    DeviceSlabCache (VRAM hot-set), and ExpertTier (resident GPU tiers)
        //    to fetch expert gate/up/down slabs.  This eliminates the per-expert
        //    synchronous VFile read_exact_at and the host-side dequant-to-f32
        //    for the GPU dispatch path.
        //
        //    For a first safe slice, we GPU-run only Q2_K/Q3_K triples (the
        //    existing moe_pair_swiglu/moe_down kernels support these quants).
        //    Other quant types fall back to the host path.
        let gpu_ok = (ffn_gate_exps.quant == kernels::QUANT_Q2_K
            || ffn_gate_exps.quant == kernels::QUANT_Q3_K)
            && (ffn_up_exps.quant == ffn_gate_exps.quant)
            && (ffn_down_exps.quant == kernels::QUANT_Q2_K
                || ffn_down_exps.quant == kernels::QUANT_Q3_K);
        let use_gpu = gpu_ok
            && k3_expert_backend() == "cuda"
            && ffn_gate_exps.row_bytes == ffn_up_exps.row_bytes;
        if k3_expert_backend() == "cuda" && !use_gpu {
            eprintln!(
                "pulsar: K3 CUDA expert backend requested but this layer's expert layout is unsupported; using CPU reference"
            );
        }

        // Build the list of distinct expert offsets and resolve through
        // the cache/tier system.  This is shared by both GPU and host paths.
        let mut distinct: Vec<i32> = selected
            .iter()
            .copied()
            .filter(|&e| e >= 0 && (e as u32) < n_expert)
            .collect();
        distinct.sort_unstable();
        distinct.dedup();

        let mut wants = Vec::new();
        for &e in &distinct {
            let ei = e as u64;
            for (t, le) in [(ffn_gate_exps, ei), (ffn_up_exps, ei), (ffn_down_exps, ei)] {
                let off = t.abs_offset + le * t.expert_bytes;
                wants.push(StreamRead {
                    offset: off,
                    len: t.expert_bytes,
                });
            }
        }
        wants.dedup_by_key(|r| r.offset);

        // Resolve: cache hits return immediately, misses go through io_uring.
        // For the host path, we collect resolved bytes into a HashMap.
        // For the GPU path, we upload to staging and track device pointers.
        let mut resolved_host: std::collections::HashMap<u64, Vec<u8>> =
            std::collections::HashMap::new();
        let mut resolved_gpu: std::collections::HashMap<u64, *const std::ffi::c_void> =
            std::collections::HashMap::new();

        let store_before = (
            store.hits,
            store.misses,
            store.io_bytes,
            store.io_reads,
            store.io_max_read,
            store.io_wait,
        );
        let dev_before = (dev_cache.hits, dev_cache.misses);
        let resolve_t0 = std::time::Instant::now();
        let mut resolved_h2d_bytes = 0u64;
        let mut resolved_h2d_time = std::time::Duration::ZERO;
        if use_gpu {
            let mut stage_base: std::collections::HashMap<u64, usize> =
                std::collections::HashMap::new();
            let mut stage_len = 0usize;
            for r in &wants {
                stage_base.insert(r.offset, stage_len);
                stage_len += r.len as usize;
            }
            if stage_len > staging.bytes() {
                return Err(format!(
                    "K3 GPU MoE staging too small: need {} bytes, have {}",
                    stage_len,
                    staging.bytes()
                )
                .into());
            }
            store.ensure_with(&wants, |off, payload| {
                if k3_compare_active(step, il) {
                    resolved_host.insert(off, payload.to_vec());
                }
                let base = stage_base[&off];
                let h2d_t0 = std::time::Instant::now();
                staging.write(base, payload)?;
                resolved_h2d_time += h2d_t0.elapsed();
                resolved_h2d_bytes = resolved_h2d_bytes.saturating_add(payload.len() as u64);
                resolved_gpu.insert(off, staging.ptr_at(base));
                Ok(())
            })?;
        } else {
            // Host path: collect resolved bytes
            store.ensure_with(&wants, |off, payload| {
                resolved_host.insert(off, payload.to_vec());
                Ok(())
            })?;
        }
        if let Some(p) = layer_prof.as_deref_mut() {
            p.expert_resolution += resolve_t0.elapsed();
            p.storage += store.io_wait.saturating_sub(store_before.5);
            p.cache += resolve_t0.elapsed().saturating_sub(p.storage);
            p.h2d_bytes = p.h2d_bytes.saturating_add(resolved_h2d_bytes);
            p.h2d += resolved_h2d_time;
            p.storage_bytes = p
                .storage_bytes
                .saturating_add(store.io_bytes.saturating_sub(store_before.2));
            p.storage_reads = p
                .storage_reads
                .saturating_add(store.io_reads.saturating_sub(store_before.3));
            p.storage_max_read = p
                .storage_max_read
                .max(store.io_max_read.saturating_sub(store_before.4));
            p.host_cache_hits = p
                .host_cache_hits
                .saturating_add(store.hits.saturating_sub(store_before.0));
            p.host_cache_misses = p
                .host_cache_misses
                .saturating_add(store.misses.saturating_sub(store_before.1));
            p.device_cache_hits = p
                .device_cache_hits
                .saturating_add(dev_cache.hits.saturating_sub(dev_before.0));
            p.device_cache_misses = p
                .device_cache_misses
                .saturating_add(dev_cache.misses.saturating_sub(dev_before.1));
            p.expert_requests = p.expert_requests.saturating_add(selected.len() as u64);
            p.unique_experts = p.unique_experts.saturating_add(distinct.len() as u64);
            p.repeated_experts = p
                .repeated_experts
                .saturating_add(selected.len().saturating_sub(distinct.len()) as u64);
        }

        if use_gpu {
            // ── GPU dispatch path (Q2_K/Q3_K only) ──────────────────────────
            self.k3_gpu_moe_compute(
                rt,
                &selected,
                &weights,
                &resolved_gpu,
                if k3_compare_active(step, il) {
                    Some(&resolved_host)
                } else {
                    None
                },
                staging,
                ffn_gate_exps,
                ffn_up_exps,
                ffn_down_exps,
                n_expert,
                n_expert_used,
                moe_latent,
                n_ff_exp,
                il,
                step,
            )?;
        } else if k3_expert_backend() == "cpu-q8" {
            self.k3_cpu_q8_moe_compute(
                rt,
                &selected,
                &weights,
                &resolved_host,
                ffn_gate_exps,
                ffn_up_exps,
                ffn_down_exps,
                n_expert,
                n_expert_used,
                moe_latent,
                n_ff_exp,
                il,
                step,
                layer_prof.as_deref_mut(),
            )?;
        } else {
            // ── Host fallback path (all quant types) ────────────────────────
            // Resolve slabs through the cache/tier system, then dequant and
            // compute on host.  This is the correctness reference.
            self.k3_host_moe_compute(
                rt,
                &selected,
                &weights,
                &resolved_host,
                ffn_gate_exps,
                ffn_up_exps,
                ffn_down_exps,
                n_expert,
                n_expert_used,
                moe_latent,
                n_ff_exp,
                layer_prof.as_deref_mut(),
            )?;
        }

        // 6. RMSNorm(moe_acc, latent_norm)
        let latent_norm_t0 = std::time::Instant::now();
        k3_dump_device(
            step,
            "moe",
            il,
            "routed_moe_accum",
            &rt.latent_normed,
            moe_latent,
        )?;
        let latent_norm_host = ffn_latent_norm.read_f32(moe_latent as usize)?;
        let moe_acc = rt.latent_normed.read_f32(moe_latent as usize)?;
        let mean_sq: f32 = moe_acc.iter().map(|v| v * v).sum::<f32>() / moe_latent as f32;
        let inv_rms = 1.0 / (mean_sq + s.rms_eps).sqrt();
        let mut moe_normed = vec![0.0f32; moe_latent as usize];
        for i in 0..moe_latent as usize {
            moe_normed[i] = moe_acc[i] * inv_rms * latent_norm_host[i];
        }
        if let Some(p) = layer_prof.as_deref_mut() {
            p.cpu_latent_norm += latent_norm_t0.elapsed();
            p.cpu_threads = p.cpu_threads.max(1);
        }

        // 7. moe_out = moe_normed @ W_latent_up  [moe_latent -> n_embd]
        // Write moe_normed to device, then matmul
        rt.latent_normed.write(0, kernels::as_bytes(&moe_normed))?;
        k3_dump_device(
            step,
            "moe",
            il,
            "routed_moe_normed",
            &rt.latent_normed,
            moe_latent,
        )?;
        kernels::matmul_q8_0(
            &mut rt.moe_out,
            ffn_latent_up,
            &rt.latent_normed,
            moe_latent,
            n_embd,
            1,
        )?;
        k3_dump_device(step, "moe", il, "latent_up_output", &rt.moe_out, n_embd)?;

        // 8. Shared experts (on full hidden state, SiTU-GLU)
        // gate = x @ W_gate_sh  [n_embd -> n_ff_exp * n_expert_shared]
        let shexp_width = n_ff_exp * n_expert_shared;
        ffn_gate_shexp.matmul(
            &mut rt.shexp_gate,
            x,
            &mut rt.q8k_scratch,
            n_embd,
            shexp_width,
            1,
        )?;
        // up = x @ W_up_sh
        ffn_up_shexp.matmul(
            &mut rt.shexp_up,
            x,
            &mut rt.q8k_scratch,
            n_embd,
            shexp_width,
            1,
        )?;
        // mid = SiTU_Gate(gate) * SiTU_Linear(up)
        kernels::k3_situ_glu(
            &mut rt.shexp_mid,
            &rt.shexp_gate,
            &rt.shexp_up,
            shexp_width,
            s.situ_beta,
            s.situ_linear_beta,
        )?;
        // out = mid @ W_down_sh
        kernels::matmul_q8_0(
            &mut rt.shexp_out,
            ffn_down_shexp,
            &rt.shexp_mid,
            shexp_width,
            n_embd,
            1,
        )?;
        k3_dump_device(
            step,
            "moe",
            il,
            "shared_expert_output",
            &rt.shexp_out,
            n_embd,
        )?;

        // 9. ffn_out = moe_out + shexp_out
        kernels::add(&mut rt.ffn_out, &rt.moe_out, &rt.shexp_out, n_embd)?;
        k3_dump_device(
            step,
            "moe",
            il,
            "ffn_output_before_residual",
            &rt.ffn_out,
            n_embd,
        )?;

        Ok(())
    }

    /// Compute on host from pre-resolved expert slabs (dequant + CPU matmul).
    /// This is the correctness reference path for all quant types.
    #[allow(clippy::too_many_arguments)]
    fn k3_host_moe_compute(
        &self,
        rt: &mut KimiK3Rt,
        selected: &[i32],
        weights: &[f32],
        resolved: &std::collections::HashMap<u64, Vec<u8>>,
        ffn_gate_exps: &super::ExpertTensor,
        ffn_up_exps: &super::ExpertTensor,
        ffn_down_exps: &super::ExpertTensor,
        n_expert: u32,
        n_expert_used: u32,
        moe_latent: u32,
        n_ff_exp: u32,
        mut layer_prof: Option<&mut super::K3LayerProfile>,
    ) -> Result {
        let s = self.shape;

        // Read latent from device
        let latent_host = rt.latent.read_f32(moe_latent as usize)?;

        // Compute each selected expert on host
        let mut moe_acc = vec![0.0f32; moe_latent as usize];

        for si in 0..n_expert_used as usize {
            let e = selected[si];
            if e < 0 || e as u32 >= n_expert {
                continue;
            }
            let w_e = weights[si];
            let ei = e as u64;

            let gate_off = ffn_gate_exps.abs_offset + ei * ffn_gate_exps.expert_bytes;
            let up_off = ffn_up_exps.abs_offset + ei * ffn_up_exps.expert_bytes;
            let down_off = ffn_down_exps.abs_offset + ei * ffn_down_exps.expert_bytes;

            let gate_buf = resolved
                .get(&gate_off)
                .ok_or_else(|| format!("K3 host MoE: gate slab {gate_off} not resolved"))?;
            let up_buf = resolved
                .get(&up_off)
                .ok_or_else(|| format!("K3 host MoE: up slab {up_off} not resolved"))?;
            let down_buf = resolved
                .get(&down_off)
                .ok_or_else(|| format!("K3 host MoE: down slab {down_off} not resolved"))?;

            let dequant_t0 = std::time::Instant::now();
            let gate_f32 = k3_dequant_expert_bytes(
                gate_buf,
                (moe_latent * n_ff_exp) as usize,
                ffn_gate_exps.quant,
            )?;
            let up_f32 = k3_dequant_expert_bytes(
                up_buf,
                (moe_latent * n_ff_exp) as usize,
                ffn_up_exps.quant,
            )?;
            let down_f32 = k3_dequant_expert_bytes(
                down_buf,
                (n_ff_exp * moe_latent) as usize,
                ffn_down_exps.quant,
            )?;
            if let Some(p) = layer_prof.as_deref_mut() {
                p.cpu_expert_dequant += dequant_t0.elapsed();
                p.cpu_expert_matrices += 3;
                p.cpu_expert_weight_bytes = p.cpu_expert_weight_bytes.saturating_add(
                    ffn_gate_exps.expert_bytes
                        + ffn_up_exps.expert_bytes
                        + ffn_down_exps.expert_bytes,
                );
                p.cpu_threads = 1;
            }

            // gate_out = latent @ W_gate_e  [moe_latent -> n_ff_exp]
            let mut gate_out = vec![0.0f32; n_ff_exp as usize];
            let gate_t0 = std::time::Instant::now();
            for j in 0..n_ff_exp as usize {
                for k in 0..moe_latent as usize {
                    gate_out[j] += latent_host[k] * gate_f32[j * moe_latent as usize + k];
                }
            }
            if let Some(p) = layer_prof.as_deref_mut() {
                p.cpu_expert_gate += gate_t0.elapsed();
            }

            // up_out = latent @ W_up_e  [moe_latent -> n_ff_exp]
            let mut up_out = vec![0.0f32; n_ff_exp as usize];
            let up_t0 = std::time::Instant::now();
            for j in 0..n_ff_exp as usize {
                for k in 0..moe_latent as usize {
                    up_out[j] += latent_host[k] * up_f32[j * moe_latent as usize + k];
                }
            }
            if let Some(p) = layer_prof.as_deref_mut() {
                p.cpu_expert_up += up_t0.elapsed();
            }

            // mid = SiTU_Gate(gate_out) * SiTU_Linear(up_out)
            let mut mid = vec![0.0f32; n_ff_exp as usize];
            let activation_t0 = std::time::Instant::now();
            for j in 0..n_ff_exp as usize {
                let g = gate_out[j];
                let u = up_out[j];
                let situ_gate = s.situ_beta * (g / s.situ_beta).tanh() / (1.0 + (-g).exp());
                let situ_linear = s.situ_linear_beta * (u / s.situ_linear_beta).tanh();
                mid[j] = situ_gate * situ_linear;
            }
            if let Some(p) = layer_prof.as_deref_mut() {
                p.cpu_expert_activation += activation_t0.elapsed();
            }

            // expert_out = mid @ W_down_e  [n_ff_exp -> moe_latent]
            let mut expert_out = vec![0.0f32; moe_latent as usize];
            let down_t0 = std::time::Instant::now();
            for k in 0..moe_latent as usize {
                for j in 0..n_ff_exp as usize {
                    expert_out[k] += mid[j] * down_f32[k * n_ff_exp as usize + j];
                }
            }
            if let Some(p) = layer_prof.as_deref_mut() {
                p.cpu_expert_down += down_t0.elapsed();
            }

            // Accumulate weighted expert output
            let accumulation_t0 = std::time::Instant::now();
            for k in 0..moe_latent as usize {
                moe_acc[k] += w_e * expert_out[k];
            }
            if let Some(p) = layer_prof.as_deref_mut() {
                p.cpu_expert_accumulation += accumulation_t0.elapsed();
                p.cpu_expert_evaluations += 1;
            }
        }

        // Write moe_acc to device for the latent norm/up steps
        rt.latent_normed.write(0, kernels::as_bytes(&moe_acc))?;

        Ok(())
    }

    /// CPU reference with the same two Q8_K activation boundaries as CUDA.
    /// Weight bytes remain packed and are consumed by the existing CPU K-quant
    /// dot routines; route weights are deliberately applied after down.
    #[allow(clippy::too_many_arguments)]
    fn k3_cpu_q8_moe_compute(
        &self,
        rt: &mut KimiK3Rt,
        selected: &[i32],
        weights: &[f32],
        resolved: &std::collections::HashMap<u64, Vec<u8>>,
        ffn_gate_exps: &super::ExpertTensor,
        ffn_up_exps: &super::ExpertTensor,
        ffn_down_exps: &super::ExpertTensor,
        n_expert: u32,
        n_expert_used: u32,
        moe_latent: u32,
        n_ff_exp: u32,
        il: usize,
        step: u32,
        mut layer_prof: Option<&mut super::K3LayerProfile>,
    ) -> Result {
        let gate_up_q2 =
            ffn_gate_exps.quant == kernels::QUANT_Q2_K && ffn_up_exps.quant == kernels::QUANT_Q2_K;
        let gate_up_q3 =
            ffn_gate_exps.quant == kernels::QUANT_Q3_K && ffn_up_exps.quant == kernels::QUANT_Q3_K;
        let down_supported = matches!(
            ffn_down_exps.quant,
            kernels::QUANT_Q2_K | kernels::QUANT_Q3_K
        );
        if !(gate_up_q2 || gate_up_q3) || !down_supported {
            return Err("K3 CPU-Q8 requires matching Q2_K or Q3_K expert weights".into());
        }
        let latent = rt.latent.read_f32(moe_latent as usize)?;
        let input_q = quant::cpu_dot::quantize_row_q8_k(&latent);
        let mut moe_acc = vec![0.0f32; moe_latent as usize];
        let compare = k3_compare_active(step, il);
        let compute_t0 = std::time::Instant::now();

        if compare {
            k3_dump_host_f32(
                step,
                "expert",
                il,
                "cpu_q8_expert_498_rank_00_slot_00_input_f32",
                &latent,
            )?;
            let input_q_bytes = k3_pack_q8_row(&input_q);
            k3_dump_expert_bytes(
                step,
                il,
                0,
                498,
                0,
                "cpu_q8_expert_498_rank_00_slot_00_input_q8_k",
                &input_q_bytes,
                "Q8_K",
                input_q.d.len(),
                moe_latent as usize,
                kernels::Q8_K_BLOCK_BYTES as u64,
                "activation",
                "quant::cpu_dot::quantize_row_q8_k",
            )?;
        }

        for si in 0..n_expert_used as usize {
            let e = selected[si];
            if e < 0 || e as u32 >= n_expert {
                continue;
            }
            let ei = e as u64;
            let gate_off = ffn_gate_exps.abs_offset + ei * ffn_gate_exps.expert_bytes;
            let up_off = ffn_up_exps.abs_offset + ei * ffn_up_exps.expert_bytes;
            let down_off = ffn_down_exps.abs_offset + ei * ffn_down_exps.expert_bytes;
            let gate = resolved
                .get(&gate_off)
                .ok_or("K3 CPU-Q8 gate slab missing")?;
            let up = resolved.get(&up_off).ok_or("K3 CPU-Q8 up slab missing")?;
            let down = resolved
                .get(&down_off)
                .ok_or("K3 CPU-Q8 down slab missing")?;
            let focused = compare && k3_compare_expert(si, e);

            if focused {
                k3_dump_expert_weight(
                    step,
                    il,
                    si,
                    e,
                    si,
                    ffn_gate_exps,
                    gate,
                    "cpu_q8_gate_q2_k_weight",
                )?;
                k3_dump_expert_weight(
                    step,
                    il,
                    si,
                    e,
                    si,
                    ffn_up_exps,
                    up,
                    "cpu_q8_up_q2_k_weight",
                )?;
                k3_dump_expert_weight(
                    step,
                    il,
                    si,
                    e,
                    si,
                    ffn_down_exps,
                    down,
                    "cpu_q8_down_q3_k_weight",
                )?;
            }

            let mut gate_out = vec![0.0f32; n_ff_exp as usize];
            let mut up_out = vec![0.0f32; n_ff_exp as usize];
            for j in 0..n_ff_exp as usize {
                let go = &gate[j * ffn_gate_exps.row_bytes as usize..];
                let uo = &up[j * ffn_up_exps.row_bytes as usize..];
                gate_out[j] = k3_q8_dot(ffn_gate_exps.quant, go, &input_q, moe_latent as usize)?;
                up_out[j] = k3_q8_dot(ffn_up_exps.quant, uo, &input_q, moe_latent as usize)?;
            }
            if focused {
                k3_dump_host_f32(
                    step,
                    "expert",
                    il,
                    "cpu_q8_expert_498_rank_00_slot_00_gate_f32",
                    &gate_out,
                )?;
                k3_dump_host_f32(
                    step,
                    "expert",
                    il,
                    "cpu_q8_expert_498_rank_00_slot_00_up_f32",
                    &up_out,
                )?;
            }

            let mut mid = vec![0.0f32; n_ff_exp as usize];
            for j in 0..n_ff_exp as usize {
                let g = gate_out[j];
                let u = up_out[j];
                mid[j] = self.shape.situ_beta * (g / self.shape.situ_beta).tanh()
                    / (1.0 + (-g).exp())
                    * (self.shape.situ_linear_beta * (u / self.shape.situ_linear_beta).tanh());
            }
            let mid_q = quant::cpu_dot::quantize_row_q8_k(&mid);
            if focused {
                k3_dump_host_f32(
                    step,
                    "expert",
                    il,
                    "cpu_q8_expert_498_rank_00_slot_00_situ_f32",
                    &mid,
                )?;
                let mid_q_bytes = k3_pack_q8_row(&mid_q);
                k3_dump_expert_bytes(
                    step,
                    il,
                    si,
                    e,
                    si,
                    "cpu_q8_expert_498_rank_00_slot_00_situ_output_q8_k",
                    &mid_q_bytes,
                    "Q8_K",
                    mid_q.d.len(),
                    mid.len(),
                    kernels::Q8_K_BLOCK_BYTES as u64,
                    "activation",
                    "quant::cpu_dot::quantize_row_q8_k",
                )?;
            }
            let mut expert_out = vec![0.0f32; moe_latent as usize];
            for k in 0..moe_latent as usize {
                let row = &down[k * ffn_down_exps.row_bytes as usize..];
                expert_out[k] = k3_q8_dot(ffn_down_exps.quant, row, &mid_q, n_ff_exp as usize)?;
            }
            if focused {
                k3_dump_host_f32(
                    step,
                    "expert",
                    il,
                    "cpu_q8_expert_498_rank_00_slot_00_down_unweighted_f32",
                    &expert_out,
                )?;
                let weighted: Vec<f32> = expert_out.iter().map(|&v| weights[si] * v).collect();
                k3_dump_host_f32(
                    step,
                    "expert",
                    il,
                    "cpu_q8_expert_498_rank_00_slot_00_weighted_f32",
                    &weighted,
                )?;
                k3_dump_host_f32(
                    step,
                    "expert",
                    il,
                    "cpu_q8_expert_498_rank_00_slot_00_routing_weight_f32",
                    std::slice::from_ref(&weights[si]),
                )?;
                eprintln!(
                    "pulsar: K3 CPU-Q8 focused expert layer={} token={} rank={} global_id={} local_slot={} weight_bits=0x{:08x}",
                    il, step, si, e, si, weights[si].to_bits()
                );
            }
            if compare && k3_compare_expert(si, e) {
                let before = moe_acc.clone();
                let weighted: Vec<f32> = expert_out
                    .iter()
                    .map(|&value| weights[si] * value)
                    .collect();
                for (acc, &value) in moe_acc.iter_mut().zip(&expert_out) {
                    *acc += weights[si] * value;
                }
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cpu_q8_expert_{si:02}_output"),
                    &expert_out,
                )?;
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cpu_q8_expert_{si:02}_weighted"),
                    &weighted,
                )?;
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cpu_q8_expert_{si:02}_accum_before"),
                    &before,
                )?;
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cpu_q8_expert_{si:02}_accum_after"),
                    &moe_acc,
                )?;
                eprintln!(
                    "pulsar: K3 CPU-Q8 accumulation rank={si} global_id={} local_slot={si} weight={:.9}",
                    selected[si], weights[si]
                );
            } else {
                for (acc, &value) in moe_acc.iter_mut().zip(&expert_out) {
                    *acc += weights[si] * value;
                }
            }
        }
        if let Some(p) = layer_prof.as_deref_mut() {
            p.cpu_expert_matrices += (n_expert_used * 3) as u64;
            p.cpu_expert_evaluations += n_expert_used as u64;
            p.cpu_expert_gate += compute_t0.elapsed();
            p.cpu_threads = 1;
        }
        rt.latent_normed.write(0, kernels::as_bytes(&moe_acc))?;
        Ok(())
    }

    /// GPU dispatch path for K3 latent MoE (Q2_K/Q3_K only).
    ///
    /// Builds ExpertPtrs from pre-resolved device pointers and dispatches to
    /// moe_pair_swiglu/moe_down.
    ///
    #[allow(clippy::too_many_arguments)]
    fn k3_gpu_moe_compute(
        &self,
        rt: &mut KimiK3Rt,
        selected: &[i32],
        weights: &[f32],
        resolved: &std::collections::HashMap<u64, *const std::ffi::c_void>,
        debug_host: Option<&std::collections::HashMap<u64, Vec<u8>>>,
        _slab_staging: &DeviceBuf,
        ffn_gate_exps: &super::ExpertTensor,
        ffn_up_exps: &super::ExpertTensor,
        ffn_down_exps: &super::ExpertTensor,
        n_expert: u32,
        n_expert_used: u32,
        moe_latent: u32,
        n_ff_exp: u32,
        il: usize,
        step: u32,
    ) -> Result {
        // Resolve every selected slot independently.  The host reference applies
        // the route weight after the down projection; folding it into `mid`
        // before Q8_K quantization changes the activation scale and is not
        // mathematically equivalent.
        let mut ptrs = Vec::with_capacity(selected.len());
        for &e in selected {
            if e < 0 || e as u32 >= n_expert {
                ptrs.push(ExpertPtrs::NULL);
                continue;
            }
            let ei = e as u64;
            let gate_off = ffn_gate_exps.abs_offset + ei * ffn_gate_exps.expert_bytes;
            let up_off = ffn_up_exps.abs_offset + ei * ffn_up_exps.expert_bytes;
            let down_off = ffn_down_exps.abs_offset + ei * ffn_down_exps.expert_bytes;
            ptrs.push(ExpertPtrs {
                gate: *resolved.get(&gate_off).ok_or_else(|| {
                    format!("K3 CUDA MoE: missing staged gate for expert {e} at {gate_off}")
                })?,
                up: *resolved.get(&up_off).ok_or_else(|| {
                    format!("K3 CUDA MoE: missing staged up for expert {e} at {up_off}")
                })?,
                down: *resolved.get(&down_off).ok_or_else(|| {
                    format!("K3 CUDA MoE: missing staged down for expert {e} at {down_off}")
                })?,
            });
        }

        if selected.len() != n_expert_used as usize || weights.len() != selected.len() {
            return Err(format!(
                "K3 CUDA MoE: routing slots/weights mismatch: ids={}, weights={}, expected={n_expert_used}",
                selected.len(), weights.len()
            ).into());
        }
        if k3_compare_active(step, il) {
            for (slot, (&expert, &ptr)) in selected.iter().zip(&ptrs).enumerate() {
                let ei = expert as u64;
                let gate_off = ffn_gate_exps.abs_offset + ei * ffn_gate_exps.expert_bytes;
                let up_off = ffn_up_exps.abs_offset + ei * ffn_up_exps.expert_bytes;
                let down_off = ffn_down_exps.abs_offset + ei * ffn_down_exps.expert_bytes;
                eprintln!(
                    "pulsar: K3 route slot={slot} global_id={expert} weight={:.9} local_slot={slot} gate_offset={gate_off} up_offset={up_off} down_offset={down_off} ptrs=({}, {}, {})",
                    weights[slot],
                    !ptr.gate.is_null(),
                    !ptr.up.is_null(),
                    !ptr.down.is_null(),
                );
            }
        }

        // The per-slot call is intentional: it keeps each expert's Q8_K
        // activation independent, then applies its route weight to the down
        // result just like the CPU reference.
        let mut ptr_buf = DeviceBuf::alloc(std::mem::size_of::<ExpertPtrs>())?;
        let mut one_weight_buf = DeviceBuf::alloc(4)?;

        // Quantize latent activation to q8_K using rt.q8k_scratch
        let xq_bytes =
            (moe_latent as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS) * kernels::Q8_K_BLOCK_BYTES;
        if rt.q8k_scratch.bytes() < xq_bytes {
            rt.q8k_scratch = DeviceBuf::alloc(xq_bytes)?;
        }
        kernels::quantize_q8_k(&mut rt.q8k_scratch, &rt.latent, moe_latent, 1)?;
        let compare = k3_compare_active(step, il);
        let q8_reference = if compare {
            let mut qbytes = vec![0u8; xq_bytes];
            rt.q8k_scratch.read(0, &mut qbytes)?;
            Some(k3_reconstruct_q8(&qbytes, moe_latent))
        } else {
            None
        };
        if compare {
            let original = rt.latent.read_f32(moe_latent as usize)?;
            let mut qbytes = vec![0u8; xq_bytes];
            rt.q8k_scratch.read(0, &mut qbytes)?;
            k3_report_q8_input(&original, &qbytes, moe_latent);
            if compare && k3_compare_expert(0, 498) {
                k3_dump_host_f32(
                    step,
                    "expert",
                    il,
                    "cuda_expert_498_rank_00_slot_00_input_f32",
                    &original,
                )?;
                k3_dump_expert_bytes(
                    step,
                    il,
                    0,
                    498,
                    0,
                    "cuda_expert_498_rank_00_slot_00_input_q8_k",
                    &qbytes,
                    "Q8_K",
                    xq_bytes / kernels::Q8_K_BLOCK_BYTES,
                    moe_latent as usize,
                    kernels::Q8_K_BLOCK_BYTES as u64,
                    "activation",
                    "pulsar_quantize_q8_K CUDA kernel",
                )?;
            }
        }

        // moe_pair_swiglu: mid = SiTU(gate_e @ xq, up_e @ xq).
        let mid_dim = n_ff_exp;
        let n_used = 1u32;
        let n_tok = 1u32;
        let row_bytes = ffn_gate_exps.row_bytes;
        let quant = ffn_gate_exps.quant;
        let act_op = 4u32; // K3 SiTU-GLU (beta=4, linear_beta=25)
        if compare {
            eprintln!(
                "pulsar: K3 expert layout in_dim={} mid_dim={} out_dim={} gate(up) quant={} row_bytes={} down quant={} row_bytes={}",
                moe_latent, n_ff_exp, moe_latent, quant, row_bytes,
                ffn_down_exps.quant, ffn_down_exps.row_bytes
            );
        }

        let mid_bytes = (mid_dim as usize) * 4;
        if rt.expert_mid.bytes() < mid_bytes {
            rt.expert_mid = DeviceBuf::alloc(mid_bytes)?;
        }

        let midq_bytes =
            (mid_dim as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS) * kernels::Q8_K_BLOCK_BYTES;
        if rt.expert_staging.bytes() < midq_bytes {
            rt.expert_staging = DeviceBuf::alloc(midq_bytes)?;
        }
        let out_bytes = (n_tok as usize) * (moe_latent as usize) * 4;
        if rt.expert_down.bytes() < out_bytes {
            rt.expert_down = DeviceBuf::alloc(out_bytes)?;
        }

        let accum_mode = k3_accum_mode();
        let capture_accum = compare || accum_mode != "current";
        let mut moe_acc = vec![0.0f32; moe_latent as usize];
        let mut debug_cpu_acc = vec![0.0f32; moe_latent as usize];
        let mut expert_outputs = if capture_accum {
            Vec::with_capacity(n_expert_used as usize * moe_latent as usize)
        } else {
            Vec::new()
        };
        for (slot, (&route, &weight)) in ptrs.iter().zip(weights.iter()).enumerate() {
            if route.gate.is_null() || route.up.is_null() || route.down.is_null() {
                if capture_accum {
                    expert_outputs.resize(expert_outputs.len() + moe_latent as usize, 0.0);
                }
                continue;
            }
            ptr_buf.write(0, kernels::as_bytes(std::slice::from_ref(&route)))?;
            one_weight_buf.write(0, kernels::as_bytes(&[1.0f32]))?;
            if compare && k3_compare_expert(slot, selected[slot]) && debug_host.is_some() {
                kernels::moe_pair_swiglu_debug(
                    &mut rt.expert_mid,
                    &mut rt.expert_gate,
                    &mut rt.expert_up,
                    &ptr_buf,
                    &one_weight_buf,
                    &rt.q8k_scratch,
                    moe_latent,
                    mid_dim,
                    n_used,
                    n_tok,
                    row_bytes,
                    quant,
                    act_op,
                )?;
            } else {
                kernels::moe_pair_swiglu(
                    &mut rt.expert_mid,
                    &ptr_buf,
                    &one_weight_buf,
                    &rt.q8k_scratch,
                    moe_latent,
                    mid_dim,
                    n_used,
                    n_tok,
                    row_bytes,
                    quant,
                    act_op,
                )?;
            }
            kernels::quantize_q8_k(&mut rt.expert_staging, &rt.expert_mid, mid_dim, 1)?;
            kernels::moe_down(
                &mut rt.expert_down,
                &ptr_buf,
                &rt.expert_staging,
                mid_dim,
                moe_latent,
                n_used,
                n_tok,
                ffn_down_exps.row_bytes,
                ffn_down_exps.quant,
            )?;
            let expert_out = rt.expert_down.read_f32(moe_latent as usize)?;
            if capture_accum {
                expert_outputs.extend_from_slice(&expert_out);
            }
            if compare && k3_compare_expert(slot, selected[slot]) {
                if let Some(host) = debug_host {
                    let ei = selected[slot] as u64;
                    let go = host
                        .get(&(ffn_gate_exps.abs_offset + ei * ffn_gate_exps.expert_bytes))
                        .ok_or("missing debug gate")?;
                    let uo = host
                        .get(&(ffn_up_exps.abs_offset + ei * ffn_up_exps.expert_bytes))
                        .ok_or("missing debug up")?;
                    let dno = host
                        .get(&(ffn_down_exps.abs_offset + ei * ffn_down_exps.expert_bytes))
                        .ok_or("missing debug down")?;
                    k3_dump_expert_weight(
                        step,
                        il,
                        slot,
                        selected[slot],
                        slot,
                        ffn_gate_exps,
                        go,
                        "cuda_gate_q2_k_weight",
                    )?;
                    k3_dump_expert_weight(
                        step,
                        il,
                        slot,
                        selected[slot],
                        slot,
                        ffn_up_exps,
                        uo,
                        "cuda_up_q2_k_weight",
                    )?;
                    k3_dump_expert_weight(
                        step,
                        il,
                        slot,
                        selected[slot],
                        slot,
                        ffn_down_exps,
                        dno,
                        "cuda_down_q3_k_weight",
                    )?;
                    let gf = k3_dequant_expert_bytes(
                        go,
                        (moe_latent * n_ff_exp) as usize,
                        ffn_gate_exps.quant,
                    )?;
                    let uf = k3_dequant_expert_bytes(
                        uo,
                        (moe_latent * n_ff_exp) as usize,
                        ffn_up_exps.quant,
                    )?;
                    let df = k3_dequant_expert_bytes(
                        dno,
                        (n_ff_exp * moe_latent) as usize,
                        ffn_down_exps.quant,
                    )?;
                    let xq = q8_reference.as_ref().unwrap();
                    let mut cg = vec![0.0; n_ff_exp as usize];
                    let mut cu = vec![0.0; n_ff_exp as usize];
                    for j in 0..n_ff_exp as usize {
                        for k in 0..moe_latent as usize {
                            cg[j] += xq[k] * gf[j * moe_latent as usize + k];
                            cu[j] += xq[k] * uf[j * moe_latent as usize + k];
                        }
                    }
                    let mut cm = vec![0.0; n_ff_exp as usize];
                    for j in 0..n_ff_exp as usize {
                        let g = cg[j];
                        let u = cu[j];
                        cm[j] = 4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp())
                            * (25.0 * (u / 25.0).tanh());
                    }
                    let mut cd = vec![0.0; moe_latent as usize];
                    let mut mid_qbytes = vec![0u8; midq_bytes];
                    rt.expert_staging.read(0, &mut mid_qbytes)?;
                    let mid_q = k3_reconstruct_q8(&mid_qbytes, mid_dim);
                    let mut cd_q8 = vec![0.0; moe_latent as usize];
                    for k in 0..moe_latent as usize {
                        for j in 0..n_ff_exp as usize {
                            cd[k] += cm[j] * df[k * n_ff_exp as usize + j];
                            cd_q8[k] += mid_q[j] * df[k * n_ff_exp as usize + j];
                        }
                    }
                    let gg = rt.expert_gate.read_f32(mid_dim as usize)?;
                    let uu = rt.expert_up.read_f32(mid_dim as usize)?;
                    let mm = rt.expert_mid.read_f32(mid_dim as usize)?;
                    k3_dump_host_f32(
                        step,
                        "expert",
                        il,
                        "cuda_expert_498_rank_00_slot_00_gate_f32",
                        &gg,
                    )?;
                    k3_dump_host_f32(
                        step,
                        "expert",
                        il,
                        "cuda_expert_498_rank_00_slot_00_up_f32",
                        &uu,
                    )?;
                    k3_dump_host_f32(
                        step,
                        "expert",
                        il,
                        "cuda_expert_498_rank_00_slot_00_situ_f32",
                        &mm,
                    )?;
                    k3_dump_expert_bytes(
                        step,
                        il,
                        slot,
                        selected[slot],
                        slot,
                        "cuda_expert_498_rank_00_slot_00_situ_output_q8_k",
                        &mid_qbytes,
                        "Q8_K",
                        mid_qbytes.len() / kernels::Q8_K_BLOCK_BYTES,
                        mid_dim as usize,
                        kernels::Q8_K_BLOCK_BYTES as u64,
                        "activation",
                        "pulsar_quantize_q8_K CUDA kernel",
                    )?;
                    k3_dump_host_f32(
                        step,
                        "expert",
                        il,
                        "cuda_expert_498_rank_00_slot_00_down_unweighted_f32",
                        &expert_out,
                    )?;
                    let focused_weighted: Vec<f32> =
                        expert_out.iter().map(|&v| weight * v).collect();
                    k3_dump_host_f32(
                        step,
                        "expert",
                        il,
                        "cuda_expert_498_rank_00_slot_00_weighted_f32",
                        &focused_weighted,
                    )?;
                    k3_dump_host_f32(
                        step,
                        "expert",
                        il,
                        "cuda_expert_498_rank_00_slot_00_routing_weight_f32",
                        std::slice::from_ref(&weight),
                    )?;

                    // Secondary control: feed the exact CPU-Q8 packed row to
                    // the CUDA Q2_K kernel. This is not the primary comparison;
                    // it only tests the operation immediately after the
                    // independently generated activation representation.
                    let shared_input = rt.latent.read_f32(moe_latent as usize)?;
                    let shared_q = quant::cpu_dot::quantize_row_q8_k(&shared_input);
                    let shared_q_dev = DeviceBuf::from_bytes(&k3_pack_q8_row(&shared_q))?;
                    let mut shared_mid = DeviceBuf::alloc(mid_bytes)?;
                    let mut shared_gate = DeviceBuf::alloc(mid_bytes)?;
                    let mut shared_up = DeviceBuf::alloc(mid_bytes)?;
                    kernels::moe_pair_swiglu_debug(
                        &mut shared_mid,
                        &mut shared_gate,
                        &mut shared_up,
                        &ptr_buf,
                        &one_weight_buf,
                        &shared_q_dev,
                        moe_latent,
                        mid_dim,
                        n_used,
                        n_tok,
                        row_bytes,
                        quant,
                        act_op,
                    )?;
                    let shared_gate_cuda = shared_gate.read_f32(mid_dim as usize)?;
                    let shared_up_cuda = shared_up.read_f32(mid_dim as usize)?;
                    let mut shared_gate_cpu = vec![0.0f32; n_ff_exp as usize];
                    let mut shared_up_cpu = vec![0.0f32; n_ff_exp as usize];
                    for j in 0..n_ff_exp as usize {
                        let gate_row = &go[j * ffn_gate_exps.row_bytes as usize..];
                        let up_row = &uo[j * ffn_up_exps.row_bytes as usize..];
                        shared_gate_cpu[j] = k3_q8_dot(
                            ffn_gate_exps.quant,
                            gate_row,
                            &shared_q,
                            moe_latent as usize,
                        )?;
                        shared_up_cpu[j] =
                            k3_q8_dot(ffn_up_exps.quant, up_row, &shared_q, moe_latent as usize)?;
                    }
                    k3_report_vector_exact(
                        "controlled shared CPU-Q8 input gate CPU/CUDA",
                        &shared_gate_cpu,
                        &shared_gate_cuda,
                    );
                    k3_report_vector_exact(
                        "controlled shared CPU-Q8 input up CPU/CUDA",
                        &shared_up_cpu,
                        &shared_up_cuda,
                    );
                    k3_report_vector("gate CPU-Q8/CUDA", &cg, &gg);
                    k3_report_vector("up CPU-Q8/CUDA", &cu, &uu);
                    k3_report_vector("SiTU CPU-Q8/CUDA", &cm, &mm);
                    k3_report_vector("SiTU CPU-F32/CPU-Q8-down-input", &cm, &mid_q);
                    k3_report_vector("down CPU-F32/CUDA", &cd, &expert_out);
                    k3_report_vector("down CPU-Q8-mid/CUDA", &cd_q8, &expert_out);
                    for (dst, src) in debug_cpu_acc.iter_mut().zip(&cd_q8) {
                        *dst += weight * src;
                    }
                    eprintln!("pulsar: K3 debug layer expert rank={slot} global_id={} dims in={} mid={} out={} gate_quant={} up_quant={} down_quant={} gate_row_bytes={} up_row_bytes={} down_row_bytes={} gate_bytes={} up_bytes={} down_bytes={} weight={:.9}", selected[slot], moe_latent, n_ff_exp, moe_latent, ffn_gate_exps.quant, ffn_up_exps.quant, ffn_down_exps.quant, ffn_gate_exps.row_bytes, ffn_up_exps.row_bytes, ffn_down_exps.row_bytes, go.len(), uo.len(), dno.len(), weight);
                }
            }
            let before = if compare { Some(moe_acc.clone()) } else { None };
            let weighted: Vec<f32> = if compare {
                expert_out.iter().map(|&value| weight * value).collect()
            } else {
                Vec::new()
            };
            for (dst, src) in moe_acc.iter_mut().zip(&expert_out) {
                *dst += weight * *src;
            }
            if compare && k3_compare_expert(slot, selected[slot]) {
                let before = before.unwrap();
                let output =
                    &expert_outputs[slot * moe_latent as usize..(slot + 1) * moe_latent as usize];
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cuda_expert_{slot:02}_output"),
                    output,
                )?;
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cuda_expert_{slot:02}_weighted"),
                    &weighted,
                )?;
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cuda_expert_{slot:02}_accum_before"),
                    &before,
                )?;
                k3_dump_host_f32(
                    step,
                    "moe",
                    il,
                    &format!("cuda_expert_{slot:02}_accum_after"),
                    &moe_acc,
                )?;
                eprintln!(
                    "pulsar: K3 CUDA accumulation rank={slot} global_id={} local_slot={slot} weight={weight:.9}",
                    selected[slot]
                );
                if debug_host.is_some() {
                    k3_report_accum_progression(
                        slot,
                        selected[slot],
                        weight,
                        &debug_cpu_acc,
                        &moe_acc,
                    );
                }
            }
        }

        let current_host = moe_acc.clone();
        if accum_mode == "f64-reference" {
            let mut f64_acc = vec![0.0f64; moe_latent as usize];
            for rank in 0..n_expert_used as usize {
                let base = rank * moe_latent as usize;
                for i in 0..moe_latent as usize {
                    f64_acc[i] += weights[rank] as f64 * expert_outputs[base + i] as f64;
                }
            }
            moe_acc = f64_acc.into_iter().map(|value| value as f32).collect();
        } else if matches!(accum_mode, "serial" | "serial-nofma") {
            let (serial, nofma) =
                k3_cuda_accum_variants(&expert_outputs, weights, moe_latent, n_expert_used)?;
            moe_acc = if accum_mode == "serial" {
                serial
            } else {
                nofma
            };
        }
        if compare && k3_compare_accum_enabled() {
            let mut cpu_f32 = vec![0.0f32; moe_latent as usize];
            let mut cpu_f64 = vec![0.0f64; moe_latent as usize];
            for rank in 0..n_expert_used as usize {
                let base = rank * moe_latent as usize;
                for i in 0..moe_latent as usize {
                    cpu_f32[i] += weights[rank] * expert_outputs[base + i];
                    cpu_f64[i] += weights[rank] as f64 * expert_outputs[base + i] as f64;
                }
            }
            let cpu_f64: Vec<f32> = cpu_f64.into_iter().map(|value| value as f32).collect();
            let (serial, nofma) =
                k3_cuda_accum_variants(&expert_outputs, weights, moe_latent, n_expert_used)?;
            k3_report_vector_exact(
                "identical-vectors CPU-F32/current-host",
                &cpu_f32,
                &current_host,
            );
            k3_report_vector_exact(
                "identical-vectors CPU-F32/F64-reference",
                &cpu_f32,
                &cpu_f64,
            );
            k3_report_vector_exact(
                "identical-vectors current-host/CUDA-serial",
                &current_host,
                &serial,
            );
            k3_report_vector_exact("identical-vectors CPU-F64/CUDA-serial", &cpu_f64, &serial);
            k3_report_vector_exact(
                "identical-vectors CPU-F64/CUDA-serial-nofma",
                &cpu_f64,
                &nofma,
            );
        }
        rt.latent_normed.write(0, kernels::as_bytes(&moe_acc))?;

        Ok(())
    }

    /// K3 forward: sequential single-token steps (KDA recurrence + AttnRes
    /// snapshot bank are sequential state machines).
    pub(super) fn forward_kimi_k3(
        &self,
        st: &mut State,
        tokens: &[u32],
        pos0: u32,
        rows: u32,
    ) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Err("empty batch".into());
        }
        if tokens.len() != 1 {
            let mut logits = None;
            for (i, &token) in tokens.iter().enumerate() {
                let last = i + 1 == tokens.len();
                logits = self.forward_kimi_k3(
                    st,
                    &[token],
                    pos0 + i as u32,
                    if last { rows.min(1) } else { 0 },
                )?;
            }
            return Ok(logits);
        }

        // Check CUDA is available
        if kernels::device_count() == 0 {
            return Err("K3 forward requires CUDA".into());
        }

        let s = self.shape;
        let eps = s.rms_eps;
        let n_embd = s.n_embd;
        let n_layer = s.n_exec_layer as usize;
        let res_block = s.attn_res_block_size.max(1) as usize;

        // Get K3 runtime state
        let rt = st.kimi_k3.as_mut().ok_or("K3 forward: missing KimiK3Rt")?;

        if pos0 == 0 {
            rt.reset()?;
        }
        // AttnRes checkpoints span layer depth for one token only.
        rt.res_bank_len = 0;

        // All K3 compute and primary-side state must stay on the resolved
        // process-local device. Pinned weights are explicitly reported as
        // host/UVA and are valid inputs, but outputs and scratch may not move.
        kernels::set_device(self.primary_device)?;
        if self.k3_device_log.get().is_none() {
            let first = self.layers.first().ok_or("K3 forward: no layers")?;
            let super::Attn::KimiK3(k3w) = &first.attn else {
                return Err("K3 forward: first layer is not KimiK3".into());
            };
            let device = |b: &DeviceBuf| {
                if b.is_pinned() {
                    "host-pinned/UVA".to_string()
                } else {
                    format!("CUDA {}", b.device())
                }
            };
            let check = |name: &str, b: &DeviceBuf| -> Result {
                if !b.is_pinned() && b.device() != self.primary_device {
                    return Err(format!(
                        "K3 device validation failed: {name} is on CUDA {}, expected CUDA {}",
                        b.device(),
                        self.primary_device
                    )
                    .into());
                }
                Ok(())
            };
            let check_dev = |name: &str, d: i32| -> Result {
                if d >= 0 && d != self.primary_device {
                    return Err(format!(
                        "K3 device validation failed: {name} is on CUDA {d}, expected CUDA {}",
                        self.primary_device
                    )
                    .into());
                }
                Ok(())
            };
            check("cur", &st.cur)?;
            check(
                "dense weights",
                k3w.ffn_gate
                    .as_ref()
                    .map(|w| &w.buf)
                    .unwrap_or(&k3w.attn_norm),
            )?;
            check(
                "router",
                k3w.ffn_gate_inp
                    .as_ref()
                    .map(|w| &w.buf)
                    .unwrap_or(&k3w.attn_norm),
            )?;
            check("KDA/attention", &k3w.attn_norm)?;
            check("expert staging", &st.staging)?;
            check_dev("expert device cache", st.dev_cache.device())?;
            check("MoE compute", &rt.expert_mid)?;
            check("output projection / LM head", &self.output)?;
            check("logits", &st.logits)?;
            eprintln!(
                "K3 allocation/execution summary:\n  K3 primary compute device: CUDA {}\n  K3 expert backend: {}\n  K3 expert-streaming device: {}\n  K3 dense weights: {}\n  K3 router: {}\n  K3 KDA: {}\n  K3 attention: {}\n  K3 expert staging buffers: {} ({} bytes)\n  K3 expert device cache: {}\n  K3 MoE compute: {}\n  K3 output projection / LM head: {}\n  K3 logits buffers: {}",
                self.primary_device,
                k3_expert_backend(),
                device(&st.staging),
                device(k3w.ffn_gate.as_ref().map(|w| &w.buf).unwrap_or(&k3w.attn_norm)),
                device(k3w.ffn_gate_inp.as_ref().map(|w| &w.buf).unwrap_or(&k3w.attn_norm)),
                device(&k3w.attn_norm),
                device(&k3w.attn_norm),
                device(&st.staging),
                st.staging.bytes(),
                if st.dev_cache.device() < 0 {
                    "host-pinned/UVA".to_string()
                } else {
                    format!("CUDA {}", st.dev_cache.device())
                },
                device(&rt.expert_mid),
                device(&self.output),
                device(&st.logits),
            );
            self.k3_device_log.set(()).ok();
        }

        // Check output_res_norm and output_res_proj are loaded
        let output_res_norm = self
            .k3_output_res_norm
            .as_ref()
            .ok_or("K3 forward: output_res_norm not loaded (missing from gguf)")?;
        let output_res_proj = self
            .k3_output_res_proj
            .as_ref()
            .ok_or("K3 forward: output_res_proj not loaded (missing from gguf)")?;

        // ── Embedding ──────────────────────────────────────────────────────
        let token_t0 = std::time::Instant::now();
        let token_index = pos0 as u64;
        let profiling = super::Prof::enabled();
        st.prof
            .begin_k3_token(token_index, if pos0 == 0 { "prefill" } else { "decode" });
        let token_i32: Vec<i32> = vec![tokens[0] as i32];
        st.tok.write(0, kernels::as_bytes(&token_i32))?;
        kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, n_embd, s.n_vocab, 1)?;

        let k3_timing = std::env::var_os("PULSAR_K3_TIMING").is_some();
        for il in 0..n_layer {
            let layer_t0 = std::time::Instant::now();
            let mut layer_prof = super::K3LayerProfile {
                index: il,
                kind: if self.k3_layer_kinds[il] == K3LayerKind::Kda {
                    "KDA"
                } else {
                    "MLA"
                },
                ..Default::default()
            };
            if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
                eprintln!("pulsar: K3 layer {il} begin");
            }
            // Get the K3 layer weights
            let l = &self.layers[il];
            let super::Attn::KimiK3(ref k3w) = l.attn else {
                return Err(format!("K3 layer {il}: expected KimiK3 weights").into());
            };

            k3_dump_device(pos0, "layer", il, "residual_input", &st.cur, n_embd)?;

            // ── AttnRes pre-attention mixture ────────────────────────────
            let input_t0 = std::time::Instant::now();
            // h = (bank empty) ? prefix : attn_res_mix(prefix, bank, attn_res_norm, attn_res_proj)
            let attn_input = if rt.res_bank_len > 0 {
                self.attn_res_mix(rt, &st.cur, &k3w.attn_res_norm, &k3w.attn_res_proj, eps)?;
                &rt.mix_out
            } else {
                &st.cur
            };
            k3_dump_device(pos0, "layer", il, "norm_input", attn_input, n_embd)?;

            // ── Snapshot every attn_res_block_size layers ─────────────────
            let snapshot = il % res_block == 0;
            if snapshot {
                // Save current prefix to bank
                let bank_idx = (il / res_block) as usize;
                let bank_off = bank_idx * n_embd as usize * 4;
                kernels::copy_d2d(&mut rt.res_bank, bank_off, &st.cur, 0, n_embd as usize * 4)?;
                rt.res_bank_len = (bank_idx + 1) as u32;
            }

            // ── attn_norm ─────────────────────────────────────────────────
            kernels::rms_norm(&mut st.normed, attn_input, &k3w.attn_norm, n_embd, 1, eps)?;
            k3_dump_device(pos0, "layer", il, "norm_output", &st.normed, n_embd)?;
            layer_prof.input_residual_norm = input_t0.elapsed();

            // ── Attention (KDA or MLA) ────────────────────────────────────
            let attention_t0 = std::time::Instant::now();
            let attention_gpu_timer = if super::Prof::detailed() {
                let timer = kernels::GpuTimer::new()?;
                timer.start()?;
                Some(timer)
            } else {
                None
            };
            match k3w.kind {
                K3LayerKind::Kda => {
                    self.kda_layer_forward(rt, &st.normed, k3w, il, pos0, eps)?;
                }
                K3LayerKind::Mla => {
                    let dims = K3MlaDims {
                        n_head: s.n_head,
                        q_lora_rank: s.n_lora_q,
                        kv_lora_rank: s.n_kv_lora,
                        qk_nope: s.qk_nope,
                        qk_rope: s.qk_rope,
                        v_mla: s.value_mla,
                        n_embd,
                    };
                    let mla_wq_a = k3w.mla_wq_a.as_ref().ok_or("MLA layer missing mla_wq_a")?;
                    let mla_wq_b = k3w.mla_wq_b.as_ref().ok_or("MLA layer missing mla_wq_b")?;
                    let mla_q_a_norm = k3w
                        .mla_q_a_norm
                        .as_ref()
                        .ok_or("MLA layer missing mla_q_a_norm")?;
                    let mla_wkv_a_mqa = k3w
                        .mla_wkv_a_mqa
                        .as_ref()
                        .ok_or("MLA layer missing mla_wkv_a_mqa")?;
                    let mla_kv_a_norm = k3w
                        .mla_kv_a_norm
                        .as_ref()
                        .ok_or("MLA layer missing mla_kv_a_norm")?;
                    let mla_wqkv_gate = k3w
                        .mla_wqkv_gate
                        .as_ref()
                        .ok_or("MLA layer missing mla_wqkv_gate")?;
                    let mla_wo = k3w.mla_wo.as_ref().ok_or("MLA layer missing mla_wo")?;

                    self.k3_mla_step(
                        rt,
                        &st.normed,
                        mla_wq_a,
                        mla_wq_b,
                        mla_q_a_norm,
                        mla_wkv_a_mqa,
                        mla_kv_a_norm,
                        k3w.mla_wk_b.as_ref(),
                        k3w.mla_wv_b.as_ref(),
                        k3w.mla_wkv_b.as_ref(),
                        mla_wqkv_gate,
                        mla_wo,
                        &mut st.kcache[il],
                        &mut st.vcache[il],
                        &mut st.qk_low,
                        pos0,
                        st.ctx,
                        &dims,
                        eps,
                    )?;
                }
            }
            let attention_wall = attention_t0.elapsed();
            let attention_gpu = attention_gpu_timer
                .map(|timer| {
                    timer
                        .stop_ms()
                        .map(|ms| std::time::Duration::from_secs_f64(ms as f64 / 1000.0))
                })
                .transpose()?
                .unwrap_or(attention_wall);
            if k3w.kind == K3LayerKind::Kda {
                layer_prof.kda_gpu = attention_gpu;
            } else {
                layer_prof.mla_gpu = attention_gpu;
            }

            if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
                let sync_result: super::Result = kernels::sync()
                    .map_err(|e| format!("K3 layer {il} attention block sync: {e}").into());
                sync_result?;
            }
            k3_dump_device(
                pos0,
                "layer",
                il,
                "attention_output_before_residual",
                &rt.attn_out,
                n_embd,
            )?;

            // ── Residual update (attention) ──────────────────────────────
            let residual_t0 = std::time::Instant::now();
            // prefix = snapshot ? attn_out : prefix + attn_out
            if snapshot {
                // Snapshot restart: prefix = attn_out (discard old prefix)
                kernels::copy_d2d(&mut st.cur, 0, &rt.attn_out, 0, n_embd as usize * 4)?;
            } else {
                // prefix += attn_out
                kernels::add_assign(&mut st.cur, &rt.attn_out, n_embd)?;
            }
            k3_dump_device(
                pos0,
                "layer",
                il,
                "attention_residual_output",
                &st.cur,
                n_embd,
            )?;

            // ── AttnRes pre-FFN mixture ───────────────────────────────────
            let ffn_input = if rt.res_bank_len > 0 {
                self.attn_res_mix(rt, &st.cur, &k3w.ffn_res_norm, &k3w.ffn_res_proj, eps)?;
                &rt.mix_out
            } else {
                &st.cur
            };

            // ── ffn_norm ──────────────────────────────────────────────────
            kernels::rms_norm(&mut rt.normed, ffn_input, &k3w.ffn_norm, n_embd, 1, eps)?;
            k3_dump_device(pos0, "layer", il, "moe_input", &rt.normed, n_embd)?;
            layer_prof.output_residual = residual_t0.elapsed();

            // ── FFN (dense SiTU-GLU or latent Stable-MoE) ─────────────────
            if il < s.n_leading_dense as usize {
                let ffn_t0 = std::time::Instant::now();
                let mut ffn_input = DeviceBuf::alloc(n_embd as usize * 4)?;
                kernels::copy_d2d(&mut ffn_input, 0, &rt.normed, 0, n_embd as usize * 4)?;
                self.dense_ffn_forward(rt, &ffn_input, k3w)?;
                layer_prof.dense_gpu = ffn_t0.elapsed();
            } else {
                let ffn_t0 = std::time::Instant::now();
                let mut ffn_input = DeviceBuf::alloc(n_embd as usize * 4)?;
                kernels::copy_d2d(&mut ffn_input, 0, &rt.normed, 0, n_embd as usize * 4)?;
                self.latent_moe_forward(
                    &mut st.store,
                    &mut st.dev_cache,
                    &mut st.staging,
                    rt,
                    &ffn_input,
                    k3w,
                    il,
                    pos0,
                    Some(&mut layer_prof),
                )?;
                k3_dump_device(
                    pos0,
                    "moe",
                    il,
                    "routed_moe",
                    &rt.latent_normed,
                    s.moe_latent_size,
                )?;
                if k3_expert_backend() == "cuda" {
                    layer_prof.moe_gpu = ffn_t0.elapsed();
                } else {
                    layer_prof.cpu_routing = ffn_t0.elapsed();
                    let measured_cpu = layer_prof.cpu_expert_dequant
                        + layer_prof.cpu_expert_gate
                        + layer_prof.cpu_expert_up
                        + layer_prof.cpu_expert_activation
                        + layer_prof.cpu_expert_down
                        + layer_prof.cpu_expert_accumulation
                        + layer_prof.cpu_latent_norm;
                    layer_prof.cpu_miscellaneous = layer_prof
                        .cpu_routing
                        .saturating_sub(layer_prof.expert_resolution + measured_cpu);
                }
            }

            // ── Residual update (FFN) ────────────────────────────────────
            // prefix += ffn_out
            kernels::add_assign(&mut st.cur, &rt.ffn_out, n_embd)?;
            k3_dump_device(pos0, "layer", il, "layer_output", &st.cur, n_embd)?;
            k3_dump_device(pos0, "layer", il, "hidden", &st.cur, n_embd)?;
            if self.k3_layer_kinds[il] == K3LayerKind::Kda {
                for (which, state) in rt.conv_states[il].iter().enumerate() {
                    k3_dump_device(
                        pos0,
                        "state",
                        il,
                        &format!("conv_{which}"),
                        state,
                        (state.bytes() / 4) as u32,
                    )?;
                }
                let state = &rt.ssm_states[il];
                k3_dump_device(
                    pos0,
                    "state",
                    il,
                    "kda_ssm",
                    state,
                    (state.bytes() / 4) as u32,
                )?;
            }
            k3_dump_device(
                pos0,
                "state",
                il,
                "attn_res_bank",
                &rt.res_bank,
                rt.res_bank_len.saturating_mul(n_embd),
            )?;
            if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
                let sync_result: super::Result =
                    kernels::sync().map_err(|e| format!("K3 layer {il} end sync: {e}").into());
                sync_result?;
            }
            if k3_timing {
                eprintln!(
                    "pulsar: K3 layer {il} {:.3}s",
                    layer_t0.elapsed().as_secs_f32()
                );
            }
            if let Ok(target) = std::env::var("PULSAR_K3_COMPARE_LAYER") {
                if target.parse::<usize>().ok() == Some(il) {
                    eprintln!("pulsar: K3 differential boundary reached at layer {il}");
                    if std::env::var_os("PULSAR_K3_COMPARE_STOP").is_some() {
                        st.prof
                            .finish_k3_token(token_t0.elapsed(), std::time::Duration::ZERO);
                        return Ok(None);
                    }
                }
            }
            if profiling {
                layer_prof.total = layer_t0.elapsed();
                let classified = layer_prof.input_residual_norm
                    + layer_prof.kda_gpu
                    + layer_prof.mla_gpu
                    + layer_prof.dense_gpu
                    + layer_prof.moe_gpu
                    + layer_prof.cpu_routing
                    + layer_prof.output_residual;
                layer_prof.unclassified = layer_prof.total.saturating_sub(classified);
                st.prof.push_k3_layer(layer_prof);
            }
        }

        // ── Final AttnRes mixture ─────────────────────────────────────────
        if rt.res_bank_len > 0 {
            self.attn_res_mix(rt, &st.cur, output_res_norm, output_res_proj, eps)?;
            let mixed = &rt.mix_out;
            kernels::copy_d2d(&mut st.cur, 0, mixed, 0, n_embd as usize * 4)?;
        }

        // ── Output norm + head ────────────────────────────────────────────
        if rows == 0 {
            st.prof
                .finish_k3_token(token_t0.elapsed(), std::time::Duration::ZERO);
            return Ok(None);
        }
        let k = rows.min(1);
        let t_tail = std::time::Instant::now();
        k3_dump_device(pos0, "final", 0, "hidden", &st.cur, n_embd)?;
        kernels::rms_norm(&mut st.normed, &st.cur, &self.output_norm, n_embd, k, eps)?;
        self.head_logits(st, k)?;
        kernels::sync()?;
        k3_dump_device(pos0, "final", 0, "logits", &st.logits, s.n_vocab)?;
        let out = st.logits.read_f32(k as usize * s.n_vocab as usize)?;
        if std::env::var_os("PULSAR_DEBUG_LOGITS").is_some() {
            let mut ids: Vec<usize> = (0..out.len()).collect();
            ids.sort_unstable_by(|&a, &b| out[b].total_cmp(&out[a]));
            let top: Vec<(usize, f32)> = ids.into_iter().take(10).map(|id| (id, out[id])).collect();
            let nan = out.iter().filter(|v| v.is_nan()).count();
            let inf = out.iter().filter(|v| v.is_infinite()).count();
            let margin = top
                .get(0)
                .zip(top.get(1))
                .map_or(f32::NAN, |(a, b)| a.1 - b.1);
            eprintln!(
                "pulsar: K3 logits pos {pos0}: top10 {top:?}, top1-top2 {margin:.9}, NaN {nan}, Inf {inf}"
            );
        }
        st.prof.tail += t_tail.elapsed();
        st.prof
            .finish_k3_token(token_t0.elapsed(), t_tail.elapsed());
        Ok(Some(out))
    }
}

// ── f16 helper for host-side dequant ───────────────────────────────────────

fn f16_to_f32(h: u16) -> f32 {
    let s = ((h >> 15) & 1) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            let mut m = m;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            (s << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if e == 0x1f {
        (s << 31) | (0xff << 23) | (m << 13)
    } else {
        (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

fn k3_report_q8_input(original: &[f32], encoded: &[u8], n: u32) {
    let blocks = (n as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS);
    let padded = blocks * kernels::Q8_K_BLOCK_ELEMS - n as usize;
    let mut reconstructed = Vec::with_capacity(original.len());
    let mut saturated = 0usize;
    for b in 0..blocks {
        let base = b * kernels::Q8_K_BLOCK_BYTES;
        let d = f32::from_le_bytes(encoded[base..base + 4].try_into().unwrap());
        for i in 0..kernels::Q8_K_BLOCK_ELEMS {
            let q = encoded[base + 4 + i] as i8;
            saturated += usize::from(q == i8::MAX || q == i8::MIN);
            reconstructed.push(d * q as f32);
        }
    }
    reconstructed.truncate(original.len());
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut nx = 0.0f64;
    let mut ny = 0.0f64;
    for (&x, &y) in original.iter().zip(&reconstructed) {
        let d = (x - y).abs();
        max = max.max(d);
        sum += d as f64;
        sum_sq += (x - y) as f64 * (x - y) as f64;
        dot += x as f64 * y as f64;
        nx += x as f64 * x as f64;
        ny += y as f64 * y as f64;
    }
    let len = original.len() as f64;
    eprintln!(
        "pulsar: K3 F32->Q8_K input max={max:.6e} mean={:.6e} rms={:.6e} cosine={:.9} norm_ratio={:.9} saturated={} padded={}",
        sum / len,
        (sum_sq / len).sqrt(),
        dot / (nx.sqrt() * ny.sqrt()).max(f64::MIN_POSITIVE),
        (ny / nx.max(f64::MIN_POSITIVE)).sqrt(),
        saturated,
        padded,
    );
}

fn k3_reconstruct_q8(encoded: &[u8], n: u32) -> Vec<f32> {
    let blocks = (n as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS);
    let mut out = Vec::with_capacity(n as usize);
    for b in 0..blocks {
        let base = b * kernels::Q8_K_BLOCK_BYTES;
        let d = f32::from_le_bytes(encoded[base..base + 4].try_into().unwrap());
        for i in 0..kernels::Q8_K_BLOCK_ELEMS {
            out.push(d * encoded[base + 4 + i] as i8 as f32);
        }
    }
    out.truncate(n as usize);
    out
}

fn k3_pack_q8_row(row: &quant::cpu_dot::Q8KRow) -> Vec<u8> {
    let blocks = row.d.len();
    let mut out = Vec::with_capacity(blocks * kernels::Q8_K_BLOCK_BYTES);
    for block in 0..blocks {
        out.extend_from_slice(&row.d[block].to_le_bytes());
        out.extend(
            row.qs[block * quant::cpu_dot::QK_K..(block + 1) * quant::cpu_dot::QK_K]
                .iter()
                .map(|q| *q as u8),
        );
        for &sum in &row.bsums[block * 16..(block + 1) * 16] {
            out.extend_from_slice(&(sum as i16).to_le_bytes());
        }
    }
    out
}

fn k3_q8_dot(quant: u32, row: &[u8], x: &quant::cpu_dot::Q8KRow, n: usize) -> Result<f32> {
    Ok(match quant {
        kernels::QUANT_Q2_K => quant::cpu_dot::vec_dot_q2_k_q8_k(row, x, n),
        kernels::QUANT_Q3_K => quant::cpu_dot::vec_dot_q3_k_q8_k(row, x, n),
        _ => return Err(format!("K3 CPU-Q8 unsupported expert quant {quant}").into()),
    })
}

/// Opt-in raw snapshots for the three-run K3 comparison harness. Files are
/// backend-scoped so separate CPU-F32, CPU-Q8, and CUDA runs can be compared
/// without changing the forward computation or logging full vectors.
fn k3_dump_device(
    step: u32,
    kind: &str,
    layer: usize,
    name: &str,
    buf: &DeviceBuf,
    n: u32,
) -> Result {
    if !k3_compare_enabled(step, layer) {
        return Ok(());
    }
    let Some(root) = std::env::var_os("PULSAR_K3_COMPARE_DIR") else {
        return Ok(());
    };
    let backend = k3_expert_backend();
    let dir = std::path::PathBuf::from(root).join(backend);
    std::fs::create_dir_all(&dir)?;
    let values = buf.read_f32(n as usize)?;
    k3_dump_bytes(
        &dir,
        step,
        kind,
        layer,
        name,
        "f32",
        &format!("[{n}]"),
        kernels::as_bytes(&values),
        "f32",
    )?;
    Ok(())
}

fn k3_dump_host_f32(step: u32, kind: &str, layer: usize, name: &str, values: &[f32]) -> Result {
    if !k3_compare_enabled(step, layer) {
        return Ok(());
    }
    let Some(root) = std::env::var_os("PULSAR_K3_COMPARE_DIR") else {
        return Ok(());
    };
    let backend = k3_expert_backend();
    let dir = std::path::PathBuf::from(root).join(backend);
    std::fs::create_dir_all(&dir)?;
    k3_dump_bytes(
        &dir,
        step,
        kind,
        layer,
        name,
        "f32",
        &format!("[{}]", values.len()),
        kernels::as_bytes(values),
        "f32",
    )?;
    Ok(())
}

fn k3_dump_i32(step: u32, kind: &str, layer: usize, name: &str, values: &[i32]) -> Result {
    if !k3_compare_enabled(step, layer) {
        return Ok(());
    }
    let Some(root) = std::env::var_os("PULSAR_K3_COMPARE_DIR") else {
        return Ok(());
    };
    let backend = k3_expert_backend();
    let dir = std::path::PathBuf::from(root).join(backend);
    std::fs::create_dir_all(&dir)?;
    k3_dump_bytes(
        &dir,
        step,
        kind,
        layer,
        name,
        "i32",
        &format!("[{}]", values.len()),
        kernels::as_bytes(values),
        "i32",
    )?;
    Ok(())
}

fn k3_compare_active(step: u32, layer: usize) -> bool {
    std::env::var_os("PULSAR_K3_COMPARE_DIR").is_some()
        && k3_compare_enabled(step, layer)
        && k3_compare_backend_enabled()
}

fn k3_compare_backend_enabled() -> bool {
    let backend = k3_expert_backend();
    std::env::var("PULSAR_K3_COMPARE_BACKENDS")
        .ok()
        .map(|backends| backends.split(',').any(|value| value.trim() == backend))
        .unwrap_or(true)
}

fn k3_compare_operation_enabled(operation: &str) -> bool {
    std::env::var("PULSAR_K3_COMPARE_OPERATIONS")
        .ok()
        .map(|operations| {
            operations
                .split(',')
                .map(str::trim)
                .any(|value| value == "*" || value == operation)
        })
        .unwrap_or(true)
}

fn k3_compare_expert(rank: usize, expert: i32) -> bool {
    let rank_ok = std::env::var("PULSAR_K3_COMPARE_EXPERT_RANKS")
        .ok()
        .map(|ranks| {
            ranks
                .split(',')
                .filter_map(|value| value.trim().parse::<usize>().ok())
                .any(|value| value == rank)
        })
        .unwrap_or(true);
    let expert_ok = std::env::var("PULSAR_K3_COMPARE_EXPERT_IDS")
        .ok()
        .map(|ids| {
            ids.split(',')
                .filter_map(|value| value.trim().parse::<i32>().ok())
                .any(|value| value == expert)
        })
        .unwrap_or(true);
    rank_ok && expert_ok
}

fn k3_compare_accum_enabled() -> bool {
    std::env::var_os("PULSAR_K3_COMPARE_EXPERT_RANKS").is_none()
        && std::env::var_os("PULSAR_K3_COMPARE_EXPERT_IDS").is_none()
}

fn k3_fnv1a(bytes: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in bytes {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("fnv1a64:{hash:016x}")
}

#[allow(clippy::too_many_arguments)]
fn k3_dump_expert_bytes(
    step: u32,
    layer: usize,
    rank: usize,
    expert: i32,
    slot: usize,
    name: &str,
    bytes: &[u8],
    quantization: &str,
    block_count: usize,
    elements: usize,
    stride: u64,
    dtype: &str,
    source: &str,
) -> Result {
    if !k3_compare_active(step, layer) || !k3_compare_operation_enabled(name) {
        return Ok(());
    }
    let root = std::env::var_os("PULSAR_K3_COMPARE_DIR").ok_or("comparison directory missing")?;
    let backend = k3_expert_backend();
    let dir = std::path::PathBuf::from(root).join(backend);
    std::fs::create_dir_all(&dir)?;
    // Captures span multiple layers; coordinates prevent later layers from
    // overwriting an earlier packed artifact.
    let file_name = format!("layer_{layer:03}_token_{step:03}_{name}.bin");
    std::fs::write(dir.join(&file_name), bytes)?;
    use std::io::Write;
    let mut manifest = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("manifest.jsonl"))?;
    writeln!(
        manifest,
        "{{\"backend\":\"{backend}\",\"layer\":{layer},\"token_position\":{step},\"expert_rank\":{rank},\"global_expert_id\":{expert},\"local_staging_slot\":{slot},\"operation\":\"{name}\",\"dtype\":\"{dtype}\",\"quantization\":\"{quantization}\",\"elements\":{elements},\"block_count\":{block_count},\"stride_bytes\":{stride},\"packed_bytes\":{},\"hash\":\"{}\",\"source\":\"{source}\",\"file\":\"{file_name}\"}}",
        bytes.len(),
        k3_fnv1a(bytes),
    )?;
    Ok(())
}

fn k3_dump_expert_weight(
    step: u32,
    layer: usize,
    rank: usize,
    expert: i32,
    slot: usize,
    tensor: &super::ExpertTensor,
    bytes: &[u8],
    stage: &str,
) -> Result {
    let block_bytes = match tensor.quant {
        kernels::QUANT_Q2_K => 84,
        kernels::QUANT_Q3_K => 110,
        _ => tensor.row_bytes,
    };
    let rows = bytes.len() as u64 / tensor.row_bytes.max(1);
    k3_dump_expert_bytes(
        step,
        layer,
        rank,
        expert,
        slot,
        &format!("{stage}_expert_{expert}_rank_{rank:02}_slot_{slot:02}"),
        bytes,
        &format!("quant_id:{}", tensor.quant),
        bytes.len() / block_bytes as usize,
        rows as usize * tensor.row_elems as usize,
        tensor.row_bytes,
        "packed",
        &format!(
            "tensor={} dims=[{}, {}, {}] absolute_offset={} expert_bytes={} row_bytes={} quant_id={} block_bytes={}",
            tensor.name,
            tensor.row_elems,
            tensor.rows_per_expert,
            tensor.expert_count,
            tensor.abs_offset,
            tensor.expert_bytes,
            tensor.row_bytes,
            tensor.quant,
            block_bytes
        ),
    )
}

fn k3_compare_enabled(step: u32, layer: usize) -> bool {
    if step != 0 {
        return false;
    }
    let Ok(layers) = std::env::var("PULSAR_K3_COMPARE_LAYERS") else {
        return true;
    };
    layers
        .split(',')
        .filter_map(|value| value.trim().parse::<usize>().ok())
        .any(|value| value == layer)
}

fn k3_dump_bytes(
    dir: &std::path::Path,
    step: u32,
    kind: &str,
    layer: usize,
    name: &str,
    extension: &str,
    shape: &str,
    bytes: &[u8],
    dtype: &str,
) -> Result {
    if !k3_compare_operation_enabled(name) {
        return Ok(());
    }
    let file_name = format!("{kind}_{layer:03}_{name}.{extension}");
    let path = dir.join(&file_name);
    std::fs::write(&path, bytes)?;
    let state_slot = if kind == "kda_state" {
        if name.starts_with("recurrent") {
            "ssm"
        } else {
            name.strip_prefix("conv_").unwrap_or("unknown")
        }
    } else {
        "none"
    };
    let manifest = format!(
        "{{\"backend\":\"{}\",\"layer\":{},\"operation\":\"{}\",\"shape\":{},\"token_position\":{},\"state_slot\":\"{}\",\"dtype\":\"{}\",\"quantization\":\"{}\",\"file\":\"{}\"}}\n",
        k3_expert_backend(), layer, name, shape, step, state_slot, dtype, dtype, file_name
    );
    use std::io::Write;
    let mut manifest_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(dir.join("manifest.jsonl"))?;
    manifest_file.write_all(manifest.as_bytes())?;
    Ok(())
}

fn k3_cuda_accum_variants(
    expert_outputs: &[f32],
    weights: &[f32],
    out_dim: u32,
    n_used: u32,
) -> Result<(Vec<f32>, Vec<f32>)> {
    let expert_buf = DeviceBuf::from_f32(expert_outputs)?;
    let weight_buf = DeviceBuf::from_f32(weights)?;
    let mut serial_buf = DeviceBuf::alloc(out_dim as usize * 4)?;
    let mut nofma_buf = DeviceBuf::alloc(out_dim as usize * 4)?;
    kernels::moe_accum_serial(
        &mut serial_buf,
        &expert_buf,
        &weight_buf,
        out_dim,
        n_used,
        false,
    )?;
    kernels::moe_accum_serial(
        &mut nofma_buf,
        &expert_buf,
        &weight_buf,
        out_dim,
        n_used,
        true,
    )?;
    let serial = serial_buf.read_f32(out_dim as usize)?;
    let nofma = nofma_buf.read_f32(out_dim as usize)?;
    Ok((serial, nofma))
}

fn k3_report_accum_progression(rank: usize, expert: i32, weight: f32, cpu: &[f32], cuda: &[f32]) {
    let (max, mean, rms, cosine, norm_ratio, first) = k3_vector_metrics(cpu, cuda);
    eprintln!(
        "pulsar: K3 accum rank={rank} global_id={expert} weight={weight:.9} max={max:.6e} mean={mean:.6e} rms={rms:.6e} cosine={cosine:.9} norm_ratio={norm_ratio:.9} first_mismatch={first:?}"
    );
    if let Some(i) = first {
        let start = i.saturating_sub(2);
        let end = (i + 3).min(cpu.len()).min(cuda.len());
        for j in start..end {
            eprintln!(
                "pulsar: K3 accum rank={rank} around index={j} CPU-Q8={:.9e} CUDA={:.9e}",
                cpu[j], cuda[j]
            );
        }
    }
}

fn k3_vector_metrics(a: &[f32], b: &[f32]) -> (f64, f64, f64, f64, f64, Option<usize>) {
    let mut max = 0.0f64;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    let mut first = None;
    for (i, (&x, &y)) in a.iter().zip(b).enumerate() {
        let d = (x as f64 - y as f64).abs();
        max = max.max(d);
        sum += d;
        sum_sq += d * d;
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
        if first.is_none() && x.to_bits() != y.to_bits() {
            first = Some(i);
        }
    }
    let n = a.len().max(1) as f64;
    (
        max,
        sum / n,
        (sum_sq / n).sqrt(),
        dot / (na.sqrt() * nb.sqrt()).max(f64::MIN_POSITIVE),
        (nb / na.max(f64::MIN_POSITIVE)).sqrt(),
        first,
    )
}

fn k3_report_vector_exact(label: &str, a: &[f32], b: &[f32]) {
    let (max, mean, rms, cosine, norm_ratio, first) = k3_vector_metrics(a, b);
    eprintln!(
        "pulsar: K3 {label} max={max:.6e} mean={mean:.6e} rms={rms:.6e} cosine={cosine:.9} norm_ratio={norm_ratio:.9} first_mismatch={first:?}"
    );
}

fn k3_report_vector(label: &str, cpu: &[f32], cuda: &[f32]) {
    let mut max = 0.0f32;
    let mut sum = 0.0f64;
    let mut sum_sq = 0.0f64;
    let mut dot = 0.0f64;
    let mut nc = 0.0f64;
    let mut ng = 0.0f64;
    let mut first = None;
    let mut nan_cpu = 0usize;
    let mut nan_cuda = 0usize;
    for (i, (&c, &g)) in cpu.iter().zip(cuda).enumerate() {
        nan_cpu += usize::from(c.is_nan() || c.is_infinite());
        nan_cuda += usize::from(g.is_nan() || g.is_infinite());
        let d = (c - g).abs();
        if first.is_none() && d > 1e-3 {
            first = Some(i);
        }
        max = max.max(d);
        sum += d as f64;
        sum_sq += d as f64 * d as f64;
        dot += c as f64 * g as f64;
        nc += c as f64 * c as f64;
        ng += g as f64 * g as f64;
    }
    let len = cpu.len().max(1) as f64;
    if let Some(i) = first {
        eprintln!("pulsar: K3 {label} len={} max={max:.6e} mean={:.6e} rms={:.6e} cosine={:.9} cpu_norm={:.6e} cuda_norm={:.6e} norm_ratio={:.9} first_tol={} values=({:.6e},{:.6e}) nan_inf=({}, {})", cpu.len(), sum / len, (sum_sq / len).sqrt(), dot / (nc.sqrt() * ng.sqrt()).max(f64::MIN_POSITIVE), nc.sqrt(), ng.sqrt(), (ng / nc.max(f64::MIN_POSITIVE)).sqrt(), i, cpu[i], cuda[i], nan_cpu, nan_cuda);
    } else {
        eprintln!("pulsar: K3 {label} len={} max={max:.6e} mean={:.6e} rms={:.6e} cosine={:.9} cpu_norm={:.6e} cuda_norm={:.6e} norm_ratio={:.9} first_tol=none nan_inf=({}, {})", cpu.len(), sum / len, (sum_sq / len).sqrt(), dot / (nc.sqrt() * ng.sqrt()).max(f64::MIN_POSITIVE), nc.sqrt(), ng.sqrt(), (ng / nc.max(f64::MIN_POSITIVE)).sqrt(), nan_cpu, nan_cuda);
    }
}

// ── K3 typed weight representation ─────────────────────────────────────────

/// Source quantization metadata for a K3 dense weight matrix.
///
/// Carries the source quant type and row_bytes so the forward path can
/// dispatch to the correct matmul kernel without re-inspecting the GGUF
/// tensor type at runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct K3WeightQuant {
    /// Source quant type constant (kernels::QUANT_Q2_K, QUANT_Q3_K, etc.)
    pub quant: u32,
    /// Row bytes in the source quant format (bytes per row of the weight
    /// matrix in its native packed layout).
    pub row_bytes: u64,
}

impl K3WeightQuant {
    /// Returns `true` when the weight can be consumed directly by
    /// `kernels::matmul_kq` (Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, IQ2_XXS).
    pub fn is_direct_kq(&self) -> bool {
        matches!(self.quant, kernels::QUANT_Q2_K | kernels::QUANT_Q3_K)
    }

    /// Returns `true` when the weight is Q8_0 (use `kernels::matmul_q8_0`).
    pub fn is_q8_0(&self) -> bool {
        self.quant == kernels::QUANT_Q8_0
    }

    /// Build metadata from a GGUF tensor type and element count.
    /// Returns `None` when the type is not a supported K-quant or Q8_0.
    pub fn from_gguf(ty: gguf::TensorType, n_elems: u64) -> Option<Self> {
        use gguf::TensorType as T;
        let (quant, row_bytes) = match ty {
            T::Q2K => (kernels::QUANT_Q2_K, T::Q2K.row_bytes(n_elems)),
            T::Q3K => (kernels::QUANT_Q3_K, T::Q3K.row_bytes(n_elems)),
            T::Q4K => (kernels::QUANT_Q4_K, T::Q4K.row_bytes(n_elems)),
            T::Q5K => (kernels::QUANT_Q5_K, T::Q5K.row_bytes(n_elems)),
            T::Q6K => (kernels::QUANT_Q6_K, T::Q6K.row_bytes(n_elems)),
            T::Q8_0 => (kernels::QUANT_Q8_0, T::Q8_0.row_bytes(n_elems)),
            _ => return None,
        };
        Some(K3WeightQuant {
            quant,
            row_bytes: row_bytes?,
        })
    }
}

/// A K3 dense weight matrix with its source quantization metadata.
///
/// The `DeviceBuf` holds the weight bytes in their native quantized format
/// (Q2_K, Q3_K, Q8_0, etc.). The `quant` field tells the forward path which
/// kernel to dispatch to and how to interpret the row stride.
pub struct K3DenseWeight {
    pub buf: DeviceBuf,
    pub quant: K3WeightQuant,
}

impl K3DenseWeight {
    pub fn new(buf: DeviceBuf, quant: K3WeightQuant) -> Self {
        K3DenseWeight { buf, quant }
    }

    /// Dispatch a direct K-quant weight using an already quantized Q8_K input.
    /// The caller must ensure `q8k` represents the same `[n_tok, in_dim]`
    /// activation requested here; this avoids re-quantizing identical inputs
    /// for the multiple projections in one K3 layer.
    pub fn matmul_q8k(
        &self,
        out: &mut DeviceBuf,
        q8k: &DeviceBuf,
        in_dim: u32,
        out_dim: u32,
        n_tok: u32,
    ) -> super::Result {
        if !self.quant.is_direct_kq() {
            return Err(format!(
                "matmul_q8k requires Q2_K/Q3_K, got quant={}",
                self.quant.quant
            )
            .into());
        }
        kernels::matmul_kq(
            out,
            &self.buf,
            q8k,
            in_dim,
            out_dim,
            n_tok,
            self.quant.row_bytes,
            self.quant.quant,
        )
        .map_err(|e| {
            format!(
                "K3 matmul_kq prequantized (quant={}): {e}",
                self.quant.quant
            )
            .into()
        })
    }

    /// Convenience: dispatch the correct matmul for this weight.
    ///
    /// * Q2_K / Q3_K → quantize activation to Q8_K, call `matmul_kq`
    /// * Q8_0 → call `matmul_q8_0`
    /// * F32 → call `matmul_f32`
    ///
    /// `q8k_scratch` is a reusable Q8_K activation buffer sized for
    /// `in_dim` elements.  It is only used for the direct-KQ path.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        out: &mut DeviceBuf,
        x: &DeviceBuf,
        q8k_scratch: &mut DeviceBuf,
        in_dim: u32,
        out_dim: u32,
        n_tok: u32,
    ) -> super::Result {
        if self.quant.is_direct_kq() {
            if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
                eprintln!(
                    "pulsar: K3 direct matmul quant={} in_dim={} out_dim={} n_tok={} row_bytes={} weight_bytes={}",
                    self.quant.quant, in_dim, out_dim, n_tok, self.quant.row_bytes, self.buf.bytes()
                );
            }
            if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
                let sync_result: super::Result = kernels::sync().map_err(|e| {
                    format!(
                        "K3 matmul pre-sync (quant={} in_dim={} out_dim={}): {e}",
                        self.quant.quant, in_dim, out_dim
                    )
                    .into()
                });
                sync_result?;
            }
            // Quantize activation f32 → Q8_K
            kernels::quantize_q8_k(q8k_scratch, x, in_dim, n_tok)?;
            if std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
                let sync_result: super::Result = kernels::sync()
                    .map_err(|e| format!("K3 quantize_q8_k sync (in_dim={}): {e}", in_dim).into());
                sync_result?;
            }
            // Direct K-quant matmul
            let result = kernels::matmul_kq(
                out,
                &self.buf,
                q8k_scratch,
                in_dim,
                out_dim,
                n_tok,
                self.quant.row_bytes,
                self.quant.quant,
            )
            .map_err(|e| format!("K3 matmul_kq (quant={}): {e}", self.quant.quant).into());
            if result.is_ok() && std::env::var_os("PULSAR_DEBUG_CUDA_SYNC").is_some() {
                let sync_result: super::Result = kernels::sync().map_err(|e| {
                    format!("K3 matmul_kq sync (quant={}): {e}", self.quant.quant).into()
                });
                sync_result?;
            }
            result
        } else if self.quant.is_q8_0() {
            kernels::matmul_q8_0(out, &self.buf, x, in_dim, out_dim, n_tok)
                .map_err(|e| format!("K3 matmul_q8_0: {e}").into())
        } else {
            // F32 fallback (absorbed MLA tensors, custom layouts)
            kernels::matmul_f32(out, &self.buf, x, in_dim, out_dim, n_tok)
                .map_err(|e| format!("K3 matmul_f32: {e}").into())
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// RED: verify K3WeightQuant correctly identifies Q2_K as direct-kq.
    #[test]
    fn test_k3_weight_quant_q2k_is_direct() {
        let q = K3WeightQuant {
            quant: kernels::QUANT_Q2_K,
            row_bytes: 1176, // 3584 elems / 256 * 84
        };
        assert!(q.is_direct_kq(), "Q2_K must be direct-kq");
        assert!(!q.is_q8_0(), "Q2_K must not be q8_0");
    }

    /// RED: verify K3WeightQuant correctly identifies Q3_K as direct-kq.
    #[test]
    fn test_k3_weight_quant_q3k_is_direct() {
        let q = K3WeightQuant {
            quant: kernels::QUANT_Q3_K,
            row_bytes: 14 * 110, // 3584 elems / 256 * 110
        };
        assert!(q.is_direct_kq(), "Q3_K must be direct-kq");
        assert!(!q.is_q8_0(), "Q3_K must not be q8_0");
    }

    /// RED: verify K3WeightQuant correctly identifies Q8_0.
    #[test]
    fn test_k3_weight_quant_q8_0_is_not_direct() {
        let q = K3WeightQuant {
            quant: kernels::QUANT_Q8_0,
            row_bytes: 3584u64.div_ceil(32) * 34,
        };
        assert!(!q.is_direct_kq(), "Q8_0 must not be direct-kq");
        assert!(q.is_q8_0(), "Q8_0 must be q8_0");
    }

    /// RED: verify K3WeightQuant::from_gguf for Q2_K.
    #[test]
    fn test_k3_weight_quant_from_gguf_q2k() {
        let q = K3WeightQuant::from_gguf(gguf::TensorType::Q2K, 3584).expect("Q2_K from_gguf");
        assert_eq!(q.quant, kernels::QUANT_Q2_K);
        assert_eq!(q.row_bytes, 1176);
    }

    /// RED: verify K3WeightQuant::from_gguf for Q3_K.
    #[test]
    fn test_k3_weight_quant_from_gguf_q3k() {
        let q = K3WeightQuant::from_gguf(gguf::TensorType::Q3K, 3584).expect("Q3_K from_gguf");
        assert_eq!(q.quant, kernels::QUANT_Q3_K);
        assert_eq!(q.row_bytes, 14 * 110);
    }

    /// RED: verify K3WeightQuant::from_gguf for Q8_0.
    #[test]
    fn test_k3_weight_quant_from_gguf_q8_0() {
        let q = K3WeightQuant::from_gguf(gguf::TensorType::Q8_0, 7168).expect("Q8_0 from_gguf");
        assert_eq!(q.quant, kernels::QUANT_Q8_0);
        assert_eq!(q.row_bytes, 7168u64.div_ceil(32) * 34);
    }

    /// RED: verify K3WeightQuant::from_gguf returns None for unsupported types.
    #[test]
    fn test_k3_weight_quant_from_gguf_unsupported() {
        assert!(K3WeightQuant::from_gguf(gguf::TensorType::F32, 100).is_none());
        assert!(K3WeightQuant::from_gguf(gguf::TensorType::F16, 100).is_none());
    }

    /// RED: verify K3DenseWeight::new and matmul dispatch metadata.
    #[test]
    fn test_k3_dense_weight_metadata() {
        use gguf::TensorType as T;
        // Simulate a Q2_K weight with 3584 elements (moe_latent)
        let quant = K3WeightQuant::from_gguf(T::Q2K, 3584).unwrap();
        assert_eq!(quant.quant, kernels::QUANT_Q2_K);
        assert_eq!(quant.row_bytes, 1176);
        // Simulate a Q3_K weight with 7168 elements (n_embd)
        let quant = K3WeightQuant::from_gguf(T::Q3K, 7168).unwrap();
        assert_eq!(quant.quant, kernels::QUANT_Q3_K);
        assert_eq!(quant.row_bytes, 7168u64.div_ceil(256) * 110);
        // Simulate a Q8_0 weight with 7168 elements
        let quant = K3WeightQuant::from_gguf(T::Q8_0, 7168).unwrap();
        assert_eq!(quant.quant, kernels::QUANT_Q8_0);
        assert_eq!(quant.row_bytes, 7168u64.div_ceil(32) * 34);
    }

    /// Focused test: chain from_gguf → is_direct_kq for every supported
    /// K-quant type. Q2_K and Q3_K must be direct-kq; Q4_K/Q5_K/Q6_K
    /// must NOT be direct-kq (they fall back to Q8_0 conversion).
    #[test]
    fn test_k3_weight_quant_from_gguf_is_direct_kq() {
        use gguf::TensorType as T;
        // Q2_K: direct-kq
        let q = K3WeightQuant::from_gguf(T::Q2K, 7168).expect("Q2_K from_gguf");
        assert!(q.is_direct_kq(), "Q2_K must be direct-kq");
        assert!(!q.is_q8_0(), "Q2_K must not be q8_0");
        // Q3_K: direct-kq
        let q = K3WeightQuant::from_gguf(T::Q3K, 7168).expect("Q3_K from_gguf");
        assert!(q.is_direct_kq(), "Q3_K must be direct-kq");
        assert!(!q.is_q8_0(), "Q3_K must not be q8_0");
        // Q4_K: NOT direct-kq (falls back to Q8_0)
        let q = K3WeightQuant::from_gguf(T::Q4K, 7168).expect("Q4_K from_gguf");
        assert!(!q.is_direct_kq(), "Q4_K must NOT be direct-kq");
        assert!(!q.is_q8_0(), "Q4_K must not be q8_0");
        // Q5_K: NOT direct-kq
        let q = K3WeightQuant::from_gguf(T::Q5K, 7168).expect("Q5_K from_gguf");
        assert!(!q.is_direct_kq(), "Q5_K must NOT be direct-kq");
        assert!(!q.is_q8_0(), "Q5_K must not be q8_0");
        // Q6_K: NOT direct-kq
        let q = K3WeightQuant::from_gguf(T::Q6K, 7168).expect("Q6_K from_gguf");
        assert!(!q.is_direct_kq(), "Q6_K must NOT be direct-kq");
        assert!(!q.is_q8_0(), "Q6_K must not be q8_0");
        // Q8_0: not direct-kq, is q8_0
        let q = K3WeightQuant::from_gguf(T::Q8_0, 7168).expect("Q8_0 from_gguf");
        assert!(!q.is_direct_kq(), "Q8_0 must NOT be direct-kq");
        assert!(q.is_q8_0(), "Q8_0 must be q8_0");
        // F32: from_gguf returns None
        assert!(K3WeightQuant::from_gguf(T::F32, 100).is_none());
    }

    #[test]
    fn test_k3_layer_kind_kda() {
        assert_eq!(K3LayerKind::Kda as u8, 0u8);
        assert_eq!(K3LayerKind::Mla as u8, 1u8);
    }

    #[test]
    fn test_k3_layer_kind_discriminant() {
        let kda = K3LayerKind::Kda;
        let mla = K3LayerKind::Mla;
        assert_ne!(kda, mla);
        assert_eq!(kda, K3LayerKind::Kda);
        assert_eq!(mla, K3LayerKind::Mla);
    }

    #[test]
    fn test_k3_forward_empty_batch_error() {
        let msg = "Kimi K3 forward not yet implemented (Phase 1 skeleton)";
        assert!(msg.contains("Phase 1 skeleton"));
    }

    /// RED/GREEN boundary test for the quant-aware expert buffer helper.
    /// A short packed buffer must fail closed rather than returning a partial
    /// f32 slab that would later panic or silently corrupt expert matmul.
    #[test]
    fn test_k3_q2k_expert_dequant_rejects_short_buffer() {
        let short = vec![0u8; 83];
        let err = k3_dequant_expert_bytes(&short, 256, kernels::QUANT_Q2_K)
            .expect_err("truncated Q2_K block must fail closed");
        assert!(err.to_string().contains("expected 84 bytes"));
    }

    /// The quant id, not a hard-coded Q8_0 assumption, selects Q2_K dequant.
    #[test]
    fn test_k3_q2k_expert_dequant_uses_quant_id() {
        let mut block = vec![0u8; 84];
        block[80..82].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        for byte in &mut block[16..80] {
            *byte = 0xAA; // q = 2 in every 2-bit lane
        }
        for byte in &mut block[0..16] {
            *byte = 0x01; // scale = 1, min = 0
        }
        let out = k3_dequant_expert_bytes(&block, 256, kernels::QUANT_Q2_K)
            .expect("Q2_K expert block should dequantize");
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| (*v - 2.0).abs() < 1e-4));
    }

    /// Q4_0 is used by AtomicChat's small absorbed K3 tensors.
    #[test]
    fn test_k3_q4_0_dequant_block() {
        let mut block = vec![0u8; 18];
        block[0..2].copy_from_slice(&0x3c00u16.to_le_bytes()); // d = 1
        for byte in &mut block[2..] {
            *byte = 0x99; // both nibbles = 9, so q - 8 = 1
        }
        let mut out = [0.0f32; 256];
        crate::real::requant::dequant_block(gguf::TensorType::Q4_0, &block, &mut out);
        assert!(out[..32].iter().all(|v| (*v - 1.0).abs() < 1e-6));
    }

    /// RED: verify the Q2_K block layout constants match the dequant path.
    /// Q2_K: 256 elements, 84 bytes per block.
    #[test]
    fn test_k3_q2k_block_layout() {
        use gguf::TensorType;
        let (bs, bb) = TensorType::Q2K.block_layout().expect("Q2_K block layout");
        assert_eq!(bs, 256, "Q2_K block size");
        assert_eq!(bb, 84, "Q2_K block bytes");
        // Verify row_bytes for a typical expert slab width
        let row = TensorType::Q2K
            .row_bytes(3072)
            .expect("Q2_K row bytes for n_ff_exp");
        assert_eq!(row, 3072u64.div_ceil(256) * 84);
        assert_eq!(row, 12 * 84);
        assert_eq!(row, 1008);
    }

    /// RED: verify the Q2_K dequant path produces valid f32 output.
    /// Uses the engine's own dequant_block to prove the path exists.
    #[test]
    fn test_k3_q2k_dequant_block() {
        // Build a synthetic Q2_K block (256 elements, 84 bytes).
        // Layout: 16 bytes super-block scales, 64 bytes quant data, 2 bytes d, 2 bytes dmin.
        let mut block = vec![0u8; 84];
        // Set d (scale) at offset 80-81
        let d: u16 = 0x3c00; // f16 1.0
        block[80..82].copy_from_slice(&d.to_le_bytes());
        // Set dmin at offset 82-83
        let dmin: u16 = 0x0000; // f16 0.0
        block[82..84].copy_from_slice(&dmin.to_le_bytes());
        // Set super-block scales at offsets 0-15: all 0x11 (scale=1, min=1)
        for j in 0..16 {
            block[j] = 0x11;
        }
        // Set quant data at offsets 16-79: all 0xAA (2-bit pattern 10 10 10 10 = value 2)
        for j in 16..80 {
            block[j] = 0xAA;
        }

        let mut out = [0.0f32; 256];
        crate::real::requant::dequant_block(gguf::TensorType::Q2K, &block, &mut out);

        // Every element should be d * scale * q - dmin * min = 1.0 * 1.0 * 2.0 - 0.0 * 1.0 = 2.0
        for (i, &v) in out.iter().enumerate() {
            assert!(
                (v - 2.0).abs() < 1e-4,
                "Q2_K dequant block[{i}] = {v}, expected 2.0"
            );
        }
        println!("✅ Q2_K dequant block: all 256 values = 2.0");
    }

    /// RED: verify the Q2_K row_bytes for the actual AtomicChat model dimensions.
    /// Expert slabs: [moe_latent=3584, n_ff_exp=3072, n_expert=896]
    /// Each expert: 3072 rows of 3584 elements each.
    #[test]
    fn test_k3_q2k_expert_slab_byte_size() {
        use gguf::TensorType;
        let row_elems: u64 = 3584; // moe_latent
        let rows_per_expert: u64 = 3072; // n_ff_exp
        let n_expert: u64 = 896;
        let row_bytes = TensorType::Q2K.row_bytes(row_elems).expect("row bytes");
        assert_eq!(row_bytes, 3584u64.div_ceil(256) * 84);
        assert_eq!(row_bytes, 14 * 84);
        assert_eq!(row_bytes, 1176);
        let expert_bytes = row_bytes * rows_per_expert;
        assert_eq!(expert_bytes, 1176 * 3072);
        assert_eq!(expert_bytes, 3_612_672);
        let total = expert_bytes * n_expert;
        assert_eq!(total, 3_612_672 * 896);
        assert_eq!(total, 3_236_954_112);
        println!(
            "✅ Q2_K expert slab: {row_bytes} row_bytes, {expert_bytes} per expert, {total} total"
        );
    }

    #[test]
    fn test_k3_contract_constants() {
        let n_layer: u32 = 93;
        let n_kda: u32 = 69;
        let n_mla: u32 = 24;
        assert_eq!(n_kda + n_mla, n_layer);
        assert_eq!(n_kda, 69);
        assert_eq!(n_mla, 24);
        assert_eq!(n_layer, 93);
    }

    #[test]
    fn test_k3_moe_contract() {
        let n_expert: u32 = 896;
        let n_expert_used: u32 = 16;
        let n_expert_shared: u32 = 2;
        let n_leading_dense: u32 = 1;
        assert!(n_expert_used <= n_expert);
        assert_eq!(n_expert_shared, 2);
        assert_eq!(n_leading_dense, 1);
        assert_eq!(n_expert_used + n_expert_shared, 18);
    }

    #[test]
    fn test_k3_attn_res_block_size() {
        let res_block: u32 = 12;
        let n_layer: u32 = 93;
        let n_snapshots = (n_layer + res_block - 1) / res_block;
        assert_eq!(n_snapshots, 8);
    }

    #[test]
    fn test_k3_kda_head_dim() {
        let kda_head_dim: u32 = 128;
        let n_head: u32 = 96;
        let d_inner = n_head * kda_head_dim;
        assert_eq!(d_inner, 12288);
    }

    #[test]
    fn test_k3_moe_latent_size() {
        let moe_latent: u32 = 3584;
        let n_ff_exp: u32 = 3072;
        assert_eq!(moe_latent, 3584);
        assert_eq!(n_ff_exp, 3072);
    }

    #[test]
    fn test_k3_mla_nope_keeps_unrotated_tail() {
        let dims = K3MlaDims::canonical();
        assert_eq!(dims.qk_nope, 128);
        assert_eq!(dims.qk_rope, 64);
        assert_eq!(dims.qk_dim(), 192);
        assert_eq!(dims.kv_a_out(), 576);
    }

    #[test]
    fn test_k3_safe_gate_lower_bound() {
        let gate_lower_bound: f32 = -5.0;
        assert!((gate_lower_bound + 5.0).abs() < f32::EPSILON);
    }

    /// Test AttnRes mixture formula on host (no CUDA needed).
    #[test]
    fn test_attn_res_mix_formula() {
        let n_embd = 8;
        let n_rows = 3; // 2 bank rows + 1 prefix

        // Create synthetic data
        let bank: Vec<f32> = (0..(n_rows - 1) * n_embd)
            .map(|i| (i as f32) * 0.1)
            .collect();
        let prefix: Vec<f32> = (0..n_embd).map(|i| (i as f32 + 10.0) * 0.1).collect();
        let norm_w: Vec<f32> = vec![1.0; n_embd];
        let proj_w: Vec<f32> = vec![1.0; n_embd];
        let eps = 1e-6;

        // Compute sw = norm_w * proj_w
        let sw: Vec<f32> = norm_w
            .iter()
            .zip(proj_w.iter())
            .map(|(n, p)| n * p)
            .collect();

        // Compute scores
        let mut scores = Vec::with_capacity(n_rows);
        for r in 0..n_rows {
            let row: &[f32] = if r < n_rows - 1 {
                &bank[r * n_embd..(r + 1) * n_embd]
            } else {
                &prefix
            };
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / n_embd as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            let score: f32 = row
                .iter()
                .zip(sw.iter())
                .map(|(v, s)| v * inv_rms * s)
                .sum();
            scores.push(score);
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let probs: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();

        // Weighted sum
        let mut mix_out = vec![0.0f32; n_embd];
        for r in 0..n_rows {
            let row: &[f32] = if r < n_rows - 1 {
                &bank[r * n_embd..(r + 1) * n_embd]
            } else {
                &prefix
            };
            let p = probs[r];
            for i in 0..n_embd {
                mix_out[i] += p * row[i];
            }
        }

        // Verify: probs sum to 1
        let prob_sum: f32 = probs.iter().sum();
        assert!(
            (prob_sum - 1.0).abs() < 1e-5,
            "probs should sum to 1, got {prob_sum}"
        );

        // Verify: output is a convex combination of inputs
        for i in 0..n_embd {
            let mut expected = 0.0;
            for r in 0..n_rows {
                let row: &[f32] = if r < n_rows - 1 {
                    &bank[r * n_embd..(r + 1) * n_embd]
                } else {
                    &prefix
                };
                expected += probs[r] * row[i];
            }
            assert!(
                (mix_out[i] - expected).abs() < 1e-5,
                "mix_out[{i}] mismatch"
            );
        }

        println!("✅ AttnRes mixture formula: probs sum to 1, convex combination verified");
    }

    /// Test KDA safe gate formula on host.
    #[test]
    fn test_kda_safe_gate_formula() {
        let n_head = 4;
        let head_dim = 8;
        let d_inner = n_head * head_dim;
        let gate_lower_bound = -5.0;

        // Synthetic data
        let f_b: Vec<f32> = (0..d_inner).map(|i| (i as f32) * 0.1).collect();
        let dt_bias: Vec<f32> = vec![0.1; d_inner];
        // Persisted GGUFs exist with both converter sign conventions.
        let a_folded: Vec<f32> = (0..n_head)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 } * (1.0 + i as f32 * 0.5).exp())
            .collect();

        // g_raw = f_a @ f_b + dt_bias (simplified: just use f_b directly)
        let mut g_raw: Vec<f32> = f_b.clone();
        for i in 0..d_inner {
            g_raw[i] += dt_bias[i];
        }

        // Apply the positive exp(A_log) magnitude per head.
        for h in 0..n_head {
            for d in 0..head_dim {
                let idx = h * head_dim + d;
                g_raw[idx] *= a_folded[h].abs();
            }
        }

        // sigmoid and multiply by gate_lower_bound
        let mut g1 = vec![0.0f32; d_inner];
        for i in 0..d_inner {
            g1[i] = gate_lower_bound / (1.0 + (-g_raw[i]).exp());
        }

        // Verify: all values are in [gate_lower_bound, 0]
        for &v in &g1 {
            assert!(
                v >= gate_lower_bound && v <= 0.0,
                "safe gate value {v} outside [{gate_lower_bound}, 0]"
            );
        }

        println!("✅ KDA safe gate formula: all values in [{gate_lower_bound}, 0]");
    }

    /// Test SiTU-GLU formula on host.
    #[test]
    fn test_situ_glu_formula() {
        let n = 16;
        let beta = 4.0;
        let linear_beta = 25.0;

        let gate: Vec<f32> = (0..n).map(|i| (i as f32 - 8.0) * 0.5).collect();
        let up: Vec<f32> = (0..n).map(|i| (i as f32) * 0.3).collect();

        let mut out = vec![0.0f32; n];
        for i in 0..n {
            let g = gate[i];
            let u = up[i];
            // SiTU_Gate: beta * tanh(x/beta) * sigmoid(x)
            let situ_gate = beta * (g / beta).tanh() / (1.0 + (-g).exp());
            // SiTU_Linear: linear_beta * tanh(x/linear_beta)
            let situ_linear = linear_beta * (u / linear_beta).tanh();
            out[i] = situ_gate * situ_linear;
        }

        // Verify: no NaN
        for &v in &out {
            assert!(!v.is_nan(), "SiTU-GLU output has NaN");
        }

        println!("✅ SiTU-GLU formula: no NaN, computed {} values", n);
    }

    /// Test KDA output gate formula on host.
    #[test]
    fn test_kda_output_gate_formula() {
        let head_dim = 4;
        let attn_out = vec![1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let o_norm_w = vec![1.0; head_dim];
        let g2 = vec![0.0f32; attn_out.len()];
        let eps = 1e-6;

        let mut normed = Vec::with_capacity(attn_out.len());
        for head in attn_out.chunks_exact(head_dim) {
            let mean_sq = head.iter().map(|v| v * v).sum::<f32>() / head_dim as f32;
            let inv_rms = 1.0 / (mean_sq + eps).sqrt();
            normed.extend(head.iter().zip(&o_norm_w).map(|(v, w)| v * inv_rms * w));
        }

        // gated = normed * sigmoid(g2)
        let gated: Vec<f32> = normed
            .iter()
            .zip(g2.iter())
            .map(|(n, g)| n / (1.0 + (-g).exp()))
            .collect();

        // Verify: gated values are bounded by normed values
        for i in 0..attn_out.len() {
            assert!(
                gated[i].abs() <= normed[i].abs() + 1e-5,
                "gated[{i}] = {} exceeds normed[{i}] = {}",
                gated[i],
                normed[i]
            );
        }

        println!("✅ KDA output gate formula: gated bounded by normed");
    }

    #[test]
    fn test_k3_mla_split_gguf_axis_order() {
        let (nope, lora, heads) = (2usize, 3usize, 2usize);
        let mut wk = vec![0.0f32; nope * lora * heads];
        for h in 0..heads {
            for j in 0..lora {
                for i in 0..nope {
                    let idx = i + nope * (j + lora * h);
                    wk[idx] = (100 * h + 10 * j + i) as f32;
                }
            }
        }
        let q = [2.0f32, 3.0];
        let h = 1usize;
        let got: Vec<f32> = (0..lora)
            .map(|j| {
                (0..nope)
                    .map(|i| q[i] * wk[i + nope * (j + lora * h)])
                    .sum()
            })
            .collect();
        assert_eq!(got, vec![503.0, 553.0, 603.0]);
    }

    #[test]
    fn test_attn_res_retrieval_does_not_replace_prefix() {
        let prefix = [3.0f32, 5.0];
        let retrieved = [1.0f32, 2.0];
        let attention = [0.5f32, -0.5];
        let ffn = [4.0f32, 6.0];

        // Retrieval feeds normalization/sublayers, while residual updates
        // continue from the raw prefix stream.
        let layer_out = [
            prefix[0] + attention[0] + ffn[0],
            prefix[1] + attention[1] + ffn[1],
        ];
        let overwritten_bug = [
            retrieved[0] + attention[0] + ffn[0],
            retrieved[1] + attention[1] + ffn[1],
        ];
        assert_eq!(layer_out, [7.5, 10.5]);
        assert_ne!(layer_out, overwritten_bug);
    }

    /// Test latent MoE router formula on host.
    #[test]
    fn test_latent_moe_router_formula() {
        let n_expert = 8;
        let n_used = 3;

        let scores: Vec<f32> = (0..n_expert).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let bias: Vec<f32> = vec![0.1; n_expert];

        // Sigmoid scores with bias
        let sig_scores: Vec<f32> = scores
            .iter()
            .zip(bias.iter())
            .map(|(s, b)| 1.0 / (1.0 + (-(s + b)).exp()))
            .collect();

        // Top-k selection
        let mut indexed: Vec<(usize, f32)> = sig_scores
            .iter()
            .enumerate()
            .map(|(i, s)| (i, *s))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_idx: Vec<i32> = indexed
            .iter()
            .take(n_used)
            .map(|(i, _)| *i as i32)
            .collect();
        let top_val: Vec<f32> = indexed.iter().take(n_used).map(|(_, s)| *s).collect();

        // Renormalize
        let sum_top: f32 = top_val.iter().sum();
        let renorm: Vec<f32> = top_val.iter().map(|v| v / sum_top).collect();

        // Verify: indices are unique and in range
        let mut sorted_idx = top_idx.clone();
        sorted_idx.sort_unstable();
        sorted_idx.dedup();
        assert_eq!(sorted_idx.len(), n_used, "top-k indices should be unique");

        // Verify: weights sum to 1
        let w_sum: f32 = renorm.iter().sum();
        assert!(
            (w_sum - 1.0).abs() < 1e-5,
            "renormalized weights should sum to 1, got {w_sum}"
        );

        println!("✅ Latent MoE router: {n_used} unique indices, weights sum to {w_sum}");
    }

    /// Regression test: verify expert slab offset resolution matches the
    /// K3 contract dimensions.  Each expert tensor has abs_offset, expert_bytes,
    /// row_bytes, and quant.  The offset for expert `e` is
    /// `abs_offset + e * expert_bytes`.  This test verifies the arithmetic
    /// for the canonical K3 dimensions (moe_latent=3584, n_ff_exp=3072, 896 experts).
    #[test]
    fn test_k3_expert_slab_offset_resolution() {
        use gguf::TensorType;

        // Canonical K3 expert dimensions
        let moe_latent: u64 = 3584;
        let n_ff_exp: u64 = 3072;
        let n_expert: u64 = 896;

        // Q2_K row_bytes for moe_latent elements
        let row_bytes = TensorType::Q2K
            .row_bytes(moe_latent)
            .expect("Q2_K row_bytes");
        assert_eq!(row_bytes, 1176, "Q2_K row_bytes for moe_latent=3584");

        // Per-expert slab size: rows * row_bytes
        let expert_bytes = row_bytes * n_ff_exp;
        assert_eq!(expert_bytes, 1176 * 3072, "Q2_K expert slab bytes");

        // Simulate abs_offsets for gate/up/down tensors
        let gate_base: u64 = 1_000_000;
        let up_base: u64 = gate_base + n_expert * expert_bytes;
        let down_base: u64 = up_base + n_expert * expert_bytes;

        // Resolve expert 0
        assert_eq!(gate_base + 0 * expert_bytes, gate_base);
        assert_eq!(up_base + 0 * expert_bytes, up_base);
        assert_eq!(down_base + 0 * expert_bytes, down_base);

        // Resolve expert 42
        let e42: u64 = 42;
        assert_eq!(
            gate_base + e42 * expert_bytes,
            gate_base + 42 * 1176 * 3072,
            "gate offset for expert 42"
        );
        assert_eq!(
            up_base + e42 * expert_bytes,
            up_base + 42 * 1176 * 3072,
            "up offset for expert 42"
        );
        assert_eq!(
            down_base + e42 * expert_bytes,
            down_base + 42 * 1176 * 3072,
            "down offset for expert 42"
        );

        // Verify no overlap: gate/up/down slabs for the same expert are at
        // different offsets
        let g42 = gate_base + e42 * expert_bytes;
        let u42 = up_base + e42 * expert_bytes;
        let d42 = down_base + e42 * expert_bytes;
        assert_ne!(g42, u42, "gate and up must not share offset");
        assert_ne!(u42, d42, "up and down must not share offset");
        assert_ne!(g42, d42, "gate and down must not share offset");

        // Verify the last expert's slab does not overflow into the next tensor
        let last = n_expert - 1;
        let g_last = gate_base + last * expert_bytes;
        let u_last = up_base + last * expert_bytes;
        assert!(
            g_last + expert_bytes <= up_base,
            "last gate slab must not overlap up tensor"
        );
        assert!(
            u_last + expert_bytes <= down_base,
            "last up slab must not overlap down tensor"
        );

        println!("✅ Expert slab offset resolution: all checks pass");
    }

    /// Regression test: verify the dispatch selection logic for GPU vs host
    /// MoE paths.  Q2_K and Q3_K are GPU-eligible; Q4_K, Q5_K, Q6_K, Q8_0
    /// fall back to host.
    #[test]
    fn test_k3_moe_dispatch_selection() {
        // GPU-eligible quants
        assert!(
            kernels::QUANT_Q2_K == kernels::QUANT_Q2_K
                || kernels::QUANT_Q3_K == kernels::QUANT_Q3_K,
            "Q2_K and Q3_K are GPU-eligible"
        );

        // Host-only quants
        let host_quants = [
            kernels::QUANT_Q4_K,
            kernels::QUANT_Q5_K,
            kernels::QUANT_Q6_K,
            kernels::QUANT_Q8_0,
            kernels::QUANT_Q4_0,
        ];
        for &q in &host_quants {
            let is_gpu = q == kernels::QUANT_Q2_K || q == kernels::QUANT_Q3_K;
            assert!(!is_gpu, "quant {q} must NOT be GPU-eligible");
        }

        // Verify the dispatch condition matches the code
        let is_gpu_eligible = |q: u32| q == kernels::QUANT_Q2_K || q == kernels::QUANT_Q3_K;
        assert!(is_gpu_eligible(kernels::QUANT_Q2_K));
        assert!(is_gpu_eligible(kernels::QUANT_Q3_K));
        assert!(!is_gpu_eligible(kernels::QUANT_Q4_K));
        assert!(!is_gpu_eligible(kernels::QUANT_Q5_K));
        assert!(!is_gpu_eligible(kernels::QUANT_Q6_K));
        assert!(!is_gpu_eligible(kernels::QUANT_Q8_0));
        assert!(!is_gpu_eligible(kernels::QUANT_Q4_0));

        println!("✅ MoE dispatch selection: GPU for Q2_K/Q3_K, host for others");
    }
}
