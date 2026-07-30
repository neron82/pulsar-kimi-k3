# Kimi K3 GGUF Contract

> **Canonical reference:** AtomicBot fork at `/tmp/atomic-llama-cpp-turboquant-kimi`, commit `b2f13d7be`
> **HF config:** `moonshotai/Kimi-K3` (fetched 2026-07-28)
> **Target repo:** `/home/neron/projects/pulsar-kimi-k3`
>
> Every claim below is verified against the actual source code and HF config. No guessing.

---

## 1. Architecture Strings

### 1.1 GGUF `general.architecture` value

| Context | String | Source |
|---------|--------|--------|
| C++ enum → name | `"kimi-k3"` | `llama-arch.cpp:144` (`LLM_ARCH_KIMI_K3 → "kimi-k3"`) |
| Python enum → name | `"kimi-k3"` | `constants.py:552,1150` (`MODEL_ARCH.KIMI_K3 → "kimi-k3"`) |
| Converter HF arch → module | `"kimi_k3"` | `conversion/__init__.py:124` (`"KimiK3ForConditionalGeneration" → "kimi_k3"`) |

**The GGUF file MUST have `general.architecture = "kimi-k3"`.**

### 1.2 HuggingFace architecture name

| HF `architectures[]` | Converter class | Source |
|----------------------|----------------|--------|
| `"KimiK3ForConditionalGeneration"` | `KimiK3Model` (inherits `KimiLinearModel` → `TextModel`) | `kimi_k3.py:14-15` |

### 1.3 Related but distinct architecture

| Enum | String | Purpose |
|------|--------|---------|
| `LLM_ARCH_KIMI_LINEAR` | `"kimi-linear"` | Kimi Linear (predecessor, no AttnRes, no latent MoE, low-rank KDA gate) |

---

## 2. Metadata Key Namespaces and Types

All K3-specific keys use the `{arch}` = `"kimi-k3"` prefix. The C++ `llm_kv` enum and Python `Keys` class define the canonical names.

### 2.1 K3-Specific Keys

| GGUF Key | C++ Enum | Python Constant | Type | Source (C++ KV name) |
|----------|----------|-----------------|------|---------------------|
| `kimi-k3.kda.head_dim` | `LLM_KV_KDA_HEAD_DIM` | `Keys.KDA.HEAD_DIM` | `uint32` | `llama-arch.cpp:306` |
| `kimi-k3.kda.gate_lower_bound` | `LLM_KV_KDA_GATE_LOWER_BOUND` | `Keys.KDA.GATE_LOWER_BOUND` | `float32` | `llama-arch.cpp:307` |
| `kimi-k3.situ_beta` | `LLM_KV_SITU_BETA` | `Keys.LLM.SITU_BETA` | `float32` | `llama-arch.cpp:309` |
| `kimi-k3.situ_linear_beta` | `LLM_KV_SITU_LINEAR_BETA` | `Keys.LLM.SITU_LINEAR_BETA` | `float32` | `llama-arch.cpp:310` |
| `kimi-k3.attn_res_block_size` | `LLM_KV_ATTN_RES_BLOCK_SIZE` | `Keys.LLM.ATTN_RES_BLOCK_SIZE` | `uint32` | `llama-arch.cpp:311` |
| `kimi-k3.moe_latent_size` | `LLM_KV_MOE_LATENT_SIZE` | `Keys.LLM.MOE_LATENT_SIZE` | `uint32` | `llama-arch.cpp:205` |
| `kimi-k3.ssm.conv_kernel` | `LLM_KV_SSM_CONV_KERNEL` | `Keys.SSM.CONV_KERNEL` | `uint32` | `llama-arch.cpp:299` |

### 2.2 Standard Keys Used by K3

| GGUF Key | C++ Enum | Type | Source |
|----------|----------|------|--------|
| `kimi-k3.attention.layer_norm_rms_epsilon` | `LLM_KV_ATTENTION_LAYERNORM_RMS_EPS` | `float32` | `llama-arch.cpp:241` |
| `kimi-k3.attention.key_length_mla` | `LLM_KV_ATTENTION_KEY_LENGTH_MLA` | `uint32` | `llama-arch.cpp:260` |
| `kimi-k3.attention.value_length_mla` | `LLM_KV_ATTENTION_VALUE_LENGTH_MLA` | `uint32` | `llama-arch.cpp:261` |
| `kimi-k3.attention.kv_lora_rank` | `LLM_KV_ATTENTION_KV_LORA_RANK` | `uint32` | `llama-arch.cpp:246` |
| `kimi-k3.attention.q_lora_rank` | `LLM_KV_ATTENTION_Q_LORA_RANK` | `uint32` | `llama-arch.cpp:245` |
| `kimi-k3.attention.head_count` | `LLM_KV_ATTENTION_HEAD_COUNT` | `uint32` | `llama-arch.cpp:234` |
| `kimi-k3.attention.head_count_kv` | `LLM_KV_ATTENTION_HEAD_COUNT_KV` | `array[uint32]` | `llama-arch.cpp:235` |
| `kimi-k3.rope.dimension_count` | `LLM_KV_ROPE_DIMENSION_COUNT` | `uint32` | `llama-arch.cpp:282` |
| `kimi-k3.expert_feed_forward_length` | `LLM_KV_EXPERT_FEED_FORWARD_LENGTH` | `uint32` | `llama-arch.cpp:192` |
| `kimi-k3.expert_shared_count` | `LLM_KV_EXPERT_SHARED_COUNT` | `uint32` | `llama-arch.cpp:201` |
| `kimi-k3.leading_dense_block_count` | `LLM_KV_LEADING_DENSE_BLOCK_COUNT` | `uint32` | `llama-arch.cpp:190` |
| `kimi-k3.expert_weights_scale` | `LLM_KV_EXPERT_WEIGHTS_SCALE` | `float32` | `llama-arch.cpp:204` |
| `kimi-k3.expert_gating_func` | `LLM_KV_EXPERT_GATING_FUNC` | `uint32` | `llama-arch.cpp:206` |
| `kimi-k3.expert_count` | `LLM_KV_EXPERT_COUNT` | `uint32` | `llama-arch.cpp:199` |
| `kimi-k3.expert_used_count` | `LLM_KV_EXPERT_USED_COUNT` | `uint32` | `llama-arch.cpp:200` |
| `kimi-k3.hidden_activation` | `LLM_KV_HIDDEN_ACT` | `string` | `llama-arch.cpp:214` |

### 2.3 K3-Specific HParam Fields (C++ `llama_hparams`)

| Field | Type | Default | Set From | Source |
|-------|------|---------|----------|--------|
| `n_embd_head_kda` | `uint32` | `0` | `kimi-k3.kda.head_dim` | `llama-hparams.h:163` |
| `kda_gate_lower_bound` | `float` | `0.0f` | `kimi-k3.kda.gate_lower_bound` | `llama-hparams.h:166` |
| `situ_beta` | `float` | `0.0f` | `kimi-k3.situ_beta` | `llama-hparams.h:167` |
| `situ_linear_beta` | `float` | `0.0f` | `kimi-k3.situ_linear_beta` | `llama-hparams.h:168` |
| `attn_res_block_size` | `uint32` | `0` | `kimi-k3.attn_res_block_size` | `llama-hparams.h:169` |
| `moe_latent_size` | `uint32` | `0` | `kimi-k3.moe_latent_size` | `llama-hparams.h:103` |
| `ssm_d_conv` | `uint32` | `0` | `kimi-k3.ssm.conv_kernel` | `llama-hparams.h:156` |
| `expert_weights_norm` | `bool` | `false` | Hard-coded `true` for K3 | `kimi-k3.cpp:48` |

---

## 3. Per-Layer Discriminator Semantics and Indexing

### 3.1 The `attention.head_count_kv` Array

K3 uses `attention.head_count_kv` as an **array** of length `block_count` (93). Each element discriminates the layer type:

| Value | Layer Type | C++ `is_recr(i)` | Description |
|-------|-----------|-------------------|-------------|
| `0` | **KDA** (recurrent) | `true` | Kimi Delta Attention with safe gate, full-rank output gate |
| `>0` | **Gated MLA** (full attention) | `false` | NoPE gated Multi-head Latent Attention with output gate |

**Source:** `kimi-k3.cpp:35-37`:
```cpp
for (uint32_t i = 0; i < hparams.n_layer(); ++i) {
    hparams.is_recr_impl[i] = hparams.n_head_kv(i) == 0;
}
```

### 3.2 Layer Assignment from HF Config

From `moonshotai/Kimi-K3` `text_config.linear_attn_config`:

- **`full_attn_layers`** (1-indexed): `[4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 68, 72, 76, 80, 84, 88, 92, 93]` → **24 MLA layers**
- **`kda_layers`** (1-indexed): All other layers 1–93 → **69 KDA layers**

**0-indexed mapping:** Layer `i` is MLA if `(i+1) ∈ full_attn_layers`, otherwise KDA.

The converter (`kimi_linear.py:89-97`) builds the `head_count_kv` array:
```python
for il in range(num_hidden_layers):
    if il + 1 in full_attn_layers:
        _num_kv_heads.append(num_key_value_heads)  # 1 (MQA)
    else:
        _num_kv_heads.append(0)  # KDA
```

### 3.3 Layer 0 Special Case

Layer 0 (0-indexed) is **KDA** (not in `full_attn_layers`). Layer 92 (0-indexed, value 93 in 1-indexed) is **MLA**.

### 3.4 Dense FFN vs MoE Discriminator

`leading_dense_block_count` (`first_k_dense_replace` in HF config) = **1**. Layers `i < 1` (i.e., layer 0 only) use dense FFN; layers `i >= 1` use Stable LatentMoE.

---

## 4. Exact Tensor Names, Shapes, and Order

### 4.1 GGUF Tensor Name Format

All per-layer tensors use the `blk.%d.` prefix where `%d` is the 0-indexed layer number. Global tensors have no prefix.

### 4.2 Global Tensors

| GGUF Name | C++ Tensor Enum | Shape | Source |
|-----------|-----------------|-------|--------|
| `token_embd` | `LLM_TENSOR_TOKEN_EMBD` | `[n_embd, n_vocab]` = `[7168, 163840]` | `kimi-k3.cpp:58` |
| `output_norm` | `LLM_TENSOR_OUTPUT_NORM` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:61` |
| `output` | `LLM_TENSOR_OUTPUT` | `[n_embd, n_vocab]` = `[7168, 163840]` | `kimi-k3.cpp:62` |
| `output_res_norm` | `LLM_TENSOR_OUTPUT_RES_NORM` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:65` |
| `output_res_proj` | `LLM_TENSOR_OUTPUT_RES_PROJ` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:66` |

### 4.3 Per-Layer Tensors (All Layers)

| GGUF Name | C++ Tensor Enum | Shape | Source |
|-----------|-----------------|-------|--------|
| `blk.%d.attn_norm` | `LLM_TENSOR_ATTN_NORM` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:71` |
| `blk.%d.attn_res_norm` | `LLM_TENSOR_ATTN_RES_NORM` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:74` |
| `blk.%d.attn_res_proj` | `LLM_TENSOR_ATTN_RES_PROJ` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:75` |
| `blk.%d.ffn_res_norm` | `LLM_TENSOR_FFN_RES_NORM` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:76` |
| `blk.%d.ffn_res_proj` | `LLM_TENSOR_FFN_RES_PROJ` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:77` |
| `blk.%d.ffn_norm` | `LLM_TENSOR_FFN_NORM` | `[n_embd]` = `[7168]` | `kimi-k3.cpp:156` |

### 4.4 KDA Layer Tensors (69 layers: `is_recr(i) == true`)

**Dimensions:**
- `n_embd` = 7168, `n_head` = 96, `n_embd_head_kda` = 128
- `d_inner` = `n_head * n_embd_head_kda` = 96 × 128 = 12288
- `ssm_d_conv` = 4

| GGUF Name | C++ Tensor Enum | Shape | Notes | Source |
|-----------|-----------------|-------|-------|--------|
| `blk.%d.ssm_conv1d_q` | `LLM_TENSOR_SSM_CONV1D_Q` | `[4, 1, 12288, 1]` or `[4, 1, 12288]` | 4D preferred, 3D fallback | `kimi-k3.cpp:87-90` |
| `blk.%d.ssm_conv1d_k` | `LLM_TENSOR_SSM_CONV1D_K` | `[4, 1, 12288, 1]` or `[4, 1, 12288]` | Same | `kimi-k3.cpp:91-94` |
| `blk.%d.ssm_conv1d_v` | `LLM_TENSOR_SSM_CONV1D_V` | `[4, 1, 12288, 1]` or `[4, 1, 12288]` | Same | `kimi-k3.cpp:95-98` |
| `blk.%d.attn_q` | `LLM_TENSOR_ATTN_Q` | `[7168, 12288]` | Q projection | `kimi-k3.cpp:101` |
| `blk.%d.attn_k` | `LLM_TENSOR_ATTN_K` | `[7168, 12288]` | K projection | `kimi-k3.cpp:101` |
| `blk.%d.attn_v` | `LLM_TENSOR_ATTN_V` | `[7168, 12288]` | V projection | `kimi-k3.cpp:101` |
| `blk.%d.ssm_f_a` | `LLM_TENSOR_SSM_F_A` | `[7168, 128]` | Forget gate A (low-rank) | `kimi-k3.cpp:104` |
| `blk.%d.ssm_f_b` | `LLM_TENSOR_SSM_F_B` | `[128, 12288]` | Forget gate B (low-rank) | `kimi-k3.cpp:105` |
| `blk.%d.ssm_beta` | `LLM_TENSOR_SSM_BETA` | `[7168, 96]` | Beta mixing coefficient | `kimi-k3.cpp:108` |
| `blk.%d.ssm_a` | `LLM_TENSOR_SSM_A` | `[96]` or `[1, 96]` | exp(A_log), per head, positive | `kimi-k3.cpp:111-114` |
| `blk.%d.ssm_dt` | `LLM_TENSOR_SSM_DT` | `[12288]` | dt_bias | `kimi-k3.cpp:117` |
| `blk.%d.attn_gate` | `LLM_TENSOR_ATTN_GATE` | `[7168, 12288]` | Full-rank output gate g2 | `kimi-k3.cpp:120` |
| `blk.%d.ssm_norm` | `LLM_TENSOR_SSM_NORM` | `[128]` | o_norm (RMSNorm) | `kimi-k3.cpp:123` |
| `blk.%d.attn_output` | `LLM_TENSOR_ATTN_OUT` | `[12288, 7168]` | o_proj | `kimi-k3.cpp:126` |

**HF → GGUF tensor name mapping** (from `tensor_mapping.py:923-948`):
| HF Name | GGUF Name |
|---------|-----------|
| `model.layers.{bid}.self_attn.q_conv1d` | `blk.%d.ssm_conv1d_q` |
| `model.layers.{bid}.self_attn.k_conv1d` | `blk.%d.ssm_conv1d_k` |
| `model.layers.{bid}.self_attn.v_conv1d` | `blk.%d.ssm_conv1d_v` |
| `model.layers.{bid}.self_attn.f_a_proj` | `blk.%d.ssm_f_a` |
| `model.layers.{bid}.self_attn.f_b_proj` | `blk.%d.ssm_f_b` |
| `model.layers.{bid}.self_attn.b_proj` | `blk.%d.ssm_beta` |
| `model.layers.{bid}.A_log` | `blk.%d.ssm_a` |
| `model.layers.{bid}.self_attn.o_norm` | `blk.%d.ssm_norm` |
| `model.layers.{bid}.self_attn.gate_proj` | `blk.%d.attn_gate` |

**Conv1d weight reshape** (from `kimi_linear.py:158-170`):
HF stores as `[d_inner, d_conv]` (2D). Converter reshapes to `(1, d_inner, 1, d_conv)` → GGUF `ne = [d_conv, 1, d_inner, 1]`.

**A_log conversion** (from `kimi_k3.py:58-63`):
HF stores `A_log` as `[head_dim]` (128). K3 converter slices to `[:n_head]` (96), then applies `exp()` (positive). Unlike Kimi Linear which stores `-exp(A_log)`, K3 stores `+exp(A_log)`.

### 4.5 Gated MLA Layer Tensors (24 layers: `is_recr(i) == false`)

**Dimensions:**
- `n_embd` = 7168, `n_head` = 96
- `q_lora_rank` = 1536, `kv_lora_rank` = 512
- `n_embd_head_k_mla` = `qk_nope_head_dim + qk_rope_head_dim` = 128 + 64 = 192
- `n_embd_head_v_mla` = `v_head_dim` = 128
- `qk_rope_head_dim` = 64, `qk_nope_head_dim` = 128

| GGUF Name | C++ Tensor Enum | Shape | Notes | Source |
|-----------|-----------------|-------|-------|--------|
| `blk.%d.attn_q_a_norm` | `LLM_TENSOR_ATTN_Q_A_NORM` | `[1536]` | Q-A RMSNorm | `kimi-k3.cpp:135` |
| `blk.%d.attn_kv_a_norm` | `LLM_TENSOR_ATTN_KV_A_NORM` | `[512]` | KV-A RMSNorm | `kimi-k3.cpp:136` |
| `blk.%d.attn_q_a` | `LLM_TENSOR_ATTN_Q_A` | `[7168, 1536]` | Q down-projection | `kimi-k3.cpp:138` |
| `blk.%d.attn_q_b` | `LLM_TENSOR_ATTN_Q_B` | `[1536, 18432]` | Q up-projection (96×192) | `kimi-k3.cpp:139` |
| `blk.%d.attn_kv_a_mqa` | `LLM_TENSOR_ATTN_KV_A_MQA` | `[7168, 576]` | KV-A MQA (512+64) | `kimi-k3.cpp:141` |
| `blk.%d.attn_kv_b` | `LLM_TENSOR_ATTN_KV_B` | `[512, 96×(128+128)]` = `[512, 24576]` | Optional, fallback path | `kimi-k3.cpp:143-144` |
| `blk.%d.attn_k_b` | `LLM_TENSOR_ATTN_K_B` | `[128, 512, 96]` | MLA KV cache enabled | `kimi-k3.cpp:146` |
| `blk.%d.attn_v_b` | `LLM_TENSOR_ATTN_V_B` | `[512, 128, 96]` | MLA KV cache enabled | `kimi-k3.cpp:147` |
| `blk.%d.attn_gate` | `LLM_TENSOR_ATTN_GATE` | `[7168, 12288]` | Output gate (96×128) | `kimi-k3.cpp:151` |
| `blk.%d.attn_output` | `LLM_TENSOR_ATTN_OUT` | `[12288, 7168]` | o_proj | `kimi-k3.cpp:153` |

**Note on `attn_kv_b` vs `attn_k_b` + `attn_v_b`:** These are mutually exclusive. The converter (`kimi_linear.py:207-221`) splits `kv_b_proj.weight` into `k_b_proj` and `v_b_proj` when MLA KV cache absorption is enabled. The runtime checks `layer.wk_b && layer.wv_b` to decide which path to use (`kimi-k3.cpp:145-148`).

**HF → GGUF tensor name mapping** (from `tensor_mapping.py`):
| HF Name | GGUF Name |
|---------|-----------|
| `model.layers.{bid}.self_attn.q_a_proj` | `blk.%d.attn_q_a` |
| `model.layers.{bid}.self_attn.q_b_proj` | `blk.%d.attn_q_b` |
| `model.layers.{bid}.self_attn.kv_a_proj` | `blk.%d.attn_kv_a_mqa` |
| `model.layers.{bid}.self_attn.kv_b_proj` | `blk.%d.attn_kv_b` |
| `model.layers.{bid}.self_attn.k_b_proj` | `blk.%d.attn_k_b` |
| `model.layers.{bid}.self_attn.v_b_proj` | `blk.%d.attn_v_b` |
| `model.layers.{bid}.self_attn.gate_proj` | `blk.%d.attn_gate` |
| `model.layers.{bid}.self_attn.o_proj` | `blk.%d.attn_output` |

### 4.6 Dense FFN Tensors (Layer 0 only, `i < leading_dense_block_count`)

**Dimensions:** `n_ff` = `intermediate_size` = 33792

| GGUF Name | C++ Tensor Enum | Shape | Source |
|-----------|-----------------|-------|--------|
| `blk.0.ffn_gate` | `LLM_TENSOR_FFN_GATE` | `[7168, 33792]` | `kimi-k3.cpp:162` |
| `blk.0.ffn_down` | `LLM_TENSOR_FFN_DOWN` | `[33792, 7168]` | `kimi-k3.cpp:163` |
| `blk.0.ffn_up` | `LLM_TENSOR_FFN_UP` | `[7168, 33792]` | `kimi-k3.cpp:164` |

### 4.7 Stable LatentMoE Tensors (Layers 1–92, `i >= leading_dense_block_count`)

**Dimensions:**
- `n_expert` = 896, `n_expert_used` = 16, `n_expert_shared` = 2
- `moe_latent` = `routed_expert_hidden_size` = 3584
- `n_ff_exp` = `moe_intermediate_size` = 3072
- `n_ff_shexp` = `n_ff_exp * n_expert_shared` = 3072 × 2 = 6144

| GGUF Name | C++ Tensor Enum | Shape | Source |
|-----------|-----------------|-------|--------|
| `blk.%d.ffn_gate_inp` | `LLM_TENSOR_FFN_GATE_INP` | `[7168, 896]` | Router | `kimi-k3.cpp:167` |
| `blk.%d.exp_probs_b` | `LLM_TENSOR_FFN_EXP_PROBS_B` | `[896]` | Router bias | `kimi-k3.cpp:168` |
| `blk.%d.ffn_latent_down` | `LLM_TENSOR_FFN_LATENT_DOWN` | `[7168, 3584]` | Latent down-proj | `kimi-k3.cpp:170` |
| `blk.%d.ffn_latent_norm` | `LLM_TENSOR_FFN_LATENT_NORM` | `[3584]` | Latent RMSNorm | `kimi-k3.cpp:171` |
| `blk.%d.ffn_latent_up` | `LLM_TENSOR_FFN_LATENT_UP` | `[3584, 7168]` | Latent up-proj | `kimi-k3.cpp:172` |
| `blk.%d.ffn_gate_exps` | `LLM_TENSOR_FFN_GATE_EXPS` | `[3584, 3072, 896]` | Routed expert gates | `kimi-k3.cpp:174` |
| `blk.%d.ffn_down_exps` | `LLM_TENSOR_FFN_DOWN_EXPS` | `[3072, 3584, 896]` | Routed expert downs | `kimi-k3.cpp:175` |
| `blk.%d.ffn_up_exps` | `LLM_TENSOR_FFN_UP_EXPS` | `[3584, 3072, 896]` | Routed expert ups | `kimi-k3.cpp:176` |
| `blk.%d.ffn_gate_shexp` | `LLM_TENSOR_FFN_GATE_SHEXP` | `[7168, 6144]` | Shared expert gate | `kimi-k3.cpp:180` |
| `blk.%d.ffn_down_shexp` | `LLM_TENSOR_FFN_DOWN_SHEXP` | `[6144, 7168]` | Shared expert down | `kimi-k3.cpp:181` |
| `blk.%d.ffn_up_shexp` | `LLM_TENSOR_FFN_UP_SHEXP` | `[7168, 6144]` | Shared expert up | `kimi-k3.cpp:182` |

**HF → GGUF tensor name mapping** (from `tensor_mapping.py`):
| HF Name | GGUF Name |
|---------|-----------|
| `model.layers.{bid}.block_sparse_moe.gate.weight` | `blk.%d.ffn_gate_inp` |
| `model.layers.{bid}.block_sparse_moe.gate.e_score_correction` | `blk.%d.exp_probs_b` |
| `model.layers.{bid}.block_sparse_moe.routed_expert_down_proj` | `blk.%d.ffn_latent_down` |
| `model.layers.{bid}.block_sparse_moe.routed_expert_norm` | `blk.%d.ffn_latent_norm` |
| `model.layers.{bid}.block_sparse_moe.routed_expert_up_proj` | `blk.%d.ffn_latent_up` |
| `model.layers.{bid}.block_sparse_moe.experts.{eid}.w1.weight` | `blk.%d.ffn_gate.%d` (per-expert, merged to `ffn_gate_exps`) |
| `model.layers.{bid}.block_sparse_moe.experts.{eid}.w2.weight` | `blk.%d.ffn_down.%d` (per-expert, merged to `ffn_down_exps`) |
| `model.layers.{bid}.block_sparse_moe.experts.{eid}.w3.weight` | `blk.%d.ffn_up.%d` (per-expert, merged to `ffn_up_exps`) |
| `model.layers.{bid}.block_sparse_moe.shared_experts.gate_proj` | `blk.%d.ffn_gate_shexp` |
| `model.layers.{bid}.block_sparse_moe.shared_experts.down_proj` | `blk.%d.ffn_down_shexp` |
| `model.layers.{bid}.block_sparse_moe.shared_experts.up_proj` | `blk.%d.ffn_up_shexp` |

**Expert merging** (from `kimi_linear.py:182-205`): Individual expert tensors `model.layers.{bid}.block_sparse_moe.experts.{eid}.w{1,2,3}.weight` are stacked into 3D tensors `[moe_latent, n_ff_exp, n_expert]` / `[n_ff_exp, moe_latent, n_expert]`.

### 4.8 AttnRes HF → GGUF Mapping

| HF Name | GGUF Name | Source |
|---------|-----------|--------|
| `model.layers.{bid}.self_attention_res_norm` | `blk.%d.attn_res_norm` | `tensor_mapping.py:634-636` |
| `model.layers.{bid}.self_attention_res_proj` | `blk.%d.attn_res_proj` | `tensor_mapping.py:638-640` |
| `model.layers.{bid}.mlp_res_norm` | `blk.%d.ffn_res_norm` | `tensor_mapping.py:642-644` |
| `model.layers.{bid}.mlp_res_proj` | `blk.%d.ffn_res_proj` | `tensor_mapping.py:646-648` |
| `model.output_attn_res_norm` | `output_res_norm` | `tensor_mapping.py:121-123` |
| `model.output_attn_res_proj` | `output_res_proj` | `tensor_mapping.py:125-127` |

**Res projection reshape** (from `kimi_k3.py:66-67`): HF stores `_res_proj.weight` and `_res_norm.weight` as `[1, n_embd]`; converter flattens to `[n_embd]`.

---

## 5. KDA (Kimi Delta Attention) — Exact Formulas

### 5.1 Causal Conv1D

```
Q = causal_conv1d(x, W_q, conv1d_q)  # [d_inner, n_tokens]
K = causal_conv1d(x, W_k, conv1d_k)
V = causal_conv1d(x, W_v, conv1d_v)
```

Each conv1d: `silu(conv(x_proj, conv_weight))` reshaped to `[head_dim, n_head, n_seq_tokens, n_seqs]`.

**Source:** `kimi-k3.cpp:192-227`

### 5.2 Safe Gate (g1)

```
f_a = x @ W_f_a           # [n_embd] → [head_dim=128]
g1_raw = f_a @ W_f_b      # [head_dim] → [d_inner=12288]
g1_raw = g1_raw + dt_bias # [d_inner]
g1 = reshape(g1_raw, [head_dim, n_head, n_tokens])
A = reshape(exp(A_log), [1, n_head, 1])  # positive, per head
g1 = g1 * A
g1 = sigmoid(g1)
g1 = g1 * gate_lower_bound  # -5.0
g1 = reshape(g1, [head_dim, n_head, n_seq_tokens, n_seqs])
```

**Source:** `kimi-k3.cpp:355-370`

### 5.3 Beta (Mixing Coefficient)

```
beta = x @ W_beta          # [n_embd] → [n_head=96]
beta = reshape(beta, [1, n_head, n_seq_tokens, n_seqs])
beta = sigmoid(beta)
```

**Source:** `kimi-k3.cpp:372-375`

### 5.4 L2-Norm and Delta Net Recurrence

```
Q = l2_norm(Q, eps)
K = l2_norm(K, eps)
attn_out, new_state = delta_net(Q, K, V, g1, beta, state)
```

**Source:** `kimi-k3.cpp:384-390`

### 5.5 Full-Rank Output Gate (g2)

```
g2 = x @ W_gate            # [n_embd] → [d_inner=12288]
g2 = reshape(g2, [head_dim, n_head, n_tokens])
attn_out = reshape(attn_out, [head_dim, n_head, n_tokens])
normed = RMSNorm(attn_out, o_norm)
gated = normed * sigmoid(g2)
out = gated @ W_o          # [d_inner] → [n_embd]
```

**Source:** `kimi-k3.cpp:398-411`

---

## 6. Gated MLA (NoPE) — Exact Formulas

### 6.1 Q Projection

```
Q = q_b(rms_norm(q_a(x @ W_q_a), q_a_norm)) @ W_q_b
# [n_embd] → [q_lora_rank=1536] → [n_head * n_embd_head_k_mla=96×192=18432]
```

**Source:** `kimi-k3.cpp:415-418`

### 6.2 KV Compression

```
kv_cmpr_pe = x @ W_kv_a_mqa  # [n_embd] → [kv_lora_rank + qk_rope_head_dim=512+64=576]
kv_cmpr = kv_cmpr_pe[:, :512]  # latent part
k_pe = kv_cmpr_pe[:, 512:]     # positional part (NoPE: no RoPE applied)
kv_cmpr = rms_norm(kv_cmpr, kv_a_norm)
```

**Source:** `kimi-k3.cpp:421-430`

### 6.3 Attention (MLA KV Cache Enabled Path)

```
q_nope = Q[:, :128]  # first 128 dims
q_pe = Q[:, 128:]    # last 64 dims
q_nope_absorbed = q_nope @ W_k_b  # [128, kv_lora_rank, n_head]
Q_mla = concat(q_nope_absorbed, q_pe, dim=0)
K = concat(kv_cmpr, k_pe, dim=0)
V = kv_cmpr
attn_pregate = attention(Q_mla, K, V, W_v_b, scale=1/sqrt(192))
```

**Source:** `kimi-k3.cpp:434-453`

### 6.4 Output Gate

```
gate = sigmoid(x @ W_gate)  # [n_embd] → [n_head * n_embd_head_v_mla=12288]
attn = attn_pregate * gate
out = attn @ W_o
```

**Source:** `kimi-k3.cpp:476-483`

---

## 7. Attention Residuals (AttnRes)

### 7.1 Snapshot Bank

Every `attn_res_block_size` (12) layers, the current residual stream `prefix` is snapshotted into `res_bank`:

```cpp
const bool snapshot = (il % res_block) == 0;
if (snapshot) {
    res_bank.push_back(prefix);
}
```

**Source:** `kimi-k3.cpp:333-336`

### 7.2 Pre-Attention Mixture

```
h = (bank empty) ? prefix : attn_res_mix(prefix, bank, attn_res_norm, attn_res_proj)
```

**Source:** `kimi-k3.cpp:328-330`

### 7.3 Pre-FFN Mixture

```
h2 = attn_res_mix(prefix, bank, ffn_res_norm, ffn_res_proj)
```

**Source:** `kimi-k3.cpp:491`

### 7.4 Output Mixture

```
cur = attn_res_mix(prefix, bank, output_res_norm, output_res_proj)
```

**Source:** `kimi-k3.cpp:554`

### 7.5 Mixture Formula

```
v = concat(bank_rows..., prefix)  # [n_embd, n_rows, n_toks]
k = RMSNorm(v)
sw = norm_w * proj_w              # elementwise, [n_embd]
scores = k @ sw                   # [1, n_rows, n_toks]
probs = softmax(scores)           # over rows
out = sum_r probs_r * v_r         # weighted sum of raw (unnormalized) rows
```

**Source:** `kimi-k3.cpp:235-278`

### 7.6 Residual Stream Update

After attention: `prefix = snapshot ? cur : prefix + cur`
After FFN: `prefix = prefix + cur`

**Source:** `kimi-k3.cpp:487-488, 548`

---

## 8. Stable LatentMoE

### 8.1 Router

```
router_logits = x @ W_gate_inp  # [n_embd] → [n_expert=896]
# Sigmoid activation, top-16 selection, renormalize
# exp_probs_b is added as bias
```

**Source:** `kimi-k3.cpp:507-508`

**Router config** (from HF config):
- `moe_router_activation_func = "sigmoid"`
- `topk_method = "noaux_tc"`
- `moe_renormalize = true` (hard-coded in `kimi-k3.cpp:48`)
- `n_expert_used = 16`

### 8.2 Latent Projection

```
latent = x @ W_latent_down  # [n_embd=7168] → [moe_latent=3584]
```

**Source:** `kimi-k3.cpp:511`

### 8.3 Routed Expert Computation (in Latent Space)

```
moe_out = sum_e w_e * SiTU_Gate(latent @ W_gate_e)
                 * SiTU_Linear(latent @ W_up_e)
                 @ W_down_e
```

Where:

```
SiTU_Gate(x)   = beta * tanh(x / beta) * sigmoid(x)
SiTU_Linear(x) = linear_beta * tanh(x / linear_beta)
```

**Source:** `llama-graph.cpp` SiTU FFN implementation and `pulsar_kernels.cu:k3_situ_glu_kernel`.

### 8.4 Latent Norm and Up-Projection

```
moe_out = RMSNorm(moe_out, latent_norm)
moe_out = moe_out @ W_latent_up  # [moe_latent=3584] → [n_embd=7168]
```

**Source:** `kimi-k3.cpp:531-533`

### 8.5 Shared Experts (on Full Hidden State)

```
shared = SiTU_Gate(x @ W_gate_sh) * SiTU_Linear(x @ W_up_sh) @ W_down_sh
```

The same gate/linear definitions from §8.3 apply. `out = moe_out + shared`.

---

## 9. SiTU-GLU Activation

Used in dense FFN, routed experts, and shared experts.

```
SiTU_Gate(x)   = beta * tanh(x / beta) * sigmoid(x)
SiTU_Linear(x) = linear_beta * tanh(x / linear_beta)
```

The full SiTU-GLU FFN is:

```
out = SiTU_Gate(x @ W_gate)
    * SiTU_Linear(x @ W_up)
    @ W_down
```

The production K3 defaults are `situ_beta = 4.0` and `situ_linear_beta = 25.0`; scaled test fixtures may use smaller values but must preserve the same formula.

**Source:** `llama-graph.cpp` SiTU FFN implementation; CUDA parity path: `crates/kernels/cuda/pulsar_kernels.cu:k3_situ_glu_kernel`.

---

## 10. Tokenizer Requirements

| Property | Value | Source |
|----------|-------|--------|
| Model type | `"gpt2"` (BPE) | `kimi_linear.py:64` |
| Pre-tokenizer | `"kimi-k2"` | `kimi_linear.py:33,65` |
| Vocabulary size | 163840 | HF config `vocab_size` |
| BOS token ID | 163584 | HF config |
| EOS token ID | 163586 | HF config |
| Pad token ID | 163839 | HF config |
| Tie word embeddings | `false` | HF config |

The tokenizer is a GPT-2 BPE tokenizer with the `kimi-k2` pre-tokenizer. The converter builds the token list from `tokenizer.model._mergeable_ranks` and `tokenizer.special_tokens`.

---

## 11. HF Config Values (Canonical)

| Parameter | Value | GGUF Key |
|-----------|-------|----------|
| `num_hidden_layers` | 93 | `kimi-k3.block_count` |
| `hidden_size` | 7168 | `kimi-k3.embedding_length` |
| `num_attention_heads` | 96 | `kimi-k3.attention.head_count` |
| `num_key_value_heads` | 96 (MQA: 1 per MLA layer, 0 per KDA) | `kimi-k3.attention.head_count_kv` (array) |
| `intermediate_size` | 33792 | `kimi-k3.feed_forward_length` |
| `linear_attn_config.head_dim` | 128 | `kimi-k3.kda.head_dim` |
| `linear_attn_config.gate_lower_bound` | -5.0 | `kimi-k3.kda.gate_lower_bound` |
| `linear_attn_config.short_conv_kernel_size` | 4 | `kimi-k3.ssm.conv_kernel` |
| `linear_attn_config.num_heads` | 96 | (same as `num_attention_heads`) |
| `linear_attn_config.use_full_rank_gate` | true | (hard-coded in K3) |
| `q_lora_rank` | 1536 | `kimi-k3.attention.q_lora_rank` |
| `kv_lora_rank` | 512 | `kimi-k3.attention.kv_lora_rank` |
| `qk_nope_head_dim` | 128 | (part of `key_length_mla`) |
| `qk_rope_head_dim` | 64 | `kimi-k3.rope.dimension_count` |
| `v_head_dim` | 128 | `kimi-k3.attention.value_length_mla` |
| `n_embd_head_k_mla` | 192 (128+64) | `kimi-k3.attention.key_length_mla` |
| `n_embd_head_v_mla` | 128 | `kimi-k3.attention.value_length_mla` |
| `num_experts` | 896 | `kimi-k3.expert_count` |
| `num_experts_per_token` | 16 | `kimi-k3.expert_used_count` |
| `num_shared_experts` | 2 | `kimi-k3.expert_shared_count` |
| `moe_intermediate_size` | 3072 | `kimi-k3.expert_feed_forward_length` |
| `routed_expert_hidden_size` | 3584 | `kimi-k3.moe_latent_size` |
| `first_k_dense_replace` | 1 | `kimi-k3.leading_dense_block_count` |
| `routed_scaling_factor` | 1.0 | `kimi-k3.expert_weights_scale` |
| `activation_situ_beta` | 4.0 | `kimi-k3.situ_beta` |
| `activation_situ_linear_beta` | 25.0 | `kimi-k3.situ_linear_beta` |
| `attn_res_block_size` | 12 | `kimi-k3.attn_res_block_size` |
| `rms_norm_eps` | 1e-05 | `kimi-k3.attention.layer_norm_rms_epsilon` |
| `max_position_embeddings` | 1048576 | `kimi-k3.context_length` |
| `vocab_size` | 163840 | `kimi-k3.vocab_size` |
| `hidden_act` | `"situ"` | `kimi-k3.hidden_activation` |
| `moe_renormalize` | true | (hard-coded in `kimi-k3.cpp:48`) |
| `moe_router_activation_func` | `"sigmoid"` | `kimi-k3.expert_gating_func` (value 2) |
| `mla_use_nope` | true | (NoPE: no RoPE applied) |
| `mla_use_output_gate` | true | (MLA output gate enabled) |
| `tie_word_embeddings` | false | (separate `output` tensor) |

---

## 12. RoPE

K3 uses **no RoPE** anywhere. `LLAMA_ROPE_TYPE_NONE` is returned for `LLM_ARCH_KIMI_K3`.

**Source:** `llama-model.cpp:2458`

The `qk_rope_head_dim` (64) in MLA is a **positional-encoding-free shared key dimension** — no RoPE is applied to it. See `kimi-k3.cpp:429` comment: `// NoPE: k_pe is a positional-encoding-free shared key dimension, no RoPE applied`.

---

## 13. Complete Tensor List (All 103 Tensors)

For a full K3 GGUF with 93 layers, 1 dense FFN layer, 92 MoE layers, 69 KDA layers, 24 MLA layers:

| Count | Tensor Name Pattern | Present In |
|-------|-------------------|------------|
| 1 | `token_embd` | Global |
| 1 | `output_norm` | Global |
| 1 | `output` | Global |
| 1 | `output_res_norm` | Global |
| 1 | `output_res_proj` | Global |
| 93 | `blk.%d.attn_norm` | All layers |
| 93 | `blk.%d.attn_res_norm` | All layers |
| 93 | `blk.%d.attn_res_proj` | All layers |
| 93 | `blk.%d.ffn_res_norm` | All layers |
| 93 | `blk.%d.ffn_res_proj` | All layers |
| 93 | `blk.%d.ffn_norm` | All layers |
| 69 | `blk.%d.ssm_conv1d_q` | KDA layers |
| 69 | `blk.%d.ssm_conv1d_k` | KDA layers |
| 69 | `blk.%d.ssm_conv1d_v` | KDA layers |
| 69 | `blk.%d.attn_q` | KDA layers |
| 69 | `blk.%d.attn_k` | KDA layers |
| 69 | `blk.%d.attn_v` | KDA layers |
| 69 | `blk.%d.ssm_f_a` | KDA layers |
| 69 | `blk.%d.ssm_f_b` | KDA layers |
| 69 | `blk.%d.ssm_beta` | KDA layers |
| 69 | `blk.%d.ssm_a` | KDA layers |
| 69 | `blk.%d.ssm_dt` | KDA layers |
| 69 | `blk.%d.attn_gate` | KDA layers |
| 69 | `blk.%d.ssm_norm` | KDA layers |
| 69 | `blk.%d.attn_output` | KDA layers |
| 24 | `blk.%d.attn_q_a_norm` | MLA layers |
| 24 | `blk.%d.attn_kv_a_norm` | MLA layers |
| 24 | `blk.%d.attn_q_a` | MLA layers |
| 24 | `blk.%d.attn_q_b` | MLA layers |
| 24 | `blk.%d.attn_kv_a_mqa` | MLA layers |
| 24 | `blk.%d.attn_kv_b` OR `blk.%d.attn_k_b` + `blk.%d.attn_v_b` | MLA layers |
| 24 | `blk.%d.attn_gate` | MLA layers |
| 24 | `blk.%d.attn_output` | MLA layers |
| 1 | `blk.0.ffn_gate` | Dense FFN (layer 0) |
| 1 | `blk.0.ffn_down` | Dense FFN (layer 0) |
| 1 | `blk.0.ffn_up` | Dense FFN (layer 0) |
| 92 | `blk.%d.ffn_gate_inp` | MoE layers (1–92) |
| 92 | `blk.%d.exp_probs_b` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_latent_down` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_latent_norm` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_latent_up` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_gate_exps` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_down_exps` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_up_exps` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_gate_shexp` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_down_shexp` | MoE layers (1–92) |
| 92 | `blk.%d.ffn_up_shexp` | MoE layers (1–92) |

**Total: 5 global + 93×5 (shared) + 69×14 (KDA) + 24×8 (MLA) + 3 (dense FFN) + 92×11 (MoE) = 103 tensors** (minimum, assuming `attn_kv_b` path; 105 with `attn_k_b` + `attn_v_b` split).

---

## 14. Source Citations

| File | Lines | Content |
|------|-------|---------|
| `src/llama-arch.h` | 146 | `LLM_ARCH_KIMI_K3` enum |
| `src/llama-arch.cpp` | 144 | `"kimi-k3"` string mapping |
| `src/llama-arch.cpp` | 306-311 | K3-specific KV keys (KDA, SiTU, AttnRes) |
| `src/llama-arch.h` | 311-316 | `LLM_KV_KDA_HEAD_DIM`, `LLM_KV_KDA_GATE_LOWER_BOUND`, `LLM_KV_SITU_BETA`, `LLM_KV_SITU_LINEAR_BETA`, `LLM_KV_ATTN_RES_BLOCK_SIZE` |
| `src/llama-hparams.h` | 162-169 | K3 hparam fields |
| `src/llama-hparams.h` | 10 | `LLAMA_MAX_EXPERTS 512` (note: K3 needs 896) |
| `src/models/kimi-k3.cpp` | 1-570 | Full K3 model implementation |
| `src/models/kimi-k3.cpp` | 20-51 | `load_arch_hparams` — metadata parsing |
| `src/models/kimi-k3.cpp` | 35-37 | Per-layer KDA/MLA discriminator |
| `src/models/kimi-k3.cpp` | 53-184 | `load_arch_tensors` — all tensor definitions |
| `src/models/kimi-k3.cpp` | 84-127 | KDA layer tensors |
| `src/models/kimi-k3.cpp` | 128-153 | Gated MLA layer tensors |
| `src/models/kimi-k3.cpp` | 160-164 | Dense FFN tensors |
| `src/models/kimi-k3.cpp` | 165-183 | Stable LatentMoE tensors |
| `src/models/kimi-k3.cpp` | 235-278 | AttnRes mixture formula |
| `src/models/kimi-k3.cpp` | 343-411 | KDA forward (safe gate, beta, delta net, output gate) |
| `src/models/kimi-k3.cpp` | 412-483 | Gated MLA forward |
| `src/models/kimi-k3.cpp` | 496-544 | Dense FFN + Stable LatentMoE forward |
| `src/models/kimi-k3.cpp` | 553-554 | Output AttnRes mixture |
| `src/llama-model.cpp` | 312-313 | `LLM_ARCH_KIMI_K3` → `llama_model_kimi_k3` |
| `src/llama-model.cpp` | 2457-2459 | `LLAMA_ROPE_TYPE_NONE` for K3 |
| `conversion/kimi_k3.py` | 1-69 | K3 converter (inherits KimiLinear) |
| `conversion/kimi_linear.py` | 1-223 | Kimi Linear converter (K3 base) |
| `conversion/__init__.py` | 124 | `"KimiK3ForConditionalGeneration" → "kimi_k3"` |
| `gguf-py/gguf/constants.py` | 552, 1150 | `MODEL_ARCH.KIMI_K3 = "kimi-k3"` |
| `gguf-py/gguf/constants.py` | 4461-4509 | K3 tensor list |
| `gguf-py/gguf/constants.py` | 128-131 | K3-specific Keys.LLM constants |
| `gguf-py/gguf/constants.py` | 248-250 | `Keys.KDA` constants |
| `gguf-py/gguf/constants.py` | 626-634 | `MODEL_TENSOR` AttnRes/MoE latent entries |
| `gguf-py/gguf/tensor_mapping.py` | 121-127 | `output_res_norm`/`output_res_proj` mapping |
| `gguf-py/gguf/tensor_mapping.py` | 392-394 | `attn_gate` mapping |
| `gguf-py/gguf/tensor_mapping.py` | 620-632 | `moe_latent_down`/`up`/`norm` mapping |
| `gguf-py/gguf/tensor_mapping.py` | 634-648 | `attn_res_norm`/`proj`, `ffn_res_norm`/`proj` mapping |
| `gguf-py/gguf/tensor_mapping.py` | 923-948 | KDA tensor mapping (conv1d, f_a, f_b, beta, A_log, norm) |
| `convert_hf_to_gguf_update.py` | 184 | `"kimi-k2"` tokenizer checksum entry |
| HF config | — | `moonshotai/Kimi-K3` `text_config` (fetched 2026-07-28) |

---

## 15. Verification Commands

```bash
# Verify architecture string in C++ runtime
grep -n 'KIMI_K3' /tmp/atomic-llama-cpp-turboquant-kimi/src/llama-arch.cpp

# Verify architecture string in Python converter
grep -n 'KIMI_K3' /tmp/atomic-llama-cpp-turboquant-kimi/gguf-py/gguf/constants.py

# Verify HF config
curl -s https://huggingface.co/moonshotai/Kimi-K3/raw/main/config.json | python3 -m json.tool

# Verify tensor names in the model implementation
grep -n 'create_tensor' /tmp/atomic-llama-cpp-turboquant-kimi/src/models/kimi-k3.cpp

# Verify per-layer discriminator logic
grep -n 'is_recr' /tmp/atomic-llama-cpp-turboquant-kimi/src/models/kimi-k3.cpp

# Verify RoPE type
grep -n 'KIMI_K3' /tmp/atomic-llama-cpp-turboquant-kimi/src/llama-model.cpp
```
