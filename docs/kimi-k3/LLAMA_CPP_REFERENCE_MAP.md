# Kimi K3 llama.cpp Reference Map

## Pinned Reference

- PR: `https://github.com/ggml-org/llama.cpp/pull/26185`
- Repository: `https://github.com/ggml-org/llama.cpp.git`
- Checkout: `/home/neron/llama.cpp-kimi-k3-reference`
- Branch: `kimi-k3-pr-26185`
- Commit: `7b990cf5721b2ecb865e65beb63413a25d73cd3e`
- Retrieval commands:

```bash
cd /home/neron
git clone https://github.com/ggml-org/llama.cpp.git llama.cpp-kimi-k3-reference
cd llama.cpp-kimi-k3-reference
git fetch origin pull/26185/head:kimi-k3-pr-26185
git checkout kimi-k3-pr-26185
git remote -v
git rev-parse HEAD
git log -1 --oneline --decorate
git status --short
git diff --name-status origin/master...HEAD
git diff --stat origin/master...HEAD
git log --oneline origin/master..HEAD
```

The checkout was clean after retrieval. A separate `build-cpu/` directory was
created for the dynamic attempt. No upstream source files were modified.

## Changed PR Files

The PR changes these files: `common/chat.cpp`, `conversion/__init__.py`,
`conversion/base.py`, `conversion/deepseek.py`, `conversion/kimi_k3.py`,
`gguf-py/gguf/constants.py`, `gguf-py/gguf/gguf_writer.py`,
`gguf-py/gguf/tensor_mapping.py`, `models/templates/Kimi-K3.jinja`,
`src/llama-arch.cpp`, `src/llama-arch.h`, `src/llama-context.cpp`,
`src/llama-graph.cpp`, `src/llama-graph.h`, `src/llama-hparams.h`,
`src/llama-model.cpp`, `src/llama-model.h`, `src/models/kimi-k3.cpp`,
`src/models/models.h`, `tests/test-chat.cpp`, and `tests/test-llama-archs.cpp`.

Only the following are material to K3 mathematics or GGUF semantics:

| llama.cpp location | Symbol | Purpose | Pulsar counterpart | Status |
|---|---|---|---|---|
| `src/models/kimi-k3.cpp:17-162` | `load_arch_hparams`, `load_arch_tensors` | K3 metadata and tensor shapes | `crates/engine/src/lib.rs:264-459,3231-3405` | Match after MLA tail fix; Pulsar supports the existing `moe_latent_size` GGUF alias |
| `src/models/kimi-k3.cpp:175-183` | `kimi_k3_situ` | Exact SiTU formula | `crates/kernels/cuda/pulsar_kernels.cu:4913-4928`, host path `kimi_k3.rs:1796-1805` | Match |
| `src/models/kimi-k3.cpp:190-247` | `res_push`, `res_stack`, `res_mix` | Attention Residual scoring and weighted retrieval | `kimi_k3.rs` AttnRes helpers and layer loop | Corrected: retrieval feeds sublayers without replacing the raw residual; bank resets per token |
| `src/models/kimi-k3.cpp:291-343` | K3 layer graph | Layer ordering, checkpoint restart, residual updates | `kimi_k3.rs` K3 layer loop | Corrected residual and sequence-reset semantics |
| `src/models/kimi-k3.cpp:373-406` | `kimi_k3_conv1d` | Conv state layout and causal tap order | `kimi_k3.rs:876-1046`, `pulsar_kernels.cu:4178-4233` | Equivalent implementation |
| `src/models/kimi-k3.cpp:408-495` | `build_kda_layer` | KDA gate, beta, delta state, output gate | `kimi_k3.rs` KDA forward | Corrected folded-sign compatibility, `1e-12` Q/K L2 epsilon, and per-head output RMSNorm |
| `src/models/kimi-k3.cpp:502-579` | `build_mla_layer` | MLA Q/K/V, absorbed causal attention, gate and output | `kimi_k3.rs` MLA forward and `pulsar_k3_mla_cached_attn_split` | Corrected 64-wide tail and compact causal history for split weights |
| `src/models/kimi-k3.cpp:587-635` | `build_latent_moe` | Latent projection, router, experts, norm/up, shared path | `kimi_k3.rs:1259-1694` | Match |
| `src/llama-graph.cpp:1810-2100` | `build_moe_ffn` | Sigmoid scoring, bias selection, top-k, normalization, weighted down output | `pulsar_kernels.cu:4944-5055`, `kimi_k3.rs:1367-1378,2039-2091` | Match |
| `conversion/kimi_k3.py:201-262` | `set_gguf_parameters` | Converter metadata contract | `Shape::from_gguf` and `crates/gguf/src/lib.rs` | PR names latent field `expert_latent_length`; existing model uses `moe_latent_size` |
| `conversion/kimi_k3.py:277-382` | `modify_tensors` | Residual fusion, conv reshape, A folding, expert stacking, MLA absorption | `crates/engine/src/lib.rs:2328-2479` | Match; A sign correction applied in Pulsar compute |
| `gguf-py/gguf/tensor_mapping.py:893-924` | KDA/latent tensor mappings | GGUF tensor names | `crates/engine/src/lib.rs:3287-3404` | Match |
| `gguf-py/gguf/constants.py:1301-1307` | K3 GGUF names | Serialized tensor names | `crates/gguf/src/lib.rs:409-449` | Match |
| `src/llama-hparams.h:164-172` | K3 hparams | Latent, residual, gate, SiTU constants | `crates/engine/src/lib.rs:434-455` | Match after MLA tail correction |
| `src/models/delta-net-base.cpp:289-370` | `build_delta_net_autoregressive` | Recurrent KDA update | `pulsar_kernels.cu:5200-5263` | Equivalent implementation |

## Reference Forward Pass

The following is the ordered single-token graph reconstructed from the PR,
not a source-level copy:

```text
x = token_embedding(token)
for layer il = 0..92:
    prefix = x
    cur = attn_res_mix(prefix, attn_res_score[il], bank) if bank is nonempty
    if il % attn_res_block_size == 0:
        bank.push(prefix)                         # raw layer input
        checkpoint = true
    else:
        checkpoint = false

    a = RMSNorm(cur, attn_norm[il])
    if layer is KDA:
        q = SiLU(causal_conv(x @ Wq, q_conv, state_q))
        k = SiLU(causal_conv(x @ Wk, k_conv, state_k))
        v = SiLU(causal_conv(x @ Wv, v_conv, state_v))
        g_raw = f_b(f_a(a)) + dt_bias
        g = -5 * sigmoid(exp(A_log) * g_raw)       # K3 lower-bound form
        beta = sigmoid(a @ W_beta)
        q = L2Norm(q, 1e-12); k = L2Norm(k, 1e-12)
        attn = delta_rule(q, k, v, g, beta, ssm_state)
        gated = RMSNorm(attn, o_norm) * sigmoid(a @ W_full_gate)
        attn_out = gated @ W_o
    else:
        q = q_b(RMSNorm(q_a(a), q_a_norm))
        kv_raw = a @ W_kv_a_mqa
        kv = RMSNorm(kv_raw[:512], kv_a_norm)
        k_tail = kv_raw[512:576]
        q = [q_nope(128), q_tail(64)]
        k = [absorbed_k_nope(kv), k_tail]
        v = kv
        cache.append(RMSNorm(kv_raw[:512]), k_tail)
        attn = causal_attention(q, cache, scale=1/sqrt(192))
        attn_out = (attn * sigmoid(a @ W_mla_gate)) @ W_o

    prefix = attn_out if checkpoint else prefix + attn_out
    cur = attn_res_mix(prefix, ffn_res_score[il], bank) if bank is nonempty
    f = RMSNorm(cur, ffn_norm[il])

    if il < leading_dense_count:
        ff = SiTU(f @ W_gate, f @ W_up, beta=4, linear_beta=25) @ W_down
    else:
        identity = f
        latent = f @ routed_down                    # 7168 -> 3584
        logits = identity @ router                 # full-width router
        p = sigmoid(logits)
        selection = top_k(p + score_bias, 16)
        weights = normalize(p[selection]) * expert_scale
        routed = sum(weights[e] *
                     SiTU(latent @ gate_e, latent @ up_e, 4, 25) @ down_e)
        routed = RMSNorm(routed, routed_norm)
        routed = routed @ routed_up                  # 3584 -> 7168
        shared = SiTU(identity @ shared_gate,
                      identity @ shared_up, 4, 25) @ shared_down
        ff = routed + shared

    x = prefix + ff

cur = attn_res_mix(x, output_res_score, bank) if bank is nonempty
cur = RMSNorm(cur, output_norm)
logits = cur @ output
```

`SiTU(g,u) = beta*tanh(g/beta)*sigmoid(g) *
linear_beta*tanh(u/linear_beta)`. The up transform is disabled only when
`linear_beta <= 0`; K3 has `25`.

## Pulsar Forward Pass

Pulsar uses the same ordering while retaining SSD expert staging and its
custom quantized kernels:

```text
embed token into st.cur
for il:
    prefix = st.cur
    retrieve AttnRes from raw bank + prefix using normalized scores
    snapshot raw prefix at il = 0, 12, 24, ...
    RMSNorm -> KDA or compact-cache split-weight MLA
    checkpoint ? st.cur = attention_out : st.cur += attention_out
    retrieve FFN AttnRes
    RMSNorm
    layer 0: dense SiTU; other layers: latent MoE
    latent = f @ routed_down; router = f @ router_weight
    sigmoid(router), add bias for selection, top-16, normalize selected probs
    stream selected expert slabs; gate/up SiTU; down; apply route weight
    latent RMSNorm -> routed_up; add two shared SiTU experts
    st.cur += ffn_out
final AttnRes -> output RMSNorm -> output head
```

The CPU-Q8 and CUDA expert paths share the corrected attention, recurrent, and
residual graph. They differ in expert quantization and accumulation precision.
Both preserve route weights after expert down projection.

## Metadata Comparison

The exact Pulsar GGUF was attempted with the reference loader. Its metadata
dump reported:

| Field | Existing GGUF | llama.cpp K3 interpretation | Pulsar |
|---|---:|---:|---:|
| architecture | `kimi-k3` | K3 | K3 |
| block count | 93 | 93 | 93 |
| context length | 1,048,576 | 1,048,576 | parsed |
| embedding length | 7168 | 7168 | 7168 |
| attention heads | 96 | 96 | 96 |
| KDA/MLA discriminator | 93-element `head_count_kv` array | 0 = KDA, nonzero = MLA | same |
| KDA layers / MLA layers | 69 / 24 | same | same |
| vocab | 163840 | 163840 | parsed |
| RMS epsilon | 1e-5 | 1e-5 | parsed |
| conv kernel | 4 | 4 | 4 |
| KDA head dim | 128 | 128 | 128 |
| q LoRA rank | 1536 | 1536 | 1536 |
| KV LoRA rank | 512 | 512 | 512 |
| rope dimension count | 64 | 64-dimensional tail, no rotation | now 64-dimensional tail, `rot_dim=0` |
| MLA key length | 192 | 128 + 64 | 128 + 64 |
| stored key length | 576 | 512 + 64 | 576 |
| MLA value length | 128 | 128 | 128 |
| routed experts / active | 896 / 16 | 896 / 16 | 896 / 16 |
| expert FFN width | 3072 | 3072 | 3072 |
| latent width | `moe_latent_size=3584` | PR expects `expert_latent_length` | 3584 |
| shared experts | 2 | 2 | 2 |
| leading dense layers | 1 | 1 | 1 |
| SiTU beta / linear beta | 4 / 25 | 4 / 25 | 4 / 25 |
| AttnRes block | 12 | 12 | 12 |
| KDA gate lower bound | -5 | -5 | -5 |
| router function | sigmoid | sigmoid | sigmoid |
| router normalization | selected sigmoid probabilities normalized, scale 1 | same | same |

The `moe_latent_size` versus `expert_latent_length` naming difference is a
GGUF compatibility difference, not a K3 math difference. The PR checkout
therefore cannot load this exact existing file without a metadata alias.

## Tensor Layout Comparison

Reference `create_tensor` shapes are logical GGML shapes, listed in the same
order as the converter's GGUF dimensions:

| Tensor family | Reference logical shape | Pulsar interpretation | Quant/layout status |
|---|---|---|---|
| KDA Q/K/V | `[7168, 12288]` matrix | rows are output channels, input activation is 7168 | Q2/Q3 direct K-quant or converted dense quant |
| KDA conv | `[4, 1, 12288, 1]` or rank-3 equivalent | channel-major `[d_inner][d_conv]`, three independent states | equivalent reshape; causal tap `K-1` is current token |
| KDA `f_a`, `f_b` | `[7168,128]`, `[128,12288]` | same matrix orientation; `f_b` decoded F32 for custom use | equivalent |
| KDA A | `[96]` | one value per head | stored as `-exp(A_log)`; fixed compute sign |
| KDA norm | `[128]` | expanded across 96 heads | equivalent per-head RMSNorm |
| MLA q_a/q_b | `[7168,1536]`, `[1536,18432]` | same | equivalent |
| MLA kv_a | `[7168,576]` | split first 512 to normalized latent and last 64 to K tail | fixed from prior zero-tail interpretation |
| MLA split k_b/v_b | k `[128,512,96]`, v `[512,128,96]` | custom F32 absorbed layout | equivalent |
| routed gate/up | `[3584,3072,896]` | expert `e` offset is `base + e*expert_bytes`; each row has 3584 elements | Q2_K/Q3_K packed bytes preserved |
| routed down | `[3072,3584,896]` | each output row has 3072 elements and emits 3584 latent values | Q2_K/Q3_K packed bytes preserved |
| routed down/up projections | `[7168,3584]`, `[3584,7168]` | latent down before router experts; norm/up after reduction | equivalent |
| shared experts | gate/up `[7168,6144]`, down `[6144,7168]` | full-width input, two experts concatenated | equivalent |
| AttnRes scores | `[7168]` fused `norm*proj` | host computes weightless RMSNorm then dot with fused vector | equivalent |

Successful dimensions do not alone establish axis correctness; the expert
offset and row identity checks in the existing differential harness establish
the packed-slab identity. The reference's converter stacks experts in expert
ID order, matching Pulsar's `abs_offset + expert_id * expert_bytes` mapping.

## Operation Comparison

| Operation | llama.cpp | Pulsar CPU-Q8 | Pulsar CUDA | Classification |
|---|---|---|---|---|
| RMSNorm | weighted RMSNorm, epsilon 1e-5 | host/device weighted RMSNorm | CUDA RMSNorm | equivalent, arithmetic difference expected |
| AttnRes score | RMSNorm row, multiply fused score, sum | same host formula | same host formula | equivalent |
| AttnRes weighting | softmax over bank plus current; raw rows weighted | same | same | equivalent |
| Conv | causal `ssm_conv`, then SiLU | `sconv` gives `x+conv`, subtract x, then SiLU | same | equivalent expression |
| KDA decay | positive `exp(A_log)` magnitude in bounded gate | uses `abs(ssm_a)` to accept both persisted folded-sign conventions | same | corrected semantic defect |
| KDA Q/K norm | per-head L2 norm, epsilon `1e-12` | same | same | corrected semantic defect |
| KDA delta update | decay state, `d=(v-S*k)*beta`, rank-1 update, output `S*q/sqrt(dim)` | same | same | equivalent |
| KDA output | RMSNorm per 128-wide head, sigmoid full gate, output projection | same | same | corrected semantic defect |
| MLA scale | `1/sqrt(192)` | same | same | equivalent |
| MLA tail | 64 learned unrotated coordinates included in Q/K | preserved | preserved | corrected semantic defect |
| MLA history | compact causal KV history | cached latent and tail for split weights | same CUDA kernel | corrected semantic defect |
| MLA gate | sigmoid of normalized-input projection before output projection | same | same | equivalent |
| Router | sigmoid logits; bias affects selection only | same | same | equivalent |
| Top-k | top 16, deterministic lower-ID tie break | same selection kernel | same selection kernel | equivalent |
| Route weights | selected sigmoid probs, sum clamp, scale 1 | same | same | equivalent |
| Expert SiTU | activation on gate and up branches, then down | same | same kernel | equivalent |
| Route application | after down projection | after down projection | after down projection | equivalent |
| Expert reduction | rank-ordered graph adds | rank-ordered serial host add | serial CUDA default/diagnostic variants | expected precision/backend difference |
| Latent norm/up | norm reduced latent, then up projection | same | same | equivalent |
| Shared path | full-width input, two shared experts, add | same | same | equivalent |
| Final output | final AttnRes, RMSNorm, logits | same | same | equivalent |

## Discrepancies and Fixes

1. **Genuine K3 semantic discrepancy, fixed:** Pulsar set
   `qk_rope=0` because K3 is NoPE. The PR and actual GGUF retain
   `qk_rope_head_dim=64`; NoPE disables rotation, not the learned tail. This
   caused Pulsar to ignore the final 64 query/key coordinates and to use a
   512-wide KV-A projection against a 576-wide tensor.
2. **GGUF/conversion semantic discrepancy, fixed:** the PR converter folds
   `A_log` to `-exp(A_log)`. Pulsar multiplied by the stored negative value but
   applied `sigmoid` without the reference negation, reversing the decay gate's
   intended sigmoid argument. Pulsar now uses `-ssm_a` as `exp(A_log)`.
3. **PR-specific metadata detail, applicable as compatibility:** the PR
   requires `expert_latent_length`, while the existing Pulsar GGUF exposes
   `moe_latent_size`. Pulsar's parser already handles the latter; no change to
   model mathematics is needed.
4. **Expected backend difference:** CPU-Q8 and CUDA quantize activations at
   Q8_K boundaries and use different accumulation implementations. Existing
   layer-1 evidence is byte-identical for packed expert inputs and weights,
   with routed RMS about `9.5e-10` and hidden RMS about `1.8e-9`.
5. **Dynamic-reference unresolved:** llama.cpp's exact PR checkout builds in
   a CPU-only configuration but rejects the existing GGUF before graph build
   because of the latent metadata name. A temporary source alias was not
   introduced, so no external token or snapshot is claimed.
6. **Genuine AttnRes semantic discrepancy, fixed:** retrieval had overwritten
   the raw residual stream and the checkpoint bank survived across token
   positions. Retrieval now only feeds the sublayer and the bank is rebuilt
   through layer depth for every token.
7. **Genuine MLA semantic discrepancy, fixed for split weights:** Pulsar had
   computed only current-token attention. It now stores and consumes compact
   causal latent/tail caches. Fused `wkv_b` remains fail-closed until it has a
   cache-aware implementation.
8. **Genuine KDA normalization discrepancies, fixed:** Q/K L2 epsilon is
   `1e-12`, and output RMSNorm is independent for each 128-wide head.

## Dynamic Attempt

CPU build commands:

```bash
cmake -S . -B build-cpu -DGGML_CUDA=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_SERVER=OFF
cmake --build build-cpu --config Release -j4 --target llama-simple
build-cpu/bin/llama-simple -m /home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf \
  -n 1 -c 2048 -ngl 0 ''
```

The build succeeded. The loader found all 24 shards, 2760 tensors, and the
metadata listed above, then failed before model construction with:

```text
error loading model hyperparameters: key not found in model: kimi-k3.expert_latent_length
```

The CUDA build was also attempted in `build/`; it was still compiling CUDA
translation units when the execution time budget ended. This does not change
the static reference result or the exact-GGUF incompatibility.

## Remaining Uncertainty

- No three-way llama.cpp/Pulsar tensor snapshot is possible until the PR
  loader accepts the existing `moe_latent_size` metadata alias or a compatible
  GGUF is supplied.
- The old CPU-Q8 token `51960` and CUDA token `3592` comparison predates the
  residual, cache, KDA, state-reset, and XTML corrections. It remains useful
  historical evidence but is not an oracle for the corrected graph.
- Corrected CPU-Q8 generated `Paris`; corrected CUDA passed the seven-prompt
  behavioral suite and 32K server validation in
  `BEHAVIOURAL_VALIDATION.md`. CUDA is now the default expert backend; exact
  layerwise CPU-Q8/CUDA numerical parity remains an open diagnostic question.
- Fused `wkv_b` MLA is not cache-aware and is rejected explicitly. The tested
  Q2_K checkpoint uses split `wk_b`/`wv_b` tensors.
