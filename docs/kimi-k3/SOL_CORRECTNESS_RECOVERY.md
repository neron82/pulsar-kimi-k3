# Kimi K3 Correctness Recovery

## Scope

This recovery started from Pulsar commit
`355d1c1fdd682232e0154011dab697d5e0ba4a4e` and the 24-shard model:

`/home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf`

The baseline CPU-Q8 completion for `The capital of France is` was the
repetitive sequence `abfabfabfabf weighs weighs weighs weighs weighs`. The
goal was semantic correctness, not performance tuning.

References used during the audit:

- WASTE: `/home/neron/waste-k3-reference`, commit
  `69315701f634648f7a790915a0a525ed8aabf218`.
- llama.cpp K3 PR checkout: `/home/neron/llama.cpp-kimi-k3-reference`, commit
  `7b990cf5721b2ecb865e65beb63413a25d73cd3e`.
- Official Kimi K3: `/home/neron/kimi-k3-official-reference`, commit
  `7c5be9599120d7993748de66a76128614f15f210`.

## Root Causes

The failure was not one quantization defect. Several shared graph semantics
were wrong:

1. AttnRes retrieval replaced `st.cur`. Retrieval is a sublayer input, while
   the raw residual prefix must remain the base of residual updates.
2. The AttnRes checkpoint bank survived across tokens. It is a depth-local
   bank for one token and must start empty on every forward position.
3. MLA computed attention from only the current token. The allocated compact
   KV caches were never written or read, so MLA had no causal history.
4. KDA output RMSNorm treated all 96 heads as one row. K3 normalizes each
   128-wide head independently using the shared 128-element norm weight.
5. KDA Q/K L2 normalization used the model RMS epsilon instead of the
   reference epsilon `1e-12`.
6. Persisted K3 GGUFs use both positive and negative folded `ssm_a`
   conventions. The bounded gate requires the positive `exp(A_log)`
   magnitude, so consuming the stored sign directly can reverse the gate.
7. A new sequence did not explicitly clear all KDA, convolution, MLA, and
   AttnRes runtime state.
8. The generic chat renderer did not emit K3 XTML. Raw instruction text is
   not equivalent to the model's trained chat protocol.

## Corrections

### Forward Graph

- AttnRes writes retrieval output to `rt.mix_out` and passes that buffer only
  to the following normalization/sublayer. `st.cur` remains the raw residual
  stream.
- `rt.res_bank_len` is reset before every token. `rt.reset()` is called when
  `pos0 == 0` to clear recurrent and cache state for a new sequence.
- Multi-token K3 prefill falls back to ordered one-token forwards. This is a
  correctness fallback that preserves recurrent semantics; it is not a
  performance claim.

### MLA

- Split `wk_b`/`wv_b` MLA now stores normalized KV latent rows and the learned
  unrotated 64-wide key tail in compact causal caches.
- `pulsar_k3_mla_cached_attn_split` computes the absorbed low-rank query,
  scores every cached position, applies causal softmax, and reconstructs the
  value output using the GGUF split-weight axis order.
- The fused `wkv_b` representation remains rejected explicitly because it
  does not yet have a cache-aware implementation. The tested Q2_K model uses
  split weights.

### KDA

- Q and K use per-head L2 normalization with epsilon `1e-12`.
- The decay gate multiplies by `abs(ssm_a)` to accept either persisted folded
  sign convention while preserving the required positive magnitude.
- Output RMSNorm runs as 96 independent rows of width 128.

### Tokenizer And Serving

- `ChatMarkers` recognizes K3 XTML tokens and renders system, user, assistant
  opening, and assistant-history messages.
- Non-thinking generation opens the XTML `response` channel immediately.
- K3 sampling stops on `<|close|>` at the end of response content; the model's
  `<|end_of_msg|>` token remains recognized as an end-of-generation token.

### Diagnostics And Placement

- `PULSAR_DEBUG_LOGITS` reports top-10 logits, top-1 margin, and NaN/Inf
  counts. `PULSAR_K3_DEBUG_A` reports the loaded `ssm_a` range.
- The automatic K3 resident-weight reserve increased from 1536 MiB to 2048
  MiB. On the RTX 3090 this leaves enough room for runtime state, the CUDA MoE
  staging slot, and its safety margin while keeping most dense weights
  resident.

## Evidence

The first corrected raw CPU-Q8 completion for `The capital of France is`
selected token `17374`, decoded as ` Paris`. Its top-1 margin was `3.0556`,
with no NaN or Inf logits. The final CPU-Q8 K3 XTML server response was
exactly `Paris`, with 33 prompt tokens and one completion token.

The final mostly resident CUDA server completed all seven deterministic XTML
smokes with the expected concise answers; see `BEHAVIOURAL_VALIDATION.md`.

The merged release workspace suite passed. It includes 47 engine tests, 17
CUDA kernel self-tests, 19 quant tests, and 10 tokenizer tests. Formatting and
`git diff --check` also passed.

## Remaining Limits

- Cache-aware MLA currently requires split `wk_b` and `wv_b` tensors.
- K3 prefill is sequential, so no prefill throughput claim is made.
- Raw, unformatted instruction prompts are not a supported substitute for
  K3 XTML chat prompts.
- This sprint establishes coherent behavior for the tested Q2_K model. It
  does not claim exact CPU-Q8/CUDA logit parity at every layer.
