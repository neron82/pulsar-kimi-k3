# Pulsar Kimi K3 Implementation Plan

> **Status:** execution authorized by user; implementation starts after this plan is written.
>
> **Working tree:** `/home/neron/projects/pulsar-kimi-k3`
>
> **Source baseline:** Pulsar `de343bc73e01162acd175ae454421cfce2193879`, including the pre-existing local CUDA build fix in `crates/kernels/build.rs`.
>
> **Reference implementation:** `AtomicBot-ai/atomic-llama-cpp-turboquant`, branch `feat/kimi-k3-support`, commit `b2f13d7be`.

## Goal

Add a new `KimiK3` inference family to Pulsar for text-only Kimi K3 GGUFs, initially targeting standard GGUF quant types (`Q8_0`, `Q2_K`, `Q3_K`) produced from MXFP4 checkpoints. Preserve Pulsar's NVMe expert streaming, tier cache, CUDA placement, and OpenAI-compatible serving architecture.

Native MXFP4 tensor execution and MoonViT vision input are explicitly deferred until the text-only path is correct and teacher-forced against the reference implementation.

## Non-goals for the first milestone

- No vision encoder or multimodal prompt path.
- No native MXFP4 CUDA matmul.
- No speculative decoding or MTP for K3.
- No changes to the original `/home/neron/projects/pulsar` checkout.
- No weakening of existing model-family paths.

## K3 contract

- 93 transformer layers: 69 KDA layers and 24 gated MLA layers.
- One leading dense FFN layer, then 92 Stable LatentMoE layers.
- 896 routed experts, top-16 active experts, 2 shared experts.
- Hidden size 7168, KDA head size 128, 96 heads.
- Latent expert size 3584, expert FFN size 3072.
- KDA safe gate and full-rank output gate.
- Gated MLA without RoPE and with an output gate.
- SiTU-GLU in dense, routed, and shared FFNs.
- Attention Residual snapshot/mixing before attention, before FFN, and at output.
- 1M context metadata must load without allocating a 1M-token dense KV cache for KDA layers.

## Architecture strategy

Introduce a dedicated `Family::KimiK3` path instead of adding K3 branches throughout the generic path. Reuse the following existing infrastructure:

- `DeviceBuf`, GGUF parsing, split-shard virtual files.
- `ExpertTensor`, host cache, VRAM tiers, prefetch and CPU expert lane.
- Quantized matmul primitives and existing MLA/GDN kernel conventions.
- Existing CLI/server and tokenizer contracts.

K3-specific code lives primarily in:

- `crates/engine/src/real/kimi_k3.rs`
- `crates/kernels/src/lib.rs`
- `crates/kernels/cuda/pulsar_kernels.cu`
- `crates/gguf/src/lib.rs` only when native MXFP4 or K3 metadata requires it.

## Execution phases

### Phase 0 — Baseline and handoffs

1. Record source and target Git status, toolchain, and test baseline.
2. Add a process log at `docs/kimi-k3/PROCESS_LOG.md`.
3. Create a K3 metadata/tensor-name handoff from the AtomicBot converter and graph.
4. Define file ownership so parallel workers do not edit the same files.

**Gate:** target tree is reviewable; no source files changed except process/plan artifacts.

### Phase 1 — GGUF contract and shape model

1. Add `Family::KimiK3`.
2. Parse K3 architecture metadata, including per-layer KDA/MLA pattern.
3. Add K3 shape fields: latent size, SiTU parameters, AttnRes block size, KDA dimensions, safe-gate bound, per-layer attention mode.
4. Add loader acceptance for K3 tensor names and standard quant types.
5. Add header fixtures/tests using a synthetic tiny K3-shaped GGUF header; no full model required.

**Gate:** synthetic K3 header parses, invalid/missing K3 metadata fails closed, existing GGUF tests remain green.

### Phase 2 — K3 runtime data structures

1. Add `KimiK3W` and per-layer weight structures.
2. Add KDA recurrent state and convolution state allocation/reset.
3. Add AttnRes scratch/state structures.
4. Add latent-MoE resident projections and expert slab geometry.
5. Generalize expert resolve dimensions instead of assuming hidden→FFN→hidden.

**Gate:** model allocation tests pass against synthetic tensor tables; no CUDA execution yet.

### Phase 3 — Correctness kernels and forward graph

Implement test-first, one primitive at a time:

1. SiTU-GLU kernel and host reference.
2. K3 router: sigmoid scoring, bias, top-16, normalization, 896 expert support.
3. KDA causal Conv1D.
4. KDA safe gate and beta calculation.
5. KDA delta-state recurrence and output gate.
6. NoPE gated MLA and compressed KV cache.
7. AttnRes scalar scoring, softmax mixture, and residual restart.
8. Latent-MoE down → routed experts → latent norm → up.
9. Shared expert addition and layer residual update.

**Gate:** deterministic tiny-model tests match a CPU/reference implementation for each primitive; CUDA self-tests pass where CUDA is available.

### Phase 4 — Integrated K3 forward

1. Add `forward_kimi_k3()` and dispatch from `forward_rows()`.
2. Preserve sequential semantics for KDA layers while allowing safe batching around dense/MLA work.
3. Make K3 KV/recurrent state reset correctly at position zero.
4. Add output norm/head path and tokenizer integration.
5. Add teacher-forced parity harness against the AtomicBot fork on a tiny/synthetic checkpoint.

**Gate:** greedy token IDs and logits agree within an explicitly documented tolerance on the tiny fixture; errors identify layer/primitive.

### Phase 5 — Streaming and performance integration

1. Route 896 expert slabs through existing host/VRAM tiers.
2. Add K3-specific slab sizing and cache accounting.
3. Add prefetch/read batching for the 48 selected expert matrices per token.
4. Keep KDA state resident and avoid accidental per-token host round trips.
5. Add warm/cold decode benchmark with raw I/O, cache hit, kernel, and token timings.

**Gate:** full K3 GGUF can load and run on a sufficiently provisioned machine; no OOM or out-of-bounds expert reads; performance report separates compute from storage latency.

### Phase 6 — Serving and documentation

1. Expose K3 through existing `pulsar-cli` and `pulsar-serve` without K3-specific API forks.
2. Document model requirements, text-only status, supported quant types, RAM/NVMe expectations, and deferred MXFP4/vision work.
3. Add rollback notes and a reference command line.

**Gate:** CLI and server smoke tests pass with a tiny fixture; README and process log reflect measured behavior only.

## Parallel work allocation

Workers must stay within their file authority and return a patch/diff plus exact test output.

- **Architecture/loader lane:** `crates/engine/src/lib.rs`, `crates/engine/src/real/kimi_k3.rs` skeleton, GGUF fixtures. Must not edit CUDA kernels.
- **Kernel lane:** `crates/kernels/src/lib.rs`, `crates/kernels/cuda/pulsar_kernels.cu`, kernel self-tests. Must not edit engine dispatch.
- **Quant/expert lane:** `crates/gguf/src/lib.rs`, `crates/quant/*`, expert slab geometry and standard-quant fixtures. Must not redesign KDA.
- **Reference/test lane:** `tests/`, `docs/kimi-k3/`, tiny reference harness and parity tooling. Must not change production runtime code.

The parent integrates only after re-reading every changed file, checking conflicts, and running the merged-tree gates.

## Verification commands

Baseline and every integration checkpoint:

```bash
git status --short --branch
git diff --check
cargo test --workspace
```

When CUDA is available:

```bash
PULSAR_CUDA_ARCH=<supported-arch> cargo test --workspace
cargo build --release -p engine
cargo build --release -p serve
```

Reference CPU build:

```bash
cmake -S /tmp/atomic-llama-cpp-turboquant-kimi \
  -B /tmp/k3-cpu-build \
  -DGGML_CUDA=OFF -DLLAMA_CURL=OFF
cmake --build /tmp/k3-cpu-build --target llama-cli -j2
```

No claim of successful GPU inference is valid without a real K3 model/fixture and captured logits or generated tokens.

## Current risks

- The current Pulsar router hard limit is 512 experts; K3 requires 896.
- Existing `Ffn::Moe` assumes a non-latent expert input/output geometry.
- Qwen35 GDN is not K3 KDA; only state/scheduling patterns are reusable.
- Existing MLA assumes RoPE-capable geometry and lacks the K3 output gate.
- AttnRes changes the residual contract across the layer loop.
- Full K3 GGUFs are approximately 1–3 TB depending on quantization.
- The local development host may not have `nvcc`; CUDA gates must report infrastructure blockers separately from code failures.

## Acceptance criteria for the first completed port

- `Family::KimiK3` loads a standard K3 GGUF header.
- A tiny K3 fixture runs through the full forward path.
- KDA, gated MLA, AttnRes, SiTU, latent MoE, shared experts, and 896-way routing are covered by focused tests.
- Existing Pulsar families remain regression-free.
- No native MXFP4 or vision claims are made.
- The final report includes exact files changed, commands run, test output, and known limitations.
