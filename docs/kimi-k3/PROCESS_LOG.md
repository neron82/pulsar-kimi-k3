# Kimi K3 Port Process Log

## 2026-07-28 — setup

- Created isolated working tree: `/home/neron/projects/pulsar-kimi-k3`.
- Source baseline: `de343bc73e01162acd175ae454421cfce2193879`.
- Preserved pre-existing source modification: `crates/kernels/build.rs`.
- Original `/home/neron/projects/pulsar` was not modified.
- Added implementation contract: `docs/kimi-k3/PLAN.md`.
- Baseline `git diff --check`: passed.
- Baseline `cargo test --workspace`: blocked in `crates/kernels` because this host has no `nvcc`; Rust crates before CUDA compilation built, with one pre-existing `unused_mut` warning in `crates/quant/src/iq.rs:133`.
- CUDA implementation cannot be claimed verified until a host with a working CUDA toolkit runs the kernel gates.

## 2026-07-28 — reference/test harness lane (Lane D)

### Files created (all under shared-tree authority)

| File | Purpose |
|------|---------|
| `tests/k3_contract_fixture.py` | Deterministic fixture generator — produces `k3_contract_fixture.json` |
| `tests/k3_contract_fixture.json` | Host-side reference vectors for all 7 K3 components (generated) |
| `tests/k3_parity_harness.py` | RED/GREEN test harness — Python self-test + AtomicBot reference comparison |
| `tests/k3_contract_parity.rs` | Rust-side parity test — cross-language RED phase (requires `serde_json` dev-dep) |
| `tests/k3_make_tiny_gguf.py` | Synthetic K3-shaped GGUF fixture for AtomicBot GREEN phase |

### Contract components covered

1. **SiTU-GLU** — `beta*tanh(x/beta)*sigmoid(x)` gate and `linear_beta*tanh(x/linear_beta)` up branch in dense FFN, routed experts, and shared experts
2. **Router top-k / sigmoid + renormalize** — sigmoid scoring, top-2 selection, weight renormalization
3. **KDA safe gate** — `g1 = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))`
4. **KDA full-rank output gate** — `g2 = g_proj(x)`; `out = RMSNorm(attn) * sigmoid(g2)`
5. **NoPE gated MLA output gate** — `attn = attn_pregate * sigmoid(g_proj(x))`
6. **Latent-MoE dimension flow** — `down → experts (SiTU-GLU) → RMSNorm → up`
7. **AttnRes mixture** — softmax over bank + prefix scored by `sum(k * (norm_w * proj_w))`

### RED phase results

```
$ python3 tests/k3_parity_harness.py red
✅ SiTU-GLU: max_diff=0.00e+00
✅ Router top-k: idx_match=True, weights_match=True
✅ KDA safe gate: max_diff=0.00e+00
✅ KDA output gate: max_diff=0.00e+00
✅ MLA output gate: max_diff=0.00e+00
✅ Latent-MoE: max_diff=0.00e+00, idx_match=True
✅ AttnRes mixture: max_diff=0.00e+00, scores_match=True, probs_match=True

Overall: ✅ ALL PASS
```

All 7 components pass with exact (deterministic) match at tolerance `1e-10`.

### GREEN phase (AtomicBot reference)

The GREEN phase is intentionally gated. It requires:
1. A CPU build of the AtomicBot fork at `/tmp/k3-cpu-build`
2. A real, quantized Atomic-compatible K3 GGUF selected via `PULSAR_K3_ATOMIC_MODEL`

The synthetic `tests/k3_tiny_fixture.gguf` is F32/zero-weight contract material for parser/loader rejection tests only; it is not a valid Atomic inference checkpoint and must not be passed to GREEN.

```bash
export PULSAR_K3_ATOMIC_MODEL=/path/to/real-k3-q8.gguf
python3 tests/k3_parity_harness.py green
# or with auto-build:
python3 tests/k3_parity_harness.py green --build
```

### Rust parity test

The Rust test `tests/k3_contract_parity.rs` recomputes all 7 components in pure Rust and compares against the JSON fixture. To enable:

```bash
# Add serde_json as workspace dev-dependency:
# In Cargo.toml: [workspace.dependencies] serde_json = "1"
# In tests/k3_contract_parity.rs: serde_json is used via dev-dependency
cargo test --test k3_contract_parity -- --nocapture
```

Note: `serde_json` is not currently a workspace dependency; it exists only in `crates/serve/Cargo.toml`. The Rust test is provided as a ready-to-compile reference — add `serde_json` to `[dev-dependencies]` in the workspace or the crate that owns the test directory.

### Tiny GGUF fixture

`tests/k3_tiny_fixture.gguf` (296 KB) is a synthetic K3-shaped GGUF with:
- 4 layers: 1 KDA + 3 gated MLA
- Scaled-down dimensions matching the contract fixture
- All 103 tensors declared with correct K3 tensor names and shapes
- Zero-initialized weights (synthetic — for load/parse testing only)

### Key formulas extracted from reference

From `kimi-k3.cpp` (AtomicBot `b2f13d7be`):

**KDA safe gate** (lines 355–367):
```
f_a = x @ W_f_a           # [n_embd] → [head_dim]
f_b = f_a @ W_f_b         # [head_dim] → [d_inner]
g1_raw = f_b + dt_bias    # [d_inner]
g1 = lower_bound * sigmoid(exp(A_log) * g1_raw)  # per-head, per-head_dim
```

**KDA output gate** (lines 398–410):
```
g2 = x @ W_gate           # [n_embd] → [d_inner]
out = RMSNorm(attn) * sigmoid(g2)  # elementwise
out = out @ W_o
```

**MLA output gate** (lines 476–483):
```
gate = sigmoid(x @ W_gate)
attn = attn_pregate * gate
out = attn @ W_o
```

**AttnRes mixture** (lines 235–278):
```
v = [bank_rows..., prefix]  # stacked
k = RMSNorm(v)
sw = norm_w * proj_w        # elementwise
scores = k @ sw             # per-row scalar
probs = softmax(scores)
out = sum_r probs_r * v_r
```

**Latent-MoE** (lines 506–544):
```
latent = x @ W_down
router on FULL hidden state (x), not latent
moe_out = sum_e w_e * SiTU_Gate(latent @ W_gate_e) * SiTU_Linear(latent @ W_up_e) @ W_down_e
moe_out = RMSNorm(moe_out) @ W_up
shared = SiTU_Gate(x @ W_gate_sh) * SiTU_Linear(x @ W_up_sh) @ W_down_sh
out = moe_out + shared
```

- The authoritative HF config was rechecked: `situ_beta=4.0`, `situ_linear_beta=25.0`, `attn_res_block_size=12`, `kda.gate_lower_bound=-5.0`, `kda.head_dim=128`.
- The AtomicBot converter registers the GGUF architecture as `kimi-k3`; its `attention.head_count_kv` array is the per-layer discriminator (`0 = KDA`, nonzero = gated MLA).
- The first engine skeleton had stale placeholder defaults (`1/1/4/0.0625`) and was corrected before integration verification.

## Operating rules

- Parallel workers have explicit file ownership; no two workers edit the same production file.
- Every worker must return changed paths, a diff/commit handle, and exact test output.
- Parent re-reads and verifies all worker output before integration.
- Failed CUDA/toolchain gates are reported separately from code failures.
- No native MXFP4 or vision support is claimed in the first milestone.

## 2026-07-28 — verified loader and KDA slice

- Added `docs/kimi-k3/ATOMIC_CONTRACT.md` from AtomicBot `b2f13d7be` plus HF config.
- Canonicalized K3 architecture and metadata to `kimi-k3`; production loading requires the exact `attention.head_count_kv` array with one entry per layer.
- Added strict K3 loader path in `crates/engine/src/lib.rs`: KDA/MLA weights, AttnRes weights, dense FFN or latent-MoE slabs; generic FFN duplication is avoided.
- Added exact KDA delta-rule CUDA primitive with CPU-reference tests, including H_v/H_k repeat cases.
- Corrected host and documentation SiTU formula to `beta*tanh(x/beta)*sigmoid(x)` and `linear_beta*tanh(x/linear_beta)`.
- Added gated MLA absorbed-attention primitives for split `W_k_b/W_v_b` and fused `W_kv_b` single-token paths; NoPE preserves the rope-tail dimensions but performs no rotation.
- Added explicit K3 sigmoid and elementwise-multiply helpers; the engine no longer abuses Qwen35's `x *= sigmoid(gate)` helper for a plain sigmoid/multiply gate.
- Resized K3 runtime scratch buffers for canonical MLA intermediates (`n_head * qk_dim`, `n_head * value_mla`, and fused KV width); this was caught by review before forward wiring.
- The K3 MLA engine helper is not yet wired into the full forward and currently assumes F32 projection buffers; standard-quantized MLA weight dispatch is an explicit next gate, not silently treated as F32.
- Verification: `cargo test --workspace` passed with `PATH=/usr/local/cuda/bin:$PATH`; K3 GGUF tests 19/19, CUDA selftests 15/15, engine tests 17/17, quant K3 test 1/1, RED harness all 7 components pass.
- Full K3 forward remains intentionally unimplemented: gated MLA, AttnRes execution, latent-MoE execution, and output projection are still explicit next slices.

## 2026-07-30 — AtomicChat Q2_K real-model continuation

- Verified `/home/neron/models/kimi-k3/Q2_K/` contains all 24/24 shards, no missing names, total `1,008,952,301,088` bytes.
- Parsed all shard headers with a bounded prefix reader. Tensor types: `F32=1299`, `Q2_K=1020`, `Q3_K=347`, `Q4_0=93`, `Q6_K=1`.
- The first quant compatibility slice was implemented without discarding the pre-existing K3 work:
  - `upload_k3_mla_q8()` now normalizes K2–K6 source tensors through the existing K→Q8 path.
  - K3 latent-MoE host resolve now dequantizes Q2_K/Q3_K (and guarded Q4_K/Q5_K/Q6_K/Q8_0/Q4_0 paths) with strict packed-buffer validation instead of assuming Q8_0.
  - K3 token embedding accepts the actual AtomicChat Q2_K source and normalizes it to the existing Q8_0 embedding kernel contract.
  - K3 absorbed MLA path now handles the actual Q4_0 `ssm_f_b`/`attn_k_b` class through a tested Q4_0 block dequantizer.
- Parent verification:
  - strict RED test first failed to compile because the new helper did not exist (`k3_dequant_expert_bytes`); after implementation, focused Q2_K short-buffer and quant-ID tests passed.
  - Q4_0 block dequant test passed.
  - `/home/neron/.cargo/bin/cargo test --workspace --quiet`: all runnable tests passed; CUDA device tests remain ignored because this host has no CUDA device.
  - `git diff --check`: clean.
- Real CLI load smoke against the 24-shard model reached CUDA initialization and then exposed the next missing source type: `blk.0.ssm_f_b.weight` was Q4_0. That path is now patched and tested.
- A fresh real-model load is currently running with the rebuilt `pulsar-cli`; no inference success is claimed until it completes and produces logits/tokens.
- The first full real load reached `cudaMalloc(0.05 GB) failed on device 0` after 13:22. The host maps CUDA device 0 to the 8-GiB RTX 3060 Ti and device 1 to the 24-GiB RTX 3090. The next smoke is pinned with `PULSAR_GPU=1`.

## 2026-07-30 — WASTE cross-check

- Reviewed `https://github.com/sqliteai/waste` at commit `49c71e327654e8f99d35893ceba2748b75fc228f` (2026-07-30). It is an independently verified CPU-only C engine for the same Kimi K3 family, using a custom `.waste` container rather than GGUF.
- High-value transferable ideas:
  - Pack each layer's gate/up/down expert payload into one 4-KiB-aligned record so one direct-I/O read obtains one complete expert. Pulsar's current GGUF path treats the three K3 expert tensors separately; a K3 sidecar bank could reduce read/syscall overhead without changing the GGUF loader.
  - Use an explicit bounded cache policy with frequency plus recency (WASTE calls it LFRU), not plain LRU. Their measured K3-relevant result was materially better than LRU at the cache margin.
  - Persist a usage/hotlist file and warm the cache on the next open. Pulsar already has warm/cache machinery, but WASTE's learned per-(layer,expert) evidence is a useful target for a later persistent policy rather than a blind preload.
  - Size cache around one token's routed working set and avoid spending every last byte. WASTE measured the K3 floor as 16 experts × 92 MoE layers × ~11.8 MB ≈ 17 GB per cold token; its later measurements show paging can make a larger cache dramatically slower. Pulsar's VRAM has no OS paging cliff, but the same whole-working-set and allocator-reserve logic applies to device cache sizing.
  - Stream `embed_tokens` row-wise when only one vocabulary row is consumed per token. WASTE freed ~1.1 GB this way. Pulsar's current K3 compatibility path normalizes the full Q2_K embedding table into a resident Q8 buffer; row-wise device staging or a host-side row path is a concrete future VRAM reduction candidate.
  - Chunked prefill should union routed experts across tokens and read each expert once. WASTE measured 3.3x fewer expert reads on a 32-token chunk with identical logits. K3's recurrent KDA/AttnRes state complicates this in Pulsar, so it is a later optimization, not a drop-in loop rewrite.
  - Never materialize sub-4-bit expert weights if the arithmetic can consume packed codes through lookup tables. WASTE's VQ3R/LUT path removed dequantization from its CPU hot loop. This is not directly reusable for the existing Q2_K/Q3_K GGUF block format or CUDA kernels, but the principle is relevant to a future K3 expert-bank format.
- Important non-transferable parts:
  - WASTE's VQ3R format, converter, tokenizer sidecars, and container-specific parser are a different product boundary. They do not make the current GGUF weights directly CUDA-readable.
  - Its 27.28-GB resident trunk is host RAM, not VRAM; it proves the model's dense trunk is too large for a 24-GB GPU, but does not imply mapped pinned host reads will be fast on PCIe. A CUDA path should keep hot dense weights resident or stage them deliberately, not blindly turn the whole trunk into UVA loads.
  - The Docker/container setup is packaging, not an inference insight.
- Useful independent correctness reference: WASTE reports an end-to-end K3 CPU oracle agreement of 3.56e-06 final-logit max error and identical argmax/top-5, but its VQ3R/custom-container arithmetic is not a direct numerical oracle for Pulsar's Q2_K GGUF path.
- Conclusion: do not port WASTE wholesale. The strongest Pulsar follow-ups are (1) row-wise/streamed K3 embeddings, (2) a packed per-layer expert sidecar or coalesced-read path, (3) LFRU plus persisted routing telemetry, and (4) chunked-prefill expert union after the current K3 forward is correct. The active host-pinned resident-weight experiment remains the immediate VRAM gate.

## 2026-07-30 — host-pinned K3 load experiment

- The coding-agent partial added `PULSAR_K3_HOST=1` routing for K3 layer weights, embedding, absorbed tensors, and global `output_norm`/`output`; default non-K3 placement remains unchanged.
- Parent review replaced the agent's env-only tests with real CUDA placement tests: `DeviceBuf::is_pinned()` is asserted for a 256-byte host-mode allocation and false for the default VRAM allocation. Both passed; CPU-only builders would skip after `device_count() == 0`.
- `/home/neron/.cargo/bin/cargo test --workspace --quiet`: passed after the host-mode fixes. `cargo build --bin pulsar-cli --quiet` and `git diff --check` also passed.
- Real load command: `PULSAR_GPU=1 PULSAR_K3_HOST=1 ./target/debug/pulsar-cli -m /home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf -p test -n 1`.
- Result: CUDA device 1 initialized and no tensor/CUDA allocation error appeared, but the process remained at ~99.8% CPU for 40:17, grew to `64,685,900 KiB` RSS (~61.7 GiB reported by `ps`, briefly ~65.4 GiB before termination), emitted no `Model loaded`/logit/token output, and was stopped at the agreed ~65 GiB host-RAM ceiling. This is a placement/load experiment, not a successful model load.
- Interpretation: mapped pinned placement works for small buffers, but the current K3 implementation still expands too much resident Q2_K/Q3_K material into Q8/F32 host buffers. WASTE's ~27.28 GiB resident Q4 trunk is not directly comparable; its custom container keeps a much denser trunk representation. The next useful direction is selective retention/row streaming/packed-Q2 or a sidecar expert bank, not simply pinning every K3 weight.

## 2026-07-30 — full K3 implementation mission kickoff

- User acceptance target: real `pulsar-cli` load of the 24-shard AtomicChat Kimi-K3 Q2_K model, one successful inference step with visible generated answer, SSD-streaming pipeline, direct Q2_K/Q3_K CUDA matmul without the current Q8 conversion step, RAM/VRAM hot-expert tiering, at least 0.33 tok/s cold, and a measured 2–3x hot-expert improvement.
- Baseline resources captured before implementation: RTX 3060 Ti device 0 (8 GiB; ~7.0 GiB free), RTX 3090 device 1 (24 GiB; ~23.6 GiB free), 94 GiB host RAM, ~218 GiB free on the model NVMe filesystem. Honcho health is `{"status":"ok"}` on `127.0.0.1:8001` and must remain alive.
- Current tree is a large uncommitted K3 rewrite; no existing changes are to be discarded or committed by delegated workers. Four bounded lanes were dispatched under delegation `deleg_156f75cb`: read-only runtime audit, kernel-only Q2/Q3 CUDA primitives, stream-crate-only SSD pipeline, and read-only hot-tier architecture audit.
- Parent verification gate remains strict: delegated summaries are hypotheses until the merged tree compiles, focused tests and full workspace tests pass, the real model produces output, and cold/hot timings are captured from completed runs rather than exit codes alone.

## 2026-07-30 — SSD stream lane reconciliation

- The stream worker's first partial patch added an unsafe/incomplete `PipelineBatch` API and did not compile: duplicate fields, stale test signatures, and mutable-borrow errors. It was not accepted.
- Parent replaced that partial block with the smaller existing contract: Linux `io_uring`, `O_DIRECT`, bounded queue depth, optional CUDA-pinned buffer allocator, split-shard routing, and `Fetcher::fetch_each` callback overlap. This is the API already consumed by the engine's `Prefetcher` and `StreamingStore`.
- Verification: `cargo test -p stream --quiet` passed; `cargo test -p kernels --quiet` passed with Q2_K/Q3_K direct matmul selftests; `cargo test --workspace --quiet` passed. The SSD stream is therefore compile- and unit-verified, but the full K3 loader has not yet been wired to use direct dense Q2/Q3 weights or benchmarked end-to-end.

## 2026-07-30 — direct KQ engine integration gate

- The K3 engine now carries `K3DenseWeight { DeviceBuf, K3WeightQuant }` metadata through the K3 loader and forward paths. Native Q2_K/Q3_K bytes stay native; activations are quantized once into reusable Q8_K scratch and dispatched through the existing CUDA `matmul_kq`. Q8_0/F32/custom paths retain explicit fallbacks.
- Parent repaired the first partial contract merge (27 compile errors) rather than accepting casts or silently re-expanding everything to Q8_0.
- Verification: `cargo check --release --bin pulsar-cli` passed; `cargo test -p engine --quiet` passed 39/39; `cargo test --workspace --quiet` passed all reported suites, including 17/17 CUDA selftests and Q2_K/Q3_K matmul checks.
- E2E command launched on GPU 1 with raw direct-KQ path and bounded cache settings: `PULSAR_GPU=1 PULSAR_DEV_CACHE_GB=12 PULSAR_HOST_POOL_GB=64 PULSAR_BATCH=1 timeout --signal=TERM --kill-after=30 1800s ./target/release/pulsar-cli -m /home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf -p 'What is the capital of France?' -n 4 --ctx 128`. Raw output is being captured in `/tmp/pulsar-k3-e2e-direct-kq.log`; no inference success is claimed until `loaded` plus generated tokens are observed.

## 2026-07-30 — runtime gate and tiering blocker

- The first VRAM-resident E2E attempt exited with evidence, not speculation: `split gguf: 24 shards as one virtual file`, `using CUDA device 1`, then `cudaMalloc(0.18 GB) failed on device 1`. The direct-KQ path is compiling, but the resident K3 trunk overcommits the 24-GiB card.
- The host-pinned retry is running with `PULSAR_K3_HOST=1`, `PULSAR_DEV_CACHE_GB=4`, and the real host-cache variable `PULSAR_CACHE_GB=64` in `/tmp/pulsar-k3-e2e-host-direct-kq.log`. It has reached the 24-shard split/load phase; no `loaded` or token output yet.
- Read-only tier audit confirmed the critical performance gap: `latent_moe_forward()` in `crates/engine/src/real/kimi_k3.rs` still performs synchronous `VFile::read_exact_at`, host dequantization, and O(n²) CPU matmul for every selected expert. It bypasses `StreamingStore`, `DeviceSlabCache`, `ExpertTier`, and `ExpertPtrs` despite those paths being implemented for other MoE families.
- Geometry evidence: one Q2_K expert slab is ~3.45 MiB; a gate/up/down triple is ~10.35 MiB. A 3090 tier can hold roughly 2,175 triples (~22 GiB); the 3060 Ti can hold roughly 500 after fixed runtime allocations; 46 GiB pinned host cache can hold roughly 4,500 additional triples. Those capacities are useful only after K3 resolves through the shared tier/cache path.
- A file-bounded K3 latent-MoE GPU/cache integration lane is dispatched as `deleg_38a0b858`; it may reuse existing `moe_pair_swiglu`/`moe_down` contracts but may not add untested CUDA kernels.
- The first host-pinned forward failed because `DeviceBuf::read()` used `cudaMemcpy(..., D2H)` even when `DeviceBuf` held mapped pinned host memory. Parent added the symmetric CPU copy path already present in `DeviceBuf::write()`. `cargo test -p kernels` (17 CUDA selftests) and `cargo test -p engine` (39 tests) pass after the fix.
- A rebuilt release binary is now running the second host-pinned E2E attempt as `proc_73d1b3cc39fa`, capturing `/tmp/pulsar-k3-e2e-host-direct-kq-v2.log`.

## 2026-07-30 — persistent K3 tiering and SiTU GPU gate

- Parent merged the file-bounded latent-MoE integration: selected expert slabs resolve through `StreamingStore::ensure_with()` instead of per-expert synchronous `VFile::read_exact_at`; Q2_K/Q3_K triples can dispatch through existing `ExpertPtrs`, `moe_pair_swiglu`, and `moe_down` kernels. Unsupported quant/layout combinations retain the host correctness path.
- `State.dev_cache` and `State.staging` are now threaded into `latent_moe_forward()`. The K3 capacity solver also accounts for `Attn::KimiK3` expert tensors; startup reports a nonzero staging budget instead of the old 256-byte dummy buffer.
- The first GPU lane used `act_op=0` (SiLU) as an approximation. This was rejected for correctness: K3 requires SiTU-GLU with `situ_beta=4.0` and `situ_linear_beta=25.0`.
- Added `act_op=4` to `pulsar_glu()` in `crates/kernels/cuda/pulsar_kernels.cu` and switched the K3 GPU dispatch to it. Other gated-FFN act ops are unchanged.
- Verification after the SiTU change: `cargo test --workspace --quiet` passed, including 41 engine tests, 17 CUDA selftests, Q2_K/Q3_K matmul checks, and `git diff --check`.
- Performance evidence remains a blocker: previous opt-in timing measured approximately 0.60–0.68 s per K3 layer (~56–63 s per complete forward). No generated token or cold/hot throughput claim is made.
- Resource incident: the historical 64-GiB host-cache run caused a cgroup global OOM; kernel evidence records `pulsar-cli` with ~84.9 GiB shmem. Subsequent bounded runs use `PULSAR_CACHE_GB=8` or lower. Honcho remained healthy at `127.0.0.1:8001`.
- A bounded real-model run with the SiTU path and 8-GiB host cache was attempted, but was killed before `loaded`/forward output; no inference success is claimed.
