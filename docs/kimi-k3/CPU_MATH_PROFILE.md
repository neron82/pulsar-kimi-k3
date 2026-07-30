# K3 CPU Math Profile

This report records the K3 host-execution investigation. It does not enable
`PULSAR_K3_GPU_MOE` and does not change model math, placement, cache policy, or
storage behavior.

## Measured Run

Command: the requested `CUDA_VISIBLE_DEVICES=GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b`
run with `PULSAR_GPU=0`, `PULSAR_PROFILE=1`, `PULSAR_PROFILE_DETAIL=layers`,
`--tokens 1 -n 1`, and the K3 Q2_K shard. The header confirmed an RTX 3090
with that UUID. The one-token invocation produces token 0 (`prefill`) and
token 1 (`decode`) in this CLI.

The final run measured:

| phase | wall | existing CPU work | measured categories | expert evaluations | packed weight bytes |
|---|---:|---:|---:|---:|---:|
| prefill | 277.584 s | 276.805 s | 267.150 s | 1,472 | 17,599,561,728 |
| decode | 279.432 s | 278.656 s | 268.763 s | 1,472 | 17,599,561,728 |

The category total includes the measured `CPU miscellaneous` remainder. The
separately reported expert-resolution time was 9.654 s and 9.894 s. H2D was
zero in both phases; D2H was 11,776 bytes. The run used 92 MoE layers, 16
selected experts per MoE layer, and one CPU thread.

## Actual Forward Path

The complete path is in `crates/engine/src/real/kimi_k3.rs`:

1. `forward_kimi_k3` embeds the token into `st.cur`, then loops over 93 layers.
2. AttnRes mixture and snapshots run in `attn_res_mix`; `rms_norm` performs
   attention normalization on CUDA.
3. KDA layers call `kda_layer_forward`; MLA layers call `k3_mla_step`. Their
   projections and elementwise kernels are CUDA dispatches, with several
   explicit D2H host round trips in the KDA correctness path.
4. The FFN input is normalized with CUDA `rms_norm`. Non-leading layers call
   `latent_moe_forward`.
5. `latent_moe_forward` performs latent-down and router projection on CUDA,
   runs `k3_router_select` on CUDA, then reads 16 IDs and 16 weights to host.
6. `StreamingStore::ensure_with` resolves three expert slabs per selected
   expert. With `PULSAR_K3_GPU_MOE` absent, it fills `resolved_host` with
   host byte vectors.
7. `k3_host_moe_compute` calls `k3_dequant_expert_bytes` for gate, up, and
   down, then executes scalar Rust loops for gate, up, SiTU, down, and the
   weighted accumulation. This proves selected expert inference is CPU-side.
8. Host latent RMS normalization reads `rt.latent_normed`, applies the norm,
   and writes back. Latent-up and shared-expert projections then dispatch CUDA
   kernels, followed by the CUDA residual update.
9. The final AttnRes mixture, output norm, output head, and sampling complete
   the token.

The host expert dimensions are latent `3584`, expert FFN width `3072`, and
one-token input. Each selected expert performs gate `[3584 x 3072]`, up
`[3584 x 3072]`, SiTU-GLU `[3072]`, down `[3072 x 3584]`, and accumulation
`[3584]`. The loops are serial, scalar Rust loops: no BLAS, Rayon, OpenMP,
or SIMD dispatch is used by this function. The packed slabs are copied from
the resolver into host `Vec<u8>` values, dequantized into temporary `Vec<f32>`
values, and are not host-mapped GPU inputs.

## CPU Category Breakdown

The detailed profiler now reports these non-overlapping host timers per layer:

| category | source operation |
|---|---|
| CPU expert unpack/dequantization | three `k3_dequant_expert_bytes` calls per selected expert |
| CPU expert gate projection | scalar `latent_host * gate_f32` loop |
| CPU expert up projection | scalar `latent_host * up_f32` loop |
| CPU expert activation | scalar SiTU gate/linear/tanh/sigmoid loop |
| CPU expert down projection | scalar `mid * down_f32` loop |
| CPU expert weighted accumulation | scalar `w_e * expert_out` loop |
| CPU latent normalization | host RMS calculation and elementwise normalization |
| CPU miscellaneous | non-overlapping residual of the enclosing host FFN timer after expert resolution and the above timers |

For the decode phase, measured categories accounted for 268.763 s of the
278.656 s enclosing CPU-work timer. The dominant per-layer measurements were
approximately dequant 1.25 s, gate 0.48 s, up 0.48 s, down 0.48 s, and
miscellaneous 0.20 s. Activation, accumulation, and latent norm were each
about 0.001 s or less per layer. Thus dequantization is the largest single
category, while the three projection/dequantization groups together dominate.
The residual is intentionally visible as `CPU miscellaneous`; it includes
host resolver bookkeeping/readback and timing not attributable to one math
loop, rather than being silently assigned to a projection.

The profiler detail output remains behind `PULSAR_PROFILE_DETAIL=layers`.

## CPU Versus CUDA

| operation | execution | format / dimensions |
|---|---|---|
| normalization, residual and attention kernels | CUDA | f32 vectors, hidden 7168; KDA/MLA dimensions from K3 metadata |
| latent-down and router projection | CUDA | 7168 to 3584 and 7168 to 896 |
| router selection | CUDA, then D2H IDs/weights | 896 logits, top-16 |
| storage/cache resolution | CPU/I/O, then no upload in current path | three packed slabs per selected expert |
| expert unpack/dequantization | CPU | Q2_K/Q3_K-compatible host dequant path into f32 |
| expert gate/up/down | CPU | 3584 -> 3072, 3584 -> 3072, 3072 -> 3584 |
| expert SiTU and weighted sum | CPU | 3072 and 3584 element scalar loops |
| latent norm/up and shared experts | CUDA except host latent norm | latent 3584, hidden 7168 |

The exact dominant function is `KimiK3::k3_host_moe_compute` at
`crates/engine/src/real/kimi_k3.rs:1433`; its gate/up/down loops are the
dominant CPU math. The measured thread field is 1 for every expert layer.

## Why H2D Is Zero

The active branch is the `else` branch at `kimi_k3.rs:1323`: resolver payloads
are copied into `resolved_host`, and `k3_host_moe_compute` consumes those
vectors. The `staging.write` call that increments H2D bytes exists only in
the `use_gpu` branch, which requires `PULSAR_K3_GPU_MOE` and a supported
Q2_K/Q3_K layout. No host-mapped weights are passed to CUDA and no
uninstrumented expert-weight transfer exists in this execution. The measured
H2D zero is therefore genuine, not a missing counter.

## Existing CUDA Expert Support

`crates/kernels/cuda/pulsar_kernels.cu` and the wrappers in
`crates/kernels/src/lib.rs` provide `moe_pair_swiglu` and `moe_down`, with
Q2_K and Q3_K dispatches, Q8_K activations, `ExpertPtrs` device pointers, and
single-token dimensions. `k3_gpu_moe_compute` in `kimi_k3.rs` can dispatch one
selected-expert list at a time and accepts grouped support elsewhere in the
generic DSV4 path.

The K3 packed expert slab geometry is three independent contiguous slabs
addressed as `abs_offset + expert_id * expert_bytes`; that pointer contract is
compatible with the CUDA kernels when each slab is staged in device memory.
The current K3 format selection is Q2_K/Q3_K, so no unpack/repack is needed
for the existing dot kernels. However, the existing kernel path is not a
mathematical replacement for the host reference in this tree: its documented
K3 path notes that the legacy activation implementation is approximate with
respect to SiTU. It also requires valid device-resident pointers and performs
single-token expert-list dispatch; it does not itself stream storage.

## VRAM Staging

The profiler measured one selected gate/up/down triple as **191,299,584
bytes** (182.4 MiB). Therefore the minimum packed expert storage is:

| staging choice | bytes | GiB |
|---|---:|---:|
| one selected expert triple | 191,299,584 | 0.178 |
| all 16 selected experts for one layer | 3,060,793,344 | 2.850 |
| double-buffered 16-expert layer | 6,121,586,688 | 5.701 |

The one-token CUDA scratch minimum from repository constants is approximately
70 KiB: Q8_K latent activation 4,088 bytes, 16-expert mid Q8_K scratch
56,064 bytes, output 14,336 bytes, plus pointer/weight arrays and the current
4,096-byte staging slack. This is negligible beside packed expert storage.
Adding a conservative 1 GiB CUDA runtime and allocation headroom gives about
3.85 GiB for one-layer 16-expert staging, or about 6.70 GiB for double
buffering, before other persistent model/runtime allocations. A 24 GiB RTX
3090 has sufficient capacity for the minimum staging choices, but this is a
capacity calculation only, not an implementation recommendation.

With no device expert cache and one layer at a time, moving the current
selection to GPU would transfer approximately **17.60 GB of packed expert
weights per token** (1,472 evaluations x the measured per-layer triple,
subject to cache hits). The observed storage path was 17.48 GB to 17.06 GB per
phase, consistent with this order of magnitude.

## Scope And Verification

Changed files are `crates/engine/src/lib.rs`,
`crates/engine/src/real/kimi_k3.rs`, and this document. No CUDA kernel,
mathematics, cache policy, residency, storage layout, batching, prefetching,
or scheduling was changed.

`cargo build --release -p engine`, `cargo test -p engine` (44 tests), and
`git diff --check` passed. The profile run completed on the requested RTX
3090 UUID and emitted 93 layer records per phase. No test source was changed.

Proposed commit message: `profile: decompose K3 CPU expert math`
