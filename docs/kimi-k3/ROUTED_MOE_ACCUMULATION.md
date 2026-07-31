# K3 Routed-MoE Accumulation Investigation

This investigation is limited to layer 1, token 0, on the RTX 3090 selected
with `PULSAR_GPU=GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b`. The focused runs
used `--tokens 1 -n 1 --ctx 2048 --temp 0 --top-p 1 --min-p 0 --seed 1` and
stopped at layer 1 with `PULSAR_K3_COMPARE_STOP=1`.

## Result

The first CPU-Q8 versus CUDA accumulator mismatch is rank 0, global expert
498, local slot 0. This is not an accumulation-order defect: the unweighted
expert vectors already differ. At rank 0, index 0 is:

| value | CPU-Q8 | CUDA |
|---|---:|---:|
| unweighted expert output | -8.286854625e-2 | -8.286876976e-2 |
| weighted output | -1.365943439e-2 | -1.365947165e-2 |
| accumulator after rank 0 | -1.365943439e-2 | -1.365947165e-2 |

The final `7.254e-6` accumulator difference is therefore classification 1:
different expert-vector inputs to the accumulator. The difference grows at
ranks 7, 14, and 15; it is not created by a reordered or parallel sum.

## Expert Order

CPU-Q8 and CUDA used the same rank order and the same routing weights:

| rank | global ID | local slot | routing weight |
|---:|---:|---:|---:|
| 0 | 498 | 0 | 0.164832562 |
| 1 | 730 | 1 | 0.104517967 |
| 2 | 748 | 2 | 0.079839930 |
| 3 | 15 | 3 | 0.078882933 |
| 4 | 66 | 4 | 0.052121956 |
| 5 | 873 | 5 | 0.052849110 |
| 6 | 236 | 6 | 0.051112358 |
| 7 | 14 | 7 | 0.050274745 |
| 8 | 303 | 8 | 0.045444570 |
| 9 | 104 | 9 | 0.050911084 |
| 10 | 212 | 10 | 0.045883402 |
| 11 | 162 | 11 | 0.045135010 |
| 12 | 5 | 12 | 0.044431228 |
| 13 | 271 | 13 | 0.043244559 |
| 14 | 32 | 14 | 0.043520983 |
| 15 | 658 | 15 | 0.046997521 |

The weights are host `f32` readbacks from the CUDA router output. The CPU-Q8
run and CUDA run have bit-identical selected IDs and weights.

## Rank Progression

These metrics compare the captured CPU-Q8 and CUDA `accum_after` vectors.
`first` is the first exact bit mismatch, not a tolerance threshold.

| rank | max abs | mean abs | RMS | cosine | norm ratio | first |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1.639e-7 | 2.306e-8 | 2.931e-8 | 1.000000000 | 1.000001717 | 0 |
| 1 | 1.788e-7 | 2.590e-8 | 3.284e-8 | 1.000000000 | 1.000001726 | 0 |
| 2 | 1.788e-7 | 2.665e-8 | 3.390e-8 | 1.000000000 | 1.000001718 | 0 |
| 3 | 1.788e-7 | 2.872e-8 | 3.668e-8 | 1.000000000 | 1.000001782 | 0 |
| 4 | 1.863e-7 | 3.019e-8 | 3.854e-8 | 1.000000000 | 1.000001863 | 0 |
| 5 | 2.384e-7 | 3.555e-8 | 4.609e-8 | 1.000000000 | 1.000001944 | 0 |
| 6 | 2.310e-7 | 3.559e-8 | 4.609e-8 | 1.000000000 | 1.000001892 | 0 |
| 7 | 4.325e-6 | 9.397e-7 | 1.156e-6 | 0.999999998 | 0.999995602 | 0 |
| 8 | 4.330e-6 | 9.397e-7 | 1.156e-6 | 0.999999998 | 0.999995653 | 0 |
| 9 | 4.335e-6 | 9.397e-7 | 1.156e-6 | 0.999999998 | 0.999995587 | 0 |
| 10 | 4.338e-6 | 9.398e-7 | 1.156e-6 | 0.999999999 | 0.999995657 | 0 |
| 11 | 4.340e-6 | 9.398e-7 | 1.157e-6 | 0.999999999 | 0.999995753 | 0 |
| 12 | 4.335e-6 | 9.399e-7 | 1.157e-6 | 0.999999999 | 0.999995848 | 0 |
| 13 | 4.342e-6 | 9.398e-7 | 1.157e-6 | 0.999999999 | 0.999995931 | 0 |
| 14 | 6.406e-6 | 1.400e-6 | 1.738e-6 | 0.999999997 | 0.999994227 | 0 |
| 15 | 7.254e-6 | 1.492e-6 | 1.854e-6 | 0.999999997 | 0.999994348 | 0 |

The final routed-MoE CPU-Q8 versus current CUDA metrics are max `7.254304e-6`,
mean `1.491955e-6`, RMS `1.854037e-6`, cosine `0.999999997043`, norm ratio
`0.999994348349`, first mismatch `0`. Layer-1 hidden metrics are max
`4.5422465e-5`, mean `9.337556e-6`, RMS `1.1665878e-5`, cosine
`0.999999914223`, norm ratio `1.000005567666`, first mismatch `0`.

## Implementations

CPU-Q8 is `Model::k3_cpu_q8_moe_compute` in
`crates/engine/src/real/kimi_k3.rs`. It computes one expert at a time, stores
the unweighted down output in a host `Vec<f32>`, and performs:

```text
moe_acc[k] += weights[rank] * expert_out[k]
```

Ranks are visited `0..16`, the accumulator is `vec![0.0f32; moe_latent]`,
and the output is materialized before weighting. There is no BLAS, Rayon, or
parallel reduction. The operation has no explicit FMA. Release disassembly
contained no `vfmadd` instruction in the active binary; the no-FMA CUDA
diagnostic also reproduced the host result bit-for-bit for the captured data.

The active CUDA path is `Model::k3_gpu_moe_compute`. `moe_pair_swiglu` and
`moe_down` are launched once per selected expert with `n_used=1`; their output
is synchronously read into a host `Vec<f32>`, then the cross-expert sum uses the
same host expression and rank loop as CPU-Q8. The CUDA down kernel uses one
warp per output row for its within-expert quant-block dot reduction, but no
thread from one expert writes the cross-expert accumulator. No cross-expert
atomic operation is used.

The debug `serial` kernel is `moe_accum_serial_kernel` in
`crates/kernels/cuda/pulsar_kernels.cu`. It assigns one thread to each output
element and loops over ranks serially. Its normal expression may contract
under the CUDA build's `--use_fast_math`. `serial-nofma` uses `__fmul_rn` and
`__fadd_rn` for separate rounded operations. `f64-reference` performs the
rank-ordered sum in host `f64` and converts to `f32` once at the end.

## Identical-Vector Isolation

The CUDA expert outputs and weights were captured once and fed to CPU F32,
CPU F64, current host accumulation, serial CUDA, and serial no-FMA CUDA.
This removes gate, up, SiTU, down, and independent expert-vector computation
from the accumulation comparison.

| comparison | max abs | mean abs | RMS | cosine | norm ratio | first |
|---|---:|---:|---:|---:|---:|---:|
| CPU F32 vs current host | 0 | 0 | 0 | 1.000000000 | 1.000000000 | none |
| CPU F32 vs F64 reference | 1.490116e-8 | 1.205643e-9 | 2.145173e-9 | 1.000000000 | 0.999999996 | 1 |
| current host vs serial CUDA | 7.450581e-9 | 5.651860e-10 | 1.179329e-9 | 1.000000000 | 0.999999998 | 0 |
| F64 reference vs serial CUDA | 1.490116e-8 | 1.188081e-9 | 2.137634e-9 | 1.000000000 | 1.000000002 | 0 |
| F64 reference vs serial no-FMA | 1.490116e-8 | 1.205643e-9 | 2.145173e-9 | 1.000000000 | 1.000000004 | 1 |

`serial-nofma` is bit-identical to current host accumulation for this captured
vector set. Replacing the production result with serial or F64 does not
materially improve parity:

| mode | routed max/RMS | hidden max/RMS |
|---|---:|---:|
| current | 7.254304e-6 / 1.854037e-6 | 4.5422465e-5 / 1.1665878e-5 |
| serial | 7.254072e-6 / 1.854040e-6 | 4.5420602e-5 / 1.1665870e-5 |
| serial-nofma | 7.254304e-6 / 1.854037e-6 | 4.5422465e-5 / 1.1665878e-5 |
| f64-reference | 7.254537e-6 / 1.854085e-6 | 4.5422465e-5 / 1.1665858e-5 |

The focused no-snapshot layer-1 stop measured `0.04 s` for both current and
serial. The debug snapshot runs measured 2.56 s current, 2.68 s serial, 2.51
s serial-no-FMA, and 2.50 s F64; those numbers include diagnostics and are not
production latency measurements.

## Buffer and Synchronization Checks

The host accumulator is initialized exactly once per MoE call with exact zero
f32 values. CUDA expert output storage is allocated for exactly one
`moe_latent` vector and `moe_down` writes every output element on every launch.
The debug identical-vector buffer is exactly `16 * 3584 * sizeof(f32)` and
the weight buffer is exactly `16 * sizeof(f32)`; neither includes padding.

All 16 valid routed slots are visited once in rank order. The route weight is
applied once after down and is neither omitted nor duplicated. `ptr_buf.write`
and each expert-output `read_f32` are synchronous CUDA copies on the default
stream, so staging pointers and output buffers are not reused before the
snapshot. The serial diagnostic writes every output element from a fresh zero
local accumulator; no stale output or atomic update is involved.

## Decision

No production correction was made. The debug modes are opt-in through
`PULSAR_K3_ACCUM_MODE=current|serial|serial-nofma|f64-reference`; `current`
remains the default. CUDA remains non-default because the unresolved
CPU-Q8/CUDA expert-vector difference still produces the established end-to-end
mismatch. A full 93-layer run was not repeated because the focused parity did
not materially improve under any accumulation mode.

Proposed commit message: `debug: isolate K3 routed MoE accumulation`
