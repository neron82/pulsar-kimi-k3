# Kimi K3 CUDA Correctness Sprint

## Scope

This sprint compares the opt-in K3 CUDA routed-expert backend with the
CPU-Q8 reference on the Q2_K full model:

`/home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf`

The required deterministic invocation was:

```text
--tokens 1 -n 1 --ctx 2048 --temp 0 --top-p 1 --min-p 0 --seed 1
```

The RTX 3090 was selected by UUID with
`PULSAR_GPU=GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b`. Pulsar resolved that
request to physical GPU UUID `GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b`,
process-local CUDA index 0, an NVIDIA GeForce RTX 3090.

## Code Changes

- CUDA Q8_K packing now clamps to `[-127, 127]` and uses `roundf`.
- CPU Q8_K packing supports padded partial blocks.
- CUDA compilation no longer uses fast math; it uses non-contracted,
  precise division and square-root settings.
- K3 MLA preserves the learned 64-wide query/key tail when rotation is
  disabled.
- K3 KDA applies the reference sign when using folded `ssm_a` decay values.
- Expert capture filenames include layer/token coordinates.
- Comparison tooling supports backend and operation filters and compares
  packed artifacts through manifests.
- CPU-Q8 quantization tests cover extrema, zero blocks, tie rounding, and
  partial blocks.

## Focused Comparison

The focused comparison showed:

- Q2/Q3 expert weight packed bytes: exact.
- Initial expert input Q8_K packed bytes: exact.
- Routed-MoE output: max absolute difference `5.59e-9`, RMS
  `9.54e-10`.
- Layer hidden output: max absolute difference `1.49e-8`, RMS
  `1.81e-9`, cosine `1.0`.
- Expert f32 outputs differed only at small f32 arithmetic levels.
- Later input Q8_K packs differed in the scale fields as small upstream
  differences crossed quantization boundaries; this is expected to require
  end-to-end behavioural validation rather than byte equality.
- Layer 2 hidden output: max absolute difference `3.65e-5`, RMS `8.06e-6`,
  cosine `0.999999994`.
- Layer 3 hidden output: max absolute difference `1.59e-3`, RMS `3.42e-4`,
  cosine `0.999978047`; this is the first material Q8/CUDA divergence under
  the harness threshold.

## Full-Model Result

All four runs loaded all 24 shards and completed one token without NaN or
CUDA execution failure. Each backend was deterministic across its two runs,
but the CPU-Q8/CUDA token mismatch is stable.

| Backend | Run 1 load/prefill/total | Run 2 load/prefill/total | Output token |
|---|---:|---:|---:|
| CPU-Q8 | 47.9 / 16.69 / 17.47 s | 50.2 / 16.88 / 17.33 s | `51960`, both runs |
| CUDA | 50.4 / 4.11 / 4.07 s | 50.2 / 4.14 / 4.09 s | `3592`, both runs |

Commands:

```bash
PULSAR_GPU=GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b \
  PULSAR_K3_EXPERT_BACKEND=cpu-q8 \
  ./target/release/pulsar-cli -m MODEL.gguf \
  --tokens 1 -n 1 --ctx 2048 --temp 0 --top-p 1 --min-p 0 --seed 1

PULSAR_GPU=GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b \
  PULSAR_K3_EXPERT_BACKEND=cuda \
  ./target/release/pulsar-cli -m MODEL.gguf \
  --tokens 1 -n 1 --ctx 2048 --temp 0 --top-p 1 --min-p 0 --seed 1
```

## Decision

CUDA remains opt-in. The full-model token mismatch is a correctness failure,
despite the small focused layer-1 and layer-2 numerical differences. The
first material divergence currently appears at layer 3. The CPU reference
path remains the default until that later-layer divergence is explained and
fixed.

## Validation Notes

- `cargo test -p engine --release --quiet k3_`: passed, 29 tests.
- `cargo test -p quant --release --quiet`: passed, 19 tests.
- `cargo test -p kernels --release -- --test-threads=1`: passed, 17
  self-tests; 10 device tests remained ignored by the test harness.
- `cargo test --workspace --release --quiet`: passed.
- `cargo build --release -p engine` and `cargo build --release -p serve`:
  passed.
- `cargo fmt --all -- --check` and `git diff --check`: passed.
