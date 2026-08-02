# Kimi K3 Behavioural Validation

## Configuration

- Date: 2026-08-02
- Model: `/home/neron/models/kimi-k3/Q2_K/Kimi-K3-Q2_K-00001-of-00024.gguf`
- GPU: RTX 3090,
  `GPU-7ac9a486-bc2d-f429-bcab-2e6fd1aee04b`
- Backend: `PULSAR_K3_EXPERT_BACKEND=cuda`
- Expert cache: `PULSAR_DEV_CACHE_GB=0`; K3 uses its synchronous per-layer
  CUDA staging slot.
- Context: 4096
- Sampling: temperature 0, seed 42, at most 8 output tokens
- Prompt protocol: K3 non-thinking XTML through the OpenAI-compatible server
- State policy: each request starts at position zero and resets recurrent,
  cache, and AttnRes state.

The server used the normal mostly resident placement after increasing the K3
model-load VRAM reserve to 2048 MiB. Startup reported a 191,299,840-byte CUDA
expert staging buffer and then accepted requests normally.

## Smoke Results

| Prompt | Exact response | Prompt tokens | Completion tokens |
|---|---|---:|---:|
| `What is the capital of France? Reply with one word.` | `Paris` | 33 | 1 |
| `What is 2+2? Reply with one word.` | `4` | 33 | 1 |
| `Name a primary color. Reply with one word.` | `red` | 31 | 1 |
| `Opposite of hot? Reply with one word.` | `Cold` | 30 | 1 |
| `What gas do humans breathe to survive? Reply with one word.` | `Oxygen` | 34 | 2 |
| `Who wrote Hamlet? Reply with two words.` | `William Shakespeare` | 31 | 2 |
| `What planet do we live on? Reply with one word.` | `Earth` | 33 | 1 |

All seven responses are coherent, satisfy the requested answer length, and
stop before XTML structural closing tokens are exposed to the client.

## CPU-Q8 Evidence

The final reset-per-request server was also run with
`PULSAR_K3_EXPERT_BACKEND=cpu-q8` and context 256. For
`What is the capital of France? Reply with one word.`, it returned exactly
`Paris` using 33 prompt tokens and one completion token. No XTML structural
tokens were exposed. This establishes that the shared semantic fixes are not
CUDA-only.

Before the XTML stop correction, the same corrected CPU-Q8 graph had generated
`Paris` followed by structural close tokens. Stopping on the first K3
`<|close|>` is what makes the final client response exactly `Paris`.

The raw-completion diagnostic `The capital of France is` also selected
` Paris` with a top-1 margin of `3.0556` and no NaN/Inf values. These raw
completion diagnostics are graph evidence only; instruction use must use
XTML.

## Verification

The following gates passed after the behavioral run:

```text
cargo test --workspace --release --quiet
cargo fmt --all -- --check
cargo check -p tokenizer
git diff --check
```

Reported release test groups included:

- Engine: 47 passed.
- CUDA kernel self-tests: 17 passed.
- Quant: 19 passed.
- Tokenizer: 10 passed, plus 2 integration tests.

## Interpretation

This evidence supersedes the earlier repetitive-output baseline and establishes
usable K3 text behavior for this split-weight Q2_K checkpoint on both the
shared CPU-Q8 graph and the CUDA expert path. It is not a claim of bitwise
backend parity, fused-MLA support, optimized prefill, or support for raw chat
prompts without XTML.
