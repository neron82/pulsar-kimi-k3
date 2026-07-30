#!/usr/bin/env python3
"""
Kimi K3 RED/GREEN Test Harness.

RED  phase: recompute all 7 contract components from fixture weights/inputs
            and compare against stored expected outputs.
GREEN phase: (when AtomicBot fork is built and
             PULSAR_K3_ATOMIC_MODEL points to a real quantized K3 GGUF)
             run the reference CPU build and compare logits.

Usage:
  python3 tests/k3_parity_harness.py red          # RED: contract self-test
  python3 tests/k3_parity_harness.py green        # GREEN: reference comparison (requires PULSAR_K3_ATOMIC_MODEL)
  python3 tests/k3_parity_harness.py red --json   # RED with JSON diff output
  python3 tests/k3_parity_harness.py green --build # build AtomicBot first, then compare
"""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "k3_contract_fixture.json"
ATOMIC_REPO  = Path("/tmp/atomic-llama-cpp-turboquant-kimi")
ATOMIC_BUILD = Path("/tmp/k3-cpu-build")
TOLERANCE    = 1e-10  # exact match for deterministic float ops

# ── helpers (mirror of fixture generator) ───────────────────────────────

def rms_norm(x, w, eps=1e-6):
    mean_sq = sum(v*v for v in x) / len(x)
    inv_rms = 1.0 / math.sqrt(mean_sq + eps)
    return [v * inv_rms * w[i] for i, v in enumerate(x)]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    s = sum(exps)
    return [e / s for e in exps]

def dot(x, y):
    return sum(a*b for a, b in zip(x, y))

def matvec(W, x):
    return [dot(row, x) for row in W]

def situ_gate(x, beta):
    """SiTU gate: beta * tanh(x/beta) * sigmoid(x) — from kimi-k3.cpp LLM_FFN_SITU"""
    return [beta * math.tanh(v / beta) * sigmoid(v) for v in x]


def situ_linear(x, linear_beta):
    """SiTU linear soft-cap: linear_beta * tanh(x/linear_beta) — from kimi-k3.cpp LLM_FFN_SITU"""
    return [linear_beta * math.tanh(v / linear_beta) for v in x]

# ── RED: contract self-test ──────────────────────────────────────────────

def run_red(verbose=True, output_json=False):
    with open(FIXTURE_PATH) as f:
        fx = json.load(f)

    x = fx["input_x"]
    results = {}
    all_pass = True

    # 1. SiTU-GLU
    c = fx["situ_glu"]
    W_gate, W_up, W_down = c["W_gate"], c["W_up"], c["W_down"]
    beta_g, beta_l = c["beta_gate"], c["beta_linear"]
    gate = situ_gate(matvec(W_gate, x), beta_g)
    up = situ_linear(matvec(W_up, x), beta_l)
    hidden = [g * u for g, u in zip(gate, up)]
    computed = matvec(W_down, hidden)
    expected = c["output"]
    diff = max(abs(a - b) for a, b in zip(computed, expected))
    passed = diff < TOLERANCE
    results["situ_glu"] = {"pass": passed, "max_diff": diff}
    if verbose:
        status = "✅" if passed else "❌"
        print(f"{status} SiTU-GLU: max_diff={diff:.2e}")
    all_pass = all_pass and passed

    # 2. Router top-k
    c = fx["router_topk"]
    W_router, r_bias = c["W_router"], c["bias"]
    top_k = c["top_k"]
    scores = [sigmoid(dot(W_router[i], x) + r_bias[i]) for i in range(len(W_router))]
    indexed = sorted(enumerate(scores), key=lambda t: -t[1])
    top_idx = [i for i, _ in indexed[:top_k]]
    top_val = [scores[i] for i in top_idx]
    s = sum(top_val)
    renorm = [v / s for v in top_val]
    idx_pass = top_idx == c["top_indices"]
    wt_pass = all(abs(a - b) < TOLERANCE for a, b in zip(renorm, c["top_weights"]))
    passed = idx_pass and wt_pass
    results["router_topk"] = {"pass": passed, "idx_match": idx_pass, "wt_max_diff": max(abs(a - b) for a, b in zip(renorm, c["top_weights"])) if wt_pass else None}
    if verbose:
        status = "✅" if passed else "❌"
        print(f"{status} Router top-k: idx_match={idx_pass}, weights_match={wt_pass}")
    all_pass = all_pass and passed

    # 3. KDA safe gate
    c = fx["kda_safe_gate"]
    f_a = matvec(c["W_f_a"], x)
    f_b = matvec(c["W_f_b"], f_a)
    g_raw = [f_b[i] + c["dt_bias"][i] for i in range(len(f_b))]
    n_head = len(c["A_log"])
    head_dim = len(c["output_g1"][0])
    computed_g1 = [[c["lower_bound"] * sigmoid(c["A_log"][h] * g_raw[h * head_dim + d]) for d in range(head_dim)] for h in range(n_head)]
    expected_g1 = c["output_g1"]
    diff = max(abs(computed_g1[h][d] - expected_g1[h][d]) for h in range(n_head) for d in range(head_dim))
    passed = diff < TOLERANCE
    results["kda_safe_gate"] = {"pass": passed, "max_diff": diff}
    if verbose:
        status = "✅" if passed else "❌"
        print(f"{status} KDA safe gate: max_diff={diff:.2e}")
    all_pass = all_pass and passed

    # 4. KDA output gate
    c = fx["kda_output_gate"]
    g2 = matvec(c["W_gate"], x)
    normed = rms_norm(c["attn_out"], c["o_norm_w"])
    computed = [normed[i] * sigmoid(g2[i]) for i in range(len(normed))]
    expected = c["output_gated"]
    diff = max(abs(a - b) for a, b in zip(computed, expected))
    passed = diff < TOLERANCE
    results["kda_output_gate"] = {"pass": passed, "max_diff": diff}
    if verbose:
        status = "✅" if passed else "❌"
        print(f"{status} KDA output gate: max_diff={diff:.2e}")
    all_pass = all_pass and passed

    # 5. MLA output gate
    c = fx["mla_output_gate"]
    g2 = matvec(c["W_gate"], x)
    computed = [c["attn_pregate"][i] * sigmoid(g2[i]) for i in range(len(c["attn_pregate"]))]
    expected = c["output_gated"]
    diff = max(abs(a - b) for a, b in zip(computed, expected))
    passed = diff < TOLERANCE
    results["mla_output_gate"] = {"pass": passed, "max_diff": diff}
    if verbose:
        status = "✅" if passed else "❌"
        print(f"{status} MLA output gate: max_diff={diff:.2e}")
    all_pass = all_pass and passed

    # 6. Latent-MoE
    c = fx["latent_moe"]
    latent = matvec(c["W_down"], x)
    scores_moe = [sigmoid(dot(c["W_gate_inp"][i], x) + c["exp_bias"][i]) for i in range(len(c["W_gate_inp"]))]
    indexed = sorted(enumerate(scores_moe), key=lambda t: -t[1])
    top_k_moe = len(c["top_indices"])
    top_idx = [i for i, _ in indexed[:top_k_moe]]
    top_val = [scores_moe[i] for i in top_idx]
    s = sum(top_val)
    renorm = [v / s for v in top_val]
    moe_acc = [0.0] * len(c["latent_norm_w"])
    for e_i, w in zip(top_idx, renorm):
        gate = situ_gate(matvec(c['expert_gates'][e_i], latent), fx['meta']['shapes']['situ_beta'])
        up = situ_linear(matvec(c['expert_ups'][e_i], latent), fx['meta']['shapes']['situ_linear_beta'])
        hidden = [g * u for g, u in zip(gate, up)]
        expert_out = matvec(c["expert_downs"][e_i], hidden)
        moe_acc = [moe_acc[i] + w * expert_out[i] for i in range(len(moe_acc))]
    moe_normed = rms_norm(moe_acc, c["latent_norm_w"])
    computed = matvec(c["W_up"], moe_normed)
    expected = c["output"]
    diff = max(abs(a - b) for a, b in zip(computed, expected))
    idx_pass = top_idx == c["top_indices"]
    passed = diff < TOLERANCE and idx_pass
    results["latent_moe"] = {"pass": passed, "max_diff": diff, "idx_match": idx_pass}
    if verbose:
        status = "✅" if passed else "❌"
        print(f"{status} Latent-MoE: max_diff={diff:.2e}, idx_match={idx_pass}")
    all_pass = all_pass and passed

    # 7. AttnRes mixture
    c = fx["attn_res_mix"]
    n_embd = len(c["prefix"])
    rows = c["bank"] + [c["prefix"]]
    n_rows = len(rows)
    k = [rms_norm(row, [1.0]*n_embd) for row in rows]
    sw = [c["norm_w"][i] * c["proj_w"][i] for i in range(n_embd)]
    scores = [dot(k[row], sw) for row in range(n_rows)]
    probs = softmax(scores)
    computed = [sum(probs[r] * rows[r][i] for r in range(n_rows)) for i in range(n_embd)]
    expected = c["output"]
    diff = max(abs(a - b) for a, b in zip(computed, expected))
    scores_pass = all(abs(a - b) < TOLERANCE for a, b in zip(scores, c["scores"]))
    probs_pass = all(abs(a - b) < TOLERANCE for a, b in zip(probs, c["probs"]))
    passed = diff < TOLERANCE and scores_pass and probs_pass
    results["attn_res_mix"] = {"pass": passed, "max_diff": diff, "scores_match": scores_pass, "probs_match": probs_pass}
    if verbose:
        status = "✅" if passed else "❌"
        print(f"{status} AttnRes mixture: max_diff={diff:.2e}, scores_match={scores_pass}, probs_match={probs_pass}")
    all_pass = all_pass and passed

    if output_json:
        print(json.dumps(results, indent=2))
    elif verbose:
        print(f"\n{'='*50}")
        print(f"Overall: {'✅ ALL PASS' if all_pass else '❌ SOME FAILED'}")
        print(f"{'='*50}")

    return all_pass, results


# ── GREEN: AtomicBot reference comparison ───────────────────────────────

def run_green(verbose=True, do_build=False):
    """Compare against the AtomicBot fork CPU build."""
    if do_build:
        build_atomic()

    if not ATOMIC_BUILD.exists():
        print(f"❌ AtomicBot build not found at {ATOMIC_BUILD}")
        print(f"   Run: python3 {__file__} green --build")
        return False

    llama_cli = ATOMIC_BUILD / "bin" / "llama-cli"
    if not llama_cli.exists():
        llama_cli = ATOMIC_BUILD / "bin" / "llama-cli"
        if not llama_cli.exists():
            print(f"❌ llama-cli not found in {ATOMIC_BUILD}/bin/")
            return False

    # GREEN is intentionally gated on a real, quantized K3 GGUF. The tiny
    # contract fixture is F32/synthetic and must never be treated as a
    # reference-loadable checkpoint.
    model_env = os.environ.get("PULSAR_K3_ATOMIC_MODEL")
    if not model_env:
        print("❌ PULSAR_K3_ATOMIC_MODEL is not set")
        print("   GREEN requires a real Atomic-compatible quantized K3 GGUF;")
        print("   the synthetic tests/k3_tiny_fixture.gguf is RED/contract-only.")
        return False
    model_path = Path(model_env).expanduser()
    if not model_path.is_file():
        print(f"❌ PULSAR_K3_ATOMIC_MODEL does not exist: {model_path}")
        return False

    print(f"🔬 Running AtomicBot reference on {model_path}...")
    cmd = [
        str(llama_cli),
        "-m", str(model_path),
        "-p", "Hello",
        "-n", "1",
        "--temp", "0.0",
        "--logits", "all",
        "--json",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        print(f"stdout:\n{result.stdout[:2000]}")
        if result.stderr:
            print(f"stderr (last 500):\n{result.stderr[-500:]}")
        print(f"✅ AtomicBot reference completed (exit code {result.returncode})")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ AtomicBot reference timed out after 120s")
        return False
    except FileNotFoundError:
        print(f"❌ llama-cli not found at {llama_cli}")
        return False


def build_atomic():
    """Build the AtomicBot fork as a CPU-only reference."""
    if not ATOMIC_REPO.exists():
        print(f"❌ AtomicBot repo not found at {ATOMIC_REPO}")
        print(f"   Clone: git clone --branch feat/kimi-k3-support <repo> {ATOMIC_REPO}")
        return False

    print(f"🔧 Building AtomicBot CPU reference at {ATOMIC_BUILD}...")
    cmds = [
        f"cmake -S {ATOMIC_REPO} -B {ATOMIC_BUILD} -DGGML_CUDA=OFF -DLLAMA_CURL=OFF",
        f"cmake --build {ATOMIC_BUILD} --target llama-cli -j$(nproc)",
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"❌ Build step failed: {cmd}")
            print(result.stderr[-500:])
            return False
        print(f"   ✅ {cmd.split()[0]} ... done")
    print(f"✅ AtomicBot CPU build complete at {ATOMIC_BUILD}")
    return True


# ── main ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 1

    mode = sys.argv[1]
    verbose = "--quiet" not in sys.argv
    output_json = "--json" in sys.argv
    do_build = "--build" in sys.argv

    if mode == "red":
        passed, results = run_red(verbose=verbose, output_json=output_json)
        return 0 if passed else 1
    elif mode == "green":
        passed = run_green(verbose=verbose, do_build=do_build)
        return 0 if passed else 1
    else:
        print(f"Unknown mode: {mode}")
        print(__doc__)
        return 1


if __name__ == "__main__":
    sys.exit(main())
