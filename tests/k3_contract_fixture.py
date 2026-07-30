#!/usr/bin/env python3
"""
Kimi K3 Deterministic Contract Fixture — host-side reference vectors.

Covers all 6 K3-specific components:
  1. SiTU-GLU activation
  2. Router top-k / sigmoid + renormalize
  3. KDA safe gate formula (g1 = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias)))
  4. KDA full-rank output gate (g2 = g_proj(x); out = RMSNorm(attn) * sigmoid(g2))
  5. NoPE gated MLA output gate (attn = attn * sigmoid(g_proj(x)))
  6. Latent-MoE dimension flow (down → experts → RMSNorm → up)
  7. AttnRes mixture (softmax over bank + prefix)

Output: JSON fixture at tests/k3_contract_fixture.json
         consumed by both Python and Rust test harnesses.

Dependency-free: uses only math, json, struct (no numpy, no torch).
"""

import json
import math
import struct
from pathlib import Path

# ── tiny K3 shape constants ──────────────────────────────────────────────
# These are scaled-down versions of the real K3 contract for fast CPU tests.
N_EMBD      = 64     # hidden size (real: 7168)
N_HEAD      = 4      # heads (real: 96)
HEAD_DIM    = 8      # KDA head size (real: 128)
D_INNER     = N_HEAD * HEAD_DIM  # 32
N_EXPERT    = 8      # routed experts (real: 896)
TOP_K       = 2      # active experts (real: 16)
N_EXP_SHARED = 1     # shared experts (real: 2)
MOE_LATENT  = 16     # latent expert size (real: 3584)
N_FF_EXP    = 8      # expert FFN size (real: 3072)
N_FF_DENSE  = 32     # dense FFN size (scaled)
RES_BLOCK   = 2      # attn_res_block_size (real: 6)
N_LAYER     = 4      # total layers (real: 93)
N_LAYER_DENSE_LEAD = 1  # one dense FFN, then MoE
GATE_LOWER_BOUND = 0.5  # kda_gate_lower_bound
SITU_BETA   = 0.25   # activation_situ_beta
SITU_LINEAR_BETA = 0.5  # activation_situ_linear_beta
RMS_EPS     = 1e-6

# ── helpers ──────────────────────────────────────────────────────────────

def rms_norm(x, w, eps=RMS_EPS):
    """RMSNorm: out = x / sqrt(mean(x^2) + eps) * w"""
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
    """W is list of rows (each row is list of floats). Returns W @ x."""
    return [dot(row, x) for row in W]

def matmul(A, B):
    """A: [M,K], B: [K,N] → [M,N]"""
    M = len(A)
    K = len(A[0])
    N = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(K)) for j in range(N)] for i in range(M)]

def transpose(M):
    return [[M[i][j] for i in range(len(M))] for j in range(len(M[0]))]

def make_weight(rows, cols, seed, scale=0.1):
    """Deterministic weight matrix from a seed."""
    rng = seed
    def rand():
        nonlocal rng
        rng = (rng * 1103515245 + 12345) & 0x7fffffff
        return (rng & 0xffff) / 65536.0 - 0.5
    return [[rand() * scale for _ in range(cols)] for _ in range(rows)]

def make_bias(n, seed, scale=0.01):
    rng = seed
    def rand():
        nonlocal rng
        rng = (rng * 1103515245 + 12345) & 0x7fffffff
        return (rng & 0xffff) / 65536.0 - 0.5
    return [rand() * scale for _ in range(n)]

# ── 1. SiTU-GLU ─────────────────────────────────────────────────────────
def situ_gate(x, beta):
    """SiTU gate activation: beta * tanh(x/beta) * sigmoid(x)
       From kimi-k3.cpp LLM_FFN_SITU (llama-graph.cpp:1679-1685)."""
    return [beta * math.tanh(v / beta) * sigmoid(v) for v in x]


def situ_linear(x, linear_beta):
    """SiTU linear (up) soft-cap: linear_beta * tanh(x/linear_beta)
       From kimi-k3.cpp LLM_FFN_SITU (llama-graph.cpp:1689)."""
    return [linear_beta * math.tanh(v / linear_beta) for v in x]


def situ_glu(x, W_gate, W_up, W_down, beta_gate, beta_linear):
    """SiTU-GLU FFN: [beta*tanh(gate/beta)*sigmoid(gate)] * [linear_beta*tanh(up/linear_beta)] @ W_down
       From kimi-k3.cpp LLM_FFN_SITU (llama-graph.cpp:1677-1696).
       NOTE: beta_gate = situ_beta, beta_linear = situ_linear_beta (test-only scale)."""
    gate = situ_gate(matvec(W_gate, x), beta_gate)
    up = situ_linear(matvec(W_up, x), beta_linear)
    # elementwise multiply
    hidden = [g * u for g, u in zip(gate, up)]
    return matvec(W_down, hidden)

# ── 2. Router top-k / sigmoid + renormalize ─────────────────────────────

def router_forward(x, W_router, bias, top_k=TOP_K):
    """Sigmoid scores → top-k → renormalize."""
    scores = [sigmoid(dot(W_router[i], x) + bias[i]) for i in range(len(W_router))]
    # top-k indices
    indexed = sorted(enumerate(scores), key=lambda t: -t[1])
    top_idx = [i for i, _ in indexed[:top_k]]
    top_val = [scores[i] for i in top_idx]
    # renormalize
    s = sum(top_val)
    renorm = [v / s for v in top_val]
    return top_idx, renorm, scores

# ── 3. KDA safe gate ────────────────────────────────────────────────────

def kda_safe_gate(x, W_f_a, W_f_b, dt_bias, A_log, lower_bound=GATE_LOWER_BOUND):
    """
    g1 = lower_bound * sigmoid(exp(A_log) * (f_b(f_a(x)) + dt_bias))
    A_log is per-head, dt_bias is per-d_inner.
    Returns g1 shaped [head_dim, n_head] (or flattened).
    """
    f_a = matvec(W_f_a, x)          # [n_embd_head_kda]
    f_b = matvec(W_f_b, f_a)        # [d_inner]
    g_raw = [f_b[i] + dt_bias[i] for i in range(len(f_b))]
    # reshape to [head_dim, n_head]
    g_2d = [[g_raw[h * HEAD_DIM + d] for d in range(HEAD_DIM)] for h in range(N_HEAD)]
    g1 = [[lower_bound * sigmoid(A_log[h] * g_2d[h][d]) for d in range(HEAD_DIM)] for h in range(N_HEAD)]
    return g1  # [n_head, head_dim]

# ── 4. KDA full-rank output gate ────────────────────────────────────────

def kda_output_gate(attn_out, x, W_gate, o_norm_w):
    """
    g2 = g_proj(x)  [d_inner]
    out = RMSNorm(attn_out, o_norm_w) * sigmoid(g2)
    Then projected by wo.
    """
    g2 = matvec(W_gate, x)  # [d_inner]
    normed = rms_norm(attn_out, o_norm_w)
    gated = [normed[i] * sigmoid(g2[i]) for i in range(len(normed))]
    return gated, g2

# ── 5. NoPE gated MLA output gate ───────────────────────────────────────

def mla_output_gate(attn_pregate, x, W_gate):
    """
    attn = attn_pregate * sigmoid(g_proj(x))
    Returns gated attention output (before o_proj).
    """
    g2 = matvec(W_gate, x)
    return [attn_pregate[i] * sigmoid(g2[i]) for i in range(len(attn_pregate))], g2

# ── 6. Latent-MoE dimension flow ───────────────────────────────────────

def latent_moe(x, W_down, W_gate_inp, exp_bias, expert_gates, expert_downs, expert_ups,
               latent_norm_w, W_up, n_expert=N_EXPERT, top_k=TOP_K):
    """
    latent = x @ W_down
    router on FULL hidden state (x), not latent
    moe_out = sum_e w_e * SiTU(latent @ W_gate_e) * (latent @ W_up_e) @ W_down_e
    moe_out = RMSNorm(moe_out) @ W_up
    """
    latent = matvec(W_down, x)  # [moe_latent]

    # router on x (full hidden state)
    idx, weights, _ = router_forward(x, W_gate_inp, exp_bias, top_k)

    # expert computation in latent space
    moe_acc = [0.0] * MOE_LATENT
    for e_i, w in zip(idx, weights):
        gate = situ_gate(matvec(expert_gates[e_i], latent), SITU_BETA)
        up = situ_linear(matvec(expert_ups[e_i], latent), SITU_LINEAR_BETA)
        hidden = [g * u for g, u in zip(gate, up)]
        expert_out = matvec(expert_downs[e_i], hidden)
        moe_acc = [moe_acc[i] + w * expert_out[i] for i in range(MOE_LATENT)]

    # latent norm
    moe_normed = rms_norm(moe_acc, latent_norm_w)
    # project back up
    out = matvec(W_up, moe_normed)
    return out, idx, weights

# ── 7. AttnRes mixture ──────────────────────────────────────────────────

def attn_res_mix(prefix, bank, norm_w, proj_w):
    """
    v = [bank_rows..., prefix]  stacked along dim=1
    k = RMSNorm(v)
    scores = sum(k * (norm_w * proj_w), dim=embd)
    probs = softmax(scores over rows)
    out = sum_r probs_r * v_r
    """
    n_embd = len(prefix)
    n_toks = 1  # single token for deterministic test
    rows = bank + [prefix]
    n_rows = len(rows)

    # v: [n_rows, n_embd]
    v = rows

    # k = RMSNorm(v) per row
    k = [rms_norm(row, [1.0]*n_embd) for row in v]  # unit norm weight

    # score weight = norm_w * proj_w (elementwise)
    sw = [norm_w[i] * proj_w[i] for i in range(n_embd)]

    # scores: dot(k[row], sw) for each row
    scores = [dot(k[row], sw) for row in range(n_rows)]
    probs = softmax(scores)

    # weighted sum
    out = [sum(probs[r] * v[r][i] for r in range(n_rows)) for i in range(n_embd)]
    return out, scores, probs


# ══════════════════════════════════════════════════════════════════════════
#  Generate deterministic fixture
# ══════════════════════════════════════════════════════════════════════════

def generate_fixture():
    fixture = {
        "meta": {
            "description": "Kimi K3 deterministic contract fixture — host-side reference vectors",
            "shapes": {
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "head_dim": HEAD_DIM,
                "d_inner": D_INNER,
                "n_expert": N_EXPERT,
                "top_k": TOP_K,
                "n_expert_shared": N_EXP_SHARED,
                "moe_latent": MOE_LATENT,
                "n_ff_exp": N_FF_EXP,
                "n_ff_dense": N_FF_DENSE,
                "res_block": RES_BLOCK,
                "n_layer": N_LAYER,
                "n_layer_dense_lead": N_LAYER_DENSE_LEAD,
                "gate_lower_bound": GATE_LOWER_BOUND,
                "situ_beta": SITU_BETA,
                "situ_linear_beta": SITU_LINEAR_BETA,
                "rms_eps": RMS_EPS,
            },
            "components": [
                "situ_glu",
                "router_topk",
                "kda_safe_gate",
                "kda_output_gate",
                "mla_output_gate",
                "latent_moe",
                "attn_res_mix",
            ]
        }
    }

    # Deterministic input vector
    x = [math.sin(i * 0.7) * 0.5 for i in range(N_EMBD)]
    fixture["input_x"] = x

    # ── 1. SiTU-GLU ──
    W_gate = make_weight(N_FF_DENSE, N_EMBD, seed=1001)
    W_up   = make_weight(N_FF_DENSE, N_EMBD, seed=1002)
    W_down = make_weight(N_EMBD, N_FF_DENSE, seed=1003)
    situ_out = situ_glu(x, W_gate, W_up, W_down, SITU_BETA, SITU_LINEAR_BETA)
    fixture["situ_glu"] = {
        "W_gate": W_gate,
        "W_up": W_up,
        "W_down": W_down,
        "beta_gate": SITU_BETA,
        "beta_linear": SITU_LINEAR_BETA,
        "output": situ_out,
    }

    # ── 2. Router top-k ──
    W_router = make_weight(N_EXPERT, N_EMBD, seed=2001)
    r_bias = make_bias(N_EXPERT, seed=2002)
    top_idx, top_weights, raw_scores = router_forward(x, W_router, r_bias, TOP_K)
    fixture["router_topk"] = {
        "W_router": W_router,
        "bias": r_bias,
        "top_k": TOP_K,
        "raw_scores": raw_scores,
        "top_indices": top_idx,
        "top_weights": top_weights,
    }

    # ── 3. KDA safe gate ──
    W_f_a = make_weight(HEAD_DIM, N_EMBD, seed=3001)
    W_f_b = make_weight(D_INNER, HEAD_DIM, seed=3002)
    dt_bias = make_bias(D_INNER, seed=3003, scale=0.05)
    A_log = [0.5 + i * 0.1 for i in range(N_HEAD)]  # exp(A_log) > 0
    g1 = kda_safe_gate(x, W_f_a, W_f_b, dt_bias, A_log, GATE_LOWER_BOUND)
    fixture["kda_safe_gate"] = {
        "W_f_a": W_f_a,
        "W_f_b": W_f_b,
        "dt_bias": dt_bias,
        "A_log": A_log,
        "lower_bound": GATE_LOWER_BOUND,
        "output_g1": g1,  # [n_head, head_dim]
    }

    # ── 4. KDA output gate ──
    W_gate_kda = make_weight(D_INNER, N_EMBD, seed=4001)
    o_norm_w = [1.0 + i * 0.05 for i in range(D_INNER)]
    # synthetic attn_out
    attn_out = [math.cos(i * 0.3) * 0.5 for i in range(D_INNER)]
    gated_out, g2_raw = kda_output_gate(attn_out, x, W_gate_kda, o_norm_w)
    fixture["kda_output_gate"] = {
        "W_gate": W_gate_kda,
        "o_norm_w": o_norm_w,
        "attn_out": attn_out,
        "g2_raw": g2_raw,
        "output_gated": gated_out,
    }

    # ── 5. MLA output gate ──
    W_gate_mla = make_weight(D_INNER, N_EMBD, seed=5001)
    attn_pregate = [math.sin(i * 0.5) * 0.5 for i in range(D_INNER)]
    mla_gated, mla_g2 = mla_output_gate(attn_pregate, x, W_gate_mla)
    fixture["mla_output_gate"] = {
        "W_gate": W_gate_mla,
        "attn_pregate": attn_pregate,
        "g2_raw": mla_g2,
        "output_gated": mla_gated,
    }

    # ── 6. Latent-MoE ──
    W_down_moe = make_weight(MOE_LATENT, N_EMBD, seed=6001)
    W_gate_inp = make_weight(N_EXPERT, N_EMBD, seed=6002)
    exp_bias = make_bias(N_EXPERT, seed=6003)
    expert_gates = [make_weight(N_FF_EXP, MOE_LATENT, seed=6100 + i) for i in range(N_EXPERT)]
    expert_ups   = [make_weight(N_FF_EXP, MOE_LATENT, seed=6200 + i) for i in range(N_EXPERT)]
    expert_downs = [make_weight(MOE_LATENT, N_FF_EXP, seed=6300 + i) for i in range(N_EXPERT)]
    latent_norm_w = [1.0 + i * 0.1 for i in range(MOE_LATENT)]
    W_up_moe = make_weight(N_EMBD, MOE_LATENT, seed=6004)
    moe_out, moe_idx, moe_weights = latent_moe(
        x, W_down_moe, W_gate_inp, exp_bias,
        expert_gates, expert_downs, expert_ups,
        latent_norm_w, W_up_moe,
        N_EXPERT, TOP_K
    )
    fixture["latent_moe"] = {
        "W_down": W_down_moe,
        "W_gate_inp": W_gate_inp,
        "exp_bias": exp_bias,
        "expert_gates": expert_gates,
        "expert_ups": expert_ups,
        "expert_downs": expert_downs,
        "latent_norm_w": latent_norm_w,
        "W_up": W_up_moe,
        "top_indices": moe_idx,
        "top_weights": moe_weights,
        "output": moe_out,
    }

    # ── 7. AttnRes mixture ──
    norm_w = [1.0 + i * 0.02 for i in range(N_EMBD)]
    proj_w = [0.5 + i * 0.01 for i in range(N_EMBD)]
    # bank: 2 snapshot rows
    bank = [
        [math.sin(i * 0.1 + j) * 0.3 for i in range(N_EMBD)]
        for j in range(2)
    ]
    prefix = [math.cos(i * 0.2) * 0.4 for i in range(N_EMBD)]
    mix_out, mix_scores, mix_probs = attn_res_mix(prefix, bank, norm_w, proj_w)
    fixture["attn_res_mix"] = {
        "norm_w": norm_w,
        "proj_w": proj_w,
        "bank": bank,
        "prefix": prefix,
        "scores": mix_scores,
        "probs": mix_probs,
        "output": mix_out,
    }

    return fixture


def main():
    fixture = generate_fixture()

    out_path = Path(__file__).parent / "k3_contract_fixture.json"
    with open(out_path, "w") as f:
        json.dump(fixture, f, indent=2, cls=_Encoder)
    print(f"✅ Fixture written: {out_path}")
    print(f"   Components: {len(fixture['meta']['components'])}")
    for c in fixture['meta']['components']:
        print(f"   - {c}")

    # Quick self-test: verify determinism
    fixture2 = generate_fixture()
    assert json.dumps(fixture, cls=_Encoder) == json.dumps(fixture2, cls=_Encoder), \
        "NON-DETERMINISTIC: two runs produced different output"
    print("✅ Determinism verified: re-run produces identical fixture")


class _Encoder(json.JSONEncoder):
    """Custom encoder to handle nested lists of floats compactly."""
    def default(self, obj):
        return super().default(obj)


if __name__ == "__main__":
    main()
