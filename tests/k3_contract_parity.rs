//! Kimi K3 Contract Fixture Parity Test (Rust side)
//!
//! Loads the deterministic JSON fixture and recomputes all 7 contract components
//! in pure Rust, comparing against the Python-generated expected outputs.
//! This is the cross-language RED phase: Rust vs Python reference.
//!
//! Run: cargo test --test k3_contract_parity -- --nocapture

use std::path::Path;

const FIXTURE_PATH: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/k3_contract_fixture.json");
const TOLERANCE: f64 = 1e-10;

// ── helpers ──────────────────────────────────────────────────────────────

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
    let s: f64 = exps.iter().sum();
    exps.iter().map(|e| e / s).collect()
}

fn dot(x: &[f64], y: &[f64]) -> f64 {
    x.iter().zip(y.iter()).map(|(a, b)| a * b).sum()
}

fn matvec(w: &[Vec<f64>], x: &[f64]) -> Vec<f64> {
    w.iter().map(|row| dot(row, x)).collect()
}

fn rms_norm(x: &[f64], w: &[f64], eps: f64) -> Vec<f64> {
    let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
    let inv_rms = 1.0 / (mean_sq + eps).sqrt();
    x.iter().zip(w.iter()).map(|(v, wi)| v * inv_rms * wi).collect()
}

fn situ(x: &[f64], beta: f64) -> Vec<f64> {
    x.iter().map(|v| v * sigmoid(beta * *v)).collect()
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

// ── JSON parsing helpers ─────────────────────────────────────────────────

fn parse_f64_matrix(val: &serde_json::Value) -> Vec<Vec<f64>> {
    val.as_array().unwrap().iter().map(|row| {
        row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
    }).collect()
}

fn parse_f64_vec(val: &serde_json::Value) -> Vec<f64> {
    val.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn parse_i64_vec(val: &serde_json::Value) -> Vec<i64> {
    val.as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect()
}

// ── tests ─────────────────────────────────────────────────────────────────

#[test]
fn test_situ_glu() {
    let fx = load_fixture();
    let c = &fx["situ_glu"];
    let x = parse_f64_vec(&fx["input_x"]);
    let w_gate = parse_f64_matrix(&c["W_gate"]);
    let w_up = parse_f64_matrix(&c["W_up"]);
    let w_down = parse_f64_matrix(&c["W_down"]);
    let beta_gate = c["beta_gate"].as_f64().unwrap();

    let gate = situ(&matvec(&w_gate, &x), beta_gate);
    let up = matvec(&w_up, &x);
    let hidden: Vec<f64> = gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();
    let computed = matvec(&w_down, &hidden);
    let expected = parse_f64_vec(&c["output"]);

    let diff = max_abs_diff(&computed, &expected);
    assert!(diff < TOLERANCE, "SiTU-GLU: max_diff={:.2e}", diff);
    println!("✅ Rust SiTU-GLU: max_diff={:.2e}", diff);
}

#[test]
fn test_router_topk() {
    let fx = load_fixture();
    let c = &fx["router_topk"];
    let x = parse_f64_vec(&fx["input_x"]);
    let w_router = parse_f64_matrix(&c["W_router"]);
    let bias = parse_f64_vec(&c["bias"]);
    let top_k = c["top_k"].as_i64().unwrap() as usize;

    let scores: Vec<f64> = w_router.iter().zip(bias.iter())
        .map(|(row, b)| sigmoid(dot(row, &x) + b)).collect();

    let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, s)| (i, *s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_idx: Vec<i64> = indexed.iter().take(top_k).map(|(i, _)| *i as i64).collect();
    let top_val: Vec<f64> = indexed.iter().take(top_k).map(|(_, s)| *s).collect();
    let s: f64 = top_val.iter().sum();
    let renorm: Vec<f64> = top_val.iter().map(|v| v / s).collect();

    let expected_idx = parse_i64_vec(&c["top_indices"]);
    let expected_wt = parse_f64_vec(&c["top_weights"]);

    assert_eq!(top_idx, expected_idx, "Router top-k: index mismatch");
    let wt_diff = max_abs_diff(&renorm, &expected_wt);
    assert!(wt_diff < TOLERANCE, "Router top-k: weight max_diff={:.2e}", wt_diff);
    println!("✅ Rust Router top-k: idx_match=true, wt_max_diff={:.2e}", wt_diff);
}

#[test]
fn test_kda_safe_gate() {
    let fx = load_fixture();
    let c = &fx["kda_safe_gate"];
    let x = parse_f64_vec(&fx["input_x"]);
    let w_f_a = parse_f64_matrix(&c["W_f_a"]);
    let w_f_b = parse_f64_matrix(&c["W_f_b"]);
    let dt_bias = parse_f64_vec(&c["dt_bias"]);
    let a_log = parse_f64_vec(&c["A_log"]);
    let lower_bound = c["lower_bound"].as_f64().unwrap();
    let expected_g1: Vec<Vec<f64>> = parse_f64_matrix(&c["output_g1"]);

    let f_a = matvec(&w_f_a, &x);
    let f_b = matvec(&w_f_b, &f_a);
    let g_raw: Vec<f64> = f_b.iter().zip(dt_bias.iter()).map(|(fb, dt)| fb + dt).collect();

    let n_head = a_log.len();
    let head_dim = expected_g1[0].len();
    let mut computed_g1 = vec![vec![0.0; head_dim]; n_head];
    for h in 0..n_head {
        for d in 0..head_dim {
            computed_g1[h][d] = lower_bound * sigmoid(a_log[h] * g_raw[h * head_dim + d]);
        }
    }

    let mut max_diff = 0.0;
    for h in 0..n_head {
        for d in 0..head_dim {
            let diff = (computed_g1[h][d] - expected_g1[h][d]).abs();
            if diff > max_diff { max_diff = diff; }
        }
    }
    assert!(max_diff < TOLERANCE, "KDA safe gate: max_diff={:.2e}", max_diff);
    println!("✅ Rust KDA safe gate: max_diff={:.2e}", max_diff);
}

#[test]
fn test_kda_output_gate() {
    let fx = load_fixture();
    let c = &fx["kda_output_gate"];
    let x = parse_f64_vec(&fx["input_x"]);
    let w_gate = parse_f64_matrix(&c["W_gate"]);
    let o_norm_w = parse_f64_vec(&c["o_norm_w"]);
    let attn_out = parse_f64_vec(&c["attn_out"]);

    let g2 = matvec(&w_gate, &x);
    let normed = rms_norm(&attn_out, &o_norm_w, 1e-6);
    let computed: Vec<f64> = normed.iter().zip(g2.iter()).map(|(n, g)| n * sigmoid(*g)).collect();
    let expected = parse_f64_vec(&c["output_gated"]);

    let diff = max_abs_diff(&computed, &expected);
    assert!(diff < TOLERANCE, "KDA output gate: max_diff={:.2e}", diff);
    println!("✅ Rust KDA output gate: max_diff={:.2e}", diff);
}

#[test]
fn test_mla_output_gate() {
    let fx = load_fixture();
    let c = &fx["mla_output_gate"];
    let x = parse_f64_vec(&fx["input_x"]);
    let w_gate = parse_f64_matrix(&c["W_gate"]);
    let attn_pregate = parse_f64_vec(&c["attn_pregate"]);

    let g2 = matvec(&w_gate, &x);
    let computed: Vec<f64> = attn_pregate.iter().zip(g2.iter()).map(|(a, g)| a * sigmoid(*g)).collect();
    let expected = parse_f64_vec(&c["output_gated"]);

    let diff = max_abs_diff(&computed, &expected);
    assert!(diff < TOLERANCE, "MLA output gate: max_diff={:.2e}", diff);
    println!("✅ Rust MLA output gate: max_diff={:.2e}", diff);
}

#[test]
fn test_latent_moe() {
    let fx = load_fixture();
    let c = &fx["latent_moe"];
    let x = parse_f64_vec(&fx["input_x"]);
    let w_down = parse_f64_matrix(&c["W_down"]);
    let w_gate_inp = parse_f64_matrix(&c["W_gate_inp"]);
    let exp_bias = parse_f64_vec(&c["exp_bias"]);
    let expert_gates: Vec<Vec<Vec<f64>>> = c["expert_gates"].as_array().unwrap().iter()
        .map(|e| parse_f64_matrix(e)).collect();
    let expert_ups: Vec<Vec<Vec<f64>>> = c["expert_ups"].as_array().unwrap().iter()
        .map(|e| parse_f64_matrix(e)).collect();
    let expert_downs: Vec<Vec<Vec<f64>>> = c["expert_downs"].as_array().unwrap().iter()
        .map(|e| parse_f64_matrix(e)).collect();
    let latent_norm_w = parse_f64_vec(&c["latent_norm_w"]);
    let w_up = parse_f64_matrix(&c["W_up"]);
    let top_k = fx["meta"]["shapes"]["top_k"].as_i64().unwrap() as usize;
    let situ_beta = fx["meta"]["shapes"]["situ_beta"].as_f64().unwrap();

    let latent = matvec(&w_down, &x);
    let scores: Vec<f64> = w_gate_inp.iter().zip(exp_bias.iter())
        .map(|(row, b)| sigmoid(dot(row, &x) + b)).collect();
    let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, s)| (i, *s)).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    let top_idx: Vec<i64> = indexed.iter().take(top_k).map(|(i, _)| *i as i64).collect();
    let top_val: Vec<f64> = indexed.iter().take(top_k).map(|(_, s)| *s).collect();
    let s: f64 = top_val.iter().sum();
    let renorm: Vec<f64> = top_val.iter().map(|v| v / s).collect();

    let mut moe_acc = vec![0.0; latent_norm_w.len()];
    for (&e_i, &w) in top_idx.iter().zip(renorm.iter()) {
        let ei = e_i as usize;
        let gate = situ(&matvec(&expert_gates[ei], &latent), situ_beta);
        let up = matvec(&expert_ups[ei], &latent);
        let hidden: Vec<f64> = gate.iter().zip(up.iter()).map(|(g, u)| g * u).collect();
        let expert_out = matvec(&expert_downs[ei], &hidden);
        for i in 0..moe_acc.len() {
            moe_acc[i] += w * expert_out[i];
        }
    }
    let moe_normed = rms_norm(&moe_acc, &latent_norm_w, 1e-6);
    let computed = matvec(&w_up, &moe_normed);
    let expected = parse_f64_vec(&c["output"]);

    let expected_idx = parse_i64_vec(&c["top_indices"]);
    assert_eq!(top_idx, expected_idx, "Latent-MoE: index mismatch");
    let diff = max_abs_diff(&computed, &expected);
    assert!(diff < TOLERANCE, "Latent-MoE: max_diff={:.2e}", diff);
    println!("✅ Rust Latent-MoE: max_diff={:.2e}, idx_match=true", diff);
}

#[test]
fn test_attn_res_mix() {
    let fx = load_fixture();
    let c = &fx["attn_res_mix"];
    let n_embd = c["prefix"].as_array().unwrap().len();
    let norm_w = parse_f64_vec(&c["norm_w"]);
    let proj_w = parse_f64_vec(&c["proj_w"]);
    let bank: Vec<Vec<f64>> = c["bank"].as_array().unwrap().iter()
        .map(|row| parse_f64_vec(row)).collect();
    let prefix = parse_f64_vec(&c["prefix"]);

    let mut rows = bank.clone();
    rows.push(prefix.clone());
    let n_rows = rows.len();

    let k: Vec<Vec<f64>> = rows.iter().map(|row| rms_norm(row, &vec![1.0; n_embd], 1e-6)).collect();
    let sw: Vec<f64> = norm_w.iter().zip(proj_w.iter()).map(|(n, p)| n * p).collect();
    let scores: Vec<f64> = k.iter().map(|k_row| dot(k_row, &sw)).collect();
    let probs = softmax(&scores);

    let mut computed = vec![0.0; n_embd];
    for r in 0..n_rows {
        for i in 0..n_embd {
            computed[i] += probs[r] * rows[r][i];
        }
    }

    let expected = parse_f64_vec(&c["output"]);
    let expected_scores = parse_f64_vec(&c["scores"]);
    let expected_probs = parse_f64_vec(&c["probs"]);

    let diff = max_abs_diff(&computed, &expected);
    let scores_diff = max_abs_diff(&scores, &expected_scores);
    let probs_diff = max_abs_diff(&probs, &expected_probs);

    assert!(diff < TOLERANCE, "AttnRes mixture output: max_diff={:.2e}", diff);
    assert!(scores_diff < TOLERANCE, "AttnRes mixture scores: max_diff={:.2e}", scores_diff);
    assert!(probs_diff < TOLERANCE, "AttnRes mixture probs: max_diff={:.2e}", probs_diff);
    println!("✅ Rust AttnRes mixture: max_diff={:.2e}, scores_match=true, probs_match=true", diff);
}

// ── fixture loader ───────────────────────────────────────────────────────

fn load_fixture() -> serde_json::Value {
    let path = Path::new(FIXTURE_PATH);
    if !path.exists() {
        panic!(
            "Fixture not found at {}. Generate it first:\n  python3 tests/k3_contract_fixture.py",
            path.display()
        );
    }
    let text = std::fs::read_to_string(path).expect("Failed to read fixture");
    serde_json::from_str(&text).expect("Failed to parse fixture JSON")
}
