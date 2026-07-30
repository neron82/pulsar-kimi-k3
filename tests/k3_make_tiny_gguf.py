#!/usr/bin/env python3
"""
Generate a tiny synthetic K3-shaped GGUF fixture for the AtomicBot reference.

Dependency-free: uses only struct, json, math (no numpy, no torch).

Output: tests/k3_tiny_fixture.gguf
"""

import json
import math
import struct
import sys
from pathlib import Path

FIXTURE_PATH = Path(__file__).parent / "k3_contract_fixture.json"
OUTPUT_PATH  = Path(__file__).parent / "k3_tiny_fixture.gguf"

# ── GGUF constants ───────────────────────────────────────────────────────
GGUF_MAGIC  = 0x46554747  # "GGUF"
GGUF_VERSION = 3

GGUF_TYPE_UINT8   = 0
GGUF_TYPE_INT8    = 1
GGUF_TYPE_UINT16  = 2
GGUF_TYPE_INT16   = 3
GGUF_TYPE_UINT32  = 4
GGUF_TYPE_INT32   = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL    = 7
GGUF_TYPE_STRING  = 8
GGUF_TYPE_ARRAY   = 9
GGUF_TYPE_UINT64  = 10
GGUF_TYPE_INT64   = 11
GGUF_TYPE_FLOAT64 = 12

GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8

# K3 KV keys
LLM_KV_GENERAL_ARCHITECTURE        = "general.architecture"
LLM_KV_CONTEXT_LENGTH              = "context_length"
LLM_KV_EMBEDDING_LENGTH            = "embedding_length"
LLM_KV_BLOCK_COUNT                 = "block_count"
LLM_KV_HEAD_COUNT                  = "attention.head_count"
LLM_KV_HEAD_COUNT_KV               = "attention.head_count_kv"
LLM_KV_KEY_LENGTH                  = "attention.key_length"
LLM_KV_VALUE_LENGTH                = "attention.value_length"
LLM_KV_FEED_FORWARD_LENGTH         = "feed_forward_length"
LLM_KV_EXPERT_COUNT                = "expert_count"
LLM_KV_EXPERT_USED_COUNT           = "expert_used_count"
LLM_KV_EXPERT_SHARED_COUNT         = "expert_shared_count"
LLM_KV_LAYER_NORM_RMS_EPS          = "attention.layer_norm_rms_epsilon"
LLM_KV_ATTENTION_KEY_LENGTH_MLA    = "attention.key_length_mla"
LLM_KV_ATTENTION_VALUE_LENGTH_MLA  = "attention.value_length_mla"
LLM_KV_ATTENTION_KV_LORA_RANK      = "attention.kv_lora_rank"
LLM_KV_ATTENTION_Q_LORA_RANK       = "attention.q_lora_rank"
LLM_KV_SSM_CONV_KERNEL             = "ssm.conv_kernel"
LLM_KV_KDA_HEAD_DIM                = "kda.head_dim"
LLM_KV_KDA_GATE_LOWER_BOUND        = "kda.gate_lower_bound"
LLM_KV_SITU_BETA                  = "situ_beta"
LLM_KV_SITU_LINEAR_BETA            = "situ_linear_beta"
LLM_KV_ATTN_RES_BLOCK_SIZE         = "attn_res_block_size"
LLM_KV_MOE_LATENT_SIZE              = "moe_latent_size"
LLM_KV_EXPERT_FEED_FORWARD_LENGTH  = "expert_feed_forward_length"
LLM_KV_LEADING_DENSE_BLOCK_COUNT   = "leading_dense_block_count"
LLM_KV_EXPERT_WEIGHTS_SCALE         = "expert_weights_scale"
LLM_KV_EXPERT_GATING_FUNC          = "expert_gating_func"


def pack_f32(data):
    """Pack a list of floats as F32 bytes."""
    return struct.pack(f"<{len(data)}f", *data)


def pack_zeros(n):
    """Pack n zero floats."""
    return b'\x00' * (n * 4)


def make_tiny_gguf():
    with open(FIXTURE_PATH) as f:
        fx = json.load(f)

    shapes = fx["meta"]["shapes"]
    n_embd = shapes["n_embd"]
    n_head = shapes["n_head"]
    head_dim = shapes["head_dim"]
    d_inner = shapes["d_inner"]
    n_expert = shapes["n_expert"]
    n_expert_shared = shapes["n_expert_shared"]
    moe_latent = shapes["moe_latent"]
    n_ff_exp = shapes["n_ff_exp"]
    n_ff_dense = shapes["n_ff_dense"]
    n_layer = shapes["n_layer"]
    n_layer_dense_lead = shapes["n_layer_dense_lead"]
    res_block = shapes["res_block"]
    gate_lower_bound = shapes["gate_lower_bound"]
    situ_beta = shapes["situ_beta"]
    situ_linear_beta = shapes["situ_linear_beta"]

    kv_lora_rank = 32
    q_lora_rank = 32
    n_embd_head_k_mla = head_dim
    n_embd_head_v_mla = head_dim
    ssm_d_conv = 4
    n_vocab = 128

    # ── Build KV metadata ────────────────────────────────────────────────
    kv_pairs = []

    def encoded_key(key):
        if key == LLM_KV_GENERAL_ARCHITECTURE or key.startswith("tokenizer."):
            return key.encode("utf-8")
        return ("kimi-k3." + key).encode("utf-8")

    def add_kv_str(key, val):
        k_enc = encoded_key(key)
        v_enc = val.encode("utf-8")
        kv_pairs.append(struct.pack("<Q", len(k_enc)) + k_enc +
                        struct.pack("<I", GGUF_TYPE_STRING) +
                        struct.pack("<Q", len(v_enc)) + v_enc)

    def add_kv_u32(key, val):
        k_enc = encoded_key(key)
        kv_pairs.append(struct.pack("<Q", len(k_enc)) + k_enc +
                        struct.pack("<I", GGUF_TYPE_UINT32) +
                        struct.pack("<I", val))

    def add_kv_f32(key, val):
        k_enc = encoded_key(key)
        kv_pairs.append(struct.pack("<Q", len(k_enc)) + k_enc +
                        struct.pack("<I", GGUF_TYPE_FLOAT32) +
                        struct.pack("<f", val))

    def add_kv_u32_array(key, vals):
        k_enc = encoded_key(key)
        payload = (struct.pack("<I", GGUF_TYPE_ARRAY) +
                   struct.pack("<I", GGUF_TYPE_UINT32) +
                   struct.pack("<Q", len(vals)) +
                   b"".join(struct.pack("<I", v) for v in vals))
        kv_pairs.append(struct.pack("<Q", len(k_enc)) + k_enc + payload)

    def add_kv_string_array(key, vals):
        k_enc = encoded_key(key)
        payload = (struct.pack("<I", GGUF_TYPE_ARRAY) +
                   struct.pack("<I", GGUF_TYPE_STRING) +
                   struct.pack("<Q", len(vals)) +
                   b"".join(struct.pack("<Q", len(v.encode("utf-8"))) + v.encode("utf-8") for v in vals))
        kv_pairs.append(struct.pack("<Q", len(k_enc)) + k_enc + payload)

    def add_kv_f32_array(key, vals):
        k_enc = encoded_key(key)
        payload = (struct.pack("<I", GGUF_TYPE_ARRAY) +
                   struct.pack("<I", GGUF_TYPE_FLOAT32) +
                   struct.pack("<Q", len(vals)) +
                   b"".join(struct.pack("<f", v) for v in vals))
        kv_pairs.append(struct.pack("<Q", len(k_enc)) + k_enc + payload)

    add_kv_str(LLM_KV_GENERAL_ARCHITECTURE, "kimi-k3")
    add_kv_str("tokenizer.ggml.model", "gpt2")
    add_kv_string_array("tokenizer.ggml.tokens", [f"t{i}" for i in range(n_vocab)])
    add_kv_f32_array("tokenizer.ggml.scores", [0.0] * n_vocab)
    add_kv_string_array("tokenizer.ggml.merges", [])

    add_kv_u32(LLM_KV_CONTEXT_LENGTH, 128)
    add_kv_u32(LLM_KV_EMBEDDING_LENGTH, n_embd)
    add_kv_u32(LLM_KV_BLOCK_COUNT, n_layer)
    add_kv_u32(LLM_KV_HEAD_COUNT, n_head)
    add_kv_u32(LLM_KV_KEY_LENGTH, head_dim)
    add_kv_u32(LLM_KV_VALUE_LENGTH, head_dim)
    add_kv_u32_array(LLM_KV_HEAD_COUNT_KV, [0] + [1] * (n_layer - 1))
    add_kv_f32(LLM_KV_LAYER_NORM_RMS_EPS, 1e-6)
    add_kv_u32(LLM_KV_FEED_FORWARD_LENGTH, n_ff_dense)
    add_kv_u32(LLM_KV_EXPERT_COUNT, n_expert)
    add_kv_u32(LLM_KV_EXPERT_USED_COUNT, shapes["top_k"])
    add_kv_u32(LLM_KV_EXPERT_SHARED_COUNT, n_expert_shared)
    add_kv_u32(LLM_KV_ATTENTION_KEY_LENGTH_MLA, n_embd_head_k_mla)
    add_kv_u32(LLM_KV_ATTENTION_VALUE_LENGTH_MLA, n_embd_head_v_mla)
    add_kv_u32(LLM_KV_ATTENTION_KV_LORA_RANK, kv_lora_rank)
    add_kv_u32(LLM_KV_ATTENTION_Q_LORA_RANK, q_lora_rank)
    add_kv_u32(LLM_KV_SSM_CONV_KERNEL, ssm_d_conv)
    add_kv_u32(LLM_KV_KDA_HEAD_DIM, head_dim)
    add_kv_f32(LLM_KV_KDA_GATE_LOWER_BOUND, gate_lower_bound)
    add_kv_f32(LLM_KV_SITU_BETA, situ_beta)
    add_kv_f32(LLM_KV_SITU_LINEAR_BETA, situ_linear_beta)
    add_kv_u32(LLM_KV_ATTN_RES_BLOCK_SIZE, res_block)
    add_kv_u32(LLM_KV_MOE_LATENT_SIZE, moe_latent)
    add_kv_u32(LLM_KV_EXPERT_FEED_FORWARD_LENGTH, n_ff_exp)
    add_kv_u32(LLM_KV_LEADING_DENSE_BLOCK_COUNT, n_layer_dense_lead)
    add_kv_f32(LLM_KV_EXPERT_WEIGHTS_SCALE, 1.0)
    add_kv_u32(LLM_KV_EXPERT_GATING_FUNC, 0)

    kv_data = b''.join(kv_pairs)

    # ── Build tensor info list ────────────────────────────────────────────
    tensor_infos = []  # (name, n_dims, [dim0, dim1, ...], ggml_type)

    def ti(name, dims):
        ggml_type = GGML_TYPE_Q8_0 if name.endswith("_exps.weight") else GGML_TYPE_F32
        tensor_infos.append((name, len(dims), dims, ggml_type))

    # Global tensors
    ti("token_embd.weight", [n_embd, n_vocab])
    ti("output_norm.weight", [n_embd])
    ti("output.weight", [n_embd, n_vocab])
    ti("output_res_norm.weight", [n_embd])
    ti("output_res_proj.weight", [n_embd])

    for i in range(n_layer):
        ti(f"blk.{i}.attn_norm.weight", [n_embd])
        ti(f"blk.{i}.attn_res_norm.weight", [n_embd])
        ti(f"blk.{i}.attn_res_proj.weight", [n_embd])
        ti(f"blk.{i}.ffn_res_norm.weight", [n_embd])
        ti(f"blk.{i}.ffn_res_proj.weight", [n_embd])
        ti(f"blk.{i}.ffn_norm.weight", [n_embd])

        if i == 0:  # KDA layer
            ti(f"blk.{i}.ssm_conv1d_q.weight", [ssm_d_conv, 1, d_inner, 1])
            ti(f"blk.{i}.ssm_conv1d_k.weight", [ssm_d_conv, 1, d_inner, 1])
            ti(f"blk.{i}.ssm_conv1d_v.weight", [ssm_d_conv, 1, d_inner, 1])
            ti(f"blk.{i}.attn_q.weight", [n_embd, d_inner])
            ti(f"blk.{i}.attn_k.weight", [n_embd, d_inner])
            ti(f"blk.{i}.attn_v.weight", [n_embd, d_inner])
            ti(f"blk.{i}.ssm_f_a.weight", [n_embd, head_dim])
            ti(f"blk.{i}.ssm_f_b.weight", [head_dim, d_inner])
            ti(f"blk.{i}.ssm_beta.weight", [n_embd, n_head])
            ti(f"blk.{i}.ssm_a", [n_head])
            ti(f"blk.{i}.ssm_dt.bias", [d_inner])
            ti(f"blk.{i}.attn_gate.weight", [n_embd, d_inner])
            ti(f"blk.{i}.ssm_norm.weight", [head_dim])
            ti(f"blk.{i}.attn_output.weight", [d_inner, n_embd])
        else:  # MLA layer
            ti(f"blk.{i}.attn_q_a.weight", [n_embd, q_lora_rank])
            ti(f"blk.{i}.attn_q_a_norm.weight", [q_lora_rank])
            ti(f"blk.{i}.attn_q_b.weight", [q_lora_rank, n_head * n_embd_head_k_mla])
            ti(f"blk.{i}.attn_kv_a_mqa.weight", [n_embd, kv_lora_rank + head_dim])
            ti(f"blk.{i}.attn_kv_b.weight", [kv_lora_rank, n_head * n_embd_head_k_mla])
            ti(f"blk.{i}.attn_kv_a_norm.weight", [kv_lora_rank])
            ti(f"blk.{i}.attn_gate.weight", [n_embd, n_head * n_embd_head_v_mla])
            ti(f"blk.{i}.attn_output.weight", [n_head * n_embd_head_v_mla, n_embd])

        if i < n_layer_dense_lead:
            ti(f"blk.{i}.ffn_gate.weight", [n_embd, n_ff_dense])
            ti(f"blk.{i}.ffn_down.weight", [n_ff_dense, n_embd])
            ti(f"blk.{i}.ffn_up.weight", [n_embd, n_ff_dense])
        else:
            ti(f"blk.{i}.ffn_gate_inp.weight", [n_embd, n_expert])
            ti(f"blk.{i}.exp_probs_b.bias", [n_expert])
            ti(f"blk.{i}.ffn_latent_down.weight", [n_embd, moe_latent])
            ti(f"blk.{i}.ffn_latent_norm.weight", [moe_latent])
            ti(f"blk.{i}.ffn_latent_up.weight", [moe_latent, n_embd])
            ti(f"blk.{i}.ffn_gate_exps.weight", [moe_latent, n_ff_exp, n_expert])
            ti(f"blk.{i}.ffn_down_exps.weight", [n_ff_exp, moe_latent, n_expert])
            ti(f"blk.{i}.ffn_up_exps.weight", [moe_latent, n_ff_exp, n_expert])
            ti(f"blk.{i}.ffn_gate_shexp.weight", [n_embd, n_ff_exp * n_expert_shared])
            ti(f"blk.{i}.ffn_down_shexp.weight", [n_ff_exp * n_expert_shared, n_embd])
            ti(f"blk.{i}.ffn_up_shexp.weight", [n_embd, n_ff_exp * n_expert_shared])

    # Compute sizes and deterministic zero payloads using the actual GGML type.
    tensor_data_offsets = []
    tensor_payloads = []
    current_offset = 0
    for name, n_dims, dims, ggml_type in tensor_infos:
        if ggml_type == GGML_TYPE_Q8_0:
            row_elems = dims[0]
            row_bytes = ((row_elems + 31) // 32) * 34
            rows = 1
            for d in dims[1:]:
                rows *= d
            n_bytes = row_bytes * rows
            payload = b"\x00" * n_bytes
        else:
            n_elems = 1
            for d in dims:
                n_elems *= d
            n_bytes = n_elems * 4
            payload = b"\x00" * n_bytes
        # Align each tensor payload to 32 bytes.
        pad = (32 - (n_bytes % 32)) % 32
        tensor_data_offsets.append((current_offset, n_bytes, pad))
        tensor_payloads.append(payload)
        current_offset += n_bytes + pad

    # ── Write the file ───────────────────────────────────────────────────
    with open(OUTPUT_PATH, "wb") as f:
        # GGUF v3 header: tensor_count followed by metadata KV count.
        f.write(struct.pack("<I", GGUF_MAGIC))
        f.write(struct.pack("<I", GGUF_VERSION))
        f.write(struct.pack("<Q", len(tensor_infos)))
        f.write(struct.pack("<Q", len(kv_pairs)))

        # KV data
        f.write(kv_data)

        # Tensor infos carry offsets relative to the aligned tensor-data start.
        for index, (name, n_dims, dims, ggml_type) in enumerate(tensor_infos):
            name_enc = name.encode("utf-8")
            f.write(struct.pack("<Q", len(name_enc)))
            f.write(name_enc)
            f.write(struct.pack("<I", n_dims))
            for d in dims:
                f.write(struct.pack("<Q", d))
            f.write(struct.pack("<I", ggml_type))
            f.write(struct.pack("<Q", tensor_data_offsets[index][0]))

        # GGUF tensor data begins at the next 32-byte boundary.
        while f.tell() % 32:
            f.write(b"\x00")
        for (offset, n_bytes, pad), payload in zip(tensor_data_offsets, tensor_payloads):
            assert len(payload) == n_bytes
            f.write(payload)
            if pad:
                f.write(b"\x00" * pad)

    file_size = OUTPUT_PATH.stat().st_size
    print(f"✅ Tiny K3 GGUF fixture written: {OUTPUT_PATH}")
    print(f"   Layers: {n_layer} (KDA: 1, MLA: {n_layer - 1})")
    print(f"   Tensors: {len(tensor_infos)}")
    print(f"   File size: {file_size} bytes")
    print(f"\n   To use with AtomicBot reference:")
    print(f"     /tmp/k3-cpu-build/bin/llama-cli -m {OUTPUT_PATH} -p 'Hello' -n 1 --temp 0.0 --logits all")


if __name__ == "__main__":
    make_tiny_gguf()
