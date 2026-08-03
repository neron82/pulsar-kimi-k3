//! GPU kernel selftests as a cargo test gate (see scripts/check.sh).
//! Each wrapper runs a device-vs-host reference comparison implemented in
//! pulsar_kernels.cu. Needs a CUDA GPU; run serially (--test-threads=1).
#![cfg(target_os = "linux")]

macro_rules! selftest {
    ($name:ident) => {
        #[test]
        fn $name() {
            assert!(kernels::$name(), stringify!($name));
        }
    };
}

selftest!(gqa_selftest);
selftest!(sconv_selftest);
selftest!(q8_0_matmul_selftest);
selftest!(qk_matmul_selftest);
selftest!(router_selftest);
selftest!(moe_selftest);
selftest!(glue_selftest);
selftest!(mla_selftest);
selftest!(idx_selftest);
selftest!(dsv4_selftest);
selftest!(qwen35_selftest);
selftest!(k3_situ_glu_selftest);
selftest!(k3_router_selftest);
selftest!(k3_kda_step_selftest);
selftest!(k3_mla_absorbed_attn_split_selftest);
selftest!(k3_mla_cached_attn_32k_selftest);
selftest!(k3_mla_absorbed_attn_split_q8_selftest);
selftest!(k3_mla_absorbed_attn_fused_selftest);
