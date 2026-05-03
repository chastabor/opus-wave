#![cfg(any(feature = "dnn-dred", feature = "dnn-osce", feature = "dnn-deep-plc"))]

//! RDOVAE sparse linear divergence investigation.
//!
//! Tests sparse_sgemv8x4 and the first sparse dense layer of RDOVAE encoder
//! against C using actual model weights.

mod common;

/// Test the first sparse layer (gdense1) of RDOVAE encoder: Rust compute_linear vs
/// C c_compute_linear using the EXACT SAME weight data from the Rust model.
///
/// gdense1 has: nb_inputs=544, nb_outputs=128, weights_idx present (sparse).
#[test]
fn test_rdovae_gdense1_sparse_vs_c() {
    let Some(blob) = common::load_dnn_blob("rdovae_enc.bin") else {
        eprintln!("Skipping: rdovae_enc.bin not found");
        return;
    };

    let arrays = opus_wave::dnn::nnet::weights::parse_weights(&blob).unwrap();
    let model = opus_wave::dnn::dred::rdovae_enc::init_rdovae_enc(&arrays).unwrap();

    let gdense1 = &model.gdense1;
    let nb_inputs = gdense1.nb_inputs;
    let nb_outputs = gdense1.nb_outputs;
    eprintln!("gdense1: ni={nb_inputs}, no={nb_outputs}");
    eprintln!("  has float_weights: {}", gdense1.float_weights.is_some());
    eprintln!("  has weights (int8): {}", gdense1.weights.is_some());
    eprintln!("  has weights_idx: {}", gdense1.weights_idx.is_some());
    eprintln!("  has bias: {}", gdense1.bias.is_some());
    eprintln!("  has subias: {}", gdense1.subias.is_some());
    eprintln!("  has scale: {}", gdense1.scale.is_some());

    let mut seed = 42u32;
    let input = common::gen_random_vec(nb_inputs, &mut seed);

    // Check sparsity structure: count blocks per 8-row group
    let idx = gdense1.weights_idx.as_ref().unwrap();
    eprintln!("  weights_idx len: {}", idx.len());
    eprintln!("  first few idx values: {:?}", &idx[..10.min(idx.len())]);

    let mut pos = 0;
    let mut total_blocks = 0;
    let mut group = 0;
    while pos < idx.len() {
        let nb = idx[pos] as usize;
        pos += 1 + nb;
        total_blocks += nb;
        group += 1;
        if group <= 3 {
            eprintln!("  row group {}: {} blocks", group - 1, nb);
        }
    }
    eprintln!("  total groups: {group}, total blocks: {total_blocks}");

    // The sparse float weights should have SPARSE_BLOCK_SIZE * total_blocks floats
    if let Some(ref fw) = gdense1.float_weights {
        let expected_floats = 32 * total_blocks;
        eprintln!(
            "  float_weights: {} floats (expected {})",
            fw.len(),
            expected_floats
        );
        assert_eq!(
            fw.len(),
            expected_floats,
            "Float weight count mismatch for sparse layer"
        );
    }

    // Compare Rust sparse_sgemv8x4 vs C sparse_sgemv8x4 using same weight data
    let fw = gdense1.float_weights.as_ref().unwrap();
    let idx = gdense1.weights_idx.as_ref().unwrap();

    let mut rust_sgemv = vec![0.0f32; nb_outputs];
    opus_wave::dnn::nnet::linear::compute_linear(gdense1, &mut rust_sgemv, &input);
    // Subtract bias to get raw sgemv output (float path uses regular bias)
    if let Some(ref bias) = gdense1.bias {
        for i in 0..nb_outputs {
            rust_sgemv[i] -= bias[i];
        }
    }

    let mut c_sgemv = vec![0.0f32; nb_outputs];
    opus_ffi::c_sparse_sgemv8x4(&mut c_sgemv, fw, idx, nb_outputs, &input);

    let (max_sgemv_diff, _) = common::max_abs_diff(&rust_sgemv, &c_sgemv);

    eprintln!("  sparse_sgemv8x4 max_diff = {max_sgemv_diff:.6e}");
    eprintln!(
        "  rust_sgemv[0..4] = {:?}",
        &rust_sgemv[..4.min(nb_outputs)]
    );
    eprintln!("  c_sgemv[0..4]    = {:?}", &c_sgemv[..4.min(nb_outputs)]);

    assert!(
        max_sgemv_diff < 1e-4,
        "sparse_sgemv8x4 mismatch: max_diff={max_sgemv_diff}"
    );
}

/// Compare enc_dense1 output through C model init vs Rust model init.
/// Both load from the same blob but through different init paths.
/// This isolates whether the weight loading path produces different models.
#[test]
fn test_rdovae_enc_dense1_via_c_model_init() {
    let Some(blob) = common::load_dnn_blob("rdovae_enc.bin") else {
        eprintln!("Skipping: rdovae_enc.bin not found");
        return;
    };

    let arrays = opus_wave::dnn::nnet::weights::parse_weights(&blob).unwrap();
    let model = opus_wave::dnn::dred::rdovae_enc::init_rdovae_enc(&arrays).unwrap();
    let nb_inputs = model.enc_dense1.nb_inputs;
    let nb_outputs = model.enc_dense1.nb_outputs;

    let input = vec![0.0f32; nb_inputs]; // zero input

    // Rust: using Rust-loaded model
    let mut rust_out = vec![0.0f32; nb_outputs];
    opus_wave::dnn::nnet::ops::compute_generic_dense(
        &model.enc_dense1,
        &mut rust_out,
        &input,
        opus_wave::dnn::nnet::Activation::Tanh,
    );

    // C: using C-loaded model (via wrap_rdovae_enc_dense1_only which calls C init_rdovaeenc)
    let c_out = opus_ffi::c_rdovae_enc_dense1(&blob, &input).expect("C enc_dense1 failed");

    let (max_diff, _) = common::max_abs_diff(&rust_out, &c_out);

    eprintln!("RDOVAE enc_dense1 (C model init vs Rust model init):");
    eprintln!("  ni={nb_inputs}, no={nb_outputs}");
    eprintln!("  max_diff = {max_diff:.6e}");
    eprintln!("  rust[0..4] = {:?}", &rust_out[..4.min(nb_outputs)]);
    eprintln!("  c[0..4]    = {:?}", &c_out[..4.min(nb_outputs)]);

    // If this is large, the model initialization paths load different weights
    assert!(
        max_diff < 0.01,
        "enc_dense1 C-vs-Rust model init divergence: {max_diff}"
    );
}

/// Test RDOVAE encode with different input magnitudes against C.
/// Zero input tests structural correctness; tiny input (near-zero) stays in
/// the linear region of activations, isolating weight-loading from precision issues.
#[test]
fn test_rdovae_encode_special_inputs_vs_c() {
    let Some(enc_blob) = common::load_dnn_blob("rdovae_enc.bin") else {
        eprintln!("Skipping: rdovae_enc.bin not found");
        return;
    };

    let arrays = opus_wave::dnn::nnet::weights::parse_weights(&enc_blob).unwrap();
    let model = opus_wave::dnn::dred::rdovae_enc::init_rdovae_enc(&arrays).unwrap();
    let input_dim = model.enc_dense1.nb_inputs;

    // Zero input
    rdovae_encode_vs_c_helper(&enc_blob, &model, &vec![0.0f32; input_dim], "zero");

    // Tiny input: all near-zero so tanh(x) ~ x in linear region
    rdovae_encode_vs_c_helper(&enc_blob, &model, &vec![0.001f32; input_dim], "tiny");
}

fn rdovae_encode_vs_c_helper(
    enc_blob: &[u8],
    model: &opus_wave::dnn::dred::rdovae_enc::RdovaeEnc,
    input: &[f32],
    label: &str,
) {
    let latent_dim = model.latent_dim;
    let state_dim = model.state_dim;
    let mut enc_state = opus_wave::dnn::dred::rdovae_enc::rdovae_enc_state_init(model);

    let mut rust_latents = vec![0.0f32; latent_dim];
    let mut rust_state = vec![0.0f32; state_dim];
    opus_wave::dnn::dred::rdovae_enc::dred_rdovae_encode_dframe(
        &mut enc_state,
        model,
        &mut rust_latents,
        &mut rust_state,
        input,
    );

    let (c_latents, c_state) =
        opus_ffi::c_rdovae_encode_dframe(enc_blob, input, latent_dim, state_dim)
            .expect("C RDOVAE encode failed");

    let (max_lat, _) = common::max_abs_diff(&rust_latents, &c_latents);
    let (max_st, _) = common::max_abs_diff(&rust_state, &c_state);

    eprintln!("RDOVAE encode ({label} input):");
    eprintln!("  latent max_diff = {max_lat:.6}");
    eprintln!("  state max_diff = {max_st:.6}");
    eprintln!(
        "  rust_latents[0..4] = {:?}",
        &rust_latents[..4.min(latent_dim)]
    );
    eprintln!(
        "  c_latents[0..4]    = {:?}",
        &c_latents[..4.min(latent_dim)]
    );
    eprintln!(
        "  rust_state[0..4]   = {:?}",
        &rust_state[..4.min(state_dim)]
    );
    eprintln!("  c_state[0..4]      = {:?}", &c_state[..4.min(state_dim)]);
}

/// Test: compare full RDOVAE encoder first layer (enc_dense1, NON-sparse)
/// to prove the dense path works. Then run gdense1 (sparse) to find divergence.
#[test]
fn test_rdovae_enc_dense1_vs_c() {
    let Some(blob) = common::load_dnn_blob("rdovae_enc.bin") else {
        eprintln!("Skipping: rdovae_enc.bin not found");
        return;
    };

    let arrays = opus_wave::dnn::nnet::weights::parse_weights(&blob).unwrap();
    let model = opus_wave::dnn::dred::rdovae_enc::init_rdovae_enc(&arrays).unwrap();

    let layer = &model.enc_dense1;
    let nb_inputs = layer.nb_inputs;
    let nb_outputs = layer.nb_outputs;
    eprintln!("enc_dense1: ni={nb_inputs}, no={nb_outputs}");
    eprintln!("  sparse: {}", layer.weights_idx.is_some());

    let mut seed = 42u32;
    let input = common::gen_random_vec(nb_inputs, &mut seed);

    // Rust
    let mut rust_out = vec![0.0f32; nb_outputs];
    opus_wave::dnn::nnet::linear::compute_linear(layer, &mut rust_out, &input);

    // C (using same weights — enc_dense1 uses float_weights, no int8 path)
    let fw = layer.float_weights.as_ref().unwrap();
    let bias = layer.bias.as_ref().unwrap();
    let mut c_out = vec![0.0f32; nb_outputs];
    opus_ffi::c_compute_linear(&mut c_out, fw, bias, nb_inputs, nb_outputs, &input);

    let (max_diff, _) = common::max_abs_diff(&rust_out, &c_out);

    eprintln!("  enc_dense1 max_diff = {max_diff:.6e}");
    assert!(
        max_diff < 1e-4,
        "enc_dense1 (non-sparse) diff {max_diff} too large"
    );
}

/// Verify Rust sparse_sgemv8x4 on gdense1 produces reasonable output.
#[test]
fn test_rdovae_gdense1_sparse_sanity() {
    let Some(blob) = common::load_dnn_blob("rdovae_enc.bin") else {
        eprintln!("Skipping: rdovae_enc.bin not found");
        return;
    };

    let arrays = opus_wave::dnn::nnet::weights::parse_weights(&blob).unwrap();
    let model = opus_wave::dnn::dred::rdovae_enc::init_rdovae_enc(&arrays).unwrap();

    // gdense1 is public and has weights_idx (sparse)
    let layer = &model.gdense1;
    let nb_inputs = layer.nb_inputs;
    let nb_outputs = layer.nb_outputs;
    eprintln!("gdense1 (sparse): ni={nb_inputs}, no={nb_outputs}");
    eprintln!("  float_weights: {}", layer.float_weights.is_some());
    eprintln!("  int8 weights: {}", layer.weights.is_some());
    eprintln!("  sparse (weights_idx): {}", layer.weights_idx.is_some());

    let mut seed = 42u32;
    let input = common::gen_random_vec(nb_inputs, &mut seed);

    let mut rust_out = vec![0.0f32; nb_outputs];
    opus_wave::dnn::nnet::linear::compute_linear(layer, &mut rust_out, &input);

    let has_nan = rust_out.iter().any(|v| v.is_nan() || v.is_infinite());
    let all_zero = rust_out.iter().all(|&v| v == 0.0);
    let max_val = rust_out.iter().cloned().fold(0.0f32, |a, b| a.max(b.abs()));

    eprintln!("  output: has_nan={has_nan}, all_zero={all_zero}, max_abs={max_val:.4}");
    eprintln!("  output[0..4] = {:?}", &rust_out[..4.min(nb_outputs)]);

    assert!(!has_nan, "Sparse sgemv produced NaN/Inf");
    assert!(!all_zero, "Sparse sgemv produced all zeros");
    assert!(
        max_val < 1000.0,
        "Sparse sgemv produced unreasonably large values: {max_val}"
    );
}
