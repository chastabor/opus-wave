#![cfg(any(feature = "dnn-dred", feature = "dnn-osce", feature = "dnn-deep-plc"))]

//! DNN NN primitive comparison tests: Rust opus-dnn vs C libopus.
//!
//! Tests Layer 0 (activations), Layer 1 (linear), and Layer 2 (GRU)
//! using small handcrafted layers with known weights.

mod common;

use opus_wave::dnn::nnet::Activation;
use opus_wave::dnn::nnet::LinearLayer;
use opus_wave::dnn::nnet::activations::compute_activation;
use opus_wave::dnn::nnet::linear::compute_linear;
use opus_wave::dnn::nnet::ops::{compute_generic_dense, compute_generic_gru};

use common::assert_f32_slice_close as assert_close;

// ============ Layer 0: Activations ============

#[test]
fn test_activation_sigmoid_vs_c() {
    let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.25).collect();
    let n = input.len();

    // Rust: in-place activation
    let mut rust_out = input.clone();
    compute_activation(&mut rust_out, Activation::Sigmoid);

    // C: out-of-place activation (output, input are separate)
    let mut c_out = vec![0.0f32; n];
    opus_ffi::c_compute_activation(&mut c_out, &input, 1);

    // Tolerance: ~1e-4 due to different polynomial approximations (rational vs table-based SIMD).
    assert_close(&rust_out, &c_out, 5e-4, "sigmoid");
}

#[test]
fn test_activation_tanh_vs_c() {
    let input: Vec<f32> = (-20..20).map(|i| i as f32 * 0.25).collect();
    let n = input.len();

    let mut rust_out = input.clone();
    compute_activation(&mut rust_out, Activation::Tanh);

    let mut c_out = vec![0.0f32; n];
    opus_ffi::c_compute_activation(&mut c_out, &input, 2);

    // Tolerance: ~5e-4 due to different approximation paths (Rust scalar vs C SSE2).
    assert_close(&rust_out, &c_out, 5e-4, "tanh");
}

#[test]
fn test_activation_relu_vs_c() {
    let input: Vec<f32> = (-10..10).map(|i| i as f32 * 0.5).collect();
    let n = input.len();

    let mut rust_out = input.clone();
    compute_activation(&mut rust_out, Activation::Relu);

    let mut c_out = vec![0.0f32; n];
    opus_ffi::c_compute_activation(&mut c_out, &input, 3);

    assert_close(&rust_out, &c_out, 1e-6, "relu");
}

#[test]
fn test_activation_swish_vs_c() {
    let input: Vec<f32> = (-16..16).map(|i| i as f32 * 0.3).collect();
    let n = input.len();

    let mut rust_out = input.clone();
    compute_activation(&mut rust_out, Activation::Swish);

    let mut c_out = vec![0.0f32; n];
    opus_ffi::c_compute_activation(&mut c_out, &input, 5);

    assert_close(&rust_out, &c_out, 5e-4, "swish");
}

// ============ Layer 1: Linear ============

#[test]
fn test_linear_identity_vs_c() {
    // 8x8 identity matrix + bias=0.5
    let n = 8;
    let mut weights = vec![0.0f32; n * n];
    for i in 0..n {
        weights[i * n + i] = 1.0;
    }
    let bias = vec![0.5f32; n];
    let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut rust_out = vec![0.0f32; n];
    let layer = LinearLayer {
        bias: Some(bias.clone()),
        subias: None,
        weights: None,
        float_weights: Some(weights.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: n,
        nb_outputs: n,
    };
    compute_linear(&layer, &mut rust_out, &input);

    let mut c_out = vec![0.0f32; n];
    opus_ffi::c_compute_linear(&mut c_out, &weights, &bias, n, n, &input);

    assert_close(&rust_out, &c_out, 1e-5, "linear_identity");
}

#[test]
fn test_linear_random_vs_c() {
    // 16x8 matrix with pseudo-random weights
    let ni = 8;
    let no = 16;
    let mut weights = vec![0.0f32; ni * no];
    let mut rng = 42u32;
    for w in weights.iter_mut() {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        *w = ((rng >> 16) as f32 / 32768.0) - 1.0;
    }
    let bias: Vec<f32> = (0..no).map(|i| i as f32 * 0.1 - 0.8).collect();
    let input: Vec<f32> = (0..ni).map(|i| (i as f32 - 3.5) * 0.3).collect();

    let mut rust_out = vec![0.0f32; no];
    let layer = LinearLayer {
        bias: Some(bias.clone()),
        subias: None,
        weights: None,
        float_weights: Some(weights.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: no,
    };
    compute_linear(&layer, &mut rust_out, &input);

    let mut c_out = vec![0.0f32; no];
    opus_ffi::c_compute_linear(&mut c_out, &weights, &bias, ni, no, &input);

    assert_close(&rust_out, &c_out, 1e-4, "linear_random");
}

// ============ Layer 2: Dense + GRU ============

#[test]
fn test_generic_dense_relu_vs_c() {
    let ni = 8;
    let no = 8;
    let mut weights = vec![0.0f32; ni * no];
    for i in 0..ni {
        weights[i * no + i] = 1.0;
    }
    let bias = vec![-3.0f32; no];
    let input = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let mut rust_out = vec![0.0f32; no];
    let layer = LinearLayer {
        bias: Some(bias.clone()),
        subias: None,
        weights: None,
        float_weights: Some(weights.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: no,
    };
    compute_generic_dense(&layer, &mut rust_out, &input, Activation::Relu);

    let mut c_out = vec![0.0f32; no];
    opus_ffi::c_compute_generic_dense(&mut c_out, &input, &weights, &bias, ni, no, 3);

    assert_close(&rust_out, &c_out, 1e-5, "dense_relu");
}

#[test]
fn test_gru_vs_c() {
    // Small 4-neuron GRU with 4-dim input
    let ni = 4;
    let nn = 4;
    let n3 = 3 * nn;

    // Generate pseudo-random weights
    let mut rng = 123u32;
    let mut rand_vec = |size: usize| -> Vec<f32> {
        (0..size)
            .map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    };

    let input_w = rand_vec(ni * n3);
    let input_b = rand_vec(n3);
    let recur_w = rand_vec(nn * n3);
    let recur_b = rand_vec(n3);
    let recur_diag = rand_vec(n3);

    let input_data = [0.5f32, -0.3, 0.8, -0.1];

    // Rust
    let input_layer = LinearLayer {
        bias: Some(input_b.clone()),
        subias: None,
        weights: None,
        float_weights: Some(input_w.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: n3,
    };
    let recur_layer = LinearLayer {
        bias: Some(recur_b.clone()),
        subias: None,
        weights: None,
        float_weights: Some(recur_w.clone()),
        weights_idx: None,
        diag: Some(recur_diag.clone()),
        scale: None,
        nb_inputs: nn,
        nb_outputs: n3,
    };

    let mut rust_state = vec![0.0f32; nn];
    compute_generic_gru(&input_layer, &recur_layer, &mut rust_state, &input_data);

    // C
    let mut c_state = vec![0.0f32; nn];
    opus_ffi::c_compute_generic_gru(
        &mut c_state,
        &input_w,
        &input_b,
        &recur_w,
        &recur_b,
        &recur_diag,
        ni,
        nn,
        &input_data,
    );

    // Small numerical differences expected from operation ordering.
    assert_close(&rust_state, &c_state, 2e-4, "gru_state");
}

#[test]
fn test_gru_multi_step_vs_c() {
    // Run GRU for several steps to verify state accumulation matches
    let ni = 8;
    let nn = 8;
    let n3 = 3 * nn;

    let mut rng = 456u32;
    let mut rand_vec = |size: usize| -> Vec<f32> {
        (0..size)
            .map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng >> 16) as f32 / 65536.0) - 0.5
            })
            .collect()
    };

    let input_w = rand_vec(ni * n3);
    let input_b = rand_vec(n3);
    let recur_w = rand_vec(nn * n3);
    let recur_b = rand_vec(n3);
    let recur_diag = rand_vec(n3);

    let input_layer = LinearLayer {
        bias: Some(input_b.clone()),
        subias: None,
        weights: None,
        float_weights: Some(input_w.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: n3,
    };
    let recur_layer = LinearLayer {
        bias: Some(recur_b.clone()),
        subias: None,
        weights: None,
        float_weights: Some(recur_w.clone()),
        weights_idx: None,
        diag: Some(recur_diag.clone()),
        scale: None,
        nb_inputs: nn,
        nb_outputs: n3,
    };

    let mut rust_state = vec![0.0f32; nn];
    let mut c_state = vec![0.0f32; nn];

    // Run 10 steps with different inputs
    for step in 0..10 {
        let input_data: Vec<f32> = (0..ni)
            .map(|i| ((step * ni + i) as f32 * 0.1).sin())
            .collect();

        compute_generic_gru(&input_layer, &recur_layer, &mut rust_state, &input_data);
        opus_ffi::c_compute_generic_gru(
            &mut c_state,
            &input_w,
            &input_b,
            &recur_w,
            &recur_b,
            &recur_diag,
            ni,
            nn,
            &input_data,
        );
    }

    assert_close(&rust_state, &c_state, 5e-4, "gru_10step_state");
}

// ============ Coverage gap: edge-case float activations ============

#[test]
fn test_activation_edge_cases_vs_c() {
    // Test extreme values, zero, negative zero, very small values.
    let input = [
        0.0f32, -0.0, 1e-38, -1e-38, // zero and subnormals
        1e10, -1e10, 100.0, -100.0, // large magnitudes
        1e-7, -1e-7, 0.5, -0.5, // typical range
    ];
    let n = input.len();

    for (act_id, act_enum, label) in [
        (1i32, Activation::Sigmoid, "sigmoid_edge"),
        (2, Activation::Tanh, "tanh_edge"),
        (3, Activation::Relu, "relu_edge"),
        (5, Activation::Swish, "swish_edge"),
    ] {
        let mut rust_out = input.to_vec();
        compute_activation(&mut rust_out, act_enum);

        let mut c_out = vec![0.0f32; n];
        opus_ffi::c_compute_activation(&mut c_out, &input, act_id);

        // Larger tolerance for extreme inputs where approximations diverge most.
        assert_close(&rust_out, &c_out, 2e-3, label);
    }
}

// ============ Coverage gap: non-multiple-of-8 dimensions ============

#[test]
fn test_linear_odd_dimensions_vs_c() {
    // 13x7 matrix — neither dimension is a multiple of 8.
    // This exercises the scalar fallback path in sgemv (not 8x or 16x unrolled).
    let ni = 7;
    let no = 13;
    let mut rng = 789u32;
    let mut rand_f32 = || -> f32 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        ((rng >> 16) as f32 / 32768.0) - 1.0
    };

    let weights: Vec<f32> = (0..ni * no).map(|_| rand_f32()).collect();
    let bias: Vec<f32> = (0..no).map(|_| rand_f32() * 0.5).collect();
    let input: Vec<f32> = (0..ni).map(|_| rand_f32()).collect();

    let layer = LinearLayer {
        bias: Some(bias.clone()),
        subias: None,
        weights: None,
        float_weights: Some(weights.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: no,
    };
    let mut rust_out = vec![0.0f32; no];
    compute_linear(&layer, &mut rust_out, &input);

    let mut c_out = vec![0.0f32; no];
    opus_ffi::c_compute_linear(&mut c_out, &weights, &bias, ni, no, &input);

    assert_close(&rust_out, &c_out, 1e-4, "linear_7x13");
}

#[test]
fn test_linear_5x3_vs_c() {
    // Very small odd dimensions.
    let ni = 3;
    let no = 5;
    let weights: Vec<f32> = (0..ni * no).map(|i| (i as f32 * 0.3 - 2.0).sin()).collect();
    let bias = vec![0.1f32; no];
    let input = [1.0f32, -0.5, 2.0];

    let layer = LinearLayer {
        bias: Some(bias.clone()),
        subias: None,
        weights: None,
        float_weights: Some(weights.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: no,
    };
    let mut rust_out = vec![0.0f32; no];
    compute_linear(&layer, &mut rust_out, &input);

    let mut c_out = vec![0.0f32; no];
    opus_ffi::c_compute_linear(&mut c_out, &weights, &bias, ni, no, &input);

    assert_close(&rust_out, &c_out, 1e-5, "linear_3x5");
}

// ============ Coverage gap: int8 quantized weights ============

#[test]
fn test_linear_int8_vs_c() {
    // 8x4 int8 quantized layer with small known weights.
    // cgemv8x4 requires nb_outputs % 8 == 0 and nb_inputs % 4 == 0.
    let ni = 4;
    let no = 8;

    // Small known weights — identity-like pattern scaled to int8 range.
    let mut weights = vec![0i8; ni * no];
    for i in 0..ni.min(no) {
        weights[i * 4 + (i % 4)] = 64; // ~0.5 after quantization
    }
    let bias = vec![0.0f32; no];
    let scale = vec![1.0f32; no]; // uniform scale
    // Input in [-1, 1] range (the cgemv8x4 quantizes to round(127*x))
    let input = [0.5f32, -0.5, 0.25, -0.25];

    // Rust
    let layer = LinearLayer {
        bias: Some(bias.clone()),
        subias: None,
        weights: Some(weights.clone()),
        float_weights: None,
        weights_idx: None,
        diag: None,
        scale: Some(scale.clone()),
        nb_inputs: ni,
        nb_outputs: no,
    };
    let mut rust_out = vec![0.0f32; no];
    compute_linear(&layer, &mut rust_out, &input);

    // C
    let mut c_out = vec![0.0f32; no];
    opus_ffi::c_compute_linear_int8(&mut c_out, &weights, &bias, &scale, ni, no, &input);

    // Unsigned quantization path: tolerance depends on int8 rounding.
    assert_close(&rust_out, &c_out, 1.0, "linear_int8_4x8");
}

#[test]
fn test_linear_int8_16x8_vs_c() {
    // 16x8 int8 layer — larger matrix with random weights.
    let ni = 8;
    let no = 16;

    let mut rng = 654u32;
    let mut rand_i8 = || -> i8 {
        rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
        (rng >> 16) as i8
    };

    let weights: Vec<i8> = (0..ni * no).map(|_| rand_i8()).collect();
    let bias: Vec<f32> = (0..no).map(|i| (i as f32 - 8.0) * 0.05).collect();
    let scale: Vec<f32> = vec![1.0f32; no];
    let input: Vec<f32> = (0..ni).map(|i| (i as f32 * 0.7).sin()).collect();

    let layer = LinearLayer {
        bias: Some(bias.clone()),
        subias: None,
        weights: Some(weights.clone()),
        float_weights: None,
        weights_idx: None,
        diag: None,
        scale: Some(scale.clone()),
        nb_inputs: ni,
        nb_outputs: no,
    };
    let mut rust_out = vec![0.0f32; no];
    compute_linear(&layer, &mut rust_out, &input);

    let mut c_out = vec![0.0f32; no];
    opus_ffi::c_compute_linear_int8(&mut c_out, &weights, &bias, &scale, ni, no, &input);

    assert_close(&rust_out, &c_out, 5.0, "linear_int8_8x16");
}

// ============ Coverage gap: non-zero initial GRU state ============

#[test]
fn test_gru_nonzero_initial_state_vs_c() {
    // GRU with pre-filled state to verify reset/update gate interaction
    // with existing state values on the very first step.
    let ni = 8;
    let nn = 8;
    let n3 = 3 * nn;

    let mut rng = 999u32;
    let mut rand_vec = |size: usize| -> Vec<f32> {
        (0..size)
            .map(|_| {
                rng = rng.wrapping_mul(1103515245).wrapping_add(12345);
                ((rng >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    };

    let input_w = rand_vec(ni * n3);
    let input_b = rand_vec(n3);
    let recur_w = rand_vec(nn * n3);
    let recur_b = rand_vec(n3);
    let recur_diag = rand_vec(n3);

    let input_layer = LinearLayer {
        bias: Some(input_b.clone()),
        subias: None,
        weights: None,
        float_weights: Some(input_w.clone()),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: n3,
    };
    let recur_layer = LinearLayer {
        bias: Some(recur_b.clone()),
        subias: None,
        weights: None,
        float_weights: Some(recur_w.clone()),
        weights_idx: None,
        diag: Some(recur_diag.clone()),
        scale: None,
        nb_inputs: nn,
        nb_outputs: n3,
    };

    // Non-zero initial state
    let initial_state: Vec<f32> = (0..nn).map(|i| (i as f32 * 0.3 - 1.0).tanh()).collect();
    let mut rust_state = initial_state.clone();
    let mut c_state = initial_state.clone();

    let input_data: Vec<f32> = (0..ni).map(|i| (i as f32 * 0.5).sin()).collect();

    // Single step from non-zero state
    compute_generic_gru(&input_layer, &recur_layer, &mut rust_state, &input_data);
    opus_ffi::c_compute_generic_gru(
        &mut c_state,
        &input_w,
        &input_b,
        &recur_w,
        &recur_b,
        &recur_diag,
        ni,
        nn,
        &input_data,
    );

    assert_close(&rust_state, &c_state, 2e-4, "gru_nonzero_init");

    // Continue for 5 more steps to verify accumulation from non-zero start
    for step in 1..6 {
        let input_data: Vec<f32> = (0..ni)
            .map(|i| ((step * ni + i) as f32 * 0.2).cos())
            .collect();
        compute_generic_gru(&input_layer, &recur_layer, &mut rust_state, &input_data);
        opus_ffi::c_compute_generic_gru(
            &mut c_state,
            &input_w,
            &input_b,
            &recur_w,
            &recur_b,
            &recur_diag,
            ni,
            nn,
            &input_data,
        );
    }

    assert_close(&rust_state, &c_state, 5e-4, "gru_nonzero_init_6steps");
}
