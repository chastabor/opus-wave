use super::*;
use crate::nnet::activations::compute_activation;
use crate::nnet::ops::compute_generic_conv1d;
use crate::nnet::{Activation, LinearLayer};

/// Process one frame through an adaptive spectral shaping layer.
/// Matches C `adashape_process_frame` from nndsp.c.
pub fn adashape_process_frame(
    state: &mut AdaShapeState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    alpha1f: &LinearLayer,
    alpha1t: &LinearLayer,
    alpha2: &LinearLayer,
    feature_dim: usize,
    frame_size: usize,
    avg_pool_k: usize,
    interpolate_k: usize,
) {
    let hidden_dim = frame_size / interpolate_k;
    let tenv_size = frame_size / avg_pool_k;
    let f = 1.0 / avg_pool_k as f32;

    debug_assert_eq!(frame_size % avg_pool_k, 0);
    debug_assert_eq!(frame_size % interpolate_k, 0);
    debug_assert!(feature_dim + tenv_size + 1 < ADASHAPE_MAX_INPUT_DIM);

    // Build input: [features | temporal_envelope | mean]
    let mut in_buffer = [0.0f32; ADASHAPE_MAX_INPUT_DIM + ADASHAPE_MAX_FRAME_SIZE];
    in_buffer[..feature_dim].copy_from_slice(&features[..feature_dim]);

    let tenv = &mut in_buffer[feature_dim..feature_dim + tenv_size + 1];
    for v in tenv.iter_mut() {
        *v = 0.0;
    }

    // Temporal envelope: average absolute value per block
    let mut mean = 0.0f32;
    for i in 0..tenv_size {
        for k in 0..avg_pool_k {
            tenv[i] += x_in[i * avg_pool_k + k].abs();
        }
        tenv[i] = (tenv[i] * f + 1.52587890625e-05).ln();
        mean += tenv[i];
    }
    mean /= tenv_size as f32;
    for t in &mut tenv[..tenv_size] {
        *t -= mean;
    }
    tenv[tenv_size] = mean;

    // Alpha1 paths: feature conv + temporal conv, then leaky ReLU
    let mut out_buffer = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    let mut tmp_buffer = [0.0f32; ADASHAPE_MAX_FRAME_SIZE];
    compute_generic_conv1d(
        alpha1f,
        &mut out_buffer[..hidden_dim],
        &mut state.conv_alpha1f_state,
        &in_buffer[..feature_dim],
        feature_dim,
        Activation::Linear,
    );
    compute_generic_conv1d(
        alpha1t,
        &mut tmp_buffer[..hidden_dim],
        &mut state.conv_alpha1t_state,
        &in_buffer[feature_dim..feature_dim + tenv_size + 1],
        tenv_size + 1,
        Activation::Linear,
    );

    // Leaky ReLU: max(x, 0.2*x). Reuses in_buffer (features+tenv data already consumed).
    for i in 0..hidden_dim {
        let tmp = out_buffer[i] + tmp_buffer[i];
        in_buffer[i] = if tmp >= 0.0 { tmp } else { 0.2 * tmp };
    }

    compute_generic_conv1d(
        alpha2,
        &mut tmp_buffer[..hidden_dim],
        &mut state.conv_alpha2_state,
        &in_buffer[..hidden_dim],
        hidden_dim,
        Activation::Linear,
    );

    // Upsample by linear interpolation
    for i in 0..hidden_dim {
        for k in 0..interpolate_k {
            let alpha = (k + 1) as f32 / interpolate_k as f32;
            out_buffer[i * interpolate_k + k] =
                alpha * tmp_buffer[i] + (1.0 - alpha) * state.interpolate_state[0];
        }
        state.interpolate_state[0] = tmp_buffer[i];
    }

    // Apply exp activation and shape signal
    compute_activation(&mut out_buffer[..frame_size], Activation::Exp);
    for i in 0..frame_size {
        x_out[i] = out_buffer[i] * x_in[i];
    }
}
