use super::*;
use crate::nnet::ops::compute_generic_dense;
use crate::nnet::{Activation, LinearLayer};

fn scale_kernel_1d(kernel: &mut [f32], kernel_size: usize, gain: f32) {
    let mut norm = 0.0f32;
    for k in &kernel[..kernel_size] {
        norm += k * k;
    }
    norm = 1.0 / (1e-6 + norm.sqrt());
    for k in &mut kernel[..kernel_size] {
        *k *= norm * gain;
    }
}

/// Process one frame through an adaptive comb filter.
/// Matches C `adacomb_process_frame` from nndsp.c.
pub fn adacomb_process_frame(
    state: &mut AdaCombState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    global_gain_layer: &LinearLayer,
    pitch_lag: usize,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    log_gain_limit: f32,
    window: &[f32],
) {
    let mut output_buffer = [0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut output_buffer_last = [0.0f32; ADACOMB_MAX_FRAME_SIZE];
    let mut kernel_buffer = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut input_buffer =
        [0.0f32; ADACOMB_MAX_FRAME_SIZE + ADACOMB_MAX_LAG + ADACOMB_MAX_KERNEL_SIZE];

    // Prepare input: [history | x_in]
    let hist_len = kernel_size + ADACOMB_MAX_LAG;
    input_buffer[..hist_len].copy_from_slice(&state.history[..hist_len]);
    input_buffer[hist_len..hist_len + frame_size].copy_from_slice(&x_in[..frame_size]);
    let p_input_off = kernel_size + ADACOMB_MAX_LAG; // offset into input_buffer for current frame

    // Calculate kernel and gains
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer[..kernel_size],
        features,
        Activation::Linear,
    );
    let mut gain = [0.0f32; 1];
    compute_generic_dense(gain_layer, &mut gain, features, Activation::Relu);
    let mut global_gain = [0.0f32; 1];
    compute_generic_dense(
        global_gain_layer,
        &mut global_gain,
        features,
        Activation::Tanh,
    );

    let gain = (log_gain_limit - gain[0]).exp();
    let global_gain = (filter_gain_a * global_gain[0] + filter_gain_b).exp();
    scale_kernel_1d(&mut kernel_buffer, kernel_size, gain);

    // Pad kernels to ADACOMB_MAX_KERNEL_SIZE for celt_pitch_xcorr
    let mut kernel = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    let mut last_kernel = [0.0f32; ADACOMB_MAX_KERNEL_SIZE];
    kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    last_kernel[..kernel_size].copy_from_slice(&state.last_kernel[..kernel_size]);

    // Buffer layout guarantees: p_input_off - left_padding - max_lag >= 0
    debug_assert!(p_input_off >= left_padding + state.last_pitch_lag);
    debug_assert!(p_input_off >= left_padding + pitch_lag);

    let last_start = p_input_off - left_padding - state.last_pitch_lag;
    let new_start = p_input_off - left_padding - pitch_lag;

    opus_celt::pitch::celt_pitch_xcorr(
        &last_kernel,
        &input_buffer[last_start..],
        &mut output_buffer_last,
        ADACOMB_MAX_KERNEL_SIZE,
        overlap_size,
    );
    opus_celt::pitch::celt_pitch_xcorr(
        &kernel,
        &input_buffer[new_start..],
        &mut output_buffer,
        ADACOMB_MAX_KERNEL_SIZE,
        frame_size,
    );

    // Overlap blend
    for s in 0..overlap_size {
        output_buffer[s] = state.last_global_gain * window[s] * output_buffer_last[s]
            + global_gain * (1.0 - window[s]) * output_buffer[s];
    }
    // Add input signal with gain blending
    for s in 0..overlap_size {
        output_buffer[s] += (window[s] * state.last_global_gain + (1.0 - window[s]) * global_gain)
            * input_buffer[p_input_off + s];
    }
    for s in overlap_size..frame_size {
        output_buffer[s] = global_gain * (output_buffer[s] + input_buffer[p_input_off + s]);
    }

    x_out[..frame_size].copy_from_slice(&output_buffer[..frame_size]);

    // Update state
    state.last_kernel[..kernel_size].copy_from_slice(&kernel_buffer[..kernel_size]);
    let hist_start = p_input_off + frame_size - kernel_size - ADACOMB_MAX_LAG;
    state.history[..hist_len].copy_from_slice(&input_buffer[hist_start..hist_start + hist_len]);
    state.last_pitch_lag = pitch_lag;
    state.last_global_gain = global_gain;
}
