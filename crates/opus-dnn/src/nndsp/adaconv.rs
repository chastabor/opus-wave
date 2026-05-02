use super::*;
use crate::nnet::ops::compute_generic_dense;
use crate::nnet::{Activation, LinearLayer};

/// Normalize kernel and apply per-channel gain.
/// Matches C `scale_kernel` from nndsp.c.
fn scale_kernel(
    kernel: &mut [f32],
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    gain: &[f32],
) {
    for (oc, &g) in gain.iter().enumerate().take(out_channels) {
        let mut norm = 0.0f32;
        for ic in 0..in_channels {
            for k in 0..kernel_size {
                let idx = (oc * in_channels + ic) * kernel_size + k;
                norm += kernel[idx] * kernel[idx];
            }
        }
        norm = 1.0 / (1e-6 + norm.sqrt());
        for ic in 0..in_channels {
            for k in 0..kernel_size {
                let idx = (oc * in_channels + ic) * kernel_size + k;
                kernel[idx] *= norm * g;
            }
        }
    }
}

fn transform_gains(gains: &mut [f32], filter_gain_a: f32, filter_gain_b: f32) {
    for g in gains.iter_mut() {
        *g = (filter_gain_a * *g + filter_gain_b).exp();
    }
}

/// Process one frame through an adaptive convolution layer.
/// Matches C `adaconv_process_frame` from nndsp.c.
pub fn adaconv_process_frame(
    state: &mut AdaConvState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    kernel_layer: &LinearLayer,
    gain_layer: &LinearLayer,
    _feature_dim: usize,
    frame_size: usize,
    overlap_size: usize,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    left_padding: usize,
    filter_gain_a: f32,
    filter_gain_b: f32,
    _shape_gain: f32,
    window: &[f32],
) {
    debug_assert_eq!(left_padding, kernel_size - 1);

    let mut output_buffer = [0.0f32; ADACONV_MAX_FRAME_SIZE * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut kernel_buffer = [0.0f32;
        ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS];
    let mut input_buffer =
        [0.0f32; ADACONV_MAX_INPUT_CHANNELS * (ADACONV_MAX_FRAME_SIZE + ADACONV_MAX_KERNEL_SIZE)];
    let mut gain_buffer = [0.0f32; ADACONV_MAX_OUTPUT_CHANNELS];

    // Prepare input: [history | x_in] per channel
    for ic in 0..in_channels {
        let buf_start = ic * (kernel_size + frame_size);
        input_buffer[buf_start..buf_start + kernel_size]
            .copy_from_slice(&state.history[ic * kernel_size..(ic + 1) * kernel_size]);
        input_buffer[buf_start + kernel_size..buf_start + kernel_size + frame_size]
            .copy_from_slice(&x_in[frame_size * ic..frame_size * (ic + 1)]);
    }

    // Calculate new kernel and gain
    compute_generic_dense(
        kernel_layer,
        &mut kernel_buffer[..in_channels * out_channels * kernel_size],
        features,
        Activation::Linear,
    );
    compute_generic_dense(
        gain_layer,
        &mut gain_buffer[..out_channels],
        features,
        Activation::Tanh,
    );
    transform_gains(
        &mut gain_buffer[..out_channels],
        filter_gain_a,
        filter_gain_b,
    );
    scale_kernel(
        &mut kernel_buffer,
        in_channels,
        out_channels,
        kernel_size,
        &gain_buffer,
    );

    // Convolution with overlap blending between old and new kernels.
    // Uses celt_pitch_xcorr for batched correlation (matching C nndsp.c lines 215-216).
    for oc in 0..out_channels {
        for ic in 0..in_channels {
            let p_input_start = kernel_size + ic * (frame_size + kernel_size) - left_padding;
            let k_base = (oc * in_channels + ic) * kernel_size;

            // Pad kernels to ADACONV_MAX_KERNEL_SIZE for celt_pitch_xcorr
            let mut kernel0 = [0.0f32; ADACONV_MAX_KERNEL_SIZE];
            let mut kernel1 = [0.0f32; ADACONV_MAX_KERNEL_SIZE];
            kernel0[..kernel_size]
                .copy_from_slice(&state.last_kernel[k_base..k_base + kernel_size]);
            kernel1[..kernel_size].copy_from_slice(&kernel_buffer[k_base..k_base + kernel_size]);

            let mut channel_buffer0 = [0.0f32; ADACONV_MAX_OVERLAP_SIZE];
            let mut channel_buffer1 = [0.0f32; ADACONV_MAX_FRAME_SIZE];
            opus_celt::pitch::celt_pitch_xcorr(
                &kernel0,
                &input_buffer[p_input_start..],
                &mut channel_buffer0,
                ADACONV_MAX_KERNEL_SIZE,
                overlap_size,
            );
            opus_celt::pitch::celt_pitch_xcorr(
                &kernel1,
                &input_buffer[p_input_start..],
                &mut channel_buffer1,
                ADACONV_MAX_KERNEL_SIZE,
                frame_size,
            );

            for s in 0..overlap_size {
                output_buffer[s + oc * frame_size] +=
                    window[s] * channel_buffer0[s] + (1.0 - window[s]) * channel_buffer1[s];
            }
            for s in overlap_size..frame_size {
                output_buffer[s + oc * frame_size] += channel_buffer1[s];
            }
        }
    }

    x_out[..out_channels * frame_size].copy_from_slice(&output_buffer[..out_channels * frame_size]);

    // Update state
    for ic in 0..in_channels {
        let p_start = kernel_size + ic * (frame_size + kernel_size) + frame_size - kernel_size;
        state.history[ic * kernel_size..(ic + 1) * kernel_size]
            .copy_from_slice(&input_buffer[p_start..p_start + kernel_size]);
    }
    state.last_kernel[..kernel_size * in_channels * out_channels]
        .copy_from_slice(&kernel_buffer[..kernel_size * in_channels * out_channels]);
}
