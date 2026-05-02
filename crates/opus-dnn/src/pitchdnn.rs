use crate::nnet::WeightArray;
use crate::nnet::conv2d::compute_conv2d;
use crate::nnet::ops::{compute_generic_dense, compute_generic_gru};
use crate::nnet::weights::{WeightError, conv2d_init, linear_init, weight_output_dim};
use crate::nnet::{Activation, Conv2dLayer, LinearLayer};

pub const PITCH_MIN_PERIOD: usize = 32;
pub const PITCH_MAX_PERIOD: usize = 256;
pub const NB_XCORR_FEATURES: usize = PITCH_MAX_PERIOD - PITCH_MIN_PERIOD;

pub const PITCH_IF_MAX_FREQ: usize = 30;
pub const NB_IF_FEATURES: usize = 3 * PITCH_IF_MAX_FREQ - 2;

/// Convert a log-scale DNN pitch value to an integer period in samples.
/// Used by FARGAN and LPCNet encoder. Inverse of the pitch representation
/// that PitchDNN produces.
pub fn pitch_period_from_dnn(dnn_pitch: f32) -> usize {
    (0.5 + 256.0 / 2.0f32.powf((1.0 / 60.0) * ((dnn_pitch + 1.5) * 60.0))).floor() as usize
}

/// Upper bounds on PitchDNN layer sizes for stack allocation.
const MAX_IF_UP1_SIZE: usize = 256;
const MAX_IF_UP2_SIZE: usize = 256;
const MAX_DS_OUT_SIZE: usize = 256;
const MAX_FINAL_OUT_SIZE: usize = 192;
const MAX_CONV_CHANNELS: usize = 16;

/// PitchDNN model: neural network layers for pitch estimation.
/// Matches the auto-generated C `struct PitchDNN` from pitchdnn_data.h.
pub struct PitchDnn {
    pub dense_if_upsampler_1: LinearLayer,
    pub dense_if_upsampler_2: LinearLayer,
    pub conv2d_1: Conv2dLayer,
    pub conv2d_2: Conv2dLayer,
    pub dense_downsampler: LinearLayer,
    pub gru_1_input: LinearLayer,
    pub gru_1_recurrent: LinearLayer,
    pub dense_final_upsampler: LinearLayer,
}

/// PitchDNN state: model plus persistent recurrent state.
/// Matches C `PitchDNNState` from pitchdnn.h.
pub struct PitchDnnState {
    pub model: PitchDnn,
    pub gru_state: Vec<f32>,
    pub xcorr_mem1: Vec<f32>,
    pub xcorr_mem2: Vec<f32>,
}

/// Initialize PitchDNN model from weight arrays.
/// Matches the auto-generated C `init_pitchdnn`.
pub fn init_pitchdnn(arrays: &[WeightArray]) -> Result<PitchDnn, WeightError> {
    let if_up1_out = weight_output_dim(arrays, "dense_if_upsampler_1_bias")?;
    let if_up2_out = weight_output_dim(arrays, "dense_if_upsampler_2_bias")?;
    let conv1_out = weight_output_dim(arrays, "conv2d_1_bias")?;
    let conv2_out = weight_output_dim(arrays, "conv2d_2_bias")?;
    let ds_out = weight_output_dim(arrays, "dense_downsampler_bias")?;
    let gru_3n = weight_output_dim(arrays, "gru_1_input_bias")?;

    Ok(PitchDnn {
        dense_if_upsampler_1: linear_init(
            arrays,
            Some("dense_if_upsampler_1_bias"),
            Some("dense_if_upsampler_1_weights"),
            Some("dense_if_upsampler_1_weights"),
            None,
            None,
            Some("dense_if_upsampler_1_scale"),
            NB_IF_FEATURES,
            if_up1_out,
        )?,
        dense_if_upsampler_2: linear_init(
            arrays,
            Some("dense_if_upsampler_2_bias"),
            Some("dense_if_upsampler_2_weights"),
            Some("dense_if_upsampler_2_weights"),
            None,
            None,
            Some("dense_if_upsampler_2_scale"),
            if_up1_out,
            if_up2_out,
        )?,
        conv2d_1: conv2d_init(
            arrays,
            Some("conv2d_1_bias"),
            Some("conv2d_1_weight"),
            1,
            conv1_out,
            3,
            3,
        )?,
        conv2d_2: conv2d_init(
            arrays,
            Some("conv2d_2_bias"),
            Some("conv2d_2_weight"),
            conv1_out,
            conv2_out,
            3,
            3,
        )?,
        dense_downsampler: linear_init(
            arrays,
            Some("dense_downsampler_bias"),
            Some("dense_downsampler_weights"),
            Some("dense_downsampler_weights"),
            None,
            None,
            Some("dense_downsampler_scale"),
            NB_XCORR_FEATURES + if_up2_out,
            ds_out,
        )?,
        gru_1_input: linear_init(
            arrays,
            Some("gru_1_input_bias"),
            Some("gru_1_input_weights"),
            Some("gru_1_input_weights"),
            None,
            None,
            Some("gru_1_input_scale"),
            ds_out,
            gru_3n,
        )?,
        gru_1_recurrent: linear_init(
            arrays,
            Some("gru_1_recurrent_bias"),
            Some("gru_1_recurrent_weights"),
            Some("gru_1_recurrent_weights"),
            None,
            None,
            Some("gru_1_recurrent_scale"),
            gru_3n / 3,
            gru_3n,
        )?,
        dense_final_upsampler: linear_init(
            arrays,
            Some("dense_final_upsampler_bias"),
            Some("dense_final_upsampler_weights"),
            Some("dense_final_upsampler_weights"),
            None,
            None,
            Some("dense_final_upsampler_scale"),
            gru_3n / 3,
            weight_output_dim(arrays, "dense_final_upsampler_bias")?,
        )?,
    })
}

/// Initialize PitchDNN state with model.
pub fn pitchdnn_state_init(model: PitchDnn) -> PitchDnnState {
    let gru_size = model.gru_1_recurrent.nb_inputs;
    let conv2d_1_out = model.conv2d_1.out_channels;
    PitchDnnState {
        model,
        gru_state: vec![0.0; gru_size],
        xcorr_mem1: vec![0.0; (NB_XCORR_FEATURES + 2) * 2],
        xcorr_mem2: vec![0.0; (NB_XCORR_FEATURES + 2) * 2 * conv2d_1_out],
    }
}

/// Compute pitch estimate from instantaneous frequency and cross-correlation features.
/// Returns a normalized pitch value (log-scale). Matches C `compute_pitchdnn`.
pub fn compute_pitchdnn(
    st: &mut PitchDnnState,
    if_features: &[f32],
    xcorr_features: &[f32],
) -> f32 {
    let model = &st.model;
    let if_up1_size = model.dense_if_upsampler_1.nb_outputs;
    let if_up2_size = model.dense_if_upsampler_2.nb_outputs;
    let ds_out_size = model.dense_downsampler.nb_outputs;
    let final_out_size = model.dense_final_upsampler.nb_outputs;
    let conv1_out_ch = model.conv2d_1.out_channels;

    debug_assert!(if_up1_size <= MAX_IF_UP1_SIZE);
    debug_assert!(if_up2_size <= MAX_IF_UP2_SIZE);
    debug_assert!(ds_out_size <= MAX_DS_OUT_SIZE);
    debug_assert!(final_out_size <= MAX_FINAL_OUT_SIZE);
    debug_assert!(conv1_out_ch <= MAX_CONV_CHANNELS);

    let mut if1_out = [0.0f32; MAX_IF_UP1_SIZE];
    compute_generic_dense(
        &model.dense_if_upsampler_1,
        &mut if1_out[..if_up1_size],
        if_features,
        Activation::Tanh,
    );

    let ds_in_size = NB_XCORR_FEATURES + if_up2_size;
    let mut downsampler_in = [0.0f32; NB_XCORR_FEATURES + MAX_IF_UP2_SIZE];
    compute_generic_dense(
        &model.dense_if_upsampler_2,
        &mut downsampler_in[NB_XCORR_FEATURES..NB_XCORR_FEATURES + if_up2_size],
        &if1_out[..if_up1_size],
        Activation::Tanh,
    );

    let conv_stride = NB_XCORR_FEATURES + 2;
    let mut conv1_tmp1 = [0.0f32; (NB_XCORR_FEATURES + 2) * MAX_CONV_CHANNELS];
    conv1_tmp1[1..1 + NB_XCORR_FEATURES].copy_from_slice(&xcorr_features[..NB_XCORR_FEATURES]);

    let mut conv1_tmp2 = [0.0f32; (NB_XCORR_FEATURES + 2) * MAX_CONV_CHANNELS];
    compute_conv2d(
        &model.conv2d_1,
        &mut conv1_tmp2[1..],
        &mut st.xcorr_mem1,
        &conv1_tmp1,
        NB_XCORR_FEATURES,
        conv_stride,
        Activation::Tanh,
    );

    compute_conv2d(
        &model.conv2d_2,
        &mut downsampler_in[..NB_XCORR_FEATURES],
        &mut st.xcorr_mem2,
        &conv1_tmp2,
        NB_XCORR_FEATURES,
        NB_XCORR_FEATURES,
        Activation::Tanh,
    );

    let mut downsampler_out = [0.0f32; MAX_DS_OUT_SIZE];
    compute_generic_dense(
        &model.dense_downsampler,
        &mut downsampler_out[..ds_out_size],
        &downsampler_in[..ds_in_size],
        Activation::Tanh,
    );

    compute_generic_gru(
        &model.gru_1_input,
        &model.gru_1_recurrent,
        &mut st.gru_state,
        &downsampler_out[..ds_out_size],
    );

    let mut output = [0.0f32; MAX_FINAL_OUT_SIZE];
    compute_generic_dense(
        &model.dense_final_upsampler,
        &mut output[..final_out_size],
        &st.gru_state,
        Activation::Linear,
    );

    // Find peak and compute weighted average position
    let mut pos = 0;
    let mut maxval = -1.0f32;
    let search_len = 180.min(final_out_size);
    for (i, &val) in output.iter().enumerate().take(search_len) {
        if val > maxval {
            pos = i;
            maxval = val;
        }
    }

    let start = pos.saturating_sub(2);
    let end = (pos + 2).min(179.min(final_out_size - 1));
    let mut sum = 0.0f32;
    let mut count = 0.0f32;
    for (i, &val) in output.iter().enumerate().take(end + 1).skip(start) {
        let p = val.exp();
        sum += p * i as f32;
        count += p;
    }

    (1.0 / 60.0) * (sum / count) - 1.5
}
