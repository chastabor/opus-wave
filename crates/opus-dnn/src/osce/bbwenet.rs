use crate::nndsp::adaconv::adaconv_process_frame;
use crate::nndsp::adashape::adashape_process_frame;
use crate::nndsp::{AdaConvState, AdaShapeState, compute_overlap_window};
use crate::nnet::ops::{compute_generic_conv1d, compute_generic_dense, compute_generic_gru};
use crate::nnet::weights::{WeightError, linear_init};
use crate::nnet::{Activation, LinearLayer, WeightArray};

use super::config::*;

// BBWENet constants from bbwenet_data.h.
const BBWENET_FEATURE_DIM: usize = 114;
const BBWENET_FRAME_SIZE16: usize = 80;
const BBWENET_COND_DIM: usize = 128;

// Adaptive filter constants.
const AF1_FRAME_SIZE: usize = 80;
const AF1_OVERLAP_SIZE: usize = 40;
const AF1_IN_CH: usize = 1;
const AF1_OUT_CH: usize = 3;
const AF1_KERNEL_SIZE: usize = 16;
const AF2_FRAME_SIZE: usize = 160;
const AF2_OVERLAP_SIZE: usize = 80;
const AF2_IN_CH: usize = 3;
const AF2_OUT_CH: usize = 3;
const AF2_KERNEL_SIZE: usize = 32;
const AF3_FRAME_SIZE: usize = 240;
const AF3_OVERLAP_SIZE: usize = 120;
const AF3_IN_CH: usize = 3;
const AF3_OUT_CH: usize = 1;
const AF3_KERNEL_SIZE: usize = 16;

const AF_FILTER_GAIN_A: f32 = 1.381551;
const AF_FILTER_GAIN_B: f32 = 0.0;

// TDShape constants.
const TDSHAPE1_FRAME_SIZE: usize = 160;
const TDSHAPE1_AVG_POOL_K: usize = 8;
const TDSHAPE1_INTERP_K: usize = 2;
const TDSHAPE2_FRAME_SIZE: usize = 240;
const TDSHAPE2_AVG_POOL_K: usize = 12;
const TDSHAPE2_INTERP_K: usize = 2;

// Resampler coefficients from osce.c.
const HQ_2X_EVEN: [f32; 3] = [0.026641845703125, 0.228668212890625, -0.4036407470703125];
const HQ_2X_ODD: [f32; 3] = [0.104583740234375, 0.3932037353515625, -0.152496337890625];
const FRAC_01_24: [f32; 8] = [
    0.00576782,
    -0.01831055,
    0.01882935,
    0.9328308,
    0.09143066,
    -0.04196167,
    0.01296997,
    -0.00140381,
];
const FRAC_17_24: [f32; 8] = [
    -3.14331055e-03,
    2.73437500e-02,
    -1.06414795e-01,
    3.64685059e-01,
    8.03863525e-01,
    -1.02233887e-01,
    1.61437988e-02,
    -1.22070312e-04,
];
const FRAC_09_24: [f32; 8] = [
    -0.00146484,
    0.02313232,
    -0.12072754,
    0.7315979,
    0.4621277,
    -0.12075806,
    0.0295105,
    -0.00326538,
];

const DELAY_SAMPLES: usize = 8;

/// BBWENet model layers (17 layers). Matches C `BBWENETLayers` from bbwenet_data.h.
pub struct BbwenetLayers {
    pub fnet_conv1: LinearLayer,
    pub fnet_conv2: LinearLayer,
    pub fnet_gru_input: LinearLayer,
    pub fnet_gru_recurrent: LinearLayer,
    pub fnet_tconv: LinearLayer,
    pub tdshape1_alpha1_f: LinearLayer,
    pub tdshape1_alpha1_t: LinearLayer,
    pub tdshape1_alpha2: LinearLayer,
    pub tdshape2_alpha1_f: LinearLayer,
    pub tdshape2_alpha1_t: LinearLayer,
    pub tdshape2_alpha2: LinearLayer,
    pub af1_kernel: LinearLayer,
    pub af1_gain: LinearLayer,
    pub af2_kernel: LinearLayer,
    pub af2_gain: LinearLayer,
    pub af3_kernel: LinearLayer,
    pub af3_gain: LinearLayer,
}

/// BBWENet model with layers + overlap windows.
pub struct Bbwenet {
    pub layers: BbwenetLayers,
    pub window16: [f32; AF1_OVERLAP_SIZE],
    pub window32: [f32; AF2_OVERLAP_SIZE],
    pub window48: [f32; AF3_OVERLAP_SIZE],
}

/// Per-channel resampler state.
#[derive(Default)]
pub struct ResampState {
    pub upsamp_buffer: [[f32; 3]; 2],
    pub interpol_buffer: [f32; DELAY_SAMPLES],
}

/// BBWENet processing state.
pub struct BbwenetState {
    pub fnet_conv1_state: Vec<f32>,
    pub fnet_conv2_state: Vec<f32>,
    pub gru_state: Vec<f32>,
    pub output_buffer: [i16; OSCE_BWE_OUTPUT_DELAY],
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub resampler_state: [ResampState; 3],
}

/// Initialize BBWENet model from weight arrays.
pub fn init_bbwenet(arrays: &[WeightArray]) -> Result<Bbwenet, WeightError> {
    // Float-only layer: bias + float weights, no int8/scale.
    // Matches C: linear_init(_, bias, NULL, NULL, float, NULL, NULL, NULL, ni, no)
    let lf =
        |bias: &str, weights: &str, ni: usize, no: usize| -> Result<LinearLayer, WeightError> {
            linear_init(
                arrays,
                Some(bias),
                None,
                Some(weights),
                None,
                None,
                None,
                ni,
                no,
            )
        };
    // Int8 layer: bias + int8 weights (subias, float companion, and scale are auto-loaded).
    // Matches C: linear_init(_, bias, subias, int8, float, NULL, NULL, scale, ni, no)
    let li =
        |bias: &str, weights: &str, ni: usize, no: usize| -> Result<LinearLayer, WeightError> {
            linear_init(
                arrays,
                Some(bias),
                Some(weights),
                None,
                None,
                None,
                None,
                ni,
                no,
            )
        };

    let layers = BbwenetLayers {
        // C: linear_init(bbwenet_fnet_conv1, bias, NULL, NULL, float, NULL, NULL, NULL, 342, 128)
        fnet_conv1: lf(
            "bbwenet_fnet_conv1_bias",
            "bbwenet_fnet_conv1_weights",
            342,
            128,
        )?,
        // C: linear_init(bbwenet_fnet_conv2, bias, subias, int8, float, NULL, NULL, scale, 384, 128)
        fnet_conv2: li(
            "bbwenet_fnet_conv2_bias",
            "bbwenet_fnet_conv2_weights",
            384,
            128,
        )?,
        // C: linear_init(bbwenet_fnet_gru_input, bias, subias, int8, float, NULL, NULL, scale, 128, 384)
        fnet_gru_input: li(
            "bbwenet_fnet_gru_input_bias",
            "bbwenet_fnet_gru_input_weights",
            128,
            384,
        )?,
        // C: linear_init(bbwenet_fnet_gru_recurrent, bias, subias, int8, float, NULL, NULL, scale, 128, 384)
        fnet_gru_recurrent: li(
            "bbwenet_fnet_gru_recurrent_bias",
            "bbwenet_fnet_gru_recurrent_weights",
            128,
            384,
        )?,
        // C: linear_init(bbwenet_fnet_tconv, bias, subias, int8, float, NULL, NULL, scale, 128, 256)
        fnet_tconv: li(
            "bbwenet_fnet_tconv_bias",
            "bbwenet_fnet_tconv_weights",
            128,
            256,
        )?,
        // C: linear_init(bbwenet_tdshape1_alpha1_f, bias, subias, int8, float, NULL, NULL, scale, 256, 80)
        tdshape1_alpha1_f: li(
            "bbwenet_tdshape1_alpha1_f_bias",
            "bbwenet_tdshape1_alpha1_f_weights",
            256,
            80,
        )?,
        // C: linear_init(bbwenet_tdshape1_alpha1_t, bias, NULL, NULL, float, NULL, NULL, NULL, 42, 80)
        tdshape1_alpha1_t: lf(
            "bbwenet_tdshape1_alpha1_t_bias",
            "bbwenet_tdshape1_alpha1_t_weights",
            42,
            80,
        )?,
        // C: linear_init(bbwenet_tdshape1_alpha2, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 80)
        tdshape1_alpha2: lf(
            "bbwenet_tdshape1_alpha2_bias",
            "bbwenet_tdshape1_alpha2_weights",
            160,
            80,
        )?,
        // C: linear_init(bbwenet_tdshape2_alpha1_f, bias, subias, int8, float, NULL, NULL, scale, 256, 120)
        tdshape2_alpha1_f: li(
            "bbwenet_tdshape2_alpha1_f_bias",
            "bbwenet_tdshape2_alpha1_f_weights",
            256,
            120,
        )?,
        // C: linear_init(bbwenet_tdshape2_alpha1_t, bias, NULL, NULL, float, NULL, NULL, NULL, 42, 120)
        tdshape2_alpha1_t: lf(
            "bbwenet_tdshape2_alpha1_t_bias",
            "bbwenet_tdshape2_alpha1_t_weights",
            42,
            120,
        )?,
        // C: linear_init(bbwenet_tdshape2_alpha2, bias, NULL, NULL, float, NULL, NULL, NULL, 240, 120)
        tdshape2_alpha2: lf(
            "bbwenet_tdshape2_alpha2_bias",
            "bbwenet_tdshape2_alpha2_weights",
            240,
            120,
        )?,
        // C: linear_init(bbwenet_af1_kernel, bias, subias, int8, float, NULL, NULL, scale, 128, 48)
        af1_kernel: li(
            "bbwenet_af1_kernel_bias",
            "bbwenet_af1_kernel_weights",
            128,
            48,
        )?,
        // C: linear_init(bbwenet_af1_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 128, 3)
        af1_gain: lf("bbwenet_af1_gain_bias", "bbwenet_af1_gain_weights", 128, 3)?,
        // C: linear_init(bbwenet_af2_kernel, bias, subias, int8, float, NULL, NULL, scale, 128, 288)
        af2_kernel: li(
            "bbwenet_af2_kernel_bias",
            "bbwenet_af2_kernel_weights",
            128,
            288,
        )?,
        // C: linear_init(bbwenet_af2_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 128, 3)
        af2_gain: lf("bbwenet_af2_gain_bias", "bbwenet_af2_gain_weights", 128, 3)?,
        // C: linear_init(bbwenet_af3_kernel, bias, subias, int8, float, NULL, NULL, scale, 128, 48)
        af3_kernel: li(
            "bbwenet_af3_kernel_bias",
            "bbwenet_af3_kernel_weights",
            128,
            48,
        )?,
        // C: linear_init(bbwenet_af3_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 128, 1)
        af3_gain: lf("bbwenet_af3_gain_bias", "bbwenet_af3_gain_weights", 128, 1)?,
    };

    let mut window16 = [0.0f32; AF1_OVERLAP_SIZE];
    let mut window32 = [0.0f32; AF2_OVERLAP_SIZE];
    let mut window48 = [0.0f32; AF3_OVERLAP_SIZE];
    compute_overlap_window(&mut window16, AF1_OVERLAP_SIZE);
    compute_overlap_window(&mut window32, AF2_OVERLAP_SIZE);
    compute_overlap_window(&mut window48, AF3_OVERLAP_SIZE);

    Ok(Bbwenet {
        layers,
        window16,
        window32,
        window48,
    })
}

/// Create BBWENet processing state.
pub fn bbwenet_state_init(model: &Bbwenet) -> BbwenetState {
    BbwenetState {
        fnet_conv1_state: vec![0.0; model.layers.fnet_conv1.nb_inputs],
        fnet_conv2_state: vec![0.0; model.layers.fnet_conv2.nb_inputs],
        gru_state: vec![0.0; BBWENET_COND_DIM],
        output_buffer: [0; OSCE_BWE_OUTPUT_DELAY],
        af1_state: AdaConvState::default(),
        af2_state: AdaConvState::default(),
        af3_state: AdaConvState::default(),
        tdshape1_state: AdaShapeState::default(),
        tdshape2_state: AdaShapeState::default(),
        resampler_state: [
            ResampState::default(),
            ResampState::default(),
            ResampState::default(),
        ],
    }
}

/// 2x upsampling using allpass halfband filter. Matches C `upsamp_2x`.
fn upsamp_2x(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32], num_samples: usize) {
    let [ref mut s_even, ref mut s_odd] = state.upsamp_buffer;

    for k in 0..num_samples {
        let x = x_in[k];
        // Even sample
        let mut y = x - s_even[0];
        let mut xi = y * HQ_2X_EVEN[0];
        let tmp1 = s_even[0] + xi;
        s_even[0] = x + xi;

        y = tmp1 - s_even[1];
        xi = y * HQ_2X_EVEN[1];
        let tmp2 = s_even[1] + xi;
        s_even[1] = tmp1 + xi;

        y = tmp2 - s_even[2];
        xi = y * (1.0 + HQ_2X_EVEN[2]);
        let tmp3 = s_even[2] + xi;
        s_even[2] = tmp2 + xi;
        x_out[2 * k] = tmp3;

        // Odd sample
        y = x - s_odd[0];
        xi = y * HQ_2X_ODD[0];
        let tmp1 = s_odd[0] + xi;
        s_odd[0] = x + xi;

        y = tmp1 - s_odd[1];
        xi = y * HQ_2X_ODD[1];
        let tmp2 = s_odd[1] + xi;
        s_odd[1] = tmp1 + xi;

        y = tmp2 - s_odd[2];
        xi = y * (1.0 + HQ_2X_ODD[2]);
        let tmp3 = s_odd[2] + xi;
        s_odd[2] = tmp2 + xi;
        x_out[2 * k + 1] = tmp3;
    }
}

/// 3/2 polyphase interpolation. Matches C `interpol_3_2`.
fn interpol_3_2(state: &mut ResampState, x_out: &mut [f32], x_in: &[f32], num_samples: usize) {
    let mut buffer = [0.0f32; 8 * BBWENET_FRAME_SIZE16 + DELAY_SAMPLES];
    buffer[..DELAY_SAMPLES].copy_from_slice(&state.interpol_buffer);
    buffer[DELAY_SAMPLES..DELAY_SAMPLES + num_samples].copy_from_slice(&x_in[..num_samples]);

    let mut i_out = 0;
    let mut i_sample = 0;
    while i_sample < num_samples {
        let b = &buffer[i_sample..];
        x_out[i_out] = b[0] * FRAC_01_24[0]
            + b[1] * FRAC_01_24[1]
            + b[2] * FRAC_01_24[2]
            + b[3] * FRAC_01_24[3]
            + b[4] * FRAC_01_24[4]
            + b[5] * FRAC_01_24[5]
            + b[6] * FRAC_01_24[6]
            + b[7] * FRAC_01_24[7];
        i_out += 1;
        x_out[i_out] = b[0] * FRAC_17_24[0]
            + b[1] * FRAC_17_24[1]
            + b[2] * FRAC_17_24[2]
            + b[3] * FRAC_17_24[3]
            + b[4] * FRAC_17_24[4]
            + b[5] * FRAC_17_24[5]
            + b[6] * FRAC_17_24[6]
            + b[7] * FRAC_17_24[7];
        i_out += 1;
        x_out[i_out] = b[1] * FRAC_09_24[0]
            + b[2] * FRAC_09_24[1]
            + b[3] * FRAC_09_24[2]
            + b[4] * FRAC_09_24[3]
            + b[5] * FRAC_09_24[4]
            + b[6] * FRAC_09_24[5]
            + b[7] * FRAC_09_24[6]
            + b[8] * FRAC_09_24[7];
        i_out += 1;
        i_sample += 2;
    }
    state
        .interpol_buffer
        .copy_from_slice(&buffer[num_samples..num_samples + DELAY_SAMPLES]);
}

/// Valin activation: x * sin(log(|x| + eps)). Matches C `apply_valin_activation`.
fn apply_valin_activation(x: &mut [f32]) {
    for sample in x.iter_mut() {
        let y = (sample.abs() + 1e-6).ln();
        *sample *= y.sin();
    }
}

/// BBWENet feature network. Matches C `bbwe_feature_net` from osce.c.
/// Pipeline: conv1 per-frame → conv2 per-frame → tconv per-frame (upsamples) → GRU per-subframe.
fn bbwe_feature_net(
    model: &Bbwenet,
    state: &mut BbwenetState,
    output: &mut [f32],
    features: &[f32],
    num_frames: usize,
) {
    let num_subframes = 2 * num_frames;
    let cd = BBWENET_COND_DIM;
    let conv1_dim = model.layers.fnet_conv1.nb_outputs;
    let conv2_dim = model.layers.fnet_conv2.nb_outputs;
    let mut input_buf = [0.0f32; 4 * BBWENET_COND_DIM];
    let mut output_buf = [0.0f32; 4 * BBWENET_COND_DIM];

    // Conv1: per-frame (not per-subframe — matches C loop over i_frame)
    for f in 0..num_frames {
        compute_generic_conv1d(
            &model.layers.fnet_conv1,
            &mut output_buf[f * conv1_dim..(f + 1) * conv1_dim],
            &mut state.fnet_conv1_state,
            &features[f * BBWENET_FEATURE_DIM..(f + 1) * BBWENET_FEATURE_DIM],
            BBWENET_FEATURE_DIM,
            Activation::Tanh,
        );
    }
    input_buf[..num_frames * conv1_dim].copy_from_slice(&output_buf[..num_frames * conv1_dim]);

    // Conv2: per-frame
    for f in 0..num_frames {
        compute_generic_conv1d(
            &model.layers.fnet_conv2,
            &mut output_buf[f * conv2_dim..(f + 1) * conv2_dim],
            &mut state.fnet_conv2_state,
            &input_buf[f * conv1_dim..(f + 1) * conv1_dim],
            conv1_dim,
            Activation::Tanh,
        );
    }
    input_buf[..num_frames * conv2_dim].copy_from_slice(&output_buf[..num_frames * conv2_dim]);

    // Tconv upsampling: per-frame, each frame produces 2 subframes (stride=2)
    for f in 0..num_frames {
        compute_generic_dense(
            &model.layers.fnet_tconv,
            &mut output_buf[f * 2 * cd..(f + 1) * 2 * cd],
            &input_buf[f * conv2_dim..(f + 1) * conv2_dim],
            Activation::Tanh,
        );
    }
    input_buf[..num_subframes * cd].copy_from_slice(&output_buf[..num_subframes * cd]);

    // GRU: per-subframe (runs at 2x frame rate after tconv upsampling)
    // Output is the GRU state (not tconv output), matching C line 952
    for sf in 0..num_subframes {
        compute_generic_gru(
            &model.layers.fnet_gru_input,
            &model.layers.fnet_gru_recurrent,
            &mut state.gru_state,
            &input_buf[sf * cd..(sf + 1) * cd],
        );
        output[sf * cd..(sf + 1) * cd].copy_from_slice(&state.gru_state);
    }
}

/// Process BBWENet frames: 16kHz → 48kHz bandwidth extension.
/// Matches C `bbwenet_process_frames` from osce.c.
pub fn bbwenet_process_frames(
    model: &Bbwenet,
    state: &mut BbwenetState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    num_frames: usize,
) {
    let num_subframes = 2 * num_frames;
    let mut latent = [0.0f32; 4 * BBWENET_COND_DIM];
    bbwe_feature_net(model, state, &mut latent, features, num_frames);

    // Working buffers for the 3-channel processing chain (stack, matching C).
    const BUF_SIZE: usize = AF1_OUT_CH * 4 * 3 * BBWENET_FRAME_SIZE16;
    let mut buf1 = [0.0f32; BUF_SIZE];
    let mut buf2 = [0.0f32; BUF_SIZE];

    // Stage 1: af1 (1→3 channels at 16kHz)
    for sf in 0..num_subframes {
        adaconv_process_frame(
            &mut state.af1_state,
            &mut buf1[sf * AF1_FRAME_SIZE * AF1_OUT_CH
                ..sf * AF1_FRAME_SIZE * AF1_OUT_CH + AF1_FRAME_SIZE * AF1_OUT_CH],
            &x_in[sf * AF1_FRAME_SIZE..(sf + 1) * AF1_FRAME_SIZE],
            &latent[sf * BBWENET_COND_DIM..(sf + 1) * BBWENET_COND_DIM],
            &model.layers.af1_kernel,
            &model.layers.af1_gain,
            BBWENET_COND_DIM,
            AF1_FRAME_SIZE,
            AF1_OVERLAP_SIZE,
            AF1_IN_CH,
            AF1_OUT_CH,
            AF1_KERNEL_SIZE,
            AF1_KERNEL_SIZE - 1,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window16,
        );
    }

    // Stage 2: 2x upsample per channel → tdshape1 on ch2 → valin activation on ch3
    for sf in 0..num_subframes {
        for ch in 0..3 {
            upsamp_2x(
                &mut state.resampler_state[ch],
                &mut buf2[sf * TDSHAPE1_FRAME_SIZE * AF1_OUT_CH + ch * TDSHAPE1_FRAME_SIZE..],
                &buf1[sf * AF1_FRAME_SIZE * AF1_OUT_CH + ch * AF1_FRAME_SIZE..],
                AF1_FRAME_SIZE,
            );
        }

        // tdshape1 on second channel
        let ch2_start = sf * TDSHAPE1_FRAME_SIZE * AF1_OUT_CH + TDSHAPE1_FRAME_SIZE;
        let mut ts_in = [0.0f32; TDSHAPE1_FRAME_SIZE];
        ts_in.copy_from_slice(&buf2[ch2_start..ch2_start + TDSHAPE1_FRAME_SIZE]);
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut buf2[ch2_start..ch2_start + TDSHAPE1_FRAME_SIZE],
            &ts_in,
            &latent[sf * BBWENET_COND_DIM..(sf + 1) * BBWENET_COND_DIM],
            &model.layers.tdshape1_alpha1_f,
            &model.layers.tdshape1_alpha1_t,
            &model.layers.tdshape1_alpha2,
            BBWENET_COND_DIM,
            TDSHAPE1_FRAME_SIZE,
            TDSHAPE1_AVG_POOL_K,
            TDSHAPE1_INTERP_K,
        );

        // Valin activation on third channel
        let ch3_start = sf * TDSHAPE1_FRAME_SIZE * AF1_OUT_CH + 2 * TDSHAPE1_FRAME_SIZE;
        apply_valin_activation(&mut buf2[ch3_start..ch3_start + TDSHAPE1_FRAME_SIZE]);
    }

    // Stage 3: af2 mixing (3→3 channels at 32kHz)
    for sf in 0..num_subframes {
        adaconv_process_frame(
            &mut state.af2_state,
            &mut buf1[sf * AF2_FRAME_SIZE * AF2_OUT_CH
                ..sf * AF2_FRAME_SIZE * AF2_OUT_CH + AF2_FRAME_SIZE * AF2_OUT_CH],
            &buf2[sf * AF2_FRAME_SIZE * AF2_IN_CH
                ..sf * AF2_FRAME_SIZE * AF2_IN_CH + AF2_FRAME_SIZE * AF2_IN_CH],
            &latent[sf * BBWENET_COND_DIM..(sf + 1) * BBWENET_COND_DIM],
            &model.layers.af2_kernel,
            &model.layers.af2_gain,
            BBWENET_COND_DIM,
            AF2_FRAME_SIZE,
            AF2_OVERLAP_SIZE,
            AF2_IN_CH,
            AF2_OUT_CH,
            AF2_KERNEL_SIZE,
            AF2_KERNEL_SIZE - 1,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window32,
        );
    }

    // Stage 4: 3/2 interpolation per channel → tdshape2 on ch2 → valin on ch3
    for sf in 0..num_subframes {
        for ch in 0..3 {
            interpol_3_2(
                &mut state.resampler_state[ch],
                &mut buf2[sf * AF3_FRAME_SIZE * AF2_OUT_CH + ch * TDSHAPE2_FRAME_SIZE..],
                &buf1[sf * TDSHAPE1_FRAME_SIZE * AF2_OUT_CH + ch * TDSHAPE1_FRAME_SIZE..],
                TDSHAPE1_FRAME_SIZE,
            );
        }

        let ch2_start = sf * TDSHAPE2_FRAME_SIZE * AF2_OUT_CH + TDSHAPE2_FRAME_SIZE;
        let mut ts_in = [0.0f32; TDSHAPE2_FRAME_SIZE];
        ts_in.copy_from_slice(&buf2[ch2_start..ch2_start + TDSHAPE2_FRAME_SIZE]);
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut buf2[ch2_start..ch2_start + TDSHAPE2_FRAME_SIZE],
            &ts_in,
            &latent[sf * BBWENET_COND_DIM..(sf + 1) * BBWENET_COND_DIM],
            &model.layers.tdshape2_alpha1_f,
            &model.layers.tdshape2_alpha1_t,
            &model.layers.tdshape2_alpha2,
            BBWENET_COND_DIM,
            TDSHAPE2_FRAME_SIZE,
            TDSHAPE2_AVG_POOL_K,
            TDSHAPE2_INTERP_K,
        );

        let ch3_start = sf * TDSHAPE2_FRAME_SIZE * AF2_OUT_CH + 2 * TDSHAPE2_FRAME_SIZE;
        apply_valin_activation(&mut buf2[ch3_start..ch3_start + TDSHAPE2_FRAME_SIZE]);
    }

    // Stage 5: af3 final mixing (3→1 channel at 48kHz)
    for sf in 0..num_subframes {
        adaconv_process_frame(
            &mut state.af3_state,
            &mut x_out[sf * AF3_FRAME_SIZE..(sf + 1) * AF3_FRAME_SIZE],
            &buf2[sf * TDSHAPE2_FRAME_SIZE * AF2_OUT_CH
                ..sf * TDSHAPE2_FRAME_SIZE * AF2_OUT_CH + AF3_FRAME_SIZE * AF3_IN_CH],
            &latent[sf * BBWENET_COND_DIM..(sf + 1) * BBWENET_COND_DIM],
            &model.layers.af3_kernel,
            &model.layers.af3_gain,
            BBWENET_COND_DIM,
            AF3_FRAME_SIZE,
            AF3_OVERLAP_SIZE,
            AF3_IN_CH,
            AF3_OUT_CH,
            AF3_KERNEL_SIZE,
            AF3_KERNEL_SIZE - 1,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window48,
        );
    }
}
