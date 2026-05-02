use crate::nndsp::adacomb::adacomb_process_frame;
use crate::nndsp::adaconv::adaconv_process_frame;
use crate::nndsp::{AdaCombState, AdaConvState, compute_overlap_window};
use crate::nnet::weights::{WeightError, linear_init, weight_output_dim};
use crate::nnet::{LinearLayer, WeightArray};

use super::common::*;
use super::config::*;

pub const LACE_FRAME_SIZE: usize = 80;
pub const LACE_OVERLAP_SIZE: usize = 40;

// Filter gain constants from lace_data.h (verified against downloaded weights).
const CF_FILTER_GAIN_A: f32 = 0.690776;
const CF_FILTER_GAIN_B: f32 = 0.0;
const CF_LOG_GAIN_LIMIT: f32 = 1.151293;
const AF_FILTER_GAIN_A: f32 = 1.381551;
const AF_FILTER_GAIN_B: f32 = 0.0;

/// LACE model layers.
pub struct LaceLayers {
    pub pitch_embedding: LinearLayer,
    pub fnet_conv1: LinearLayer,
    pub fnet_conv2: LinearLayer,
    pub fnet_tconv: LinearLayer,
    pub fnet_gru_input: LinearLayer,
    pub fnet_gru_recurrent: LinearLayer,
    pub cf1_kernel: LinearLayer,
    pub cf1_gain: LinearLayer,
    pub cf1_global_gain: LinearLayer,
    pub cf2_kernel: LinearLayer,
    pub cf2_gain: LinearLayer,
    pub cf2_global_gain: LinearLayer,
    pub af1_kernel: LinearLayer,
    pub af1_gain: LinearLayer,
}

impl FeatureNetLayers for LaceLayers {
    fn pitch_embedding(&self) -> &LinearLayer {
        &self.pitch_embedding
    }
    fn fnet_conv1(&self) -> &LinearLayer {
        &self.fnet_conv1
    }
    fn fnet_conv2(&self) -> &LinearLayer {
        &self.fnet_conv2
    }
    fn fnet_tconv(&self) -> &LinearLayer {
        &self.fnet_tconv
    }
    fn fnet_gru_input(&self) -> &LinearLayer {
        &self.fnet_gru_input
    }
    fn fnet_gru_recurrent(&self) -> &LinearLayer {
        &self.fnet_gru_recurrent
    }
}

/// LACE model with layers + overlap window + params.
pub struct Lace {
    pub layers: LaceLayers,
    pub window: [f32; LACE_OVERLAP_SIZE],
    pub params: FeatureNetParams,
    pub cf1_kernel_size: usize,
    pub cf2_kernel_size: usize,
    pub af1_kernel_size: usize,
    pub af1_in_channels: usize,
    pub af1_out_channels: usize,
}

/// LACE processing state.
pub struct LaceState {
    pub fnet_conv2_state: Vec<f32>,
    pub gru_state: Vec<f32>,
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl FeatureNetState for LaceState {
    fn fnet_conv2_state(&mut self) -> &mut [f32] {
        &mut self.fnet_conv2_state
    }
    fn gru_state(&mut self) -> &mut [f32] {
        &mut self.gru_state
    }
}

/// Initialize LACE model from weight arrays.
pub fn init_lace(arrays: &[WeightArray]) -> Result<Lace, WeightError> {
    let dim = |name: &str| weight_output_dim(arrays, name);
    let cond_dim = dim("lace_fnet_gru_input_bias")? / 3;
    let hidden_dim = dim("lace_fnet_conv1_bias")?;
    let pitch_embed_dim = dim("lace_pitch_embedding_bias")?;
    let numbits_embed_dim = 8;
    let feat_in = OSCE_FEATURE_DIM + pitch_embed_dim + 2 * numbits_embed_dim;
    // Conv2 uses kernel_size=2 over 4*hidden_dim frames.
    let conv2_out = dim("lace_fnet_conv2_bias")?;

    let cf1_kernel_size = dim("lace_cf1_kernel_bias")?;
    let cf2_kernel_size = dim("lace_cf2_kernel_bias")?;
    let af1_kernel_size = dim("lace_af1_kernel_bias")?;

    // Dimensions match C `init_lacelayers` from lace_data.c.
    let pitch_max = 300;
    let layers = LaceLayers {
        pitch_embedding: linear_init(
            arrays,
            Some("lace_pitch_embedding_bias"),
            None,
            Some("lace_pitch_embedding_weights"),
            None,
            None,
            None,
            pitch_max + 1,
            pitch_embed_dim,
        )?,
        fnet_conv1: linear_init(
            arrays,
            Some("lace_fnet_conv1_bias"),
            None,
            Some("lace_fnet_conv1_weights"),
            None,
            None,
            None,
            feat_in,
            hidden_dim,
        )?,
        fnet_conv2: linear_init(
            arrays,
            Some("lace_fnet_conv2_bias"),
            Some("lace_fnet_conv2_weights"),
            Some("lace_fnet_conv2_weights"),
            None,
            None,
            Some("lace_fnet_conv2_scale"),
            2 * 4 * hidden_dim,
            conv2_out,
        )?,
        fnet_tconv: linear_init(
            arrays,
            Some("lace_fnet_tconv_bias"),
            Some("lace_fnet_tconv_weights"),
            Some("lace_fnet_tconv_weights"),
            None,
            None,
            Some("lace_fnet_tconv_scale"),
            conv2_out,
            4 * cond_dim,
        )?,
        fnet_gru_input: linear_init(
            arrays,
            Some("lace_fnet_gru_input_bias"),
            Some("lace_fnet_gru_input_weights"),
            Some("lace_fnet_gru_input_weights"),
            None,
            None,
            Some("lace_fnet_gru_input_scale"),
            cond_dim,
            3 * cond_dim,
        )?,
        fnet_gru_recurrent: linear_init(
            arrays,
            Some("lace_fnet_gru_recurrent_bias"),
            Some("lace_fnet_gru_recurrent_weights"),
            Some("lace_fnet_gru_recurrent_weights"),
            None,
            None,
            Some("lace_fnet_gru_recurrent_scale"),
            cond_dim,
            3 * cond_dim,
        )?,
        cf1_kernel: linear_init(
            arrays,
            Some("lace_cf1_kernel_bias"),
            Some("lace_cf1_kernel_weights"),
            Some("lace_cf1_kernel_weights"),
            None,
            None,
            Some("lace_cf1_kernel_scale"),
            cond_dim,
            cf1_kernel_size,
        )?,
        cf1_gain: linear_init(
            arrays,
            Some("lace_cf1_gain_bias"),
            None,
            Some("lace_cf1_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        cf1_global_gain: linear_init(
            arrays,
            Some("lace_cf1_global_gain_bias"),
            None,
            Some("lace_cf1_global_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        cf2_kernel: linear_init(
            arrays,
            Some("lace_cf2_kernel_bias"),
            Some("lace_cf2_kernel_weights"),
            Some("lace_cf2_kernel_weights"),
            None,
            None,
            Some("lace_cf2_kernel_scale"),
            cond_dim,
            cf2_kernel_size,
        )?,
        cf2_gain: linear_init(
            arrays,
            Some("lace_cf2_gain_bias"),
            None,
            Some("lace_cf2_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        cf2_global_gain: linear_init(
            arrays,
            Some("lace_cf2_global_gain_bias"),
            None,
            Some("lace_cf2_global_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        af1_kernel: linear_init(
            arrays,
            Some("lace_af1_kernel_bias"),
            Some("lace_af1_kernel_weights"),
            Some("lace_af1_kernel_weights"),
            None,
            None,
            Some("lace_af1_kernel_scale"),
            cond_dim,
            af1_kernel_size,
        )?,
        af1_gain: linear_init(
            arrays,
            Some("lace_af1_gain_bias"),
            None,
            Some("lace_af1_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
    };

    let mut window = [0.0f32; LACE_OVERLAP_SIZE];
    compute_overlap_window(&mut window, LACE_OVERLAP_SIZE);

    Ok(Lace {
        layers,
        window,
        params: FeatureNetParams {
            cond_dim,
            hidden_dim,
            pitch_embed_dim,
            numbits_embed_dim,
            pitch_max,
            numbits_range_low: 50.0,
            numbits_range_high: 650.0,
            numbits_scales: [
                1.0983515, 2.0509143, 3.5729940, 4.4780359, 5.9265194, 7.1522822, 8.2774124,
                8.9268303,
            ],
        },
        cf1_kernel_size,
        cf2_kernel_size,
        af1_kernel_size,
        af1_in_channels: 1,
        af1_out_channels: 1,
    })
}

pub fn lace_state_init(model: &Lace) -> LaceState {
    LaceState {
        fnet_conv2_state: vec![0.0; model.layers.fnet_conv2.nb_inputs],
        gru_state: vec![0.0; model.params.cond_dim],
        cf1_state: AdaCombState::default(),
        cf2_state: AdaCombState::default(),
        af1_state: AdaConvState::default(),
        preemph_mem: 0.0,
        deemph_mem: 0.0,
    }
}

/// Process a 20ms LACE frame (4 subframes of 80 samples each).
pub fn lace_process_20ms_frame(
    model: &Lace,
    state: &mut LaceState,
    x_out: &mut [f32],
    x_in: &[f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[usize; 4],
) {
    let cd = model.params.cond_dim;
    let mut cond = [0.0f32; 1024];
    osce_feature_net(
        &model.layers,
        state,
        &model.params,
        &mut cond[..4 * cd],
        features,
        numbits,
        periods,
    );

    let total = 4 * LACE_FRAME_SIZE;
    let mut x_pre = [0.0f32; 4 * 80];
    apply_preemphasis(&mut x_pre[..total], &x_in[..total], &mut state.preemph_mem);

    let mut x_buf = [0.0f32; 4 * 80];

    for sf in 0..4 {
        let sc = &cond[sf * cd..(sf + 1) * cd];
        let si = sf * LACE_FRAME_SIZE;
        let ei = si + LACE_FRAME_SIZE;

        let mut tmp = [0.0f32; LACE_FRAME_SIZE];
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut tmp,
            &x_pre[si..ei],
            sc,
            &model.layers.cf1_kernel,
            &model.layers.cf1_gain,
            &model.layers.cf1_global_gain,
            periods[sf],
            cd,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            model.cf1_kernel_size,
            model.cf1_kernel_size - 1,
            CF_FILTER_GAIN_A,
            CF_FILTER_GAIN_B,
            CF_LOG_GAIN_LIMIT,
            &model.window,
        );

        adacomb_process_frame(
            &mut state.cf2_state,
            &mut x_buf[si..ei],
            &tmp,
            sc,
            &model.layers.cf2_kernel,
            &model.layers.cf2_gain,
            &model.layers.cf2_global_gain,
            periods[sf],
            cd,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            model.cf2_kernel_size,
            model.cf2_kernel_size - 1,
            CF_FILTER_GAIN_A,
            CF_FILTER_GAIN_B,
            CF_LOG_GAIN_LIMIT,
            &model.window,
        );

        adaconv_process_frame(
            &mut state.af1_state,
            &mut x_out[si..ei],
            &x_buf[si..ei],
            sc,
            &model.layers.af1_kernel,
            &model.layers.af1_gain,
            cd,
            LACE_FRAME_SIZE,
            LACE_OVERLAP_SIZE,
            model.af1_in_channels,
            model.af1_out_channels,
            model.af1_kernel_size,
            model.af1_kernel_size - 1,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window,
        );
    }

    apply_deemphasis(&mut x_out[..total], &mut state.deemph_mem);
}
