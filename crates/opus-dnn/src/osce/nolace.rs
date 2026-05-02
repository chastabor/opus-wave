use crate::nndsp::adacomb::adacomb_process_frame;
use crate::nndsp::adaconv::adaconv_process_frame;
use crate::nndsp::adashape::adashape_process_frame;
use crate::nndsp::{AdaCombState, AdaConvState, AdaShapeState, compute_overlap_window};
use crate::nnet::ops::compute_generic_conv1d;
use crate::nnet::weights::{WeightError, linear_init, weight_output_dim};
use crate::nnet::{Activation, LinearLayer, WeightArray};

use super::common::*;
use super::config::*;

pub const NOLACE_FRAME_SIZE: usize = 80;
const NOLACE_OVERLAP_SIZE: usize = 40;

// Filter gain constants from nolace_data.h.
const CF_FILTER_GAIN_A: f32 = 0.690776;
const CF_FILTER_GAIN_B: f32 = 0.0;
const CF_LOG_GAIN_LIMIT: f32 = 1.151293;
const AF_FILTER_GAIN_A: f32 = 1.381551;
const AF_FILTER_GAIN_B: f32 = 0.0;

/// NoLACE model layers (34 total). Matches C `NOLACELayers` from nolace_data.h.
pub struct NoLaceLayers {
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
    pub tdshape1_alpha1_f: LinearLayer,
    pub tdshape1_alpha1_t: LinearLayer,
    pub tdshape1_alpha2: LinearLayer,
    pub tdshape2_alpha1_f: LinearLayer,
    pub tdshape2_alpha1_t: LinearLayer,
    pub tdshape2_alpha2: LinearLayer,
    pub tdshape3_alpha1_f: LinearLayer,
    pub tdshape3_alpha1_t: LinearLayer,
    pub tdshape3_alpha2: LinearLayer,
    pub af2_kernel: LinearLayer,
    pub af2_gain: LinearLayer,
    pub af3_kernel: LinearLayer,
    pub af3_gain: LinearLayer,
    pub af4_kernel: LinearLayer,
    pub af4_gain: LinearLayer,
    pub post_cf1: LinearLayer,
    pub post_cf2: LinearLayer,
    pub post_af1: LinearLayer,
    pub post_af2: LinearLayer,
    pub post_af3: LinearLayer,
}

impl FeatureNetLayers for NoLaceLayers {
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

/// NoLACE model with layers + window + shared params.
pub struct NoLace {
    pub layers: NoLaceLayers,
    pub window: [f32; NOLACE_OVERLAP_SIZE],
    pub params: FeatureNetParams,
}

/// NoLACE processing state.
pub struct NoLaceState {
    pub fnet_conv2_state: Vec<f32>,
    pub gru_state: Vec<f32>,
    pub post_cf1_state: Vec<f32>,
    pub post_cf2_state: Vec<f32>,
    pub post_af1_state: Vec<f32>,
    pub post_af2_state: Vec<f32>,
    pub post_af3_state: Vec<f32>,
    pub cf1_state: AdaCombState,
    pub cf2_state: AdaCombState,
    pub af1_state: AdaConvState,
    pub af2_state: AdaConvState,
    pub af3_state: AdaConvState,
    pub af4_state: AdaConvState,
    pub tdshape1_state: AdaShapeState,
    pub tdshape2_state: AdaShapeState,
    pub tdshape3_state: AdaShapeState,
    pub preemph_mem: f32,
    pub deemph_mem: f32,
}

impl FeatureNetState for NoLaceState {
    fn fnet_conv2_state(&mut self) -> &mut [f32] {
        &mut self.fnet_conv2_state
    }
    fn gru_state(&mut self) -> &mut [f32] {
        &mut self.gru_state
    }
}

/// Initialize NoLACE model from weight arrays.
/// Dimensions match C `init_nolacelayers` from nolace_data.c.
pub fn init_nolace(arrays: &[WeightArray]) -> Result<NoLace, WeightError> {
    let dim = |name: &str| weight_output_dim(arrays, name);
    let cond_dim = dim("nolace_fnet_gru_input_bias")? / 3;
    let hidden_dim = dim("nolace_fnet_conv1_bias")?;
    let pitch_embed_dim = dim("nolace_pitch_embedding_bias")?;
    let numbits_embed_dim = 8;
    let feat_in = OSCE_FEATURE_DIM + pitch_embed_dim + 2 * numbits_embed_dim;
    let pitch_max = 300;
    let conv2_out = dim("nolace_fnet_conv2_bias")?;

    let cf1_ks = dim("nolace_cf1_kernel_bias")?;
    let cf2_ks = dim("nolace_cf2_kernel_bias")?;
    let af1_ks_total = dim("nolace_af1_kernel_bias")?;
    let af2_ks_total = dim("nolace_af2_kernel_bias")?;
    let af3_ks_total = dim("nolace_af3_kernel_bias")?;
    let af4_ks_total = dim("nolace_af4_kernel_bias")?;
    let post_dim = dim("nolace_post_cf1_bias")?;
    let tdshape_dim = dim("nolace_tdshape1_alpha1_f_bias")?;

    let layers = NoLaceLayers {
        // C: linear_init(pitch_embedding, bias, NULL, NULL, float, NULL, NULL, NULL, 301, 64)
        pitch_embedding: linear_init(
            arrays,
            Some("nolace_pitch_embedding_bias"),
            None,
            Some("nolace_pitch_embedding_weights"),
            None,
            None,
            None,
            pitch_max + 1,
            pitch_embed_dim,
        )?,
        // C: linear_init(fnet_conv1, bias, NULL, NULL, float, NULL, NULL, NULL, 173, 96)
        fnet_conv1: linear_init(
            arrays,
            Some("nolace_fnet_conv1_bias"),
            None,
            Some("nolace_fnet_conv1_weights"),
            None,
            None,
            None,
            feat_in,
            hidden_dim,
        )?,
        // C: linear_init(fnet_conv2, bias, subias, int8, float, NULL, NULL, scale, 768, 160)
        fnet_conv2: linear_init(
            arrays,
            Some("nolace_fnet_conv2_bias"),
            Some("nolace_fnet_conv2_weights"),
            Some("nolace_fnet_conv2_weights"),
            None,
            None,
            Some("nolace_fnet_conv2_scale"),
            2 * 4 * hidden_dim,
            conv2_out,
        )?,
        // C: linear_init(fnet_tconv, bias, subias, int8, float, NULL, NULL, scale, 160, 640)
        fnet_tconv: linear_init(
            arrays,
            Some("nolace_fnet_tconv_bias"),
            Some("nolace_fnet_tconv_weights"),
            Some("nolace_fnet_tconv_weights"),
            None,
            None,
            Some("nolace_fnet_tconv_scale"),
            conv2_out,
            4 * cond_dim,
        )?,
        // C: linear_init(fnet_gru_input, bias, subias, int8, float, NULL, NULL, scale, 160, 480)
        fnet_gru_input: linear_init(
            arrays,
            Some("nolace_fnet_gru_input_bias"),
            Some("nolace_fnet_gru_input_weights"),
            Some("nolace_fnet_gru_input_weights"),
            None,
            None,
            Some("nolace_fnet_gru_input_scale"),
            cond_dim,
            3 * cond_dim,
        )?,
        // C: linear_init(fnet_gru_recurrent, bias, subias, int8, float, NULL, NULL, scale, 160, 480)
        fnet_gru_recurrent: linear_init(
            arrays,
            Some("nolace_fnet_gru_recurrent_bias"),
            Some("nolace_fnet_gru_recurrent_weights"),
            Some("nolace_fnet_gru_recurrent_weights"),
            None,
            None,
            Some("nolace_fnet_gru_recurrent_scale"),
            cond_dim,
            3 * cond_dim,
        )?,
        // C: linear_init(cf1_kernel, bias, subias, int8, float, NULL, NULL, scale, 160, 16)
        cf1_kernel: linear_init(
            arrays,
            Some("nolace_cf1_kernel_bias"),
            Some("nolace_cf1_kernel_weights"),
            Some("nolace_cf1_kernel_weights"),
            None,
            None,
            Some("nolace_cf1_kernel_scale"),
            cond_dim,
            cf1_ks,
        )?,
        // C: linear_init(cf1_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 1)
        cf1_gain: linear_init(
            arrays,
            Some("nolace_cf1_gain_bias"),
            None,
            Some("nolace_cf1_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        // C: linear_init(cf1_global_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 1)
        cf1_global_gain: linear_init(
            arrays,
            Some("nolace_cf1_global_gain_bias"),
            None,
            Some("nolace_cf1_global_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        // C: linear_init(cf2_kernel, bias, subias, int8, float, NULL, NULL, scale, 160, 16)
        cf2_kernel: linear_init(
            arrays,
            Some("nolace_cf2_kernel_bias"),
            Some("nolace_cf2_kernel_weights"),
            Some("nolace_cf2_kernel_weights"),
            None,
            None,
            Some("nolace_cf2_kernel_scale"),
            cond_dim,
            cf2_ks,
        )?,
        // C: linear_init(cf2_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 1)
        cf2_gain: linear_init(
            arrays,
            Some("nolace_cf2_gain_bias"),
            None,
            Some("nolace_cf2_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        // C: linear_init(cf2_global_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 1)
        cf2_global_gain: linear_init(
            arrays,
            Some("nolace_cf2_global_gain_bias"),
            None,
            Some("nolace_cf2_global_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        // C: linear_init(af1_kernel, bias, subias, int8, float, NULL, NULL, scale, 160, 32)
        af1_kernel: linear_init(
            arrays,
            Some("nolace_af1_kernel_bias"),
            Some("nolace_af1_kernel_weights"),
            Some("nolace_af1_kernel_weights"),
            None,
            None,
            Some("nolace_af1_kernel_scale"),
            cond_dim,
            af1_ks_total,
        )?,
        // C: linear_init(af1_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 2)
        af1_gain: linear_init(
            arrays,
            Some("nolace_af1_gain_bias"),
            None,
            Some("nolace_af1_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            2,
        )?,
        // C: linear_init(tdshape1_alpha1_f, bias, subias, int8, float, NULL, NULL, scale, 320, 80)
        tdshape1_alpha1_f: linear_init(
            arrays,
            Some("nolace_tdshape1_alpha1_f_bias"),
            Some("nolace_tdshape1_alpha1_f_weights"),
            Some("nolace_tdshape1_alpha1_f_weights"),
            None,
            None,
            Some("nolace_tdshape1_alpha1_f_scale"),
            2 * cond_dim,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape1_alpha1_t, bias, NULL, NULL, float, NULL, NULL, NULL, 42, 80)
        tdshape1_alpha1_t: linear_init(
            arrays,
            Some("nolace_tdshape1_alpha1_t_bias"),
            None,
            Some("nolace_tdshape1_alpha1_t_weights"),
            None,
            None,
            None,
            42,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape1_alpha2, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 80)
        tdshape1_alpha2: linear_init(
            arrays,
            Some("nolace_tdshape1_alpha2_bias"),
            None,
            Some("nolace_tdshape1_alpha2_weights"),
            None,
            None,
            None,
            cond_dim,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape2_alpha1_f, bias, subias, int8, float, NULL, NULL, scale, 320, 80)
        tdshape2_alpha1_f: linear_init(
            arrays,
            Some("nolace_tdshape2_alpha1_f_bias"),
            Some("nolace_tdshape2_alpha1_f_weights"),
            Some("nolace_tdshape2_alpha1_f_weights"),
            None,
            None,
            Some("nolace_tdshape2_alpha1_f_scale"),
            2 * cond_dim,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape2_alpha1_t, bias, NULL, NULL, float, NULL, NULL, NULL, 42, 80)
        tdshape2_alpha1_t: linear_init(
            arrays,
            Some("nolace_tdshape2_alpha1_t_bias"),
            None,
            Some("nolace_tdshape2_alpha1_t_weights"),
            None,
            None,
            None,
            42,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape2_alpha2, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 80)
        tdshape2_alpha2: linear_init(
            arrays,
            Some("nolace_tdshape2_alpha2_bias"),
            None,
            Some("nolace_tdshape2_alpha2_weights"),
            None,
            None,
            None,
            cond_dim,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape3_alpha1_f, bias, subias, int8, float, NULL, NULL, scale, 320, 80)
        tdshape3_alpha1_f: linear_init(
            arrays,
            Some("nolace_tdshape3_alpha1_f_bias"),
            Some("nolace_tdshape3_alpha1_f_weights"),
            Some("nolace_tdshape3_alpha1_f_weights"),
            None,
            None,
            Some("nolace_tdshape3_alpha1_f_scale"),
            2 * cond_dim,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape3_alpha1_t, bias, NULL, NULL, float, NULL, NULL, NULL, 42, 80)
        tdshape3_alpha1_t: linear_init(
            arrays,
            Some("nolace_tdshape3_alpha1_t_bias"),
            None,
            Some("nolace_tdshape3_alpha1_t_weights"),
            None,
            None,
            None,
            42,
            tdshape_dim,
        )?,
        // C: linear_init(tdshape3_alpha2, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 80)
        tdshape3_alpha2: linear_init(
            arrays,
            Some("nolace_tdshape3_alpha2_bias"),
            None,
            Some("nolace_tdshape3_alpha2_weights"),
            None,
            None,
            None,
            cond_dim,
            tdshape_dim,
        )?,
        // C: linear_init(af2_kernel, bias, subias, int8, float, NULL, NULL, scale, 160, 64)
        af2_kernel: linear_init(
            arrays,
            Some("nolace_af2_kernel_bias"),
            Some("nolace_af2_kernel_weights"),
            Some("nolace_af2_kernel_weights"),
            None,
            None,
            Some("nolace_af2_kernel_scale"),
            cond_dim,
            af2_ks_total,
        )?,
        // C: linear_init(af2_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 2)
        af2_gain: linear_init(
            arrays,
            Some("nolace_af2_gain_bias"),
            None,
            Some("nolace_af2_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            2,
        )?,
        // C: linear_init(af3_kernel, bias, subias, int8, float, NULL, NULL, scale, 160, 64)
        af3_kernel: linear_init(
            arrays,
            Some("nolace_af3_kernel_bias"),
            Some("nolace_af3_kernel_weights"),
            Some("nolace_af3_kernel_weights"),
            None,
            None,
            Some("nolace_af3_kernel_scale"),
            cond_dim,
            af3_ks_total,
        )?,
        // C: linear_init(af3_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 2)
        af3_gain: linear_init(
            arrays,
            Some("nolace_af3_gain_bias"),
            None,
            Some("nolace_af3_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            2,
        )?,
        // C: linear_init(af4_kernel, bias, subias, int8, float, NULL, NULL, scale, 160, 32)
        af4_kernel: linear_init(
            arrays,
            Some("nolace_af4_kernel_bias"),
            Some("nolace_af4_kernel_weights"),
            Some("nolace_af4_kernel_weights"),
            None,
            None,
            Some("nolace_af4_kernel_scale"),
            cond_dim,
            af4_ks_total,
        )?,
        // C: linear_init(af4_gain, bias, NULL, NULL, float, NULL, NULL, NULL, 160, 1)
        af4_gain: linear_init(
            arrays,
            Some("nolace_af4_gain_bias"),
            None,
            Some("nolace_af4_gain_weights"),
            None,
            None,
            None,
            cond_dim,
            1,
        )?,
        // C: linear_init(post_cf1, bias, subias, int8, float, NULL, NULL, scale, 320, 160)
        post_cf1: linear_init(
            arrays,
            Some("nolace_post_cf1_bias"),
            Some("nolace_post_cf1_weights"),
            Some("nolace_post_cf1_weights"),
            None,
            None,
            Some("nolace_post_cf1_scale"),
            2 * cond_dim,
            post_dim,
        )?,
        // C: linear_init(post_cf2, bias, subias, int8, float, NULL, NULL, scale, 320, 160)
        post_cf2: linear_init(
            arrays,
            Some("nolace_post_cf2_bias"),
            Some("nolace_post_cf2_weights"),
            Some("nolace_post_cf2_weights"),
            None,
            None,
            Some("nolace_post_cf2_scale"),
            2 * cond_dim,
            post_dim,
        )?,
        // C: linear_init(post_af1, bias, subias, int8, float, NULL, NULL, scale, 320, 160)
        post_af1: linear_init(
            arrays,
            Some("nolace_post_af1_bias"),
            Some("nolace_post_af1_weights"),
            Some("nolace_post_af1_weights"),
            None,
            None,
            Some("nolace_post_af1_scale"),
            2 * cond_dim,
            post_dim,
        )?,
        // C: linear_init(post_af2, bias, subias, int8, float, NULL, NULL, scale, 320, 160)
        post_af2: linear_init(
            arrays,
            Some("nolace_post_af2_bias"),
            Some("nolace_post_af2_weights"),
            Some("nolace_post_af2_weights"),
            None,
            None,
            Some("nolace_post_af2_scale"),
            2 * cond_dim,
            post_dim,
        )?,
        // C: linear_init(post_af3, bias, subias, int8, float, NULL, NULL, scale, 320, 160)
        post_af3: linear_init(
            arrays,
            Some("nolace_post_af3_bias"),
            Some("nolace_post_af3_weights"),
            Some("nolace_post_af3_weights"),
            None,
            None,
            Some("nolace_post_af3_scale"),
            2 * cond_dim,
            post_dim,
        )?,
    };

    let mut window = [0.0f32; NOLACE_OVERLAP_SIZE];
    compute_overlap_window(&mut window, NOLACE_OVERLAP_SIZE);

    Ok(NoLace {
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
                1.0357312, 1.7355591, 3.6004558, 4.5524783, 5.9325595, 7.1769705, 8.1149988,
                8.7706327,
            ],
        },
    })
}

pub fn nolace_state_init(model: &NoLace) -> NoLaceState {
    let cd = model.params.cond_dim;
    NoLaceState {
        fnet_conv2_state: vec![0.0; model.layers.fnet_conv2.nb_inputs],
        gru_state: vec![0.0; cd],
        post_cf1_state: vec![0.0; cd],
        post_cf2_state: vec![0.0; cd],
        post_af1_state: vec![0.0; cd],
        post_af2_state: vec![0.0; cd],
        post_af3_state: vec![0.0; cd],
        cf1_state: AdaCombState::default(),
        cf2_state: AdaCombState::default(),
        af1_state: AdaConvState::default(),
        af2_state: AdaConvState::default(),
        af3_state: AdaConvState::default(),
        af4_state: AdaConvState::default(),
        tdshape1_state: AdaShapeState::default(),
        tdshape2_state: AdaShapeState::default(),
        tdshape3_state: AdaShapeState::default(),
        preemph_mem: 0.0,
        deemph_mem: 0.0,
    }
}

/// Process a 20ms NoLACE frame.
/// Chain: feature net → cf1 → post_cf1 → cf2 → post_cf2 →
///        af1 → post_af1 → tdshape1 → af2 → post_af2 → tdshape2 →
///        af3 → post_af3 → tdshape3 → af4 → de-emphasis
pub fn nolace_process_20ms_frame(
    model: &NoLace,
    state: &mut NoLaceState,
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

    let total = 4 * NOLACE_FRAME_SIZE;
    let mut x = [0.0f32; 4 * 80];
    apply_preemphasis(&mut x[..total], &x_in[..total], &mut state.preemph_mem);

    let mut buf1 = [0.0f32; 4 * 80];
    let mut buf2 = [0.0f32; 4 * 80 * 2];

    for sf in 0..4 {
        let sc = &cond[sf * cd..(sf + 1) * cd];
        let si = sf * NOLACE_FRAME_SIZE;
        let ei = si + NOLACE_FRAME_SIZE;
        let si2 = sf * NOLACE_FRAME_SIZE * 2;
        let ch2 = si2 + NOLACE_FRAME_SIZE; // second channel offset

        let mut post_out = [0.0f32; 256];

        // cf1
        adacomb_process_frame(
            &mut state.cf1_state,
            &mut buf1[si..ei],
            &x[si..ei],
            sc,
            &model.layers.cf1_kernel,
            &model.layers.cf1_gain,
            &model.layers.cf1_global_gain,
            periods[sf],
            cd,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            16,
            8,
            CF_FILTER_GAIN_A,
            CF_FILTER_GAIN_B,
            CF_LOG_GAIN_LIMIT,
            &model.window,
        );

        compute_generic_conv1d(
            &model.layers.post_cf1,
            &mut post_out[..cd],
            &mut state.post_cf1_state,
            sc,
            cd,
            Activation::Tanh,
        );

        // cf2
        adacomb_process_frame(
            &mut state.cf2_state,
            &mut x[si..ei],
            &buf1[si..ei],
            &post_out[..cd],
            &model.layers.cf2_kernel,
            &model.layers.cf2_gain,
            &model.layers.cf2_global_gain,
            periods[sf],
            cd,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            16,
            8,
            CF_FILTER_GAIN_A,
            CF_FILTER_GAIN_B,
            CF_LOG_GAIN_LIMIT,
            &model.window,
        );

        compute_generic_conv1d(
            &model.layers.post_cf2,
            &mut post_out[..cd],
            &mut state.post_cf2_state,
            sc,
            cd,
            Activation::Tanh,
        );

        // af1 (1 → 2 channels)
        adaconv_process_frame(
            &mut state.af1_state,
            &mut buf2[si2..si2 + NOLACE_FRAME_SIZE * 2],
            &x[si..ei],
            &post_out[..cd],
            &model.layers.af1_kernel,
            &model.layers.af1_gain,
            cd,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            1,
            2,
            16,
            15,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window,
        );

        compute_generic_conv1d(
            &model.layers.post_af1,
            &mut post_out[..cd],
            &mut state.post_af1_state,
            sc,
            cd,
            Activation::Tanh,
        );

        // tdshape1 — second channel
        let mut ts_in = [0.0f32; NOLACE_FRAME_SIZE];
        ts_in.copy_from_slice(&buf2[ch2..ch2 + NOLACE_FRAME_SIZE]);
        adashape_process_frame(
            &mut state.tdshape1_state,
            &mut buf2[ch2..ch2 + NOLACE_FRAME_SIZE],
            &ts_in,
            &post_out[..cd],
            &model.layers.tdshape1_alpha1_f,
            &model.layers.tdshape1_alpha1_t,
            &model.layers.tdshape1_alpha2,
            cd,
            NOLACE_FRAME_SIZE,
            4,
            1,
        );

        // af2 (2 → 2)
        let mut af2_buf = [0.0f32; NOLACE_FRAME_SIZE * 2];
        adaconv_process_frame(
            &mut state.af2_state,
            &mut af2_buf,
            &buf2[si2..si2 + NOLACE_FRAME_SIZE * 2],
            &post_out[..cd],
            &model.layers.af2_kernel,
            &model.layers.af2_gain,
            cd,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            2,
            2,
            16,
            15,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window,
        );

        compute_generic_conv1d(
            &model.layers.post_af2,
            &mut post_out[..cd],
            &mut state.post_af2_state,
            sc,
            cd,
            Activation::Tanh,
        );

        // tdshape2 — second channel
        ts_in.copy_from_slice(&af2_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2]);
        adashape_process_frame(
            &mut state.tdshape2_state,
            &mut af2_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2],
            &ts_in,
            &post_out[..cd],
            &model.layers.tdshape2_alpha1_f,
            &model.layers.tdshape2_alpha1_t,
            &model.layers.tdshape2_alpha2,
            cd,
            NOLACE_FRAME_SIZE,
            4,
            1,
        );

        // af3 (2 → 2)
        let mut af3_buf = [0.0f32; NOLACE_FRAME_SIZE * 2];
        adaconv_process_frame(
            &mut state.af3_state,
            &mut af3_buf,
            &af2_buf,
            &post_out[..cd],
            &model.layers.af3_kernel,
            &model.layers.af3_gain,
            cd,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            2,
            2,
            16,
            15,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window,
        );

        compute_generic_conv1d(
            &model.layers.post_af3,
            &mut post_out[..cd],
            &mut state.post_af3_state,
            sc,
            cd,
            Activation::Tanh,
        );

        // tdshape3 — second channel
        ts_in.copy_from_slice(&af3_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2]);
        adashape_process_frame(
            &mut state.tdshape3_state,
            &mut af3_buf[NOLACE_FRAME_SIZE..NOLACE_FRAME_SIZE * 2],
            &ts_in,
            &post_out[..cd],
            &model.layers.tdshape3_alpha1_f,
            &model.layers.tdshape3_alpha1_t,
            &model.layers.tdshape3_alpha2,
            cd,
            NOLACE_FRAME_SIZE,
            4,
            1,
        );

        // af4 (2 → 1)
        adaconv_process_frame(
            &mut state.af4_state,
            &mut x_out[si..ei],
            &af3_buf,
            &post_out[..cd],
            &model.layers.af4_kernel,
            &model.layers.af4_gain,
            cd,
            NOLACE_FRAME_SIZE,
            NOLACE_OVERLAP_SIZE,
            2,
            1,
            16,
            15,
            AF_FILTER_GAIN_A,
            AF_FILTER_GAIN_B,
            1.0,
            &model.window,
        );
    }

    apply_deemphasis(&mut x_out[..total], &mut state.deemph_mem);
}
