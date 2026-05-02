//! Shared OSCE types and functions used by both LACE and NoLACE.

use super::config::*;
use crate::nnet::ops::{compute_generic_conv1d, compute_generic_dense, compute_generic_gru};
use crate::nnet::{Activation, LinearLayer};

const MAX_FEATURE_BUF: usize = 1024;

/// Trait for accessing the shared feature network layers.
/// Both LACE and NoLACE implement this — the feature net architecture is identical,
/// only the dimensions and weights differ.
pub trait FeatureNetLayers {
    fn pitch_embedding(&self) -> &LinearLayer;
    fn fnet_conv1(&self) -> &LinearLayer;
    fn fnet_conv2(&self) -> &LinearLayer;
    fn fnet_tconv(&self) -> &LinearLayer;
    fn fnet_gru_input(&self) -> &LinearLayer;
    fn fnet_gru_recurrent(&self) -> &LinearLayer;
}

/// Trait for accessing the shared feature net state.
pub trait FeatureNetState {
    fn fnet_conv2_state(&mut self) -> &mut [f32];
    fn gru_state(&mut self) -> &mut [f32];
}

/// Parameters for the feature net that differ between LACE and NoLACE.
pub struct FeatureNetParams {
    pub cond_dim: usize,
    pub hidden_dim: usize,
    pub pitch_embed_dim: usize,
    pub numbits_embed_dim: usize,
    pub pitch_max: usize,
    pub numbits_range_low: f32,
    pub numbits_range_high: f32,
    pub numbits_scales: [f32; 8],
}

/// Sinusoidal numbits embedding matching C `compute_{lace,nolace}_numbits_embedding`.
pub fn compute_numbits_embedding(
    out: &mut [f32],
    numbits: f32,
    dim: usize,
    log_low: f32,
    log_high: f32,
    scales: &[f32],
) {
    let log_numbits = numbits.ln();
    let mid = (log_high + log_low) * 0.5;
    let x = log_numbits.clamp(log_low, log_high) - mid;
    for i in 0..dim.min(scales.len()) {
        out[i] = (x * scales[i] - 0.5).sin();
    }
}

/// Shared OSCE feature network: features → conditioning vectors.
/// Used by both LACE and NoLACE with their respective layer weights and dimensions.
pub fn osce_feature_net(
    layers: &impl FeatureNetLayers,
    state: &mut impl FeatureNetState,
    params: &FeatureNetParams,
    output: &mut [f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[usize; 4],
) {
    let cd = params.cond_dim;
    let hd = params.hidden_dim;
    let ed = params.pitch_embed_dim;
    let nd = params.numbits_embed_dim;
    let log_low = params.numbits_range_low.ln();
    let log_high = params.numbits_range_high.ln();

    let mut nb_emb = [0.0f32; 16];
    compute_numbits_embedding(
        &mut nb_emb[..nd],
        numbits[0],
        nd,
        log_low,
        log_high,
        &params.numbits_scales,
    );
    compute_numbits_embedding(
        &mut nb_emb[nd..2 * nd],
        numbits[1],
        nd,
        log_low,
        log_high,
        &params.numbits_scales,
    );

    let embed_w = layers.pitch_embedding().float_weights.as_ref().unwrap();
    let feat_dim = OSCE_FEATURE_DIM;
    let conv1_in = feat_dim + ed + 2 * nd;

    let mut input_buf = [0.0f32; MAX_FEATURE_BUF];
    let mut output_buf = [0.0f32; MAX_FEATURE_BUF];

    // Per-subframe: features + pitch embedding + numbits embedding → conv1
    for sf in 0..4 {
        input_buf[..feat_dim].copy_from_slice(&features[sf * feat_dim..(sf + 1) * feat_dim]);
        let pe_start = periods[sf].min(params.pitch_max) * ed;
        if pe_start + ed <= embed_w.len() {
            input_buf[feat_dim..feat_dim + ed].copy_from_slice(&embed_w[pe_start..pe_start + ed]);
        }
        input_buf[feat_dim + ed..feat_dim + ed + 2 * nd].copy_from_slice(&nb_emb[..2 * nd]);
        compute_generic_conv1d(
            layers.fnet_conv1(),
            &mut output_buf[sf * hd..(sf + 1) * hd],
            &mut [],
            &input_buf[..conv1_in],
            conv1_in,
            Activation::Tanh,
        );
    }

    // Subframe accumulation: conv2
    input_buf[..4 * hd].copy_from_slice(&output_buf[..4 * hd]);
    compute_generic_conv1d(
        layers.fnet_conv2(),
        &mut output_buf[..4 * cd],
        state.fnet_conv2_state(),
        &input_buf[..4 * hd],
        4 * hd,
        Activation::Tanh,
    );

    // Tconv upsampling
    input_buf[..4 * cd].copy_from_slice(&output_buf[..4 * cd]);
    compute_generic_dense(
        layers.fnet_tconv(),
        &mut output_buf[..4 * cd],
        &input_buf[..4 * cd],
        Activation::Tanh,
    );

    // GRU per subframe
    input_buf[..4 * cd].copy_from_slice(&output_buf[..4 * cd]);
    for sf in 0..4 {
        compute_generic_gru(
            layers.fnet_gru_input(),
            layers.fnet_gru_recurrent(),
            state.gru_state(),
            &input_buf[sf * cd..(sf + 1) * cd],
        );
        output[sf * cd..(sf + 1) * cd].copy_from_slice(state.gru_state());
    }
}

/// Apply pre-emphasis filter in-place. Matches C OSCE pre-emphasis.
pub fn apply_preemphasis(out: &mut [f32], input: &[f32], mem: &mut f32) {
    for i in 0..input.len() {
        let xi = input[i];
        out[i] = xi - OSCE_PREEMPH * *mem;
        *mem = xi;
    }
}

/// Apply de-emphasis filter in-place. Matches C OSCE de-emphasis.
pub fn apply_deemphasis(buf: &mut [f32], mem: &mut f32) {
    for sample in buf.iter_mut() {
        *sample += OSCE_PREEMPH * *mem;
        *mem = *sample;
    }
}
