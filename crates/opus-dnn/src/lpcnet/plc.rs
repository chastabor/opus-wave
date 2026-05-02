use super::enc::*;
use crate::fargan::*;
use crate::freq::*;
use crate::nnet::ops::{compute_generic_dense, compute_generic_gru};
use crate::nnet::weights::{WeightError, linear_init, weight_output_dim};
use crate::nnet::{Activation, LinearLayer, WeightArray};

/// Maximum FEC frames buffered (DRED can provide up to 104 frames).
pub const PLC_MAX_FEC: usize = 104;

/// Number of continuation feature vectors stored.
pub const CONT_VECTORS: usize = 5;

/// PCM buffer size in samples.
pub const PLC_BUF_SIZE: usize = (CONT_VECTORS + 10) * FRAME_SIZE;

/// Attenuation table for loss concealment (indexed by loss_count).
const ATT_TABLE: [f32; 10] = [0.0, 0.0, -0.2, -0.2, -0.4, -0.4, -0.8, -0.8, -1.6, -1.6];

/// PLC prediction network model (6 layers).
/// Matches auto-generated C `PLCModel` from plc_data.h.
pub struct PlcModel {
    pub plc_dense_in: LinearLayer,
    pub plc_gru1_input: LinearLayer,
    pub plc_gru1_recurrent: LinearLayer,
    pub plc_gru2_input: LinearLayer,
    pub plc_gru2_recurrent: LinearLayer,
    pub plc_dense_out: LinearLayer,
}

/// PLC recurrent network state (GRU states).
/// Matches C `PLCNetState`.
#[derive(Clone)]
pub struct PlcNetState {
    pub gru1_state: Vec<f32>,
    pub gru2_state: Vec<f32>,
}

/// Full LPCNet PLC state.
/// Matches C `LPCNetPLCState` from lpcnet_private.h.
pub struct LpcnetPlcState {
    pub model: PlcModel,
    pub fargan: FarganState,
    pub enc: LpcnetEncState,
    pub loaded: bool,

    pub fec: Vec<f32>,
    pub analysis_gap: bool,
    pub fec_read_pos: usize,
    pub fec_fill_pos: usize,
    pub fec_skip: usize,
    pub analysis_pos: usize,
    pub predict_pos: usize,
    pub pcm: Vec<f32>,
    pub blend: bool,
    pub features: [f32; NB_TOTAL_FEATURES],
    pub cont_features: Vec<f32>,
    pub loss_count: usize,
    pub plc_net: PlcNetState,
    pub plc_bak: [PlcNetState; 2],
}

/// Initialize PLC prediction model from weight arrays.
pub fn init_plcmodel(arrays: &[WeightArray]) -> Result<PlcModel, WeightError> {
    let dim = |name| weight_output_dim(arrays, name);
    let dense_in_out = dim("plc_dense_in_bias")?;
    let gru1_3n = dim("plc_gru1_input_bias")?;
    let gru1_out = gru1_3n / 3;
    let gru2_3n = dim("plc_gru2_input_bias")?;
    let gru2_out = gru2_3n / 3;
    let dense_out_size = dim("plc_dense_out_bias")?;
    let dense_in_in = 2 * NB_BANDS + NB_FEATURES + 1;

    Ok(PlcModel {
        plc_dense_in: linear_init(
            arrays,
            Some("plc_dense_in_bias"),
            None,
            Some("plc_dense_in_weights"),
            None,
            None,
            None,
            dense_in_in,
            dense_in_out,
        )?,
        plc_gru1_input: linear_init(
            arrays,
            Some("plc_gru1_input_bias"),
            Some("plc_gru1_input_weights"),
            Some("plc_gru1_input_weights"),
            None,
            None,
            Some("plc_gru1_input_scale"),
            dense_in_out,
            gru1_3n,
        )?,
        plc_gru1_recurrent: linear_init(
            arrays,
            Some("plc_gru1_recurrent_bias"),
            Some("plc_gru1_recurrent_weights"),
            Some("plc_gru1_recurrent_weights"),
            None,
            None,
            Some("plc_gru1_recurrent_scale"),
            gru1_out,
            gru1_3n,
        )?,
        plc_gru2_input: linear_init(
            arrays,
            Some("plc_gru2_input_bias"),
            Some("plc_gru2_input_weights"),
            Some("plc_gru2_input_weights"),
            None,
            None,
            Some("plc_gru2_input_scale"),
            gru1_out,
            gru2_3n,
        )?,
        plc_gru2_recurrent: linear_init(
            arrays,
            Some("plc_gru2_recurrent_bias"),
            Some("plc_gru2_recurrent_weights"),
            Some("plc_gru2_recurrent_weights"),
            None,
            None,
            Some("plc_gru2_recurrent_scale"),
            gru2_out,
            gru2_3n,
        )?,
        plc_dense_out: linear_init(
            arrays,
            Some("plc_dense_out_bias"),
            None,
            Some("plc_dense_out_weights"),
            None,
            None,
            None,
            gru2_out,
            dense_out_size,
        )?,
    })
}

fn new_plc_net_state(model: &PlcModel) -> PlcNetState {
    let gru1_size = model.plc_gru1_recurrent.nb_inputs;
    let gru2_size = model.plc_gru2_recurrent.nb_inputs;
    PlcNetState {
        gru1_state: vec![0.0; gru1_size],
        gru2_state: vec![0.0; gru2_size],
    }
}

/// Initialize the full PLC state. Requires pre-initialized FARGAN and encoder models.
pub fn lpcnet_plc_init(
    plc_model: PlcModel,
    fargan_state: FarganState,
    enc_state: LpcnetEncState,
) -> LpcnetPlcState {
    let net = new_plc_net_state(&plc_model);
    LpcnetPlcState {
        plc_bak: [net.clone(), net.clone()],
        plc_net: net,
        model: plc_model,
        fargan: fargan_state,
        enc: enc_state,
        loaded: true,
        fec: vec![0.0; PLC_MAX_FEC * NB_FEATURES],
        analysis_gap: true,
        fec_read_pos: 0,
        fec_fill_pos: 0,
        fec_skip: 0,
        analysis_pos: PLC_BUF_SIZE,
        predict_pos: PLC_BUF_SIZE,
        pcm: vec![0.0; PLC_BUF_SIZE],
        blend: false,
        features: [0.0; NB_TOTAL_FEATURES],
        cont_features: vec![0.0; CONT_VECTORS * NB_FEATURES],
        loss_count: 0,
    }
}

/// Add FEC (Forward Error Correction) features from DRED.
/// Matches C `lpcnet_plc_fec_add`.
pub fn lpcnet_plc_fec_add(st: &mut LpcnetPlcState, features: Option<&[f32]>) {
    match features {
        None => {
            st.fec_skip += 1;
        }
        Some(f) => {
            debug_assert!(st.fec_fill_pos < PLC_MAX_FEC);
            let off = st.fec_fill_pos * NB_FEATURES;
            st.fec[off..off + NB_FEATURES].copy_from_slice(&f[..NB_FEATURES]);
            st.fec_fill_pos += 1;
        }
    }
}

/// Clear FEC buffer. Matches C `lpcnet_plc_fec_clear`.
pub fn lpcnet_plc_fec_clear(st: &mut LpcnetPlcState) {
    st.fec_read_pos = 0;
    st.fec_fill_pos = 0;
    st.fec_skip = 0;
}

fn compute_plc_pred(st: &mut LpcnetPlcState, out: &mut [f32], input: &[f32]) {
    debug_assert!(st.loaded);
    let model = &st.model;
    let dense_out = model.plc_dense_in.nb_outputs;
    let mut tmp = [0.0f32; 256];
    compute_generic_dense(
        &model.plc_dense_in,
        &mut tmp[..dense_out],
        input,
        Activation::Tanh,
    );
    compute_generic_gru(
        &model.plc_gru1_input,
        &model.plc_gru1_recurrent,
        &mut st.plc_net.gru1_state,
        &tmp[..dense_out],
    );
    let PlcNetState {
        gru1_state,
        gru2_state,
    } = &mut st.plc_net;
    compute_generic_gru(
        &model.plc_gru2_input,
        &model.plc_gru2_recurrent,
        gru2_state,
        gru1_state,
    );
    let out_size = model.plc_dense_out.nb_outputs;
    compute_generic_dense(
        &model.plc_dense_out,
        &mut out[..out_size],
        &st.plc_net.gru2_state,
        Activation::Linear,
    );
}

fn get_fec_or_pred(st: &mut LpcnetPlcState) -> bool {
    if st.fec_read_pos != st.fec_fill_pos && st.fec_skip == 0 {
        let mut plc_features = [0.0f32; 2 * NB_BANDS + NB_FEATURES + 1];
        let mut features = [0.0f32; NB_FEATURES];
        let off = st.fec_read_pos * NB_FEATURES;
        features.copy_from_slice(&st.fec[off..off + NB_FEATURES]);
        st.fec_read_pos += 1;
        // Update PLC state using FEC (without Burg features)
        plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES].copy_from_slice(&features);
        plc_features[2 * NB_BANDS + NB_FEATURES] = -1.0;
        let mut discard = [0.0f32; NB_FEATURES];
        compute_plc_pred(st, &mut discard, &plc_features);
        st.features[..NB_FEATURES].copy_from_slice(&features);
        true
    } else {
        let zeros = [0.0f32; 2 * NB_BANDS + NB_FEATURES + 1];
        let mut out = [0.0f32; NB_FEATURES];
        compute_plc_pred(st, &mut out, &zeros);
        st.features[..NB_FEATURES].copy_from_slice(&out);
        if st.fec_skip > 0 {
            st.fec_skip -= 1;
        }
        false
    }
}

fn queue_features(cont_features: &mut [f32], features: &[f32]) {
    let nf = NB_FEATURES;
    cont_features.copy_within(nf.., 0);
    let start = (CONT_VECTORS - 1) * nf;
    cont_features[start..start + nf].copy_from_slice(&features[..nf]);
}

/// Update PLC state with a good (received) packet.
/// Matches C `lpcnet_plc_update`.
pub fn lpcnet_plc_update(st: &mut LpcnetPlcState, pcm: &[i16]) {
    if st.analysis_pos >= FRAME_SIZE {
        st.analysis_pos -= FRAME_SIZE;
    } else {
        st.analysis_gap = true;
    }
    if st.predict_pos >= FRAME_SIZE {
        st.predict_pos -= FRAME_SIZE;
    }
    st.pcm.copy_within(FRAME_SIZE.., 0);
    for (i, &p) in pcm.iter().enumerate().take(FRAME_SIZE) {
        st.pcm[PLC_BUF_SIZE - FRAME_SIZE + i] = (1.0 / 32768.0) * p as f32;
    }
    st.loss_count = 0;
    st.blend = false;
}

/// Conceal a lost packet by synthesizing replacement audio.
/// Matches C `lpcnet_plc_conceal`.
pub fn lpcnet_plc_conceal(st: &mut LpcnetPlcState, pcm: &mut [i16]) {
    debug_assert!(st.loaded);

    if !st.blend {
        let mut count = 0;
        st.plc_net = st.plc_bak[0].clone();
        while st.analysis_pos + FRAME_SIZE <= PLC_BUF_SIZE {
            let mut x = [0.0f32; FRAME_SIZE];
            let mut plc_features = [0.0f32; 2 * NB_BANDS + NB_FEATURES + 1];
            for (i, x_val) in x.iter_mut().enumerate().take(FRAME_SIZE) {
                *x_val = 32768.0 * st.pcm[st.analysis_pos + i];
            }
            burg_cepstral_analysis(&mut plc_features, &x);
            let x_copy = x;
            preemphasis(
                &mut x,
                &mut st.enc.mem_preemph,
                &x_copy,
                PREEMPHASIS,
                FRAME_SIZE,
            );
            compute_frame_features(&mut st.enc, &x);

            if (!st.analysis_gap || count > 0) && st.analysis_pos >= st.predict_pos {
                queue_features(&mut st.cont_features, &st.enc.features);
                plc_features[2 * NB_BANDS..2 * NB_BANDS + NB_FEATURES]
                    .copy_from_slice(&st.enc.features[..NB_FEATURES]);
                plc_features[2 * NB_BANDS + NB_FEATURES] = 1.0;
                st.plc_bak[0] = st.plc_bak[1].clone();
                st.plc_bak[1] = st.plc_net.clone();
                let mut pred_out = [0.0f32; NB_TOTAL_FEATURES];
                compute_plc_pred(st, &mut pred_out, &plc_features);
                st.features = pred_out;
            }
            st.analysis_pos += FRAME_SIZE;
            count += 1;
        }

        st.plc_bak[0] = st.plc_bak[1].clone();
        st.plc_bak[1] = st.plc_net.clone();
        get_fec_or_pred(st);
        let features_snap = st.features;
        queue_features(&mut st.cont_features, &features_snap);

        st.plc_bak[0] = st.plc_bak[1].clone();
        st.plc_bak[1] = st.plc_net.clone();
        get_fec_or_pred(st);
        let features_snap = st.features;
        queue_features(&mut st.cont_features, &features_snap);

        let cont_start = PLC_BUF_SIZE - FARGAN_CONT_SAMPLES;
        fargan_cont(&mut st.fargan, &st.pcm[cont_start..], &st.cont_features);
        st.analysis_gap = false;
    }

    st.plc_bak[0] = st.plc_bak[1].clone();
    st.plc_bak[1] = st.plc_net.clone();
    if get_fec_or_pred(st) {
        st.loss_count = 0;
    } else {
        st.loss_count += 1;
    }

    if st.loss_count >= 10 {
        st.features[0] =
            (-15.0f32).max(st.features[0] + ATT_TABLE[9] - 2.0 * (st.loss_count as f32 - 9.0));
    } else {
        st.features[0] = (-15.0f32).max(st.features[0] + ATT_TABLE[st.loss_count]);
    }

    let features_copy: [f32; NB_TOTAL_FEATURES] = st.features;
    fargan_synthesize_int(&mut st.fargan, pcm, &features_copy);

    let features_snap = st.features;
    queue_features(&mut st.cont_features, &features_snap);

    if st.analysis_pos >= FRAME_SIZE {
        st.analysis_pos -= FRAME_SIZE;
    } else {
        st.analysis_gap = true;
    }
    st.predict_pos = PLC_BUF_SIZE;
    st.pcm.copy_within(FRAME_SIZE.., 0);
    for (i, &p) in pcm.iter().enumerate().take(FRAME_SIZE) {
        st.pcm[PLC_BUF_SIZE - FRAME_SIZE + i] = (1.0 / 32768.0) * p as f32;
    }
    st.blend = true;
}
