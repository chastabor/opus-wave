use crate::freq::NB_BANDS;
use crate::nnet::ops::{
    compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu,
    compute_glu_inplace,
};
use crate::nnet::weights::{WeightError, linear_init};
use crate::nnet::{Activation, LinearLayer, WeightArray};

use crate::pitchdnn::{PITCH_MAX_PERIOD, pitch_period_from_dnn};

/// Number of features per frame used by FARGAN (cepstral + pitch + corr).
pub const NB_FEATURES: usize = 20;

pub const FARGAN_CONT_SAMPLES: usize = 320;
pub const FARGAN_NB_SUBFRAMES: usize = 4;
pub const FARGAN_SUBFRAME_SIZE: usize = 40;
pub const FARGAN_FRAME_SIZE: usize = FARGAN_NB_SUBFRAMES * FARGAN_SUBFRAME_SIZE;
pub const FARGAN_DEEMPHASIS: f32 = 0.85;

/// Maximum layer sizes for stack allocation.
const MAX_COND_SIZE: usize = 512;
const MAX_FWC0_IN_SIZE: usize = 512;
const MAX_SKIP_SIZE: usize = 1024;

/// FARGAN model: all neural network layers for the vocoder.
/// Matches the auto-generated C `struct FARGAN` from fargan_data.h.
pub struct Fargan {
    pub cond_net_pembed: LinearLayer,
    pub cond_net_fdense1: LinearLayer,
    pub cond_net_fconv1: LinearLayer,
    pub cond_net_fdense2: LinearLayer,
    pub sig_net_cond_gain_dense: LinearLayer,
    pub sig_net_fwc0_conv: LinearLayer,
    pub sig_net_fwc0_glu_gate: LinearLayer,
    pub sig_net_gain_dense_out: LinearLayer,
    pub sig_net_gru1_input: LinearLayer,
    pub sig_net_gru1_recurrent: LinearLayer,
    pub sig_net_gru1_glu_gate: LinearLayer,
    pub sig_net_gru2_input: LinearLayer,
    pub sig_net_gru2_recurrent: LinearLayer,
    pub sig_net_gru2_glu_gate: LinearLayer,
    pub sig_net_gru3_input: LinearLayer,
    pub sig_net_gru3_recurrent: LinearLayer,
    pub sig_net_gru3_glu_gate: LinearLayer,
    pub sig_net_skip_dense: LinearLayer,
    pub sig_net_skip_glu_gate: LinearLayer,
    pub sig_net_sig_dense_out: LinearLayer,
}

/// FARGAN state: model + recurrent states + buffers.
/// Matches C `FARGANState` from fargan.h.
pub struct FarganState {
    pub model: Fargan,
    pub cont_initialized: bool,
    pub deemph_mem: f32,
    pub pitch_buf: [f32; PITCH_MAX_PERIOD],
    pub cond_conv1_state: Vec<f32>,
    pub fwc0_mem: Vec<f32>,
    pub gru1_state: Vec<f32>,
    pub gru2_state: Vec<f32>,
    pub gru3_state: Vec<f32>,
    pub last_period: usize,
    /// Cached dimension: conditioning vector size per subframe.
    cond_size: usize,
    /// Cached dimension: total conditioning output.
    total_cond_size: usize,
    /// Pitch embedding dimension.
    pembed_size: usize,
}

/// Initialize FARGAN model from weight arrays.
/// Dimensions match C `fargan_init` from fargan_data.c exactly.
pub fn init_fargan(arrays: &[WeightArray]) -> Result<Fargan, WeightError> {
    Ok(Fargan {
        // linear_init(arrays, bias, weights_int8, float_weights, weights_idx, diag, scale, nb_inputs, nb_outputs)
        // C: linear_init(&model->cond_net_pembed, ..., 224, 12)
        cond_net_pembed: linear_init(
            arrays,
            Some("cond_net_pembed_bias"),
            None,
            Some("cond_net_pembed_weights"),
            None,
            None,
            None,
            224,
            12,
        )?,
        // C: linear_init(&model->cond_net_fdense1, ..., 32, 64)
        cond_net_fdense1: linear_init(
            arrays,
            Some("cond_net_fdense1_bias"),
            None,
            Some("cond_net_fdense1_weights"),
            None,
            None,
            None,
            32,
            64,
        )?,
        // C: linear_init(&model->cond_net_fconv1, ..., int8+scale, 192, 128)
        cond_net_fconv1: linear_init(
            arrays,
            Some("cond_net_fconv1_bias"),
            Some("cond_net_fconv1_weights"),
            Some("cond_net_fconv1_weights"),
            None,
            None,
            Some("cond_net_fconv1_scale"),
            192,
            128,
        )?,
        // C: linear_init(&model->cond_net_fdense2, ..., int8+scale, 128, 320)
        cond_net_fdense2: linear_init(
            arrays,
            Some("cond_net_fdense2_bias"),
            Some("cond_net_fdense2_weights"),
            Some("cond_net_fdense2_weights"),
            None,
            None,
            Some("cond_net_fdense2_scale"),
            128,
            320,
        )?,
        // C: linear_init(&model->sig_net_cond_gain_dense, ..., 80, 1)
        sig_net_cond_gain_dense: linear_init(
            arrays,
            Some("sig_net_cond_gain_dense_bias"),
            None,
            Some("sig_net_cond_gain_dense_weights"),
            None,
            None,
            None,
            80,
            1,
        )?,
        // C: linear_init(&model->sig_net_fwc0_conv, ..., int8+scale, 328, 192)
        sig_net_fwc0_conv: linear_init(
            arrays,
            Some("sig_net_fwc0_conv_bias"),
            Some("sig_net_fwc0_conv_weights"),
            Some("sig_net_fwc0_conv_weights"),
            None,
            None,
            Some("sig_net_fwc0_conv_scale"),
            328,
            192,
        )?,
        // C: linear_init(&model->sig_net_fwc0_glu_gate, ..., int8+scale, 192, 192)
        sig_net_fwc0_glu_gate: linear_init(
            arrays,
            Some("sig_net_fwc0_glu_gate_bias"),
            Some("sig_net_fwc0_glu_gate_weights"),
            Some("sig_net_fwc0_glu_gate_weights"),
            None,
            None,
            Some("sig_net_fwc0_glu_gate_scale"),
            192,
            192,
        )?,
        // C: linear_init(&model->sig_net_gain_dense_out, ..., 192, 4)
        sig_net_gain_dense_out: linear_init(
            arrays,
            Some("sig_net_gain_dense_out_bias"),
            None,
            Some("sig_net_gain_dense_out_weights"),
            None,
            None,
            None,
            192,
            4,
        )?,
        // C: linear_init(&model->sig_net_gru1_input, ..., bias=NULL, int8+scale, 272, 480)
        sig_net_gru1_input: linear_init(
            arrays,
            None,
            Some("sig_net_gru1_input_weights"),
            Some("sig_net_gru1_input_weights"),
            None,
            None,
            Some("sig_net_gru1_input_scale"),
            272,
            480,
        )?,
        // C: linear_init(&model->sig_net_gru1_recurrent, ..., bias=NULL, int8+scale, 160, 480)
        sig_net_gru1_recurrent: linear_init(
            arrays,
            None,
            Some("sig_net_gru1_recurrent_weights"),
            Some("sig_net_gru1_recurrent_weights"),
            None,
            None,
            Some("sig_net_gru1_recurrent_scale"),
            160,
            480,
        )?,
        // C: linear_init(&model->sig_net_gru1_glu_gate, ..., int8+scale, 160, 160)
        sig_net_gru1_glu_gate: linear_init(
            arrays,
            Some("sig_net_gru1_glu_gate_bias"),
            Some("sig_net_gru1_glu_gate_weights"),
            Some("sig_net_gru1_glu_gate_weights"),
            None,
            None,
            Some("sig_net_gru1_glu_gate_scale"),
            160,
            160,
        )?,
        // C: linear_init(&model->sig_net_gru2_input, ..., bias=NULL, int8+scale, 240, 384)
        sig_net_gru2_input: linear_init(
            arrays,
            None,
            Some("sig_net_gru2_input_weights"),
            Some("sig_net_gru2_input_weights"),
            None,
            None,
            Some("sig_net_gru2_input_scale"),
            240,
            384,
        )?,
        // C: linear_init(&model->sig_net_gru2_recurrent, ..., bias=NULL, int8+scale, 128, 384)
        sig_net_gru2_recurrent: linear_init(
            arrays,
            None,
            Some("sig_net_gru2_recurrent_weights"),
            Some("sig_net_gru2_recurrent_weights"),
            None,
            None,
            Some("sig_net_gru2_recurrent_scale"),
            128,
            384,
        )?,
        // C: linear_init(&model->sig_net_gru2_glu_gate, ..., int8+scale, 128, 128)
        sig_net_gru2_glu_gate: linear_init(
            arrays,
            Some("sig_net_gru2_glu_gate_bias"),
            Some("sig_net_gru2_glu_gate_weights"),
            Some("sig_net_gru2_glu_gate_weights"),
            None,
            None,
            Some("sig_net_gru2_glu_gate_scale"),
            128,
            128,
        )?,
        // C: linear_init(&model->sig_net_gru3_input, ..., bias=NULL, int8+scale, 208, 384)
        sig_net_gru3_input: linear_init(
            arrays,
            None,
            Some("sig_net_gru3_input_weights"),
            Some("sig_net_gru3_input_weights"),
            None,
            None,
            Some("sig_net_gru3_input_scale"),
            208,
            384,
        )?,
        // C: linear_init(&model->sig_net_gru3_recurrent, ..., bias=NULL, int8+scale, 128, 384)
        sig_net_gru3_recurrent: linear_init(
            arrays,
            None,
            Some("sig_net_gru3_recurrent_weights"),
            Some("sig_net_gru3_recurrent_weights"),
            None,
            None,
            Some("sig_net_gru3_recurrent_scale"),
            128,
            384,
        )?,
        // C: linear_init(&model->sig_net_gru3_glu_gate, ..., int8+scale, 128, 128)
        sig_net_gru3_glu_gate: linear_init(
            arrays,
            Some("sig_net_gru3_glu_gate_bias"),
            Some("sig_net_gru3_glu_gate_weights"),
            Some("sig_net_gru3_glu_gate_weights"),
            None,
            None,
            Some("sig_net_gru3_glu_gate_scale"),
            128,
            128,
        )?,
        // C: linear_init(&model->sig_net_skip_dense, ..., int8+scale, 688, 128)
        sig_net_skip_dense: linear_init(
            arrays,
            Some("sig_net_skip_dense_bias"),
            Some("sig_net_skip_dense_weights"),
            Some("sig_net_skip_dense_weights"),
            None,
            None,
            Some("sig_net_skip_dense_scale"),
            688,
            128,
        )?,
        // C: linear_init(&model->sig_net_skip_glu_gate, ..., int8+scale, 128, 128)
        sig_net_skip_glu_gate: linear_init(
            arrays,
            Some("sig_net_skip_glu_gate_bias"),
            Some("sig_net_skip_glu_gate_weights"),
            Some("sig_net_skip_glu_gate_weights"),
            None,
            None,
            Some("sig_net_skip_glu_gate_scale"),
            128,
            128,
        )?,
        // C: linear_init(&model->sig_net_sig_dense_out, ..., int8+scale, 128, 40)
        sig_net_sig_dense_out: linear_init(
            arrays,
            Some("sig_net_sig_dense_out_bias"),
            Some("sig_net_sig_dense_out_weights"),
            Some("sig_net_sig_dense_out_weights"),
            None,
            None,
            Some("sig_net_sig_dense_out_scale"),
            128,
            40,
        )?,
    })
}

/// Initialize FARGAN state from a loaded model.
pub fn fargan_state_init(model: Fargan) -> FarganState {
    let total_cond_size = model.cond_net_fdense2.nb_outputs;
    let cond_size = total_cond_size / FARGAN_NB_SUBFRAMES;
    let pembed_size = model.cond_net_pembed.nb_outputs;
    let fconv1_in = model.cond_net_fconv1.nb_inputs;
    let sig_input_size = cond_size + 2 * FARGAN_SUBFRAME_SIZE + 4;
    let gru1_out = model.sig_net_gru1_recurrent.nb_inputs;
    let gru2_out = model.sig_net_gru2_recurrent.nb_inputs;
    let gru3_out = model.sig_net_gru3_recurrent.nb_inputs;

    FarganState {
        model,
        cont_initialized: false,
        deemph_mem: 0.0,
        pitch_buf: [0.0; PITCH_MAX_PERIOD],
        cond_conv1_state: vec![0.0; fconv1_in],
        fwc0_mem: vec![0.0; 2 * sig_input_size],
        gru1_state: vec![0.0; gru1_out],
        gru2_state: vec![0.0; gru2_out],
        gru3_state: vec![0.0; gru3_out],
        last_period: 0,
        cond_size,
        total_cond_size,
        pembed_size,
    }
}

fn compute_fargan_cond(st: &mut FarganState, cond: &mut [f32], features: &[f32], period: usize) {
    let model = &st.model;
    let fdense1_out = model.cond_net_fdense1.nb_outputs;
    let fconv1_out = model.cond_net_fconv1.nb_outputs;

    let pembed_idx = period.saturating_sub(32).min(223);
    let pembed_weights = model.cond_net_pembed.float_weights.as_ref().unwrap();
    let pembed_start = pembed_idx * st.pembed_size;

    let mut dense_in = [0.0f32; MAX_COND_SIZE];
    dense_in[..NB_FEATURES].copy_from_slice(&features[..NB_FEATURES]);
    dense_in[NB_FEATURES..NB_FEATURES + st.pembed_size]
        .copy_from_slice(&pembed_weights[pembed_start..pembed_start + st.pembed_size]);

    let mut conv1_in = [0.0f32; MAX_COND_SIZE];
    compute_generic_dense(
        &model.cond_net_fdense1,
        &mut conv1_in[..fdense1_out],
        &dense_in[..NB_FEATURES + st.pembed_size],
        Activation::Tanh,
    );

    let mut fdense2_in = [0.0f32; MAX_COND_SIZE];
    compute_generic_conv1d(
        &model.cond_net_fconv1,
        &mut fdense2_in[..fconv1_out],
        &mut st.cond_conv1_state,
        &conv1_in[..fdense1_out],
        fdense1_out,
        Activation::Tanh,
    );

    compute_generic_dense(
        &model.cond_net_fdense2,
        &mut cond[..st.total_cond_size],
        &fdense2_in[..fconv1_out],
        Activation::Tanh,
    );
}

fn fargan_deemphasis(pcm: &mut [f32], deemph_mem: &mut f32) {
    for sample in pcm[..FARGAN_SUBFRAME_SIZE].iter_mut() {
        *sample += FARGAN_DEEMPHASIS * *deemph_mem;
        *deemph_mem = *sample;
    }
}

fn run_fargan_subframe(st: &mut FarganState, pcm: &mut [f32], cond: &[f32], period: usize) {
    debug_assert!(st.cont_initialized);
    let model = &st.model;
    let cond_size = st.cond_size;
    let gru1_out = model.sig_net_gru1_recurrent.nb_inputs;
    let gru2_out = model.sig_net_gru2_recurrent.nb_inputs;
    let gru3_out = model.sig_net_gru3_recurrent.nb_inputs;
    let fwc0_out = model.sig_net_fwc0_conv.nb_outputs;
    let skip_dense_out = model.sig_net_skip_dense.nb_outputs;
    let sig_input_size = cond_size + 2 * FARGAN_SUBFRAME_SIZE + 4;

    let mut gain = [0.0f32; 1];
    compute_generic_dense(
        &model.sig_net_cond_gain_dense,
        &mut gain,
        &cond[..cond_size],
        Activation::Linear,
    );
    let gain = gain[0].exp();
    let gain_1 = 1.0 / (1e-5 + gain);

    // Compute pitch prediction and previous samples
    let mut pred = [0.0f32; FARGAN_SUBFRAME_SIZE + 4];
    let mut pos = (PITCH_MAX_PERIOD as isize - period as isize - 2).max(0) as usize;
    for p in pred.iter_mut() {
        *p = (gain_1 * st.pitch_buf[pos.min(PITCH_MAX_PERIOD - 1)]).clamp(-1.0, 1.0);
        pos += 1;
        if pos == PITCH_MAX_PERIOD {
            pos -= period;
        }
    }
    let mut prev = [0.0f32; FARGAN_SUBFRAME_SIZE];
    for (i, prev_val) in prev.iter_mut().enumerate() {
        *prev_val =
            (gain_1 * st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE + i]).clamp(-1.0, 1.0);
    }

    // Build FWC0 input: [cond | pred | prev]
    let mut fwc0_in = [0.0f32; MAX_FWC0_IN_SIZE];
    fwc0_in[..cond_size].copy_from_slice(&cond[..cond_size]);
    fwc0_in[cond_size..cond_size + FARGAN_SUBFRAME_SIZE + 4].copy_from_slice(&pred);
    fwc0_in[cond_size + FARGAN_SUBFRAME_SIZE + 4..sig_input_size].copy_from_slice(&prev);

    let mut gru1_in = [0.0f32; MAX_SKIP_SIZE];
    compute_generic_conv1d(
        &model.sig_net_fwc0_conv,
        &mut gru1_in[..fwc0_out],
        &mut st.fwc0_mem,
        &fwc0_in[..sig_input_size],
        sig_input_size,
        Activation::Tanh,
    );
    compute_glu_inplace(&model.sig_net_fwc0_glu_gate, &mut gru1_in[..fwc0_out]);

    let mut pitch_gate = [0.0f32; 4];
    compute_generic_dense(
        &model.sig_net_gain_dense_out,
        &mut pitch_gate,
        &gru1_in[..fwc0_out],
        Activation::Sigmoid,
    );

    // GRU 1
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru1_in[fwc0_out + i] = pitch_gate[0] * pred[i + 2];
    }
    gru1_in[fwc0_out + FARGAN_SUBFRAME_SIZE..fwc0_out + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru1_input,
        &model.sig_net_gru1_recurrent,
        &mut st.gru1_state,
        &gru1_in[..fwc0_out + 2 * FARGAN_SUBFRAME_SIZE],
    );
    let mut gru2_in = [0.0f32; MAX_SKIP_SIZE];
    compute_glu(
        &model.sig_net_gru1_glu_gate,
        &mut gru2_in[..gru1_out],
        &st.gru1_state,
    );

    // GRU 2
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru2_in[gru1_out + i] = pitch_gate[1] * pred[i + 2];
    }
    gru2_in[gru1_out + FARGAN_SUBFRAME_SIZE..gru1_out + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru2_input,
        &model.sig_net_gru2_recurrent,
        &mut st.gru2_state,
        &gru2_in[..gru1_out + 2 * FARGAN_SUBFRAME_SIZE],
    );
    let mut gru3_in = [0.0f32; MAX_SKIP_SIZE];
    compute_glu(
        &model.sig_net_gru2_glu_gate,
        &mut gru3_in[..gru2_out],
        &st.gru2_state,
    );

    // GRU 3
    for i in 0..FARGAN_SUBFRAME_SIZE {
        gru3_in[gru2_out + i] = pitch_gate[2] * pred[i + 2];
    }
    gru3_in[gru2_out + FARGAN_SUBFRAME_SIZE..gru2_out + 2 * FARGAN_SUBFRAME_SIZE]
        .copy_from_slice(&prev);
    compute_generic_gru(
        &model.sig_net_gru3_input,
        &model.sig_net_gru3_recurrent,
        &mut st.gru3_state,
        &gru3_in[..gru2_out + 2 * FARGAN_SUBFRAME_SIZE],
    );

    // Skip connections: [gru1_glu_out | gru2_glu_out | gru3_glu_out | fwc0_glu_out | pitch_gate[3]*pred | prev]
    let mut skip_cat = [0.0f32; MAX_SKIP_SIZE];
    skip_cat[..gru1_out].copy_from_slice(&gru2_in[..gru1_out]);
    skip_cat[gru1_out..gru1_out + gru2_out].copy_from_slice(&gru3_in[..gru2_out]);
    let gru3_glu_off = gru1_out + gru2_out;
    compute_glu(
        &model.sig_net_gru3_glu_gate,
        &mut skip_cat[gru3_glu_off..gru3_glu_off + gru3_out],
        &st.gru3_state,
    );
    let fwc_off = gru3_glu_off + gru3_out;
    skip_cat[fwc_off..fwc_off + fwc0_out].copy_from_slice(&gru1_in[..fwc0_out]);
    let pred_off = fwc_off + fwc0_out;
    for i in 0..FARGAN_SUBFRAME_SIZE {
        skip_cat[pred_off + i] = pitch_gate[3] * pred[i + 2];
    }
    let prev_off = pred_off + FARGAN_SUBFRAME_SIZE;
    skip_cat[prev_off..prev_off + FARGAN_SUBFRAME_SIZE].copy_from_slice(&prev);

    let skip_in_size = prev_off + FARGAN_SUBFRAME_SIZE;
    let mut skip_out = [0.0f32; MAX_SKIP_SIZE];
    compute_generic_dense(
        &model.sig_net_skip_dense,
        &mut skip_out[..skip_dense_out],
        &skip_cat[..skip_in_size],
        Activation::Tanh,
    );
    compute_glu_inplace(
        &model.sig_net_skip_glu_gate,
        &mut skip_out[..skip_dense_out],
    );

    compute_generic_dense(
        &model.sig_net_sig_dense_out,
        &mut pcm[..FARGAN_SUBFRAME_SIZE],
        &skip_out[..skip_dense_out],
        Activation::Tanh,
    );
    for sample in pcm[..FARGAN_SUBFRAME_SIZE].iter_mut() {
        *sample *= gain;
    }

    // Update pitch buffer
    st.pitch_buf
        .copy_within(FARGAN_SUBFRAME_SIZE..PITCH_MAX_PERIOD, 0);
    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..PITCH_MAX_PERIOD]
        .copy_from_slice(&pcm[..FARGAN_SUBFRAME_SIZE]);
    fargan_deemphasis(pcm, &mut st.deemph_mem);
}

/// Pre-load FARGAN state from previous PCM and features for continuation.
/// Matches C `fargan_cont` from fargan.c.
pub fn fargan_cont(st: &mut FarganState, pcm0: &[f32], features0: &[f32]) {
    let mut cond = [0.0f32; MAX_COND_SIZE];
    let mut period = 0usize;

    // Pre-load features (5 frames worth of conditioning).
    for i in 0..5 {
        let features = &features0[i * NB_FEATURES..];
        st.last_period = period;
        period = pitch_period_from_dnn(features[NB_BANDS]);
        compute_fargan_cond(st, &mut cond, features, period);
    }

    // Pre-emphasis on continuation samples
    let mut x0 = [0.0f32; FARGAN_CONT_SAMPLES];
    for i in 1..FARGAN_CONT_SAMPLES {
        x0[i] = pcm0[i] - FARGAN_DEEMPHASIS * pcm0[i - 1];
    }

    st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_FRAME_SIZE..PITCH_MAX_PERIOD]
        .copy_from_slice(&x0[..FARGAN_FRAME_SIZE]);
    st.cont_initialized = true;

    // Warm up subframes
    let cond_size = st.cond_size;
    let mut dummy = [0.0f32; FARGAN_SUBFRAME_SIZE];
    for i in 0..FARGAN_NB_SUBFRAMES {
        run_fargan_subframe(
            st,
            &mut dummy,
            &cond[i * cond_size..(i + 1) * cond_size],
            st.last_period,
        );
        st.pitch_buf[PITCH_MAX_PERIOD - FARGAN_SUBFRAME_SIZE..PITCH_MAX_PERIOD].copy_from_slice(
            &x0[FARGAN_FRAME_SIZE + i * FARGAN_SUBFRAME_SIZE
                ..FARGAN_FRAME_SIZE + (i + 1) * FARGAN_SUBFRAME_SIZE],
        );
    }
    st.deemph_mem = pcm0[FARGAN_CONT_SAMPLES - 1];
}

/// Synthesize one frame of audio from features.
/// Matches C `fargan_synthesize` from fargan.c.
pub fn fargan_synthesize(st: &mut FarganState, pcm: &mut [f32], features: &[f32]) {
    debug_assert!(st.cont_initialized);
    let mut cond = [0.0f32; MAX_COND_SIZE];
    let period = pitch_period_from_dnn(features[NB_BANDS]);
    compute_fargan_cond(st, &mut cond, features, period);
    let cond_size = st.cond_size;
    for subframe in 0..FARGAN_NB_SUBFRAMES {
        run_fargan_subframe(
            st,
            &mut pcm[subframe * FARGAN_SUBFRAME_SIZE..],
            &cond[subframe * cond_size..(subframe + 1) * cond_size],
            st.last_period,
        );
    }
    st.last_period = period;
}

/// Synthesize one frame as i16 samples.
/// Matches C `fargan_synthesize_int` from fargan.c.
pub fn fargan_synthesize_int(st: &mut FarganState, pcm: &mut [i16], features: &[f32]) {
    let mut fpcm = [0.0f32; FARGAN_FRAME_SIZE];
    fargan_synthesize(st, &mut fpcm, features);
    for i in 0..FARGAN_FRAME_SIZE {
        pcm[i] = (0.5 + (32768.0 * fpcm[i]).clamp(-32767.0, 32767.0)).floor() as i16;
    }
}
