use opus_celt::fft::KissFftCpx;

use crate::freq::*;
use crate::pitchdnn::*;

/// Total features per frame: NB_BANDS cepstral + pitch + corr + LPC_ORDER LPC coefs.
pub const NB_TOTAL_FEATURES: usize = NB_BANDS + 2 + LPC_ORDER;

/// LPCNet encoder state for feature extraction.
/// Matches C `LPCNetEncState` from lpcnet_private.h.
pub struct LpcnetEncState {
    pub pitchdnn: PitchDnnState,
    pub analysis_mem: [f32; OVERLAP_SIZE],
    pub prev_if: [KissFftCpx; PITCH_IF_MAX_FREQ],
    pub if_features: [f32; NB_IF_FEATURES],
    pub xcorr_features: [f32; NB_XCORR_FEATURES],
    pub features: [f32; NB_TOTAL_FEATURES],
    pub lpc: [f32; LPC_ORDER],
    pub exc_buf: Vec<f32>,
    pub lp_buf: Vec<f32>,
    pub pitch_mem: [f32; LPC_ORDER],
    pub pitch_filt: f32,
    pub lp_mem: [f32; 2],
    pub mem_preemph: f32,
    pub dnn_pitch: f32,
}

impl LpcnetEncState {
    pub fn new(pitchdnn_state: PitchDnnState) -> Self {
        LpcnetEncState {
            pitchdnn: pitchdnn_state,
            analysis_mem: [0.0; OVERLAP_SIZE],
            prev_if: [KissFftCpx::default(); PITCH_IF_MAX_FREQ],
            if_features: [0.0; NB_IF_FEATURES],
            xcorr_features: [0.0; NB_XCORR_FEATURES],
            features: [0.0; NB_TOTAL_FEATURES],
            lpc: [0.0; LPC_ORDER],
            exc_buf: vec![0.0; PITCH_MAX_PERIOD + FRAME_SIZE],
            lp_buf: vec![0.0; PITCH_MAX_PERIOD + FRAME_SIZE],
            pitch_mem: [0.0; LPC_ORDER],
            pitch_filt: 0.0,
            lp_mem: [0.0; 2],
            mem_preemph: 0.0,
            dnn_pitch: 0.0,
        }
    }
}

fn frame_analysis(
    st: &mut LpcnetEncState,
    x_out: &mut [KissFftCpx],
    ex: &mut [f32; NB_BANDS],
    input: &[f32],
) {
    let mut x = [0.0f32; WINDOW_SIZE];
    x[..OVERLAP_SIZE].copy_from_slice(&st.analysis_mem);
    x[OVERLAP_SIZE..OVERLAP_SIZE + FRAME_SIZE].copy_from_slice(&input[..FRAME_SIZE]);
    st.analysis_mem
        .copy_from_slice(&input[FRAME_SIZE - OVERLAP_SIZE..FRAME_SIZE]);
    apply_window(&mut x);
    forward_transform(x_out, &x);
    lpcn_compute_band_energy(ex, x_out);
}

fn biquad(y: &mut [f32], mem: &mut [f32; 2], x: &[f32], b: &[f32; 2], a: &[f32; 2], n: usize) {
    let mut mem0 = mem[0];
    let mut mem1 = mem[1];
    for i in 0..n {
        let xi = x[i];
        let yi = xi + mem0;
        let mem00 = mem0;
        mem0 = (b[0] - a[0]) * xi + mem1 - a[0] * mem0;
        mem1 = (b[1] - a[1]) * xi + 1e-30 - a[1] * mem00;
        y[i] = yi;
    }
    mem[0] = mem0;
    mem[1] = mem1;
}

pub fn preemphasis(y: &mut [f32], mem: &mut f32, x: &[f32], coef: f32, n: usize) {
    for i in 0..n {
        let yi = x[i] + *mem;
        *mem = -coef * x[i];
        y[i] = yi;
    }
}

/// Compute frame features from pre-emphasized input.
/// Matches C `compute_frame_features` from lpcnet_enc.c.
pub fn compute_frame_features(st: &mut LpcnetEncState, input: &[f32]) {
    const LP_B: [f32; 2] = [-0.84946, 1.0];
    const LP_A: [f32; 2] = [-1.54220, 0.70781];

    let mut aligned_in = [0.0f32; FRAME_SIZE];
    aligned_in[..TRAINING_OFFSET]
        .copy_from_slice(&st.analysis_mem[OVERLAP_SIZE - TRAINING_OFFSET..OVERLAP_SIZE]);

    let mut x_fft = [KissFftCpx::default(); FREQ_SIZE];
    let mut ex = [0.0f32; NB_BANDS];
    frame_analysis(st, &mut x_fft, &mut ex, input);

    // Instantaneous frequency features
    st.if_features[0] = ((1.0 / 64.0)
        * (10.0 * opus_celt::mathops::celt_log10(1e-15 + x_fft[0].r * x_fft[0].r) - 6.0))
        .clamp(-1.0, 1.0);
    for (i, xf) in x_fft.iter().enumerate().take(PITCH_IF_MAX_FREQ).skip(1) {
        let prod_r = xf.r * st.prev_if[i].r + xf.i * st.prev_if[i].i;
        let prod_i = xf.i * st.prev_if[i].r - xf.r * st.prev_if[i].i;
        let norm_1 = 1.0 / (1e-15 + prod_r * prod_r + prod_i * prod_i).sqrt();
        st.if_features[3 * i - 2] = prod_r * norm_1;
        st.if_features[3 * i - 1] = prod_i * norm_1;
        st.if_features[3 * i] = ((1.0 / 64.0)
            * (10.0 * opus_celt::mathops::celt_log10(1e-15 + xf.r * xf.r + xf.i * xf.i) - 6.0))
            .clamp(-1.0, 1.0);
    }
    st.prev_if[..PITCH_IF_MAX_FREQ].copy_from_slice(&x_fft[..PITCH_IF_MAX_FREQ]);

    // Band energy to cepstral features
    let mut ly = [0.0f32; NB_BANDS];
    let mut log_max = -2.0f32;
    let mut follow = -2.0f32;
    for i in 0..NB_BANDS {
        ly[i] = opus_celt::mathops::celt_log10(1e-2 + ex[i]);
        ly[i] = ly[i].max(log_max - 8.0).max(follow - 2.5);
        log_max = log_max.max(ly[i]);
        follow = (follow - 2.5).max(ly[i]);
    }
    let mut ceps = [0.0f32; NB_BANDS];
    dct(&mut ceps, &ly);
    st.features[..NB_BANDS].copy_from_slice(&ceps);
    st.features[0] -= 4.0;

    lpc_from_cepstrum(
        &mut st.lpc,
        <&[f32; NB_BANDS]>::try_from(&st.features[..NB_BANDS]).unwrap(),
    );
    for i in 0..LPC_ORDER {
        st.features[NB_BANDS + 2 + i] = st.lpc[i];
    }

    let pm = PITCH_MAX_PERIOD;
    st.exc_buf.copy_within(FRAME_SIZE..pm + FRAME_SIZE, 0);
    st.lp_buf.copy_within(FRAME_SIZE..pm + FRAME_SIZE, 0);

    aligned_in[TRAINING_OFFSET..FRAME_SIZE].copy_from_slice(&input[..FRAME_SIZE - TRAINING_OFFSET]);
    let mut x_fir = [0.0f32; FRAME_SIZE + LPC_ORDER];
    x_fir[..LPC_ORDER].copy_from_slice(&st.pitch_mem);
    x_fir[LPC_ORDER..LPC_ORDER + FRAME_SIZE].copy_from_slice(&aligned_in);
    st.pitch_mem
        .copy_from_slice(&aligned_in[FRAME_SIZE - LPC_ORDER..FRAME_SIZE]);

    // FIR LPC analysis filter (sign convention matches C lpcnet_enc.c)
    for i in 0..FRAME_SIZE {
        let mut sum = x_fir[LPC_ORDER + i];
        for j in 0..LPC_ORDER {
            sum -= st.lpc[j] * x_fir[LPC_ORDER + i - j - 1];
        }
        st.lp_buf[pm + i] = sum;
    }

    for i in 0..FRAME_SIZE {
        st.exc_buf[pm + i] = st.lp_buf[pm + i] + 0.7 * st.pitch_filt;
        st.pitch_filt = st.lp_buf[pm + i];
    }

    // Biquad LP filter (needs separate input/output buffers)
    let mut lp_tmp = [0.0f32; FRAME_SIZE];
    lp_tmp.copy_from_slice(&st.lp_buf[pm..pm + FRAME_SIZE]);
    biquad(
        &mut st.lp_buf[pm..pm + FRAME_SIZE],
        &mut st.lp_mem,
        &lp_tmp,
        &LP_B,
        &LP_A,
        FRAME_SIZE,
    );

    // Pitch cross-correlation
    let buf = &st.exc_buf;
    let ener0: f32 = opus_celt::mathops::celt_inner_prod(&buf[pm..], &buf[pm..], FRAME_SIZE);
    let mut ener1: f64 =
        opus_celt::mathops::celt_inner_prod(&buf[0..], &buf[0..], FRAME_SIZE) as f64;

    let mut xcorr = [0.0f32; PITCH_MAX_PERIOD];
    opus_celt::pitch::celt_pitch_xcorr(
        &buf[pm..pm + FRAME_SIZE],
        buf,
        &mut xcorr,
        FRAME_SIZE,
        pm - PITCH_MIN_PERIOD,
    );

    let mut ener_norm = [0.0f32; PITCH_MAX_PERIOD - PITCH_MIN_PERIOD];
    for i in 0..PITCH_MAX_PERIOD - PITCH_MIN_PERIOD {
        let ener = 1.0 + ener0 as f64 + ener1;
        st.xcorr_features[i] = 2.0 * xcorr[i];
        ener_norm[i] = ener as f32;
        ener1 +=
            buf[i + FRAME_SIZE] as f64 * buf[i + FRAME_SIZE] as f64 - buf[i] as f64 * buf[i] as f64;
    }
    for (xf, en) in st.xcorr_features[..PITCH_MAX_PERIOD - PITCH_MIN_PERIOD]
        .iter_mut()
        .zip(ener_norm.iter())
    {
        *xf /= *en;
    }

    st.dnn_pitch = compute_pitchdnn(&mut st.pitchdnn, &st.if_features, &st.xcorr_features);

    let pitch = pitch_period_from_dnn(st.dnn_pitch);
    let pitch = pitch.clamp(PITCH_MIN_PERIOD, PITCH_MAX_PERIOD - 1);

    let lp = &st.lp_buf;
    let xx = opus_celt::mathops::celt_inner_prod(&lp[pm..], &lp[pm..], FRAME_SIZE);
    let yy = opus_celt::mathops::celt_inner_prod(&lp[pm - pitch..], &lp[pm - pitch..], FRAME_SIZE);
    let xy = opus_celt::mathops::celt_inner_prod(&lp[pm..], &lp[pm - pitch..], FRAME_SIZE);

    let frame_corr = xy / (1.0 + xx * yy).sqrt();
    let frame_corr = (1.0 + (5.0 * frame_corr).exp()).ln() / (1.0 + (5.0f32).exp()).ln();

    st.features[NB_BANDS] = st.dnn_pitch;
    st.features[NB_BANDS + 1] = frame_corr - 0.5;
}

/// Shared implementation for single-frame feature extraction.
fn compute_single_frame_features_impl(st: &mut LpcnetEncState, x: &mut [f32; FRAME_SIZE]) {
    let input_copy = *x;
    preemphasis(x, &mut st.mem_preemph, &input_copy, PREEMPHASIS, FRAME_SIZE);
    compute_frame_features(st, x);
}

/// Compute features for a single frame of PCM (i16 input).
pub fn lpcnet_compute_single_frame_features(
    st: &mut LpcnetEncState,
    pcm: &[i16],
    features: &mut [f32; NB_TOTAL_FEATURES],
) {
    let mut x = [0.0f32; FRAME_SIZE];
    for i in 0..FRAME_SIZE {
        x[i] = pcm[i] as f32;
    }
    compute_single_frame_features_impl(st, &mut x);
    features.copy_from_slice(&st.features);
}

/// Compute features for a single frame of PCM (f32 input).
pub fn lpcnet_compute_single_frame_features_float(
    st: &mut LpcnetEncState,
    pcm: &[f32],
    features: &mut [f32; NB_TOTAL_FEATURES],
) {
    let mut x = [0.0f32; FRAME_SIZE];
    x.copy_from_slice(&pcm[..FRAME_SIZE]);
    compute_single_frame_features_impl(st, &mut x);
    features.copy_from_slice(&st.features);
}
