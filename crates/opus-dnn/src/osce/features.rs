use opus_celt::fft::KissFftCpx;
use std::sync::LazyLock;

use super::config::*;
use super::structs::OsceFeatureState;
use crate::freq::{self, FREQ_SIZE, NB_BANDS};

const OSCE_SPEC_WINDOW_SIZE: usize = 320;
const OSCE_SPEC_NUM_FREQS: usize = 161;

/// LTP_ORDER from SILK (number of LTP taps).
pub const LTP_ORDER: usize = 5;
/// TYPE_VOICED signal type from SILK.
const TYPE_VOICED: i32 = 2;

/// Bridge struct for passing SILK decoder state to OSCE feature extraction.
/// The caller populates this from SILK's `ChannelState` and `DecoderControl`
/// to avoid a direct opus-dnn → opus-silk dependency.
pub struct OsceInput<'a> {
    /// LPC prediction coefficients per half-frame (Q12). [2][lpc_order]
    pub pred_coef_q12: &'a [[i16; 16]; 2],
    /// Pitch lag per subframe.
    pub pitch_lags: &'a [i32],
    /// LTP coefficients per subframe (Q14). [nb_subfr][LTP_ORDER]
    pub ltp_coef_q14: &'a [i16],
    /// Gain per subframe (Q16).
    pub gains_q16: &'a [i32],
    /// LPC order.
    pub lpc_order: usize,
    /// Signal type (TYPE_VOICED=2, etc.). i8 in SILK, widened here.
    pub signal_type: i32,
    /// Number of subframes (2 or 4).
    pub nb_subfr: usize,
    /// Number of bits in the SILK payload.
    pub num_bits: i32,
}

// Filterbank tables from osce_features.c
#[rustfmt::skip]
const CENTER_BINS_CLEAN: [usize; 64] = [
    0, 2, 5, 8, 10, 12, 15, 18, 20, 22, 25, 28, 30, 33, 35, 38,
    40, 42, 45, 48, 50, 52, 55, 58, 60, 62, 65, 68, 70, 73, 75, 78,
    80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 105, 108, 110, 112, 115, 118,
    120, 122, 125, 128, 130, 132, 135, 138, 140, 142, 145, 148, 150, 152, 155, 160,
];

#[rustfmt::skip]
const CENTER_BINS_NOISY: [usize; 18] = [
    0, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64, 80, 96, 112, 136, 160,
];

#[rustfmt::skip]
const BAND_WEIGHTS_CLEAN: [f32; 64] = [
    0.666666666667, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.400000000000, 0.400000000000, 0.400000000000, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.400000000000, 0.400000000000, 0.400000000000, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.333333333333, 0.400000000000,
    0.500000000000, 0.400000000000, 0.250000000000, 0.333333333333,
];

#[rustfmt::skip]
const BAND_WEIGHTS_NOISY: [f32; 18] = [
    0.400000000000, 0.250000000000, 0.250000000000, 0.250000000000,
    0.250000000000, 0.250000000000, 0.250000000000, 0.250000000000,
    0.166666666667, 0.125000000000, 0.125000000000, 0.125000000000,
    0.083333333333, 0.062500000000, 0.062500000000, 0.050000000000,
    0.041666666667, 0.080000000000,
];

/// Pre-computed OSCE analysis window. Matches C static `osce_window[320]`.
/// w[n] = sin(pi*(n+0.5)/320), symmetric.
static OSCE_WINDOW: LazyLock<[f32; OSCE_SPEC_WINDOW_SIZE]> = LazyLock::new(|| {
    let mut w = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    for (n, w_val) in w.iter_mut().enumerate() {
        let t = std::f64::consts::PI * (n as f64 + 0.5) / OSCE_SPEC_WINDOW_SIZE as f64;
        *w_val = t.sin() as f32;
    }
    w
});

/// Apply triangular filterbank. Matches C `apply_filterbank`.
fn apply_filterbank(
    x_out: &mut [f32],
    x_in: &[f32],
    center_bins: &[usize],
    band_weights: &[f32],
    num_bands: usize,
) {
    x_out[0] = 0.0;
    for b in 0..num_bands - 1 {
        x_out[b + 1] = 0.0;
        for (i, x_val) in x_in
            .iter()
            .enumerate()
            .take(center_bins[b + 1])
            .skip(center_bins[b])
        {
            let frac =
                (center_bins[b + 1] - i) as f32 / (center_bins[b + 1] - center_bins[b]) as f32;
            x_out[b] += band_weights[b] * frac * x_val;
            x_out[b + 1] += band_weights[b + 1] * (1.0 - frac) * x_val;
        }
    }
    x_out[num_bands - 1] += band_weights[num_bands - 1] * x_in[center_bins[num_bands - 1]];
}

/// Compute 320-point magnitude spectrum. Matches C `mag_spec_320_onesided`.
fn mag_spec_320_onesided(out: &mut [f32], input: &[f32; OSCE_SPEC_WINDOW_SIZE]) {
    let mut fft_out = [KissFftCpx::default(); FREQ_SIZE];
    freq::forward_transform(&mut fft_out, input);
    for k in 0..OSCE_SPEC_NUM_FREQS {
        out[k] = OSCE_SPEC_WINDOW_SIZE as f32
            * (fft_out[k].r * fft_out[k].r + fft_out[k].i * fft_out[k].i).sqrt();
    }
}

/// Compute log spectrum from LPC coefficients. Matches C `calculate_log_spectrum_from_lpc`.
fn calculate_log_spectrum_from_lpc(spec: &mut [f32], a_q12: &[i16], lpc_order: usize) {
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    buffer[0] = 1.0;
    for i in 0..lpc_order {
        buffer[i + 1] = -(a_q12[i] as f32) / (1 << 12) as f32;
    }

    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    mag_spec_320_onesided(&mut mag, &buffer);

    let mut inv_mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    for i in 0..OSCE_SPEC_NUM_FREQS {
        inv_mag[i] = 1.0 / (mag[i] + 1e-9);
    }

    apply_filterbank(
        spec,
        &inv_mag,
        &CENTER_BINS_CLEAN,
        &BAND_WEIGHTS_CLEAN,
        OSCE_CLEAN_SPEC_NUM_BANDS,
    );
    for s in &mut spec[..OSCE_CLEAN_SPEC_NUM_BANDS] {
        *s = 0.3 * (*s + 1e-9).ln();
    }
}

/// Compute noisy cepstrum from signal. Matches C `calculate_cepstrum`.
fn calculate_cepstrum(cepstrum: &mut [f32], signal: &[f32]) {
    // signal should have at least OSCE_SPEC_WINDOW_SIZE samples accessible
    // (starting from some offset that includes history)
    let mut buffer = [0.0f32; OSCE_SPEC_WINDOW_SIZE];

    let window = &*OSCE_WINDOW;
    for n in 0..OSCE_SPEC_WINDOW_SIZE {
        buffer[n] = window[n] * signal[n];
    }

    let mut mag = [0.0f32; OSCE_SPEC_NUM_FREQS];
    mag_spec_320_onesided(&mut mag, &buffer);

    let mut spec_buf = [0.0f32; OSCE_SPEC_WINDOW_SIZE];
    apply_filterbank(
        &mut spec_buf[..OSCE_NOISY_SPEC_NUM_BANDS],
        &mag,
        &CENTER_BINS_NOISY,
        &BAND_WEIGHTS_NOISY,
        OSCE_NOISY_SPEC_NUM_BANDS,
    );

    for s in &mut spec_buf[..OSCE_NOISY_SPEC_NUM_BANDS] {
        *s = (*s + 1e-9).ln();
    }

    let spec_input: [f32; NB_BANDS] = spec_buf[..NB_BANDS].try_into().unwrap();
    let mut cep_out = [0.0f32; NB_BANDS];
    freq::dct(&mut cep_out, &spec_input);
    cepstrum[..NB_BANDS].copy_from_slice(&cep_out);
}

/// Compute autocorrelation around pitch lag. Matches C `calculate_acorr`.
/// `signal` must be a slice starting from the frame position within the full
/// buffer (with history before it accessible via negative offsets).
/// `frame_offset` is the index of the current frame within the full buffer.
fn calculate_acorr(acorr: &mut [f32], full_buffer: &[f32], frame_offset: usize, lag: usize) {
    for (k_off, ac) in acorr.iter_mut().enumerate().take(5) {
        let k = k_off as isize - 2;
        let mut xx = 0.0f32;
        let mut yy = 0.0f32;
        let mut xy = 0.0f32;
        for n in 0..80usize {
            let s_n = full_buffer[frame_offset + n];
            let lag_idx = (frame_offset + n) as isize - lag as isize + k;
            let s_lag = if lag_idx >= 0 && (lag_idx as usize) < full_buffer.len() {
                full_buffer[lag_idx as usize]
            } else {
                0.0
            };
            xx += s_n * s_n;
            yy += s_lag * s_lag;
            xy += s_n * s_lag;
        }
        *ac = xy / (xx * yy + 1e-9).sqrt();
    }
}

/// Pitch post-processing (hangover logic). Matches C `pitch_postprocessing`.
fn pitch_postprocessing(state: &mut OsceFeatureState, lag: i32, signal_type: i32) -> usize {
    // OSCE_PITCH_HANGOVER is 0, so hangover is effectively disabled
    let new_lag = if signal_type != TYPE_VOICED {
        state.pitch_hangover_count = 0;
        OSCE_NO_PITCH_VALUE
    } else {
        state.last_lag = lag;
        state.pitch_hangover_count = 0;
        lag as usize
    };
    state.last_type = signal_type;
    new_lag
}

/// Extract OSCE features from decoded SILK output.
/// Matches C `osce_calculate_features` from osce_features.c.
///
/// `xq` is the decoded speech signal (i16, `nb_subfr * 80` samples).
/// `features` receives `nb_subfr * OSCE_FEATURE_DIM` floats.
/// `numbits` receives [raw_bits, smoothed_bits].
/// `periods` receives the pitch lag per subframe.
pub fn osce_calculate_features(
    state: &mut OsceFeatureState,
    input: &OsceInput,
    xq: &[i16],
    features: &mut [f32],
    numbits: &mut [f32; 2],
    periods: &mut [usize],
) {
    let num_subframes = input.nb_subfr;
    let num_samples = num_subframes * 80;

    // Smooth bit count
    state.numbits_smooth = 0.9 * state.numbits_smooth + 0.1 * input.num_bits as f32;
    numbits[0] = input.num_bits as f32;
    numbits[1] = state.numbits_smooth;

    // Build signal buffer: [history | current frame]
    const BUF_SIZE: usize = OSCE_FEATURES_MAX_HISTORY + OSCE_MAX_FEATURE_FRAMES * 80;
    let mut buffer = [0.0f32; BUF_SIZE];
    buffer[..OSCE_FEATURES_MAX_HISTORY].copy_from_slice(&state.signal_history);
    for n in 0..num_samples {
        buffer[OSCE_FEATURES_MAX_HISTORY + n] = xq[n] as f32 / 32768.0;
    }

    for (k, period) in periods.iter_mut().enumerate().take(num_subframes) {
        let frame_start = OSCE_FEATURES_MAX_HISTORY + k * 80;
        let feat_base = k * OSCE_FEATURE_DIM;

        for v in features[feat_base..feat_base + OSCE_FEATURE_DIM].iter_mut() {
            *v = 0.0;
        }

        // Clean spectrum from LPC (update every other subframe)
        if k % 2 == 0 {
            let off = k * OSCE_FEATURE_DIM + OSCE_CLEAN_SPEC_START;
            calculate_log_spectrum_from_lpc(
                &mut features[off..off + OSCE_CLEAN_SPEC_LENGTH],
                &input.pred_coef_q12[k >> 1][..input.lpc_order],
                input.lpc_order,
            );
        } else {
            let prev_off = (k - 1) * OSCE_FEATURE_DIM + OSCE_CLEAN_SPEC_START;
            let cur_off = k * OSCE_FEATURE_DIM + OSCE_CLEAN_SPEC_START;
            features.copy_within(prev_off..prev_off + OSCE_CLEAN_SPEC_LENGTH, cur_off);
        }

        // Noisy cepstrum (update every other subframe)
        if k % 2 == 0 {
            let cep_start = frame_start.saturating_sub(160);
            if cep_start + OSCE_SPEC_WINDOW_SIZE <= buffer.len() {
                let off = k * OSCE_FEATURE_DIM + OSCE_NOISY_CEPSTRUM_START;
                calculate_cepstrum(
                    &mut features[off..off + OSCE_NOISY_CEPSTRUM_LENGTH],
                    &buffer[cep_start..cep_start + OSCE_SPEC_WINDOW_SIZE],
                );
            }
        } else {
            let prev_off = (k - 1) * OSCE_FEATURE_DIM + OSCE_NOISY_CEPSTRUM_START;
            let cur_off = k * OSCE_FEATURE_DIM + OSCE_NOISY_CEPSTRUM_START;
            features.copy_within(prev_off..prev_off + OSCE_NOISY_CEPSTRUM_LENGTH, cur_off);
        }

        // Pitch post-processing
        *period = pitch_postprocessing(state, input.pitch_lags[k], input.signal_type);

        // Autocorrelation around pitch lag (pass full buffer so negative lag offsets reach history)
        let acorr_off = feat_base + OSCE_ACORR_START;
        calculate_acorr(
            &mut features[acorr_off..acorr_off + OSCE_ACORR_LENGTH],
            &buffer,
            frame_start,
            *period,
        );

        // LTP coefficients
        for i in 0..OSCE_LTP_LENGTH {
            features[feat_base + OSCE_LTP_START + i] =
                input.ltp_coef_q14[k * LTP_ORDER + i] as f32 / (1 << 14) as f32;
        }

        // Log gain
        features[feat_base + OSCE_LOG_GAIN_START] =
            (input.gains_q16[k] as f32 / (1u32 << 16) as f32 + 1e-9).ln();
    }

    // Update signal history
    let hist_start = num_samples;
    state
        .signal_history
        .copy_from_slice(&buffer[hist_start..hist_start + OSCE_FEATURES_MAX_HISTORY]);

    if state.reset {
        state.reset = false;
    }
}
