// Noise shaping analysis for the SILK encoder.
//
// Port of silk/fixed/noise_shape_analysis_FIX.c
//
// Computes per-subframe spectral shaping filter parameters that the NSQ
// uses to shape quantization noise under the speech spectral envelope.

use crate::nlsf::silk_bwexpander_32 as bwexpander_32_nlsf;
use crate::nsq::MAX_SHAPE_LPC_ORDER;
use crate::signal_processing::{
    silk_apply_sine_window, silk_k2a_q16 as k2a_q16_sp, silk_schur64 as schur64_sp,
    silk_sigm_q15 as sigm_q15,
};
use crate::*;

// ---- Tuning parameters (from silk/tuning_parameters.h) ----

/// Fraction added to first autocorrelation value (white noise floor)
const SHAPE_WHITE_NOISE_FRACTION_Q20: i32 = 31; // 3e-5 * (1 << 20) ~ 31

/// Noise shaping filter chirp factor: 0.94 in Q16
const BANDWIDTH_EXPANSION_Q16: i32 = 61604; // (0.94 * 65536 + 0.5) as i32

/// Harmonic shaping base: 0.3 in Q16
const HARMONIC_SHAPING_Q16: i32 = 19661;

/// Extra harmonic shaping for high bitrates or noisy input: 0.2 in Q16
const HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING_Q16: i32 = 13107;

/// HP noise tilt coefficient: 0.25 in Q16
const HP_NOISE_COEF_Q16: i32 = 16384;

/// Harmonic HP noise coefficient: 0.35 in Q24
const HARM_HP_NOISE_COEF_Q24: i32 = 5872026; // (0.35 * (1<<24) + 0.5) as i32

/// Low frequency shaping: 4.0
const LOW_FREQ_SHAPING: i32 = 4;

/// Less LF shaping for low quality: 0.5 in Q13
const LOW_QUALITY_LOW_FREQ_SHAPING_DECR_Q13: i32 = 4096;

/// Subframe smoothing coefficient: 0.4 in Q16
const SUBFR_SMTH_COEF_Q16: i32 = 26214;

/// Background SNR decrease: 2.0 dB
#[allow(dead_code)] // May be used when full SNR tracking is implemented
const BG_SNR_DECR_DB: f64 = 2.0;

/// Harmonic SNR increase: 2.0 in Q8
const HARM_SNR_INCR_DB_Q8: i32 = 512; // 2.0 * (1 << 8)

/// Lambda parameters for R/D tradeoff
const LAMBDA_OFFSET_Q16: i32 = 78643; // 1.2 * 65536
const LAMBDA_SPEECH_ACT_Q16: i32 = -13107; // -0.2 * 65536
const LAMBDA_INPUT_QUALITY_Q16: i32 = -6554; // -0.1 * 65536
const LAMBDA_CODING_QUALITY_Q16: i32 = -13107; // -0.2 * 65536
const LAMBDA_QUANT_OFFSET_Q16: i32 = 52429; // 0.8 * 65536

/// Energy variation threshold for quantization offset: 0.6 in Q7
const ENERGY_VARIATION_THRESHOLD_QNT_OFFSET_Q7: i32 = 77; // 0.6 * 128

/// Pitch white noise fraction for BWE control: 1e-3 in Q16
const FIND_PITCH_WHITE_NOISE_FRACTION_Q16: i32 = 66; // (1e-3 * 65536 + 0.5) as i32

// Duplicated LUTs and helper functions removed -- now delegated to
// signal_processing.rs, nlsf.rs, and lpc_analysis.rs via imports above.

/// Results of noise shape analysis for one frame.
pub struct NoiseShapeAnalysis {
    /// AR shaping coefficients, Q13, layout: [nb_subfr * shaping_lpc_order]
    /// Padded to MAX_SHAPE_LPC_ORDER per subframe for NSQ compatibility.
    pub ar_q13: [i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER],
    /// Harmonic noise shaping gain per subframe, Q14
    pub harm_shape_gain_q14: [i32; MAX_NB_SUBFR],
    /// Noise tilt per subframe, Q14
    pub tilt_q14: [i32; MAX_NB_SUBFR],
    /// Low-frequency shaping per subframe, Q14 (packed: high 16 = MA, low 16 = AR)
    pub lf_shp_q14: [i32; MAX_NB_SUBFR],
    /// Subframe gains, Q16
    pub gains_q16: [i32; MAX_NB_SUBFR],
    /// Rate-distortion lambda, Q10
    pub lambda_q10: i32,
    /// Coding quality, Q14
    pub coding_quality_q14: i32,
    /// Input quality, Q14
    pub input_quality_q14: i32,
    /// Quantization offset type: 0 (low) or 1 (high)
    pub quant_offset_type: i8,
}

/// Delegate to signal_processing::silk_sigm_q15.
#[inline]
fn silk_sigm_q15(in_q5: i32) -> i32 {
    sigm_q15(in_q5)
}

/// Apply sine window to a signal segment.
///
/// win_type 1: sine from 0 to pi/2 (rising)
/// win_type 2: sine from pi/2 to pi (falling)
///
/// Port of silk_apply_sine_window from silk/fixed/apply_sine_window_FIX.c.
/// Delegate to signal_processing::silk_apply_sine_window.
#[inline]
fn apply_sine_window(px_win: &mut [i16], px: &[i16], win_type: i32, length: usize) {
    silk_apply_sine_window(px_win, px, win_type, length as i32);
}

/// Delegate to signal_processing::silk_schur64.
#[inline]
fn silk_schur64(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
    schur64_sp(rc_q16, c, order)
}

/// Delegate to signal_processing::silk_k2a_q16.
#[inline]
fn silk_k2a_q16(a_q24: &mut [i32], rc_q16: &[i32], order: usize) {
    k2a_q16_sp(a_q24, rc_q16, order)
}

/// Delegate to nlsf::silk_bwexpander_32.
#[inline]
fn bwexpander_32(ar: &mut [i32], d: usize, chirp_q16: i32) {
    bwexpander_32_nlsf(ar, d, chirp_q16)
}

/// silk_LPC_fit: Convert Q24 AR coefficients to Q13 i16 with overflow protection.
///
/// Port of silk_LPC_fit from silk/LPC_fit.c.
fn silk_lpc_fit(a_q13_out: &mut [i16], a_q24_in: &mut [i32], d: usize) {
    let qin = 24;
    let qout = 13;
    let shift = qin - qout; // 11

    for _iter in 0..10 {
        let mut maxabs = 0i32;
        let mut idx = 0usize;
        for (k, item) in a_q24_in.iter().enumerate().take(d) {
            let absval = (item.unsigned_abs().min(i32::MAX as u32)) as i32;
            if absval > maxabs {
                maxabs = absval;
                idx = k;
            }
        }
        maxabs = silk_rshift_round(maxabs, shift);

        if maxabs > i16::MAX as i32 {
            maxabs = maxabs.min(163838);
            let chirp_q16 = 65471 // 0.999 in Q16
                - silk_div32(
                    (maxabs - i16::MAX as i32) << 14,
                    (maxabs.wrapping_mul((idx as i32) + 1)) >> 2,
                );
            bwexpander_32(a_q24_in, d, chirp_q16);
        } else {
            // Coefficients fit -- convert and return
            for k in 0..d {
                a_q13_out[k] = silk_rshift_round(a_q24_in[k], shift) as i16;
            }
            return;
        }
    }

    // Last resort: clip
    for k in 0..d {
        a_q13_out[k] = silk_sat16(silk_rshift_round(a_q24_in[k], shift));
        a_q24_in[k] = (a_q13_out[k] as i32) << shift;
    }
}

/// Autocorrelation with scale output (used by schur64).
///
/// Like silk_autocorrelation in lpc_analysis.rs, but also returns the right-shift
/// applied, which is needed for gain computation.
fn autocorrelation_with_scale(
    results: &mut [i32],
    input: &[i16],
    input_len: usize,
    correlation_count: usize,
) -> i32 {
    // Compute energy (lag 0) to determine shift
    let mut nrg: i64 = 0;
    for item in input.iter().take(input_len) {
        nrg += (*item as i64) * (*item as i64);
    }

    let shift = if nrg > i32::MAX as i64 {
        let mut s = 0;
        let mut tmp = nrg;
        while tmp > i32::MAX as i64 {
            tmp >>= 1;
            s += 1;
        }
        s
    } else {
        0i32
    };

    results[0] = (nrg >> shift) as i32;

    for k in 1..=correlation_count {
        let mut acc: i64 = 0;
        for i in k..input_len {
            acc += (input[i] as i64) * (input[i - k] as i64);
        }
        results[k] = (acc >> shift) as i32;
    }

    shift
}

/// Compute noise shaping filter parameters for one frame.
///
/// This is a faithful port of silk_noise_shape_analysis_FIX, simplified to
/// use standard (non-warped) autocorrelation. The output parameters feed
/// directly into silk_nsq.
pub fn silk_noise_shape_analysis(
    input: &[i16],
    pitch_lags: &[i32],
    is_voiced: bool,
    prev_tilt_q16: &mut i32,
    prev_harm_q16: &mut i32,
    fs_khz: i32,
    nb_subfr: i32,
    subfr_length: i32,
    _frame_length: i32,
    _lpc_order: i32,
    shaping_lpc_order: i32,
    _warping_q16: i32,
    speech_activity_q8: i32,
    _coding_quality_q14: i32,
    snr_db_q7: i32,
) -> NoiseShapeAnalysis {
    let nb_subfr_usize = nb_subfr as usize;
    let shaping_order = shaping_lpc_order as usize;

    // ========================================================================
    // Gain control
    // ========================================================================
    let mut snr_adj_db_q7 = snr_db_q7;

    // Input quality: for simplicity use a moderate fixed value since we do
    // not have VAD band quality here. 0.5 in Q14 = 8192.
    let input_quality_q14: i32 = 8192;

    // Coding quality via sigmoid of SNR
    // coding_quality_Q14 = silk_RSHIFT( silk_sigm_Q15(
    //     silk_RSHIFT_ROUND( SNR_adj_dB_Q7 - 20.0_Q7, 4 ) ), 1 );
    let computed_coding_quality_q14 = silk_sigm_q15(
        silk_rshift_round(snr_adj_db_q7 - (20 * 128), 4), // 20.0 in Q7 = 2560
    ) >> 1;

    // Reduce coding SNR during low speech activity (non-CBR path)
    {
        let b_q8 = (1i32 << 8) - speech_activity_q8;
        let b_q8_sq = silk_smulwb(b_q8 << 8, b_q8);
        // BG_SNR_DECR_dB = 2.0 => 2.0 in Q7 = 256; shifted >> (4+1) = 256 >> 5 = 8
        let bg_snr_decr_q7_shifted: i32 = -((2.0f64 * 128.0) as i32) >> 5; // = -8 (Q(7-5) = Q2)
        let quality_factor = silk_smulwb(
            (1i32 << 14) + input_quality_q14,
            computed_coding_quality_q14,
        ); // Q12
        snr_adj_db_q7 = silk_smlawb(
            snr_adj_db_q7,
            silk_smulbb(bg_snr_decr_q7_shifted, b_q8_sq),
            quality_factor,
        );
    }

    if is_voiced {
        // Reduce gains for periodic signals
        // HARM_SNR_INCR_dB = 2.0 in Q8 = 512
        // We approximate LTPCorr_Q15 ~ 0.5 (moderate pitch correlation) = 16384 in Q15
        let ltp_corr_q15: i32 = 16384;
        snr_adj_db_q7 = silk_smlawb(snr_adj_db_q7, HARM_SNR_INCR_DB_Q8, ltp_corr_q15);
    } else {
        // For unvoiced: adjust quality slower than SNR
        let adj = silk_smlawb(
            6i32 << 9, // 6.0 in Q9
            -(26214),  // -0.4 in Q18 ~ -26214
            snr_db_q7,
        );
        snr_adj_db_q7 = silk_smlawb(snr_adj_db_q7, adj, (1i32 << 14) - input_quality_q14);
    }

    // ========================================================================
    // Quantizer offset type
    // ========================================================================
    let quant_offset_type: i8;
    if is_voiced {
        quant_offset_type = 0;
    } else {
        // Sparseness measure based on energy variation per 2ms segments
        let n_samples = (fs_khz * 2) as usize;
        let n_segs = ((SUB_FRAME_LENGTH_MS as i32 * nb_subfr) / 2) as usize;
        let mut energy_variation_q7 = 0i32;
        let mut log_energy_prev_q7 = 0i32;

        // We use the input signal itself as an approximation of pitch residual
        for seg in 0..n_segs {
            let start = seg * n_samples;
            let end = (start + n_samples).min(input.len());
            if start >= input.len() {
                break;
            }
            let segment = &input[start..end];
            let mut nrg = 0i32;
            let mut scale = 0i32;
            silk_sum_sqr_shift(&mut nrg, &mut scale, segment, segment.len());
            nrg += (n_samples as i32) >> scale;

            let log_energy_q7 = silk_lin2log(nrg);
            if seg > 0 {
                energy_variation_q7 += (log_energy_q7 - log_energy_prev_q7).abs();
            }
            log_energy_prev_q7 = log_energy_q7;
        }

        if n_segs > 1
            && energy_variation_q7 > ENERGY_VARIATION_THRESHOLD_QNT_OFFSET_Q7 * (n_segs as i32 - 1)
        {
            quant_offset_type = 0;
        } else {
            quant_offset_type = 1;
        }
    }

    // ========================================================================
    // Bandwidth expansion control
    // ========================================================================
    // More BWE for signals with high prediction gain.
    // We use a moderate fixed predGain since we lack the LTP analysis gain here.
    let pred_gain_q16: i32 = 1 << 16; // 1.0 in Q16 (conservative)
    let strength_q16 = silk_smulwb(pred_gain_q16, FIND_PITCH_WHITE_NOISE_FRACTION_Q16);
    let bw_exp_q16 = silk_div32_varq(
        BANDWIDTH_EXPANSION_Q16,
        silk_smlawb(1i32 << 16, strength_q16, strength_q16),
        16,
    );

    // ========================================================================
    // Window parameters
    // ========================================================================
    // la_shape: lookahead for noise shape analysis.
    // For complexity 2 (our default): la_shape = 3 * fs_kHz (complexity 0-1)
    // or 5 * fs_kHz (complexity >= 1). We use the simpler la = 5 * fs_kHz.
    let la_shape = 5 * fs_khz;
    let shape_win_length = (SUB_FRAME_LENGTH_MS as i32) * fs_khz + 2 * la_shape;

    // ========================================================================
    // Compute noise shaping AR coefficients and gains per subframe
    // ========================================================================
    let mut ar_q13 = [0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
    let mut gains_q16 = [0i32; MAX_NB_SUBFR];

    // Stack arrays for per-subframe analysis (bounded by MAX_SHAPE_LPC_ORDER=24)
    let mut x_windowed = [0i16; 240]; // max shape_win_length = 15 * 16 = 240
    let mut auto_corr = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut refl_coef_q16 = [0i32; MAX_SHAPE_LPC_ORDER];
    let mut ar_q24 = [0i32; MAX_SHAPE_LPC_ORDER];

    // x_ptr starts at input - la_shape (but since input already includes look-ahead,
    // we just start from the beginning for the first subframe)
    let _la_shape_usize = la_shape as usize;

    for k in 0..nb_subfr_usize {
        // The input pointer for subframe k. In the C code: x_ptr = x - la_shape,
        // then x_ptr advances by subfr_length each subframe.
        // The caller provides input[0..frame_length + la_shape], and the analysis
        // window is centered on each subframe.
        let x_ptr_start = k * (subfr_length as usize);

        // Apply window: sine slope + flat part + cosine slope
        let flat_part = (fs_khz * 3) as usize;
        let slope_part = ((shape_win_length as usize) - flat_part) / 2;
        // Round slope_part down to multiple of 4 for sine window
        let slope_part = (slope_part / 4) * 4;
        let slope_part = slope_part.clamp(16, 120);

        let win_len = 2 * slope_part + flat_part;
        let actual_win_len = win_len.min(x_windowed.len());

        // Make sure we don't go out of bounds on input
        let avail = input.len().saturating_sub(x_ptr_start);
        let actual_win_len = actual_win_len.min(avail);
        if actual_win_len < 16 {
            // Not enough samples, skip this subframe
            continue;
        }

        let src = &input[x_ptr_start..x_ptr_start + actual_win_len];

        // Rising sine slope
        let rising_len = slope_part.min(actual_win_len);
        if rising_len >= 16 {
            apply_sine_window(
                &mut x_windowed[..rising_len],
                &src[..rising_len],
                1,
                rising_len,
            );
        }

        // Flat part (copy)
        let flat_start = rising_len;
        let flat_end = (flat_start + flat_part).min(actual_win_len);
        x_windowed[flat_start..flat_end].copy_from_slice(&src[flat_start..flat_end]);

        // Falling cosine slope
        let falling_start = flat_end;
        let falling_len_raw = actual_win_len.saturating_sub(falling_start);
        let falling_len = (falling_len_raw / 4) * 4;
        let falling_len = falling_len
            .min(120)
            .max(if falling_len_raw >= 16 { 16 } else { 0 });
        if falling_len >= 16 {
            apply_sine_window(
                &mut x_windowed[falling_start..falling_start + falling_len],
                &src[falling_start..falling_start + falling_len],
                2,
                falling_len,
            );
        }

        // Calculate autocorrelation
        let analysis_len = if falling_len >= 16 {
            falling_start + falling_len
        } else {
            flat_end
        };
        let analysis_len = analysis_len.max(shaping_order + 1);

        let scale = autocorrelation_with_scale(
            &mut auto_corr,
            &x_windowed[..analysis_len],
            analysis_len,
            shaping_order,
        );

        // Add white noise as fraction of energy
        let wn_add = silk_smulwb(auto_corr[0] >> 4, SHAPE_WHITE_NOISE_FRACTION_Q20).max(1);
        auto_corr[0] = auto_corr[0].saturating_add(wn_add);

        // Schur recursion: autocorrelation -> reflection coefficients
        let nrg = silk_schur64(&mut refl_coef_q16, &auto_corr, shaping_order);

        // Convert reflection coefficients to prediction coefficients
        for item in ar_q24.iter_mut().take(shaping_order) {
            *item = 0;
        }
        silk_k2a_q16(&mut ar_q24, &refl_coef_q16, shaping_order);

        // Compute gain from residual energy
        let mut q_nrg = -(scale);
        // Make q_nrg even
        let mut nrg_val = nrg;
        if q_nrg & 1 != 0 {
            q_nrg -= 1;
            nrg_val >>= 1;
        }

        let tmp32 = silk_sqrt_approx(nrg_val);
        q_nrg >>= 1;

        gains_q16[k] = silk_lshift_sat32(tmp32, 16 - q_nrg);
        // Bandwidth expansion
        bwexpander_32(&mut ar_q24, shaping_order, bw_exp_q16);

        // Convert AR Q24 -> AR Q13 with overflow protection (silk_LPC_fit)
        silk_lpc_fit(
            &mut ar_q13[k * MAX_SHAPE_LPC_ORDER..],
            &mut ar_q24,
            shaping_order,
        );
    }

    // ========================================================================
    // Gain tweaking
    // ========================================================================
    // gain_mult_Q16 = silk_log2lin(-silk_SMLAWB(-16.0_Q7, SNR_adj_dB_Q7, 0.16_Q16))
    let gain_mult_q16 = silk_log2lin(-silk_smlawb(
        -(16 * 128), // -16.0 in Q7
        snr_adj_db_q7,
        10486, // 0.16 in Q16
    ));
    // gain_add_Q16 = silk_log2lin(silk_SMLAWB(16.0_Q7, MIN_QGAIN_DB_Q7, 0.16_Q16))
    let gain_add_q16 = silk_log2lin(silk_smlawb(
        16 * 128,           // 16.0 in Q7
        MIN_QGAIN_DB * 128, // MIN_QGAIN_DB in Q7
        10486,
    ));

    // Apply gain_mult (SNR-based reduction) and gain_add (minimum floor).
    // The encoder's process_gains stage will floor these using residual energy.
    for item in gains_q16.iter_mut().take(nb_subfr_usize) {
        *item = silk_smulww_correct(*item, gain_mult_q16);
        if *item < 0 {
            *item = i32::MAX;
        }
        *item = item.saturating_add(gain_add_q16);
    }

    // ========================================================================
    // Low-frequency shaping and noise tilt
    // ========================================================================
    let mut lf_shp_q14 = [0i32; MAX_NB_SUBFR];
    let tilt_q16: i32;

    // strength_Q16 for LF shaping
    // Less LF shaping for noisy inputs. We approximate input_quality_bands_Q15[0] = 0.5 * 32768 = 16384
    let input_quality_band0_q15: i32 = 16384;
    let lf_strength_q16 = LOW_FREQ_SHAPING
        * silk_smlawb(
            1i32 << 12, // 1.0 in Q12
            LOW_QUALITY_LOW_FREQ_SHAPING_DECR_Q13,
            input_quality_band0_q15 - (1i32 << 15),
        ); // Q(4+12) = Q16
    let lf_strength_q16 = ((lf_strength_q16 as i64 * speech_activity_q8 as i64) >> 8) as i32;

    if is_voiced {
        // For voiced: LF shaping depends on pitch lag per subframe
        let fs_khz_inv = silk_div32_varq(3277, fs_khz, 0); // 0.2 in Q14 / fs_kHz
        for k in 0..nb_subfr_usize {
            let lag = pitch_lags[k].max(1);
            let b_q14 = fs_khz_inv + silk_div32_varq(49152, lag, 0); // 3.0_Q14 / pitchL[k]

            // Pack two coefficients in one i32:
            // High 16 bits = LF_MA = 1.0_Q14 - b_Q14 - strength * b_Q14
            // Low 16 bits  = LF_AR = b_Q14 - 1.0_Q14
            let lf_ma = ((1i32 << 14) - b_q14 - silk_smulwb(lf_strength_q16, b_q14)) & 0xFFFF;
            let lf_ar = (b_q14 - (1i32 << 14)) as u16;
            lf_shp_q14[k] = (lf_ma << 16) | (lf_ar as i32);
        }

        // Tilt for voiced: more HP tilt during voiced speech with activity
        // Tilt_Q16 = -HP_NOISE_COEF_Q16 -
        //   SMULWB(1.0_Q16 - HP_NOISE_COEF_Q16,
        //          SMULWB(HARM_HP_NOISE_COEF_Q24, speech_activity_Q8))
        tilt_q16 = -HP_NOISE_COEF_Q16
            - silk_smulwb(
                (1i32 << 16) - HP_NOISE_COEF_Q16,
                silk_smulwb(HARM_HP_NOISE_COEF_Q24, speech_activity_q8),
            );
    } else {
        // For unvoiced: fixed b coefficient
        let b_q14 = silk_div32_varq(21299, fs_khz, 0); // 1.3_Q14 / fs_kHz
        let lf_ma = ((1i32 << 14) - b_q14
            - silk_smulwb(lf_strength_q16, silk_smulwb(39322, b_q14))) // 0.6_Q16 = 39322
            & 0xFFFF;
        let lf_ar = (b_q14 - (1i32 << 14)) as u16;
        let packed = (lf_ma << 16) | (lf_ar as i32);
        for item in lf_shp_q14.iter_mut().take(nb_subfr_usize) {
            *item = packed;
        }

        // Tilt for unvoiced: just HP noise coefficient
        tilt_q16 = -HP_NOISE_COEF_Q16;
    }

    // ========================================================================
    // Harmonic shaping control
    // ========================================================================
    let harm_shape_gain_q16: i32;
    if is_voiced {
        // More harmonic noise shaping for high bitrates or noisy input
        let mut hsg = silk_smlawb(
            HARMONIC_SHAPING_Q16,
            (1i32 << 16)
                - silk_smulwb(
                    (1i32 << 18) - ((computed_coding_quality_q14) << 4),
                    input_quality_q14,
                ),
            HIGH_RATE_OR_LOW_QUALITY_HARMONIC_SHAPING_Q16,
        );

        // Less harmonic shaping for less periodic signals
        // Approximate LTPCorr_Q15 ~ 0.5 = 16384
        let ltp_corr_q15: i32 = 16384;
        hsg = silk_smulwb(hsg << 1, silk_sqrt_approx(ltp_corr_q15 << 15));

        harm_shape_gain_q16 = hsg;
    } else {
        harm_shape_gain_q16 = 0;
    }

    // ========================================================================
    // Smooth over subframes
    // ========================================================================
    let mut harm_shape_gain_q14 = [0i32; MAX_NB_SUBFR];
    let mut tilt_q14_arr = [0i32; MAX_NB_SUBFR];

    for k in 0..nb_subfr as usize {
        *prev_harm_q16 = silk_smlawb(
            *prev_harm_q16,
            harm_shape_gain_q16 - *prev_harm_q16,
            SUBFR_SMTH_COEF_Q16,
        );
        *prev_tilt_q16 = silk_smlawb(
            *prev_tilt_q16,
            tilt_q16 - *prev_tilt_q16,
            SUBFR_SMTH_COEF_Q16,
        );

        harm_shape_gain_q14[k] = silk_rshift_round(*prev_harm_q16, 2);
        tilt_q14_arr[k] = silk_rshift_round(*prev_tilt_q16, 2);
    }

    // ========================================================================
    // Lambda (rate-distortion tradeoff)
    // ========================================================================
    // lambda = LAMBDA_OFFSET
    //        + LAMBDA_SPEECH_ACT * speech_activity
    //        + LAMBDA_INPUT_QUALITY * input_quality
    //        + LAMBDA_CODING_QUALITY * coding_quality
    //        + LAMBDA_QUANT_OFFSET * quant_offset_type
    let lambda_q16 = LAMBDA_OFFSET_Q16
        + silk_smulwb(LAMBDA_SPEECH_ACT_Q16, speech_activity_q8 << 8)
        + silk_smulwb(LAMBDA_INPUT_QUALITY_Q16, input_quality_q14 << 2)
        + silk_smulwb(LAMBDA_CODING_QUALITY_Q16, computed_coding_quality_q14 << 2)
        + silk_smulwb(LAMBDA_QUANT_OFFSET_Q16, (quant_offset_type as i32) << 16);
    // Convert from Q16 to Q10
    let lambda_q10 = silk_rshift_round(lambda_q16, 6).max(0);

    NoiseShapeAnalysis {
        ar_q13,
        harm_shape_gain_q14,
        tilt_q14: tilt_q14_arr,
        lf_shp_q14,
        gains_q16,
        lambda_q10,
        coding_quality_q14: computed_coding_quality_q14,
        input_quality_q14,
        quant_offset_type,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigm_q15() {
        // At 0 => 0.5 in Q15 = 16384
        assert_eq!(silk_sigm_q15(0), 16384);
        // Large positive => near 32767
        assert_eq!(silk_sigm_q15(6 * 32), 32767);
        // Large negative => 0
        assert_eq!(silk_sigm_q15(-6 * 32), 0);
    }

    #[test]
    fn test_schur64_basic() {
        // Simple autocorrelation: energy = 1000, corr[1] = 500 (highly correlated)
        let c = [1000, 500, 200, 50, 10, 0, 0, 0, 0, 0, 0];
        let mut rc = [0i32; 10];
        let nrg = silk_schur64(&mut rc, &c, 10);
        assert!(nrg > 0);
        // First reflection coefficient should be negative (positive correlation)
        assert!(rc[0] < 0);
    }

    #[test]
    fn test_schur64_zero_energy() {
        let c = [0; 11];
        let mut rc = [0i32; 10];
        let nrg = silk_schur64(&mut rc, &c, 10);
        assert_eq!(nrg, 0);
        assert!(rc.iter().all(|&x| x == 0));
    }

    #[test]
    fn test_k2a_q16_order1() {
        // Single reflection coefficient
        let rc = [-32768i32]; // -0.5 in Q16
        let mut a = [0i32; 1];
        silk_k2a_q16(&mut a, &rc, 1);
        // a[0] = -(-32768 << 8) = 32768 << 8 = 8388608
        assert_eq!(a[0], 32768 << 8);
    }

    #[test]
    fn test_bwexpander_32() {
        let mut ar = [1 << 24, 1 << 23, 1 << 22];
        bwexpander_32(&mut ar, 3, 61604); // 0.94 in Q16
        // First coefficient should be scaled by 0.94
        assert!(ar[0] < (1 << 24));
        assert!(ar[0] > 0);
    }

    #[test]
    fn test_apply_sine_window_basic() {
        let input = vec![10000i16; 16];
        let mut output = vec![0i16; 16];

        // Rising window: first sample should be near 0, last near full amplitude
        apply_sine_window(&mut output, &input, 1, 16);
        assert!(output[0].abs() < output[15].abs());

        // Falling window: first sample near full, last near 0
        apply_sine_window(&mut output, &input, 2, 16);
        assert!(output[0].abs() > output[15].abs());
    }

    #[test]
    fn test_noise_shape_analysis_silence() {
        let fs_khz = 16;
        let nb_subfr = 4;
        let subfr_length = 5 * fs_khz;
        let frame_length = nb_subfr * subfr_length;
        let la_shape = 5 * fs_khz;

        // Silent input with look-ahead
        let input = vec![0i16; (frame_length + la_shape) as usize];
        let pitch_lags = [80i32; MAX_NB_SUBFR];
        let mut prev_tilt = 0i32;
        let mut prev_harm = 0i32;

        let result = silk_noise_shape_analysis(
            &input,
            &pitch_lags,
            false,
            &mut prev_tilt,
            &mut prev_harm,
            fs_khz,
            nb_subfr,
            subfr_length,
            frame_length,
            16,       // lpc_order
            16,       // shaping_lpc_order
            0,        // warping_q16
            128,      // speech_activity_q8 (0.5)
            8192,     // coding_quality_q14 (0.5)
            20 * 128, // snr_db_q7 (20 dB)
        );

        assert_eq!(result.ar_q13.len(), 4 * MAX_SHAPE_LPC_ORDER);
        assert!(result.lambda_q10 >= 0);
    }

    #[test]
    fn test_noise_shape_analysis_tone() {
        let fs_khz = 16;
        let nb_subfr = 4;
        let subfr_length = 5 * fs_khz;
        let frame_length = nb_subfr * subfr_length;
        let la_shape = 5 * fs_khz;
        let total_len = (frame_length + la_shape) as usize;

        // Generate a 200Hz tone
        let mut input = vec![0i16; total_len];
        for (i, sample) in input.iter_mut().enumerate() {
            *sample =
                (8000.0 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let pitch_lags = [80i32; MAX_NB_SUBFR]; // 200Hz at 16kHz
        let mut prev_tilt = 0i32;
        let mut prev_harm = 0i32;

        let result = silk_noise_shape_analysis(
            &input,
            &pitch_lags,
            true, // voiced
            &mut prev_tilt,
            &mut prev_harm,
            fs_khz,
            nb_subfr,
            subfr_length,
            frame_length,
            16,
            16,
            0,
            200,      // speech_activity_q8 (~0.78)
            12000,    // coding_quality_q14 (~0.73)
            25 * 128, // snr_db_q7 (25 dB)
        );

        // For voiced signal, harmonic shaping gain should be nonzero
        assert!(result.harm_shape_gain_q14.iter().any(|&x| x != 0));
        // Tilt should be nonzero (HP tilt)
        assert!(result.tilt_q14.iter().any(|&x| x != 0));
        // Gains should be positive
        assert!(result.gains_q16.iter().all(|&x| x > 0));
        // AR coefficients should have some nonzero values (the tone has spectral structure)
        let has_nonzero_ar = result.ar_q13.iter().any(|&x| x != 0);
        assert!(
            has_nonzero_ar,
            "AR coefficients should be nonzero for a tonal signal"
        );
        // Quant offset type should be 0 for voiced
        assert_eq!(result.quant_offset_type, 0);
    }

    #[test]
    fn test_noise_shape_analysis_voiced_lf_shaping() {
        let fs_khz = 16;
        let nb_subfr = 4;
        let subfr_length = 5 * fs_khz;
        let frame_length = nb_subfr * subfr_length;
        let la_shape = 5 * fs_khz;
        let total_len = (frame_length + la_shape) as usize;

        let mut input = vec![0i16; total_len];
        for (i, sample) in input.iter_mut().enumerate() {
            *sample =
                (5000.0 * (2.0 * std::f64::consts::PI * 150.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let pitch_lags = [107i32; MAX_NB_SUBFR]; // ~150Hz
        let mut prev_tilt = 0i32;
        let mut prev_harm = 0i32;

        let result = silk_noise_shape_analysis(
            &input,
            &pitch_lags,
            true,
            &mut prev_tilt,
            &mut prev_harm,
            fs_khz,
            nb_subfr,
            subfr_length,
            frame_length,
            16,
            16,
            0,
            200,
            10000,
            25 * 128,
        );

        // LF shaping should be packed nonzero values for voiced
        for k in 0..nb_subfr as usize {
            assert_ne!(
                result.lf_shp_q14[k], 0,
                "LF shaping should be nonzero for voiced subframe {}",
                k
            );
        }
    }
}
