// Port of silk/stereo_LR_to_MS.c, silk/stereo_find_predictor.c,
// silk/stereo_quant_pred.c, silk/stereo_encode_pred.c
//
// Stereo encoding: convert L/R to adaptive M/S with prediction.

use crate::signal_processing::silk_inner_prod_aligned_scale;
use crate::tables::*;
use crate::*;
use opus_range_coder::EcCtx;

// Constants from silk/define.h
const LA_SHAPE_MS: i32 = 5;

/// STEREO_RATIO_SMOOTH_COEF = 0.01 in Q16 = 655
const STEREO_RATIO_SMOOTH_COEF_Q16: i32 = 655;

/// STEREO_RATIO_SMOOTH_COEF / 2 in Q16 = 328
const STEREO_RATIO_SMOOTH_COEF_HALF_Q16: i32 = 328;

/// Stereo encoder state (from silk/structs.h stereo_enc_state)
#[derive(Clone)]
pub struct StereoEncState {
    pub pred_prev_q13: [i16; 2],
    pub s_mid: [i16; 2],
    pub s_side: [i16; 2],
    pub mid_side_amp_q0: [i32; 4],
    pub smth_width_q14: i16,
    pub width_prev_q14: i16,
    pub silent_side_len: i16,
    pub pred_ix: [[[i8; 3]; 2]; MAX_FRAMES_PER_PACKET],
    pub mid_only_flags: [i8; MAX_FRAMES_PER_PACKET],
}

impl Default for StereoEncState {
    fn default() -> Self {
        Self {
            pred_prev_q13: [0; 2],
            s_mid: [0; 2],
            s_side: [0; 2],
            mid_side_amp_q0: [0; 4],
            smth_width_q14: 0,
            width_prev_q14: 0,
            silent_side_len: 0,
            pred_ix: [[[0i8; 3]; 2]; MAX_FRAMES_PER_PACKET],
            mid_only_flags: [0i8; MAX_FRAMES_PER_PACKET],
        }
    }
}

/// Find least-squares predictor between mid and side signals.
///
/// Port of silk_stereo_find_predictor (silk/stereo_find_predictor.c).
/// Returns predictor in Q13.
pub fn silk_stereo_find_predictor(
    ratio_q14: &mut i32,
    x: &[i16],
    y: &[i16],
    mid_res_amp_q0: &mut [i32],
    length: usize,
    smooth_coef_q16: i32,
) -> i32 {
    let mut smooth_coef_q16 = smooth_coef_q16;

    // Find predictor
    let mut nrgx = 0i32;
    let mut scale1 = 0i32;
    silk_sum_sqr_shift(&mut nrgx, &mut scale1, x, length);

    let mut nrgy = 0i32;
    let mut scale2 = 0i32;
    silk_sum_sqr_shift(&mut nrgy, &mut scale2, y, length);

    let mut scale = scale1.max(scale2);
    scale = scale + (scale & 1); // make even

    nrgy >>= scale - scale2;
    nrgx >>= scale - scale1;
    nrgx = nrgx.max(1);

    let corr = silk_inner_prod_aligned_scale(x, y, scale, length);
    let mut pred_q13 = silk_div32_varq(corr, nrgx, 13);
    pred_q13 = pred_q13.clamp(-(1 << 14), 1 << 14);
    let pred2_q10 = silk_smulwb(pred_q13, pred_q13);

    // Faster update for signals with large prediction parameters
    smooth_coef_q16 = smooth_coef_q16.max(pred2_q10.abs());

    // Smoothed mid and residual norms
    debug_assert!(smooth_coef_q16 < 32768);
    let scale_half = scale >> 1;
    mid_res_amp_q0[0] = silk_smlawb(
        mid_res_amp_q0[0],
        (silk_sqrt_approx(nrgx) << scale_half) - mid_res_amp_q0[0],
        smooth_coef_q16,
    );

    // Residual energy = nrgy - 2 * pred * corr + pred^2 * nrgx
    let mut nrgy_res = silk_sub_lshift32(nrgy, silk_smulwb(corr, pred_q13), 3 + 1);
    nrgy_res = silk_add_lshift32(nrgy_res, silk_smulwb(nrgx, pred2_q10), 6);
    mid_res_amp_q0[1] = silk_smlawb(
        mid_res_amp_q0[1],
        (silk_sqrt_approx(nrgy_res) << scale_half) - mid_res_amp_q0[1],
        smooth_coef_q16,
    );

    // Ratio of smoothed residual and mid norms
    *ratio_q14 = silk_div32_varq(mid_res_amp_q0[1], mid_res_amp_q0[0].max(1), 14);
    *ratio_q14 = (*ratio_q14).clamp(0, 32767);

    pred_q13
}

/// Quantize mid/side predictors.
///
/// Port of silk_stereo_quant_pred (silk/stereo_quant_pred.c).
/// Brute-force search over SILK_STEREO_PRED_QUANT_Q13 table with
/// STEREO_QUANT_SUB_STEPS=5 sub-steps.
/// Outputs ix[2][3] indices and updates pred_q13 to quantized values.
pub fn silk_stereo_quant_pred(pred_q13: &mut [i32; 2], ix: &mut [[i8; 3]; 2]) {
    // SILK_FIX_CONST(0.5 / STEREO_QUANT_SUB_STEPS, 16) = 0.1 * 65536 = 6554
    const STEP_SCALE_Q16: i32 = 6554;

    for n in 0..2 {
        let mut err_min_q13 = i32::MAX;
        let mut quant_pred_q13 = 0i32;

        'outer: for i in 0..(STEREO_QUANT_TAB_SIZE - 1) {
            let low_q13 = SILK_STEREO_PRED_QUANT_Q13[i] as i32;
            let next_q13 = SILK_STEREO_PRED_QUANT_Q13[i + 1] as i32;
            let step_q13 = silk_smulwb(next_q13 - low_q13, STEP_SCALE_Q16);
            for j in 0..STEREO_QUANT_SUB_STEPS {
                let lvl_q13 = silk_smlabb(low_q13, step_q13, 2 * j + 1);
                let err_q13 = (pred_q13[n] - lvl_q13).abs();
                if err_q13 < err_min_q13 {
                    err_min_q13 = err_q13;
                    quant_pred_q13 = lvl_q13;
                    ix[n][0] = i as i8;
                    ix[n][1] = j as i8;
                } else {
                    // Error increasing, so we're past the optimum
                    break 'outer;
                }
            }
        }

        ix[n][2] = silk_div32_16(ix[n][0] as i32, 3) as i8;
        ix[n][0] -= ix[n][2] * 3;
        pred_q13[n] = quant_pred_q13;
    }

    // Subtract second from first predictor (helps when actually applying these)
    pred_q13[0] -= pred_q13[1];
}

/// Entropy-code the mid/side quantization indices.
///
/// Port of silk_stereo_encode_pred (silk/stereo_encode_pred.c).
pub fn silk_stereo_encode_pred(ps_range_enc: &mut EcCtx, ix: &[[i8; 3]; 2]) {
    // Joint coarse index: 5 * ix[0][2] + ix[1][2]
    let n = (5 * ix[0][2] + ix[1][2]) as usize;
    debug_assert!(n < 25);
    ps_range_enc.enc_icdf(n, &SILK_STEREO_PRED_JOINT_ICDF, 8);

    // Fine + sub-step per channel
    for item in &ix[..2] {
        debug_assert!((item[0] as i32) < 3);
        debug_assert!((item[1] as i32) < STEREO_QUANT_SUB_STEPS);
        ps_range_enc.enc_icdf(item[0] as usize, &SILK_UNIFORM3_ICDF, 8);
        ps_range_enc.enc_icdf(item[1] as usize, &SILK_UNIFORM5_ICDF, 8);
    }
}

/// Entropy-code the mid-only flag.
///
/// Port of silk_stereo_encode_mid_only (silk/stereo_encode_pred.c).
pub fn silk_stereo_encode_mid_only(ps_range_enc: &mut EcCtx, mid_only_flag: i8) {
    ps_range_enc.enc_icdf(mid_only_flag as usize, &SILK_STEREO_ONLY_CODE_MID_ICDF, 8);
}

/// Convert Left/Right stereo signal to adaptive Mid/Side representation.
///
/// Port of silk_stereo_LR_to_MS (silk/stereo_LR_to_MS.c).
///
/// The main conversion function:
/// - Convert L/R to basic M/S (average/difference)
/// - LP/HP filter both channels
/// - Find predictors for LP and HP bands
/// - Compute stereo width and bitrate distribution
/// - Quantize predictors
/// - Subtract prediction from side (with interpolation from previous frame)
/// - Output mid (in x1), side residual (in x2), mid_only_flag, rates
///
/// # Arguments
/// * `state` - Stereo encoder state
/// * `x1` - Left input signal (length frame_length + 2, with 2 samples of look-ahead).
///   Becomes mid signal on output. Indexing starts from offset -2 relative to
///   the "current" sample, so x1[0..frame_length+2] covers the range [-2, frame_length).
/// * `x2` - Right input signal (same layout). Becomes side residual on output.
/// * `ix` - Output quantization indices [2][3]
/// * `mid_only_flag` - Output: 1 if only mid channel is coded
/// * `mid_side_rates_bps` - Output: bitrates for mid [0] and side [1] signals
/// * `total_rate_bps` - Total bitrate
/// * `prev_speech_act_q8` - Speech activity level from previous frame
/// * `to_mono` - True if this is the last frame before stereo->mono transition
/// * `fs_khz` - Sample rate in kHz
/// * `frame_length` - Number of samples per channel
pub fn silk_stereo_lr_to_ms(
    state: &mut StereoEncState,
    x1: &mut [i16], // length = frame_length + 2
    x2: &mut [i16], // length = frame_length + 2
    ix: &mut [[i8; 3]; 2],
    mid_only_flag: &mut i8,
    mid_side_rates_bps: &mut [i32; 2],
    total_rate_bps: i32,
    prev_speech_act_q8: i32,
    to_mono: bool,
    fs_khz: i32,
    frame_length: usize,
) {
    let mut total_rate_bps = total_rate_bps;

    // Allocate side buffer (frame_length + 2)
    let buf_len = frame_length + 2;
    let mut side = vec![0i16; buf_len];

    // Convert to basic mid/side signals
    // In the C code, mid = &x1[-2], so mid[n] maps to x1[n-2].
    // Here we treat x1 and x2 as having the 2-sample prefix already included,
    // so x1[0], x1[1] are the two "previous" samples and x1[2..] is the current frame.
    // We compute mid/side in-place and in the side buffer.
    for n in 0..buf_len {
        let sum = x1[n] as i32 + x2[n] as i32;
        let diff = x1[n] as i32 - x2[n] as i32;
        x1[n] = silk_rshift_round(sum, 1) as i16;
        side[n] = silk_sat16(silk_rshift_round(diff, 1));
    }

    // Buffering: swap in previous state for the first 2 samples
    let mid_saved = [x1[0], x1[1]];
    let side_saved = [side[0], side[1]];
    x1[0] = state.s_mid[0];
    x1[1] = state.s_mid[1];
    side[0] = state.s_side[0];
    side[1] = state.s_side[1];
    // Save last 2 samples for next frame's buffer
    // In the C code: state->sMid = &mid[frame_length], which is x1[frame_length..frame_length+2]
    state.s_mid[0] = if frame_length < buf_len {
        x1[frame_length]
    } else {
        mid_saved[0]
    };
    state.s_mid[1] = if frame_length + 1 < buf_len {
        x1[frame_length + 1]
    } else {
        mid_saved[1]
    };
    state.s_side[0] = if frame_length < buf_len {
        side[frame_length]
    } else {
        side_saved[0]
    };
    state.s_side[1] = if frame_length + 1 < buf_len {
        side[frame_length + 1]
    } else {
        side_saved[1]
    };

    // LP and HP filter mid signal
    // mid[n] is x1[n], and mid[n+1] is x1[n+1], mid[n+2] is x1[n+2]
    let mut lp_mid = vec![0i16; frame_length];
    let mut hp_mid = vec![0i16; frame_length];
    for n in 0..frame_length {
        let sum = silk_rshift_round(
            silk_add_lshift(x1[n] as i32 + x1[n + 2] as i32, x1[n + 1] as i32, 1),
            2,
        );
        lp_mid[n] = sum as i16;
        hp_mid[n] = (x1[n + 1] as i32 - sum) as i16;
    }

    // LP and HP filter side signal
    let mut lp_side = vec![0i16; frame_length];
    let mut hp_side = vec![0i16; frame_length];
    for n in 0..frame_length {
        let sum = silk_rshift_round(
            silk_add_lshift(side[n] as i32 + side[n + 2] as i32, side[n + 1] as i32, 1),
            2,
        );
        lp_side[n] = sum as i16;
        hp_side[n] = (side[n + 1] as i32 - sum) as i16;
    }

    // Find energies and predictors
    let is_10ms_frame = frame_length == (10 * fs_khz) as usize;
    let smooth_coef_q16 = if is_10ms_frame {
        STEREO_RATIO_SMOOTH_COEF_HALF_Q16
    } else {
        STEREO_RATIO_SMOOTH_COEF_Q16
    };
    let smooth_coef_q16 = silk_smulwb(
        silk_smulbb(prev_speech_act_q8, prev_speech_act_q8),
        smooth_coef_q16,
    );

    let mut lp_ratio_q14 = 0i32;
    let mut hp_ratio_q14 = 0i32;

    let mut pred_q13 = [0i32; 2];
    pred_q13[0] = silk_stereo_find_predictor(
        &mut lp_ratio_q14,
        &lp_mid,
        &lp_side,
        &mut state.mid_side_amp_q0[0..2],
        frame_length,
        smooth_coef_q16,
    );
    pred_q13[1] = silk_stereo_find_predictor(
        &mut hp_ratio_q14,
        &hp_mid,
        &hp_side,
        &mut state.mid_side_amp_q0[2..4],
        frame_length,
        smooth_coef_q16,
    );

    // Ratio of the norms of residual and mid signals
    // frac_Q16 = HP_ratio_Q14 + 3 * LP_ratio_Q14
    let frac_q16 = silk_smlabb(hp_ratio_q14, lp_ratio_q14, 3);
    let frac_q16 = frac_q16.min(1 << 16); // SILK_FIX_CONST(1, 16) = 65536

    // Determine bitrate distribution between mid and side, and possibly reduce stereo width
    total_rate_bps -= if is_10ms_frame { 1200 } else { 600 };
    if total_rate_bps < 1 {
        total_rate_bps = 1;
    }
    let min_mid_rate_bps = silk_smlabb(2000, fs_khz, 600);
    debug_assert!(min_mid_rate_bps < 32767);

    // Default bitrate distribution: 8 parts for Mid and (5+3*frac) parts for Side
    // mid_rate = (8 / (13 + 3 * frac)) * total_rate
    let frac_3_q16 = 3 * frac_q16;
    // SILK_FIX_CONST(8 + 5, 16) = 13 << 16 = 851968
    mid_side_rates_bps[0] = silk_div32_varq(total_rate_bps, (13 << 16) + frac_3_q16, 16 + 3);

    let mut width_q14: i32;

    // If Mid bitrate below minimum, reduce stereo width
    if mid_side_rates_bps[0] < min_mid_rate_bps {
        mid_side_rates_bps[0] = min_mid_rate_bps;
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        // width = 4 * (2 * side_rate - min_rate) / ((1 + 3 * frac) * min_rate)
        width_q14 = silk_div32_varq(
            (mid_side_rates_bps[1] << 1) - min_mid_rate_bps,
            silk_smulwb((1 << 16) + frac_3_q16, min_mid_rate_bps),
            14 + 2,
        );
        width_q14 = width_q14.clamp(0, 1 << 14); // SILK_FIX_CONST(1, 14) = 16384
    } else {
        mid_side_rates_bps[1] = total_rate_bps - mid_side_rates_bps[0];
        width_q14 = 1 << 14; // SILK_FIX_CONST(1, 14)
    }

    // Smoother
    state.smth_width_q14 = silk_smlawb(
        state.smth_width_q14 as i32,
        width_q14 - state.smth_width_q14 as i32,
        smooth_coef_q16,
    ) as i16;

    // At very low bitrates or for inputs that are nearly amplitude panned, switch to panned-mono coding
    *mid_only_flag = 0;

    if to_mono {
        // Last frame before stereo->mono transition; collapse stereo width
        width_q14 = 0;
        pred_q13[0] = 0;
        pred_q13[1] = 0;
        silk_stereo_quant_pred(&mut pred_q13, ix);
    } else if state.width_prev_q14 == 0
        && (8 * total_rate_bps < 13 * min_mid_rate_bps
            || silk_smulwb(frac_q16, state.smth_width_q14 as i32) < 819)
    // SILK_FIX_CONST(0.05, 14) = 819
    {
        // Code as panned-mono; previous frame already had zero width
        pred_q13[0] = silk_smulbb(state.smth_width_q14 as i32, pred_q13[0]) >> 14;
        pred_q13[1] = silk_smulbb(state.smth_width_q14 as i32, pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, ix);
        // Collapse stereo width
        width_q14 = 0;
        pred_q13[0] = 0;
        pred_q13[1] = 0;
        mid_side_rates_bps[0] = total_rate_bps;
        mid_side_rates_bps[1] = 0;
        *mid_only_flag = 1;
    } else if state.width_prev_q14 != 0
        && (8 * total_rate_bps < 11 * min_mid_rate_bps
            || silk_smulwb(frac_q16, state.smth_width_q14 as i32) < 328)
    // SILK_FIX_CONST(0.02, 14) = 328
    {
        // Transition to zero-width stereo
        pred_q13[0] = silk_smulbb(state.smth_width_q14 as i32, pred_q13[0]) >> 14;
        pred_q13[1] = silk_smulbb(state.smth_width_q14 as i32, pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, ix);
        // Collapse stereo width
        width_q14 = 0;
        pred_q13[0] = 0;
        pred_q13[1] = 0;
    } else if state.smth_width_q14 > 15565 {
        // SILK_FIX_CONST(0.95, 14) = 15565
        // Full-width stereo coding
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = 1 << 14; // SILK_FIX_CONST(1, 14)
    } else {
        // Reduced-width stereo coding; scale down and quantize predictors
        pred_q13[0] = silk_smulbb(state.smth_width_q14 as i32, pred_q13[0]) >> 14;
        pred_q13[1] = silk_smulbb(state.smth_width_q14 as i32, pred_q13[1]) >> 14;
        silk_stereo_quant_pred(&mut pred_q13, ix);
        width_q14 = state.smth_width_q14 as i32;
    }

    // Make sure to keep on encoding until the tapered output has been transmitted
    if *mid_only_flag == 1 {
        state.silent_side_len += (frame_length as i32 - STEREO_INTERP_LEN_MS * fs_khz) as i16;
        if (state.silent_side_len as i32) < LA_SHAPE_MS * fs_khz {
            *mid_only_flag = 0;
        } else {
            // Limit to avoid wrapping around
            state.silent_side_len = 10000;
        }
    } else {
        state.silent_side_len = 0;
    }

    if *mid_only_flag == 0 && mid_side_rates_bps[1] < 1 {
        mid_side_rates_bps[1] = 1;
        mid_side_rates_bps[0] = (total_rate_bps - mid_side_rates_bps[1]).max(1);
    }

    // Interpolate predictors and subtract prediction from side channel
    let interp_len = (STEREO_INTERP_LEN_MS * fs_khz) as usize;
    let denom_q16 = silk_div32_16(1 << 16, (STEREO_INTERP_LEN_MS * fs_khz) as i16);

    let mut pred0_q13 = -(state.pred_prev_q13[0] as i32);
    let mut pred1_q13 = -(state.pred_prev_q13[1] as i32);
    let mut w_q24 = (state.width_prev_q14 as i32) << 10;
    let delta0_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[0] - state.pred_prev_q13[0] as i32, denom_q16),
        16,
    );
    let delta1_q13 = -silk_rshift_round(
        silk_smulbb(pred_q13[1] - state.pred_prev_q13[1] as i32, denom_q16),
        16,
    );
    let deltaw_q24 = silk_smulwb(width_q14 - state.width_prev_q14 as i32, denom_q16) << 10;

    for n in 0..interp_len.min(frame_length) {
        pred0_q13 += delta0_q13;
        pred1_q13 += delta1_q13;
        w_q24 += deltaw_q24;
        // sum = (mid[n] + mid[n+2] + 2*mid[n+1]) << 9   (Q11)
        let sum = silk_add_lshift(x1[n] as i32 + x1[n + 2] as i32, x1[n + 1] as i32, 1) << 9;
        // side_residual = w * side[n+1] + sum * pred0 + mid[n+1] * 2048 * pred1
        let mut out = silk_smlawb(silk_smulwb(w_q24, side[n + 1] as i32), sum, pred0_q13);
        out = silk_smlawb(out, (x1[n + 1] as i32) << 11, pred1_q13);
        x2[n + 1] = silk_sat16(silk_rshift_round(out, 8));
    }

    // After interpolation: use final predictor values
    let pred0_q13_final = -pred_q13[0];
    let pred1_q13_final = -pred_q13[1];
    let w_q24_final = width_q14 << 10;
    for n in interp_len..frame_length {
        let sum = silk_add_lshift(x1[n] as i32 + x1[n + 2] as i32, x1[n + 1] as i32, 1) << 9;
        let mut out = silk_smlawb(
            silk_smulwb(w_q24_final, side[n + 1] as i32),
            sum,
            pred0_q13_final,
        );
        out = silk_smlawb(out, (x1[n + 1] as i32) << 11, pred1_q13_final);
        x2[n + 1] = silk_sat16(silk_rshift_round(out, 8));
    }

    // Update state
    state.pred_prev_q13[0] = pred_q13[0] as i16;
    state.pred_prev_q13[1] = pred_q13[1] as i16;
    state.width_prev_q14 = width_q14 as i16;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stereo_enc_state_default() {
        let state = StereoEncState::default();
        assert_eq!(state.pred_prev_q13, [0, 0]);
        assert_eq!(state.s_mid, [0, 0]);
        assert_eq!(state.s_side, [0, 0]);
        assert_eq!(state.mid_side_amp_q0, [0, 0, 0, 0]);
        assert_eq!(state.smth_width_q14, 0);
        assert_eq!(state.width_prev_q14, 0);
        assert_eq!(state.silent_side_len, 0);
    }

    #[test]
    fn test_stereo_quant_pred_zero() {
        let mut pred_q13 = [0i32; 2];
        let mut ix = [[0i8; 3]; 2];
        silk_stereo_quant_pred(&mut pred_q13, &mut ix);
        // Quantized zero predictors should produce small values
        // and subtract second from first
    }

    #[test]
    fn test_stereo_encode_pred_roundtrip() {
        // Quantize some predictors, encode, decode, compare
        let mut pred_q13 = [1000i32, -500i32];
        let mut ix = [[0i8; 3]; 2];
        silk_stereo_quant_pred(&mut pred_q13, &mut ix);

        // Encode
        let mut enc = EcCtx::enc_init(256);
        silk_stereo_encode_pred(&mut enc, &ix);
        enc.enc_done();

        // Decode
        let nbytes = ((enc.tell() + 7) >> 3) as usize;
        let buf = enc.buf[..nbytes].to_vec();
        let mut dec = EcCtx::dec_init(&buf);

        let mut dec_pred_q13 = [0i32; 2];
        crate::stereo::silk_stereo_decode_pred(&mut dec, &mut dec_pred_q13);

        // Decoded predictors should match the quantized ones
        assert_eq!(dec_pred_q13[0], pred_q13[0]);
        assert_eq!(dec_pred_q13[1], pred_q13[1]);
    }

    #[test]
    fn test_stereo_lr_to_ms_silence() {
        let mut state = StereoEncState::default();
        let frame_length = 320usize; // 20ms at 16kHz
        let buf_len = frame_length + 2;
        let mut x1 = vec![0i16; buf_len];
        let mut x2 = vec![0i16; buf_len];
        let mut ix = [[0i8; 3]; 2];
        let mut mid_only_flag = 0i8;
        let mut mid_side_rates_bps = [0i32; 2];

        silk_stereo_lr_to_ms(
            &mut state,
            &mut x1,
            &mut x2,
            &mut ix,
            &mut mid_only_flag,
            &mut mid_side_rates_bps,
            20000, // total_rate_bps
            128,   // prev_speech_act_q8
            false, // to_mono
            16,    // fs_khz
            frame_length,
        );

        // With silence input, mid and side should be zero
        for (i, &sample) in x1.iter().enumerate().take(buf_len) {
            assert_eq!(sample, 0, "mid[{}] should be zero for silence", i);
        }
    }

    #[test]
    fn test_stereo_encode_mid_only_roundtrip() {
        // Test mid_only_flag encode/decode
        for flag in 0..=1i8 {
            let mut enc = EcCtx::enc_init(256);
            silk_stereo_encode_mid_only(&mut enc, flag);
            enc.enc_done();

            let nbytes = ((enc.tell() + 7) >> 3) as usize;
            let buf = enc.buf[..nbytes].to_vec();
            let mut dec = EcCtx::dec_init(&buf);

            let decoded = crate::stereo::silk_stereo_decode_mid_only(&mut dec);
            assert_eq!(decoded, flag as i32);
        }
    }
}
