// Signal processing helper functions for the SILK encoder.
// Ported from the C reference implementation (fixed-point path).

use crate::{
    silk_clz32, silk_div32_varq, silk_rshift_round, silk_sat16, silk_smlawb, silk_smmul,
    silk_smulbb, silk_smulwb,
};

// Use the shared MAX_SHAPE_LPC_ORDER (=24) from nsq.rs for both Schur recursion and warped autocorrelation
use crate::nsq::MAX_SHAPE_LPC_ORDER;
const SILK_MAX_ORDER_LPC: usize = MAX_SHAPE_LPC_ORDER;

// Q-domain constants for warped autocorrelation (from silk/fixed/main_FIX.h)
const QC: i32 = 10;
const QS: i32 = 13;

// ============================================================================
// 1. silk_sigm_Q15 - Sigmoid approximation
// Ported from silk/sigm_Q15.c
// ============================================================================

/// Lookup table: slopes for piecewise-linear sigmoid segments (Q10)
static SIGM_LUT_SLOPE_Q10: [i32; 6] = [237, 153, 73, 30, 12, 7];

/// Lookup table: positive-input sigmoid values (Q15)
static SIGM_LUT_POS_Q15: [i32; 6] = [16384, 23955, 28861, 31213, 32178, 32548];

/// Lookup table: negative-input sigmoid values (Q15)
static SIGM_LUT_NEG_Q15: [i32; 6] = [16384, 8812, 3906, 1554, 589, 219];

/// Approximate sigmoid function. Returns value in Q15 domain [0, 32767].
///
/// Input is in Q5 fixed-point format.
pub fn silk_sigm_q15(in_q5: i32) -> i32 {
    if in_q5 < 0 {
        // Negative input
        let in_q5 = -in_q5;
        if in_q5 >= 6 * 32 {
            0 // Clip
        } else {
            // Linear interpolation of look up table
            let ind = (in_q5 >> 5) as usize;
            SIGM_LUT_NEG_Q15[ind] - silk_smulbb(SIGM_LUT_SLOPE_Q10[ind], in_q5 & 0x1F)
        }
    } else {
        // Positive input
        if in_q5 >= 6 * 32 {
            32767 // Clip
        } else {
            // Linear interpolation of look up table
            let ind = (in_q5 >> 5) as usize;
            SIGM_LUT_POS_Q15[ind] + silk_smulbb(SIGM_LUT_SLOPE_Q10[ind], in_q5 & 0x1F)
        }
    }
}

// ============================================================================
// 2. silk_schur / silk_schur64 - Schur recursion for reflection coefficients
// Ported from silk/fixed/schur_FIX.c and silk/fixed/schur64_FIX.c
// ============================================================================

/// SILK_FIX_CONST(0.99, 15) = (0.99 * 32768 + 0.5) = 32440
const FIX_CONST_099_Q15: i32 = 32440;
/// SILK_FIX_CONST(0.99, 16) = (0.99 * 65536 + 0.5) = 64881
const FIX_CONST_099_Q16: i32 = 64881;

/// Compute reflection coefficients from autocorrelation sequence via Schur recursion.
///
/// Faster but less accurate than `silk_schur64`. Uses `silk_smlawb` for the update step.
///
/// - `rc_q15`: output reflection coefficients in Q15 (length >= order)
/// - `c`: input autocorrelation values (length >= order + 1)
/// - `order`: prediction order
///
/// Returns the residual energy (always >= 1).
pub fn silk_schur(rc_q15: &mut [i16], c: &[i32], order: usize) -> i32 {
    debug_assert!(order <= SILK_MAX_ORDER_LPC);

    // C[k][0] and C[k][1] arrays
    let mut big_c = [[0i32; 2]; SILK_MAX_ORDER_LPC + 1];

    // Get number of leading zeros
    let lz = silk_clz32(c[0]);

    // Copy correlations and adjust level to Q30
    if lz < 2 {
        // lz must be 1, so shift one to the right
        for k in 0..=order {
            let val = c[k] >> 1;
            big_c[k][0] = val;
            big_c[k][1] = val;
        }
    } else if lz > 2 {
        // Shift to the left
        let lz_adj = lz - 2;
        for k in 0..=order {
            let val = c[k].wrapping_shl(lz_adj as u32);
            big_c[k][0] = val;
            big_c[k][1] = val;
        }
    } else {
        // No need to shift
        for k in 0..=order {
            big_c[k][0] = c[k];
            big_c[k][1] = c[k];
        }
    }

    let mut k = 0;
    while k < order {
        // Check that we won't be getting an unstable rc, otherwise stop here.
        if (big_c[k + 1][0].unsigned_abs() as i32) >= big_c[0][1] {
            if big_c[k + 1][0] > 0 {
                rc_q15[k] = -FIX_CONST_099_Q15 as i16;
            } else {
                rc_q15[k] = FIX_CONST_099_Q15 as i16;
            }
            k += 1;
            break;
        }

        // Get reflection coefficient
        let denom = (big_c[0][1] >> 15).max(1) as i16;
        let mut rc_tmp_q15 = -(big_c[k + 1][0] / denom as i32);

        // Clip (shouldn't happen for properly conditioned inputs)
        rc_tmp_q15 = rc_tmp_q15.clamp(-32768, 32767);

        // Store
        rc_q15[k] = rc_tmp_q15 as i16;

        // Update correlations
        for n in 0..(order - k) {
            let ctmp1 = big_c[n + k + 1][0];
            let ctmp2 = big_c[n][1];
            big_c[n + k + 1][0] = silk_smlawb(ctmp1, ctmp2.wrapping_shl(1), rc_tmp_q15);
            big_c[n][1] = silk_smlawb(ctmp2, ctmp1.wrapping_shl(1), rc_tmp_q15);
        }

        k += 1;
    }

    // Zero remaining coefficients
    for item in rc_q15.iter_mut().take(order).skip(k) {
        *item = 0;
    }

    // Return residual energy
    big_c[0][1].max(1)
}

/// Compute reflection coefficients from autocorrelation sequence via Schur recursion (64-bit).
///
/// Slower but more accurate than `silk_schur`. Uses `silk_smmul` for the update step.
///
/// - `rc_q16`: output reflection coefficients in Q16 (length >= order)
/// - `c`: input autocorrelation values (length >= order + 1)
/// - `order`: prediction order
///
/// Returns the residual energy (always >= 1).
pub fn silk_schur64(rc_q16: &mut [i32], c: &[i32], order: usize) -> i32 {
    debug_assert!(order <= SILK_MAX_ORDER_LPC);

    let mut big_c = [[0i32; 2]; SILK_MAX_ORDER_LPC + 1];

    // Check for invalid input
    if c[0] <= 0 {
        for item in rc_q16.iter_mut().take(order) {
            *item = 0;
        }
        return 0;
    }

    // Copy correlations
    for k in 0..=order {
        big_c[k][0] = c[k];
        big_c[k][1] = c[k];
    }

    let mut k = 0;
    while k < order {
        // Check that we won't be getting an unstable rc, otherwise stop here.
        if (big_c[k + 1][0].unsigned_abs() as i32) >= big_c[0][1] {
            if big_c[k + 1][0] > 0 {
                rc_q16[k] = -FIX_CONST_099_Q16;
            } else {
                rc_q16[k] = FIX_CONST_099_Q16;
            }
            k += 1;
            break;
        }

        // Get reflection coefficient: divide two Q30 values and get result in Q31
        let rc_tmp_q31 = silk_div32_varq(-big_c[k + 1][0], big_c[0][1], 31);

        // Save the output
        rc_q16[k] = silk_rshift_round(rc_tmp_q31, 15);

        // Update correlations
        for n in 0..(order - k) {
            let ctmp1_q30 = big_c[n + k + 1][0];
            let ctmp2_q30 = big_c[n][1];

            // Multiply and add the highest int32
            big_c[n + k + 1][0] =
                ctmp1_q30.wrapping_add(silk_smmul(ctmp2_q30.wrapping_shl(1), rc_tmp_q31));
            big_c[n][1] = ctmp2_q30.wrapping_add(silk_smmul(ctmp1_q30.wrapping_shl(1), rc_tmp_q31));
        }

        k += 1;
    }

    // Zero remaining coefficients
    for item in rc_q16.iter_mut().take(order).skip(k) {
        *item = 0;
    }

    big_c[0][1].max(1)
}

// ============================================================================
// 3. silk_k2a / silk_k2a_q16 - Reflection coefficients to AR prediction coefficients
// Ported from silk/fixed/k2a_FIX.c and silk/fixed/k2a_Q16_FIX.c
// ============================================================================

/// Convert reflection coefficients (Q15) to AR prediction coefficients (Q24).
///
/// Step-up function using the Levinson-Durbin recursion.
///
/// - `a_q24`: output prediction coefficients (length >= order)
/// - `rc_q15`: input reflection coefficients in Q15 (length >= order)
/// - `order`: prediction order
pub fn silk_k2a(a_q24: &mut [i32], rc_q15: &[i16], order: usize) {
    for k in 0..order {
        let rc = rc_q15[k] as i32;
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlawb(tmp1, tmp2.wrapping_shl(1), rc);
            a_q24[k - n - 1] = silk_smlawb(tmp2, tmp1.wrapping_shl(1), rc);
        }
        a_q24[k] = -((rc_q15[k] as i32) << 9);
    }
}

/// silk_SMLAWW: a + ((b * c) >> 16), matching the C macro for full 32-bit multiply
#[inline(always)]
fn silk_smlaww(a: i32, b: i32, c: i32) -> i32 {
    a.wrapping_add(((b as i64 * c as i64) >> 16) as i32)
}

/// Convert reflection coefficients (Q16) to AR prediction coefficients (Q24).
///
/// Step-up function using the Levinson-Durbin recursion.
///
/// - `a_q24`: output prediction coefficients (length >= order)
/// - `rc_q16`: input reflection coefficients in Q16 (length >= order)
/// - `order`: prediction order
pub fn silk_k2a_q16(a_q24: &mut [i32], rc_q16: &[i32], order: usize) {
    for k in 0..order {
        let rc = rc_q16[k];
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a_q24[n];
            let tmp2 = a_q24[k - n - 1];
            a_q24[n] = silk_smlaww(tmp1, tmp2, rc);
            a_q24[k - n - 1] = silk_smlaww(tmp2, tmp1, rc);
        }
        a_q24[k] = -(rc << 8);
    }
}

// ============================================================================
// 4. silk_apply_sine_window - Half-sine windowing
// Ported from silk/fixed/apply_sine_window_FIX.c
// ============================================================================

/// Frequency table for sine window approximation (Q16).
/// Indexed by (length / 4) - 4.
static FREQ_TABLE_Q16: [i16; 27] = [
    12111, 9804, 8235, 7100, 6239, 5565, 5022, 4575, 4202, 3885, 3612, 3375, 3167, 2984, 2820,
    2674, 2542, 2422, 2313, 2214, 2123, 2038, 1961, 1889, 1822, 1760, 1702,
];

/// Apply sine window to signal vector.
///
/// Window types:
/// - 1: sine window from 0 to pi/2 (rising half)
/// - 2: sine window from pi/2 to pi (falling half)
///
/// Every other sample is linearly interpolated for speed.
/// Window `length` must be between 16 and 120 (inclusive) and a multiple of 4.
///
/// - `px_win`: output windowed signal
/// - `px`: input signal
/// - `win_type`: 1 for rising half, 2 for falling half
/// - `length`: window length (multiple of 4, in range [16, 120])
pub fn silk_apply_sine_window(px_win: &mut [i16], px: &[i16], win_type: i32, length: i32) {
    debug_assert!(win_type == 1 || win_type == 2);
    debug_assert!((16..=120).contains(&length));
    debug_assert!((length & 3) == 0);

    // Frequency
    let k_idx = ((length >> 2) - 4) as usize;
    debug_assert!(k_idx <= 26);
    let f_q16 = FREQ_TABLE_Q16[k_idx] as i32;

    // Factor used for cosine approximation
    let c_q16 = silk_smulwb(f_q16, -f_q16);

    // Initialize state
    let (mut s0_q16, mut s1_q16);
    if win_type == 1 {
        // Start from 0
        s0_q16 = 0i32;
        // Approximation of sin(f)
        s1_q16 = f_q16 + (length >> 3);
    } else {
        // Start from 1
        s0_q16 = 1i32 << 16;
        // Approximation of cos(f)
        s1_q16 = (1i32 << 16) + (c_q16 >> 1) + (length >> 4);
    }

    // Uses the recursive equation: sin(n*f) = 2 * cos(f) * sin((n-1)*f) - sin((n-2)*f)
    // 4 samples at a time
    let mut k = 0i32;
    while k < length {
        let ki = k as usize;
        px_win[ki] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[ki] as i32) as i16;
        px_win[ki + 1] = silk_smulwb(s1_q16, px[ki + 1] as i32) as i16;
        s0_q16 = silk_smulwb(s1_q16, c_q16) + (s1_q16 << 1) - s0_q16 + 1;
        s0_q16 = s0_q16.min(1i32 << 16);

        px_win[ki + 2] = silk_smulwb((s0_q16 + s1_q16) >> 1, px[ki + 2] as i32) as i16;
        px_win[ki + 3] = silk_smulwb(s0_q16, px[ki + 3] as i32) as i16;
        s1_q16 = silk_smulwb(s0_q16, c_q16) + (s0_q16 << 1) - s1_q16;
        s1_q16 = s1_q16.min(1i32 << 16);

        k += 4;
    }
}

// ============================================================================
// 5. silk_ana_filt_bank_1 - Analysis filter bank for VAD band splitting
// Ported from silk/ana_filt_bank_1.c
// ============================================================================

/// Coefficients for 2-band filter bank based on first-order allpass filters.
const A_FB1_20: i16 = (5394i32 << 1) as i16; // = 10788
const A_FB1_21: i16 = -24290; // (opus_int16)(20623 << 1)

/// Split signal into two decimated bands using first-order allpass filters (QMF analysis).
///
/// This version writes low-pass and high-pass outputs into a single buffer at specified offsets,
/// which is the calling convention used by the VAD code.
///
/// - `input`: input signal of length `length`
/// - `s`: filter state, 2 elements (preserved between calls)
/// - `out`: output buffer (must be large enough for both lp and hp outputs)
/// - `lp_offset`: starting index in `out` for low-pass output (length/2 samples)
/// - `hp_offset`: starting index in `out` for high-pass output (length/2 samples)
/// - `length`: number of input samples (must be even)
pub fn silk_ana_filt_bank_1(
    input: &[i16],
    s: &mut [i32; 2],
    out: &mut [i16],
    lp_offset: usize,
    hp_offset: usize,
    length: i32,
) {
    let n2 = (length >> 1) as usize;

    // Internal variables and state are in Q10 format
    for k in 0..n2 {
        // Convert to Q10
        let mut in32 = (input[2 * k] as i32) << 10;

        // All-pass section for even input sample
        let y = in32 - s[0];
        let x = silk_smlawb(y, y, A_FB1_21 as i32);
        let out_1 = s[0] + x;
        s[0] = in32 + x;

        // Convert to Q10
        in32 = (input[2 * k + 1] as i32) << 10;

        // All-pass section for odd input sample, and add to output of previous section
        let y = in32 - s[1];
        let x = silk_smulwb(y, A_FB1_20 as i32);
        let out_2 = s[1] + x;
        s[1] = in32 + x;

        // Add/subtract, convert back to int16 and store to output
        out[lp_offset + k] = silk_sat16(silk_rshift_round(out_2 + out_1, 11));
        out[hp_offset + k] = silk_sat16(silk_rshift_round(out_2 - out_1, 11));
    }
}

/// Split signal into two decimated bands using first-order allpass filters (QMF analysis).
///
/// This version writes low-pass and high-pass outputs into separate slices, matching
/// the C reference signature.
///
/// - `input`: input signal of length `len`
/// - `s`: filter state, 2 elements (preserved between calls)
/// - `out_lp`: low-pass (low band) output of length `len / 2`
/// - `out_hp`: high-pass (high band) output of length `len / 2`
/// - `len`: number of input samples (must be even)
pub fn silk_ana_filt_bank_1_separate(
    input: &[i16],
    s: &mut [i32; 2],
    out_lp: &mut [i16],
    out_hp: &mut [i16],
    len: usize,
) {
    let n2 = len >> 1;

    // Internal variables and state are in Q10 format
    for k in 0..n2 {
        // Convert to Q10
        let mut in32 = (input[2 * k] as i32) << 10;

        // All-pass section for even input sample
        let y = in32 - s[0];
        let x = silk_smlawb(y, y, A_FB1_21 as i32);
        let out_1 = s[0] + x;
        s[0] = in32 + x;

        // Convert to Q10
        in32 = (input[2 * k + 1] as i32) << 10;

        // All-pass section for odd input sample, and add to output of previous section
        let y = in32 - s[1];
        let x = silk_smulwb(y, A_FB1_20 as i32);
        let out_2 = s[1] + x;
        s[1] = in32 + x;

        // Add/subtract, convert back to int16 and store to output
        out_lp[k] = silk_sat16(silk_rshift_round(out_2 + out_1, 11));
        out_hp[k] = silk_sat16(silk_rshift_round(out_2 - out_1, 11));
    }
}

// ============================================================================
// 6. silk_warped_autocorrelation - Warped autocorrelation
// Ported from silk/fixed/warped_autocorrelation_FIX.c
// ============================================================================

/// Count leading zeros of a 64-bit integer.
#[inline(always)]
fn silk_clz64(input: i64) -> i32 {
    let in_upper = (input >> 32) as i32;
    if in_upper == 0 {
        // Search in the lower 32 bits
        32 + silk_clz32(input as i32)
    } else {
        // Search in the upper 32 bits
        silk_clz32(in_upper)
    }
}

/// Compute autocorrelation with warped frequency axis using allpass filter lattice.
///
/// This implements bilinear-warped autocorrelation for spectral analysis with
/// perceptually motivated frequency resolution.
///
/// - `corr`: output autocorrelation (length >= order + 1)
/// - `scale`: output scaling of the correlation vector
/// - `input`: input signal samples
/// - `warping_q16`: warping coefficient in Q16
/// - `length`: number of input samples
/// - `order`: correlation order (must be even)
pub fn silk_warped_autocorrelation(
    corr: &mut [i32],
    scale: &mut i32,
    input: &[i16],
    warping_q16: i32,
    length: usize,
    order: usize,
) {
    debug_assert!((order & 1) == 0);
    let mut state_qs = [0i32; MAX_SHAPE_LPC_ORDER + 1];
    let mut corr_qc = [0i64; MAX_SHAPE_LPC_ORDER + 1];

    // Loop over samples
    for item in input.iter().take(length) {
        let mut tmp1_qs = (*item as i32) << QS;
        // Loop over allpass sections
        let mut i = 0;
        while i < order {
            // Output of allpass section
            let tmp2_qs = silk_smlawb(state_qs[i], state_qs[i + 1] - tmp1_qs, warping_q16);
            state_qs[i] = tmp1_qs;
            corr_qc[i] += (tmp1_qs as i64 * state_qs[0] as i64) >> (2 * QS - QC);
            // Output of allpass section
            tmp1_qs = silk_smlawb(state_qs[i + 1], state_qs[i + 2] - tmp2_qs, warping_q16);
            state_qs[i + 1] = tmp2_qs;
            corr_qc[i + 1] += (tmp2_qs as i64 * state_qs[0] as i64) >> (2 * QS - QC);
            i += 2;
        }
        state_qs[order] = tmp1_qs;
        corr_qc[order] += (tmp1_qs as i64 * state_qs[0] as i64) >> (2 * QS - QC);
    }

    let mut lsh = silk_clz64(corr_qc[0]) - 35;
    lsh = lsh.clamp(-12 - QC, 30 - QC);
    *scale = -(QC + lsh);
    if lsh >= 0 {
        for i in 0..=order {
            corr[i] = (corr_qc[i] << lsh) as i32;
        }
    } else {
        for i in 0..=order {
            corr[i] = (corr_qc[i] >> (-lsh)) as i32;
        }
    }
}

// ============================================================================
// 7. silk_inner_prod_aligned - Fast inner product
// ============================================================================

/// Compute the inner (dot) product of two i16 vectors, returning i32.
///
/// Standard dot product: sum of in_vec1[i] * in_vec2[i] for i in 0..len.
pub fn silk_inner_prod_aligned(in_vec1: &[i16], in_vec2: &[i16], len: usize) -> i32 {
    let mut sum = 0i32;
    for i in 0..len {
        sum = sum.wrapping_add((in_vec1[i] as i32) * (in_vec2[i] as i32));
    }
    sum
}

// ============================================================================
// 7b. silk_inner_prod_aligned_scale - Inner product with per-product right-shift
// Ported from silk/inner_prod_aligned.c
// ============================================================================

/// Compute the inner product of two i16 vectors with a per-product right-shift.
///
/// Each product (in_vec1[i] * in_vec2[i]) is right-shifted by `scale` bits before
/// accumulation. This prevents overflow when working with high-energy signals.
pub fn silk_inner_prod_aligned_scale(
    in_vec1: &[i16],
    in_vec2: &[i16],
    scale: i32,
    len: usize,
) -> i32 {
    let mut sum = 0i32;
    for i in 0..len {
        let product = (in_vec1[i] as i32) * (in_vec2[i] as i32);
        // silk_ADD_RSHIFT32(sum, product, scale)
        sum = sum.wrapping_add(product >> scale);
    }
    sum
}

// ============================================================================
// 8. silk_resampler_down2 - Downsample by a factor 2
// Ported from silk/resampler_down2.c
// ============================================================================

/// Allpass filter coefficients for down2 resampler (from silk/resampler_rom.h)
const SILK_RESAMPLER_DOWN2_0: i16 = 9872;
const SILK_RESAMPLER_DOWN2_1: i16 = (39809 - 65536) as i16; // = -25727

/// Downsample by a factor 2, using allpass filter pair.
///
/// - `s`: state vector (2 elements), preserved between calls
/// - `out`: output signal, length = floor(in_len / 2)
/// - `input`: input signal, length = `in_len`
/// - `in_len`: number of input samples
pub fn silk_resampler_down2(s: &mut [i32; 2], out: &mut [i16], input: &[i16], in_len: usize) {
    let len2 = in_len >> 1;

    // Internal variables and state are in Q10 format
    for k in 0..len2 {
        // Convert to Q10
        let in32 = (input[2 * k] as i32) << 10;

        // All-pass section for even input sample
        let y = in32 - s[0];
        let x = silk_smlawb(y, y, SILK_RESAMPLER_DOWN2_1 as i32);
        let mut out32 = s[0] + x;
        s[0] = in32 + x;

        // Convert to Q10
        let in32 = (input[2 * k + 1] as i32) << 10;

        // All-pass section for odd input sample, and add to output of previous section
        let y = in32 - s[1];
        let x = silk_smulwb(y, SILK_RESAMPLER_DOWN2_0 as i32);
        out32 = out32 + s[1] + x;
        s[1] = in32 + x;

        // Add, convert back to int16 and store to output
        out[k] = silk_sat16(silk_rshift_round(out32, 11));
    }
}

// ============================================================================
// 9. silk_resampler_down2_3 - Downsample by 2/3 (12kHz -> 8kHz)
// Ported from silk/resampler_down2_3.c
// ============================================================================

/// AR filter coefficients for 2/3 resampler (from silk/resampler_rom.c)
const SILK_RESAMPLER_2_3_COEFS_LQ: [i16; 6] = [-2797, -6507, 4697, 10739, 1567, 8276];

/// Second order AR filter with single delay elements.
/// Ported from silk/resampler_private_AR2.c.
fn silk_resampler_private_ar2(
    s: &mut [i32],      // state [2]
    out_q8: &mut [i32], // output Q8
    input: &[i16],      // input
    a_q14: &[i16],      // AR coefficients, Q14
    len: usize,
) {
    for k in 0..len {
        let out32 = s[0].wrapping_add((input[k] as i32) << 8);
        out_q8[k] = out32;
        let out32_shifted = out32 << 2;
        s[0] = silk_smlawb(s[1], out32_shifted, a_q14[0] as i32);
        s[1] = silk_smulwb(out32_shifted, a_q14[1] as i32);
    }
}

/// Downsample by a factor 2/3, low quality.
/// Used for 12kHz -> 8kHz conversion.
///
/// - `s`: state vector (6 elements: [0..3] = FIR buffer, [4..5] = AR2 state)
/// - `out`: output signal, length = floor(2 * in_len / 3)
/// - `input`: input signal, length = `in_len`
/// - `in_len`: number of input samples
pub fn silk_resampler_down2_3(s: &mut [i32; 6], out: &mut [i16], input: &[i16], in_len: usize) {
    const ORDER_FIR: usize = 4;
    const BATCH_SIZE: usize = 480; // RESAMPLER_MAX_BATCH_SIZE_MS * RESAMPLER_MAX_FS_KHZ

    let mut buf = [0i32; BATCH_SIZE + ORDER_FIR];

    // Copy buffered samples to start of buffer
    buf[..ORDER_FIR].copy_from_slice(&s[..ORDER_FIR]);

    let mut in_offset = 0usize;
    let mut out_offset = 0usize;
    let mut remaining = in_len;

    loop {
        let n_samples_in = remaining.min(BATCH_SIZE);

        // Second-order AR filter (output in Q8)
        silk_resampler_private_ar2(
            &mut s[ORDER_FIR..],
            &mut buf[ORDER_FIR..ORDER_FIR + n_samples_in],
            &input[in_offset..],
            &SILK_RESAMPLER_2_3_COEFS_LQ[..2],
            n_samples_in,
        );

        // Interpolate filtered signal
        let mut buf_idx = 0usize;
        let mut counter = n_samples_in;
        while counter > 2 {
            // Inner product (first output sample per 3 inputs)
            let mut res_q6 = silk_smulwb(buf[buf_idx], SILK_RESAMPLER_2_3_COEFS_LQ[2] as i32);
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_idx + 1],
                SILK_RESAMPLER_2_3_COEFS_LQ[3] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_idx + 2],
                SILK_RESAMPLER_2_3_COEFS_LQ[5] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_idx + 3],
                SILK_RESAMPLER_2_3_COEFS_LQ[4] as i32,
            );
            out[out_offset] = silk_sat16(silk_rshift_round(res_q6, 6));
            out_offset += 1;

            // Inner product (second output sample per 3 inputs)
            res_q6 = silk_smulwb(buf[buf_idx + 1], SILK_RESAMPLER_2_3_COEFS_LQ[4] as i32);
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_idx + 2],
                SILK_RESAMPLER_2_3_COEFS_LQ[5] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_idx + 3],
                SILK_RESAMPLER_2_3_COEFS_LQ[3] as i32,
            );
            res_q6 = silk_smlawb(
                res_q6,
                buf[buf_idx + 4],
                SILK_RESAMPLER_2_3_COEFS_LQ[2] as i32,
            );
            out[out_offset] = silk_sat16(silk_rshift_round(res_q6, 6));
            out_offset += 1;

            buf_idx += 3;
            counter -= 3;
        }

        in_offset += n_samples_in;
        remaining -= n_samples_in;

        if remaining > 0 {
            // Copy last part of filtered signal to beginning of buffer
            for i in 0..ORDER_FIR {
                buf[i] = buf[n_samples_in + i];
            }
        } else {
            break;
        }
    }

    // Copy last part of filtered signal to the state for the next call
    let n_samples_in = in_len.min(BATCH_SIZE); // last batch size
    s[..ORDER_FIR].copy_from_slice(&buf[n_samples_in..n_samples_in + ORDER_FIR]);
}

// ============================================================================
// 10. silk_insertion_sort_decreasing_int16 - Partial insertion sort (decreasing)
// Ported from silk/sort.c
// ============================================================================

/// Partial insertion sort (decreasing order) for i16 values.
///
/// Sorts the array `a` in decreasing order, keeping track of the original
/// indices in `idx`. Only guarantees the first `k` elements are correctly
/// sorted (the top-K values).
///
/// - `a`: values array (modified in place), length >= `len`
/// - `idx`: index array (filled), length >= `k`
/// - `len`: number of elements in `a`
/// - `k`: number of correctly sorted positions needed
pub fn silk_insertion_sort_decreasing_int16(a: &mut [i16], idx: &mut [i32], len: usize, k: usize) {
    debug_assert!(k > 0 && len > 0 && len >= k);

    // Write start indices in index vector
    for (i, idx_i) in idx.iter_mut().enumerate().take(k) {
        *idx_i = i as i32;
    }

    // Sort first K vector elements by value, decreasing order
    for i in 1..k {
        let value = a[i] as i32;
        let mut j = i as i32 - 1;
        while j >= 0 && value > a[j as usize] as i32 {
            a[(j + 1) as usize] = a[j as usize];
            idx[(j + 1) as usize] = idx[j as usize];
            j -= 1;
        }
        a[(j + 1) as usize] = value as i16;
        idx[(j + 1) as usize] = i as i32;
    }

    // Check remaining values but only keep top K correct
    for i in k..len {
        let value = a[i] as i32;
        if value > a[k - 1] as i32 {
            let mut j = k as i32 - 2;
            while j >= 0 && value > a[j as usize] as i32 {
                a[(j + 1) as usize] = a[j as usize];
                idx[(j + 1) as usize] = idx[j as usize];
                j -= 1;
            }
            a[(j + 1) as usize] = value as i16;
            idx[(j + 1) as usize] = i as i32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigm_q15_boundaries() {
        // At input 0, sigmoid should be 0.5 -> 16384 in Q15
        assert_eq!(silk_sigm_q15(0), 16384);
        // Large positive input -> 32767
        assert_eq!(silk_sigm_q15(6 * 32), 32767);
        // Large negative input -> 0
        assert_eq!(silk_sigm_q15(-6 * 32), 0);
    }

    #[test]
    fn test_sigm_q15_symmetry() {
        // sigmoid(x) + sigmoid(-x) should be approximately 32767
        for x in 1..180 {
            let pos = silk_sigm_q15(x);
            let neg = silk_sigm_q15(-x);
            // The sum should be close to 32767 (full Q15 range)
            let sum = pos + neg;
            assert!(
                (sum - 32767).abs() <= 3,
                "x={}: pos={} neg={} sum={}",
                x,
                pos,
                neg,
                sum
            );
        }
    }

    #[test]
    fn test_inner_prod_aligned() {
        let a: [i16; 4] = [1, 2, 3, 4];
        let b: [i16; 4] = [5, 6, 7, 8];
        // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70
        assert_eq!(silk_inner_prod_aligned(&a, &b, 4), 70);
    }

    #[test]
    fn test_inner_prod_aligned_zero_length() {
        let a: [i16; 4] = [1, 2, 3, 4];
        let b: [i16; 4] = [5, 6, 7, 8];
        assert_eq!(silk_inner_prod_aligned(&a, &b, 0), 0);
    }

    #[test]
    fn test_k2a_order_1() {
        // For order 1, A_Q24[0] = -(rc_Q15[0] << 9)
        let rc_q15: [i16; 1] = [16384]; // 0.5 in Q15
        let mut a_q24 = [0i32; 1];
        silk_k2a(&mut a_q24, &rc_q15, 1);
        assert_eq!(a_q24[0], -(16384 << 9));
    }

    #[test]
    fn test_k2a_q16_order_1() {
        let rc_q16: [i32; 1] = [32768]; // 0.5 in Q16
        let mut a_q24 = [0i32; 1];
        silk_k2a_q16(&mut a_q24, &rc_q16, 1);
        assert_eq!(a_q24[0], -(32768 << 8));
    }

    #[test]
    fn test_schur_basic() {
        // Simple autocorrelation: [100, 50, 25]
        let c = [100, 50, 25];
        let mut rc_q15 = [0i16; 2];
        let residual = silk_schur(&mut rc_q15, &c, 2);
        // The function should produce valid reflection coefficients
        assert!(residual > 0);
        // rc values should be in valid range
        for &rc in &rc_q15 {
            assert!(rc >= -32767);
        }
    }

    #[test]
    fn test_schur64_zero_input() {
        let c = [0, 0, 0];
        let mut rc_q16 = [0i32; 2];
        let result = silk_schur64(&mut rc_q16, &c, 2);
        assert_eq!(result, 0);
        assert_eq!(rc_q16[0], 0);
        assert_eq!(rc_q16[1], 0);
    }

    #[test]
    fn test_ana_filt_bank_1() {
        // Simple test: all-zero input should produce all-zero output
        let input = [0i16; 8];
        let mut s = [0i32; 2];
        let mut out = [0i16; 8];
        silk_ana_filt_bank_1(&input, &mut s, &mut out, 0, 4, 8);
        assert_eq!(&out[0..4], &[0; 4]); // LP
        assert_eq!(&out[4..8], &[0; 4]); // HP
    }

    #[test]
    fn test_ana_filt_bank_1_separate() {
        // Simple test: all-zero input should produce all-zero output
        let input = [0i16; 8];
        let mut s = [0i32; 2];
        let mut out_lp = [0i16; 4];
        let mut out_hp = [0i16; 4];
        silk_ana_filt_bank_1_separate(&input, &mut s, &mut out_lp, &mut out_hp, 8);
        assert_eq!(out_lp, [0; 4]);
        assert_eq!(out_hp, [0; 4]);
    }

    #[test]
    fn test_apply_sine_window_rising() {
        // With win_type=1 (rising), the first sample should be near zero
        let px = [32767i16; 16];
        let mut px_win = [0i16; 16];
        silk_apply_sine_window(&mut px_win, &px, 1, 16);
        // First sample should be small (near zero since window starts at 0)
        assert!(px_win[0].abs() < 5000);
        // Last samples should be close to input (near window peak at pi/2)
        assert!(px_win[15] > 20000);
    }

    #[test]
    fn test_apply_sine_window_falling() {
        // With win_type=2 (falling), the first sample should be near input magnitude
        let px = [32767i16; 16];
        let mut px_win = [0i16; 16];
        silk_apply_sine_window(&mut px_win, &px, 2, 16);
        // First sample should be close to input (window starts at 1)
        assert!(px_win[0] > 20000);
    }

    #[test]
    fn test_warped_autocorrelation_silence() {
        use crate::MAX_LPC_ORDER;
        let input = [0i16; 64];
        let mut corr = [0i32; MAX_LPC_ORDER + 1];
        let mut scale = 0i32;
        silk_warped_autocorrelation(&mut corr, &mut scale, &input, 0, 64, 4);
        // All correlations should be zero for silent input
        for &c in corr.iter().take(5) {
            assert_eq!(c, 0);
        }
    }
}
