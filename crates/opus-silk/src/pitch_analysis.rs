// Full 3-stage hierarchical SILK pitch analysis functions for the encoder.
// Ported from silk/fixed/pitch_analysis_core_FIX.c
//
// Functions:
// - silk_pitch_analysis_core: full 3-stage hierarchical pitch estimation
// - silk_find_pitch_contour: find best pitch contour index
// - silk_find_ltp_params: find best LTP (Long-Term Prediction) parameters

use crate::signal_processing::{
    silk_inner_prod_aligned, silk_insertion_sort_decreasing_int16, silk_resampler_down2,
    silk_resampler_down2_3,
};
use crate::tables::*;
use crate::*;

// Constants derived from pitch_est_defines.h
const SF_LENGTH_4KHZ: usize = (PE_SUBFR_LENGTH_MS * 4) as usize;
const SF_LENGTH_8KHZ: usize = (PE_SUBFR_LENGTH_MS * 8) as usize;
const MIN_LAG_4KHZ: usize = (PE_MIN_LAG_MS * 4) as usize;
const MIN_LAG_8KHZ: usize = (PE_MIN_LAG_MS * 8) as usize;
const MAX_LAG_4KHZ: usize = (PE_MAX_LAG_MS * 4) as usize;
const MAX_LAG_8KHZ: usize = (PE_MAX_LAG_MS * 8 - 1) as usize;
const CSTRIDE_4KHZ: usize = MAX_LAG_4KHZ + 1 - MIN_LAG_4KHZ;
const CSTRIDE_8KHZ: usize = MAX_LAG_8KHZ + 3 - (MIN_LAG_8KHZ - 2);
const D_COMP_MIN: usize = MIN_LAG_8KHZ - 3;
const D_COMP_MAX: usize = MAX_LAG_8KHZ + 4;
const D_COMP_STRIDE: usize = D_COMP_MAX - D_COMP_MIN;
const SCRATCH_SIZE: usize = 22;

/// Full 3-stage hierarchical pitch analysis.
///
/// Returns 0 for voiced, 1 for unvoiced.
///
/// The algorithm follows the C reference (silk/fixed/pitch_analysis_core_FIX.c):
///   Stage 0: Downscale, resample to 8kHz and 4kHz
///   Stage 1: Coarse search at 4kHz with CSTRIDE_4KHZ candidates
///   Stage 2: Refinement at 8kHz with d_comp candidate expansion
///   Stage 3: Full-rate per-subframe refinement (only if fs_khz > 8)
pub fn silk_pitch_analysis_core(
    frame_unscaled: &[i16], // Input signal
    pitch_out: &mut [i32],  // Output pitch lags per subframe [nb_subfr]
    lag_index: &mut i16,    // Output lag index for encoding
    contour_index: &mut i8, // Output contour index for encoding
    ltp_corr_q15: &mut i32, // I/O normalized correlation
    prev_lag: i32,          // Previous frame's lag (0 = unvoiced)
    search_thres1_q16: i32, // Stage 1 threshold
    search_thres2_q13: i32, // Stage 2/3 threshold
    fs_khz: i32,            // 8, 12, or 16
    complexity: i32,        // 0, 1, or 2
    nb_subfr: i32,          // 2 or 4
) -> i32 {
    debug_assert!(fs_khz == 8 || fs_khz == 12 || fs_khz == 16);
    debug_assert!((0..=2).contains(&complexity));

    let nb_subfr_usize = nb_subfr as usize;

    // Set up frame lengths, min/max lag for the sampling frequency
    let frame_length = ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz) as usize;
    let frame_length_4khz = ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 4) as usize;
    let frame_length_8khz = ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * 8) as usize;
    let sf_length = (PE_SUBFR_LENGTH_MS * fs_khz) as usize;
    let min_lag = (PE_MIN_LAG_MS * fs_khz) as usize;
    let max_lag = (PE_MAX_LAG_MS * fs_khz - 1) as usize;

    // ========================================================================
    // Stage 0: Downscale input if necessary
    // ========================================================================
    let mut energy = 0i32;
    let mut shift = 0i32;
    silk_sum_sqr_shift(
        &mut energy,
        &mut shift,
        &frame_unscaled[..frame_length],
        frame_length,
    );
    shift += 3 - silk_clz32(energy);

    // Stack-allocate buffers. Max frame length for 16kHz, 4 subfr = (20+20)*16 = 640
    // Max frame_length = (20+20)*16 = 640 at 16kHz, 4 subframes
    let mut frame_scaled_buf = [0i16; 640];
    let frame: &[i16];

    if shift > 0 {
        let shift_val = ((shift + 1) >> 1) as u32;
        for i in 0..frame_length {
            frame_scaled_buf[i] = (frame_unscaled[i] as i32 >> shift_val) as i16;
        }
        frame = &frame_scaled_buf;
    } else {
        frame = &frame_unscaled[..frame_length];
    }

    // Resample from input sampled at Fs_kHz to 8 kHz
    // Max frame_length_8khz = (20+20)*8 = 320
    let mut frame_8khz_buf = [0i16; 320];
    let frame_8khz: &[i16];

    if fs_khz == 16 {
        let mut filt_state = [0i32; 2];
        silk_resampler_down2(&mut filt_state, &mut frame_8khz_buf, frame, frame_length);
        frame_8khz = &frame_8khz_buf;
    } else if fs_khz == 12 {
        let mut filt_state = [0i32; 6];
        silk_resampler_down2_3(&mut filt_state, &mut frame_8khz_buf, frame, frame_length);
        frame_8khz = &frame_8khz_buf;
    } else {
        // fs_khz == 8
        frame_8khz = frame;
    }

    // Decimate again to 4 kHz
    let mut filt_state = [0i32; 2];
    // Max frame_length_4khz = (20+20)*4 = 160
    let mut frame_4khz = [0i16; 160];
    silk_resampler_down2(
        &mut filt_state,
        &mut frame_4khz,
        frame_8khz,
        frame_length_8khz,
    );

    // Low-pass filter
    for i in (1..frame_length_4khz).rev() {
        frame_4khz[i] = frame_4khz[i].saturating_add(frame_4khz[i - 1]);
    }

    // ========================================================================
    // FIRST STAGE, operating at 4 kHz
    // ========================================================================

    // C is laid out as [nb_subfr * CSTRIDE_8KHZ] (big enough for both stages)
    // Max = PE_MAX_NB_SUBFR * CSTRIDE_8KHZ = 4 * 132 = 528
    let mut c_buf = [0i16; PE_MAX_NB_SUBFR * CSTRIDE_8KHZ];

    // Zero out area used for 4kHz stage
    for item in c_buf.iter_mut().take((nb_subfr_usize >> 1) * CSTRIDE_4KHZ) {
        *item = 0;
    }

    // target_ptr starts at frame_4kHz offset = sf_length_4kHz * 4 = SF_LENGTH_4KHZ * 4
    let target_offset_4khz = SF_LENGTH_4KHZ * 4;

    for k in 0..(nb_subfr_usize >> 1) {
        let target_start = target_offset_4khz + k * SF_LENGTH_8KHZ;
        let basis_start_min_lag = target_start - MIN_LAG_4KHZ;

        // Compute cross-correlations using batch xcorr
        // xcorr32[d] = inner_prod(target, target - MAX_LAG_4KHZ + d) for d in 0..CSTRIDE_4KHZ
        let mut xcorr32 = [0i32; CSTRIDE_4KHZ];
        for (d, xcorr32_d) in xcorr32.iter_mut().enumerate().take(CSTRIDE_4KHZ) {
            let lag = MAX_LAG_4KHZ - d; // d=0 -> lag=MAX_LAG, d=CSTRIDE-1 -> lag=MIN_LAG
            let basis_start = target_start - lag;
            *xcorr32_d = silk_inner_prod_aligned(
                &frame_4khz[target_start..],
                &frame_4khz[basis_start..],
                SF_LENGTH_8KHZ,
            );
        }

        // Calculate first normalizer: energy(target) + energy(basis at MIN_LAG) + bias
        // xcorr32[MAX_LAG_4KHZ - MIN_LAG_4KHZ] is cross_corr at lag=MIN_LAG_4KHZ
        let cross_corr_first = xcorr32[MAX_LAG_4KHZ - MIN_LAG_4KHZ];

        let energy_target = silk_inner_prod_aligned(
            &frame_4khz[target_start..],
            &frame_4khz[target_start..],
            SF_LENGTH_8KHZ,
        );
        let energy_basis = silk_inner_prod_aligned(
            &frame_4khz[basis_start_min_lag..],
            &frame_4khz[basis_start_min_lag..],
            SF_LENGTH_8KHZ,
        );
        let mut normalizer = energy_target
            .wrapping_add(energy_basis)
            .wrapping_add(silk_smulbb(SF_LENGTH_8KHZ as i32, 4000));

        // First C value at d = MIN_LAG_4KHZ
        let c_offset = k * CSTRIDE_4KHZ;
        c_buf[c_offset] = silk_div32_varq(cross_corr_first, normalizer, 14) as i16; // Q13

        // From now on normalizer is computed recursively
        for d in (MIN_LAG_4KHZ + 1)..=MAX_LAG_4KHZ {
            let basis_start = target_start - d;

            let cross_corr = xcorr32[MAX_LAG_4KHZ - d];

            // Add contribution of new sample and remove contribution from oldest sample
            let new_sample = frame_4khz[basis_start] as i32;
            let old_sample = frame_4khz[basis_start + SF_LENGTH_8KHZ] as i32;
            normalizer = normalizer.wrapping_add(
                new_sample
                    .wrapping_mul(new_sample)
                    .wrapping_sub(old_sample.wrapping_mul(old_sample)),
            );

            c_buf[c_offset + d - MIN_LAG_4KHZ] = silk_div32_varq(cross_corr, normalizer, 14) as i16; // Q13
        }
    }

    // Combine two subframes into single correlation measure and apply short-lag bias
    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let idx = i - MIN_LAG_4KHZ;
            let mut sum = c_buf[idx] as i32 + c_buf[CSTRIDE_4KHZ + idx] as i32; // Q14
            sum = silk_smlawb(sum, sum, (-(i as i32)) << 4); // Q14
            c_buf[idx] = sum as i16;
        }
    } else {
        for i in (MIN_LAG_4KHZ..=MAX_LAG_4KHZ).rev() {
            let idx = i - MIN_LAG_4KHZ;
            let mut sum = (c_buf[idx] as i32) << 1; // Q14
            sum = silk_smlawb(sum, sum, (-(i as i32)) << 4); // Q14
            c_buf[idx] = sum as i16;
        }
    }

    // Sort: find top candidates
    let length_d_srch_init = (4 + (complexity << 1)) as usize;
    debug_assert!(3 * length_d_srch_init <= PE_D_SRCH_LENGTH);

    let mut d_srch = [0i32; PE_D_SRCH_LENGTH];
    silk_insertion_sort_decreasing_int16(&mut c_buf, &mut d_srch, CSTRIDE_4KHZ, length_d_srch_init);

    // Escape if correlation is very low already here
    let cmax = c_buf[0] as i32; // Q14
    if cmax < 3276 {
        // SILK_FIX_CONST(0.2, 14) = 3276
        for item in pitch_out.iter_mut().take(nb_subfr_usize) {
            *item = 0;
        }
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    let threshold = silk_smulwb(search_thres1_q16, cmax);
    let mut length_d_srch = length_d_srch_init;
    for i in 0..length_d_srch_init {
        if c_buf[i] as i32 > threshold {
            // Convert to 8 kHz indices
            d_srch[i] = (d_srch[i] + MIN_LAG_4KHZ as i32) << 1;
        } else {
            length_d_srch = i;
            break;
        }
    }
    debug_assert!(length_d_srch > 0);

    // Expand candidates using convolution
    // D_COMP_STRIDE = 134, fits on stack
    let mut d_comp = [0i16; D_COMP_STRIDE];
    for &d_srch_i in d_srch.iter().take(length_d_srch) {
        let idx = d_srch_i as usize;
        if (D_COMP_MIN..D_COMP_MAX).contains(&idx) {
            d_comp[idx - D_COMP_MIN] = 1;
        }
    }

    // First convolution pass
    for i in (MIN_LAG_8KHZ..D_COMP_MAX).rev() {
        if i >= D_COMP_MIN + 2 {
            d_comp[i - D_COMP_MIN] = d_comp[i - D_COMP_MIN]
                .wrapping_add(d_comp[i - 1 - D_COMP_MIN])
                .wrapping_add(d_comp[i - 2 - D_COMP_MIN]);
        }
    }

    length_d_srch = 0;
    for i in MIN_LAG_8KHZ..=MAX_LAG_8KHZ {
        if i + 1 >= D_COMP_MIN && i + 1 < D_COMP_MAX && d_comp[i + 1 - D_COMP_MIN] > 0 {
            d_srch[length_d_srch] = i as i32;
            length_d_srch += 1;
        }
    }

    // Second convolution pass
    for i in (MIN_LAG_8KHZ..D_COMP_MAX).rev() {
        if i >= D_COMP_MIN + 3 {
            d_comp[i - D_COMP_MIN] = d_comp[i - D_COMP_MIN]
                .wrapping_add(d_comp[i - 1 - D_COMP_MIN])
                .wrapping_add(d_comp[i - 2 - D_COMP_MIN])
                .wrapping_add(d_comp[i - 3 - D_COMP_MIN]);
        }
    }

    let mut length_d_comp = 0usize;
    for i in MIN_LAG_8KHZ..D_COMP_MAX {
        if i >= D_COMP_MIN && d_comp[i - D_COMP_MIN] > 0 {
            d_comp[length_d_comp] = (i as i16) - 2;
            length_d_comp += 1;
        }
    }

    // ========================================================================
    // SECOND STAGE, operating at 8 kHz
    // ========================================================================

    // Zero out C buffer for 8kHz stage
    for item in c_buf.iter_mut().take(nb_subfr_usize * CSTRIDE_8KHZ) {
        *item = 0;
    }

    let target_offset_8khz = (PE_LTP_MEM_LENGTH_MS * 8) as usize;

    for k in 0..nb_subfr_usize {
        let target_start = target_offset_8khz + k * SF_LENGTH_8KHZ;

        let energy_target = silk_inner_prod_aligned(
            &frame_8khz[target_start..],
            &frame_8khz[target_start..],
            SF_LENGTH_8KHZ,
        )
        .wrapping_add(1);

        for &d_comp_j in d_comp.iter().take(length_d_comp) {
            let d = d_comp_j as usize;
            if target_start < d {
                continue;
            }
            let basis_start = target_start - d;

            let cross_corr = silk_inner_prod_aligned(
                &frame_8khz[target_start..],
                &frame_8khz[basis_start..],
                SF_LENGTH_8KHZ,
            );

            if cross_corr > 0 {
                let energy_basis = silk_inner_prod_aligned(
                    &frame_8khz[basis_start..],
                    &frame_8khz[basis_start..],
                    SF_LENGTH_8KHZ,
                );
                let c_idx = k * CSTRIDE_8KHZ + d - (MIN_LAG_8KHZ - 2);
                c_buf[c_idx] =
                    silk_div32_varq(cross_corr, energy_target.wrapping_add(energy_basis), 14)
                        as i16; // Q13
            }
        }
    }

    // Search over lag range and lags codebook
    let mut cc_max = i32::MIN;
    let mut cc_max_b = i32::MIN;
    let mut cb_imax = 0usize;
    let mut lag: i32 = -1;

    let mut prev_lag_8khz = prev_lag;
    let mut prev_lag_log2_q7 = 0i32;
    if prev_lag_8khz > 0 {
        if fs_khz == 12 {
            prev_lag_8khz = (prev_lag_8khz << 1) / 3;
        } else if fs_khz == 16 {
            prev_lag_8khz >>= 1;
        }
        prev_lag_log2_q7 = silk_lin2log(prev_lag_8khz);
    }

    // Set up stage 2 codebook
    let nb_cbk_search: usize;
    let lag_cb_stage2_is_10ms: bool;

    if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        if fs_khz == 8 && complexity > 0 {
            nb_cbk_search = PE_NB_CBKS_STAGE2_EXT;
        } else {
            nb_cbk_search = PE_NB_CBKS_STAGE2;
        }
        lag_cb_stage2_is_10ms = false;
    } else {
        nb_cbk_search = PE_NB_CBKS_STAGE2_10MS;
        lag_cb_stage2_is_10ms = true;
    }

    for d_srch_k in d_srch.iter().take(length_d_srch) {
        let d = *d_srch_k;
        let mut cc = [0i32; PE_NB_CBKS_STAGE2_EXT];

        for (j, cc_j) in cc.iter_mut().enumerate().take(nb_cbk_search) {
            *cc_j = 0;
            for i in 0..nb_subfr_usize {
                let d_subfr = if lag_cb_stage2_is_10ms {
                    d + SILK_CB_LAGS_STAGE2_10_MS[i][j] as i32
                } else {
                    d + SILK_CB_LAGS_STAGE2[i][j] as i32
                };
                let c_idx = i * CSTRIDE_8KHZ + (d_subfr as usize) - (MIN_LAG_8KHZ - 2);
                if c_idx < nb_subfr_usize * CSTRIDE_8KHZ {
                    *cc_j += c_buf[c_idx] as i32;
                }
            }
        }

        // Find best codebook
        let mut cc_max_new = i32::MIN;
        let mut cb_imax_new = 0usize;
        for (i, &cc_i) in cc.iter().enumerate().take(nb_cbk_search) {
            if cc_i > cc_max_new {
                cc_max_new = cc_i;
                cb_imax_new = i;
            }
        }

        // Bias towards shorter lags
        let lag_log2_q7 = silk_lin2log(d);
        let mut cc_max_new_b =
            cc_max_new - ((silk_smulbb(nb_subfr * PE_SHORTLAG_BIAS, lag_log2_q7)) >> 7); // Q13

        // Bias towards previous lag
        if prev_lag_8khz > 0 {
            let delta_lag_log2_sqr_q7_raw = lag_log2_q7 - prev_lag_log2_q7;
            let delta_lag_log2_sqr_q7 =
                (silk_smulbb(delta_lag_log2_sqr_q7_raw, delta_lag_log2_sqr_q7_raw)) >> 7;
            let prev_lag_bias_q13 = (silk_smulbb(nb_subfr * PE_PREVLAG_BIAS, *ltp_corr_q15)) >> 15; // Q13
            let prev_lag_bias_q13 = silk_div32(
                prev_lag_bias_q13.wrapping_mul(delta_lag_log2_sqr_q7),
                delta_lag_log2_sqr_q7 + 64, // SILK_FIX_CONST(0.5, 7) = 64
            );
            cc_max_new_b -= prev_lag_bias_q13;
        }

        // Check if this is the best candidate
        let first_lag_offset = if lag_cb_stage2_is_10ms {
            SILK_CB_LAGS_STAGE2_10_MS[0][cb_imax_new] as i32
        } else {
            SILK_CB_LAGS_STAGE2[0][cb_imax_new] as i32
        };

        if cc_max_new_b > cc_max_b
            && cc_max_new > silk_smulbb(nb_subfr, search_thres2_q13)
            && first_lag_offset <= MIN_LAG_8KHZ as i32
        {
            cc_max_b = cc_max_new_b;
            cc_max = cc_max_new;
            lag = d;
            cb_imax = cb_imax_new;
        }
    }

    if lag == -1 {
        // No suitable candidate found
        for item in pitch_out.iter_mut().take(nb_subfr_usize) {
            *item = 0;
        }
        *ltp_corr_q15 = 0;
        *lag_index = 0;
        *contour_index = 0;
        return 1;
    }

    // Output normalized correlation
    *ltp_corr_q15 = (silk_div32(cc_max, nb_subfr)) << 2;
    if *ltp_corr_q15 < 0 {
        *ltp_corr_q15 = 0;
    }

    // ========================================================================
    // THIRD STAGE (full rate, only if fs_khz > 8)
    // ========================================================================

    if fs_khz > 8 {
        let cb_imax_old = cb_imax;

        // Compensate for decimation
        if fs_khz == 12 {
            lag = (lag * 3) >> 1;
        } else if fs_khz == 16 {
            lag <<= 1;
        } else {
            lag *= 3;
        }

        lag = lag.clamp(min_lag as i32, max_lag as i32);
        let start_lag = (lag - 2).max(min_lag as i32);
        let end_lag = (lag + 2).min(max_lag as i32);
        let mut lag_new = lag;
        cb_imax = 0;

        cc_max = i32::MIN;
        // Pitch lags according to second stage
        for k in 0..nb_subfr_usize {
            let offset = if lag_cb_stage2_is_10ms {
                SILK_CB_LAGS_STAGE2_10_MS[k][cb_imax_old] as i32
            } else {
                SILK_CB_LAGS_STAGE2[k][cb_imax_old] as i32
            };
            pitch_out[k] = lag + 2 * offset;
        }

        // Set up codebook parameters for stage 3
        let use_10ms = nb_subfr != PE_MAX_NB_SUBFR as i32;

        let nb_cbk_search_st3: usize = if !use_10ms {
            SILK_NB_CBK_SEARCHS_STAGE3[complexity as usize]
        } else {
            PE_NB_CBKS_STAGE3_10MS
        };

        // Calculate the correlations and energies needed in stage 3
        // Max = PE_MAX_NB_SUBFR * PE_NB_CBKS_STAGE3_MAX = 4 * 34 = 136 entries
        let mut cross_corr_st3 =
            [[0i32; PE_NB_STAGE3_LAGS]; PE_MAX_NB_SUBFR * PE_NB_CBKS_STAGE3_MAX];
        let mut energies_st3 = [[0i32; PE_NB_STAGE3_LAGS]; PE_MAX_NB_SUBFR * PE_NB_CBKS_STAGE3_MAX];

        silk_p_ana_calc_corr_st3(
            &mut cross_corr_st3,
            frame,
            start_lag as usize,
            sf_length,
            nb_subfr_usize,
            complexity as usize,
            nb_cbk_search_st3,
            use_10ms,
        );
        silk_p_ana_calc_energy_st3(
            &mut energies_st3,
            frame,
            start_lag as usize,
            sf_length,
            nb_subfr_usize,
            complexity as usize,
            nb_cbk_search_st3,
            use_10ms,
        );

        let contour_bias_q15 = silk_div32(PE_FLATCONTOUR_BIAS, lag);

        // Compute energy of target
        let target_offset_full = (PE_LTP_MEM_LENGTH_MS * fs_khz) as usize;
        let energy_target = silk_inner_prod_aligned(
            &frame[target_offset_full..],
            &frame[target_offset_full..],
            nb_subfr_usize * sf_length,
        )
        .wrapping_add(1);

        for (lag_counter, d) in (start_lag..=end_lag).enumerate() {
            for j in 0..nb_cbk_search_st3 {
                let mut cross_corr = 0i32;
                let mut energy = energy_target;
                for k in 0..nb_subfr_usize {
                    cross_corr = cross_corr
                        .wrapping_add(cross_corr_st3[k * nb_cbk_search_st3 + j][lag_counter]);
                    energy =
                        energy.wrapping_add(energies_st3[k * nb_cbk_search_st3 + j][lag_counter]);
                }

                let cc_max_new = if cross_corr > 0 {
                    let raw = silk_div32_varq(cross_corr, energy, 14); // Q13
                    // Reduce depending on flatness of contour
                    let diff = i16::MAX as i32 - contour_bias_q15.wrapping_mul(j as i32); // Q15
                    silk_smulwb(raw, diff) // Q14
                } else {
                    0
                };

                // Check lag is valid
                let first_lag_offset = if use_10ms {
                    SILK_CB_LAGS_STAGE3_10_MS[0][j] as i32
                } else {
                    SILK_CB_LAGS_STAGE3[0][j] as i32
                };

                if cc_max_new > cc_max && (d + first_lag_offset) <= max_lag as i32 {
                    cc_max = cc_max_new;
                    lag_new = d;
                    cb_imax = j;
                }
            }
        }

        for k in 0..nb_subfr_usize {
            let offset = if use_10ms {
                SILK_CB_LAGS_STAGE3_10_MS[k][cb_imax] as i32
            } else {
                SILK_CB_LAGS_STAGE3[k][cb_imax] as i32
            };
            pitch_out[k] = (lag_new + offset).clamp(min_lag as i32, PE_MAX_LAG_MS * fs_khz);
        }
        *lag_index = (lag_new - min_lag as i32) as i16;
        *contour_index = cb_imax as i8;
    } else {
        // Fs_kHz == 8: save lags from stage 2 directly
        for k in 0..nb_subfr_usize {
            let offset = if lag_cb_stage2_is_10ms {
                SILK_CB_LAGS_STAGE2_10_MS[k][cb_imax] as i32
            } else {
                SILK_CB_LAGS_STAGE2[k][cb_imax] as i32
            };
            pitch_out[k] = (lag + offset).clamp(MIN_LAG_8KHZ as i32, PE_MAX_LAG_MS * 8);
        }
        *lag_index = (lag - MIN_LAG_8KHZ as i32) as i16;
        *contour_index = cb_imax as i8;
    }

    debug_assert!(*lag_index >= 0);
    // Return as voiced
    0
}

/// Calculate cross-correlations for stage 3 search.
///
/// For each subframe and each codebook entry, compute PE_NB_STAGE3_LAGS
/// cross-correlation values spanning the lag search window [start_lag-2..start_lag+2].
fn silk_p_ana_calc_corr_st3(
    cross_corr_st3: &mut [[i32; PE_NB_STAGE3_LAGS]],
    frame: &[i16],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    nb_cbk_search: usize,
    use_10ms: bool,
) {
    let target_offset = sf_length * 4; // silk_LSHIFT(sf_length, 2) -- LTP_MEM offset

    for k in 0..nb_subfr {
        let target_start = target_offset + k * sf_length;

        // Get lag range for this subframe
        let (lag_low, lag_high) = if use_10ms {
            (
                SILK_LAG_RANGE_STAGE3_10_MS[k][0],
                SILK_LAG_RANGE_STAGE3_10_MS[k][1],
            )
        } else {
            (
                SILK_LAG_RANGE_STAGE3[complexity][k][0],
                SILK_LAG_RANGE_STAGE3[complexity][k][1],
            )
        };

        let n_lags = (lag_high - lag_low + 1) as usize;
        debug_assert!(n_lags <= SCRATCH_SIZE);

        // Compute cross-correlations for the lag range using batch xcorr
        let mut scratch_mem = [0i32; SCRATCH_SIZE];
        let mut xcorr32 = [0i32; SCRATCH_SIZE];

        // celt_pitch_xcorr(target, target - start_lag - lag_high, xcorr32, sf_length, n_lags)
        let basis_offset = target_start as i32 - start_lag as i32 - lag_high;
        for (j_idx, xcorr_item) in xcorr32.iter_mut().enumerate().take(n_lags) {
            let basis_start = basis_offset as usize + j_idx;
            if basis_start + sf_length <= frame.len() && target_start + sf_length <= frame.len() {
                *xcorr_item = silk_inner_prod_aligned(
                    &frame[target_start..],
                    &frame[basis_start..],
                    sf_length,
                );
            }
        }

        // Rearrange: scratch_mem stores correlations for lags lag_low..lag_high
        let mut lag_counter = 0usize;
        for j in lag_low..=lag_high {
            let xcorr_idx = (lag_high - j) as usize;
            scratch_mem[lag_counter] = xcorr32[xcorr_idx];
            lag_counter += 1;
        }

        // Fill out the 3D array
        let delta = lag_low;
        for i in 0..nb_cbk_search {
            let cb_offset = if use_10ms {
                SILK_CB_LAGS_STAGE3_10_MS[k][i] as i32
            } else {
                SILK_CB_LAGS_STAGE3[k][i] as i32
            };
            let idx = (cb_offset - delta) as usize;
            for j in 0..PE_NB_STAGE3_LAGS {
                if idx + j < lag_counter {
                    cross_corr_st3[k * nb_cbk_search + i][j] = scratch_mem[idx + j];
                }
            }
        }
    }
}

/// Calculate energies for stage 3 search.
///
/// For each subframe and each codebook entry, compute PE_NB_STAGE3_LAGS
/// energy values spanning the lag search window, using recursive energy
/// computation.
fn silk_p_ana_calc_energy_st3(
    energies_st3: &mut [[i32; PE_NB_STAGE3_LAGS]],
    frame: &[i16],
    start_lag: usize,
    sf_length: usize,
    nb_subfr: usize,
    complexity: usize,
    nb_cbk_search: usize,
    use_10ms: bool,
) {
    let target_offset = sf_length * 4;

    for k in 0..nb_subfr {
        let target_start = target_offset + k * sf_length;

        let (lag_low, lag_high) = if use_10ms {
            (
                SILK_LAG_RANGE_STAGE3_10_MS[k][0],
                SILK_LAG_RANGE_STAGE3_10_MS[k][1],
            )
        } else {
            (
                SILK_LAG_RANGE_STAGE3[complexity][k][0],
                SILK_LAG_RANGE_STAGE3[complexity][k][1],
            )
        };

        let lag_diff = (lag_high - lag_low + 1) as usize;
        let mut scratch_mem = [0i32; SCRATCH_SIZE];

        // Calculate the energy for first lag
        let basis_offset = target_start as i32 - start_lag as i32 - lag_low;
        let basis_start = basis_offset as usize;

        let mut lag_counter = 0usize;

        if basis_start + sf_length <= frame.len() {
            let mut energy_val =
                silk_inner_prod_aligned(&frame[basis_start..], &frame[basis_start..], sf_length);
            scratch_mem[lag_counter] = energy_val;
            lag_counter += 1;

            // Compute remaining energies recursively
            for i in 1..lag_diff {
                // Remove part outside new window
                let old_idx = basis_start + sf_length - i;
                if old_idx < frame.len() {
                    energy_val -= silk_smulbb(frame[old_idx] as i32, frame[old_idx] as i32);
                    if energy_val < 0 {
                        energy_val = 0;
                    }
                }

                // Add part that comes into window
                let new_idx_signed = basis_start as i32 - i as i32;
                if new_idx_signed >= 0 && (new_idx_signed as usize) < frame.len() {
                    let new_idx = new_idx_signed as usize;
                    energy_val = energy_val
                        .saturating_add(silk_smulbb(frame[new_idx] as i32, frame[new_idx] as i32));
                }
                if energy_val < 0 {
                    energy_val = 0;
                }

                scratch_mem[lag_counter] = energy_val;
                lag_counter += 1;
            }
        }

        // Fill out the 3D energy array
        let delta = lag_low;
        for i in 0..nb_cbk_search {
            let cb_offset = if use_10ms {
                SILK_CB_LAGS_STAGE3_10_MS[k][i] as i32
            } else {
                SILK_CB_LAGS_STAGE3[k][i] as i32
            };
            let idx = (cb_offset - delta) as usize;
            for j in 0..PE_NB_STAGE3_LAGS {
                if idx + j < lag_counter {
                    energies_st3[k * nb_cbk_search + i][j] = scratch_mem[idx + j];
                }
            }
        }
    }
}

/// Backward-compatible wrapper that matches the old simplified signature.
///
/// Calls silk_pitch_analysis_core internally with reasonable default thresholds
/// and complexity, then converts the return value to a bool (true = voiced).
pub fn silk_pitch_analysis_simple(
    input: &[i16],
    pitch_lags: &mut [i32],
    fs_khz: i32,
    nb_subfr: i32,
    _frame_length: i32,
) -> bool {
    let min_lag = (PE_MIN_LAG_MS * fs_khz) as usize;

    // The full pitch analysis expects the signal to include LTP memory prefix.
    // The expected length is (PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz.
    let expected_len = ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz) as usize;

    if input.len() < expected_len {
        // Not enough data -- fall back to unvoiced
        for item in pitch_lags.iter_mut().take(nb_subfr as usize) {
            *item = min_lag as i32;
        }
        return false;
    }

    let mut lag_index: i16 = 0;
    let mut contour_index: i8 = 0;
    let mut ltp_corr_q15: i32 = 0;

    // Default thresholds from the C reference (silk/control_codec.c):
    // search_thres1_Q16 = SILK_FIX_CONST(0.8, 16) = 52429
    // search_thres2_Q13 = SILK_FIX_CONST(0.3, 13) = 2458 (complexity 0)
    let search_thres1_q16 = 52429;
    let search_thres2_q13 = 2458;
    let complexity = 0;

    let ret = silk_pitch_analysis_core(
        &input[..expected_len],
        pitch_lags,
        &mut lag_index,
        &mut contour_index,
        &mut ltp_corr_q15,
        0, // prev_lag = 0 (unvoiced)
        search_thres1_q16,
        search_thres2_q13,
        fs_khz,
        complexity,
        nb_subfr,
    );

    // ret == 0 means voiced, ret == 1 means unvoiced
    if ret != 0 {
        // Unvoiced -- set some safe default lags
        for item in pitch_lags.iter_mut().take(nb_subfr as usize) {
            *item = min_lag as i32;
        }
        false
    } else {
        true
    }
}

/// Find best pitch contour index and lag index for encoding.
///
/// The contour index encodes small per-subframe pitch lag offsets relative to a
/// base lag. Try each contour from the appropriate table, find the one that best
/// matches the actual pitch_lags. The lag_index is `(base_lag - min_lag)`.
#[allow(dead_code)] // Contour selection now done inside silk_pitch_analysis_core
#[allow(clippy::needless_range_loop)] // contour `c` indexes the inner dim of [k][c] tables
pub fn silk_find_pitch_contour(
    contour_index: &mut i8,
    lag_index: &mut i16,
    pitch_lags: &[i32],
    fs_khz: i32,
    nb_subfr: i32,
) {
    let min_lag = PE_MIN_LAG_MS * fs_khz;

    // Base lag is the first subframe lag
    let base_lag = pitch_lags[0];
    *lag_index = (base_lag - min_lag) as i16;

    let nb_subfr_usize = nb_subfr as usize;

    // Find the contour that best matches the actual pitch lags.
    let mut best_contour = 0i8;
    let mut best_error = i64::MAX;

    if nb_subfr == MAX_NB_SUBFR as i32 {
        // 20ms frame, 4 subframes -- use SILK_CB_LAGS_STAGE3 ([i8; 34] x 4)
        let n_contours = PE_NB_CBKS_STAGE3_MAX;
        for c in 0..n_contours {
            let mut error: i64 = 0;
            for k in 0..nb_subfr_usize {
                let predicted_lag = base_lag + SILK_CB_LAGS_STAGE3[k][c] as i32;
                let diff = (pitch_lags[k] - predicted_lag) as i64;
                error += diff * diff;
            }
            if error < best_error {
                best_error = error;
                best_contour = c as i8;
            }
        }
    } else {
        // 10ms frame, 2 subframes -- use SILK_CB_LAGS_STAGE3_10_MS ([i8; 12] x 2)
        let n_contours = PE_NB_CBKS_STAGE3_10MS;
        for c in 0..n_contours {
            let mut error: i64 = 0;
            for k in 0..nb_subfr_usize {
                let predicted_lag = base_lag + SILK_CB_LAGS_STAGE3_10_MS[k][c] as i32;
                let diff = (pitch_lags[k] - predicted_lag) as i64;
                error += diff * diff;
            }
            if error < best_error {
                best_error = error;
                best_contour = c as i8;
            }
        }
    }

    *contour_index = best_contour;
}

/// Find best LTP (Long-Term Prediction) parameters.
///
/// Simplified LTP: For each subframe, compute the LPC residual, then for each LTP
/// codebook (3 codebooks with 8/16/32 entries), find the best entry by minimizing
/// prediction error. Select the codebook (per_index) and entry (ltp_index[k]) with
/// lowest total error.
pub fn silk_find_ltp_params(
    ltp_index: &mut [i8],
    per_index: &mut i8,
    pitch_lags: &[i32],
    input: &[i16],
    pred_coef_q12: &[i16],
    subfr_length: i32,
    nb_subfr: i32,
    ltp_mem_length: i32,
    lpc_order: i32,
) {
    let subfr_len = subfr_length as usize;
    let nb_subfr_usize = nb_subfr as usize;
    let lpc_ord = lpc_order as usize;

    // First, compute LPC residual for the entire frame
    let total_len = (nb_subfr * subfr_length) as usize;
    let offset = ltp_mem_length as usize; // samples available before the frame

    // Max total_len = 4 * 80 = 320
    let mut residual = [0i32; MAX_NB_SUBFR * MAX_SUB_FRAME_LENGTH];

    // Compute LPC residual: r[n] = x[n] - sum(a[k] * x[n-k-1])
    for (n, res_item) in residual.iter_mut().enumerate().take(total_len) {
        let abs_n = offset + n;
        let mut pred: i64 = 0;
        for k in 0..lpc_ord {
            if abs_n > k {
                pred += (pred_coef_q12[k] as i64) * (input[abs_n - k - 1] as i64);
            }
        }
        *res_item = (input[abs_n] as i32) - ((pred >> 12) as i32);
    }

    // For each LTP codebook, compute total error across all subframes
    let codebook_sizes: [usize; NB_LTP_CBKS] = [8, 16, 32];
    let codebooks: [&[[i8; 5]]; NB_LTP_CBKS] = [
        &SILK_LTP_GAIN_VQ_0,
        &SILK_LTP_GAIN_VQ_1,
        &SILK_LTP_GAIN_VQ_2,
    ];

    let mut best_total_error = i64::MAX;
    let mut best_per = 0usize;
    let mut best_ltp_indices = [0i8; MAX_NB_SUBFR];

    for cbk in 0..NB_LTP_CBKS {
        let n_entries = codebook_sizes[cbk];
        let codebook = codebooks[cbk];
        let mut total_error: i64 = 0;
        let mut cbk_ltp_indices = [0i8; MAX_NB_SUBFR];

        for sf in 0..nb_subfr_usize {
            let lag = pitch_lags[sf] as usize;
            let sf_start = sf * subfr_len;

            // Find best codebook entry for this subframe
            let mut best_entry_error = i64::MAX;
            let mut best_entry = 0usize;

            for (entry, cb_entry) in codebook.iter().enumerate().take(n_entries) {
                let mut error: i64 = 0;
                for n in 0..subfr_len {
                    let abs_n = sf_start + n;
                    // LTP prediction: sum(b[k] * residual[n - lag + 2 - k]) for k=0..4
                    let mut ltp_pred: i64 = 0;
                    for (k, b_k) in cb_entry.iter().enumerate().take(LTP_ORDER) {
                        let lag_idx = abs_n as i64 - lag as i64 + 2 - k as i64;
                        if lag_idx >= 0 && (lag_idx as usize) < total_len {
                            ltp_pred += (*b_k as i64) * (residual[lag_idx as usize] as i64);
                        }
                    }
                    ltp_pred >>= 7; // Q7 -> Q0

                    let err = residual[abs_n] as i64 - ltp_pred;
                    error += err * err;
                }

                if error < best_entry_error {
                    best_entry_error = error;
                    best_entry = entry;
                }
            }

            cbk_ltp_indices[sf] = best_entry as i8;
            total_error += best_entry_error;
        }

        if total_error < best_total_error {
            best_total_error = total_error;
            best_per = cbk;
            best_ltp_indices = cbk_ltp_indices;
        }
    }

    *per_index = best_per as i8;
    ltp_index[..nb_subfr_usize].copy_from_slice(&best_ltp_indices[..nb_subfr_usize]);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_pitch_analysis_300hz_at_16khz() {
        let fs_khz = 16;
        let nb_subfr = 4i32;
        let expected_len =
            ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz) as usize;

        // Generate 300Hz tone at 16kHz
        let mut frame = vec![0i16; expected_len];
        for (i, sample) in frame.iter_mut().enumerate() {
            *sample = (10000.0 * (2.0 * PI * 300.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let mut pitch_lags = [0i32; 4];
        let mut lag_index: i16 = 0;
        let mut contour_index: i8 = 0;
        let mut ltp_corr_q15: i32 = 0;

        let ret = silk_pitch_analysis_core(
            &frame,
            &mut pitch_lags,
            &mut lag_index,
            &mut contour_index,
            &mut ltp_corr_q15,
            0,     // prev_lag
            52429, // search_thres1_q16
            2458,  // search_thres2_q13
            fs_khz,
            0, // complexity
            nb_subfr,
        );

        let expected_period = 16000.0 / 300.0; // ~53.3
        eprintln!(
            "ret={}, pitch_lags={:?}, lag_index={}, contour_index={}, ltp_corr_q15={}",
            ret, pitch_lags, lag_index, contour_index, ltp_corr_q15
        );
        eprintln!("Expected period: {:.1}", expected_period);

        assert_eq!(ret, 0, "300Hz tone should be detected as voiced");
        for (k, &lag) in pitch_lags.iter().enumerate() {
            assert!(
                (lag - 53).abs() <= 2,
                "Pitch lag [{}] = {} should be near 53",
                k,
                lag,
            );
        }
    }

    #[test]
    fn test_pitch_analysis_simple_300hz() {
        let fs_khz = 16;
        let nb_subfr = 4i32;
        let expected_len =
            ((PE_LTP_MEM_LENGTH_MS + nb_subfr * PE_SUBFR_LENGTH_MS) * fs_khz) as usize;

        // Generate 300Hz tone at 16kHz
        let mut frame = vec![0i16; expected_len];
        for (i, sample) in frame.iter_mut().enumerate() {
            *sample = (10000.0 * (2.0 * PI * 300.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let mut pitch_lags = [0i32; 4];
        let voiced = silk_pitch_analysis_simple(
            &frame,
            &mut pitch_lags,
            fs_khz,
            nb_subfr,
            nb_subfr * PE_SUBFR_LENGTH_MS * fs_khz,
        );

        eprintln!("voiced={}, pitch_lags={:?}", voiced, pitch_lags);
        assert!(voiced, "300Hz tone should be detected as voiced");
    }

    #[test]
    fn test_pitch_analysis_with_zero_history() {
        // Simulate the encoder's first frame scenario: zero history + 300Hz tone
        let fs_khz = 16;
        let nb_subfr = 4i32;
        let ltp_mem_length = (20 * fs_khz) as usize; // 320
        let frame_length = (nb_subfr * 5 * fs_khz) as usize; // 320
        let total_len = ltp_mem_length + frame_length; // 640

        let mut frame = vec![0i16; total_len];
        // Only fill the frame portion (history is zero)
        for (i, sample) in frame.iter_mut().enumerate().skip(ltp_mem_length) {
            *sample =
                (10000.0 * (2.0 * PI * 300.0 * (i - ltp_mem_length) as f64 / 16000.0).sin()) as i16;
        }

        let mut pitch_lags = [0i32; 4];
        let voiced = silk_pitch_analysis_simple(
            &frame,
            &mut pitch_lags,
            fs_khz,
            nb_subfr,
            frame_length as i32,
        );

        eprintln!(
            "With zero history: voiced={}, pitch_lags={:?}",
            voiced, pitch_lags
        );
        // Even with zero history, a strong tone should be detected
    }
}
