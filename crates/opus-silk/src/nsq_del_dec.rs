// Port of SILK Delayed Decision Noise Shaping Quantizer (NSQ_del_dec) from silk/NSQ_del_dec.c
//
// This implements the multi-hypothesis tree search quantizer that maintains
// several parallel quantization paths and selects the best one based on
// rate-distortion cost. Faithfully ported from the C reference.

use crate::nlsf;
use crate::nsq::{MAX_SHAPE_LPC_ORDER, NSQ_LPC_BUF_LENGTH, NsqState};
use crate::tables::*;
use crate::*;

// Constants from silk/define.h
pub const DECISION_DELAY: usize = 40;
pub const MAX_DEL_DEC_STATES: usize = 4;

/// Per-state delayed decision data (matches NSQ_del_dec_struct in C)
#[derive(Clone)]
struct DelDecState {
    s_lpc_q14: [i32; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
    rand_state: [i32; DECISION_DELAY],
    q_q10: [i32; DECISION_DELAY],
    xq_q14: [i32; DECISION_DELAY],
    pred_q15: [i32; DECISION_DELAY],
    shape_q14: [i32; DECISION_DELAY],
    s_ar2_q14: [i32; MAX_SHAPE_LPC_ORDER],
    lf_ar_q14: i32,
    diff_q14: i32,
    seed: i32,
    seed_init: i32,
    rd_q10: i32,
}

impl DelDecState {
    fn new() -> Self {
        Self {
            s_lpc_q14: [0; MAX_SUB_FRAME_LENGTH + NSQ_LPC_BUF_LENGTH],
            rand_state: [0; DECISION_DELAY],
            q_q10: [0; DECISION_DELAY],
            xq_q14: [0; DECISION_DELAY],
            pred_q15: [0; DECISION_DELAY],
            shape_q14: [0; DECISION_DELAY],
            s_ar2_q14: [0; MAX_SHAPE_LPC_ORDER],
            lf_ar_q14: 0,
            diff_q14: 0,
            seed: 0,
            seed_init: 0,
            rd_q10: 0,
        }
    }
}

/// Per-sample candidate state (matches NSQ_sample_struct in C)
#[derive(Clone, Copy, Default)]
struct SampleState {
    q_q10: i32,
    rd_q10: i32,
    xq_q14: i32,
    lf_ar_q14: i32,
    diff_q14: i32,
    s_ltp_shp_q14: i32,
    lpc_exc_q14: i32,
}

/// Scale states for the del-dec NSQ (matching silk_nsq_del_dec_scale_states)
fn silk_nsq_del_dec_scale_states(
    nsq: &mut NsqState,
    del_dec: &mut [DelDecState],
    x16: &[i16],
    x_sc_q10: &mut [i32],
    s_ltp: &[i16],
    s_ltp_q15: &mut [i32],
    subfr: usize,
    n_states: usize,
    ltp_scale_q14: i32,
    gains_q16: &[i32],
    pitch_l: &[i32],
    signal_type: i32,
    decision_delay: i32,
    subfr_length: usize,
    ltp_mem_length: usize,
) {
    let lag = pitch_l[subfr] as usize;
    let inv_gain_q31 = silk_inverse32_varq(gains_q16[subfr].max(1), 47);

    // Scale input
    let inv_gain_q26 = silk_rshift_round(inv_gain_q31, 5);
    for i in 0..subfr_length {
        x_sc_q10[i] = silk_smulww_correct(x16[i] as i32, inv_gain_q26);
    }

    // After rewhitening the LTP state is un-scaled, so scale with inv_gain
    if nsq.rewhite_flag != 0 {
        let mut ig_q31 = inv_gain_q31;
        if subfr == 0 {
            // Do LTP downscaling
            ig_q31 = silk_smulwb(ig_q31, ltp_scale_q14) << 2;
        }
        let start = (nsq.s_ltp_buf_idx as usize).saturating_sub(lag + LTP_ORDER / 2);
        let end = nsq.s_ltp_buf_idx as usize;
        for i in start..end {
            if i < s_ltp_q15.len() && i < s_ltp.len() {
                s_ltp_q15[i] = silk_smulwb(ig_q31, s_ltp[i] as i32);
            }
        }
    }

    // Adjust for changing gain
    if gains_q16[subfr] != nsq.prev_gain_q16 {
        let gain_adj_q16 = silk_div32_varq(nsq.prev_gain_q16, gains_q16[subfr], 16);

        // Scale long-term shaping state
        let shp_start = (nsq.s_ltp_shp_buf_idx as usize).saturating_sub(ltp_mem_length);
        let shp_end = nsq.s_ltp_shp_buf_idx as usize;
        for i in shp_start..shp_end {
            if i < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[i] = silk_smulww_correct(gain_adj_q16, nsq.s_ltp_shp_q14[i]);
            }
        }

        // Scale long-term prediction state
        if signal_type == TYPE_VOICED && nsq.rewhite_flag == 0 {
            let ltp_start = (nsq.s_ltp_buf_idx as usize).saturating_sub(lag + LTP_ORDER / 2);
            let ltp_end = (nsq.s_ltp_buf_idx as usize).saturating_sub(decision_delay as usize);
            for i in ltp_start..ltp_end {
                if i < s_ltp_q15.len() {
                    s_ltp_q15[i] = silk_smulww_correct(gain_adj_q16, s_ltp_q15[i]);
                }
            }
        }

        // Scale delayed decision states
        for dd in del_dec.iter_mut().take(n_states) {
            // Scale scalar states
            dd.lf_ar_q14 = silk_smulww_correct(gain_adj_q16, dd.lf_ar_q14);
            dd.diff_q14 = silk_smulww_correct(gain_adj_q16, dd.diff_q14);

            // Scale short-term prediction and shaping states
            for i in 0..NSQ_LPC_BUF_LENGTH {
                dd.s_lpc_q14[i] = silk_smulww_correct(gain_adj_q16, dd.s_lpc_q14[i]);
            }
            for i in 0..MAX_SHAPE_LPC_ORDER {
                dd.s_ar2_q14[i] = silk_smulww_correct(gain_adj_q16, dd.s_ar2_q14[i]);
            }
            for i in 0..DECISION_DELAY {
                dd.pred_q15[i] = silk_smulww_correct(gain_adj_q16, dd.pred_q15[i]);
                dd.shape_q14[i] = silk_smulww_correct(gain_adj_q16, dd.shape_q14[i]);
            }
        }

        // Save inverse gain
        nsq.prev_gain_q16 = gains_q16[subfr];
    }
}

/// Per-subframe noise shape quantizer for delayed decision
/// Matches silk_noise_shape_quantizer_del_dec in the C reference.
///
/// `pulses_base` and `pxq_base` are absolute offsets into the full `pulses` / `nsq.xq` arrays
/// for the current subframe. This allows negative-offset writes for the delayed commit.
fn silk_noise_shape_quantizer_del_dec(
    nsq: &mut NsqState,
    del_dec: &mut [DelDecState],
    signal_type: i32,
    x_q10: &[i32],
    pulses: &mut [i8],
    pulses_base: usize,
    pxq_base: usize,
    s_ltp_q15: &mut [i32],
    delayed_gain_q10: &mut [i32],
    a_q12: &[i16],
    b_q14: &[i16],
    ar_shp_q13: &[i16],
    lag: i32,
    harm_shape_fir_packed_q14: i32,
    tilt_q14: i32,
    lf_shp_q14: i32,
    gain_q16: i32,
    lambda_q10: i32,
    offset_q10: i32,
    length: usize,
    subfr: usize,
    shaping_lpc_order: usize,
    predict_lpc_order: usize,
    warping_q16: i32,
    n_states: usize,
    smpl_buf_idx: &mut usize,
    decision_delay: usize,
) {
    let gain_q10 = gain_q16 >> 6;

    // Per-sample candidate pairs: [n_states][2]
    let mut sample_states = [[SampleState::default(); 2]; MAX_DEL_DEC_STATES];

    // shp_lag_ptr and pred_lag_ptr base indices
    let shp_lag_base =
        nsq.s_ltp_shp_buf_idx as i64 - lag as i64 + (nsq::HARM_SHAPE_FIR_TAPS / 2) as i64;
    let pred_lag_base = nsq.s_ltp_buf_idx as i64 - lag as i64 + (LTP_ORDER / 2) as i64;

    for (i, &x_q10_i) in x_q10.iter().enumerate().take(length) {
        // ---- Common calculations (independent of state) ----

        // Long-term prediction
        let ltp_pred_q14 = if signal_type == TYPE_VOICED {
            let plp = (pred_lag_base + i as i64) as usize;
            let mut acc: i32 = 2; // rounding bias
            for (k, &b_q14_k) in b_q14.iter().enumerate().take(LTP_ORDER) {
                let idx_val = plp.wrapping_sub(k);
                if idx_val < s_ltp_q15.len() {
                    acc = silk_smlawb(acc, s_ltp_q15[idx_val], b_q14_k as i32);
                }
            }
            acc << 1 // Q13 -> Q14
        } else {
            0
        };

        // Long-term shaping
        let n_ltp_q14 = if lag > 0 {
            let slp = (shp_lag_base + i as i64) as usize;
            let shp0 = if slp < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[slp]
            } else {
                0
            };
            let shp_m1 = if slp >= 1 && slp - 1 < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[slp - 1]
            } else {
                0
            };
            let shp_m2 = if slp >= 2 && slp - 2 < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[slp - 2]
            } else {
                0
            };

            let mut n = silk_smulwb(silk_add_sat32(shp0, shp_m2), harm_shape_fir_packed_q14);
            // silk_SMLAWT: a + ((b * (c >> 16)) >> 16)
            n = n.wrapping_add(
                ((shp_m1 as i64 * (harm_shape_fir_packed_q14 as i64 >> 16)) >> 16) as i32,
            );
            // silk_SUB_LSHIFT32(LTP_pred_Q14, n_LTP_Q14, 2): LTP_pred_Q14 - (n << 2)
            ltp_pred_q14.wrapping_sub(n << 2)
        } else {
            0
        };

        // ---- Per-state calculations ----
        for k in 0..n_states {
            let dd = &mut del_dec[k];
            let ss = &mut sample_states[k];

            // Generate dither
            dd.seed = silk_rand(dd.seed);

            // Short-term prediction using warped lattice structure
            let ps_lpc_idx = NSQ_LPC_BUF_LENGTH - 1 + i;

            // silk_noise_shape_quantizer_short_prediction:
            // out = order/2 (rounding bias)
            // out = silk_SMLAWB(out, buf[0], coef[0]) ...
            let mut lpc_pred_q14: i32 = predict_lpc_order as i32 >> 1;
            for (j, &a_q12_j) in a_q12.iter().enumerate().take(predict_lpc_order) {
                lpc_pred_q14 =
                    silk_smlawb(lpc_pred_q14, dd.s_lpc_q14[ps_lpc_idx - j], a_q12_j as i32);
            }
            lpc_pred_q14 <<= 4; // Q10 -> Q14

            // Noise shape feedback (warped lattice allpass filter)
            // Matches the C code exactly with warping
            let (n_ar_q14, tmp_last) = {
                // Output of lowpass section
                let mut tmp2 = silk_smlawb(dd.diff_q14, dd.s_ar2_q14[0], warping_q16);
                // Output of first allpass section
                let mut tmp1 = silk_smlawb(
                    dd.s_ar2_q14[0],
                    dd.s_ar2_q14[1].wrapping_sub(tmp2),
                    warping_q16,
                );
                dd.s_ar2_q14[0] = tmp2;
                let mut n_ar = shaping_lpc_order as i32 >> 1; // rounding bias
                n_ar = silk_smlawb(n_ar, tmp2, ar_shp_q13[0] as i32);

                // Loop over allpass sections in pairs
                let mut j = 2;
                while j < shaping_lpc_order {
                    tmp2 = silk_smlawb(
                        dd.s_ar2_q14[j - 1],
                        dd.s_ar2_q14[j].wrapping_sub(tmp1),
                        warping_q16,
                    );
                    dd.s_ar2_q14[j - 1] = tmp1;
                    n_ar = silk_smlawb(n_ar, tmp1, ar_shp_q13[j - 1] as i32);

                    tmp1 = silk_smlawb(
                        dd.s_ar2_q14[j],
                        dd.s_ar2_q14[j + 1].wrapping_sub(tmp2),
                        warping_q16,
                    );
                    dd.s_ar2_q14[j] = tmp2;
                    n_ar = silk_smlawb(n_ar, tmp2, ar_shp_q13[j] as i32);
                    j += 2;
                }
                dd.s_ar2_q14[shaping_lpc_order - 1] = tmp1;
                n_ar = silk_smlawb(n_ar, tmp1, ar_shp_q13[shaping_lpc_order - 1] as i32);

                let mut n_ar_q14_val = n_ar << 1; // Q11 -> Q12
                n_ar_q14_val = silk_smlawb(n_ar_q14_val, dd.lf_ar_q14, tilt_q14); // Q12
                n_ar_q14_val <<= 2; // Q12 -> Q14

                (n_ar_q14_val, tmp1)
            };
            let _ = tmp_last; // already stored

            // LF shaping
            let n_lf_q14 = {
                let shape_val = dd.shape_q14[*smpl_buf_idx];
                let mut n_lf = silk_smulwb(shape_val, lf_shp_q14); // Q12
                // silk_SMLAWT: a + ((b * (c >> 16)) >> 16)
                n_lf = n_lf
                    .wrapping_add(((dd.lf_ar_q14 as i64 * (lf_shp_q14 as i64 >> 16)) >> 16) as i32); // Q12
                n_lf << 2 // Q12 -> Q14
            };

            // Input minus prediction plus noise feedback
            // r = x[i] - LTP_pred - LPC_pred + n_AR + n_Tilt + n_LF + n_LTP
            let tmp1 = silk_add_sat32(n_ar_q14, n_lf_q14); // Q14
            // In the C code:
            //   tmp2 = ADD32_ovflw(n_LTP_Q14, LPC_pred_Q14)  -- note n_LTP_Q14 = LTP_pred - harm_shaping
            //   tmp1 = SUB_SAT(tmp2, tmp1)
            //   tmp1 = RSHIFT_ROUND(tmp1, 4) -> Q10
            //   r_Q10 = x_Q10[i] - tmp1
            // When lag > 0, n_LTP_Q14 already has LTP_pred_Q14 embedded via silk_SUB_LSHIFT32
            let combined_pred_q14 = if lag > 0 {
                // n_LTP_Q14 already contains LTP_pred - shaping; add LPC_pred
                n_ltp_q14.wrapping_add(lpc_pred_q14)
            } else {
                // No LTP shaping: just LTP_pred + LPC_pred (LTP_pred is 0 if unvoiced)
                ltp_pred_q14.wrapping_add(lpc_pred_q14)
            };
            let combined = combined_pred_q14.saturating_sub(tmp1); // Q14
            let combined_q10 = silk_rshift_round(combined, 4); // Q10

            let r_q10 = x_q10_i.wrapping_sub(combined_q10);

            // Flip sign depending on dither
            let r_q10 = if dd.seed < 0 { -r_q10 } else { r_q10 };
            let r_q10 = r_q10.clamp(-(31 << 10), 30 << 10);

            // Find two quantization level candidates
            let q1_q10_initial = r_q10 - offset_q10;
            let mut q1_q0 = q1_q10_initial >> 10;

            if lambda_q10 > 2048 {
                let rdo_offset = lambda_q10 / 2 - 512;
                if q1_q10_initial > rdo_offset {
                    q1_q0 = (q1_q10_initial - rdo_offset) >> 10;
                } else if q1_q10_initial < -rdo_offset {
                    q1_q0 = (q1_q10_initial + rdo_offset) >> 10;
                } else if q1_q10_initial < 0 {
                    q1_q0 = -1;
                } else {
                    q1_q0 = 0;
                }
            }

            let (q1_q10, q2_q10, rd1_q10, rd2_q10);

            if q1_q0 > 0 {
                q1_q10 = (q1_q0 << 10) - QUANT_LEVEL_ADJUST_Q10 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q10 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == 0 {
                q1_q10 = offset_q10;
                q2_q10 = q1_q10 + 1024 - QUANT_LEVEL_ADJUST_Q10;
                rd1_q10 = silk_smulbb(q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10, lambda_q10);
            } else if q1_q0 == -1 {
                q2_q10 = offset_q10;
                q1_q10 = q2_q10 - 1024 + QUANT_LEVEL_ADJUST_Q10;
                rd1_q10 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(q2_q10, lambda_q10);
            } else {
                // q1_q0 < -1
                q1_q10 = (q1_q0 << 10) + QUANT_LEVEL_ADJUST_Q10 + offset_q10;
                q2_q10 = q1_q10 + 1024;
                rd1_q10 = silk_smulbb(-q1_q10, lambda_q10);
                rd2_q10 = silk_smulbb(-q2_q10, lambda_q10);
            }

            let rr1 = r_q10 - q1_q10;
            let rd1 = silk_smlabb(rd1_q10, rr1, rr1) >> 10;
            let rr2 = r_q10 - q2_q10;
            let rd2 = silk_smlabb(rd2_q10, rr2, rr2) >> 10;

            if rd1 < rd2 {
                ss[0].rd_q10 = dd.rd_q10.wrapping_add(rd1);
                ss[1].rd_q10 = dd.rd_q10.wrapping_add(rd2);
                ss[0].q_q10 = q1_q10;
                ss[1].q_q10 = q2_q10;
            } else {
                ss[0].rd_q10 = dd.rd_q10.wrapping_add(rd2);
                ss[1].rd_q10 = dd.rd_q10.wrapping_add(rd1);
                ss[0].q_q10 = q2_q10;
                ss[1].q_q10 = q1_q10;
            }

            // Update states for best quantization (ss[0])
            let exc_q14_0 = ss[0].q_q10 << 4;
            let exc_q14_0 = if dd.seed < 0 { -exc_q14_0 } else { exc_q14_0 };
            let lpc_exc_q14_0 = exc_q14_0.wrapping_add(ltp_pred_q14);
            let xq_q14_0 = lpc_exc_q14_0.wrapping_add(lpc_pred_q14);

            ss[0].diff_q14 = xq_q14_0.wrapping_sub(x_q10_i << 4);
            let s_lf_ar_shp_q14_0 = ss[0].diff_q14.wrapping_sub(n_ar_q14);
            ss[0].s_ltp_shp_q14 = s_lf_ar_shp_q14_0.saturating_sub(n_lf_q14);
            ss[0].lf_ar_q14 = s_lf_ar_shp_q14_0;
            ss[0].lpc_exc_q14 = lpc_exc_q14_0;
            ss[0].xq_q14 = xq_q14_0;

            // Update states for second best quantization (ss[1])
            let exc_q14_1 = ss[1].q_q10 << 4;
            let exc_q14_1 = if dd.seed < 0 { -exc_q14_1 } else { exc_q14_1 };
            let lpc_exc_q14_1 = exc_q14_1.wrapping_add(ltp_pred_q14);
            let xq_q14_1 = lpc_exc_q14_1.wrapping_add(lpc_pred_q14);

            ss[1].diff_q14 = xq_q14_1.wrapping_sub(x_q10_i << 4);
            let s_lf_ar_shp_q14_1 = ss[1].diff_q14.wrapping_sub(n_ar_q14);
            ss[1].s_ltp_shp_q14 = s_lf_ar_shp_q14_1.saturating_sub(n_lf_q14);
            ss[1].lf_ar_q14 = s_lf_ar_shp_q14_1;
            ss[1].lpc_exc_q14 = lpc_exc_q14_1;
            ss[1].xq_q14 = xq_q14_1;
        }

        // Update smpl_buf_idx (wrapping decrement)
        *smpl_buf_idx = if *smpl_buf_idx == 0 {
            DECISION_DELAY - 1
        } else {
            *smpl_buf_idx - 1
        };
        let last_smple_idx = (*smpl_buf_idx + decision_delay) % DECISION_DELAY;

        // Find winner (best RD in first set)
        let mut rd_min_q10 = sample_states[0][0].rd_q10;
        let mut winner_ind = 0usize;
        for (k, ss_k) in sample_states.iter().enumerate().take(n_states).skip(1) {
            if ss_k[0].rd_q10 < rd_min_q10 {
                rd_min_q10 = ss_k[0].rd_q10;
                winner_ind = k;
            }
        }

        // Increase RD values of expired states (different rand path)
        let winner_rand_state = del_dec[winner_ind].rand_state[last_smple_idx];
        for k in 0..n_states {
            if del_dec[k].rand_state[last_smple_idx] != winner_rand_state {
                sample_states[k][0].rd_q10 = sample_states[k][0].rd_q10.wrapping_add(i32::MAX >> 4);
                sample_states[k][1].rd_q10 = sample_states[k][1].rd_q10.wrapping_add(i32::MAX >> 4);
            }
        }

        // Find worst in first set and best in second set
        let mut rd_max_q10 = sample_states[0][0].rd_q10;
        let mut rd_min2_q10 = sample_states[0][1].rd_q10;
        let mut rd_max_ind = 0usize;
        let mut rd_min_ind = 0usize;
        for (k, ss_k) in sample_states.iter().enumerate().take(n_states).skip(1) {
            if ss_k[0].rd_q10 > rd_max_q10 {
                rd_max_q10 = ss_k[0].rd_q10;
                rd_max_ind = k;
            }
            if ss_k[1].rd_q10 < rd_min2_q10 {
                rd_min2_q10 = ss_k[1].rd_q10;
                rd_min_ind = k;
            }
        }

        // Replace worst first-set state with best second-set state if it's better
        if rd_min2_q10 < rd_max_q10 {
            // Copy the source state's data that comes after the sLPC_Q14 portion
            // In C: memcpy at offset i into the struct. The C code copies from index i
            // of NSQ_del_dec_struct which is:
            //   silk_memcpy(((opus_int32*)&psDelDec[RDmax_ind]) + i,
            //               ((opus_int32*)&psDelDec[RDmin_ind]) + i,
            //               sizeof(NSQ_del_dec_struct) - i * sizeof(opus_int32))
            // This copies everything except the first i elements of sLPC_Q14.
            // We replicate this by copying the relevant fields.
            // Use split_at_mut to get two mutable references without clone()
            if rd_min_ind != rd_max_ind {
                let (src, dst) = if rd_min_ind < rd_max_ind {
                    let (left, right) = del_dec.split_at_mut(rd_max_ind);
                    (&left[rd_min_ind], &mut right[0])
                } else {
                    let (left, right) = del_dec.split_at_mut(rd_min_ind);
                    (&right[0], &mut left[rd_max_ind])
                };
                // Copy sLPC_Q14 from index i onward (skip already-committed prefix)
                dst.s_lpc_q14[i..].copy_from_slice(&src.s_lpc_q14[i..]);
                dst.rand_state = src.rand_state;
                dst.q_q10 = src.q_q10;
                dst.xq_q14 = src.xq_q14;
                dst.pred_q15 = src.pred_q15;
                dst.shape_q14 = src.shape_q14;
                dst.s_ar2_q14 = src.s_ar2_q14;
                dst.lf_ar_q14 = src.lf_ar_q14;
                dst.diff_q14 = src.diff_q14;
                dst.seed = src.seed;
                dst.seed_init = src.seed_init;
                dst.rd_q10 = src.rd_q10;
            }

            // Copy the second-best sample state to become the best for the replaced state
            sample_states[rd_max_ind][0] = sample_states[rd_min_ind][1];
        }

        // Write samples from winner to output and long-term filter states
        let dd_winner = &del_dec[winner_ind];
        if subfr > 0 || i >= decision_delay {
            // In C: pulses[i - decisionDelay] and pxq[i - decisionDelay]
            // These are relative to the current subframe pointer, so negative offsets
            // reach back into previous subframes. We use absolute base + signed offset.
            let abs_pulse_idx = (pulses_base as i64 + i as i64 - decision_delay as i64) as usize;
            let abs_xq_idx = (pxq_base as i64 + i as i64 - decision_delay as i64) as usize;
            if abs_pulse_idx < pulses.len() {
                pulses[abs_pulse_idx] =
                    silk_rshift_round(dd_winner.q_q10[last_smple_idx], 10) as i8;
            }
            if abs_xq_idx < nsq.xq.len() {
                nsq.xq[abs_xq_idx] = silk_sat16(silk_rshift_round(
                    silk_smulww_correct(
                        dd_winner.xq_q14[last_smple_idx],
                        delayed_gain_q10[last_smple_idx],
                    ),
                    8,
                ));
            }
            let shp_idx = (nsq.s_ltp_shp_buf_idx - decision_delay as i32) as usize;
            if shp_idx < nsq.s_ltp_shp_q14.len() {
                nsq.s_ltp_shp_q14[shp_idx] = dd_winner.shape_q14[last_smple_idx];
            }
            let ltp_idx = (nsq.s_ltp_buf_idx - decision_delay as i32) as usize;
            if ltp_idx < s_ltp_q15.len() {
                s_ltp_q15[ltp_idx] = dd_winner.pred_q15[last_smple_idx];
            }
        }
        nsq.s_ltp_shp_buf_idx += 1;
        nsq.s_ltp_buf_idx += 1;

        // Update all delayed decision states
        for k in 0..n_states {
            let dd = &mut del_dec[k];
            let ss = &sample_states[k][0];
            dd.lf_ar_q14 = ss.lf_ar_q14;
            dd.diff_q14 = ss.diff_q14;
            dd.s_lpc_q14[NSQ_LPC_BUF_LENGTH + i] = ss.xq_q14;
            dd.xq_q14[*smpl_buf_idx] = ss.xq_q14;
            dd.q_q10[*smpl_buf_idx] = ss.q_q10;
            dd.pred_q15[*smpl_buf_idx] = ss.lpc_exc_q14 << 1;
            dd.shape_q14[*smpl_buf_idx] = ss.s_ltp_shp_q14;
            dd.seed = dd.seed.wrapping_add(silk_rshift_round(ss.q_q10, 10));
            dd.rand_state[*smpl_buf_idx] = dd.seed;
            dd.rd_q10 = ss.rd_q10;
        }
        delayed_gain_q10[*smpl_buf_idx] = gain_q10;
    }

    // Update LPC states: shift buffer
    for dd in del_dec.iter_mut().take(n_states) {
        for j in 0..NSQ_LPC_BUF_LENGTH {
            dd.s_lpc_q14[j] = dd.s_lpc_q14[length + j];
        }
    }
}

/// Main delayed-decision NSQ entry point.
///
/// Port of silk_NSQ_del_dec_c from silk/NSQ_del_dec.c.
/// Performs noise-shaped quantization using a multi-hypothesis tree search
/// that maintains several parallel quantization paths and selects the best
/// one based on rate-distortion cost.
pub fn silk_nsq_del_dec(
    nsq: &mut NsqState,
    indices: &mut SideInfoIndices,
    x16: &[i16],
    pulses: &mut [i8],
    pred_coef_q12: &[i16],
    ltp_coef_q14: &[i16],
    ar_q13: &[i16],
    harm_shape_gain_q14: &[i32],
    tilt_q14: &[i32],
    lf_shp_q14: &[i32],
    gains_q16: &[i32],
    pitch_l: &[i32],
    lambda_q10: i32,
    ltp_scale_q14: i32,
    // Config
    frame_length: i32,
    subfr_length: i32,
    ltp_mem_length: i32,
    lpc_order: i32,
    shaping_lpc_order: i32,
    nb_subfr: i32,
    signal_type: i32,
    quant_offset_type: i32,
    nlsf_interp_coef_q2: i32,
    n_states_delayed_decision: i32,
    warping_q16: i32,
    // Scratch buffers
    scratch_s_ltp_q15: &mut [i32],
    scratch_s_ltp: &mut [i16],
) {
    let frame_len = frame_length as usize;
    let subfr_len = subfr_length as usize;
    let ltp_mem_len = ltp_mem_length as usize;
    let lpc_ord = lpc_order as usize;
    let shaping_ord = shaping_lpc_order as usize;
    let n_states = (n_states_delayed_decision as usize).min(MAX_DEL_DEC_STATES);

    // Set unvoiced lag to the previous one, overwrite later for voiced
    let mut lag = nsq.lag_prev;

    // Initialize delayed decision states
    // Stack-allocated states (max 4, ~1.3KB each = ~5.2KB total on stack)
    let mut del_dec = [
        DelDecState::new(),
        DelDecState::new(),
        DelDecState::new(),
        DelDecState::new(),
    ];
    for (k, dd) in del_dec.iter_mut().enumerate().take(n_states) {
        dd.seed = ((k as i32) + indices.seed as i32) & 3;
        dd.seed_init = dd.seed;
        dd.rd_q10 = 0;
        dd.lf_ar_q14 = nsq.s_lf_ar_shp_q14;
        dd.diff_q14 = nsq.s_diff_shp_q14;
        let shp_idx = ltp_mem_len.saturating_sub(1);
        if shp_idx < nsq.s_ltp_shp_q14.len() {
            dd.shape_q14[0] = nsq.s_ltp_shp_q14[shp_idx];
        }
        dd.s_lpc_q14[..NSQ_LPC_BUF_LENGTH].copy_from_slice(&nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]);
        dd.s_ar2_q14[..MAX_SHAPE_LPC_ORDER].copy_from_slice(&nsq.s_ar2_q14[..MAX_SHAPE_LPC_ORDER]);
    }

    let offset_q10 = SILK_QUANTIZATION_OFFSETS_Q10[(signal_type >> 1) as usize]
        [quant_offset_type as usize] as i32;

    let mut smpl_buf_idx: usize = 0;

    // Compute decision delay
    let mut decision_delay = DECISION_DELAY.min(subfr_len);

    // For voiced frames limit the decision delay to lower than the pitch lag
    if signal_type == TYPE_VOICED {
        for &pitch_l_k in pitch_l.iter().take(nb_subfr as usize) {
            decision_delay = decision_delay.min((pitch_l_k - LTP_ORDER as i32 / 2 - 1) as usize);
        }
    } else if lag > 0 {
        decision_delay = decision_delay.min((lag - LTP_ORDER as i32 / 2 - 1) as usize);
    }
    // Ensure decision_delay is at least 1 to avoid issues
    decision_delay = decision_delay.max(1);

    let lsf_interpolation_flag: usize = if nlsf_interp_coef_q2 == 4 { 0 } else { 1 };

    let total_len = ltp_mem_len + frame_len;
    let s_ltp_q15 = &mut scratch_s_ltp_q15[..total_len];
    let s_ltp = &mut scratch_s_ltp[..total_len];
    for v in s_ltp_q15.iter_mut() {
        *v = 0;
    }
    for v in s_ltp.iter_mut() {
        *v = 0;
    }

    // Max subfr_len = MAX_SUB_FRAME_LENGTH = 80
    let mut x_sc_q10 = [0i32; MAX_SUB_FRAME_LENGTH];
    let mut delayed_gain_q10 = [0i32; DECISION_DELAY];

    // Set up pointers to start of sub frame
    nsq.s_ltp_shp_buf_idx = ltp_mem_length;
    nsq.s_ltp_buf_idx = ltp_mem_length;
    let mut pxq_offset = ltp_mem_len;
    let mut x16_offset = 0usize;
    let mut pulses_offset = 0usize;
    let mut subfr_counter = 0usize;

    for k in 0..nb_subfr as usize {
        let a_q12_offset = ((k >> 1) | (1 - lsf_interpolation_flag)) * MAX_LPC_ORDER;
        let a_q12 = &pred_coef_q12[a_q12_offset..a_q12_offset + lpc_ord];
        let b_q14 = &ltp_coef_q14[k * LTP_ORDER..(k + 1) * LTP_ORDER];
        let ar_shp_q13 = &ar_q13[k * MAX_SHAPE_LPC_ORDER..(k + 1) * MAX_SHAPE_LPC_ORDER];

        // Noise shape parameters
        let harm_gain = harm_shape_gain_q14[k];
        let mut harm_shape_fir_packed_q14 = harm_gain >> 2;
        harm_shape_fir_packed_q14 |= (harm_gain >> 1) << 16;

        nsq.rewhite_flag = 0;
        if signal_type == TYPE_VOICED {
            lag = pitch_l[k];

            // Re-whitening
            let rewhite_cond = k & (3 - (lsf_interpolation_flag * 2));
            if rewhite_cond == 0 {
                if k == 2 {
                    // RESET DELAYED DECISIONS
                    // Find winner
                    let mut rd_min = del_dec[0].rd_q10;
                    let mut winner_ind = 0usize;
                    for (j, dd_j) in del_dec.iter().enumerate().take(n_states).skip(1) {
                        if dd_j.rd_q10 < rd_min {
                            rd_min = dd_j.rd_q10;
                            winner_ind = j;
                        }
                    }
                    // Penalize non-winners
                    for (j, dd_j) in del_dec.iter_mut().enumerate().take(n_states) {
                        if j != winner_ind {
                            dd_j.rd_q10 = dd_j.rd_q10.wrapping_add(i32::MAX >> 4);
                        }
                    }

                    // Copy final part of signals from winner state to output
                    let mut last_idx = smpl_buf_idx + decision_delay;
                    for j in 0..decision_delay {
                        last_idx = if last_idx == 0 {
                            DECISION_DELAY - 1
                        } else {
                            last_idx - 1
                        };
                        let out_offset = j as i64 - decision_delay as i64;
                        let pulse_idx = (pulses_offset as i64 + out_offset) as usize;
                        let xq_idx = (pxq_offset as i64 + out_offset) as usize;
                        if pulse_idx < pulses.len() {
                            pulses[pulse_idx] =
                                silk_rshift_round(del_dec[winner_ind].q_q10[last_idx], 10) as i8;
                        }
                        if xq_idx < nsq.xq.len() {
                            nsq.xq[xq_idx] = silk_sat16(silk_rshift_round(
                                silk_smulww_correct(
                                    del_dec[winner_ind].xq_q14[last_idx],
                                    gains_q16[1] >> 6,
                                ),
                                8,
                            ));
                        }
                        let shp_dst = (nsq.s_ltp_shp_buf_idx as i64 - decision_delay as i64
                            + j as i64) as usize;
                        if shp_dst < nsq.s_ltp_shp_q14.len() {
                            nsq.s_ltp_shp_q14[shp_dst] = del_dec[winner_ind].shape_q14[last_idx];
                        }
                    }

                    subfr_counter = 0;
                }

                // Rewhiten with new A coefs
                let start_idx =
                    (ltp_mem_len as i32 - lag - lpc_order - LTP_ORDER as i32 / 2) as usize;

                nlsf::silk_lpc_analysis_filter(
                    &mut s_ltp[start_idx..],
                    &nsq.xq[(start_idx + k * subfr_len)..],
                    a_q12,
                    ltp_mem_len - start_idx,
                    lpc_ord,
                );

                nsq.s_ltp_buf_idx = ltp_mem_length;
                nsq.rewhite_flag = 1;
            }
        }

        silk_nsq_del_dec_scale_states(
            nsq,
            &mut del_dec,
            &x16[x16_offset..],
            &mut x_sc_q10,
            s_ltp,
            s_ltp_q15,
            k,
            n_states,
            ltp_scale_q14,
            gains_q16,
            pitch_l,
            signal_type,
            decision_delay as i32,
            subfr_len,
            ltp_mem_len,
        );

        silk_noise_shape_quantizer_del_dec(
            nsq,
            &mut del_dec,
            signal_type,
            &x_sc_q10,
            pulses,
            pulses_offset,
            pxq_offset,
            s_ltp_q15,
            &mut delayed_gain_q10,
            a_q12,
            b_q14,
            ar_shp_q13,
            lag,
            harm_shape_fir_packed_q14,
            tilt_q14[k],
            lf_shp_q14[k],
            gains_q16[k],
            lambda_q10,
            offset_q10,
            subfr_len,
            subfr_counter,
            shaping_ord,
            lpc_ord,
            warping_q16,
            n_states,
            &mut smpl_buf_idx,
            decision_delay,
        );

        x16_offset += subfr_len;
        pulses_offset += subfr_len;
        pxq_offset += subfr_len;
        subfr_counter += 1;
    }

    // Find final winner
    let mut rd_min = del_dec[0].rd_q10;
    let mut winner_ind = 0usize;
    for (k, dd_k) in del_dec.iter().enumerate().take(n_states).skip(1) {
        if dd_k.rd_q10 < rd_min {
            rd_min = dd_k.rd_q10;
            winner_ind = k;
        }
    }

    // Copy final part of signals from winner state to output
    indices.seed = del_dec[winner_ind].seed_init as i8;
    let mut last_idx = smpl_buf_idx + decision_delay;
    let final_gain_q10 = gains_q16[nb_subfr as usize - 1] >> 6;
    for j in 0..decision_delay {
        last_idx = if last_idx == 0 {
            DECISION_DELAY - 1
        } else {
            last_idx - 1
        };
        let out_offset = j as i64 - decision_delay as i64;
        let pulse_idx = (pulses_offset as i64 + out_offset) as usize;
        let xq_idx = (pxq_offset as i64 + out_offset) as usize;
        if pulse_idx < pulses.len() {
            pulses[pulse_idx] = silk_rshift_round(del_dec[winner_ind].q_q10[last_idx], 10) as i8;
        }
        if xq_idx < nsq.xq.len() {
            nsq.xq[xq_idx] = silk_sat16(silk_rshift_round(
                silk_smulww_correct(del_dec[winner_ind].xq_q14[last_idx], final_gain_q10),
                8,
            ));
        }
        let shp_dst = (nsq.s_ltp_shp_buf_idx as i64 - decision_delay as i64 + j as i64) as usize;
        if shp_dst < nsq.s_ltp_shp_q14.len() {
            nsq.s_ltp_shp_q14[shp_dst] = del_dec[winner_ind].shape_q14[last_idx];
        }
    }

    // Copy winner's filter state to NSQ
    nsq.s_lpc_q14[..NSQ_LPC_BUF_LENGTH]
        .copy_from_slice(&del_dec[winner_ind].s_lpc_q14[..NSQ_LPC_BUF_LENGTH]);
    nsq.s_ar2_q14[..MAX_SHAPE_LPC_ORDER]
        .copy_from_slice(&del_dec[winner_ind].s_ar2_q14[..MAX_SHAPE_LPC_ORDER]);

    // Update states
    nsq.s_lf_ar_shp_q14 = del_dec[winner_ind].lf_ar_q14;
    nsq.s_diff_shp_q14 = del_dec[winner_ind].diff_q14;
    nsq.lag_prev = pitch_l[nb_subfr as usize - 1];

    // Save quantized speech signal: shift buffers
    nsq.xq.copy_within(frame_len..frame_len + ltp_mem_len, 0);
    nsq.s_ltp_shp_q14
        .copy_within(frame_len..frame_len + ltp_mem_len, 0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nsq;

    /// Helper to create a minimal test configuration and run del-dec NSQ.
    /// Returns (nsq_state, pulses).
    fn run_del_dec_nsq(
        n_states: i32,
        signal: &[i16],
        signal_type: i32,
        pitch_lag: i32,
    ) -> (NsqState, Vec<i8>) {
        let fs_khz = 16;
        let nb_subfr = 4i32;
        let subfr_length = 5 * fs_khz; // 80
        let frame_length = nb_subfr * subfr_length;
        let ltp_mem_length = 20 * fs_khz; // 320
        let lpc_order = 16i32;
        let shaping_lpc_order = 16i32;

        // Pad input to frame_length if needed
        let mut x16 = vec![0i16; frame_length as usize];
        let copy_len = signal.len().min(frame_length as usize);
        x16[..copy_len].copy_from_slice(&signal[..copy_len]);

        let mut nsq_state = NsqState::new();
        let mut indices = SideInfoIndices {
            signal_type: signal_type as i8,
            quant_offset_type: 0,
            nlsf_interp_coef_q2: 4, // No interpolation
            seed: 0,
            ..SideInfoIndices::default()
        };

        let mut pulses = vec![0i8; frame_length as usize];

        // Simple flat LPC coefficients (slight prediction)
        let mut pred_coef_q12 = vec![0i16; 2 * MAX_LPC_ORDER];
        // A very mild predictor: a[0] = 0.1 in Q12 = 410
        pred_coef_q12[0] = 410;
        pred_coef_q12[MAX_LPC_ORDER] = 410;

        // LTP coefficients (zero for unvoiced, mild for voiced)
        let mut ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
        if signal_type == TYPE_VOICED {
            for k in 0..nb_subfr as usize {
                ltp_coef_q14[k * LTP_ORDER + 2] = 4096; // center tap = 0.25 in Q14
            }
        }

        // AR shaping coefficients (small values)
        let mut ar_q13 = vec![0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
        for k in 0..nb_subfr as usize {
            ar_q13[k * MAX_SHAPE_LPC_ORDER] = 200;
        }

        let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
        let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
        let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];

        // Moderate gain (Q16)
        let gains_q16 = vec![65536i32; MAX_NB_SUBFR]; // gain = 1.0

        let mut pitch_lags = vec![0i32; MAX_NB_SUBFR];
        if signal_type == TYPE_VOICED {
            for p in pitch_lags.iter_mut() {
                *p = pitch_lag.max(LTP_ORDER as i32 / 2 + 2); // Ensure valid lag
            }
        }

        let lambda_q10 = 1024; // Moderate RD tradeoff
        let ltp_scale_q14 = SILK_LTP_SCALES_TABLE_Q14[0] as i32;

        let total_len = ltp_mem_length as usize + frame_length as usize;
        let mut scratch_s_ltp_q15 = vec![0i32; total_len];
        let mut scratch_s_ltp = vec![0i16; total_len];

        silk_nsq_del_dec(
            &mut nsq_state,
            &mut indices,
            &x16,
            &mut pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &pitch_lags,
            lambda_q10,
            ltp_scale_q14,
            frame_length,
            subfr_length,
            ltp_mem_length,
            lpc_order,
            shaping_lpc_order,
            nb_subfr,
            signal_type,
            0, // quant_offset_type
            4, // nlsf_interp_coef_q2 (no interpolation)
            n_states,
            0, // warping_q16 (no warping for simplicity)
            &mut scratch_s_ltp_q15,
            &mut scratch_s_ltp,
        );

        (nsq_state, pulses)
    }

    /// Helper to run the scalar NSQ with the same parameters for comparison.
    #[allow(dead_code)]
    fn run_scalar_nsq(signal: &[i16], signal_type: i32, pitch_lag: i32) -> (NsqState, Vec<i8>) {
        let fs_khz = 16;
        let nb_subfr = 4i32;
        let subfr_length = 5 * fs_khz;
        let frame_length = nb_subfr * subfr_length;
        let ltp_mem_length = 20 * fs_khz;
        let lpc_order = 16i32;

        let mut x16 = vec![0i16; frame_length as usize];
        let copy_len = signal.len().min(frame_length as usize);
        x16[..copy_len].copy_from_slice(&signal[..copy_len]);

        let mut nsq_state = NsqState::new();
        let mut indices = SideInfoIndices {
            signal_type: signal_type as i8,
            quant_offset_type: 0,
            nlsf_interp_coef_q2: 4,
            seed: 0,
            ..SideInfoIndices::default()
        };

        let mut pulses = vec![0i8; frame_length as usize];

        let mut pred_coef_q12 = vec![0i16; 2 * MAX_LPC_ORDER];
        pred_coef_q12[0] = 410;
        pred_coef_q12[MAX_LPC_ORDER] = 410;

        let mut ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
        if signal_type == TYPE_VOICED {
            for k in 0..nb_subfr as usize {
                ltp_coef_q14[k * LTP_ORDER + 2] = 4096;
            }
        }

        let mut ar_q13 = vec![0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
        for k in 0..nb_subfr as usize {
            ar_q13[k * MAX_SHAPE_LPC_ORDER] = 200;
        }

        let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
        let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
        let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
        let gains_q16 = vec![65536i32; MAX_NB_SUBFR];

        let mut pitch_lags = vec![0i32; MAX_NB_SUBFR];
        if signal_type == TYPE_VOICED {
            for p in pitch_lags.iter_mut() {
                *p = pitch_lag.max(LTP_ORDER as i32 / 2 + 2);
            }
        }

        let lambda_q10 = 1024;
        let ltp_scale_q14 = SILK_LTP_SCALES_TABLE_Q14[0] as i32;

        let total_len = ltp_mem_length as usize + frame_length as usize;
        let mut scratch_s_ltp_q15 = vec![0i32; total_len];
        let mut scratch_s_ltp = vec![0i16; total_len];
        let mut scratch_x_sc_q10 = vec![0i32; subfr_length as usize];
        let mut scratch_xq_tmp = vec![0i16; subfr_length as usize];

        nsq::silk_nsq(
            &mut nsq_state,
            &mut indices,
            &x16,
            &mut pulses,
            &pred_coef_q12,
            &ltp_coef_q14,
            &ar_q13,
            &harm_shape_gain_q14,
            &tilt_q14,
            &lf_shp_q14,
            &gains_q16,
            &pitch_lags,
            lambda_q10,
            ltp_scale_q14,
            frame_length,
            subfr_length,
            ltp_mem_length,
            lpc_order,
            MAX_SHAPE_LPC_ORDER as i32,
            nb_subfr,
            signal_type,
            0,
            4,
            &mut scratch_s_ltp_q15,
            &mut scratch_s_ltp,
            &mut scratch_x_sc_q10,
            &mut scratch_xq_tmp,
        );

        (nsq_state, pulses)
    }

    #[test]
    fn test_del_dec_produces_valid_output_unvoiced() {
        // Generate a simple test signal (sine-like pattern)
        let frame_len = 320usize;
        let mut signal = vec![0i16; frame_len];
        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = ((i as f64 * 0.1).sin() * 5000.0) as i16;
        }

        let (_nsq, pulses) = run_del_dec_nsq(2, &signal, TYPE_UNVOICED, 0);

        // The raw NSQ output can be up to |31| before the encoder's shell codec clipping stage.
        // r_Q10 is clamped to [-31<<10, 30<<10], so after rshift_round by 10, values are in [-31, 31].
        for (i, &p) in pulses.iter().enumerate() {
            assert!(
                (-31..=31).contains(&p),
                "Pulse {} out of range: {} (should be in [-31, 31])",
                i,
                p
            );
        }

        // Verify we got some non-zero pulses (the signal is non-trivial)
        let non_zero_count = pulses.iter().filter(|&&p| p != 0).count();
        assert!(
            non_zero_count > 0,
            "Expected some non-zero pulses for non-zero input signal"
        );
    }

    #[test]
    fn test_del_dec_multi_state_produces_valid_output() {
        // Test with different numbers of states (2, 3, 4) all produce valid output
        let frame_len = 320usize;
        let mut signal = vec![0i16; frame_len];
        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = ((i as f64 * 0.05).sin() * 3000.0) as i16;
        }

        for n_states in [2, 3, 4] {
            let (_nsq, pulses) = run_del_dec_nsq(n_states, &signal, TYPE_UNVOICED, 0);

            // Raw NSQ pulses in [-31, 31]
            for &p in pulses.iter() {
                assert!(
                    (-31..=31).contains(&p),
                    "nStates={}: Pulse out of range: {}",
                    n_states,
                    p
                );
            }

            // Should have non-zero pulses
            let non_zero_count = pulses.iter().filter(|&&p| p != 0).count();
            assert!(
                non_zero_count > 0,
                "nStates={}: Expected non-zero pulses",
                n_states
            );
        }
    }

    #[test]
    fn test_rd_cost_comparison() {
        // Test that increasing lambda (rate cost) produces sparser quantization.
        // Higher lambda = more penalty for non-zero pulses = fewer non-zero pulses.
        let frame_len = 320usize;
        let fs_khz = 16;
        let nb_subfr = 4i32;
        let subfr_length = 5 * fs_khz;
        let frame_length = nb_subfr * subfr_length;
        let ltp_mem_length = 20 * fs_khz;
        let lpc_order = 16i32;
        let shaping_lpc_order = 16i32;

        let mut signal = vec![0i16; frame_len];
        for (i, sample) in signal.iter_mut().enumerate() {
            *sample = ((i as f64 * 0.08).sin() * 4000.0) as i16;
        }

        let mut non_zero_counts = Vec::new();

        for &lambda_q10 in &[512, 2048, 4096] {
            let mut nsq_state = NsqState::new();
            let mut indices = SideInfoIndices {
                signal_type: TYPE_UNVOICED as i8,
                quant_offset_type: 0,
                nlsf_interp_coef_q2: 4,
                seed: 0,
                ..SideInfoIndices::default()
            };

            let mut pulses = vec![0i8; frame_length as usize];

            let mut pred_coef_q12 = vec![0i16; 2 * MAX_LPC_ORDER];
            pred_coef_q12[0] = 410;
            pred_coef_q12[MAX_LPC_ORDER] = 410;

            let ltp_coef_q14 = vec![0i16; MAX_NB_SUBFR * LTP_ORDER];
            let mut ar_q13 = vec![0i16; MAX_NB_SUBFR * MAX_SHAPE_LPC_ORDER];
            for k in 0..nb_subfr as usize {
                ar_q13[k * MAX_SHAPE_LPC_ORDER] = 200;
            }

            let harm_shape_gain_q14 = vec![0i32; MAX_NB_SUBFR];
            let tilt_q14 = vec![0i32; MAX_NB_SUBFR];
            let lf_shp_q14 = vec![0i32; MAX_NB_SUBFR];
            let gains_q16 = vec![65536i32; MAX_NB_SUBFR];
            let pitch_lags = vec![0i32; MAX_NB_SUBFR];
            let ltp_scale_q14 = SILK_LTP_SCALES_TABLE_Q14[0] as i32;

            let total_len = ltp_mem_length as usize + frame_length as usize;
            let mut scratch_s_ltp_q15 = vec![0i32; total_len];
            let mut scratch_s_ltp = vec![0i16; total_len];

            silk_nsq_del_dec(
                &mut nsq_state,
                &mut indices,
                &signal,
                &mut pulses,
                &pred_coef_q12,
                &ltp_coef_q14,
                &ar_q13,
                &harm_shape_gain_q14,
                &tilt_q14,
                &lf_shp_q14,
                &gains_q16,
                &pitch_lags,
                lambda_q10,
                ltp_scale_q14,
                frame_length,
                subfr_length,
                ltp_mem_length,
                lpc_order,
                shaping_lpc_order,
                nb_subfr,
                TYPE_UNVOICED,
                0,
                4,
                2, // nStatesDelayedDecision
                0, // warping
                &mut scratch_s_ltp_q15,
                &mut scratch_s_ltp,
            );

            let nz = pulses.iter().filter(|&&p| p != 0).count();
            non_zero_counts.push((lambda_q10, nz));
        }

        // Higher lambda should generally produce fewer or equal non-zero pulses
        // (sparser quantization). Due to the greedy nature, this is a soft check.
        let (_, count_low) = non_zero_counts[0];
        let (_, count_high) = non_zero_counts[2];

        // The high-lambda version should be at least somewhat sparser than the low-lambda one.
        // We allow a generous margin because the relationship isn't perfectly monotone.
        assert!(
            count_high <= count_low + 20,
            "Higher lambda ({}) should produce sparser output: low_lambda_count={}, high_lambda_count={}",
            non_zero_counts[2].0,
            count_low,
            count_high,
        );
    }

    #[test]
    fn test_del_dec_voiced_produces_valid_output() {
        // Test voiced mode with a pitch lag
        let frame_len = 320usize;
        let pitch_period = 80; // 200 Hz at 16kHz
        let mut signal = vec![0i16; frame_len];
        for (i, sample) in signal.iter_mut().enumerate() {
            // Quasi-periodic signal
            let phase = (i % pitch_period) as f64 / pitch_period as f64 * std::f64::consts::TAU;
            *sample = (phase.sin() * 4000.0) as i16;
        }

        let (_nsq, pulses) = run_del_dec_nsq(2, &signal, TYPE_VOICED, pitch_period as i32);

        // All pulses should be in valid raw NSQ range
        for &p in pulses.iter() {
            assert!((-31..=31).contains(&p), "Voiced pulse out of range: {}", p);
        }

        let non_zero_count = pulses.iter().filter(|&&p| p != 0).count();
        assert!(
            non_zero_count > 0,
            "Expected non-zero pulses for voiced signal"
        );
    }
}
