// Port of silk/decode_parameters.c, silk/decode_pitch.c

use crate::gain_quant::silk_gains_dequant;
use crate::tables::*;
use crate::*;

/// Decode parameters from payload
pub fn silk_decode_parameters(
    ps_dec: &mut ChannelState,
    ps_dec_ctrl: &mut DecoderControl,
    cond_coding: i32,
) {
    // Dequant gains (shared with encoder via gain_quant module)
    silk_gains_dequant(
        &mut ps_dec_ctrl.gains_q16,
        &ps_dec.indices.gains_indices,
        &mut ps_dec.last_gain_index,
        cond_coding == CODE_CONDITIONALLY,
        ps_dec.nb_subfr as usize,
    );

    // Decode NLSFs
    let nlsf_cb = get_nlsf_cb(ps_dec.nlsf_cb);
    let mut p_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    nlsf::silk_nlsf_decode(&mut p_nlsf_q15, &ps_dec.indices.nlsf_indices, nlsf_cb);

    // Convert NLSF to AR prediction filter coefficients
    nlsf::silk_nlsf2a(
        &mut ps_dec_ctrl.pred_coef_q12[1],
        &p_nlsf_q15,
        ps_dec.lpc_order as usize,
    );

    // Handle interpolation
    if ps_dec.first_frame_after_reset {
        ps_dec.indices.nlsf_interp_coef_q2 = 4;
    }

    if ps_dec.indices.nlsf_interp_coef_q2 < 4 {
        let mut p_nlsf0_q15 = [0i16; MAX_LPC_ORDER];
        let interp = ps_dec.indices.nlsf_interp_coef_q2 as i32;
        for i in 0..ps_dec.lpc_order as usize {
            p_nlsf0_q15[i] = (ps_dec.prev_nlsf_q15[i] as i32
                + ((interp * (p_nlsf_q15[i] as i32 - ps_dec.prev_nlsf_q15[i] as i32)) >> 2))
                as i16;
        }
        nlsf::silk_nlsf2a(
            &mut ps_dec_ctrl.pred_coef_q12[0],
            &p_nlsf0_q15,
            ps_dec.lpc_order as usize,
        );
    } else {
        ps_dec_ctrl.pred_coef_q12[0] = ps_dec_ctrl.pred_coef_q12[1];
    }

    // Save NLSF for next frame
    ps_dec.prev_nlsf_q15[..ps_dec.lpc_order as usize]
        .copy_from_slice(&p_nlsf_q15[..ps_dec.lpc_order as usize]);

    // After packet loss, apply BWE
    if ps_dec.loss_cnt > 0 {
        nlsf::silk_bwexpander(
            &mut ps_dec_ctrl.pred_coef_q12[0],
            ps_dec.lpc_order as usize,
            BWE_AFTER_LOSS_Q16,
        );
        nlsf::silk_bwexpander(
            &mut ps_dec_ctrl.pred_coef_q12[1],
            ps_dec.lpc_order as usize,
            BWE_AFTER_LOSS_Q16,
        );
    }

    if ps_dec.indices.signal_type as i32 == TYPE_VOICED {
        // Decode pitch lags
        silk_decode_pitch(
            ps_dec.indices.lag_index,
            ps_dec.indices.contour_index,
            &mut ps_dec_ctrl.pitch_l,
            ps_dec.fs_khz,
            ps_dec.nb_subfr,
        );

        // Decode LTP codebook index
        let cbk_ptr = SILK_LTP_VQ_PTRS_Q7[ps_dec.indices.per_index as usize];
        for k in 0..ps_dec.nb_subfr as usize {
            let ix = ps_dec.indices.ltp_index[k] as usize;
            for (i, &coef) in cbk_ptr[ix].iter().enumerate().take(LTP_ORDER) {
                ps_dec_ctrl.ltp_coef_q14[k * LTP_ORDER + i] = (coef as i16) << 7;
            }
        }

        // Decode LTP scaling
        ps_dec_ctrl.ltp_scale_q14 =
            SILK_LTP_SCALES_TABLE_Q14[ps_dec.indices.ltp_scale_index as usize] as i32;
    } else {
        ps_dec_ctrl.pitch_l[..ps_dec.nb_subfr as usize].fill(0);
        ps_dec_ctrl.ltp_coef_q14[..LTP_ORDER * ps_dec.nb_subfr as usize].fill(0);
        ps_dec.indices.per_index = 0;
        ps_dec_ctrl.ltp_scale_q14 = 0;
    }
}

/// Decode pitch lags
fn silk_decode_pitch(
    lag_index: i16,
    contour_index: i8,
    pitch_lags: &mut [i32; MAX_NB_SUBFR],
    fs_khz: i32,
    nb_subfr: i32,
) {
    let min_lag = PE_MIN_LAG_MS * fs_khz;
    let max_lag = PE_MAX_LAG_MS * fs_khz;
    let lag = min_lag + lag_index as i32;

    if fs_khz == 8 {
        if nb_subfr == PE_MAX_NB_SUBFR as i32 {
            for k in 0..nb_subfr as usize {
                pitch_lags[k] = (lag + SILK_CB_LAGS_STAGE2[k][contour_index as usize] as i32)
                    .clamp(min_lag, max_lag);
            }
        } else {
            for k in 0..nb_subfr as usize {
                pitch_lags[k] = (lag + SILK_CB_LAGS_STAGE2_10_MS[k][contour_index as usize] as i32)
                    .clamp(min_lag, max_lag);
            }
        }
    } else if nb_subfr == PE_MAX_NB_SUBFR as i32 {
        for k in 0..nb_subfr as usize {
            pitch_lags[k] = (lag + SILK_CB_LAGS_STAGE3[k][contour_index as usize] as i32)
                .clamp(min_lag, max_lag);
        }
    } else {
        for k in 0..nb_subfr as usize {
            pitch_lags[k] = (lag + SILK_CB_LAGS_STAGE3_10_MS[k][contour_index as usize] as i32)
                .clamp(min_lag, max_lag);
        }
    }
}
