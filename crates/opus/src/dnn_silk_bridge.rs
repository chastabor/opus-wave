//! Bridge between SILK decoder internals and opus-dnn OSCE feature extraction.
//!
//! Extracts the SILK decoder state fields needed by OSCE into
//! the `OsceInput` bridge struct, avoiding a direct opus-dnn → opus-silk dependency.

use opus_dnn::osce::features::OsceInput;
use opus_silk::ChannelState;
use opus_silk::DecoderControl;

/// Extract OSCE input from SILK decoder state after a frame decode.
pub fn extract_osce_input<'a>(
    channel: &'a ChannelState,
    dec_ctrl: &'a DecoderControl,
    num_bits: i32,
) -> OsceInput<'a> {
    OsceInput {
        pred_coef_q12: &dec_ctrl.pred_coef_q12,
        pitch_lags: &dec_ctrl.pitch_l,
        ltp_coef_q14: &dec_ctrl.ltp_coef_q14,
        gains_q16: &dec_ctrl.gains_q16,
        lpc_order: channel.lpc_order as usize,
        signal_type: channel.indices.signal_type as i32,
        nb_subfr: channel.nb_subfr as usize,
        num_bits,
    }
}
