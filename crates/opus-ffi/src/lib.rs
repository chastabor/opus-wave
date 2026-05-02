//! FFI bindings to C libopus for cross-validation and benchmarking.
//!
//! Provides safe Rust wrappers (`COpusEncoder`, `COpusDecoder`) around the
//! C reference Opus encoder/decoder, allowing side-by-side comparison with
//! the pure-Rust implementation.

use std::ptr;

// ── Raw FFI declarations ──

// Opus constants (matching opus_defines.h)
pub const OPUS_APPLICATION_VOIP: i32 = 2048;
pub const OPUS_APPLICATION_AUDIO: i32 = 2049;
pub const OPUS_APPLICATION_RESTRICTED_LOWDELAY: i32 = 2051;

pub const OPUS_AUTO: i32 = -1000;
pub const OPUS_BITRATE_MAX: i32 = -1;

pub const OPUS_BANDWIDTH_NARROWBAND: i32 = 1101;
pub const OPUS_BANDWIDTH_MEDIUMBAND: i32 = 1102;
pub const OPUS_BANDWIDTH_WIDEBAND: i32 = 1103;
pub const OPUS_BANDWIDTH_SUPERWIDEBAND: i32 = 1104;
pub const OPUS_BANDWIDTH_FULLBAND: i32 = 1105;

pub const OPUS_SIGNAL_VOICE: i32 = 3001;
pub const OPUS_SIGNAL_MUSIC: i32 = 3002;

// Opaque C types
#[repr(C)]
pub struct OpusEncoderC {
    _private: [u8; 0],
}

#[repr(C)]
pub struct OpusDecoderC {
    _private: [u8; 0],
}

unsafe extern "C" {
    // SILK low-level functions (for cross-validation)
    fn silk_A2NLSF(
        nlsf: *mut i16,  // O: NLSFs in Q15 [d]
        a_q16: *mut i32, // I/O: LPC coefficients in Q16 [d]
        d: i32,          // I: filter order (must be even)
    );

    // silk_burg_modified_c: only available with OPUS_FIXED_POINT=ON
    // Verified identical to Rust with fixed-point build.

    // Float DSP leaf functions (Layer 0)
    fn silk_energy_FLP(data: *const f32, data_length: i32) -> f64;
    // Note: the C float build dispatches inner_product via arch detection.
    // The generic C fallback is silk_inner_product_FLP_c.
    #[link_name = "silk_inner_product_FLP_c"]
    fn silk_inner_product_FLP(data1: *const f32, data2: *const f32, data_length: i32) -> f64;
    fn silk_schur_FLP(refl_coef: *mut f32, auto_corr: *const f32, order: i32) -> f32;
    fn silk_k2a_FLP(a: *mut f32, rc: *const f32, order: i32);
    fn silk_bwexpander_FLP(ar: *mut f32, d: i32, chirp: f32);
    fn silk_apply_sine_window_FLP(px_win: *mut f32, px: *const f32, win_type: i32, length: i32);
    fn silk_warped_autocorrelation_FLP(
        corr: *mut f32,
        input: *const f32,
        warping: f32,
        length: i32,
        order: i32,
    );
    fn silk_scale_copy_vector_FLP(
        data_out: *mut f32,
        data_in: *const f32,
        gain: f32,
        data_size: i32,
    );
    fn silk_LPC_analysis_filter_FLP(
        r_lpc: *mut f32,
        pred_coef: *const f32,
        s: *const f32,
        length: i32,
        order: i32,
    );
    fn silk_LPC_inverse_pred_gain_FLP(a: *const f32, order: i32) -> f32;
    fn silk_autocorrelation_FLP(
        results: *mut f32,
        input: *const f32,
        input_size: i32,
        corr_count: i32,
        arch: i32,
    );

    // LTP analysis (Layer 0 / Layer 3)
    fn silk_corrVector_FLP(
        x: *const f32,
        t: *const f32,
        l: i32,
        order: i32,
        xt: *mut f32,
        arch: i32,
    );
    fn silk_corrMatrix_FLP(x: *const f32, l: i32, order: i32, xx: *mut f32, arch: i32);
    fn silk_find_LTP_FLP(
        xx: *mut f32,
        x_x: *mut f32,
        r_ptr: *const f32,
        lag: *const i32,
        subfr_length: i32,
        nb_subfr: i32,
        arch: i32,
    );
    fn silk_quant_LTP_gains(
        b_q14: *mut i16,
        cbk_index: *mut i8,
        periodicity_index: *mut i8,
        sum_log_gain_q7: *mut i32,
        pred_gain_db_q7: *mut i32,
        xx_q17: *const i32,
        x_x_q17: *const i32,
        subfr_len: i32,
        nb_subfr: i32,
        arch: i32,
    );

    fn silk_NLSF2A(
        a_q12: *mut i16,      // O: monic whitening filter coefficients in Q12 [d]
        nlsf_q15: *const i16, // I: normalized line spectral frequencies in Q15 [d]
        d: i32,               // I: filter order (should be even)
        arch: i32,            // I: run-time architecture
    );

    fn silk_gains_quant(
        ind: *mut i8,       // O: gain indices [nb_subfr]
        gain_q16: *mut i32, // I/O: gains (quantized out) [nb_subfr]
        prev_ind: *mut i8,  // I/O: last index in previous frame
        conditional: i32,   // I: first gain is delta coded if 1
        nb_subfr: i32,      // I: number of subframes
    );

    fn silk_gains_dequant(
        gain_q16: *mut i32, // O: quantized gains [nb_subfr]
        ind: *const i8,     // I: gain indices [nb_subfr]
        prev_ind: *mut i8,  // I/O: last index in previous frame
        conditional: i32,   // I: first gain is delta coded if 1
        nb_subfr: i32,      // I: number of subframes
    );

    fn silk_interpolate(
        xi: *mut i16,   // O: interpolated vector [d]
        x0: *const i16, // I: first vector [d]
        x1: *const i16, // I: second vector [d]
        ifact_q2: i32,  // I: interp. factor, weight on 2nd vector
        d: i32,         // I: number of parameters
    );

    fn silk_NLSF_VQ_weights_laroia(
        pNLSFW_Q_OUT: *mut i16, // O: NLSF weights [order]
        pNLSF_Q15: *const i16,  // I: NLSFs [order]
        order: i32,             // I: filter order
    );

    // Codebook struct is opaque from Rust side; we access it via pointer
    static silk_NLSF_CB_WB: u8; // address-only — we pass &silk_NLSF_CB_WB as *const c_void

    fn silk_NLSF_encode(
        nlsf_indices: *mut i8, // O: codebook path [order+1]
        pNLSF_Q15: *mut i16,   // I/O: quantized NLSFs [order]
        psNLSF_CB: *const u8,  // I: codebook struct pointer
        pW_QW: *const i16,     // I: NLSF weights [order]
        NLSF_mu_Q20: i32,      // I: rate weight
        nSurvivors: i32,       // I: max survivors
        signalType: i32,       // I: signal type 0/1/2
    ) -> i32;

    // Core encoder API
    fn opus_encoder_create(
        fs: i32,
        channels: i32,
        application: i32,
        error: *mut i32,
    ) -> *mut OpusEncoderC;
    fn opus_encoder_destroy(enc: *mut OpusEncoderC);
    fn opus_encode_float(
        enc: *mut OpusEncoderC,
        pcm: *const f32,
        frame_size: i32,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;

    // Core decoder API
    fn opus_decoder_create(fs: i32, channels: i32, error: *mut i32) -> *mut OpusDecoderC;
    fn opus_decoder_destroy(dec: *mut OpusDecoderC);
    fn opus_decode_float(
        dec: *mut OpusDecoderC,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: i32,
        decode_fec: i32,
    ) -> i32;

    // Non-variadic CTL wrappers (from wrapper.c)
    fn opus_enc_set_bitrate(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_complexity(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_max_bandwidth(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_bandwidth(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_vbr(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_signal(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_force_channels(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_inband_fec(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_set_packet_loss_perc(enc: *mut OpusEncoderC, val: i32) -> i32;
    fn opus_enc_get_final_range(enc: *mut OpusEncoderC, val: *mut u32) -> i32;
    fn opus_enc_reset(enc: *mut OpusEncoderC) -> i32;
    fn opus_dec_get_final_range(dec: *mut OpusDecoderC, val: *mut u32) -> i32;
    fn opus_dec_reset(dec: *mut OpusDecoderC) -> i32;
}

// ── Error handling ──

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct COpusError(pub i32);

impl std::fmt::Display for COpusError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "C opus error code {}", self.0)
    }
}

impl std::error::Error for COpusError {}

fn check(code: i32) -> Result<(), COpusError> {
    if code < 0 {
        Err(COpusError(code))
    } else {
        Ok(())
    }
}

// ── Safe Encoder Wrapper ──

pub struct COpusEncoder {
    raw: *mut OpusEncoderC,
}

// SAFETY: The C encoder is single-threaded but we never share it across threads.
unsafe impl Send for COpusEncoder {}

impl COpusEncoder {
    pub fn new(fs: i32, channels: i32, application: i32) -> Result<Self, COpusError> {
        let mut error = 0i32;
        let raw = unsafe { opus_encoder_create(fs, channels, application, &mut error) };
        if raw.is_null() || error < 0 {
            return Err(COpusError(error));
        }
        Ok(Self { raw })
    }

    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        output: &mut [u8],
    ) -> Result<i32, COpusError> {
        let ret = unsafe {
            opus_encode_float(
                self.raw,
                pcm.as_ptr(),
                frame_size,
                output.as_mut_ptr(),
                output.len() as i32,
            )
        };
        if ret < 0 {
            Err(COpusError(ret))
        } else {
            Ok(ret)
        }
    }

    pub fn set_bitrate(&mut self, bitrate: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_bitrate(self.raw, bitrate) })
    }

    pub fn set_complexity(&mut self, complexity: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_complexity(self.raw, complexity) })
    }

    pub fn set_max_bandwidth(&mut self, bw: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_max_bandwidth(self.raw, bw) })
    }

    pub fn set_bandwidth(&mut self, bw: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_bandwidth(self.raw, bw) })
    }

    pub fn set_vbr(&mut self, enabled: bool) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_vbr(self.raw, enabled as i32) })
    }

    pub fn set_signal(&mut self, signal: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_signal(self.raw, signal) })
    }

    pub fn set_force_channels(&mut self, channels: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_force_channels(self.raw, channels) })
    }

    pub fn set_inband_fec(&mut self, enabled: bool) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_inband_fec(self.raw, enabled as i32) })
    }

    pub fn set_packet_loss_perc(&mut self, perc: i32) -> Result<(), COpusError> {
        check(unsafe { opus_enc_set_packet_loss_perc(self.raw, perc) })
    }

    pub fn final_range(&mut self) -> u32 {
        let mut val = 0u32;
        unsafe { opus_enc_get_final_range(self.raw, &mut val) };
        val
    }

    pub fn reset(&mut self) -> Result<(), COpusError> {
        check(unsafe { opus_enc_reset(self.raw) })
    }
}

impl Drop for COpusEncoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_encoder_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

// ── Safe Decoder Wrapper ──

pub struct COpusDecoder {
    raw: *mut OpusDecoderC,
}

unsafe impl Send for COpusDecoder {}

impl COpusDecoder {
    pub fn new(fs: i32, channels: i32) -> Result<Self, COpusError> {
        let mut error = 0i32;
        let raw = unsafe { opus_decoder_create(fs, channels, &mut error) };
        if raw.is_null() || error < 0 {
            return Err(COpusError(error));
        }
        Ok(Self { raw })
    }

    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, COpusError> {
        let (data_ptr, data_len) = match data {
            Some(d) => (d.as_ptr(), d.len() as i32),
            None => (ptr::null(), 0),
        };
        let ret = unsafe {
            opus_decode_float(
                self.raw,
                data_ptr,
                data_len,
                pcm.as_mut_ptr(),
                frame_size,
                decode_fec as i32,
            )
        };
        if ret < 0 {
            Err(COpusError(ret))
        } else {
            Ok(ret)
        }
    }

    pub fn final_range(&mut self) -> u32 {
        let mut val = 0u32;
        unsafe { opus_dec_get_final_range(self.raw, &mut val) };
        val
    }

    pub fn reset(&mut self) -> Result<(), COpusError> {
        check(unsafe { opus_dec_reset(self.raw) })
    }
}

impl Drop for COpusDecoder {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { opus_decoder_destroy(self.raw) };
            self.raw = ptr::null_mut();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encoder_creates_and_encodes() {
        let mut enc = COpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000).unwrap();
        let pcm = vec![0.0f32; 960];
        let mut output = vec![0u8; 4000];
        let len = enc.encode_float(&pcm, 960, &mut output).unwrap();
        assert!(len > 0);
    }

    #[test]
    fn decoder_creates_and_decodes() {
        // Encode a frame, then decode it.
        let mut enc = COpusEncoder::new(48000, 1, OPUS_APPLICATION_AUDIO).unwrap();
        enc.set_bitrate(64000).unwrap();
        let pcm_in = vec![0.0f32; 960];
        let mut packet = vec![0u8; 4000];
        let pkt_len = enc.encode_float(&pcm_in, 960, &mut packet).unwrap();

        let mut dec = COpusDecoder::new(48000, 1).unwrap();
        let mut pcm_out = vec![0.0f32; 960];
        let samples = dec
            .decode_float(Some(&packet[..pkt_len as usize]), &mut pcm_out, 960, false)
            .unwrap();
        assert_eq!(samples, 960);
    }
}

// ── SILK low-level safe wrappers ──

/// Call the C reference silk_A2NLSF to convert LPC coefficients to NLSFs.
/// Both `nlsf_q15` and `a_q16` must have length >= `order`.
/// Note: `a_q16` is modified in place (bandwidth expansion during root search).
pub fn c_silk_a2nlsf(nlsf_q15: &mut [i16], a_q16: &mut [i32], order: usize) {
    assert!(nlsf_q15.len() >= order && a_q16.len() >= order);
    unsafe {
        silk_A2NLSF(nlsf_q15.as_mut_ptr(), a_q16.as_mut_ptr(), order as i32);
    }
}

// c_silk_burg_modified: only available with OPUS_FIXED_POINT=ON.
// Verified identical to Rust silk_burg_modified.

/// Call the C reference silk_NLSF2A to convert NLSFs to LPC coefficients.
/// `a_q12` and `nlsf_q15` must have length >= `order`.
pub fn c_silk_nlsf2a(a_q12: &mut [i16], nlsf_q15: &[i16], order: usize) {
    assert!(a_q12.len() >= order && nlsf_q15.len() >= order);
    unsafe {
        silk_NLSF2A(a_q12.as_mut_ptr(), nlsf_q15.as_ptr(), order as i32, 0);
    }
}

/// Call the C reference silk_gains_quant for gain scalar quantization.
/// `ind` and `gain_q16` must have length >= `nb_subfr`.
pub fn c_silk_gains_quant(
    ind: &mut [i8],
    gain_q16: &mut [i32],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    assert!(ind.len() >= nb_subfr && gain_q16.len() >= nb_subfr);
    unsafe {
        silk_gains_quant(
            ind.as_mut_ptr(),
            gain_q16.as_mut_ptr(),
            prev_ind,
            conditional as i32,
            nb_subfr as i32,
        );
    }
}

/// Call the C reference silk_gains_dequant for gain scalar dequantization.
/// `gain_q16` and `ind` must have length >= `nb_subfr`.
pub fn c_silk_gains_dequant(
    gain_q16: &mut [i32],
    ind: &[i8],
    prev_ind: &mut i8,
    conditional: bool,
    nb_subfr: usize,
) {
    assert!(gain_q16.len() >= nb_subfr && ind.len() >= nb_subfr);
    unsafe {
        silk_gains_dequant(
            gain_q16.as_mut_ptr(),
            ind.as_ptr(),
            prev_ind,
            conditional as i32,
            nb_subfr as i32,
        );
    }
}

/// Call the C reference silk_interpolate to interpolate two vectors.
/// `xi`, `x0`, and `x1` must have length >= `d`.
pub fn c_silk_interpolate(xi: &mut [i16], x0: &[i16], x1: &[i16], ifact_q2: i32, d: usize) {
    assert!(xi.len() >= d && x0.len() >= d && x1.len() >= d);
    unsafe {
        silk_interpolate(
            xi.as_mut_ptr(),
            x0.as_ptr(),
            x1.as_ptr(),
            ifact_q2,
            d as i32,
        );
    }
}

pub fn c_silk_nlsf_vq_weights_laroia(weights: &mut [i16], nlsf_q15: &[i16], order: usize) {
    assert!(weights.len() >= order && nlsf_q15.len() >= order);
    unsafe {
        silk_NLSF_VQ_weights_laroia(weights.as_mut_ptr(), nlsf_q15.as_ptr(), order as i32);
    }
}

/// Call the C reference NLSF encoder (VQ + trellis) for the WB codebook.
/// Returns RD in Q25. `nlsf_indices` must be [order+1], `nlsf_q15` and `w_q2` must be [order].
pub fn c_silk_nlsf_encode_wb(
    nlsf_indices: &mut [i8],
    nlsf_q15: &mut [i16],
    w_q2: &[i16],
    mu_q20: i32,
    n_survivors: i32,
    signal_type: i32,
) -> i32 {
    unsafe {
        silk_NLSF_encode(
            nlsf_indices.as_mut_ptr(),
            nlsf_q15.as_mut_ptr(),
            &silk_NLSF_CB_WB as *const u8,
            w_q2.as_ptr(),
            mu_q20,
            n_survivors,
            signal_type,
        )
    }
}

// ── Float DSP leaf function wrappers (Layer 0) ──

pub fn c_silk_energy_flp(data: &[f32]) -> f64 {
    unsafe { silk_energy_FLP(data.as_ptr(), data.len() as i32) }
}

pub fn c_silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    unsafe { silk_inner_product_FLP(data1.as_ptr(), data2.as_ptr(), n as i32) }
}

pub fn c_silk_schur_flp(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    unsafe { silk_schur_FLP(refl_coef.as_mut_ptr(), auto_corr.as_ptr(), order as i32) }
}

pub fn c_silk_k2a_flp(a: &mut [f32], rc: &[f32], order: usize) {
    unsafe { silk_k2a_FLP(a.as_mut_ptr(), rc.as_ptr(), order as i32) }
}

pub fn c_silk_bwexpander_flp(ar: &mut [f32], d: usize, chirp: f32) {
    unsafe { silk_bwexpander_FLP(ar.as_mut_ptr(), d as i32, chirp) }
}

pub fn c_silk_apply_sine_window_flp(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    unsafe { silk_apply_sine_window_FLP(px_win.as_mut_ptr(), px.as_ptr(), win_type, length as i32) }
}

pub fn c_silk_scale_copy_vector_flp(data_out: &mut [f32], data_in: &[f32], gain: f32, len: usize) {
    unsafe { silk_scale_copy_vector_FLP(data_out.as_mut_ptr(), data_in.as_ptr(), gain, len as i32) }
}

pub fn c_silk_lpc_analysis_filter_flp(
    r_lpc: &mut [f32],
    pred_coef: &[f32],
    s: &[f32],
    length: usize,
    order: usize,
) {
    unsafe {
        silk_LPC_analysis_filter_FLP(
            r_lpc.as_mut_ptr(),
            pred_coef.as_ptr(),
            s.as_ptr(),
            length as i32,
            order as i32,
        )
    }
}

pub fn c_silk_warped_autocorrelation_flp(
    corr: &mut [f32],
    input: &[f32],
    warping: f32,
    length: usize,
    order: usize,
) {
    unsafe {
        silk_warped_autocorrelation_FLP(
            corr.as_mut_ptr(),
            input.as_ptr(),
            warping,
            length as i32,
            order as i32,
        )
    }
}

pub fn c_silk_lpc_inverse_pred_gain_flp(a: &[f32], order: usize) -> f32 {
    unsafe { silk_LPC_inverse_pred_gain_FLP(a.as_ptr(), order as i32) }
}

pub fn c_silk_autocorrelation_flp(results: &mut [f32], input: &[f32], corr_count: usize) {
    unsafe {
        silk_autocorrelation_FLP(
            results.as_mut_ptr(),
            input.as_ptr(),
            input.len() as i32,
            corr_count as i32,
            0,
        )
    }
}

// ── LTP FFI wrappers ──

pub fn c_silk_corr_vector_flp(x: &[f32], t: &[f32], l: usize, order: usize, xt: &mut [f32]) {
    unsafe {
        silk_corrVector_FLP(
            x.as_ptr(),
            t.as_ptr(),
            l as i32,
            order as i32,
            xt.as_mut_ptr(),
            0,
        )
    }
}

pub fn c_silk_corr_matrix_flp(x: &[f32], l: usize, order: usize, xx: &mut [f32]) {
    unsafe { silk_corrMatrix_FLP(x.as_ptr(), l as i32, order as i32, xx.as_mut_ptr(), 0) }
}

pub fn c_silk_find_ltp_flp(
    xx: &mut [f32],
    x_x: &mut [f32],
    res: &[f32],
    frame_offset: usize,
    lag: &[i32],
    subfr_length: i32,
    nb_subfr: i32,
) {
    let r_ptr = unsafe { res.as_ptr().add(frame_offset) };
    unsafe {
        silk_find_LTP_FLP(
            xx.as_mut_ptr(),
            x_x.as_mut_ptr(),
            r_ptr,
            lag.as_ptr(),
            subfr_length,
            nb_subfr,
            0,
        )
    }
}

pub fn c_silk_quant_ltp_gains(
    b_q14: &mut [i16],
    cbk_index: &mut [i8],
    periodicity_index: &mut i8,
    sum_log_gain_q7: &mut i32,
    pred_gain_db_q7: &mut i32,
    xx_q17: &[i32],
    x_x_q17: &[i32],
    subfr_len: i32,
    nb_subfr: i32,
) {
    unsafe {
        silk_quant_LTP_gains(
            b_q14.as_mut_ptr(),
            cbk_index.as_mut_ptr(),
            periodicity_index,
            sum_log_gain_q7,
            pred_gain_db_q7,
            xx_q17.as_ptr(),
            x_x_q17.as_ptr(),
            subfr_len,
            nb_subfr,
            0,
        )
    }
}

// ── Float wrapper FFI (Layer 2) ──

// silk_A2NLSF_FLP and silk_NLSF2A_FLP are thin wrappers around the fixed-point
// versions. We test them by calling the C float wrapper and comparing with our
// Rust float wrapper (which calls the same fixed-point core).

// The Burg float is also needed for generating realistic test LPC coefficients.
unsafe extern "C" {
    fn silk_burg_modified_FLP(
        a: *mut f32,
        x: *const f32,
        min_inv_gain: f32,
        subfr_length: i32,
        nb_subfr: i32,
        d: i32,
        arch: i32,
    ) -> f32;

    fn silk_A2NLSF_FLP(nlsf_q15: *mut i16, a: *const f32, order: i32);
    fn silk_NLSF2A_FLP(a: *mut f32, nlsf_q15: *const i16, order: i32, arch: i32);

    fn silk_LTP_analysis_filter_FLP(
        ltp_res: *mut f32,     // O: LTP residual [nb_subfr * (pre_length + subfr_length)]
        x: *const f32,         // I: input signal, with preceding samples
        b: *const f32,         // I: LTP coefficients [LTP_ORDER * MAX_NB_SUBFR]
        pitch_l: *const i32,   // I: pitch lags [MAX_NB_SUBFR]
        inv_gains: *const f32, // I: inverse quantization gains [MAX_NB_SUBFR]
        subfr_length: i32,     // I: length of each subframe
        nb_subfr: i32,         // I: number of subframes
        pre_length: i32,       // I: preceding samples for each subframe
    );
}

pub fn c_silk_burg_modified_flp(
    a: &mut [f32],
    x: &[f32],
    min_inv_gain: f32,
    subfr_length: i32,
    nb_subfr: i32,
    d: i32,
) -> f32 {
    unsafe {
        silk_burg_modified_FLP(
            a.as_mut_ptr(),
            x.as_ptr(),
            min_inv_gain,
            subfr_length,
            nb_subfr,
            d,
            0,
        )
    }
}

pub fn c_silk_a2nlsf_flp(nlsf_q15: &mut [i16], a: &[f32], order: usize) {
    unsafe { silk_A2NLSF_FLP(nlsf_q15.as_mut_ptr(), a.as_ptr(), order as i32) }
}

pub fn c_silk_nlsf2a_flp(a: &mut [f32], nlsf_q15: &[i16], order: usize) {
    unsafe { silk_NLSF2A_FLP(a.as_mut_ptr(), nlsf_q15.as_ptr(), order as i32, 0) }
}

/// Call the C reference silk_LTP_analysis_filter_FLP.
/// `ltp_res` must have length >= `nb_subfr * (pre_length + subfr_length)`.
/// `x` is the input signal with preceding samples.
/// `b` must have length >= `LTP_ORDER * nb_subfr` (LTP coefficients per subframe).
/// `pitch_l` and `inv_gains` must have length >= `nb_subfr`.
pub fn c_silk_ltp_analysis_filter_flp(
    ltp_res: &mut [f32],
    x: &[f32],
    b: &[f32],
    pitch_l: &[i32],
    inv_gains: &[f32],
    subfr_length: usize,
    nb_subfr: usize,
    pre_length: usize,
) {
    assert!(ltp_res.len() >= nb_subfr * (pre_length + subfr_length));
    const LTP_ORDER: usize = 5;
    assert!(b.len() >= LTP_ORDER * nb_subfr);
    assert!(pitch_l.len() >= nb_subfr && inv_gains.len() >= nb_subfr);
    unsafe {
        silk_LTP_analysis_filter_FLP(
            ltp_res.as_mut_ptr(),
            x.as_ptr(),
            b.as_ptr(),
            pitch_l.as_ptr(),
            inv_gains.as_ptr(),
            subfr_length as i32,
            nb_subfr as i32,
            pre_length as i32,
        );
    }
}

// Float residual energy
unsafe extern "C" {
    fn silk_residual_energy_FLP(
        nrgs: *mut f32,
        x: *const f32,
        a: *const f32, // a[2][MAX_LPC_ORDER] flattened
        gains: *const f32,
        subfr_length: i32,
        nb_subfr: i32,
        lpc_order: i32,
    );
}

pub fn c_silk_residual_energy_flp(
    nrgs: &mut [f32],
    x: &[f32],
    a: &[[f32; 16]; 2],
    gains: &[f32],
    subfr_length: i32,
    nb_subfr: i32,
    lpc_order: i32,
) {
    unsafe {
        silk_residual_energy_FLP(
            nrgs.as_mut_ptr(),
            x.as_ptr(),
            a.as_ptr() as *const f32,
            gains.as_ptr(),
            subfr_length,
            nb_subfr,
            lpc_order,
        );
    }
}

// ══════════════════════════════════════════════════════════════════════
// CELT low-level FFI declarations
// ══════════════════════════════════════════════════════════════════════

unsafe extern "C" {
    // ── Group A: Direct extern (symbols in libopus.a) ──

    fn isqrt32(val: u32) -> u32;
    fn bitexact_cos(x: i16) -> i16;
    fn bitexact_log2tan(isin: i32, icos: i32) -> i32;
    fn celt_lcg_rand(seed: u32) -> u32;

    #[link_name = "_celt_lpc"]
    fn c_celt_lpc_raw(lpc: *mut f32, ac: *const f32, p: i32);

    fn celt_fir_c(x: *const f32, num: *const f32, y: *mut f32, n: i32, ord: i32, arch: i32);

    fn celt_iir(
        x: *const f32,
        den: *const f32,
        y: *mut f32,
        n: i32,
        ord: i32,
        mem: *mut f32,
        arch: i32,
    );

    #[link_name = "_celt_autocorr"]
    fn c_celt_autocorr_raw(
        x: *const f32,
        ac: *mut f32,
        window: *const f32,
        overlap: i32,
        lag: i32,
        n: i32,
        arch: i32,
    ) -> i32;

    fn celt_pitch_xcorr_c(
        x: *const f32,
        y: *const f32,
        xcorr: *mut f32,
        len: i32,
        max_pitch: i32,
        arch: i32,
    );

    fn renormalise_vector(x: *mut f32, n: i32, gain: f32, arch: i32);

    // ── Group B: Via celt_wrapper.c shims ──

    // Math (static inline in C headers)
    fn wrap_celt_exp2(x: f32) -> f32;
    fn wrap_celt_log2(x: f32) -> f32;
    fn wrap_celt_inner_prod(x: *const f32, y: *const f32, n: i32) -> f32;
    fn wrap_celt_maxabs16(x: *const f32, len: i32) -> f32;
    fn wrap_celt_rcp(x: f32) -> f32;
    fn wrap_frac_mul16(a: i32, b: i32) -> i32;

    // FFT (requires kiss_fft_state*)
    fn wrap_opus_fft(
        nfft: i32,
        fin_r: *const f32,
        fin_i: *const f32,
        fout_r: *mut f32,
        fout_i: *mut f32,
    );

    // MDCT (requires mdct_lookup*)
    fn wrap_clt_mdct_forward(
        input: *mut f32,
        output: *mut f32,
        n: i32,
        overlap: i32,
        shift: i32,
        stride: i32,
    );
    fn wrap_clt_mdct_backward(
        input: *mut f32,
        output: *mut f32,
        n: i32,
        overlap: i32,
        shift: i32,
        stride: i32,
    );

    // Pitch (array-of-pointers or arch param)
    fn wrap_pitch_downsample_mono(x: *mut f32, x_lp: *mut f32, len: i32);
    fn wrap_pitch_search(x_lp: *const f32, y: *mut f32, len: i32, max_pitch: i32, pitch: *mut i32);
    fn wrap_remove_doubling(
        x: *mut f32,
        maxperiod: i32,
        minperiod: i32,
        n: i32,
        t0: *mut i32,
        prev_period: i32,
        prev_gain: f32,
    ) -> f32;
    fn wrap_comb_filter(
        y: *mut f32,
        x: *mut f32,
        t0: i32,
        t1: i32,
        n: i32,
        g0: f32,
        g1: f32,
        tapset0: i32,
        tapset1: i32,
        overlap: i32,
    );

    // Band processing (CELTMode-dependent)
    fn wrap_compute_band_energies(x: *const f32, band_e: *mut f32, end: i32, c: i32, lm: i32);
    fn wrap_normalise_bands(
        freq: *const f32,
        x: *mut f32,
        band_e: *const f32,
        end: i32,
        c: i32,
        m: i32,
    );
    fn wrap_denormalise_bands(
        x: *const f32,
        freq: *mut f32,
        band_log_e: *const f32,
        start: i32,
        end: i32,
        m: i32,
        downsample: i32,
        silence: i32,
    );

    // Rate allocation (CELTMode-dependent)
    fn wrap_bits2pulses(band: i32, lm: i32, bits: i32) -> i32;
    fn wrap_pulses2bits(band: i32, lm: i32, pulses: i32) -> i32;
    fn wrap_init_caps(cap: *mut i32, lm: i32, c: i32);

    // Energy quantization (CELTMode + ec_enc/ec_dec)
    fn wrap_encode_coarse_energy(
        start: i32,
        end: i32,
        e_bands: *const f32,
        old_band_e: *mut f32,
        error: *mut f32,
        ec_buf: *mut u8,
        ec_buf_size: i32,
        c: i32,
        lm: i32,
        nb_available_bytes: i32,
        force_intra: i32,
        loss_rate: i32,
        lfe: i32,
    ) -> i32;

    fn wrap_decode_coarse_energy(
        start: i32,
        end: i32,
        old_band_e: *mut f32,
        ec_buf: *const u8,
        ec_bytes: i32,
        c: i32,
        lm: i32,
    );

    fn wrap_encode_fine_energy(
        start: i32,
        end: i32,
        old_band_e: *mut f32,
        error: *mut f32,
        fine_quant: *const i32,
        ec_buf: *mut u8,
        ec_buf_size: i32,
        c: i32,
    ) -> i32;

    fn wrap_decode_fine_energy(
        start: i32,
        end: i32,
        old_band_e: *mut f32,
        fine_quant: *const i32,
        ec_buf: *const u8,
        ec_bytes: i32,
        c: i32,
    );

    fn wrap_encode_energy_finalise(
        start: i32,
        end: i32,
        old_band_e: *mut f32,
        error: *mut f32,
        fine_quant: *const i32,
        fine_priority: *const i32,
        bits_left: i32,
        ec_buf: *mut u8,
        ec_buf_size: i32,
        c: i32,
    ) -> i32;

    fn wrap_decode_energy_finalise(
        start: i32,
        end: i32,
        old_band_e: *mut f32,
        fine_quant: *const i32,
        fine_priority: *const i32,
        bits_left: i32,
        ec_buf: *const u8,
        ec_bytes: i32,
        c: i32,
    );

    fn wrap_anti_collapse(
        x: *mut f32,
        collapse_masks: *mut u8,
        lm: i32,
        c: i32,
        size: i32,
        start: i32,
        end: i32,
        log_e: *const f32,
        prev1_log_e: *const f32,
        prev2_log_e: *const f32,
        pulses: *const i32,
        seed: u32,
        encode: i32,
    );

    // Persistent-state wrappers for fair benchmarking
    fn wrap_fft_bench_init(nfft: i32);
    fn wrap_fft_bench_run(fin_r: *const f32, fin_i: *const f32, fout_r: *mut f32, fout_i: *mut f32);
    fn wrap_mdct_bench_init(n: i32);
    fn wrap_mdct_bench_forward(
        input: *mut f32,
        output: *mut f32,
        overlap: i32,
        shift: i32,
        stride: i32,
    );
    fn wrap_mdct_bench_backward(
        input: *mut f32,
        output: *mut f32,
        overlap: i32,
        shift: i32,
        stride: i32,
    );
}

// ══════════════════════════════════════════════════════════════════════
// CELT safe wrappers
// ══════════════════════════════════════════════════════════════════════

// ── Integer math (exact match expected) ──

pub fn c_isqrt32(val: u32) -> u32 {
    unsafe { isqrt32(val) }
}

pub fn c_bitexact_cos(x: i16) -> i16 {
    unsafe { bitexact_cos(x) }
}

pub fn c_bitexact_log2tan(isin: i32, icos: i32) -> i32 {
    unsafe { bitexact_log2tan(isin, icos) }
}

pub fn c_celt_lcg_rand(seed: u32) -> u32 {
    unsafe { celt_lcg_rand(seed) }
}

pub fn c_frac_mul16(a: i32, b: i32) -> i32 {
    unsafe { wrap_frac_mul16(a, b) }
}

// ── Float math ──

pub fn c_celt_exp2(x: f32) -> f32 {
    unsafe { wrap_celt_exp2(x) }
}

pub fn c_celt_log2(x: f32) -> f32 {
    unsafe { wrap_celt_log2(x) }
}

pub fn c_celt_inner_prod(x: &[f32], y: &[f32]) -> f32 {
    let n = x.len().min(y.len());
    unsafe { wrap_celt_inner_prod(x.as_ptr(), y.as_ptr(), n as i32) }
}

pub fn c_celt_maxabs16(x: &[f32]) -> f32 {
    unsafe { wrap_celt_maxabs16(x.as_ptr(), x.len() as i32) }
}

pub fn c_celt_rcp(x: f32) -> f32 {
    unsafe { wrap_celt_rcp(x) }
}

pub fn c_renormalise_vector(x: &mut [f32], gain: f32) {
    unsafe { renormalise_vector(x.as_mut_ptr(), x.len() as i32, gain, 0) }
}

// ── LPC ──

pub fn c_celt_lpc(lpc: &mut [f32], ac: &[f32], p: usize) {
    assert!(lpc.len() >= p && ac.len() > p);
    unsafe { c_celt_lpc_raw(lpc.as_mut_ptr(), ac.as_ptr(), p as i32) }
}

pub fn c_celt_fir(x: &[f32], num: &[f32], y: &mut [f32], n: usize, ord: usize) {
    assert!(x.len() >= n && num.len() >= ord && y.len() >= n);
    unsafe {
        celt_fir_c(
            x.as_ptr(),
            num.as_ptr(),
            y.as_mut_ptr(),
            n as i32,
            ord as i32,
            0,
        )
    }
}

pub fn c_celt_iir(x: &[f32], den: &[f32], y: &mut [f32], n: usize, ord: usize, mem: &mut [f32]) {
    assert!(x.len() >= n && den.len() >= ord && y.len() >= n && mem.len() >= ord);
    unsafe {
        celt_iir(
            x.as_ptr(),
            den.as_ptr(),
            y.as_mut_ptr(),
            n as i32,
            ord as i32,
            mem.as_mut_ptr(),
            0,
        )
    }
}

pub fn c_celt_autocorr(
    x: &[f32],
    ac: &mut [f32],
    window: Option<&[f32]>,
    overlap: usize,
    lag: usize,
    n: usize,
) -> i32 {
    let win_ptr = match window {
        Some(w) => w.as_ptr(),
        None => ptr::null(),
    };
    unsafe {
        c_celt_autocorr_raw(
            x.as_ptr(),
            ac.as_mut_ptr(),
            win_ptr,
            overlap as i32,
            lag as i32,
            n as i32,
            0,
        )
    }
}

// ── Pitch ──

pub fn c_celt_pitch_xcorr(x: &[f32], y: &[f32], xcorr: &mut [f32], len: usize, max_pitch: usize) {
    unsafe {
        celt_pitch_xcorr_c(
            x.as_ptr(),
            y.as_ptr(),
            xcorr.as_mut_ptr(),
            len as i32,
            max_pitch as i32,
            0,
        )
    }
}

pub fn c_pitch_downsample_mono(x: &mut [f32], x_lp: &mut [f32], len: usize) {
    unsafe { wrap_pitch_downsample_mono(x.as_mut_ptr(), x_lp.as_mut_ptr(), len as i32) }
}

pub fn c_pitch_search(x_lp: &[f32], y: &mut [f32], len: usize, max_pitch: usize) -> i32 {
    let mut pitch = 0i32;
    unsafe {
        wrap_pitch_search(
            x_lp.as_ptr(),
            y.as_mut_ptr(),
            len as i32,
            max_pitch as i32,
            &mut pitch,
        );
    }
    pitch
}

pub fn c_remove_doubling(
    x: &mut [f32],
    maxperiod: usize,
    minperiod: usize,
    n: usize,
    t0: &mut i32,
    prev_period: i32,
    prev_gain: f32,
) -> f32 {
    unsafe {
        wrap_remove_doubling(
            x.as_mut_ptr(),
            maxperiod as i32,
            minperiod as i32,
            n as i32,
            t0,
            prev_period,
            prev_gain,
        )
    }
}

pub fn c_comb_filter(
    y: &mut [f32],
    x: &mut [f32],
    t0: i32,
    t1: i32,
    n: usize,
    g0: f32,
    g1: f32,
    tapset0: i32,
    tapset1: i32,
    overlap: usize,
) {
    unsafe {
        wrap_comb_filter(
            y.as_mut_ptr(),
            x.as_mut_ptr(),
            t0,
            t1,
            n as i32,
            g0,
            g1,
            tapset0,
            tapset1,
            overlap as i32,
        )
    }
}

// ── FFT ──

pub fn c_opus_fft(
    nfft: usize,
    fin_r: &[f32],
    fin_i: &[f32],
    fout_r: &mut [f32],
    fout_i: &mut [f32],
) {
    assert!(fin_r.len() >= nfft && fin_i.len() >= nfft);
    assert!(fout_r.len() >= nfft && fout_i.len() >= nfft);
    unsafe {
        wrap_opus_fft(
            nfft as i32,
            fin_r.as_ptr(),
            fin_i.as_ptr(),
            fout_r.as_mut_ptr(),
            fout_i.as_mut_ptr(),
        )
    }
}

// ── MDCT ──

pub fn c_clt_mdct_forward(
    input: &mut [f32],
    output: &mut [f32],
    n: usize,
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    unsafe {
        wrap_clt_mdct_forward(
            input.as_mut_ptr(),
            output.as_mut_ptr(),
            n as i32,
            overlap as i32,
            shift as i32,
            stride as i32,
        )
    }
}

pub fn c_clt_mdct_backward(
    input: &mut [f32],
    output: &mut [f32],
    n: usize,
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    unsafe {
        wrap_clt_mdct_backward(
            input.as_mut_ptr(),
            output.as_mut_ptr(),
            n as i32,
            overlap as i32,
            shift as i32,
            stride as i32,
        )
    }
}

// ── Band processing ──

pub fn c_compute_band_energies(x: &[f32], band_e: &mut [f32], end: usize, c: usize, lm: usize) {
    unsafe {
        wrap_compute_band_energies(
            x.as_ptr(),
            band_e.as_mut_ptr(),
            end as i32,
            c as i32,
            lm as i32,
        )
    }
}

pub fn c_normalise_bands(
    freq: &[f32],
    x: &mut [f32],
    band_e: &[f32],
    end: usize,
    c: usize,
    m: usize,
) {
    unsafe {
        wrap_normalise_bands(
            freq.as_ptr(),
            x.as_mut_ptr(),
            band_e.as_ptr(),
            end as i32,
            c as i32,
            m as i32,
        )
    }
}

pub fn c_denormalise_bands(
    x: &[f32],
    freq: &mut [f32],
    band_log_e: &[f32],
    start: usize,
    end: usize,
    m: usize,
    downsample: usize,
    silence: bool,
) {
    unsafe {
        wrap_denormalise_bands(
            x.as_ptr(),
            freq.as_mut_ptr(),
            band_log_e.as_ptr(),
            start as i32,
            end as i32,
            m as i32,
            downsample as i32,
            silence as i32,
        )
    }
}

// ── Rate allocation ──

pub fn c_bits2pulses(band: usize, lm: usize, bits: i32) -> i32 {
    unsafe { wrap_bits2pulses(band as i32, lm as i32, bits) }
}

pub fn c_pulses2bits(band: usize, lm: usize, pulses: i32) -> i32 {
    unsafe { wrap_pulses2bits(band as i32, lm as i32, pulses) }
}

pub fn c_init_caps(cap: &mut [i32], lm: usize, c: usize) {
    unsafe { wrap_init_caps(cap.as_mut_ptr(), lm as i32, c as i32) }
}

// ── Energy quantization ──

/// Encode coarse energy with C reference. Returns number of encoded bytes.
pub fn c_encode_coarse_energy(
    start: usize,
    end: usize,
    e_bands: &[f32],
    old_band_e: &mut [f32],
    error: &mut [f32],
    ec_buf: &mut [u8],
    c: usize,
    lm: usize,
    nb_available_bytes: usize,
    force_intra: bool,
    loss_rate: i32,
    lfe: bool,
) -> usize {
    let ret = unsafe {
        wrap_encode_coarse_energy(
            start as i32,
            end as i32,
            e_bands.as_ptr(),
            old_band_e.as_mut_ptr(),
            error.as_mut_ptr(),
            ec_buf.as_mut_ptr(),
            ec_buf.len() as i32,
            c as i32,
            lm as i32,
            nb_available_bytes as i32,
            force_intra as i32,
            loss_rate,
            lfe as i32,
        )
    };
    ret as usize
}

/// Decode coarse energy with C reference (reads intra flag from bitstream).
pub fn c_decode_coarse_energy(
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    ec_buf: &[u8],
    c: usize,
    lm: usize,
) {
    unsafe {
        wrap_decode_coarse_energy(
            start as i32,
            end as i32,
            old_band_e.as_mut_ptr(),
            ec_buf.as_ptr(),
            ec_buf.len() as i32,
            c as i32,
            lm as i32,
        )
    }
}

/// Encode fine energy with C reference. Returns number of encoded bytes.
pub fn c_encode_fine_energy(
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    ec_buf: &mut [u8],
    c: usize,
) -> usize {
    let ret = unsafe {
        wrap_encode_fine_energy(
            start as i32,
            end as i32,
            old_band_e.as_mut_ptr(),
            error.as_mut_ptr(),
            fine_quant.as_ptr(),
            ec_buf.as_mut_ptr(),
            ec_buf.len() as i32,
            c as i32,
        )
    };
    ret as usize
}

/// Decode fine energy with C reference.
pub fn c_decode_fine_energy(
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    fine_quant: &[i32],
    ec_buf: &[u8],
    c: usize,
) {
    unsafe {
        wrap_decode_fine_energy(
            start as i32,
            end as i32,
            old_band_e.as_mut_ptr(),
            fine_quant.as_ptr(),
            ec_buf.as_ptr(),
            ec_buf.len() as i32,
            c as i32,
        )
    }
}

/// Encode energy finalise with C reference. Returns number of encoded bytes.
pub fn c_encode_energy_finalise(
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    error: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    bits_left: i32,
    ec_buf: &mut [u8],
    c: usize,
) -> usize {
    unsafe {
        wrap_encode_energy_finalise(
            start as i32,
            end as i32,
            old_band_e.as_mut_ptr(),
            error.as_mut_ptr(),
            fine_quant.as_ptr(),
            fine_priority.as_ptr(),
            bits_left,
            ec_buf.as_mut_ptr(),
            ec_buf.len() as i32,
            c as i32,
        ) as usize
    }
}

/// Decode energy finalise with C reference.
pub fn c_decode_energy_finalise(
    start: usize,
    end: usize,
    old_band_e: &mut [f32],
    fine_quant: &[i32],
    fine_priority: &[i32],
    bits_left: i32,
    ec_buf: &[u8],
    c: usize,
) {
    unsafe {
        wrap_decode_energy_finalise(
            start as i32,
            end as i32,
            old_band_e.as_mut_ptr(),
            fine_quant.as_ptr(),
            fine_priority.as_ptr(),
            bits_left,
            ec_buf.as_ptr(),
            ec_buf.len() as i32,
            c as i32,
        )
    }
}

/// Anti-collapse with C reference.
pub fn c_anti_collapse(
    x: &mut [f32],
    collapse_masks: &mut [u8],
    lm: i32,
    c: usize,
    size: usize,
    start: usize,
    end: usize,
    log_e: &[f32],
    prev1_log_e: &[f32],
    prev2_log_e: &[f32],
    pulses: &[i32],
    seed: u32,
    encode: bool,
) {
    unsafe {
        wrap_anti_collapse(
            x.as_mut_ptr(),
            collapse_masks.as_mut_ptr(),
            lm,
            c as i32,
            size as i32,
            start as i32,
            end as i32,
            log_e.as_ptr(),
            prev1_log_e.as_ptr(),
            prev2_log_e.as_ptr(),
            pulses.as_ptr(),
            seed,
            encode as i32,
        )
    }
}

// ── Persistent-state wrappers for fair benchmarking ──

/// Initialize C FFT state for benchmarking (call once before bench loop).
pub fn c_fft_bench_init(nfft: usize) {
    unsafe { wrap_fft_bench_init(nfft as i32) }
}

/// Run C FFT using pre-initialized state (no per-call allocation).
pub fn c_fft_bench_run(fin_r: &[f32], fin_i: &[f32], fout_r: &mut [f32], fout_i: &mut [f32]) {
    unsafe {
        wrap_fft_bench_run(
            fin_r.as_ptr(),
            fin_i.as_ptr(),
            fout_r.as_mut_ptr(),
            fout_i.as_mut_ptr(),
        )
    }
}

/// Initialize C MDCT state for benchmarking (call once before bench loop).
pub fn c_mdct_bench_init(n: usize) {
    unsafe { wrap_mdct_bench_init(n as i32) }
}

/// Run C MDCT forward using pre-initialized state.
pub fn c_mdct_bench_forward(
    input: &mut [f32],
    output: &mut [f32],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    unsafe {
        wrap_mdct_bench_forward(
            input.as_mut_ptr(),
            output.as_mut_ptr(),
            overlap as i32,
            shift as i32,
            stride as i32,
        )
    }
}

/// Run C MDCT backward using pre-initialized state.
pub fn c_mdct_bench_backward(
    input: &mut [f32],
    output: &mut [f32],
    overlap: usize,
    shift: usize,
    stride: usize,
) {
    unsafe {
        wrap_mdct_bench_backward(
            input.as_mut_ptr(),
            output.as_mut_ptr(),
            overlap as i32,
            shift as i32,
            stride as i32,
        )
    }
}

// ============ DNN wrapper FFI declarations ============

unsafe extern "C" {
    pub fn wrap_compute_activation(output: *mut f32, input: *const f32, n: i32, activation: i32);
    pub fn wrap_compute_linear(
        out: *mut f32,
        weights: *const f32,
        bias: *const f32,
        nb_inputs: i32,
        nb_outputs: i32,
        input: *const f32,
    );
    pub fn wrap_compute_linear_int8(
        out: *mut f32,
        weights: *const i8,
        bias: *const f32,
        scale: *const f32,
        nb_inputs: i32,
        nb_outputs: i32,
        input: *const f32,
    );
    pub fn wrap_compute_generic_dense(
        output: *mut f32,
        input: *const f32,
        weights: *const f32,
        bias: *const f32,
        nb_inputs: i32,
        nb_outputs: i32,
        activation: i32,
    );
    pub fn wrap_compute_generic_gru(
        state: *mut f32,
        input_weights: *const f32,
        input_bias: *const f32,
        recurrent_weights: *const f32,
        recurrent_bias: *const f32,
        recurrent_diag: *const f32,
        nb_inputs: i32,
        nb_neurons: i32,
        input: *const f32,
    );
}

/// Safe wrapper: compare C compute_activation vs Rust.
pub fn c_compute_activation(output: &mut [f32], input: &[f32], activation: i32) {
    let n = input.len() as i32;
    unsafe {
        wrap_compute_activation(output.as_mut_ptr(), input.as_ptr(), n, activation);
    }
}

/// Safe wrapper: C compute_linear with float weights.
pub fn c_compute_linear(
    out: &mut [f32],
    weights: &[f32],
    bias: &[f32],
    nb_inputs: usize,
    nb_outputs: usize,
    input: &[f32],
) {
    unsafe {
        wrap_compute_linear(
            out.as_mut_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            nb_inputs as i32,
            nb_outputs as i32,
            input.as_ptr(),
        );
    }
}

/// Safe wrapper: C compute_linear with int8 quantized weights.
pub fn c_compute_linear_int8(
    out: &mut [f32],
    weights: &[i8],
    bias: &[f32],
    scale: &[f32],
    nb_inputs: usize,
    nb_outputs: usize,
    input: &[f32],
) {
    unsafe {
        wrap_compute_linear_int8(
            out.as_mut_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            scale.as_ptr(),
            nb_inputs as i32,
            nb_outputs as i32,
            input.as_ptr(),
        );
    }
}

/// Safe wrapper: C compute_generic_dense.
pub fn c_compute_generic_dense(
    output: &mut [f32],
    input: &[f32],
    weights: &[f32],
    bias: &[f32],
    nb_inputs: usize,
    nb_outputs: usize,
    activation: i32,
) {
    unsafe {
        wrap_compute_generic_dense(
            output.as_mut_ptr(),
            input.as_ptr(),
            weights.as_ptr(),
            bias.as_ptr(),
            nb_inputs as i32,
            nb_outputs as i32,
            activation,
        );
    }
}

/// Safe wrapper: C compute_generic_gru.
pub fn c_compute_generic_gru(
    state: &mut [f32],
    input_weights: &[f32],
    input_bias: &[f32],
    recurrent_weights: &[f32],
    recurrent_bias: &[f32],
    recurrent_diag: &[f32],
    nb_inputs: usize,
    nb_neurons: usize,
    input: &[f32],
) {
    unsafe {
        wrap_compute_generic_gru(
            state.as_mut_ptr(),
            input_weights.as_ptr(),
            input_bias.as_ptr(),
            recurrent_weights.as_ptr(),
            recurrent_bias.as_ptr(),
            recurrent_diag.as_ptr(),
            nb_inputs as i32,
            nb_neurons as i32,
            input.as_ptr(),
        );
    }
}

// ============ DNN activation + layer comparison FFI declarations ============

unsafe extern "C" {
    fn wrap_sparse_sgemv8x4(
        out: *mut f32,
        w: *const f32,
        idx: *const i32,
        rows: i32,
        x: *const f32,
    );
    fn wrap_dense_tanh_from_blob(
        blob: *const u8,
        blob_len: i32,
        bias_name: *const u8,
        weights_name: *const u8,
        output: *mut f32,
        input: *const f32,
        nb_inputs: i32,
        nb_outputs: i32,
    );
}

/// Sparse float sgemv8x4 via C.
pub fn c_sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    unsafe {
        wrap_sparse_sgemv8x4(
            out.as_mut_ptr(),
            w.as_ptr(),
            idx.as_ptr(),
            rows as i32,
            x.as_ptr(),
        );
    }
}

/// Compute dense+tanh from a weight blob via C.
pub fn c_dense_tanh_from_blob(
    blob: &[u8],
    bias_name: &str,
    weights_name: &str,
    output: &mut [f32],
    input: &[f32],
    nb_inputs: usize,
    nb_outputs: usize,
) {
    let bias_cstr = std::ffi::CString::new(bias_name).unwrap();
    let weights_cstr = std::ffi::CString::new(weights_name).unwrap();
    unsafe {
        wrap_dense_tanh_from_blob(
            blob.as_ptr(),
            blob.len() as i32,
            bias_cstr.as_ptr() as *const u8,
            weights_cstr.as_ptr() as *const u8,
            output.as_mut_ptr(),
            input.as_ptr(),
            nb_inputs as i32,
            nb_outputs as i32,
        );
    }
}

// ============ DNN model-level wrapper FFI declarations ============

unsafe extern "C" {
    fn wrap_pitchdnn_compute(
        blob: *const u8,
        blob_len: i32,
        if_features: *const f32,
        xcorr_features: *const f32,
    ) -> f32;
    fn wrap_pitchdnn_multi_step(
        blob: *const u8,
        blob_len: i32,
        if_features_seq: *const f32,
        xcorr_features_seq: *const f32,
        nb_if_features: i32,
        nb_xcorr_features: i32,
        n_steps: i32,
    ) -> f32;
    fn wrap_rdovae_enc_dense1_only(
        blob: *const u8,
        blob_len: i32,
        output: *mut f32,
        output_size: *mut i32,
        input: *const f32,
    ) -> i32;
    fn wrap_rdovae_encode_dframe(
        blob: *const u8,
        blob_len: i32,
        latents: *mut f32,
        initial_state: *mut f32,
        input: *const f32,
    ) -> i32;
    fn wrap_rdovae_decode_all(
        blob: *const u8,
        blob_len: i32,
        output: *mut f32,
        initial_state: *const f32,
        latents: *const f32,
        nb_latents: i32,
    ) -> i32;
    fn wrap_fargan_synthesize(
        blob: *const u8,
        blob_len: i32,
        pcm_out: *mut f32,
        cont_pcm: *const f32,
        cont_features: *const f32,
        features: *const f32,
    ) -> i32;
    fn wrap_fargan_synthesize_multi(
        blob: *const u8,
        blob_len: i32,
        pcm_out: *mut f32,
        cont_pcm: *const f32,
        cont_features: *const f32,
        features_seq: *const f32,
        nb_features_per_frame: i32,
        n_frames: i32,
    ) -> i32;
}

/// PitchDNN: single-step compute via C.
pub fn c_pitchdnn_compute(blob: &[u8], if_features: &[f32], xcorr_features: &[f32]) -> f32 {
    unsafe {
        wrap_pitchdnn_compute(
            blob.as_ptr(),
            blob.len() as i32,
            if_features.as_ptr(),
            xcorr_features.as_ptr(),
        )
    }
}

/// PitchDNN: multi-step compute via C (GRU state accumulation).
pub fn c_pitchdnn_multi_step(
    blob: &[u8],
    if_features_seq: &[f32],
    xcorr_features_seq: &[f32],
    nb_if_features: usize,
    nb_xcorr_features: usize,
    n_steps: usize,
) -> f32 {
    unsafe {
        wrap_pitchdnn_multi_step(
            blob.as_ptr(),
            blob.len() as i32,
            if_features_seq.as_ptr(),
            xcorr_features_seq.as_ptr(),
            nb_if_features as i32,
            nb_xcorr_features as i32,
            n_steps as i32,
        )
    }
}

/// RDOVAE: run only enc_dense1 + tanh through C. Returns output vector.
pub fn c_rdovae_enc_dense1(blob: &[u8], input: &[f32]) -> Result<Vec<f32>, i32> {
    let mut output = vec![0.0f32; 256];
    let mut size = 0i32;
    let ret = unsafe {
        wrap_rdovae_enc_dense1_only(
            blob.as_ptr(),
            blob.len() as i32,
            output.as_mut_ptr(),
            &mut size,
            input.as_ptr(),
        )
    };
    if ret == 0 {
        output.truncate(size as usize);
        Ok(output)
    } else {
        Err(ret)
    }
}

/// RDOVAE encode one frame via C. Returns (latents, initial_state) or error.
pub fn c_rdovae_encode_dframe(
    blob: &[u8],
    input: &[f32],
    latent_dim: usize,
    state_dim: usize,
) -> Result<(Vec<f32>, Vec<f32>), i32> {
    let mut latents = vec![0.0f32; latent_dim];
    let mut initial_state = vec![0.0f32; state_dim];
    let ret = unsafe {
        wrap_rdovae_encode_dframe(
            blob.as_ptr(),
            blob.len() as i32,
            latents.as_mut_ptr(),
            initial_state.as_mut_ptr(),
            input.as_ptr(),
        )
    };
    if ret == 0 {
        Ok((latents, initial_state))
    } else {
        Err(ret)
    }
}

/// RDOVAE decode all latent frames via C.
pub fn c_rdovae_decode_all(
    blob: &[u8],
    initial_state: &[f32],
    latents: &[f32],
    nb_latents: usize,
    output_dim: usize,
) -> Result<Vec<f32>, i32> {
    let mut output = vec![0.0f32; nb_latents * output_dim];
    let ret = unsafe {
        wrap_rdovae_decode_all(
            blob.as_ptr(),
            blob.len() as i32,
            output.as_mut_ptr(),
            initial_state.as_ptr(),
            latents.as_ptr(),
            nb_latents as i32,
        )
    };
    if ret == 0 { Ok(output) } else { Err(ret) }
}

/// FARGAN: single-frame synthesis via C (with continuity init).
pub fn c_fargan_synthesize(
    blob: &[u8],
    cont_pcm: &[f32],
    cont_features: &[f32],
    features: &[f32],
) -> Result<Vec<f32>, i32> {
    let mut pcm = vec![0.0f32; 160]; // FARGAN_FRAME_SIZE
    let ret = unsafe {
        wrap_fargan_synthesize(
            blob.as_ptr(),
            blob.len() as i32,
            pcm.as_mut_ptr(),
            cont_pcm.as_ptr(),
            cont_features.as_ptr(),
            features.as_ptr(),
        )
    };
    if ret == 0 { Ok(pcm) } else { Err(ret) }
}

/// FARGAN: multi-frame synthesis via C (with continuity init).
pub fn c_fargan_synthesize_multi(
    blob: &[u8],
    cont_pcm: &[f32],
    cont_features: &[f32],
    features_seq: &[f32],
    nb_features: usize,
    n_frames: usize,
) -> Result<Vec<f32>, i32> {
    let mut pcm = vec![0.0f32; n_frames * 160];
    let ret = unsafe {
        wrap_fargan_synthesize_multi(
            blob.as_ptr(),
            blob.len() as i32,
            pcm.as_mut_ptr(),
            cont_pcm.as_ptr(),
            cont_features.as_ptr(),
            features_seq.as_ptr(),
            nb_features as i32,
            n_frames as i32,
        )
    };
    if ret == 0 { Ok(pcm) } else { Err(ret) }
}
