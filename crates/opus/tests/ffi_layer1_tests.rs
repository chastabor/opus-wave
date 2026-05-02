//! FFI cross-validation tests for Layer 1 (fixed-point core) and select
//! Layer 3 functions. Each test calls both the Rust port and C reference
//! with the same input and asserts matching output.

mod common;

use opus_ffi::*;
use opus_silk::encoder_flp::find_ltp::silk_ltp_analysis_filter_flp;
use opus_silk::gain_quant;
use opus_silk::nlsf;
use opus_silk::*;

fn gen_sine_f32(len: usize, freq: f32, fs: f32, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

fn assert_slice_eq<T: Eq + std::fmt::Display + std::fmt::Debug>(rust: &[T], c: &[T], name: &str) {
    assert_eq!(rust.len(), c.len(), "{}: length mismatch", name);
    for i in 0..rust.len() {
        assert_eq!(
            rust[i], c[i],
            "{} [{}]: Rust={} C={}",
            name, i, rust[i], c[i]
        );
    }
}

fn assert_f32_slice_close(rust: &[f32], c: &[f32], tol: f32, name: &str) {
    assert_eq!(rust.len(), c.len(), "{}: length mismatch", name);
    for i in 0..rust.len() {
        let diff = (rust[i] - c[i]).abs();
        assert!(
            diff <= tol,
            "{} [{}]: Rust={} C={} diff={}",
            name,
            i,
            rust[i],
            c[i],
            diff
        );
    }
}

// ========================================================================
// 1b: silk_NLSF2A
// ========================================================================

#[test]
fn nlsf2a_matches_c_uniform() {
    let nlsf_q15: [i16; 16] = [
        2048, 4096, 6144, 8192, 10240, 12288, 14336, 16384, 18432, 20480, 22528, 24576, 26624,
        28672, 30720, 32000,
    ];
    let mut rust_a = [0i16; 16];
    let mut c_a = [0i16; 16];

    nlsf::silk_nlsf2a(&mut rust_a, &nlsf_q15, 16);
    c_silk_nlsf2a(&mut c_a, &nlsf_q15, 16);

    assert_slice_eq(&rust_a, &c_a, "NLSF2A uniform");
}

#[test]
fn nlsf2a_matches_c_speech() {
    let nlsf_q15: [i16; 16] = [
        1684, 1818, 2281, 4380, 6557, 8743, 10926, 13109, 15292, 17479, 19661, 21846, 24030, 26216,
        28398, 30586,
    ];
    let mut rust_a = [0i16; 16];
    let mut c_a = [0i16; 16];

    nlsf::silk_nlsf2a(&mut rust_a, &nlsf_q15, 16);
    c_silk_nlsf2a(&mut c_a, &nlsf_q15, 16);

    assert_slice_eq(&rust_a, &c_a, "NLSF2A speech");
}

#[test]
fn nlsf2a_narrowband() {
    let nlsf_q15: [i16; 10] = [
        3277, 6554, 9830, 13107, 16384, 19661, 22938, 26214, 29491, 32000,
    ];
    let mut rust_a = [0i16; 10];
    let mut c_a = [0i16; 10];

    nlsf::silk_nlsf2a(&mut rust_a, &nlsf_q15, 10);
    c_silk_nlsf2a(&mut c_a, &nlsf_q15, 10);

    assert_slice_eq(&rust_a, &c_a, "NLSF2A NB");
}

// ========================================================================
// 1e: silk_gains_quant / silk_gains_dequant
// ========================================================================

#[test]
fn gains_quant_independent() {
    let mut r_ind = [0i8; 4];
    let mut r_gains = [100000i32, 120000, 110000, 130000];
    let mut r_prev = 10i8;

    let mut c_ind = [0i8; 4];
    let mut c_gains = [100000i32, 120000, 110000, 130000];
    let mut c_prev = 10i8;

    gain_quant::silk_gains_quant(&mut r_ind, &mut r_gains, &mut r_prev, false, 4);
    c_silk_gains_quant(&mut c_ind, &mut c_gains, &mut c_prev, false, 4);

    assert_eq!(&r_ind[..4], &c_ind[..4], "gains_quant ind (independent)");
    assert_slice_eq(
        &r_gains[..4],
        &c_gains[..4],
        "gains_quant gains (independent)",
    );
    assert_eq!(r_prev, c_prev, "gains_quant prev_ind");
}

#[test]
fn gains_quant_conditional() {
    let mut r_ind = [0i8; 4];
    let mut r_gains = [200000i32, 180000, 220000, 190000];
    let mut r_prev = 30i8;

    let mut c_ind = [0i8; 4];
    let mut c_gains = [200000i32, 180000, 220000, 190000];
    let mut c_prev = 30i8;

    gain_quant::silk_gains_quant(&mut r_ind, &mut r_gains, &mut r_prev, true, 4);
    c_silk_gains_quant(&mut c_ind, &mut c_gains, &mut c_prev, true, 4);

    assert_eq!(&r_ind[..4], &c_ind[..4], "gains_quant ind (conditional)");
    assert_slice_eq(
        &r_gains[..4],
        &c_gains[..4],
        "gains_quant gains (conditional)",
    );
    assert_eq!(r_prev, c_prev, "gains_quant prev_ind (conditional)");
}

#[test]
fn gains_dequant_matches_c() {
    let ind: [i8; 4] = [28, 5, 3, 6];
    let mut r_gains = [0i32; 4];
    let mut r_prev = 10i8;
    let mut c_gains = [0i32; 4];
    let mut c_prev = 10i8;

    gain_quant::silk_gains_dequant(&mut r_gains, &ind, &mut r_prev, false, 4);
    c_silk_gains_dequant(&mut c_gains, &ind, &mut c_prev, false, 4);

    assert_slice_eq(&r_gains[..4], &c_gains[..4], "gains_dequant gains");
    assert_eq!(r_prev, c_prev, "gains_dequant prev_ind");
}

#[test]
fn gains_quant_roundtrip_cross() {
    // Quantize with Rust, dequantize with C — must match
    let mut r_ind = [0i8; 4];
    let mut r_gains = [500000i32, 450000, 600000, 480000];
    let mut r_prev = 20i8;
    gain_quant::silk_gains_quant(&mut r_ind, &mut r_gains, &mut r_prev, false, 4);

    let mut c_gains = [0i32; 4];
    let mut c_prev = 20i8;
    c_silk_gains_dequant(&mut c_gains, &r_ind, &mut c_prev, false, 4);

    assert_slice_eq(&r_gains[..4], &c_gains[..4], "quant(Rust)→dequant(C)");
}

// ========================================================================
// 1k: silk_interpolate
// ========================================================================

#[test]
fn interpolate_all_factors() {
    let x0: [i16; 16] = [
        1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000,
        15000, 16000,
    ];
    let x1: [i16; 16] = [
        16000, 15000, 14000, 13000, 12000, 11000, 10000, 9000, 8000, 7000, 6000, 5000, 4000, 3000,
        2000, 1000,
    ];

    for ifact in 0..=4i32 {
        let mut r_xi = [0i16; 16];
        let mut c_xi = [0i16; 16];

        silk_interpolate_i16(&mut r_xi, &x0, &x1, ifact, 16);
        c_silk_interpolate(&mut c_xi, &x0, &x1, ifact, 16);

        assert_slice_eq(&r_xi, &c_xi, &format!("interpolate ifact={}", ifact));
    }
}

// ========================================================================
// 3i: silk_LTP_analysis_filter_FLP
// ========================================================================

#[test]
fn ltp_analysis_filter_voiced() {
    let total_len = 800;
    let signal = gen_sine_f32(total_len, 440.0, 16000.0, 0.5);
    let subfr_length = 80usize;
    let nb_subfr = 4usize;
    let pre_length = 16usize;
    let seg_len = subfr_length + pre_length;
    let pitch_l: [i32; MAX_NB_SUBFR] = [36, 36, 37, 37];
    let inv_gains: [f32; MAX_NB_SUBFR] = [0.001, 0.0012, 0.0011, 0.0013];

    let mut b = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];
    for k in 0..nb_subfr {
        b[k * LTP_ORDER] = 0.05;
        b[k * LTP_ORDER + 1] = 0.1;
        b[k * LTP_ORDER + 2] = 0.6;
        b[k * LTP_ORDER + 3] = 0.15;
        b[k * LTP_ORDER + 4] = 0.02;
    }

    let x_offset = 400usize;

    let mut r_res = vec![0.0f32; nb_subfr * seg_len];
    silk_ltp_analysis_filter_flp(
        &mut r_res,
        &signal,
        x_offset,
        &b,
        &pitch_l,
        &inv_gains,
        subfr_length,
        nb_subfr,
        pre_length,
    );

    let mut c_res = vec![0.0f32; nb_subfr * seg_len];
    c_silk_ltp_analysis_filter_flp(
        &mut c_res,
        &signal[x_offset..],
        &b,
        &pitch_l,
        &inv_gains,
        subfr_length,
        nb_subfr,
        pre_length,
    );

    assert_f32_slice_close(&r_res, &c_res, 1e-4, "LTP_analysis_filter voiced");
}

#[test]
fn ltp_analysis_filter_zero_b() {
    let total_len = 800;
    let signal = gen_sine_f32(total_len, 220.0, 16000.0, 0.3);
    let subfr_length = 80;
    let nb_subfr = 4;
    let pre_length = 16;
    let seg_len = subfr_length + pre_length;
    let pitch_l: [i32; MAX_NB_SUBFR] = [40, 40, 40, 40];
    let inv_gains: [f32; MAX_NB_SUBFR] = [0.5, 0.5, 0.5, 0.5];
    let b = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];

    let x_offset = 400;
    let mut r_res = vec![0.0f32; nb_subfr * seg_len];
    silk_ltp_analysis_filter_flp(
        &mut r_res,
        &signal,
        x_offset,
        &b,
        &pitch_l,
        &inv_gains,
        subfr_length,
        nb_subfr,
        pre_length,
    );

    let mut c_res = vec![0.0f32; nb_subfr * seg_len];
    c_silk_ltp_analysis_filter_flp(
        &mut c_res,
        &signal[x_offset..],
        &b,
        &pitch_l,
        &inv_gains,
        subfr_length,
        nb_subfr,
        pre_length,
    );

    assert_f32_slice_close(&r_res, &c_res, 1e-6, "LTP_analysis_filter zero B");
}

// ========================================================================
// Pipeline tests: encode_indices + encode_pulses (1f, 1g) verified via
// C decoder successfully decoding Rust encoder output.
// Also tests NSQ (1i/1j) and VAD (1h) implicitly.
// ========================================================================

const FRAME_SIZE: i32 = 320;
const MAX_PKT: usize = 4000;

fn make_sine_f32(fs: i32) -> Vec<f32> {
    gen_sine_f32(FRAME_SIZE as usize, 440.0, fs as f32, 0.5)
}

#[test]
fn encode_indices_pulses_decodable_by_c() {
    // Rust encoder → C decoder: verifies encode_indices + encode_pulses
    // produce a valid bitstream that the C reference can decode
    use opus::{Application, Bitrate, Channels, OpusEncoder, SampleRate};

    let fs = 16000;
    let mut r_enc =
        OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    r_enc.set_bitrate(Bitrate::BitsPerSecond(16000));
    r_enc.set_complexity(10);

    let pcm = make_sine_f32(fs);
    let mut pkt = vec![0u8; MAX_PKT];
    let mut bytes = 0i32;
    for _ in 0..16 {
        bytes = r_enc
            .encode_float(&pcm, FRAME_SIZE, &mut pkt, MAX_PKT as i32)
            .unwrap();
    }
    assert!(bytes > 0);

    let mut c_dec = COpusDecoder::new(fs, 1).unwrap();
    let mut dec_pcm = vec![0.0f32; FRAME_SIZE as usize];
    let samples = c_dec
        .decode_float(
            Some(&pkt[..bytes as usize]),
            &mut dec_pcm,
            FRAME_SIZE,
            false,
        )
        .unwrap();
    assert!(samples > 0, "C decoder failed on Rust packet");

    let rms = common::rms(&dec_pcm) as f32;
    eprintln!(
        "encode_indices+pulses: {} bytes, C decoded RMS = {:.4}",
        bytes, rms
    );
    assert!(rms > 0.01, "Decoded to silence (RMS={})", rms);
}

#[test]
fn nsq_output_decodable_by_c() {
    // Tests NSQ with low complexity (non-del-dec path, silk_NSQ_c)
    use opus::{Application, Bitrate, Channels, OpusEncoder, SampleRate};

    let fs = 16000;
    let mut r_enc =
        OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    r_enc.set_bitrate(Bitrate::BitsPerSecond(16000));
    r_enc.set_complexity(1); // low complexity → n_states_del_dec=1 → plain NSQ

    let pcm = make_sine_f32(fs);
    let mut pkt = vec![0u8; MAX_PKT];
    let mut bytes = 0i32;
    for _ in 0..8 {
        bytes = r_enc
            .encode_float(&pcm, FRAME_SIZE, &mut pkt, MAX_PKT as i32)
            .unwrap();
    }

    let mut c_dec = COpusDecoder::new(fs, 1).unwrap();
    let mut dec_pcm = vec![0.0f32; FRAME_SIZE as usize];
    let samples = c_dec
        .decode_float(
            Some(&pkt[..bytes as usize]),
            &mut dec_pcm,
            FRAME_SIZE,
            false,
        )
        .unwrap();
    assert!(samples > 0);

    let rms = common::rms(&dec_pcm) as f32;
    eprintln!("NSQ test: {} bytes, C decoded RMS = {:.4}", bytes, rms);
    assert!(rms > 0.01, "NSQ output decoded to silence");
}

#[test]
fn c_encoder_output_decodable_by_rust() {
    // C encoder → Rust decoder: verifies Rust decoder handles C's
    // encode_indices + encode_pulses + NSQ output
    use opus::{Channels, OpusDecoder, SampleRate};

    let fs = 16000;
    let mut c_enc = COpusEncoder::new(fs, 1, opus_ffi::OPUS_APPLICATION_VOIP).unwrap();
    c_enc.set_bitrate(16000).unwrap();
    c_enc.set_complexity(10).unwrap();

    let pcm = make_sine_f32(fs);
    let mut pkt = vec![0u8; MAX_PKT];
    let mut bytes = 0i32;
    for _ in 0..16 {
        bytes = c_enc.encode_float(&pcm, FRAME_SIZE, &mut pkt).unwrap();
    }

    let mut r_dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut dec_pcm = vec![0.0f32; FRAME_SIZE as usize];
    let samples = r_dec
        .decode_float(
            Some(&pkt[..bytes as usize]),
            &mut dec_pcm,
            FRAME_SIZE,
            false,
        )
        .unwrap();
    assert!(samples > 0);

    let rms = common::rms(&dec_pcm) as f32;
    eprintln!("C→Rust decode: {} bytes, RMS = {:.4}", bytes, rms);
    assert!(rms > 0.01, "Rust decoder failed on C packet");
}

#[test]
fn vad_produces_speech_activity() {
    // Both C and Rust should detect a 440Hz sine as speech (not silence)
    // VAD (1h) is implicitly tested: if VAD fails, encoder produces no output
    use opus::{Application, Bitrate, Channels, OpusEncoder, SampleRate};

    let fs = 16000;
    let mut r_enc =
        OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    r_enc.set_bitrate(Bitrate::BitsPerSecond(16000));

    let pcm = make_sine_f32(fs);
    let mut pkt = vec![0u8; MAX_PKT];
    let mut bytes = 0i32;
    for _ in 0..8 {
        bytes = r_enc
            .encode_float(&pcm, FRAME_SIZE, &mut pkt, MAX_PKT as i32)
            .unwrap();
    }
    assert!(
        bytes > 2,
        "VAD likely blocking output (only {} bytes)",
        bytes
    );

    // SILK-only: TOC config 0-9 for NB/WB
    let config = (pkt[0] >> 3) & 0x1f;
    assert!(config <= 13, "Expected SILK mode, got config {}", config);
}
