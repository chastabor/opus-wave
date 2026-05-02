//! Integration tests for the float frame encoder (Layer 4).
//! Tests that silk_encode_frame_flp produces valid output for test signals.

use opus_rust::range_coder::EcCtx;
use opus_rust::silk::encoder_flp::encode_frame::silk_encode_frame_flp;
use opus_rust::silk::encoder_flp::lbrr::LbrrState;
use opus_rust::silk::nsq::NsqState;
use opus_rust::silk::*;

const FS_KHZ: i32 = 16;
const NB_SUBFR: i32 = 4;
const SUBFR_LENGTH: i32 = 80;
const FRAME_LENGTH: i32 = 320;
const LTP_MEM_LENGTH: i32 = 320; // 20ms * 16kHz
const LPC_ORDER: i32 = 16;
const SHAPING_LPC_ORDER: i32 = 16;
// C: shapeWinLength = SUB_FRAME_LENGTH_MS * fs_kHz + 2 * la_shape
const SHAPE_WIN_LENGTH: i32 = 5 * FS_KHZ + 2 * 5 * FS_KHZ; // 80 + 160 = 240
const LA_PITCH: i32 = 2 * FS_KHZ; // LA_PITCH_MS=2
const PITCH_LPC_WIN_LENGTH: i32 = 24 * FS_KHZ; // (20 + 2*2) * fs_kHz
const PITCH_EST_LPC_ORDER: i32 = 16;
const LA_SHAPE: usize = 5 * FS_KHZ as usize; // 80

// 10ms frame mode constants (2 subframes instead of 4)
const NB_SUBFR_10MS: i32 = 2;
const FRAME_LENGTH_10MS: i32 = NB_SUBFR_10MS * SUBFR_LENGTH; // 160
const PITCH_LPC_WIN_LENGTH_10MS: i32 = 14 * FS_KHZ; // (10 + 2*2) * fs_kHz = 224
// LTP_MEM_LENGTH, SUBFR_LENGTH, SHAPE_WIN_LENGTH, LA_PITCH, LPC_ORDER,
// SHAPING_LPC_ORDER, PITCH_EST_LPC_ORDER, LA_SHAPE are all unchanged for 10ms.

fn gen_sine_i16(len: usize, freq: f32, fs: f32, amplitude: f32) -> Vec<i16> {
    (0..len)
        .map(|i| {
            let s = amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin();
            (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16
        })
        .collect()
}

/// Test that the float frame encoder produces a valid packet for a sine wave.
#[test]
fn encode_frame_flp_produces_output() {
    let nlsf_cb = get_nlsf_cb(NlsfCbSel::Wb);
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;
    let mut prev_ltp_corr = 0.0f32;
    let mut sum_log_gain_q7 = 0i32;
    let mut frame_counter = 0i32;
    let input_quality_bands = [16384i32, 16384, 16384, 16384]; // moderate quality
    let snr_db_q7 = 2415; // ~19 dB, typical for 16kbps WB

    let max_packet = 1275;
    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];
    let mut lbrr = LbrrState::new();
    let mut lbrr_scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut lbrr_scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut lbrr_scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut lbrr_scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    let input = gen_sine_i16(FRAME_LENGTH as usize, 440.0, 16000.0, 0.5);
    let mut total_bytes = 0;
    for frame in 0..16 {
        let mut enc = EcCtx::enc_init(max_packet as u32);
        // Write VAD + LBRR flags (2 bits)
        enc.enc_bit_logp(true, 1);
        enc.enc_bit_logp(false, 1);

        let bytes = silk_encode_frame_flp(
            &mut x_buf,
            &mut nsq_state,
            &mut indices,
            &mut prev_nlsf_q15,
            &mut prev_signal_type,
            &mut prev_lag,
            &mut first_frame_after_reset,
            &mut last_gain_index,
            &mut prev_harm_smth,
            &mut prev_tilt_smth,
            &mut prev_ltp_corr,
            &mut sum_log_gain_q7,
            &mut frame_counter,
            255, // speech_activity_q8
            &input_quality_bands,
            0, // input_tilt_q15
            snr_db_q7,
            &input,
            FS_KHZ,
            NB_SUBFR,
            SUBFR_LENGTH,
            FRAME_LENGTH,
            LTP_MEM_LENGTH,
            LPC_ORDER,
            SHAPING_LPC_ORDER,
            SHAPE_WIN_LENGTH,
            LA_PITCH,
            PITCH_LPC_WIN_LENGTH,
            PITCH_EST_LPC_ORDER,
            0,  // warping_q16
            10, // complexity
            nlsf_cb,
            (max_packet - 1) * 8,
            0,              // packet_loss_perc
            1,              // n_frames_per_packet
            frame as usize, // n_frames_encoded
            &mut lbrr,
            &mut enc,
            &mut scratch_s_ltp_q15,
            &mut scratch_s_ltp,
            &mut scratch_x_sc_q10,
            &mut scratch_xq_tmp,
            &mut lbrr_scratch_s_ltp_q15,
            &mut lbrr_scratch_s_ltp,
            &mut lbrr_scratch_x_sc_q10,
            &mut lbrr_scratch_xq_tmp,
        );

        if frame == 15 {
            total_bytes = bytes;
            eprintln!(
                "[frame {}] bytes={} gain_idx={:?} interp={}",
                frame,
                bytes,
                &indices.gains_indices[..NB_SUBFR as usize],
                indices.nlsf_interp_coef_q2
            );
        }
    }

    eprintln!("Float frame encoder: {} bytes on frame 15", total_bytes);
    assert!(total_bytes > 0, "Frame encoder produced no output");
    assert!(
        total_bytes < 1275,
        "Frame encoder produced oversized output"
    );
}

/// Test that the encoder handles the first-frame-after-reset case.
#[test]
fn encode_frame_flp_first_frame() {
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;
    let mut prev_ltp_corr2 = 0.0f32;
    let mut sum_log_gain2 = 0i32;
    let mut frame_counter2 = 0i32;

    let input = gen_sine_i16(FRAME_LENGTH as usize, 440.0, 16000.0, 0.5);

    let mut enc = EcCtx::enc_init(1275);
    enc.enc_bit_logp(true, 1);
    enc.enc_bit_logp(false, 1);

    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];
    let mut lbrr2 = LbrrState::new();
    let mut lbrr2_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut lbrr2_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut lbrr2_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut lbrr2_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    let bytes = silk_encode_frame_flp(
        &mut x_buf,
        &mut nsq_state,
        &mut indices,
        &mut prev_nlsf_q15,
        &mut prev_signal_type,
        &mut prev_lag,
        &mut first_frame_after_reset,
        &mut last_gain_index,
        &mut prev_harm_smth,
        &mut prev_tilt_smth,
        &mut prev_ltp_corr2,
        &mut sum_log_gain2,
        &mut frame_counter2,
        255,
        &[16384; 4],
        0,
        2415,
        &input,
        FS_KHZ,
        NB_SUBFR,
        SUBFR_LENGTH,
        FRAME_LENGTH,
        LTP_MEM_LENGTH,
        LPC_ORDER,
        SHAPING_LPC_ORDER,
        SHAPE_WIN_LENGTH,
        LA_PITCH,
        PITCH_LPC_WIN_LENGTH,
        PITCH_EST_LPC_ORDER,
        0,
        10,
        get_nlsf_cb(NlsfCbSel::Wb),
        1275 * 8,
        0,
        1,
        0,
        &mut lbrr2,
        &mut enc,
        &mut scratch_s_ltp_q15,
        &mut scratch_s_ltp,
        &mut scratch_x_sc_q10,
        &mut scratch_xq_tmp,
        &mut lbrr2_s_ltp_q15,
        &mut lbrr2_s_ltp,
        &mut lbrr2_x_sc_q10,
        &mut lbrr2_xq_tmp,
    );

    eprintln!(
        "First frame: {} bytes, first_frame_after_reset now = {}",
        bytes, first_frame_after_reset
    );
    assert!(bytes > 0, "First frame produced no output");
    assert!(
        !first_frame_after_reset,
        "first_frame_after_reset should be cleared"
    );
}

/// Test that LBRR encoding produces valid redundancy data when enabled.
#[test]
fn encode_frame_flp_lbrr_enabled() {
    let nlsf_cb = get_nlsf_cb(NlsfCbSel::Wb);
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;
    let mut prev_ltp_corr3 = 0.0f32;
    let mut sum_log_gain3 = 0i32;
    let mut frame_counter3 = 0i32;
    let input_quality_bands = [16384i32; 4];
    let snr_db_q7 = 2415;

    let max_packet = 1275;
    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];
    let mut lbrr = LbrrState::new();
    lbrr.enabled = true;
    lbrr.gain_increases = 7;
    let mut lbrr_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH as usize];
    let mut lbrr_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH as usize];
    let mut lbrr_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut lbrr_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    // Encode 3 frames with LBRR enabled (max per packet)
    let input = gen_sine_i16(FRAME_LENGTH as usize, 440.0, 16000.0, 0.5);
    for frame in 0..3u32 {
        let mut enc = EcCtx::enc_init(max_packet as u32);
        enc.enc_bit_logp(true, 1);
        enc.enc_bit_logp(false, 1);

        if frame == 0 {
            lbrr.reset_for_packet(last_gain_index);
        }

        let bytes = silk_encode_frame_flp(
            &mut x_buf,
            &mut nsq_state,
            &mut indices,
            &mut prev_nlsf_q15,
            &mut prev_signal_type,
            &mut prev_lag,
            &mut first_frame_after_reset,
            &mut last_gain_index,
            &mut prev_harm_smth,
            &mut prev_tilt_smth,
            &mut prev_ltp_corr3,
            &mut sum_log_gain3,
            &mut frame_counter3,
            255,
            &input_quality_bands,
            0,
            snr_db_q7,
            &input,
            FS_KHZ,
            NB_SUBFR,
            SUBFR_LENGTH,
            FRAME_LENGTH,
            LTP_MEM_LENGTH,
            LPC_ORDER,
            SHAPING_LPC_ORDER,
            SHAPE_WIN_LENGTH,
            LA_PITCH,
            PITCH_LPC_WIN_LENGTH,
            PITCH_EST_LPC_ORDER,
            0,
            10,
            nlsf_cb,
            (max_packet - 1) * 8,
            5,
            1,
            frame as usize,
            &mut lbrr,
            &mut enc,
            &mut scratch_s_ltp_q15,
            &mut scratch_s_ltp,
            &mut scratch_x_sc_q10,
            &mut scratch_xq_tmp,
            &mut lbrr_s_ltp_q15,
            &mut lbrr_s_ltp,
            &mut lbrr_x_sc_q10,
            &mut lbrr_xq_tmp,
        );

        eprintln!(
            "[LBRR frame {}] bytes={} lbrr_flag={}",
            frame, bytes, lbrr.flags[frame as usize]
        );
    }

    // First frame is first_frame_after_reset → no pitch analysis → TYPE_UNVOICED
    // So LBRR won't fire (no voice activity detection). But subsequent frames should.
    // Check that at least one LBRR flag is set for the voiced frames (frames 1+)
    let lbrr_count: i32 = lbrr.flags.iter().sum();
    eprintln!("LBRR flags: {:?}, count={}", &lbrr.flags[..3], lbrr_count);

    assert!(
        lbrr_count >= 1,
        "LBRR should produce at least one redundancy frame"
    );

    for f in 0..3 {
        if lbrr.flags[f] == 1 {
            let pulse_sum: i32 = lbrr.pulses[f][..FRAME_LENGTH as usize]
                .iter()
                .map(|&p| p.abs() as i32)
                .sum();
            assert!(pulse_sum > 0, "LBRR frame {} has flag=1 but zero pulses", f);
            eprintln!("  LBRR frame {} pulse energy: {}", f, pulse_sum);
        }
    }
}

// =========================================================================
// 10ms frame mode tests (2-subframe)
// =========================================================================

/// Test that the float frame encoder produces a valid packet for a 10ms sine wave.
#[test]
fn encode_frame_flp_10ms_produces_output() {
    let nlsf_cb = get_nlsf_cb(NlsfCbSel::Wb);
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH_10MS as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;
    let mut prev_ltp_corr = 0.0f32;
    let mut sum_log_gain_q7 = 0i32;
    let mut frame_counter = 0i32;
    let input_quality_bands = [16384i32, 16384, 16384, 16384];
    let snr_db_q7 = 2415;

    let max_packet = 1275;
    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];
    let mut lbrr = LbrrState::new();
    let mut lbrr_scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut lbrr_scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut lbrr_scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut lbrr_scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    let input = gen_sine_i16(FRAME_LENGTH_10MS as usize, 440.0, 16000.0, 0.5);
    let mut total_bytes = 0;
    for frame in 0..16 {
        let mut enc = EcCtx::enc_init(max_packet as u32);
        enc.enc_bit_logp(true, 1);
        enc.enc_bit_logp(false, 1);

        let bytes = silk_encode_frame_flp(
            &mut x_buf,
            &mut nsq_state,
            &mut indices,
            &mut prev_nlsf_q15,
            &mut prev_signal_type,
            &mut prev_lag,
            &mut first_frame_after_reset,
            &mut last_gain_index,
            &mut prev_harm_smth,
            &mut prev_tilt_smth,
            &mut prev_ltp_corr,
            &mut sum_log_gain_q7,
            &mut frame_counter,
            255,
            &input_quality_bands,
            0,
            snr_db_q7,
            &input,
            FS_KHZ,
            NB_SUBFR_10MS,
            SUBFR_LENGTH,
            FRAME_LENGTH_10MS,
            LTP_MEM_LENGTH,
            LPC_ORDER,
            SHAPING_LPC_ORDER,
            SHAPE_WIN_LENGTH,
            LA_PITCH,
            PITCH_LPC_WIN_LENGTH_10MS,
            PITCH_EST_LPC_ORDER,
            0,
            10,
            nlsf_cb,
            (max_packet - 1) * 8,
            0,
            1,
            frame as usize,
            &mut lbrr,
            &mut enc,
            &mut scratch_s_ltp_q15,
            &mut scratch_s_ltp,
            &mut scratch_x_sc_q10,
            &mut scratch_xq_tmp,
            &mut lbrr_scratch_s_ltp_q15,
            &mut lbrr_scratch_s_ltp,
            &mut lbrr_scratch_x_sc_q10,
            &mut lbrr_scratch_xq_tmp,
        );

        if frame == 15 {
            total_bytes = bytes;
            eprintln!(
                "[10ms frame {}] bytes={} gain_idx={:?} interp={}",
                frame,
                bytes,
                &indices.gains_indices[..NB_SUBFR_10MS as usize],
                indices.nlsf_interp_coef_q2
            );
        }
    }

    eprintln!(
        "Float frame encoder (10ms): {} bytes on frame 15",
        total_bytes
    );
    assert!(total_bytes > 0, "10ms frame encoder produced no output");
    assert!(
        total_bytes < 1275,
        "10ms frame encoder produced oversized output"
    );
}

/// Test that the 10ms encoder handles the first-frame-after-reset case.
#[test]
fn encode_frame_flp_10ms_first_frame() {
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH_10MS as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;
    let mut prev_ltp_corr = 0.0f32;
    let mut sum_log_gain = 0i32;
    let mut frame_counter = 0i32;

    let input = gen_sine_i16(FRAME_LENGTH_10MS as usize, 440.0, 16000.0, 0.5);

    let mut enc = EcCtx::enc_init(1275);
    enc.enc_bit_logp(true, 1);
    enc.enc_bit_logp(false, 1);

    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];
    let mut lbrr = LbrrState::new();
    let mut lbrr_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut lbrr_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut lbrr_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut lbrr_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    let bytes = silk_encode_frame_flp(
        &mut x_buf,
        &mut nsq_state,
        &mut indices,
        &mut prev_nlsf_q15,
        &mut prev_signal_type,
        &mut prev_lag,
        &mut first_frame_after_reset,
        &mut last_gain_index,
        &mut prev_harm_smth,
        &mut prev_tilt_smth,
        &mut prev_ltp_corr,
        &mut sum_log_gain,
        &mut frame_counter,
        255,
        &[16384; 4],
        0,
        2415,
        &input,
        FS_KHZ,
        NB_SUBFR_10MS,
        SUBFR_LENGTH,
        FRAME_LENGTH_10MS,
        LTP_MEM_LENGTH,
        LPC_ORDER,
        SHAPING_LPC_ORDER,
        SHAPE_WIN_LENGTH,
        LA_PITCH,
        PITCH_LPC_WIN_LENGTH_10MS,
        PITCH_EST_LPC_ORDER,
        0,
        10,
        get_nlsf_cb(NlsfCbSel::Wb),
        1275 * 8,
        0,
        1,
        0,
        &mut lbrr,
        &mut enc,
        &mut scratch_s_ltp_q15,
        &mut scratch_s_ltp,
        &mut scratch_x_sc_q10,
        &mut scratch_xq_tmp,
        &mut lbrr_s_ltp_q15,
        &mut lbrr_s_ltp,
        &mut lbrr_x_sc_q10,
        &mut lbrr_xq_tmp,
    );

    eprintln!(
        "First 10ms frame: {} bytes, first_frame_after_reset now = {}",
        bytes, first_frame_after_reset
    );
    assert!(bytes > 0, "First 10ms frame produced no output");
    assert!(
        !first_frame_after_reset,
        "first_frame_after_reset should be cleared"
    );
}

/// Test that 10ms LBRR encoding produces valid redundancy data when enabled.
#[test]
fn encode_frame_flp_10ms_lbrr_enabled() {
    let nlsf_cb = get_nlsf_cb(NlsfCbSel::Wb);
    let ltp_mem = LTP_MEM_LENGTH as usize;
    let x_buf_len = ltp_mem + LA_SHAPE + FRAME_LENGTH_10MS as usize;
    let mut x_buf = vec![0.0f32; x_buf_len];
    let mut nsq_state = NsqState::new();
    let mut indices = SideInfoIndices::default();
    let mut prev_nlsf_q15 = [0i16; MAX_LPC_ORDER];
    let mut prev_signal_type = TYPE_NO_VOICE_ACTIVITY;
    let mut prev_lag = 0i32;
    let mut first_frame_after_reset = true;
    let mut last_gain_index = 10i8;
    let mut prev_harm_smth = 0.0f32;
    let mut prev_tilt_smth = 0.0f32;
    let mut prev_ltp_corr = 0.0f32;
    let mut sum_log_gain = 0i32;
    let mut frame_counter = 0i32;
    let input_quality_bands = [16384i32; 4];
    let snr_db_q7 = 2415;

    let max_packet = 1275;
    let mut scratch_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut scratch_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut scratch_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut scratch_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];
    let mut lbrr = LbrrState::new();
    lbrr.enabled = true;
    lbrr.gain_increases = 7;
    let mut lbrr_s_ltp_q15 = vec![0i32; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut lbrr_s_ltp = vec![0i16; ltp_mem + FRAME_LENGTH_10MS as usize];
    let mut lbrr_x_sc_q10 = vec![0i32; SUBFR_LENGTH as usize];
    let mut lbrr_xq_tmp = vec![0i16; SUBFR_LENGTH as usize];

    // Encode 3 frames with LBRR enabled
    let input = gen_sine_i16(FRAME_LENGTH_10MS as usize, 440.0, 16000.0, 0.5);
    for frame in 0..3u32 {
        let mut enc = EcCtx::enc_init(max_packet as u32);
        enc.enc_bit_logp(true, 1);
        enc.enc_bit_logp(false, 1);

        if frame == 0 {
            lbrr.reset_for_packet(last_gain_index);
        }

        let bytes = silk_encode_frame_flp(
            &mut x_buf,
            &mut nsq_state,
            &mut indices,
            &mut prev_nlsf_q15,
            &mut prev_signal_type,
            &mut prev_lag,
            &mut first_frame_after_reset,
            &mut last_gain_index,
            &mut prev_harm_smth,
            &mut prev_tilt_smth,
            &mut prev_ltp_corr,
            &mut sum_log_gain,
            &mut frame_counter,
            255,
            &input_quality_bands,
            0,
            snr_db_q7,
            &input,
            FS_KHZ,
            NB_SUBFR_10MS,
            SUBFR_LENGTH,
            FRAME_LENGTH_10MS,
            LTP_MEM_LENGTH,
            LPC_ORDER,
            SHAPING_LPC_ORDER,
            SHAPE_WIN_LENGTH,
            LA_PITCH,
            PITCH_LPC_WIN_LENGTH_10MS,
            PITCH_EST_LPC_ORDER,
            0,
            10,
            nlsf_cb,
            (max_packet - 1) * 8,
            5,
            1,
            frame as usize,
            &mut lbrr,
            &mut enc,
            &mut scratch_s_ltp_q15,
            &mut scratch_s_ltp,
            &mut scratch_x_sc_q10,
            &mut scratch_xq_tmp,
            &mut lbrr_s_ltp_q15,
            &mut lbrr_s_ltp,
            &mut lbrr_x_sc_q10,
            &mut lbrr_xq_tmp,
        );

        eprintln!(
            "[10ms LBRR frame {}] bytes={} lbrr_flag={}",
            frame, bytes, lbrr.flags[frame as usize]
        );
    }

    let lbrr_count: i32 = lbrr.flags.iter().sum();
    eprintln!(
        "10ms LBRR flags: {:?}, count={}",
        &lbrr.flags[..3],
        lbrr_count
    );

    assert!(
        lbrr_count >= 1,
        "10ms LBRR should produce at least one redundancy frame"
    );

    for f in 0..3 {
        if lbrr.flags[f] == 1 {
            let pulse_sum: i32 = lbrr.pulses[f][..FRAME_LENGTH_10MS as usize]
                .iter()
                .map(|&p| p.abs() as i32)
                .sum();
            assert!(
                pulse_sum > 0,
                "10ms LBRR frame {} has flag=1 but zero pulses",
                f
            );
            eprintln!("  10ms LBRR frame {} pulse energy: {}", f, pulse_sum);
        }
    }
}
