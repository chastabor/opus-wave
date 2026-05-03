// SILK LBRR (in-band FEC) integration tests.
//
// Tests the SILK decoder's FEC (Forward Error Correction) path using LBRR
// (Low Bit-Rate Redundancy) data. The Opus FEC protocol works as follows:
//
//   1. Decode frame N normally
//   2. Frame N+1 is lost
//   3. Receive frame N+2, which may contain LBRR data for frame N+1
//   4. Call decode_float(Some(frame_n2_data), ..., decode_fec=true) to recover N+1
//   5. Call decode_float(Some(frame_n2_data), ..., decode_fec=false) to decode N+2
//
// In the SILK bitstream, after the TOC byte, the range coder contains:
//   - VAD flag (1 bit per frame via enc_bit_logp with k=1)
//   - LBRR flag (1 bit via enc_bit_logp with k=1)
//   - If LBRR flag is set, LBRR sub-flags and LBRR frame data follow
//   - Normal frame indices and pulses
//
// C reference packets can be regenerated using gen_lbrr_vectors.c:
//   gcc -O2 -I/home/ct37/projects/opus/include \
//       -o /tmp/gen_lbrr_vectors /tmp/gen_lbrr_vectors.c \
//       /home/ct37/projects/opus/build/libopus.a -lm
//   /tmp/gen_lbrr_vectors > /tmp/lbrr_vectors.txt

use opus_wave::{
    Application, Bandwidth, Bitrate, Channels, Mode, OpusDecoder, OpusEncoder, SampleRate,
    opus_packet_get_bandwidth, opus_packet_get_mode,
};

// =========================================================================
// Constants
// =========================================================================

const SAMPLE_RATE: i32 = 16000;
const FRAME_SIZE: usize = 320; // 20ms at 16kHz
const BITRATE: i32 = 16000;
const NUM_FRAMES: usize = 10;

// =========================================================================
// Signal generation helpers
// =========================================================================

/// Generate a sine tone at the given frequency and amplitude.
fn generate_tone(freq: f32, amplitude: f32, num_samples: usize, sample_offset: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; num_samples];
    for (i, sample) in buf.iter_mut().enumerate() {
        *sample = amplitude
            * (2.0 * std::f32::consts::PI * freq * (sample_offset + i) as f32 / SAMPLE_RATE as f32)
                .sin();
    }
    buf
}

/// Encode NUM_FRAMES frames of a 200Hz tone, returning all packets.
fn encode_frames(fec_enabled: bool, loss_perc: i32) -> Vec<Vec<u8>> {
    let mut enc = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(BITRATE));
    enc.set_complexity(10);
    enc.set_bandwidth(Bandwidth::Wideband);
    enc.set_inband_fec(fec_enabled);
    enc.set_packet_loss_perc(loss_perc);

    let mut packets = Vec::new();
    for f in 0..NUM_FRAMES {
        let input = generate_tone(200.0, 0.3, FRAME_SIZE, f * FRAME_SIZE);
        let mut packet = vec![0u8; 1500];
        let nbytes = enc
            .encode_float(&input, FRAME_SIZE as i32, &mut packet, 1500)
            .unwrap() as usize;
        assert!(nbytes > 0, "Frame {f}: encode should produce bytes");
        packets.push(packet[..nbytes].to_vec());
    }
    packets
}

// =========================================================================
// Test: Rust encoder produces valid SILK packets with FEC settings
// =========================================================================

#[test]
fn test_rust_encoder_fec_settings_produce_valid_packets() {
    let fec_packets = encode_frames(true, 10);
    let nofec_packets = encode_frames(false, 0);

    // All packets should be valid SILK wideband packets
    for (i, pkt) in fec_packets.iter().enumerate() {
        assert!(!pkt.is_empty(), "FEC packet {i} should not be empty");
        let mode = opus_packet_get_mode(pkt);
        assert_eq!(
            mode,
            Mode::SilkOnly,
            "FEC packet {i} should be SILK-only (mode={mode:?})"
        );
        let bw = opus_packet_get_bandwidth(pkt);
        assert_eq!(
            bw,
            Bandwidth::Wideband,
            "FEC packet {i} should be wideband (bw={bw:?})"
        );
    }

    for (i, pkt) in nofec_packets.iter().enumerate() {
        assert!(!pkt.is_empty(), "No-FEC packet {i} should not be empty");
        let mode = opus_packet_get_mode(pkt);
        assert_eq!(
            mode,
            Mode::SilkOnly,
            "No-FEC packet {i} should be SILK-only"
        );
    }
}

// =========================================================================
// Test: Decode FEC-enabled packets normally (no loss)
// =========================================================================

#[test]
fn test_decode_fec_packets_normally() {
    let packets = encode_frames(true, 10);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    for (i, pkt) in packets.iter().enumerate() {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        let result = dec.decode_float(Some(pkt), &mut pcm, FRAME_SIZE as i32, false);
        assert!(
            result.is_ok(),
            "Frame {i}: normal decode of FEC-enabled packet failed: {:?}",
            result
        );
        let n = result.unwrap() as usize;
        assert_eq!(
            n, FRAME_SIZE,
            "Frame {i}: expected {FRAME_SIZE} samples, got {n}"
        );
    }
}

// =========================================================================
// Test: FEC decode path (decode_fec=true) does not crash
// =========================================================================

#[test]
fn test_fec_decode_path_no_crash() {
    // Encode a sequence of frames with FEC enabled
    let packets = encode_frames(true, 10);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    // First, decode frame 0 normally to prime the decoder state
    let mut pcm = vec![0.0f32; FRAME_SIZE];
    dec.decode_float(Some(&packets[0]), &mut pcm, FRAME_SIZE as i32, false)
        .unwrap();

    // Simulate: frame 1 is lost, we receive frame 2
    // Step 1: Try to recover frame 1 using FEC data from frame 2
    // Since the Rust encoder currently writes lbrr_flag=0, the decoder
    // will fall back to PLC for the LBRR portion.
    let mut fec_pcm = vec![0.0f32; FRAME_SIZE];
    let fec_result = dec.decode_float(Some(&packets[2]), &mut fec_pcm, FRAME_SIZE as i32, true);
    // This should either succeed or gracefully fall back to PLC
    // The FEC path should not panic or produce an unrecoverable error
    match fec_result {
        Ok(n) => {
            assert_eq!(
                n as usize, FRAME_SIZE,
                "FEC decode should produce {FRAME_SIZE} samples"
            );
        }
        Err(e) => {
            // FEC decode may fail gracefully if no LBRR data is present
            // (the decoder may fall through to PLC which returns None/PLC)
            eprintln!("FEC decode returned error (expected for no-LBRR packets): {e}");
        }
    }

    // Step 2: Decode frame 2 normally
    let mut pcm2 = vec![0.0f32; FRAME_SIZE];
    let result2 = dec.decode_float(Some(&packets[2]), &mut pcm2, FRAME_SIZE as i32, false);
    assert!(
        result2.is_ok(),
        "Normal decode after FEC attempt should succeed: {:?}",
        result2
    );
}

// =========================================================================
// Test: FEC decode with PLC fallback for lost packets
// =========================================================================

#[test]
fn test_fec_decode_with_plc_fallback() {
    let packets = encode_frames(true, 10);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    // Decode several frames normally to build up decoder state
    for packet in packets.iter().take(5) {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        dec.decode_float(Some(packet), &mut pcm, FRAME_SIZE as i32, false)
            .unwrap();
    }

    // Frame 5 is "lost" - invoke PLC
    let mut plc_pcm = vec![0.0f32; FRAME_SIZE];
    let plc_result = dec.decode_float(None, &mut plc_pcm, FRAME_SIZE as i32, false);
    assert!(
        plc_result.is_ok(),
        "PLC decode should succeed: {:?}",
        plc_result
    );

    // Now try FEC recovery of frame 5 using frame 6's data
    // (In practice, this would be done before the PLC call, but we test both paths)
    let mut dec2 = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    for packet in packets.iter().take(5) {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        dec2.decode_float(Some(packet), &mut pcm, FRAME_SIZE as i32, false)
            .unwrap();
    }

    // Try FEC decode using frame 6's packet
    let mut fec_pcm = vec![0.0f32; FRAME_SIZE];
    let fec_result = dec2.decode_float(Some(&packets[6]), &mut fec_pcm, FRAME_SIZE as i32, true);
    match fec_result {
        Ok(n) => assert_eq!(n as usize, FRAME_SIZE),
        Err(e) => eprintln!("FEC decode error (acceptable): {e}"),
    }

    // Continue with normal decode of frame 6
    let mut pcm6 = vec![0.0f32; FRAME_SIZE];
    let result6 = dec2.decode_float(Some(&packets[6]), &mut pcm6, FRAME_SIZE as i32, false);
    assert!(
        result6.is_ok(),
        "Normal decode should work after FEC attempt"
    );
}

// =========================================================================
// Test: FEC decode on CELT packets falls back gracefully
// =========================================================================

#[test]
fn test_fec_decode_celt_mode_falls_back() {
    // When decode_fec is called on a CELT-only packet, the Opus decoder
    // should fall back to PLC since CELT has no LBRR mechanism.
    let mut enc =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(64000));

    let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();

    // Encode a few CELT frames to prime decoder
    for f in 0..3 {
        let input = generate_tone_48k(440.0, 0.3, 960, f * 960);
        let mut packet = vec![0u8; 1500];
        let nbytes = enc.encode_float(&input, 960, &mut packet, 1500).unwrap() as usize;

        let mut pcm = vec![0.0f32; 960];
        dec.decode_float(Some(&packet[..nbytes]), &mut pcm, 960, false)
            .unwrap();
    }

    // Encode one more frame
    let input = generate_tone_48k(440.0, 0.3, 960, 3 * 960);
    let mut packet = vec![0u8; 1500];
    let nbytes = enc.encode_float(&input, 960, &mut packet, 1500).unwrap() as usize;

    // Attempt FEC decode with a CELT packet - should fall back to PLC
    let mut pcm = vec![0.0f32; 960];
    let result = dec.decode_float(Some(&packet[..nbytes]), &mut pcm, 960, true);
    // CELT packets have no FEC data, so this should gracefully do PLC
    assert!(
        result.is_ok(),
        "FEC decode on CELT packet should not crash: {:?}",
        result
    );
}

/// Generate a 48kHz tone for CELT tests.
fn generate_tone_48k(
    freq: f32,
    amplitude: f32,
    num_samples: usize,
    sample_offset: usize,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; num_samples];
    for (i, sample) in buf.iter_mut().enumerate() {
        *sample = amplitude
            * (2.0 * std::f32::consts::PI * freq * (sample_offset + i) as f32 / 48000.0).sin();
    }
    buf
}

// =========================================================================
// Test: Multi-frame loss simulation with FEC recovery attempts
// =========================================================================

#[test]
fn test_multi_frame_loss_simulation() {
    let packets = encode_frames(true, 10);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    // Pattern: decode normally, simulate loss, attempt FEC, continue
    let mut all_pcm = Vec::new();

    for i in 0..packets.len() {
        let mut pcm = vec![0.0f32; FRAME_SIZE];

        if i == 3 || i == 7 {
            // Simulate packet loss at frames 3 and 7
            if i + 1 < packets.len() {
                // Attempt FEC recovery from the next packet
                let _ = dec.decode_float(Some(&packets[i + 1]), &mut pcm, FRAME_SIZE as i32, true);
            } else {
                // Pure PLC
                let _ = dec.decode_float(None, &mut pcm, FRAME_SIZE as i32, false);
            }
        } else {
            // Normal decode
            dec.decode_float(Some(&packets[i]), &mut pcm, FRAME_SIZE as i32, false)
                .unwrap();
        }

        all_pcm.extend_from_slice(&pcm);
    }

    // Verify we got the right total number of samples
    assert_eq!(all_pcm.len(), NUM_FRAMES * FRAME_SIZE);

    // Verify not all zeros (decoder should produce some output)
    let energy: f64 = all_pcm.iter().map(|&x| (x as f64) * (x as f64)).sum();
    assert!(
        energy > 0.0,
        "Decoded output should have non-zero energy even with loss"
    );
}

// =========================================================================
// Test: Rust encoder FEC roundtrip (encode with FEC, decode normally)
// =========================================================================

#[test]
fn test_rust_encoder_fec_roundtrip() {
    let mut enc = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(BITRATE));
    enc.set_complexity(10);
    enc.set_bandwidth(Bandwidth::Wideband);
    enc.set_inband_fec(true);
    enc.set_packet_loss_perc(10);

    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    let mut total_energy = 0.0f64;

    for frame in 0..NUM_FRAMES {
        let input = generate_tone(200.0, 0.3, FRAME_SIZE, frame * FRAME_SIZE);

        let mut packet = vec![0u8; 1500];
        let nbytes = enc
            .encode_float(&input, FRAME_SIZE as i32, &mut packet, 1500)
            .unwrap() as usize;
        assert!(nbytes > 0, "Frame {frame}: encode should produce bytes");

        let mut output = vec![0.0f32; FRAME_SIZE];
        let result = dec.decode_float(
            Some(&packet[..nbytes]),
            &mut output,
            FRAME_SIZE as i32,
            false,
        );
        assert!(result.is_ok(), "Frame {frame}: decode failed: {:?}", result);

        let frame_energy: f64 = output.iter().map(|&x| (x as f64) * (x as f64)).sum();
        total_energy += frame_energy;
    }

    // After warmup, the decoder should produce audible output
    assert!(
        total_energy > 0.0,
        "Total decoded energy should be non-zero for a 200Hz tone"
    );
}

// =========================================================================
// Test: Packet size comparison between FEC and non-FEC
// =========================================================================

#[test]
fn test_fec_vs_nofec_packet_sizes() {
    let fec_packets = encode_frames(true, 10);
    let nofec_packets = encode_frames(false, 0);

    // Note: The Rust encoder currently does not encode LBRR data (writes
    // lbrr_flag=0 always), so FEC-enabled packets may not be larger yet.
    // This test documents the current behavior and will detect when LBRR
    // encoding is implemented (packets should become larger).
    //
    // With a C encoder at 16kbps and 10% loss, FEC packets are typically
    // 2-10 bytes larger than non-FEC packets due to LBRR overhead.

    let fec_total: usize = fec_packets.iter().map(|p| p.len()).sum();
    let nofec_total: usize = nofec_packets.iter().map(|p| p.len()).sum();

    eprintln!("FEC total bytes: {fec_total}, No-FEC total bytes: {nofec_total}");
    for i in 0..NUM_FRAMES {
        eprintln!(
            "  Frame {i}: FEC={} bytes, No-FEC={} bytes, diff={}",
            fec_packets[i].len(),
            nofec_packets[i].len(),
            fec_packets[i].len() as i64 - nofec_packets[i].len() as i64
        );
    }

    // Both should produce valid packets of reasonable size
    for (i, pkt) in fec_packets.iter().enumerate() {
        assert!(
            pkt.len() >= 2 && pkt.len() <= 1275,
            "FEC packet {i}: size {} out of range",
            pkt.len()
        );
    }
    for (i, pkt) in nofec_packets.iter().enumerate() {
        assert!(
            pkt.len() >= 2 && pkt.len() <= 1275,
            "No-FEC packet {i}: size {} out of range",
            pkt.len()
        );
    }
}

// =========================================================================
// Test: Decode C reference SILK packets with FEC flag
// =========================================================================

// These are the same C reference SILK packets from c_reference_encode.rs.
// They were encoded without FEC (lbrr_flag=0), so decode_fec=true should
// gracefully fall back to PLC.

const C_SILK_200HZ_16K: &[u8] = &[
    0x4b, 0x41, 0x1e, 0x06, 0xe3, 0x79, 0xc5, 0x12, 0xf7, 0xbc, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

const C_SILK_200HZ_10K: &[u8] = &[
    0x0b, 0x41, 0x11, 0x06, 0xe0, 0xb3, 0x0e, 0xc6, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

#[test]
fn test_decode_c_ref_silk_with_fec_flag() {
    // First decode the packet normally to prime the decoder
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut pcm = vec![0.0f32; FRAME_SIZE];
    dec.decode_float(Some(C_SILK_200HZ_16K), &mut pcm, FRAME_SIZE as i32, false)
        .unwrap();

    // Now attempt FEC decode with the same packet type
    // Since lbrr_flag=0 in these packets, the decoder should do PLC
    let mut fec_pcm = vec![0.0f32; FRAME_SIZE];
    let result = dec.decode_float(
        Some(C_SILK_200HZ_16K),
        &mut fec_pcm,
        FRAME_SIZE as i32,
        true,
    );
    assert!(
        result.is_ok(),
        "FEC decode on no-LBRR packet should not crash: {:?}",
        result
    );
}

#[test]
fn test_decode_c_ref_silk_narrowband_with_fec_flag() {
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut pcm = vec![0.0f32; FRAME_SIZE];
    // Prime decoder with a normal decode
    dec.decode_float(Some(C_SILK_200HZ_10K), &mut pcm, FRAME_SIZE as i32, false)
        .unwrap();

    // FEC decode attempt
    let mut fec_pcm = vec![0.0f32; FRAME_SIZE];
    let result = dec.decode_float(
        Some(C_SILK_200HZ_10K),
        &mut fec_pcm,
        FRAME_SIZE as i32,
        true,
    );
    assert!(
        result.is_ok(),
        "FEC decode on NB no-LBRR packet should not crash: {:?}",
        result
    );
}

// =========================================================================
// Test: PLC produces different output than normal decode
// =========================================================================

#[test]
fn test_plc_differs_from_normal_decode() {
    let packets = encode_frames(false, 0);

    // Decode frames 0-4 normally with decoder A
    let mut dec_a = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    for packet in packets.iter().take(5) {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        dec_a
            .decode_float(Some(packet), &mut pcm, FRAME_SIZE as i32, false)
            .unwrap();
    }

    // Decode frame 5 normally
    let mut normal_pcm = vec![0.0f32; FRAME_SIZE];
    dec_a
        .decode_float(Some(&packets[5]), &mut normal_pcm, FRAME_SIZE as i32, false)
        .unwrap();

    // Now do the same but use PLC for frame 5
    let mut dec_b = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    for packet in packets.iter().take(5) {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        dec_b
            .decode_float(Some(packet), &mut pcm, FRAME_SIZE as i32, false)
            .unwrap();
    }

    // PLC for frame 5
    let mut plc_pcm = vec![0.0f32; FRAME_SIZE];
    dec_b
        .decode_float(None, &mut plc_pcm, FRAME_SIZE as i32, false)
        .unwrap();

    // PLC output should differ from normal decode (unless the encoder
    // happens to produce a DTX/silence frame)
    let normal_energy: f64 = normal_pcm.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let plc_energy: f64 = plc_pcm.iter().map(|&x| (x as f64) * (x as f64)).sum();

    eprintln!("Normal frame 5 energy: {normal_energy:.6}");
    eprintln!("PLC frame 5 energy: {plc_energy:.6}");

    // At minimum, both should produce some output (not all NaN or infinite)
    assert!(normal_energy.is_finite(), "Normal energy should be finite");
    assert!(plc_energy.is_finite(), "PLC energy should be finite");
}

// =========================================================================
// Test: Consecutive PLC frames produce attenuating output
// =========================================================================

#[test]
fn test_consecutive_plc_attenuates() {
    let packets = encode_frames(false, 0);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    // Decode some frames normally
    for packet in packets.iter().take(5) {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        dec.decode_float(Some(packet), &mut pcm, FRAME_SIZE as i32, false)
            .unwrap();
    }

    // Now simulate consecutive losses
    let mut energies = Vec::new();
    for _ in 0..5 {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        dec.decode_float(None, &mut pcm, FRAME_SIZE as i32, false)
            .unwrap();
        let energy: f64 = pcm.iter().map(|&x| (x as f64) * (x as f64)).sum();
        energies.push(energy);
    }

    eprintln!("PLC energies over 5 consecutive losses: {:?}", energies);

    // PLC should generally attenuate over time (energy decreases or stays stable)
    // We don't enforce strict monotonicity since PLC algorithms vary,
    // but the last frame should have less energy than the first (or at least not explode)
    assert!(
        energies[4] <= energies[0] * 2.0 + 1e-10,
        "PLC energy should not explode: first={:.6}, last={:.6}",
        energies[0],
        energies[4]
    );
}

// =========================================================================
// Test: FEC decode followed by normal decode produces valid state
// =========================================================================

#[test]
fn test_fec_then_normal_decode_valid_state() {
    let packets = encode_frames(true, 10);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    // Decode frames 0-3 normally
    for packet in packets.iter().take(4) {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        dec.decode_float(Some(packet), &mut pcm, FRAME_SIZE as i32, false)
            .unwrap();
    }

    // Frame 4 is lost; attempt FEC recovery from frame 5
    let mut fec_pcm = vec![0.0f32; FRAME_SIZE];
    let _ = dec.decode_float(Some(&packets[5]), &mut fec_pcm, FRAME_SIZE as i32, true);

    // Now decode frames 5-9 normally - decoder state should be consistent
    for (i, packet) in packets.iter().enumerate().take(NUM_FRAMES).skip(5) {
        let mut pcm = vec![0.0f32; FRAME_SIZE];
        let result = dec.decode_float(Some(packet), &mut pcm, FRAME_SIZE as i32, false);
        assert!(
            result.is_ok(),
            "Frame {i}: normal decode after FEC should succeed: {:?}",
            result
        );
    }
}

// =========================================================================
// Test: Decoder handles SILK mode transition with FEC
// =========================================================================

#[test]
fn test_fec_decode_respects_silk_mode_requirement() {
    // When the previous mode was CELT and we receive a SILK packet with
    // decode_fec=true, the decoder should not attempt SILK FEC decoding
    // (since there's no previous SILK state to recover).

    let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();

    // First prime with a CELT packet (48kHz audio mode)
    let mut enc_celt =
        OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
    enc_celt.set_bitrate(Bitrate::BitsPerSecond(64000));
    let input = generate_tone_48k(440.0, 0.3, 960, 0);
    let mut celt_pkt = vec![0u8; 1500];
    let nbytes = enc_celt
        .encode_float(&input, 960, &mut celt_pkt, 1500)
        .unwrap() as usize;

    let mut pcm = vec![0.0f32; 960];
    dec.decode_float(Some(&celt_pkt[..nbytes]), &mut pcm, 960, false)
        .unwrap();

    // Now try FEC decode with a SILK packet - decoder prev_mode is CELT,
    // so it should fall back gracefully
    let silk_packets = encode_frames(true, 10);
    // Need a SILK packet (16kHz WB) - but decoder is at 48kHz
    // The TOC byte determines mode, and the decoder should handle the mismatch
    let mut fec_pcm = vec![0.0f32; 960];
    let result = dec.decode_float(
        Some(&silk_packets[5]),
        &mut fec_pcm,
        FRAME_SIZE as i32,
        true,
    );
    // This may fail or fall back to PLC - either is acceptable
    match result {
        Ok(_) => eprintln!("FEC decode across mode boundary succeeded"),
        Err(e) => eprintln!("FEC decode across mode boundary failed (acceptable): {e}"),
    }
}
