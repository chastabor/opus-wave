//! Test that Rust opus encode→decode roundtrip gain matches C reference.
//!
//! Encodes a 440Hz sine wave in CELT mode and verifies that the decoded RMS
//! level matches the C reference implementation within 6dB.

mod common;

use common::{gen_sine, rms};
use opus_wave::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};
use opus_ffi::{COpusDecoder, COpusEncoder};

const SAMPLE_RATE: i32 = 48000;
const FRAME_SIZE: i32 = 960; // 20ms at 48kHz
const MAX_PACKET: usize = 4000;
const BITRATE: i32 = 32000;
const N_WARMUP: usize = 10;

#[test]
fn gain_match_celt_mode() {
    // -- Set up encoders & decoders --
    // Use RESTRICTED_LOWDELAY to force CELT-only mode in both implementations
    let mut c_enc = COpusEncoder::new(
        SAMPLE_RATE,
        1,
        opus_ffi::OPUS_APPLICATION_RESTRICTED_LOWDELAY,
    )
    .unwrap();
    c_enc.set_bitrate(BITRATE).unwrap();

    let mut rust_enc = OpusEncoder::new(
        SampleRate::Hz48000,
        Channels::Mono,
        Application::RestrictedLowDelay,
    )
    .unwrap();
    rust_enc.set_bitrate(Bitrate::BitsPerSecond(BITRATE));

    let mut c_dec = COpusDecoder::new(SAMPLE_RATE, 1).unwrap();
    let mut rust_dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
    // Extra decoders for cross-validation
    let mut c_dec_for_rust = COpusDecoder::new(SAMPLE_RATE, 1).unwrap();
    let mut rust_dec_for_c = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();

    let mut pcm_in = vec![0.0f32; FRAME_SIZE as usize];
    let mut c_pkt = vec![0u8; MAX_PACKET];
    let mut rust_pkt = vec![0u8; MAX_PACKET];
    let mut c_c_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut rust_rust_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut c_rust_out = vec![0.0f32; FRAME_SIZE as usize]; // C dec on Rust enc
    let mut rust_c_out = vec![0.0f32; FRAME_SIZE as usize]; // Rust dec on C enc

    // -- Warm up & measure --
    let test_frame = N_WARMUP; // frame index to measure
    for frame in 0..=test_frame {
        gen_sine(&mut pcm_in, frame * FRAME_SIZE as usize, 440.0, 0.5);

        let c_len = c_enc.encode_float(&pcm_in, FRAME_SIZE, &mut c_pkt).unwrap();
        let rust_len = rust_enc
            .encode_float(&pcm_in, FRAME_SIZE, &mut rust_pkt, MAX_PACKET as i32)
            .unwrap();

        // C encode → C decode
        c_dec
            .decode_float(
                Some(&c_pkt[..c_len as usize]),
                &mut c_c_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        // Rust encode → Rust decode
        rust_dec
            .decode_float(
                Some(&rust_pkt[..rust_len as usize]),
                &mut rust_rust_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        // Rust encode → C decode
        c_dec_for_rust
            .decode_float(
                Some(&rust_pkt[..rust_len as usize]),
                &mut c_rust_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        // C encode → Rust decode
        rust_dec_for_c
            .decode_float(
                Some(&c_pkt[..c_len as usize]),
                &mut rust_c_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();

        if frame == test_frame {
            let in_rms = rms(&pcm_in);
            let cc_rms = rms(&c_c_out);
            let rr_rms = rms(&rust_rust_out);
            let cr_rms = rms(&c_rust_out); // C dec decoding Rust-encoded packet
            let rc_rms = rms(&rust_c_out); // Rust dec decoding C-encoded packet

            eprintln!("=== Gain analysis (frame {frame}, 440Hz @ 0.5 amp) ===");
            eprintln!("  Input RMS:                     {in_rms:.6}");
            eprintln!(
                "  C enc → C dec RMS:             {cc_rms:.6} (ratio {:.4})",
                cc_rms / in_rms
            );
            eprintln!(
                "  Rust enc → Rust dec RMS:       {rr_rms:.6} (ratio {:.4})",
                rr_rms / in_rms
            );
            eprintln!(
                "  Rust enc → C dec RMS:          {cr_rms:.6} (ratio {:.4})",
                cr_rms / in_rms
            );
            eprintln!(
                "  C enc → Rust dec RMS:          {rc_rms:.6} (ratio {:.4})",
                rc_rms / in_rms
            );
            let c_toc = c_pkt[0];
            let rust_toc = rust_pkt[0];
            eprintln!(
                "  C pkt: size={c_len}, TOC=0x{c_toc:02x} (config={})",
                (c_toc >> 3) & 0x1F
            );
            eprintln!(
                "  Rust pkt: size={rust_len}, TOC=0x{rust_toc:02x} (config={})",
                (rust_toc >> 3) & 0x1F
            );

            // The Rust roundtrip should produce output within 6dB of the C roundtrip
            // (gain ratio between 0.5x and 2.0x).
            let gain_ratio = rr_rms / cc_rms;
            eprintln!("  Rust/C gain ratio:             {gain_ratio:.4}");
            assert!(
                gain_ratio > 0.5 && gain_ratio < 2.0,
                "Rust roundtrip gain ({rr_rms:.6}) differs too much from C reference ({cc_rms:.6}): ratio {gain_ratio:.4}"
            );
        }
    }
}
