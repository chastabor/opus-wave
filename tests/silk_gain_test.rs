//! Test SILK mode encode→decode gain against C reference.
//! Uses 16kHz mono VOIP at 16kbps to force SILK-only mode.

mod common;

use common::rms;
use opus_rust::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};
use opus_ffi::{COpusDecoder, COpusEncoder};

const FRAME_SIZE: i32 = 320; // 20ms at 16kHz
const MAX_PACKET: usize = 4000;

#[test]
fn silk_16k_gain_analysis() {
    let fs = 16000i32;
    let bitrate = 16000i32;

    let mut c_enc = COpusEncoder::new(fs, 1, opus_ffi::OPUS_APPLICATION_VOIP).unwrap();
    c_enc.set_bitrate(bitrate).unwrap();

    let mut rust_enc =
        OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    rust_enc.set_bitrate(Bitrate::BitsPerSecond(bitrate));

    let mut c_dec = COpusDecoder::new(fs, 1).unwrap();
    let mut rust_dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut rust_dec_for_c = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut c_dec_for_rust = COpusDecoder::new(fs, 1).unwrap();

    let mut pcm_in = vec![0.0f32; FRAME_SIZE as usize];
    let mut c_pkt = vec![0u8; MAX_PACKET];
    let mut rust_pkt = vec![0u8; MAX_PACKET];
    let mut c_c_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut rust_rust_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut c_rust_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut rust_c_out = vec![0.0f32; FRAME_SIZE as usize];

    let n_warmup = 15;
    for frame in 0..=n_warmup {
        for (i, sample) in pcm_in.iter_mut().enumerate() {
            let t = (frame * FRAME_SIZE as usize + i) as f32 / fs as f32;
            *sample = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }

        let c_len = c_enc.encode_float(&pcm_in, FRAME_SIZE, &mut c_pkt).unwrap();
        let rust_len = rust_enc
            .encode_float(&pcm_in, FRAME_SIZE, &mut rust_pkt, MAX_PACKET as i32)
            .unwrap();

        c_dec
            .decode_float(
                Some(&c_pkt[..c_len as usize]),
                &mut c_c_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        rust_dec
            .decode_float(
                Some(&rust_pkt[..rust_len as usize]),
                &mut rust_rust_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        c_dec_for_rust
            .decode_float(
                Some(&rust_pkt[..rust_len as usize]),
                &mut c_rust_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();
        rust_dec_for_c
            .decode_float(
                Some(&c_pkt[..c_len as usize]),
                &mut rust_c_out,
                FRAME_SIZE,
                false,
            )
            .unwrap();

        if frame == n_warmup {
            let in_rms = rms(&pcm_in);
            let cc_rms = rms(&c_c_out);
            let rr_rms = rms(&rust_rust_out);
            let cr_rms = rms(&c_rust_out);
            let rc_rms = rms(&rust_c_out);

            let c_toc = c_pkt[0];
            let rust_toc = rust_pkt[0];

            eprintln!("=== SILK 16kbps gain analysis (frame {frame}) ===");
            eprintln!("  Input RMS:                {in_rms:.6}");
            eprintln!(
                "  C enc → C dec:            {cc_rms:.6} (ratio {:.4})",
                cc_rms / in_rms
            );
            eprintln!(
                "  Rust enc → Rust dec:      {rr_rms:.6} (ratio {:.4})",
                rr_rms / in_rms
            );
            eprintln!(
                "  Rust enc → C dec:         {cr_rms:.6} (ratio {:.4})",
                cr_rms / in_rms
            );
            eprintln!(
                "  C enc → Rust dec:         {rc_rms:.6} (ratio {:.4})",
                rc_rms / in_rms
            );
            eprintln!(
                "  C pkt: size={c_len}, TOC=0x{c_toc:02x} (config={})",
                (c_toc >> 3) & 0x1F
            );
            eprintln!(
                "  Rust pkt: size={rust_len}, TOC=0x{rust_toc:02x} (config={})",
                (rust_toc >> 3) & 0x1F
            );

            let gain_ratio = rr_rms / cc_rms.max(1e-10);
            eprintln!("  Rust/C gain ratio:        {gain_ratio:.4}");
        }
    }
}

/// Test that SILK roundtrip gain ratio is within acceptable range.
/// The Rust encoder should produce output within 3x of the input level.
#[test]
fn test_silk_roundtrip_gain_ratio() {
    let fs = 16000i32;
    let mut enc = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(16000));
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    let mut pcm = vec![0.0f32; FRAME_SIZE as usize];
    let mut pkt = vec![0u8; MAX_PACKET];
    let mut out = vec![0.0f32; FRAME_SIZE as usize];

    for f in 0..=15 {
        for (i, sample) in pcm.iter_mut().enumerate() {
            let t = (f * FRAME_SIZE as usize + i) as f32 / fs as f32;
            *sample = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }
        let len = enc
            .encode_float(&pcm, FRAME_SIZE, &mut pkt, MAX_PACKET as i32)
            .unwrap();
        dec.decode_float(Some(&pkt[..len as usize]), &mut out, FRAME_SIZE, false)
            .unwrap();
    }

    let in_rms = rms(&pcm);
    let out_rms = rms(&out);
    let ratio = out_rms / in_rms;
    assert!(
        ratio > 0.3 && ratio < 3.0,
        "SILK roundtrip gain ratio {ratio:.4} should be between 0.3 and 3.0"
    );
}

/// Test that SILK decoder correctly decodes C reference packets.
/// C enc → Rust dec should match C enc → C dec within 6dB.
#[test]
fn test_silk_decoder_matches_c_reference() {
    let fs = 16000i32;
    let mut c_enc = COpusEncoder::new(fs, 1, opus_ffi::OPUS_APPLICATION_VOIP).unwrap();
    c_enc.set_bitrate(16000).unwrap();
    let mut c_dec = COpusDecoder::new(fs, 1).unwrap();
    let mut rust_dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    let mut pcm = vec![0.0f32; FRAME_SIZE as usize];
    let mut pkt = vec![0u8; MAX_PACKET];
    let mut c_out = vec![0.0f32; FRAME_SIZE as usize];
    let mut rust_out = vec![0.0f32; FRAME_SIZE as usize];

    for f in 0..=15 {
        for (i, sample) in pcm.iter_mut().enumerate() {
            let t = (f * FRAME_SIZE as usize + i) as f32 / fs as f32;
            *sample = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }
        let len = c_enc.encode_float(&pcm, FRAME_SIZE, &mut pkt).unwrap();
        c_dec
            .decode_float(Some(&pkt[..len as usize]), &mut c_out, FRAME_SIZE, false)
            .unwrap();
        rust_dec
            .decode_float(Some(&pkt[..len as usize]), &mut rust_out, FRAME_SIZE, false)
            .unwrap();
    }

    let c_rms = rms(&c_out);
    let r_rms = rms(&rust_out);
    let ratio = r_rms / c_rms.max(1e-10);
    assert!(
        ratio > 0.5 && ratio < 2.0,
        "C enc → Rust dec gain ({r_rms:.6}) should match C enc → C dec ({c_rms:.6}), ratio={ratio:.4}"
    );
}

/// Test that SILK packet sizes are stable across frames.
/// After warmup, packet sizes shouldn't oscillate wildly.
#[test]
fn test_silk_packet_size_stability() {
    let fs = 16000i32;
    let mut enc = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(16000));

    let mut pcm = vec![0.0f32; FRAME_SIZE as usize];
    let mut pkt = vec![0u8; MAX_PACKET];
    let mut sizes = Vec::new();

    for f in 0..20 {
        for (i, sample) in pcm.iter_mut().enumerate() {
            let t = (f * FRAME_SIZE as usize + i) as f32 / fs as f32;
            *sample = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin();
        }
        let len = enc
            .encode_float(&pcm, FRAME_SIZE, &mut pkt, MAX_PACKET as i32)
            .unwrap();
        if f >= 5 {
            // Skip first 5 warmup frames
            sizes.push(len as usize);
        }
    }

    let min_sz = *sizes.iter().min().unwrap();
    let max_sz = *sizes.iter().max().unwrap();
    let ratio = max_sz as f64 / min_sz.max(1) as f64;
    assert!(
        ratio < 5.0,
        "Packet sizes should be stable after warmup: min={min_sz} max={max_sz} ratio={ratio:.1}"
    );
}
