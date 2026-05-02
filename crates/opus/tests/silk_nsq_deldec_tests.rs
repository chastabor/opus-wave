// SILK delayed-decision NSQ test vectors and integration tests.
// Tests different complexity levels which select scalar vs del-dec NSQ.

use opus::{Application, Bitrate, Channels, OpusDecoder, OpusEncoder, SampleRate};

// C reference packets at different complexities (16kHz, 16kbps, 200Hz tone)
const SILK_COMPLEXITY_C0: &[u8] = &[
    0x4b, 0x41, 0x1f, 0x06, 0xe3, 0x7d, 0x8e, 0xc8, 0x58, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

const SILK_COMPLEXITY_C8: &[u8] = &[
    0x4b, 0x41, 0x1e, 0x06, 0xe3, 0x79, 0xc8, 0xc9, 0x57, 0xc0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
];

/// Decode C reference packets at different complexities.
#[test]
fn test_decode_c_ref_complexity_0() {
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut pcm = vec![0.0f32; 320];
    let result = dec.decode_float(Some(SILK_COMPLEXITY_C0), &mut pcm, 320, false);
    assert!(
        result.is_ok(),
        "Should decode complexity-0 packet: {:?}",
        result
    );
}

#[test]
fn test_decode_c_ref_complexity_8() {
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
    let mut pcm = vec![0.0f32; 320];
    let result = dec.decode_float(Some(SILK_COMPLEXITY_C8), &mut pcm, 320, false);
    assert!(
        result.is_ok(),
        "Should decode complexity-8 packet: {:?}",
        result
    );
}

/// C reference complexity 0 and 8 produce different packets (scalar vs del-dec NSQ).
#[test]
fn test_c_ref_complexity_differs() {
    assert_ne!(
        SILK_COMPLEXITY_C0, SILK_COMPLEXITY_C8,
        "Different complexities should produce different packets"
    );
}

/// Rust encoder at different complexities produces decodable packets.
#[test]
fn test_rust_encoder_complexity_levels() {
    for complexity in [0, 2, 5, 8, 10] {
        let mut enc =
            OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(16000));
        enc.set_complexity(complexity);

        let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();
        let n = 320;

        for frame in 0..5 {
            let mut input = vec![0.0f32; n];
            for (i, sample) in input.iter_mut().enumerate() {
                *sample = 0.3
                    * (2.0 * std::f32::consts::PI * 200.0 * (frame * n + i) as f32 / 16000.0).sin();
            }

            let mut packet = vec![0u8; 1500];
            let nbytes = enc
                .encode_float(&input, n as i32, &mut packet, 1500)
                .unwrap() as usize;
            assert!(
                nbytes > 0,
                "Complexity {complexity} frame {frame}: should produce bytes"
            );

            let mut output = vec![0.0f32; n];
            let result = dec.decode_float(Some(&packet[..nbytes]), &mut output, n as i32, false);
            assert!(
                result.is_ok(),
                "Complexity {complexity} frame {frame}: decode failed: {:?}",
                result
            );
        }
    }
}

/// Higher complexity should produce different (potentially better) packets.
#[test]
fn test_complexity_produces_different_packets() {
    let n = 320;
    let mut input = vec![0.0f32; n];
    for (i, sample) in input.iter_mut().enumerate() {
        *sample = 0.3 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / 16000.0).sin();
    }

    let mut packets = Vec::new();
    for complexity in [0, 5, 10] {
        let mut enc =
            OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(16000));
        enc.set_complexity(complexity);

        let mut packet = vec![0u8; 1500];
        // Encode several frames for warmup
        for _ in 0..6 {
            enc.encode_float(&input, n as i32, &mut packet, 1500)
                .unwrap();
        }
        let nbytes = enc
            .encode_float(&input, n as i32, &mut packet, 1500)
            .unwrap() as usize;
        packets.push(packet[..nbytes].to_vec());
    }

    // At least one pair should differ (different NSQ paths)
    let mut any_differ = false;
    for i in 0..packets.len() {
        for j in (i + 1)..packets.len() {
            if packets[i] != packets[j] {
                any_differ = true;
            }
        }
    }
    assert!(
        any_differ,
        "Different complexities should produce at least some different packets"
    );
}

/// Encode-decode roundtrip stability at high complexity.
#[test]
fn test_high_complexity_roundtrip_stability() {
    let mut enc = OpusEncoder::new(SampleRate::Hz16000, Channels::Mono, Application::Voip).unwrap();
    enc.set_bitrate(Bitrate::BitsPerSecond(20000));
    enc.set_complexity(10);
    let mut dec = OpusDecoder::new(SampleRate::Hz16000, Channels::Mono).unwrap();

    for frame in 0..20 {
        let n = 320;
        let mut input = vec![0.0f32; n];
        let freq = 150.0 + frame as f32 * 30.0;
        for (i, sample) in input.iter_mut().enumerate() {
            *sample =
                0.4 * (2.0 * std::f32::consts::PI * freq * (frame * n + i) as f32 / 16000.0).sin();
        }

        let mut packet = vec![0u8; 1500];
        let nbytes = enc
            .encode_float(&input, n as i32, &mut packet, 1500)
            .unwrap() as usize;

        let mut output = vec![0.0f32; n];
        let result = dec.decode_float(Some(&packet[..nbytes]), &mut output, n as i32, false);
        assert!(
            result.is_ok(),
            "Frame {frame}: decode failed at complexity 10"
        );
    }
}
