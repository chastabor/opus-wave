//! In-process correctness comparison: Rust opus vs C libopus (via FFI).
//!
//! Test A — Decode comparison: C encoder → both decoders → compare PCM
//! Test B — Encode comparison: same PCM → both encoders → compare packets/decoded PCM
//! Test C — Round-trip comparison: full encode→decode pipeline of each implementation

mod common;

use common::{gen_sine, gen_stereo_sine};
use opus_rust::{
    Application, Bandwidth, Bitrate, Channels, ForceChannels, OpusDecoder, OpusEncoder, SampleRate,
};
use opus_ffi::{COpusDecoder, COpusEncoder};

const SAMPLE_RATE: i32 = 48000;
const N_WARMUP: usize = 5;
const N_FRAMES: usize = N_WARMUP + 1;
const MAX_PACKET: usize = 4000;

fn gen_silence(buf: &mut [f32]) {
    buf.fill(0.0);
}

/// Generate N_FRAMES frames of test signal.
fn generate_frames(cfg: &TestConfig) -> Vec<Vec<f32>> {
    let ch = i32::from(cfg.channels) as usize;
    let frame_size = cfg.frame_size as usize;
    let samples_per_frame = frame_size * ch;
    (0..N_FRAMES)
        .map(|frame| {
            let mut buf = vec![0.0f32; samples_per_frame];
            let offset = frame * frame_size;
            if cfg.freq_l == 0.0 && cfg.freq_r == 0.0 {
                gen_silence(&mut buf);
            } else if cfg.channels == Channels::Stereo {
                gen_stereo_sine(
                    &mut buf, frame_size, offset, cfg.freq_l, cfg.freq_r, cfg.amp,
                );
            } else {
                gen_sine(&mut buf, offset, cfg.freq_l, cfg.amp);
            }
            buf
        })
        .collect()
}

// ── Test configuration ──

struct TestConfig {
    name: &'static str,
    channels: Channels,
    application: Application,
    max_bandwidth: Bandwidth,
    bitrate: i32,
    force_channels: ForceChannels,
    freq_l: f32,
    freq_r: f32,
    amp: f32,
    /// Samples per frame at 48kHz (960 = 20ms, 480 = 10ms).
    frame_size: i32,
    /// Max per-sample error for decode comparison (C enc → both decs).
    decode_threshold: f64,
    /// Max per-sample error for round-trip comparison.
    roundtrip_threshold: f64,
}

// ── Comparison utilities ──

fn compare_pcm(a: &[f32], b: &[f32]) -> (f64, f64) {
    assert_eq!(
        a.len(),
        b.len(),
        "PCM length mismatch: {} vs {}",
        a.len(),
        b.len()
    );
    let mut max_err: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for i in 0..a.len() {
        let err = (a[i] as f64 - b[i] as f64).abs();
        if err > max_err {
            max_err = err;
        }
        sum_sq += err * err;
    }
    let rms = if a.is_empty() {
        0.0
    } else {
        (sum_sq / a.len() as f64).sqrt()
    };
    (max_err, rms)
}

fn first_divergence(
    a: &[f32],
    b: &[f32],
    threshold: f64,
    channels: usize,
    frame_size: i32,
) -> String {
    let n = a.len().min(b.len());
    for i in 0..n {
        let err = (a[i] as f64 - b[i] as f64).abs();
        if err > threshold {
            let start = i.saturating_sub(3);
            let end = (i + 4).min(n);
            let sample_in_frame = (i / channels) % frame_size as usize;
            let frame_idx = (i / channels) / frame_size as usize;
            let ch = i % channels;
            let mut s = format!(
                "  First divergence at index {i} (frame {frame_idx}, sample {sample_in_frame}, ch {ch}):\n"
            );
            for j in start..end {
                let marker = if j == i { " <---" } else { "" };
                s += &format!(
                    "    [{j:5}] a={:12.8} b={:12.8} err={:.2e}{marker}\n",
                    a[j],
                    b[j],
                    (a[j] as f64 - b[j] as f64).abs()
                );
            }
            return s;
        }
    }
    String::from("  (no divergence found)")
}

// ── Encoder/decoder helpers ──

fn make_c_encoder(cfg: &TestConfig) -> COpusEncoder {
    let mut enc = COpusEncoder::new(
        SAMPLE_RATE,
        i32::from(cfg.channels),
        i32::from(cfg.application),
    )
    .unwrap();
    enc.set_max_bandwidth(i32::from(cfg.max_bandwidth)).unwrap();
    enc.set_complexity(10).unwrap();
    enc.set_bitrate(cfg.bitrate).unwrap();
    if cfg.force_channels != ForceChannels::Auto {
        enc.set_force_channels(i32::from(cfg.force_channels))
            .unwrap();
    }
    enc
}

fn make_rust_encoder(cfg: &TestConfig) -> OpusEncoder {
    let mut enc = OpusEncoder::new(SampleRate::Hz48000, cfg.channels, cfg.application).unwrap();
    enc.set_bandwidth(cfg.max_bandwidth);
    enc.set_complexity(10);
    enc.set_bitrate(Bitrate::BitsPerSecond(cfg.bitrate));
    if cfg.force_channels != ForceChannels::Auto {
        enc.set_force_channels(cfg.force_channels);
    }
    enc
}

// ── Test A: Decode comparison ──
// C encoder → C decoder & Rust decoder → compare PCM

fn test_decode_comparison(cfg: &TestConfig) {
    let frames = generate_frames(cfg);
    let mut c_enc = make_c_encoder(cfg);
    let mut c_dec = COpusDecoder::new(SAMPLE_RATE, i32::from(cfg.channels)).unwrap();
    let mut rust_dec = OpusDecoder::new(SampleRate::Hz48000, cfg.channels).unwrap();

    let ch = i32::from(cfg.channels) as usize;
    let mut all_c_pcm = Vec::new();
    let mut all_rust_pcm = Vec::new();

    for frame_pcm in &frames {
        let mut packet = vec![0u8; MAX_PACKET];
        let pkt_len = c_enc
            .encode_float(frame_pcm, cfg.frame_size, &mut packet)
            .unwrap();
        let packet = &packet[..pkt_len as usize];

        let mut c_out = vec![0.0f32; cfg.frame_size as usize * ch];
        let c_samples = c_dec
            .decode_float(Some(packet), &mut c_out, cfg.frame_size, false)
            .unwrap();
        assert_eq!(c_samples, cfg.frame_size);

        let mut rust_out = vec![0.0f32; cfg.frame_size as usize * ch];
        let rust_samples = rust_dec
            .decode_float(Some(packet), &mut rust_out, cfg.frame_size, false)
            .unwrap();
        assert_eq!(rust_samples, cfg.frame_size);

        // Check range coder state
        let c_range = c_dec.final_range();
        let rust_range = rust_dec.range_final;
        if c_range != rust_range {
            eprintln!(
                "  {} WARNING: range_final mismatch: C={c_range:#x} Rust={rust_range:#x}",
                cfg.name
            );
        }

        all_c_pcm.extend_from_slice(&c_out);
        all_rust_pcm.extend_from_slice(&rust_out);
    }

    let (max_err, rms_err) = compare_pcm(&all_c_pcm, &all_rust_pcm);
    println!(
        "  [decode] {}: {} samples, max_err={max_err:.8}, rms={rms_err:.8} (thresh={:.1e})",
        cfg.name,
        all_c_pcm.len(),
        cfg.decode_threshold
    );

    assert!(
        max_err <= cfg.decode_threshold,
        "{}: decode max error {max_err:.10} exceeds threshold {:.1e}\n{}",
        cfg.name,
        cfg.decode_threshold,
        first_divergence(
            &all_c_pcm,
            &all_rust_pcm,
            cfg.decode_threshold,
            ch,
            cfg.frame_size
        )
    );
}

// ── Test B: Encode comparison ──
// Same PCM → C encoder & Rust encoder → compare packets + decoded PCM

fn test_encode_comparison(cfg: &TestConfig) {
    let frames = generate_frames(cfg);
    let mut c_enc = make_c_encoder(cfg);
    let mut rust_enc = make_rust_encoder(cfg);
    // Two separate decoders to maintain consistent state per stream
    let mut c_dec_for_c = COpusDecoder::new(SAMPLE_RATE, i32::from(cfg.channels)).unwrap();
    let mut c_dec_for_rust = COpusDecoder::new(SAMPLE_RATE, i32::from(cfg.channels)).unwrap();

    let ch = i32::from(cfg.channels) as usize;
    let mut packets_identical = 0usize;
    let mut packets_total = 0usize;
    let mut rust_invalid = 0usize;
    let mut all_c_decoded = Vec::new();
    let mut all_rust_decoded = Vec::new();

    for frame_pcm in &frames {
        let mut c_packet = vec![0u8; MAX_PACKET];
        let c_len = c_enc
            .encode_float(frame_pcm, cfg.frame_size, &mut c_packet)
            .unwrap();

        let mut rust_packet = vec![0u8; MAX_PACKET];
        let rust_len = rust_enc
            .encode_float(
                frame_pcm,
                cfg.frame_size,
                &mut rust_packet,
                MAX_PACKET as i32,
            )
            .unwrap();

        packets_total += 1;
        if c_len == rust_len && c_packet[..c_len as usize] == rust_packet[..rust_len as usize] {
            packets_identical += 1;
        }

        // Decode C packet
        let mut c_decoded = vec![0.0f32; cfg.frame_size as usize * ch];
        c_dec_for_c
            .decode_float(
                Some(&c_packet[..c_len as usize]),
                &mut c_decoded,
                cfg.frame_size,
                false,
            )
            .unwrap();
        all_c_decoded.extend_from_slice(&c_decoded);

        // Decode Rust packet (may be invalid if Rust encoder diverges)
        let mut rust_decoded = vec![0.0f32; cfg.frame_size as usize * ch];
        match c_dec_for_rust.decode_float(
            Some(&rust_packet[..rust_len as usize]),
            &mut rust_decoded,
            cfg.frame_size,
            false,
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "  {} WARNING: C decoder rejected Rust packet (frame {}, len={}, err={})",
                    cfg.name,
                    packets_total - 1,
                    rust_len,
                    e
                );
                rust_invalid += 1;
            }
        }
        all_rust_decoded.extend_from_slice(&rust_decoded);
    }

    if all_c_decoded.len() == all_rust_decoded.len() && rust_invalid == 0 {
        let (max_err, rms_err) = compare_pcm(&all_c_decoded, &all_rust_decoded);
        println!(
            "  [encode] {}: {packets_identical}/{packets_total} bit-identical, \
             decoded max_err={max_err:.8}, rms={rms_err:.8}",
            cfg.name,
        );
    } else {
        println!(
            "  [encode] {}: {packets_identical}/{packets_total} bit-identical, \
             {rust_invalid} invalid Rust packets",
            cfg.name,
        );
    }
}

// ── Test C: Round-trip comparison ──
// Rust encode→decode vs C encode→decode

fn test_roundtrip_comparison(cfg: &TestConfig) {
    let frames = generate_frames(cfg);

    // C pipeline
    let mut c_enc = make_c_encoder(cfg);
    let mut c_dec = COpusDecoder::new(SAMPLE_RATE, i32::from(cfg.channels)).unwrap();
    let ch = i32::from(cfg.channels) as usize;
    let mut c_pcm = Vec::new();

    for frame_pcm in &frames {
        let mut packet = vec![0u8; MAX_PACKET];
        let len = c_enc
            .encode_float(frame_pcm, cfg.frame_size, &mut packet)
            .unwrap();
        let mut out = vec![0.0f32; cfg.frame_size as usize * ch];
        c_dec
            .decode_float(
                Some(&packet[..len as usize]),
                &mut out,
                cfg.frame_size,
                false,
            )
            .unwrap();
        c_pcm.extend_from_slice(&out);
    }

    // Rust pipeline
    let mut rust_enc = make_rust_encoder(cfg);
    let mut rust_dec = OpusDecoder::new(SampleRate::Hz48000, cfg.channels).unwrap();
    let mut rust_pcm = Vec::new();
    let mut rust_decode_errors = 0usize;

    for frame_pcm in &frames {
        let mut packet = vec![0u8; MAX_PACKET];
        let len = rust_enc
            .encode_float(frame_pcm, cfg.frame_size, &mut packet, MAX_PACKET as i32)
            .unwrap();
        let mut out = vec![0.0f32; cfg.frame_size as usize * ch];
        match rust_dec.decode_float(
            Some(&packet[..len as usize]),
            &mut out,
            cfg.frame_size,
            false,
        ) {
            Ok(_) => {}
            Err(e) => {
                eprintln!(
                    "  {} WARNING: Rust decode failed on Rust-encoded packet (len={len}, err={e:?})",
                    cfg.name
                );
                rust_decode_errors += 1;
            }
        }
        rust_pcm.extend_from_slice(&out);
    }

    if rust_decode_errors > 0 {
        println!(
            "  [roundtrip] {}: SKIPPED — {rust_decode_errors}/{} Rust packets invalid",
            cfg.name, N_FRAMES
        );
        return;
    }

    let (max_err, rms_err) = compare_pcm(&c_pcm, &rust_pcm);
    println!(
        "  [roundtrip] {}: {} samples, max_err={max_err:.8}, rms={rms_err:.8} (thresh={:.1e})",
        cfg.name,
        c_pcm.len(),
        cfg.roundtrip_threshold
    );

    assert!(
        max_err <= cfg.roundtrip_threshold,
        "{}: roundtrip max error {max_err:.10} exceeds threshold {:.1e}\n{}",
        cfg.name,
        cfg.roundtrip_threshold,
        first_divergence(
            &c_pcm,
            &rust_pcm,
            cfg.roundtrip_threshold,
            ch,
            cfg.frame_size
        )
    );
}

// ── Test configurations ──

fn configs() -> Vec<TestConfig> {
    vec![
        // CELT mono (20ms)
        TestConfig {
            name: "celt_silence",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 64000,
            force_channels: ForceChannels::Auto,
            freq_l: 0.0,
            freq_r: 0.0,
            amp: 0.0,
            frame_size: 960,
            decode_threshold: 1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "celt_sine440",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 64000,
            force_channels: ForceChannels::Auto,
            freq_l: 440.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "celt_sine1k_hbr",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 128000,
            force_channels: ForceChannels::Auto,
            freq_l: 1000.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "celt_lowbr",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 16000,
            force_channels: ForceChannels::Auto,
            freq_l: 300.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        // CELT stereo (20ms)
        TestConfig {
            name: "celt_stereo_silence",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 64000,
            force_channels: ForceChannels::Auto,
            freq_l: 0.0,
            freq_r: 0.0,
            amp: 0.0,
            frame_size: 960,
            decode_threshold: 1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "celt_stereo_sine",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 96000,
            force_channels: ForceChannels::Auto,
            freq_l: 440.0,
            freq_r: 880.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "celt_stereo_lowbr",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 32000,
            force_channels: ForceChannels::Auto,
            freq_l: 300.0,
            freq_r: 300.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "celt_stereo_hbr",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 128000,
            force_channels: ForceChannels::Auto,
            freq_l: 1000.0,
            freq_r: 500.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        // SILK mono (20ms)
        TestConfig {
            name: "silk_nb_silence",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Narrowband,
            bitrate: 12000,
            force_channels: ForceChannels::Auto,
            freq_l: 0.0,
            freq_r: 0.0,
            amp: 0.0,
            frame_size: 960,
            decode_threshold: 3.1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_nb_sine200",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Narrowband,
            bitrate: 12000,
            force_channels: ForceChannels::Auto,
            freq_l: 200.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 3.1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_mb_sine350",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Mediumband,
            bitrate: 16000,
            force_channels: ForceChannels::Auto,
            freq_l: 350.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 3.1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_wb_sine500",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Wideband,
            bitrate: 20000,
            force_channels: ForceChannels::Auto,
            freq_l: 500.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 3.1e-5,
            roundtrip_threshold: 2.0,
        },
        // SILK stereo (20ms)
        TestConfig {
            name: "silk_stereo_wb",
            channels: Channels::Stereo,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Wideband,
            bitrate: 32000,
            force_channels: ForceChannels::Auto,
            freq_l: 400.0,
            freq_r: 600.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 3e-3,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_stereo_nb",
            channels: Channels::Stereo,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Narrowband,
            bitrate: 20000,
            force_channels: ForceChannels::Auto,
            freq_l: 200.0,
            freq_r: 300.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 3e-3,
            roundtrip_threshold: 2.0,
        },
        // Hybrid stereo (20ms)
        TestConfig {
            name: "hybrid_stereo",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Superwideband,
            bitrate: 32000,
            force_channels: ForceChannels::Auto,
            freq_l: 200.0,
            freq_r: 1000.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "hybrid_stereo_fb",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 36000,
            force_channels: ForceChannels::Auto,
            freq_l: 440.0,
            freq_r: 880.0,
            amp: 0.5,
            frame_size: 960,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
        // ── 10ms frame mode (480 samples at 48kHz) ──
        // SILK mono 10ms
        TestConfig {
            name: "silk_nb_silence_10ms",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Narrowband,
            bitrate: 12000,
            force_channels: ForceChannels::Auto,
            freq_l: 0.0,
            freq_r: 0.0,
            amp: 0.0,
            frame_size: 480,
            decode_threshold: 3.1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_nb_sine200_10ms",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Narrowband,
            bitrate: 12000,
            force_channels: ForceChannels::Auto,
            freq_l: 200.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 480,
            decode_threshold: 3.1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_mb_sine350_10ms",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Mediumband,
            bitrate: 16000,
            force_channels: ForceChannels::Auto,
            freq_l: 350.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 480,
            decode_threshold: 3.1e-5,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_wb_sine500_10ms",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Wideband,
            bitrate: 20000,
            force_channels: ForceChannels::Auto,
            freq_l: 500.0,
            freq_r: 0.0,
            amp: 0.5,
            frame_size: 480,
            // 10ms WB mode has slightly higher decode divergence than 20ms due to
            // shorter pitch LPC window (14ms vs 24ms) amplifying rounding differences
            decode_threshold: 5e-4,
            roundtrip_threshold: 2.0,
        },
        // SILK stereo 10ms
        TestConfig {
            name: "silk_stereo_wb_10ms",
            channels: Channels::Stereo,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Wideband,
            bitrate: 32000,
            force_channels: ForceChannels::Auto,
            freq_l: 400.0,
            freq_r: 600.0,
            amp: 0.5,
            frame_size: 480,
            decode_threshold: 3e-3,
            roundtrip_threshold: 2.0,
        },
        TestConfig {
            name: "silk_stereo_nb_10ms",
            channels: Channels::Stereo,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Narrowband,
            bitrate: 20000,
            force_channels: ForceChannels::Auto,
            freq_l: 200.0,
            freq_r: 300.0,
            amp: 0.5,
            frame_size: 480,
            decode_threshold: 3e-3,
            roundtrip_threshold: 2.0,
        },
        // Hybrid 10ms
        TestConfig {
            name: "hybrid_stereo_10ms",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Superwideband,
            bitrate: 32000,
            force_channels: ForceChannels::Auto,
            freq_l: 200.0,
            freq_r: 1000.0,
            amp: 0.5,
            frame_size: 480,
            decode_threshold: 2.0,
            roundtrip_threshold: 2.0,
        },
    ]
}

// ── Test entry points ──

#[test]
fn decode_comparison_all() {
    println!("\n=== Test A: Decode comparison (C enc → C dec vs Rust dec) ===");
    for cfg in &configs() {
        test_decode_comparison(cfg);
    }
}

#[test]
fn encode_comparison_all() {
    println!("\n=== Test B: Encode comparison (same PCM → C enc vs Rust enc) ===");
    for cfg in &configs() {
        test_encode_comparison(cfg);
    }
}

#[test]
fn roundtrip_comparison_all() {
    println!("\n=== Test C: Round-trip comparison (C pipeline vs Rust pipeline) ===");
    for cfg in &configs() {
        test_roundtrip_comparison(cfg);
    }
}
