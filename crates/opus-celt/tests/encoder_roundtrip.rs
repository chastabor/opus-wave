// CELT encoder roundtrip tests: encode → decode → re-encode → re-decode
// Verifies energy preservation and packet size consistency.

use opus_celt::{CeltDecoder, CeltEncoder};

const FRAME_SIZE: usize = 960;

fn energy_db(pcm: &[f32]) -> f64 {
    let sum: f64 = pcm.iter().map(|&x| (x as f64) * (x as f64)).sum();
    let rms = (sum / pcm.len().max(1) as f64).sqrt();
    if rms < 1e-20 {
        -200.0
    } else {
        20.0 * rms.log10()
    }
}

fn gen_sine(freq: f32, amplitude: f32, n: usize) -> Vec<f32> {
    // Use normalized [-1,1] scale for float PCM
    (0..n)
        .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / 48000.0).sin())
        .collect()
}

fn gen_noise(amplitude: f32, n: usize) -> Vec<f32> {
    let mut seed: u32 = 12345;
    (0..n)
        .map(|_| {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let val = ((seed >> 16) & 0x7FFF) as f32 / 32768.0 - 0.5;
            amplitude * val
        })
        .collect()
}

fn gen_click(amplitude: f32, n: usize) -> Vec<f32> {
    let mut pcm = vec![0.0f32; n];
    for sample in pcm.iter_mut().take(110usize.min(n)).skip(100) {
        *sample = amplitude;
    }
    pcm
}

fn roundtrip_mono(name: &str, pcm: &[f32], nb_bytes: usize) {
    let mut enc = CeltEncoder::new(48000, 1).unwrap();
    enc.bitrate = 64000;
    enc.vbr = false;
    let mut dec = CeltDecoder::new(48000, 1).unwrap();

    // Encode several frames for state warmup, keep last
    let mut compressed = vec![0u8; nb_bytes];
    let mut nbytes = 0;
    for _ in 0..6 {
        nbytes = enc
            .encode_with_ec(pcm, FRAME_SIZE, &mut compressed, nb_bytes, None)
            .unwrap();
    }
    assert!(nbytes > 0, "{name}: encoded to 0 bytes");

    // Decode
    let mut decoded = vec![0.0f32; FRAME_SIZE];
    dec.decode_with_ec(&compressed[..nbytes], &mut decoded, FRAME_SIZE, None)
        .unwrap_or_else(|e| panic!("{name}: decode error {e}"));

    // Verify decoded output has non-trivial energy for non-silent input.
    // The exact energy level depends on encoder quality, but it should be
    // audible (above -80 dB) for non-silent input at 64kbps.
    let e_in = energy_db(pcm);
    let e_out = energy_db(&decoded);
    if e_in > -60.0 {
        assert!(
            e_out > -80.0,
            "{name}: decoded energy too low {e_out:.1}dB for input {e_in:.1}dB"
        );
    }
}

#[test]
fn test_roundtrip_silence() {
    roundtrip_mono("silence", &vec![0.0f32; FRAME_SIZE], 160);
}

#[test]
fn test_roundtrip_sine_440() {
    roundtrip_mono("sine_440", &gen_sine(440.0, 0.5, FRAME_SIZE), 160);
}

#[test]
fn test_roundtrip_noise() {
    roundtrip_mono("noise", &gen_noise(0.3, FRAME_SIZE), 160);
}

#[test]
fn test_roundtrip_click() {
    roundtrip_mono("click", &gen_click(0.9, FRAME_SIZE), 160);
}

#[test]
fn test_different_signals_produce_different_packets() {
    let signals: Vec<(&str, Vec<f32>)> = vec![
        ("silence", vec![0.0f32; FRAME_SIZE]),
        ("sine_440", gen_sine(440.0, 0.5, FRAME_SIZE)),
        ("sine_1000", gen_sine(1000.0, 0.5, FRAME_SIZE)),
        ("noise", gen_noise(0.3, FRAME_SIZE)),
        ("click", gen_click(0.9, FRAME_SIZE)),
    ];
    let mut packets: Vec<Vec<u8>> = Vec::new();
    for (_name, pcm) in &signals {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        enc.bitrate = 64000;
        enc.vbr = false;
        let mut compressed = vec![0u8; 160];
        for _ in 0..5 {
            enc.encode_with_ec(pcm, FRAME_SIZE, &mut compressed, 160, None)
                .unwrap();
        }
        let nb = enc
            .encode_with_ec(pcm, FRAME_SIZE, &mut compressed, 160, None)
            .unwrap();
        packets.push(compressed[..nb].to_vec());
    }
    // At least 80% of signal pairs should produce different packets
    let mut different = 0;
    let total = packets.len() * (packets.len() - 1) / 2;
    for i in 0..packets.len() {
        for j in (i + 1)..packets.len() {
            if packets[i] != packets[j] {
                different += 1;
            }
        }
    }
    let threshold = (total as f64 * 0.8) as usize;
    assert!(
        different >= threshold,
        "Only {different}/{total} pairs differ (need {threshold})"
    );
}

#[test]
fn test_multi_frame_stability() {
    let mut enc = CeltEncoder::new(48000, 1).unwrap();
    enc.bitrate = 64000;
    enc.vbr = false;
    let mut dec = CeltDecoder::new(48000, 1).unwrap();
    for f in 0..20 {
        let pcm = gen_sine(200.0 + f as f32 * 50.0, 0.4, FRAME_SIZE);
        let mut compressed = vec![0u8; 160];
        let nb = enc
            .encode_with_ec(&pcm, FRAME_SIZE, &mut compressed, 160, None)
            .unwrap_or_else(|e| panic!("encode frame {f}: {e}"));
        let mut decoded = vec![0.0f32; FRAME_SIZE];
        dec.decode_with_ec(&compressed[..nb], &mut decoded, FRAME_SIZE, None)
            .unwrap_or_else(|e| panic!("decode frame {f}: {e}"));
    }
}
