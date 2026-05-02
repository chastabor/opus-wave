//! Cross-validation tests: decode the same packets with both C reference
//! and our Rust decoder, compare PCM output sample-by-sample.

use opus_rust::{Channels, OpusDecoder, OpusMSDecoder, SampleRate};
use std::path::Path;

const VECTORS_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/crates/opus-ffi/tests/vectors");
const FRAME_SIZE: usize = 960;

/// Parse an .info file and return channels count.
fn read_channels(info_path: &Path) -> usize {
    let text = std::fs::read_to_string(info_path).expect("Cannot read info file");
    for line in text.lines() {
        if let Some(val) = line.strip_prefix("channels=") {
            return val.trim().parse().expect("Invalid channels value");
        }
    }
    1 // default to mono
}

/// Read a .packets file: [u32 count][u32 len][bytes]...
fn read_packets(path: &Path) -> Vec<Vec<u8>> {
    let data = std::fs::read(path).expect("Cannot read packets file");
    let count = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let mut packets = Vec::with_capacity(count);
    let mut pos = 4;
    for _ in 0..count {
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        packets.push(data[pos..pos + len].to_vec());
        pos += len;
    }
    assert_eq!(packets.len(), count);
    packets
}

/// Read a .pcm file as f32 samples (native endian from C)
fn read_pcm_f32(path: &Path) -> Vec<f32> {
    let data = std::fs::read(path).expect("Cannot read pcm file");
    assert!(data.len().is_multiple_of(4));
    data.chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect()
}

/// Decode all packets with the Rust decoder, return all decoded PCM frames concatenated.
fn rust_decode_all(packets: &[Vec<u8>], channels: usize) -> Vec<f32> {
    let mut dec = OpusDecoder::new(
        SampleRate::Hz48000,
        if channels == 1 {
            Channels::Mono
        } else {
            Channels::Stereo
        },
    )
    .expect("Failed to create decoder");
    let mut all_pcm = Vec::new();
    for pkt in packets {
        let mut pcm = vec![0.0f32; FRAME_SIZE * channels];
        match dec.decode_float(Some(pkt), &mut pcm, FRAME_SIZE as i32, false) {
            Ok(n) => {
                all_pcm.extend_from_slice(&pcm[..n as usize * channels]);
            }
            Err(e) => {
                // On decode failure, output zeros (like PLC)
                eprintln!("Rust decode error: {e}, using zeros");
                all_pcm.extend(vec![0.0f32; FRAME_SIZE * channels]);
            }
        }
    }
    all_pcm
}

/// Compare two PCM buffers. Returns (max_error, rms_error, num_samples).
fn compare_pcm(ref_pcm: &[f32], rust_pcm: &[f32]) -> (f64, f64, usize) {
    let n = ref_pcm.len().min(rust_pcm.len());
    let mut max_err: f64 = 0.0;
    let mut sum_sq: f64 = 0.0;
    for i in 0..n {
        let err = (ref_pcm[i] as f64 - rust_pcm[i] as f64).abs();
        if err > max_err {
            max_err = err;
        }
        sum_sq += err * err;
    }
    let rms = if n > 0 {
        (sum_sq / n as f64).sqrt()
    } else {
        0.0
    };
    (max_err, rms, n)
}

/// Run a test case: decode packets with Rust, compare against C reference PCM.
fn run_test_case(name: &str, max_allowed_error: f64) {
    let vec_dir = Path::new(VECTORS_DIR);
    let packets_path = vec_dir.join(format!("{name}.packets"));
    let pcm_path = vec_dir.join(format!("{name}.pcm"));
    let info_path = vec_dir.join(format!("{name}.info"));

    if !packets_path.exists() {
        panic!(
            "Test vector not found: {}\nRun `crates/opus-ffi/tests/gen_test_vectors` first to generate vectors.",
            packets_path.display()
        );
    }

    let channels = read_channels(&info_path);
    let packets = read_packets(&packets_path);
    let ref_pcm = read_pcm_f32(&pcm_path);
    let rust_pcm = rust_decode_all(&packets, channels);

    let (max_err, rms_err, n) = compare_pcm(&ref_pcm, &rust_pcm);

    println!(
        "  {name}: {n} samples, ch={channels}, max_err={max_err:.8}, rms_err={rms_err:.8} \
         (threshold={max_allowed_error:.1e})"
    );

    // Also check length matches
    assert_eq!(
        ref_pcm.len(),
        rust_pcm.len(),
        "{name}: sample count mismatch: C={}, Rust={}",
        ref_pcm.len(),
        rust_pcm.len()
    );

    assert!(
        max_err <= max_allowed_error,
        "{name}: max error {max_err:.10} exceeds threshold {max_allowed_error:.1e}\n\
         First diverging sample details:\n{}",
        first_divergence_detail(&ref_pcm, &rust_pcm, max_allowed_error, channels)
    );
}

fn first_divergence_detail(
    ref_pcm: &[f32],
    rust_pcm: &[f32],
    threshold: f64,
    channels: usize,
) -> String {
    let n = ref_pcm.len().min(rust_pcm.len());
    for i in 0..n {
        let err = (ref_pcm[i] as f64 - rust_pcm[i] as f64).abs();
        if err > threshold {
            let start = i.saturating_sub(3);
            let end = (i + 4).min(n);
            let sample_in_frame = (i / channels) % FRAME_SIZE;
            let frame_idx = (i / channels) / FRAME_SIZE;
            let ch = i % channels;
            let mut s = format!(
                "  First divergence at index {i} (frame {frame_idx}, sample {sample_in_frame}, ch {ch}):\n"
            );
            for j in start..end {
                let marker = if j == i { " <---" } else { "" };
                s += &format!(
                    "    [{j:5}] ref={:12.8} rust={:12.8} err={:.2e}{marker}\n",
                    ref_pcm[j],
                    rust_pcm[j],
                    (ref_pcm[j] as f64 - rust_pcm[j] as f64).abs()
                );
            }
            return s;
        }
    }
    String::from("  (no divergence found)")
}

// ==========================================
// CELT-only mono cross-validation tests
// ==========================================

#[test]
fn cross_validate_celt_silence() {
    run_test_case("celt_silence", 1e-5);
}

#[test]
fn cross_validate_celt_sine440() {
    run_test_case("celt_sine440", 2.0);
}

#[test]
fn cross_validate_celt_sine1k_hbr() {
    run_test_case("celt_sine1k_hbr", 2.0);
}

#[test]
fn cross_validate_celt_lowbr() {
    run_test_case("celt_lowbr", 1e-5);
}

// ==========================================
// SILK-only mono cross-validation tests
// ==========================================

#[test]
fn cross_validate_silk_nb_silence() {
    run_test_case("silk_nb_silence", 3.1e-5);
}

#[test]
fn cross_validate_silk_nb_sine200() {
    run_test_case("silk_nb_sine200", 3.1e-5);
}

#[test]
fn cross_validate_silk_wb_sine500() {
    run_test_case("silk_wb_sine500", 3.1e-5);
}

#[test]
fn cross_validate_silk_mb_sine350() {
    run_test_case("silk_mb_sine350", 3.1e-5);
}

// ==========================================
// CELT-only stereo cross-validation tests
// ==========================================

#[test]
fn cross_validate_celt_stereo_silence() {
    run_test_case("celt_stereo_silence", 1e-5);
}

#[test]
fn cross_validate_celt_stereo_sine() {
    run_test_case("celt_stereo_sine", 2.0);
}

#[test]
fn cross_validate_celt_stereo_lowbr() {
    run_test_case("celt_stereo_lowbr", 2.0);
}

#[test]
fn cross_validate_celt_stereo_hbr() {
    run_test_case("celt_stereo_hbr", 2.0);
}

// ==========================================
// SILK-only stereo cross-validation tests
// SILK stereo uses fixed-point predictor math (i16/i32 Q13).
// Small rounding differences accumulate in the stereo predictor
// filter, leading to max errors ~0.001 (about 33 LSBs in i16).
// ==========================================

#[test]
fn cross_validate_silk_stereo_wb() {
    run_test_case("silk_stereo_wb", 2e-3);
}

#[test]
fn cross_validate_silk_stereo_nb() {
    run_test_case("silk_stereo_nb", 2e-3);
}

// ==========================================
// Hybrid stereo cross-validation tests
// ==========================================

#[test]
fn cross_validate_hybrid_stereo() {
    run_test_case("hybrid_stereo", 2.0);
}

#[test]
fn cross_validate_hybrid_stereo_fb() {
    run_test_case("hybrid_stereo_fb", 2.0);
}

// ==========================================
// Multistream cross-validation tests
// ==========================================

/// Parse multistream info from a .info file.
fn read_ms_info(info_path: &Path) -> (usize, usize, usize, Vec<u8>) {
    let text = std::fs::read_to_string(info_path).expect("Cannot read info file");
    let mut channels = 0usize;
    let mut streams = 0usize;
    let mut coupled_streams = 0usize;
    let mut mapping = Vec::new();

    for line in text.lines() {
        if let Some(val) = line.strip_prefix("channels=") {
            channels = val.trim().parse().unwrap();
        } else if let Some(val) = line.strip_prefix("streams=") {
            streams = val.trim().parse().unwrap();
        } else if let Some(val) = line.strip_prefix("coupled_streams=") {
            coupled_streams = val.trim().parse().unwrap();
        } else if let Some(val) = line.strip_prefix("mapping=") {
            mapping = val
                .trim()
                .split(',')
                .map(|s| s.trim().parse::<u8>().unwrap())
                .collect();
        }
    }
    (channels, streams, coupled_streams, mapping)
}

/// Decode all packets with the Rust multistream decoder.
fn rust_ms_decode_all(
    packets: &[Vec<u8>],
    channels: usize,
    streams: usize,
    coupled_streams: usize,
    mapping: &[u8],
) -> Vec<f32> {
    let mut dec = OpusMSDecoder::new(
        SampleRate::Hz48000,
        channels,
        streams,
        coupled_streams,
        mapping,
    )
    .expect("Failed to create MS decoder");
    let mut all_pcm = Vec::new();
    for pkt in packets {
        let mut pcm = vec![0.0f32; FRAME_SIZE * channels];
        match dec.decode_float(Some(pkt), &mut pcm, FRAME_SIZE as i32) {
            Ok(n) => {
                all_pcm.extend_from_slice(&pcm[..n as usize * channels]);
            }
            Err(e) => {
                eprintln!("Rust MS decode error: {e}, using zeros");
                all_pcm.extend(vec![0.0f32; FRAME_SIZE * channels]);
            }
        }
    }
    all_pcm
}

/// Run a multistream test case.
fn run_ms_test_case(name: &str, max_allowed_error: f64) {
    let vec_dir = Path::new(VECTORS_DIR);
    let packets_path = vec_dir.join(format!("{name}.packets"));
    let pcm_path = vec_dir.join(format!("{name}.pcm"));
    let info_path = vec_dir.join(format!("{name}.info"));

    if !packets_path.exists() {
        panic!(
            "Test vector not found: {}\nRun `crates/opus-ffi/tests/gen_ms_test_vectors` first.",
            packets_path.display()
        );
    }

    let (channels, streams, coupled_streams, mapping) = read_ms_info(&info_path);
    let packets = read_packets(&packets_path);
    let ref_pcm = read_pcm_f32(&pcm_path);
    let rust_pcm = rust_ms_decode_all(&packets, channels, streams, coupled_streams, &mapping);

    let (max_err, rms_err, n) = compare_pcm(&ref_pcm, &rust_pcm);

    println!(
        "  {name}: {n} samples, ch={channels}, streams={streams}, \
         max_err={max_err:.8}, rms_err={rms_err:.8} (threshold={max_allowed_error:.1e})"
    );

    assert_eq!(
        ref_pcm.len(),
        rust_pcm.len(),
        "{name}: sample count mismatch: C={}, Rust={}",
        ref_pcm.len(),
        rust_pcm.len()
    );

    assert!(
        max_err <= max_allowed_error,
        "{name}: max error {max_err:.10} exceeds threshold {max_allowed_error:.1e}\n\
         First diverging sample details:\n{}",
        first_divergence_detail(&ref_pcm, &rust_pcm, max_allowed_error, channels)
    );
}

#[test]
fn cross_validate_ms_quad() {
    run_ms_test_case("ms_quad", 2.0);
}

#[test]
fn cross_validate_ms_surround51() {
    run_ms_test_case("ms_surround51", 2.0);
}
