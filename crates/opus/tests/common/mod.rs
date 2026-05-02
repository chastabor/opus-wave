//! Shared test utilities for opus integration tests.
#![allow(dead_code)]

/// Sample rate used across tests.
pub const SAMPLE_RATE: i32 = 48000;

/// Generate a mono sine wave into `buf` at the given frequency and amplitude.
/// `offset` is the sample offset for phase continuity across frames.
pub fn gen_sine(buf: &mut [f32], offset: usize, freq: f32, amp: f32) {
    for (i, sample) in buf.iter_mut().enumerate() {
        *sample = amp
            * (2.0 * std::f32::consts::PI * freq * (i + offset) as f32 / SAMPLE_RATE as f32).sin();
    }
}

/// Generate a stereo sine wave with separate L/R frequencies.
pub fn gen_stereo_sine(
    buf: &mut [f32],
    samples: usize,
    offset: usize,
    freq_l: f32,
    freq_r: f32,
    amp: f32,
) {
    for i in 0..samples {
        let t = (i + offset) as f32 / SAMPLE_RATE as f32;
        buf[i * 2] = amp * (2.0 * std::f32::consts::PI * freq_l * t).sin();
        buf[i * 2 + 1] = amp * (2.0 * std::f32::consts::PI * freq_r * t).sin();
    }
}

/// Compute the RMS (root-mean-square) of a float buffer.
pub fn rms(buf: &[f32]) -> f64 {
    if buf.is_empty() {
        return 0.0;
    }
    let sum: f64 = buf.iter().map(|&x| (x as f64) * (x as f64)).sum();
    (sum / buf.len() as f64).sqrt()
}

/// Generate a sine wave as a new Vec. Frequency `freq` at sample rate `fs`.
pub fn gen_sine_vec(len: usize, freq: f32, fs: f32, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

/// Generate pseudo-random noise using LCG.
pub fn gen_noise(len: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

/// Assert two f32 values are close within tolerance.
pub fn assert_f32_close(rust: f32, c: f32, tol: f32, name: &str) {
    let diff = (rust - c).abs();
    assert!(
        diff <= tol,
        "{}: Rust={} C={} diff={} (tol={})",
        name,
        rust,
        c,
        diff,
        tol
    );
}

/// Compute the total energy (sum of squares) of a float buffer.
pub fn total_energy(buf: &[f32]) -> f64 {
    buf.iter().map(|&x| x as f64 * x as f64).sum()
}

/// Count the number of bytes that differ between two slices.
/// Bytes beyond the shorter slice's length count as differing.
pub fn count_differing_bytes(a: &[u8], b: &[u8]) -> usize {
    let min_len = a.len().min(b.len());
    let max_len = a.len().max(b.len());
    let mut count = max_len - min_len;
    for i in 0..min_len {
        if a[i] != b[i] {
            count += 1;
        }
    }
    count
}

/// Assert two f32 slices match within tolerance. Reports worst element.
pub fn assert_f32_slice_close(rust: &[f32], c: &[f32], tol: f32, name: &str) {
    assert_eq!(
        rust.len(),
        c.len(),
        "{}: length mismatch {} vs {}",
        name,
        rust.len(),
        c.len()
    );
    let mut max_diff = 0.0f32;
    let mut max_idx = 0;
    for i in 0..rust.len() {
        let diff = (rust[i] - c[i]).abs();
        if diff > max_diff {
            max_diff = diff;
            max_idx = i;
        }
    }
    assert!(
        max_diff <= tol,
        "{}: max diff={} at [{}] (Rust={} C={}), tol={}",
        name,
        max_diff,
        max_idx,
        rust[max_idx],
        c[max_idx],
        tol
    );
}

// ============ DNN test utilities ============

/// Simple LCG for reproducible DNN test data. Returns values in [-1.0, 1.0).
pub fn lcg_f32(seed: &mut u32) -> f32 {
    *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
    ((*seed >> 16) as f32 / 32768.0) - 1.0
}

/// Generate a reproducible random f32 vector using LCG.
pub fn gen_random_vec(n: usize, seed: &mut u32) -> Vec<f32> {
    (0..n).map(|_| lcg_f32(seed)).collect()
}

/// Load a DNN weight blob from the opus-dnn model-data/blobs directory.
/// Returns None if the file doesn't exist (e.g., weights not downloaded).
pub fn load_dnn_blob(name: &str) -> Option<Vec<u8>> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../opus-dnn/model-data/blobs")
        .join(name);
    std::fs::read(&path).ok()
}

/// Compute the max absolute difference between two slices, returning (max_diff, index).
pub fn max_abs_diff(a: &[f32], b: &[f32]) -> (f32, usize) {
    a.iter()
        .zip(b)
        .enumerate()
        .map(|(i, (x, y))| ((x - y).abs(), i))
        .fold(
            (0.0f32, 0),
            |(md, mi), (d, i)| if d > md { (d, i) } else { (md, mi) },
        )
}
