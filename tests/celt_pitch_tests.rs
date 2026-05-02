//! Cross-validation tests for CELT pitch analysis functions.

mod common;

use common::{assert_f32_slice_close, gen_noise, gen_sine_vec};
use opus_rust::celt::pitch;
use opus_rust::celt::tables::WINDOW_120;
use opus_ffi::*;

// ── celt_pitch_xcorr ──

#[test]
fn pitch_xcorr_sine() {
    let len = 240;
    let max_pitch = 128;
    let x = gen_sine_vec(len, 440.0, 48000.0, 0.5);
    let y = gen_sine_vec(len + max_pitch, 440.0, 48000.0, 0.5);
    let mut rust_xcorr = vec![0.0f32; max_pitch];
    let mut c_xcorr = vec![0.0f32; max_pitch];
    pitch::celt_pitch_xcorr(&x, &y, &mut rust_xcorr, len, max_pitch);
    c_celt_pitch_xcorr(&x, &y, &mut c_xcorr, len, max_pitch);
    assert_f32_slice_close(&rust_xcorr, &c_xcorr, 1e-2, "pitch_xcorr(sine)");
}

#[test]
fn pitch_xcorr_noise() {
    let len = 120;
    let max_pitch = 64;
    let x = gen_noise(len, 42);
    let y = gen_noise(len + max_pitch, 99);
    let mut rust_xcorr = vec![0.0f32; max_pitch];
    let mut c_xcorr = vec![0.0f32; max_pitch];
    pitch::celt_pitch_xcorr(&x, &y, &mut rust_xcorr, len, max_pitch);
    c_celt_pitch_xcorr(&x, &y, &mut c_xcorr, len, max_pitch);
    assert_f32_slice_close(&rust_xcorr, &c_xcorr, 1e-2, "pitch_xcorr(noise)");
}

// ── pitch_downsample (mono) ──

#[test]
fn pitch_downsample_sine_mono() {
    let len = 240;
    let input_len = len * 2 + 1;
    let signal = gen_sine_vec(input_len, 440.0, 48000.0, 0.5);
    let mut rust_lp = vec![0.0f32; len];
    let mut c_signal = signal.clone();
    let mut c_lp = vec![0.0f32; len];
    pitch::pitch_downsample(&[&signal], &mut rust_lp, len, 1);
    c_pitch_downsample_mono(&mut c_signal, &mut c_lp, len);
    assert_f32_slice_close(&rust_lp, &c_lp, 1e-2, "pitch_downsample(sine, mono)");
}

#[test]
fn pitch_downsample_noise_mono() {
    let len = 120;
    let input_len = len * 2 + 1;
    let signal = gen_noise(input_len, 42);
    let mut rust_lp = vec![0.0f32; len];
    let mut c_signal = signal.clone();
    let mut c_lp = vec![0.0f32; len];
    pitch::pitch_downsample(&[&signal], &mut rust_lp, len, 1);
    c_pitch_downsample_mono(&mut c_signal, &mut c_lp, len);
    assert_f32_slice_close(&rust_lp, &c_lp, 1e-2, "pitch_downsample(noise, mono)");
}

// ── remove_doubling ──

#[test]
fn remove_doubling_periodic_signal() {
    let maxperiod = 1024;
    let n = 480;
    let total = maxperiod + n;
    let signal = gen_sine_vec(total, 110.0, 48000.0, 0.5);
    let mut rust_t0: usize = 200;
    let mut c_t0: i32 = 200;
    let mut c_signal = signal.clone();

    let rust_gain = pitch::remove_doubling(&signal, maxperiod, 30, n, &mut rust_t0, 200, 0.5);
    let c_gain = c_remove_doubling(&mut c_signal, maxperiod, 30, n, &mut c_t0, 200, 0.5);

    let diff = (rust_t0 as i32 - c_t0).abs();
    assert!(
        diff <= 2,
        "remove_doubling pitch: Rust T0={} C T0={} (diff={})",
        rust_t0,
        c_t0,
        diff
    );
    let gain_diff = (rust_gain - c_gain).abs();
    assert!(
        gain_diff < 0.2,
        "remove_doubling gain: Rust={:.4} C={:.4} diff={:.4}",
        rust_gain,
        c_gain,
        gain_diff
    );
}

// ── pitch_search ──

#[test]
fn pitch_search_periodic_signal() {
    // Generate a 2x-downsampled periodic signal (~200Hz → period ~120 at 24kHz)
    let max_pitch = 256;
    let len = 240;
    let lag = len + max_pitch;

    // x_lp: analysis window (downsampled)
    let x_lp = gen_sine_vec(len, 200.0, 24000.0, 0.5);
    // y: longer reference signal (same frequency, used for correlation)
    let y = gen_sine_vec(lag, 200.0, 24000.0, 0.5);

    let mut rust_pitch: usize = 0;
    pitch::pitch_search(&x_lp, &y, len, max_pitch, &mut rust_pitch);

    let mut c_y = y.clone();
    let c_pitch = c_pitch_search(&x_lp, &mut c_y, len, max_pitch);

    let diff = (rust_pitch as i32 - c_pitch).abs();
    assert!(
        diff <= 2,
        "pitch_search: Rust={} C={} (diff={})",
        rust_pitch,
        c_pitch,
        diff
    );
}

// ── comb_filter ──

#[test]
fn comb_filter_matches_c() {
    // Set up buffers with enough history for negative indexing.
    // The comb filter accesses x[offset - T1 - 2] through x[offset + N - 1 + 2].
    let t1 = 100usize;
    let n = 240;
    let history = t1 + 3; // T1 + 2 + 1 margin
    let buf_size = history + n + 3;
    let signal = gen_noise(buf_size, 42);
    let offset = history;

    // Rust: uses explicit (buf, offset) pairs
    let mut rust_out = vec![0.0f32; buf_size];
    pitch::comb_filter(
        &mut rust_out,
        offset,
        &signal,
        offset,
        t1,
        t1,
        n,
        0.0,
        0.5,
        0,
        0,
        &WINDOW_120,
        120,
    );

    // C: pass pointers starting at offset so C's negative indexing works
    let mut c_signal = signal.clone();
    let mut c_out = vec![0.0f32; buf_size];
    c_comb_filter(
        &mut c_out[offset..],
        &mut c_signal[offset..],
        t1 as i32,
        t1 as i32,
        n,
        0.0,
        0.5,
        0,
        0,
        120,
    );

    // Compare only the output region [0..N)
    assert_f32_slice_close(
        &rust_out[offset..offset + n],
        &c_out[offset..offset + n],
        1e-5,
        "comb_filter(g0=0, g1=0.5)",
    );
}

#[test]
fn comb_filter_crossfade() {
    // Test with different T0/T1 to exercise the crossfade overlap region
    let t0 = 80usize;
    let t1 = 100usize;
    let n = 240;
    let history = t1.max(t0) + 3;
    let buf_size = history + n + 3;
    let signal = gen_noise(buf_size, 77);
    let offset = history;

    let mut rust_out = vec![0.0f32; buf_size];
    pitch::comb_filter(
        &mut rust_out,
        offset,
        &signal,
        offset,
        t0,
        t1,
        n,
        0.3,
        0.6,
        0,
        1,
        &WINDOW_120,
        120,
    );

    let mut c_signal = signal.clone();
    let mut c_out = vec![0.0f32; buf_size];
    c_comb_filter(
        &mut c_out[offset..],
        &mut c_signal[offset..],
        t0 as i32,
        t1 as i32,
        n,
        0.3,
        0.6,
        0,
        1,
        120,
    );

    assert_f32_slice_close(
        &rust_out[offset..offset + n],
        &c_out[offset..offset + n],
        1e-4,
        "comb_filter(crossfade)",
    );
}
