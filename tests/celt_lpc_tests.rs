//! Cross-validation tests for CELT LPC functions.

mod common;

use common::{assert_f32_slice_close, gen_noise, gen_sine_vec};
use opus_rust::celt::lpc;
use opus_ffi::*;

/// Compute autocorrelation of a signal for testing.
fn compute_autocorr(x: &[f32], order: usize) -> Vec<f32> {
    let n = x.len();
    let mut ac = vec![0.0f32; order + 1];
    for k in 0..=order {
        for i in k..n {
            ac[k] += x[i] * x[i - k];
        }
    }
    ac
}

// ── celt_lpc (Levinson-Durbin) ──

#[test]
fn celt_lpc_sine_order24() {
    let signal = gen_sine_vec(960, 440.0, 48000.0, 0.5);
    let ac = compute_autocorr(&signal, 24);
    let mut rust_lpc = vec![0.0f32; 24];
    let mut c_lpc = vec![0.0f32; 24];
    lpc::celt_lpc(&mut rust_lpc, &ac, 24);
    c_celt_lpc(&mut c_lpc, &ac, 24);
    assert_f32_slice_close(&rust_lpc, &c_lpc, 1e-4, "celt_lpc(sine, order=24)");
}

#[test]
fn celt_lpc_noise_order10() {
    let signal = gen_noise(480, 42);
    let ac = compute_autocorr(&signal, 10);
    let mut rust_lpc = vec![0.0f32; 10];
    let mut c_lpc = vec![0.0f32; 10];
    lpc::celt_lpc(&mut rust_lpc, &ac, 10);
    c_celt_lpc(&mut c_lpc, &ac, 10);
    assert_f32_slice_close(&rust_lpc, &c_lpc, 1e-4, "celt_lpc(noise, order=10)");
}

#[test]
fn celt_lpc_silence() {
    let ac = vec![0.0f32; 25];
    let mut rust_lpc = vec![1.0f32; 24];
    let mut c_lpc = vec![1.0f32; 24];
    lpc::celt_lpc(&mut rust_lpc, &ac, 24);
    c_celt_lpc(&mut c_lpc, &ac, 24);
    assert_f32_slice_close(&rust_lpc, &c_lpc, 1e-7, "celt_lpc(silence)");
}

// ── celt_fir ──

#[test]
fn celt_fir_sine_order4() {
    let x = gen_sine_vec(256, 440.0, 48000.0, 0.5);
    let coeffs = [0.5f32, -0.3, 0.1, -0.05];
    let mut rust_y = vec![0.0f32; 256];
    let mut c_y = vec![0.0f32; 256];
    lpc::celt_fir(&x, &coeffs, &mut rust_y, 256, 4);
    c_celt_fir(&x, &coeffs, &mut c_y, 256, 4);
    assert_f32_slice_close(&rust_y, &c_y, 1e-5, "celt_fir(sine, ord=4)");
}

#[test]
fn celt_fir_noise_order8() {
    let x = gen_noise(320, 77);
    let coeffs = gen_noise(8, 99);
    let mut rust_y = vec![0.0f32; 320];
    let mut c_y = vec![0.0f32; 320];
    lpc::celt_fir(&x, &coeffs, &mut rust_y, 320, 8);
    c_celt_fir(&x, &coeffs, &mut c_y, 320, 8);
    assert_f32_slice_close(&rust_y, &c_y, 1e-4, "celt_fir(noise, ord=8)");
}

// ── celt_fir5 ──
// celt_fir5 is the only FIR filter actually used in the Rust codec
// (src/celt/pitch.rs::pitch_downsample). Both impls start with zero memory
// state and apply the filter in-place.

#[test]
fn celt_fir5_sine() {
    let mut rust_x = gen_sine_vec(256, 440.0, 48000.0, 0.3);
    let mut c_x = rust_x.clone();
    let coeffs = [0.5f32, -0.3, 0.1, -0.05, 0.02];
    opus_rust::celt::pitch::celt_fir5(&mut rust_x, &coeffs, 256);
    c_celt_fir5(&mut c_x, &coeffs, 256);
    assert_f32_slice_close(&rust_x, &c_x, 1e-5, "celt_fir5(sine)");
}

#[test]
fn celt_fir5_noise() {
    let mut rust_x = gen_noise(320, 77);
    let mut c_x = rust_x.clone();
    let coeffs = [0.2f32, -0.15, 0.08, -0.04, 0.02];
    opus_rust::celt::pitch::celt_fir5(&mut rust_x, &coeffs, 320);
    c_celt_fir5(&mut c_x, &coeffs, 320);
    assert_f32_slice_close(&rust_x, &c_x, 1e-4, "celt_fir5(noise)");
}

// ── celt_iir ──

#[test]
fn celt_iir_sine_order4() {
    let x = gen_sine_vec(256, 440.0, 48000.0, 0.3);
    let den = [0.1f32, -0.05, 0.02, -0.01];
    let mut rust_y = vec![0.0f32; 256];
    let mut c_y = vec![0.0f32; 256];
    let mut rust_mem = vec![0.0f32; 4];
    let mut c_mem = vec![0.0f32; 4];
    lpc::celt_iir(&x, &den, &mut rust_y, 256, 4, &mut rust_mem);
    c_celt_iir(&x, &den, &mut c_y, 256, 4, &mut c_mem);
    assert_f32_slice_close(&rust_y, &c_y, 1e-4, "celt_iir(sine, ord=4) output");
    assert_f32_slice_close(&rust_mem, &c_mem, 1e-4, "celt_iir(sine, ord=4) memory");
}

#[test]
fn celt_iir_noise_order8() {
    let x = gen_noise(320, 42);
    let den = [0.05f32, -0.03, 0.02, -0.01, 0.005, -0.003, 0.001, -0.0005];
    let mut rust_y = vec![0.0f32; 320];
    let mut c_y = vec![0.0f32; 320];
    let mut rust_mem = vec![0.0f32; 8];
    let mut c_mem = vec![0.0f32; 8];
    lpc::celt_iir(&x, &den, &mut rust_y, 320, 8, &mut rust_mem);
    c_celt_iir(&x, &den, &mut c_y, 320, 8, &mut c_mem);
    assert_f32_slice_close(&rust_y, &c_y, 1e-3, "celt_iir(noise, ord=8) output");
    assert_f32_slice_close(&rust_mem, &c_mem, 1e-3, "celt_iir(noise, ord=8) memory");
}

// ── celt_autocorr ──

#[test]
fn celt_autocorr_sine() {
    let signal = gen_sine_vec(480, 440.0, 48000.0, 0.5);
    let lag = 24;
    let mut rust_ac = vec![0.0f32; lag + 1];
    let mut c_ac = vec![0.0f32; lag + 1];
    let empty_window: [f32; 0] = [];
    lpc::celt_autocorr(&signal, &mut rust_ac, &empty_window, 0, lag, signal.len());
    c_celt_autocorr(&signal, &mut c_ac, None, 0, lag, signal.len());
    assert_f32_slice_close(&rust_ac, &c_ac, 1e-1, "celt_autocorr(sine, lag=24)");
}

#[test]
fn celt_autocorr_noise() {
    let signal = gen_noise(256, 42);
    let lag = 10;
    let mut rust_ac = vec![0.0f32; lag + 1];
    let mut c_ac = vec![0.0f32; lag + 1];
    let empty_window: [f32; 0] = [];
    lpc::celt_autocorr(&signal, &mut rust_ac, &empty_window, 0, lag, signal.len());
    c_celt_autocorr(&signal, &mut c_ac, None, 0, lag, signal.len());
    assert_f32_slice_close(&rust_ac, &c_ac, 1e-1, "celt_autocorr(noise, lag=10)");
}
