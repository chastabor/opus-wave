//! Cross-validation tests for CELT FFT.

mod common;

use common::assert_f32_close;
use opus_wave::celt::fft::{KissFftCpx, KissFftState, opus_fft};
use opus_ffi::*;

/// Run FFT via C wrapper and return (real, imag) output arrays.
fn c_fft(nfft: usize, fin: &[KissFftCpx]) -> (Vec<f32>, Vec<f32>) {
    let fin_r: Vec<f32> = fin.iter().map(|c| c.r).collect();
    let fin_i: Vec<f32> = fin.iter().map(|c| c.i).collect();
    let mut fout_r = vec![0.0f32; nfft];
    let mut fout_i = vec![0.0f32; nfft];
    c_opus_fft(nfft, &fin_r, &fin_i, &mut fout_r, &mut fout_i);
    (fout_r, fout_i)
}

fn compare_fft(nfft: usize, fin: &[KissFftCpx], tol: f32, name: &str) {
    let st = KissFftState::new(nfft);
    let mut rust_out = vec![KissFftCpx { r: 0.0, i: 0.0 }; nfft];
    opus_fft(&st, fin, &mut rust_out);
    let (c_r, c_i) = c_fft(nfft, fin);
    for i in 0..nfft {
        assert_f32_close(rust_out[i].r, c_r[i], tol, &format!("{name}[{i}].r"));
        assert_f32_close(rust_out[i].i, c_i[i], tol, &format!("{name}[{i}].i"));
    }
}

#[test]
fn fft_delta_120() {
    let nfft = 120;
    let mut fin = vec![KissFftCpx { r: 0.0, i: 0.0 }; nfft];
    fin[0] = KissFftCpx { r: 1.0, i: 0.0 };
    compare_fft(nfft, &fin, 1e-5, "fft120_delta");
}

#[test]
fn fft_delta_240() {
    let nfft = 240;
    let mut fin = vec![KissFftCpx { r: 0.0, i: 0.0 }; nfft];
    fin[0] = KissFftCpx { r: 1.0, i: 0.0 };
    compare_fft(nfft, &fin, 1e-5, "fft240_delta");
}

#[test]
fn fft_sine_120() {
    let nfft = 120;
    let fin: Vec<KissFftCpx> = (0..nfft)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * 10.0 * i as f32 / nfft as f32;
            KissFftCpx {
                r: phase.cos(),
                i: 0.0,
            }
        })
        .collect();
    compare_fft(nfft, &fin, 1e-4, "fft120_sine");
}

#[test]
fn fft_noise_480() {
    let nfft = 480;
    let mut seed = 42u32;
    let fin: Vec<KissFftCpx> = (0..nfft)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let r = (seed as i32 as f32) / (i32::MAX as f32) * 0.5;
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            let i = (seed as i32 as f32) / (i32::MAX as f32) * 0.5;
            KissFftCpx { r, i }
        })
        .collect();
    compare_fft(nfft, &fin, 1e-3, "fft480_noise");
}
