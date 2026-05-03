//! Cross-validation tests for CELT MDCT/IMDCT transforms.

mod common;

use common::assert_f32_slice_close;
use opus_wave::celt::mdct::{MdctLookup, clt_mdct_backward, clt_mdct_forward};
use opus_wave::celt::tables::WINDOW_120;
use opus_ffi::*;

// ── MDCT Forward ──

#[test]
fn mdct_forward_sine_shift0() {
    let n = 960;
    let overlap = 120;
    let n2 = n / 2;
    let input_len = n2 + overlap;
    let input: Vec<f32> = (0..input_len)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
        .collect();
    let mut c_input = input.clone();

    let mut mdct = MdctLookup::new(n, 3);
    let mut rust_output = vec![0.0f32; n2];
    clt_mdct_forward(
        &mut mdct,
        &input,
        &mut rust_output,
        &WINDOW_120,
        overlap,
        0,
        1,
    );

    let mut c_output = vec![0.0f32; n2];
    c_clt_mdct_forward(&mut c_input, &mut c_output, n, overlap, 0, 1);

    assert_f32_slice_close(&rust_output, &c_output, 1e-3, "mdct_fwd(sine, shift=0)");
}

#[test]
fn mdct_forward_noise_shift0() {
    let n = 960;
    let overlap = 120;
    let n2 = n / 2;
    let input_len = n2 + overlap;
    let mut seed = 42u32;
    let input: Vec<f32> = (0..input_len)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as i32 as f32) / (i32::MAX as f32) * 0.3
        })
        .collect();
    let mut c_input = input.clone();

    let mut mdct = MdctLookup::new(n, 3);
    let mut rust_output = vec![0.0f32; n2];
    clt_mdct_forward(
        &mut mdct,
        &input,
        &mut rust_output,
        &WINDOW_120,
        overlap,
        0,
        1,
    );

    let mut c_output = vec![0.0f32; n2];
    c_clt_mdct_forward(&mut c_input, &mut c_output, n, overlap, 0, 1);

    assert_f32_slice_close(&rust_output, &c_output, 1e-3, "mdct_fwd(noise, shift=0)");
}

// ── MDCT Backward (IMDCT) ──

#[test]
fn mdct_backward_shift0() {
    let n = 960;
    let overlap = 120;
    let n2 = n / 2;
    let mut seed = 77u32;
    let input: Vec<f32> = (0..n2)
        .map(|_| {
            seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
            (seed as i32 as f32) / (i32::MAX as f32) * 0.1
        })
        .collect();
    let mut c_input = input.clone();

    let mut mdct = MdctLookup::new(n, 3);
    let mut rust_output = vec![0.0f32; n];
    clt_mdct_backward(
        &mut mdct,
        &input,
        &mut rust_output,
        &WINDOW_120,
        overlap,
        0,
        1,
    );

    let mut c_output = vec![0.0f32; n];
    c_clt_mdct_backward(&mut c_input, &mut c_output, n, overlap, 0, 1);

    assert_f32_slice_close(&rust_output, &c_output, 1e-3, "mdct_bwd(shift=0)");
}

// ── MDCT Shifted (short blocks) ──

#[test]
fn mdct_forward_shift1() {
    let n = 960;
    let overlap = 120;
    let shift = 1;
    let n_shifted = n >> shift;
    let n2_shifted = n_shifted / 2;
    let input_len = n2_shifted + overlap;
    let input: Vec<f32> = (0..input_len)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 1000.0 * i as f32 / 48000.0).sin())
        .collect();
    let mut c_input = input.clone();

    let mut mdct = MdctLookup::new(n, 3);
    let mut rust_output = vec![0.0f32; n2_shifted];
    clt_mdct_forward(
        &mut mdct,
        &input,
        &mut rust_output,
        &WINDOW_120,
        overlap,
        shift,
        1,
    );

    let mut c_output = vec![0.0f32; n2_shifted];
    c_clt_mdct_forward(&mut c_input, &mut c_output, n, overlap, shift, 1);

    assert_f32_slice_close(&rust_output, &c_output, 1e-3, "mdct_fwd(shift=1)");
}
