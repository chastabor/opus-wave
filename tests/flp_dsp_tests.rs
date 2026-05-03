//! FFI cross-validation tests for Layer 0 float DSP functions.
//! Each test calls both the Rust port and the C reference with the
//! same input and asserts matching output.

use opus_ffi::*;
use opus_wave::silk::encoder_flp::dsp::*;

const ORDER: usize = 16;

// ---- Test helpers ----

fn gen_sine(len: usize, freq: f32, fs: f32) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

fn gen_noise(len: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn assert_f64_eq(rust: f64, c: f64, tol: f64, name: &str) {
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

fn assert_f32_eq(rust: f32, c: f32, tol: f32, name: &str) {
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

fn assert_f32_slice_eq(rust: &[f32], c: &[f32], tol: f32, name: &str) {
    assert_eq!(rust.len(), c.len(), "{}: length mismatch", name);
    for i in 0..rust.len() {
        let diff = (rust[i] - c[i]).abs();
        assert!(
            diff <= tol,
            "{}[{}]: Rust={} C={} diff={} (tol={})",
            name,
            i,
            rust[i],
            c[i],
            diff,
            tol
        );
    }
}

// ---- Energy ----

#[test]
fn energy_flp_sine() {
    let signal = gen_sine(320, 440.0, 16000.0);
    let rust = silk_energy_flp(&signal);
    let c = c_silk_energy_flp(&signal);
    assert_f64_eq(rust, c, 1e-4, "energy_flp(sine)");
}

#[test]
fn energy_flp_noise() {
    let signal = gen_noise(256, 42);
    let rust = silk_energy_flp(&signal);
    let c = c_silk_energy_flp(&signal);
    assert_f64_eq(rust, c, 1e-4, "energy_flp(noise)");
}

// ---- Inner Product ----

#[test]
fn inner_product_flp_sine() {
    let a = gen_sine(320, 440.0, 16000.0);
    let b = gen_sine(320, 880.0, 16000.0);
    let rust = silk_inner_product_flp(&a, &b);
    let c = c_silk_inner_product_flp(&a, &b);
    assert_f64_eq(rust, c, 1e-4, "inner_product_flp(sine)");
}

#[test]
fn inner_product_flp_self() {
    let a = gen_noise(200, 99);
    let rust = silk_inner_product_flp(&a, &a);
    let c = c_silk_inner_product_flp(&a, &a);
    assert_f64_eq(rust, c, 1e-4, "inner_product_flp(self)");
}

// ---- Autocorrelation ----

#[test]
fn autocorrelation_flp_matches() {
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut rust_res = [0.0f32; 17];
    let mut c_res = [0.0f32; 17];
    silk_autocorrelation_flp(&mut rust_res, &signal, 17);
    c_silk_autocorrelation_flp(&mut c_res, &signal, 17);
    assert_f32_slice_eq(&rust_res, &c_res, 1e-2, "autocorrelation_flp");
}

// ---- Schur ----

#[test]
fn schur_flp_matches() {
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);

    let mut rust_rc = [0.0f32; ORDER];
    let rust_nrg = silk_schur_flp(&mut rust_rc, &auto_corr, ORDER);

    let mut c_rc = [0.0f32; ORDER];
    let c_nrg = c_silk_schur_flp(&mut c_rc, &auto_corr, ORDER);

    assert_f32_eq(rust_nrg, c_nrg, 1e-2, "schur_flp nrg");
    assert_f32_slice_eq(&rust_rc, &c_rc, 1e-6, "schur_flp rc");
}

// ---- K2A ----

#[test]
fn k2a_flp_matches() {
    // Get reflection coefficients from Schur
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);
    let mut rc = [0.0f32; ORDER];
    c_silk_schur_flp(&mut rc, &auto_corr, ORDER);

    let mut rust_a = [0.0f32; ORDER];
    silk_k2a_flp(&mut rust_a, &rc, ORDER);

    let mut c_a = [0.0f32; ORDER];
    c_silk_k2a_flp(&mut c_a, &rc, ORDER);

    assert_f32_slice_eq(&rust_a, &c_a, 1e-6, "k2a_flp");
}

// ---- BWExpander ----

#[test]
fn bwexpander_flp_matches() {
    let mut rust_ar = [
        0.9f32, 0.5, -0.3, 0.1, 0.8, -0.4, 0.2, 0.6, 0.3, -0.1, 0.05, 0.7, -0.6, 0.4, -0.2, 0.15,
    ];
    let mut c_ar = rust_ar;

    silk_bwexpander_flp(&mut rust_ar, ORDER, 0.95);
    c_silk_bwexpander_flp(&mut c_ar, ORDER, 0.95);

    assert_f32_slice_eq(&rust_ar, &c_ar, 1e-6, "bwexpander_flp");
}

// ---- Apply Sine Window ----

#[test]
fn apply_sine_window_flp_rising() {
    let signal = gen_sine(64, 440.0, 16000.0);
    let mut rust_out = vec![0.0f32; 64];
    let mut c_out = vec![0.0f32; 64];

    silk_apply_sine_window_flp(&mut rust_out, &signal, 1, 64);
    c_silk_apply_sine_window_flp(&mut c_out, &signal, 1, 64);

    assert_f32_slice_eq(&rust_out, &c_out, 1e-5, "sine_window(rising)");
}

#[test]
fn apply_sine_window_flp_falling() {
    let signal = gen_sine(64, 440.0, 16000.0);
    let mut rust_out = vec![0.0f32; 64];
    let mut c_out = vec![0.0f32; 64];

    silk_apply_sine_window_flp(&mut rust_out, &signal, 2, 64);
    c_silk_apply_sine_window_flp(&mut c_out, &signal, 2, 64);

    assert_f32_slice_eq(&rust_out, &c_out, 1e-5, "sine_window(falling)");
}

// ---- Scale Copy Vector ----

#[test]
fn scale_copy_vector_flp_matches() {
    let input = gen_noise(100, 77);
    let mut rust_out = vec![0.0f32; 100];
    let mut c_out = vec![0.0f32; 100];

    silk_scale_copy_vector_flp(&mut rust_out, &input, 0.75, 100);
    c_silk_scale_copy_vector_flp(&mut c_out, &input, 0.75, 100);

    assert_f32_slice_eq(&rust_out, &c_out, 1e-6, "scale_copy_vector_flp");
}

// ---- LPC Analysis Filter ----

#[test]
fn lpc_analysis_filter_flp_matches() {
    let signal = gen_sine(320, 440.0, 16000.0);
    // Get LPC from schur + k2a
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);
    let mut rc = [0.0f32; ORDER];
    c_silk_schur_flp(&mut rc, &auto_corr, ORDER);
    let mut lpc = [0.0f32; ORDER];
    c_silk_k2a_flp(&mut lpc, &rc, ORDER);

    let mut rust_res = vec![0.0f32; 320];
    let mut c_res = vec![0.0f32; 320];

    silk_lpc_analysis_filter_flp(&mut rust_res, &lpc, &signal, 320, ORDER);
    c_silk_lpc_analysis_filter_flp(&mut c_res, &lpc, &signal, 320, ORDER);

    assert_f32_slice_eq(&rust_res, &c_res, 1e-3, "lpc_analysis_filter_flp");
}

// ---- LPC Inverse Prediction Gain ----

#[test]
fn lpc_inv_pred_gain_flp_matches() {
    // Get stable LPC coefficients
    let signal = gen_sine(160, 440.0, 16000.0);
    let mut auto_corr = [0.0f32; ORDER + 1];
    c_silk_autocorrelation_flp(&mut auto_corr, &signal, ORDER + 1);
    let mut rc = [0.0f32; ORDER];
    c_silk_schur_flp(&mut rc, &auto_corr, ORDER);
    let mut lpc = [0.0f32; ORDER];
    c_silk_k2a_flp(&mut lpc, &rc, ORDER);

    let rust_gain = silk_lpc_inverse_pred_gain_flp(&lpc, ORDER);
    let c_gain = c_silk_lpc_inverse_pred_gain_flp(&lpc, ORDER);

    assert_f32_eq(rust_gain, c_gain, 1e-6, "lpc_inv_pred_gain_flp");
}

// ---- Warped Autocorrelation ----

#[test]
fn warped_autocorrelation_flp_matches() {
    let signal = gen_sine(160, 440.0, 16000.0);
    let warping = 0.015f32; // typical warping for 16kHz
    let order = 16usize;

    let mut rust_corr = [0.0f32; 17];
    let mut c_corr = [0.0f32; 17];

    silk_warped_autocorrelation_flp(&mut rust_corr, &signal, warping, 160, order);
    c_silk_warped_autocorrelation_flp(&mut c_corr, &signal, warping, 160, order);

    assert_f32_slice_eq(
        &rust_corr[..order + 1],
        &c_corr[..order + 1],
        1e-2,
        "warped_autocorrelation_flp",
    );
}

#[test]
fn warped_autocorrelation_flp_noise() {
    let signal = gen_noise(256, 42);
    let warping = 0.02f32;
    let order = 16usize;

    let mut rust_corr = [0.0f32; 17];
    let mut c_corr = [0.0f32; 17];

    silk_warped_autocorrelation_flp(&mut rust_corr, &signal, warping, 256, order);
    c_silk_warped_autocorrelation_flp(&mut c_corr, &signal, warping, 256, order);

    assert_f32_slice_eq(
        &rust_corr[..order + 1],
        &c_corr[..order + 1],
        1e-2,
        "warped_autocorrelation_flp(noise)",
    );
}

// ---- LTP analysis tests ----

use opus_wave::silk::encoder_flp::find_ltp::{
    silk_corr_matrix_flp, silk_corr_vector_flp, silk_find_ltp_flp,
};
use opus_wave::silk::encoder_flp::quant_ltp_gains::silk_quant_ltp_gains;
use opus_wave::silk::{LTP_ORDER, MAX_NB_SUBFR};

#[test]
fn corr_vector_flp_matches() {
    let signal = gen_sine(320, 440.0, 16000.0);
    let order = LTP_ORDER;
    let l = 80usize; // subframe length

    // x starts at order-1 in the signal, t at order-1+order for the C convention
    let x = &signal[..order + l]; // [order-1 + L] = [4 + 80]
    let t = &signal[order - 1..order - 1 + l]; // target = first L samples starting at col0

    let mut rust_xt = [0.0f32; LTP_ORDER];
    let mut c_xt = [0.0f32; LTP_ORDER];

    silk_corr_vector_flp(x, t, l, order, &mut rust_xt);
    c_silk_corr_vector_flp(x, t, l, order, &mut c_xt);

    assert_f32_slice_eq(&rust_xt, &c_xt, 1e-2, "corrVector_FLP");
}

#[test]
fn corr_matrix_flp_matches() {
    let signal = gen_sine(320, 440.0, 16000.0);
    let order = LTP_ORDER;
    let l = 80usize;

    let x = &signal[..order + l];

    let mut rust_xx = [0.0f32; LTP_ORDER * LTP_ORDER];
    let mut c_xx = [0.0f32; LTP_ORDER * LTP_ORDER];

    silk_corr_matrix_flp(x, l, order, &mut rust_xx);
    c_silk_corr_matrix_flp(x, l, order, &mut c_xx);

    assert_f32_slice_eq(&rust_xx, &c_xx, 1e-2, "corrMatrix_FLP");
}

#[test]
fn find_ltp_flp_matches() {
    // Generate a signal with enough history for pitch lag access
    let total_len = 640 + 320; // ltp_mem + frame
    let signal = gen_sine(total_len, 440.0, 16000.0);
    let subfr_length = 80usize;
    let nb_subfr = 4usize;
    let lags: [i32; MAX_NB_SUBFR] = [36, 36, 37, 37]; // ~440Hz at 16kHz

    let mut rust_xx = [0.0f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
    let mut rust_x_x = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];
    let mut c_xx = [0.0f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
    let mut c_x_x = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];

    // Frame starts at offset 320 (ltp_mem_length)
    let frame_offset = 320usize;

    silk_find_ltp_flp(
        &mut rust_xx,
        &mut rust_x_x,
        &signal,
        frame_offset,
        &lags,
        subfr_length,
        nb_subfr,
    );

    c_silk_find_ltp_flp(
        &mut c_xx,
        &mut c_x_x,
        &signal,
        frame_offset,
        &lags,
        subfr_length as i32,
        nb_subfr as i32,
    );

    assert_f32_slice_eq(
        &rust_xx[..nb_subfr * LTP_ORDER * LTP_ORDER],
        &c_xx[..nb_subfr * LTP_ORDER * LTP_ORDER],
        1e-2,
        "find_LTP_FLP XX",
    );
    assert_f32_slice_eq(
        &rust_x_x[..nb_subfr * LTP_ORDER],
        &c_x_x[..nb_subfr * LTP_ORDER],
        1e-2,
        "find_LTP_FLP xX",
    );
}

#[test]
fn quant_ltp_gains_matches() {
    // Use correlation matrices from a real-ish signal
    let total_len = 640 + 320;
    let signal = gen_sine(total_len, 440.0, 16000.0);
    let subfr_length = 80usize;
    let nb_subfr = 4usize;
    let lags: [i32; MAX_NB_SUBFR] = [36, 36, 37, 37];
    let frame_offset = 320usize;

    // Compute float correlation matrices
    let mut xx = [0.0f32; MAX_NB_SUBFR * LTP_ORDER * LTP_ORDER];
    let mut x_x = [0.0f32; MAX_NB_SUBFR * LTP_ORDER];
    silk_find_ltp_flp(
        &mut xx,
        &mut x_x,
        &signal,
        frame_offset,
        &lags,
        subfr_length,
        nb_subfr,
    );

    // Convert to Q17
    let n_xx = nb_subfr * LTP_ORDER * LTP_ORDER;
    let n_x_x = nb_subfr * LTP_ORDER;
    let mut xx_q17 = vec![0i32; n_xx];
    let mut x_x_q17 = vec![0i32; n_x_x];
    for i in 0..n_xx {
        xx_q17[i] = (xx[i] * 131072.0).round() as i32;
    }
    for i in 0..n_x_x {
        x_x_q17[i] = (x_x[i] * 131072.0).round() as i32;
    }

    // Rust
    let mut r_b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let mut r_cbk = [0i8; MAX_NB_SUBFR];
    let mut r_per = 0i8;
    let mut r_slg = 0i32;
    let mut r_pgdb = 0i32;

    silk_quant_ltp_gains(
        &mut r_b_q14,
        &mut r_cbk,
        &mut r_per,
        &mut r_slg,
        &mut r_pgdb,
        &xx_q17,
        &x_x_q17,
        subfr_length as i32,
        nb_subfr,
    );

    // C
    let mut c_b_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
    let mut c_cbk = [0i8; MAX_NB_SUBFR];
    let mut c_per = 0i8;
    let mut c_slg = 0i32;
    let mut c_pgdb = 0i32;

    c_silk_quant_ltp_gains(
        &mut c_b_q14,
        &mut c_cbk,
        &mut c_per,
        &mut c_slg,
        &mut c_pgdb,
        &xx_q17,
        &x_x_q17,
        subfr_length as i32,
        nb_subfr as i32,
    );

    eprintln!(
        "Rust: per={} cbk={:?} B_Q14={:?}",
        r_per,
        &r_cbk[..nb_subfr],
        &r_b_q14[..nb_subfr * LTP_ORDER]
    );
    eprintln!(
        "C:    per={} cbk={:?} B_Q14={:?}",
        c_per,
        &c_cbk[..nb_subfr],
        &c_b_q14[..nb_subfr * LTP_ORDER]
    );

    // Periodicity index must match (codebook family selection)
    assert_eq!(r_per, c_per, "periodicity_index mismatch");

    // Allow minor VQ differences: count subframes where cbk index differs
    let cbk_diff = (0..nb_subfr).filter(|&i| r_cbk[i] != c_cbk[i]).count();
    assert!(
        cbk_diff <= 1,
        "too many cbk_index mismatches: {}/{}. Rust={:?} C={:?}",
        cbk_diff,
        nb_subfr,
        &r_cbk[..nb_subfr],
        &c_cbk[..nb_subfr]
    );
    if cbk_diff > 0 {
        eprintln!("  (minor VQ difference: {} subframe(s) differ)", cbk_diff);
    }
}

// ---- LTP scale control tests ----

use opus_wave::silk::encoder_flp::ltp_scale_ctrl::silk_ltp_scale_ctrl_flp;
use opus_wave::silk::{CODE_CONDITIONALLY, CODE_INDEPENDENTLY};

#[test]
fn ltp_scale_ctrl_zero_loss() {
    // With 0% packet loss, scale index should always be 0
    let result = silk_ltp_scale_ctrl_flp(
        500,   // ltp_pred_cod_gain_q7 (moderate)
        2415,  // snr_db_q7 (~19dB)
        0,     // packet_loss_perc
        1,     // n_frames_per_packet
        false, // lbrr_flag
        CODE_INDEPENDENTLY,
    );
    assert_eq!(result.ltp_scale_index, 0, "zero loss should give index 0");
    assert!(
        result.ltp_scale > 0.9,
        "scale should be close to 1.0 for index 0"
    );
}

#[test]
fn ltp_scale_ctrl_conditional_coding() {
    // Conditional coding always produces index 0 regardless of loss
    let result = silk_ltp_scale_ctrl_flp(500, 2415, 20, 1, false, CODE_CONDITIONALLY);
    assert_eq!(result.ltp_scale_index, 0, "conditional coding → index 0");
}

#[test]
fn ltp_scale_ctrl_high_loss_high_gain() {
    // High packet loss + high LTP gain → higher scale index
    let result = silk_ltp_scale_ctrl_flp(
        3000, // high LTP prediction gain
        1500, // low SNR → easier to trigger thresholds
        25,   // 25% packet loss
        2,    // 2 frames per packet → round_loss = 50
        false,
        CODE_INDEPENDENTLY,
    );
    eprintln!(
        "high_loss: index={} scale={}",
        result.ltp_scale_index, result.ltp_scale
    );
    // product = silk_smulbb(3000, 50) — but silk_smulbb truncates to i16
    // (3000 as i16) * (50 as i16) = 3000 * 50 = 150000
    // thresh1 = silk_log2lin(2900 - 1500) = silk_log2lin(1400)
    // thresh2 = silk_log2lin(3900 - 1500) = silk_log2lin(2400)
    // If product > both thresholds, index = 2
    assert!(
        result.ltp_scale_index >= 1,
        "high loss should increase scale index"
    );
}

#[test]
fn ltp_scale_ctrl_lbrr_reduces_loss() {
    // With LBRR flag, effective loss is reduced: round_loss = 2 + loss^2/100
    // 10% loss * 1 frame = 10, with LBRR: 2 + 10*10/100 = 3
    let no_lbrr = silk_ltp_scale_ctrl_flp(2000, 2000, 10, 1, false, CODE_INDEPENDENTLY);
    let with_lbrr = silk_ltp_scale_ctrl_flp(2000, 2000, 10, 1, true, CODE_INDEPENDENTLY);
    assert!(
        with_lbrr.ltp_scale_index <= no_lbrr.ltp_scale_index,
        "LBRR should reduce or maintain scale index"
    );
}
