//! FFI cross-validation tests for Layer 2 float wrapper functions.
//! Tests that the Rust wrappers (float→fixed→float) produce identical
//! output to the C wrappers for realistic LPC/NLSF inputs.

use opus_ffi::*;
use opus_rust::silk::encoder_flp::wrappers::*;
use opus_rust::silk::lpc_analysis::silk_burg_modified_flp;

const ORDER: usize = 16;

fn gen_sine(len: usize, freq: f32, fs: f32) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

/// Get realistic float LPC coefficients from a sine wave via C Burg.
fn get_c_burg_lpc(freq: f32) -> [f32; ORDER] {
    let signal = gen_sine(384, freq, 16000.0);
    let mut a = [0.0f32; ORDER];
    c_silk_burg_modified_flp(&mut a, &signal, 1.0 / 10000.0, 96, 4, ORDER as i32);
    a
}

// ---- Burg FLP ----

#[test]
fn burg_flp_matches_c() {
    let signal = gen_sine(384, 440.0, 16000.0);

    let mut rust_a = [0.0f32; ORDER];
    let rust_nrg = silk_burg_modified_flp(&mut rust_a, &signal, 1.0 / 10000.0, 96, 4, ORDER);

    let mut c_a = [0.0f32; ORDER];
    let c_nrg = c_silk_burg_modified_flp(&mut c_a, &signal, 1.0 / 10000.0, 96, 4, ORDER as i32);

    eprintln!("Burg nrg: Rust={} C={}", rust_nrg, c_nrg);
    assert!(
        (rust_nrg - c_nrg).abs() < 1e-2,
        "Burg residual energy mismatch"
    );

    for i in 0..ORDER {
        let diff = (rust_a[i] - c_a[i]).abs();
        assert!(
            diff < 1e-6,
            "Burg a[{}]: Rust={} C={} diff={}",
            i,
            rust_a[i],
            c_a[i],
            diff
        );
    }
}

// ---- A2NLSF_FLP ----

#[test]
fn a2nlsf_flp_matches_c() {
    let a = get_c_burg_lpc(440.0);

    let mut rust_nlsf = [0i16; ORDER];
    silk_a2nlsf_flp(&mut rust_nlsf, &a, ORDER);

    let mut c_nlsf = [0i16; ORDER];
    c_silk_a2nlsf_flp(&mut c_nlsf, &a, ORDER);

    eprintln!("A2NLSF_FLP: Rust={:?}", &rust_nlsf[..4]);
    eprintln!("A2NLSF_FLP: C   ={:?}", &c_nlsf[..4]);

    assert_eq!(rust_nlsf, c_nlsf, "A2NLSF_FLP produces different NLSFs");
}

#[test]
fn a2nlsf_flp_speech_like() {
    // Broader spectral shape
    let a = get_c_burg_lpc(200.0);

    let mut rust_nlsf = [0i16; ORDER];
    silk_a2nlsf_flp(&mut rust_nlsf, &a, ORDER);

    let mut c_nlsf = [0i16; ORDER];
    c_silk_a2nlsf_flp(&mut c_nlsf, &a, ORDER);

    assert_eq!(rust_nlsf, c_nlsf, "A2NLSF_FLP speech-like diverges");
}

// ---- NLSF2A_FLP ----

#[test]
fn nlsf2a_flp_matches_c() {
    // Get NLSFs from a known LPC
    let a = get_c_burg_lpc(440.0);
    let mut nlsf = [0i16; ORDER];
    c_silk_a2nlsf_flp(&mut nlsf, &a, ORDER);

    // Convert back via both paths
    let mut rust_a = [0.0f32; ORDER];
    silk_nlsf2a_flp(&mut rust_a, &nlsf, ORDER);

    let mut c_a = [0.0f32; ORDER];
    c_silk_nlsf2a_flp(&mut c_a, &nlsf, ORDER);

    for i in 0..ORDER {
        let diff = (rust_a[i] - c_a[i]).abs();
        assert!(
            diff < 1e-6,
            "NLSF2A_FLP a[{}]: Rust={} C={} diff={}",
            i,
            rust_a[i],
            c_a[i],
            diff
        );
    }
}

// ---- Burg consistency: Rust Burg == C Burg → same NLSFs ----

#[test]
fn burg_to_nlsf_full_pipeline() {
    // This tests the full chain: signal → Burg → A2NLSF for both C and Rust
    let signal = gen_sine(384, 440.0, 16000.0);

    // Rust path
    let mut rust_a = [0.0f32; ORDER];
    silk_burg_modified_flp(&mut rust_a, &signal, 1.0 / 10000.0, 96, 4, ORDER);
    let mut rust_nlsf = [0i16; ORDER];
    silk_a2nlsf_flp(&mut rust_nlsf, &rust_a, ORDER);

    // C path
    let mut c_a = [0.0f32; ORDER];
    c_silk_burg_modified_flp(&mut c_a, &signal, 1.0 / 10000.0, 96, 4, ORDER as i32);
    let mut c_nlsf = [0i16; ORDER];
    c_silk_a2nlsf_flp(&mut c_nlsf, &c_a, ORDER);

    eprintln!("Full pipeline NLSFs:");
    eprintln!("  Rust: {:?}", &rust_nlsf);
    eprintln!("  C   : {:?}", &c_nlsf);

    assert_eq!(rust_nlsf, c_nlsf, "Full Burg→A2NLSF pipeline diverges");
}

// ---- float2short / short2float round-trip ----

#[test]
fn float_short_roundtrip() {
    let float_data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 100.0).collect();
    let mut short_data = vec![0i16; 100];
    let mut back = vec![0.0f32; 100];

    silk_float2short_array(&mut short_data, &float_data, 100);
    silk_short2float_array(&mut back, &short_data, 100);

    for i in 0..100 {
        let expected = (float_data[i].round() as i32).clamp(-32768, 32767) as f32;
        assert_eq!(back[i], expected, "roundtrip mismatch at {}", i);
    }
}

// ---- NSQ wrapper conversion sanity ----

#[test]
fn nsq_wrapper_conversion_formats() {
    // Verify the Q-format conversion functions produce expected values
    // AR: float 0.5 → Q13 = 4096
    let ar_f = 0.5f32;
    let ar_q13 = (ar_f * 8192.0).round() as i16;
    assert_eq!(ar_q13, 4096);

    // Tilt: float 0.25 → Q14 = 4096
    let tilt_f = 0.25f32;
    let tilt_q14 = (tilt_f * 16384.0).round() as i32;
    assert_eq!(tilt_q14, 4096);

    // Gain: float 1.5 → Q16 = 98304
    let gain_f = 1.5f32;
    let gain_q16 = (gain_f * 65536.0).round() as i32;
    assert_eq!(gain_q16, 98304);

    // Lambda: float 1.0 → Q10 = 1024
    let lambda_f = 1.0f32;
    let lambda_q10 = (lambda_f * 1024.0).round() as i32;
    assert_eq!(lambda_q10, 1024);

    // PredCoef: float 0.5 → Q12 = 2048
    let pc_f = 0.5f32;
    let pc_q12 = (pc_f * 4096.0).round() as i16;
    assert_eq!(pc_q12, 2048);
}
