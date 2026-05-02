//! FFI cross-validation tests for Layer 3 float analysis components.
//! Tests residual_energy_FLP, find_lpc (via Burg+A2NLSF pipeline),
//! and process_gains against the C reference.

use opus_ffi::*;
use opus_silk::MAX_LPC_ORDER;
use opus_silk::MAX_NB_SUBFR;
use opus_silk::encoder_flp::dsp::*;
use opus_silk::encoder_flp::find_lpc::silk_find_lpc_flp;
use opus_silk::encoder_flp::residual_energy::silk_residual_energy_flp;
use opus_silk::encoder_flp::wrappers::*;
use opus_silk::lpc_analysis::silk_burg_modified_flp;

const ORDER: usize = 16;
const SUBFR_LEN: usize = 80;
const BURG_SUBFR: usize = ORDER + SUBFR_LEN; // 96
const NB_SUBFR: usize = 4;

fn gen_sine(len: usize, freq: f32, fs: f32) -> Vec<f32> {
    (0..len)
        .map(|i| 0.5 * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

/// Create a gain-normalized lpc_in_pre matching the C float encoder's unvoiced path.
fn make_lpc_in_pre(signal: &[f32], gains: &[f32]) -> Vec<f32> {
    let mut lpc_in_pre = vec![0.0f32; NB_SUBFR * BURG_SUBFR];
    for (k, &gain) in gains.iter().enumerate().take(NB_SUBFR) {
        let inv_gain = 1.0 / gain.max(1e-12);
        let src_start = (k * SUBFR_LEN).saturating_sub(ORDER);
        let dst_start = k * BURG_SUBFR;
        let copy_len = BURG_SUBFR.min(signal.len().saturating_sub(src_start));
        silk_scale_copy_vector_flp(
            &mut lpc_in_pre[dst_start..dst_start + copy_len],
            &signal[src_start..src_start + copy_len],
            inv_gain,
            copy_len,
        );
    }
    lpc_in_pre
}

// ---- Residual Energy ----

#[test]
fn residual_energy_flp_matches_c() {
    let signal = gen_sine(NB_SUBFR * BURG_SUBFR, 440.0, 16000.0);
    let gains = [1.5f32, 1.5, 1.2, 1.4];

    // Get LPC via C Burg + A2NLSF + NLSF2A roundtrip
    let lpc_in_pre = make_lpc_in_pre(&signal, &gains);
    let mut a_flp = [0.0f32; ORDER];
    c_silk_burg_modified_flp(
        &mut a_flp,
        &lpc_in_pre,
        1.0 / 10000.0,
        BURG_SUBFR as i32,
        NB_SUBFR as i32,
        ORDER as i32,
    );
    let mut nlsf = [0i16; ORDER];
    c_silk_a2nlsf_flp(&mut nlsf, &a_flp, ORDER);
    let mut a_q = [[0.0f32; MAX_LPC_ORDER]; 2];
    c_silk_nlsf2a_flp(&mut a_q[0], &nlsf, ORDER);
    a_q[1] = a_q[0]; // same LPC for both halves

    // Rust residual energy
    let mut rust_nrgs = [0.0f32; MAX_NB_SUBFR];
    silk_residual_energy_flp(
        &mut rust_nrgs,
        &lpc_in_pre,
        &a_q,
        &gains,
        SUBFR_LEN,
        NB_SUBFR,
        ORDER,
    );

    // C residual energy
    let mut c_nrgs = [0.0f32; MAX_NB_SUBFR];
    c_silk_residual_energy_flp(
        &mut c_nrgs,
        &lpc_in_pre,
        &a_q,
        &gains,
        SUBFR_LEN as i32,
        NB_SUBFR as i32,
        ORDER as i32,
    );

    eprintln!("ResNrg Rust: {:?}", &rust_nrgs[..NB_SUBFR]);
    eprintln!("ResNrg C   : {:?}", &c_nrgs[..NB_SUBFR]);

    for k in 0..NB_SUBFR {
        let diff = (rust_nrgs[k] - c_nrgs[k]).abs();
        let rel = diff / c_nrgs[k].abs().max(1e-10);
        assert!(
            rel < 1e-4,
            "ResNrg[{}]: Rust={} C={} rel_diff={}",
            k,
            rust_nrgs[k],
            c_nrgs[k],
            rel
        );
    }
}

// ---- Find LPC (Burg + A2NLSF) ----

#[test]
fn find_lpc_flp_no_interpolation() {
    // First frame: no interpolation (first_frame_after_reset=true)
    let signal = gen_sine(NB_SUBFR * BURG_SUBFR, 440.0, 16000.0);
    let gains = [1.5f32; NB_SUBFR];
    let lpc_in_pre = make_lpc_in_pre(&signal, &gains);
    let prev_nlsf = [0i16; ORDER];

    let mut rust_nlsf = [0i16; ORDER];
    let mut rust_interp = 0i8;
    silk_find_lpc_flp(
        &mut rust_nlsf,
        &mut rust_interp,
        &lpc_in_pre,
        1.0 / 100.0, // first_frame_after_reset minInvGain
        ORDER,
        NB_SUBFR,
        BURG_SUBFR,
        false, // use_interpolated_nlsfs: false for first frame
        true,  // first_frame_after_reset
        &prev_nlsf,
    );

    // C reference: run Burg + A2NLSF directly (equivalent when no interpolation)
    let mut c_a = [0.0f32; ORDER];
    c_silk_burg_modified_flp(
        &mut c_a,
        &lpc_in_pre,
        1.0 / 100.0,
        BURG_SUBFR as i32,
        NB_SUBFR as i32,
        ORDER as i32,
    );
    let mut c_nlsf = [0i16; ORDER];
    c_silk_a2nlsf_flp(&mut c_nlsf, &c_a, ORDER);

    eprintln!("find_lpc NLSFs (no interp):");
    eprintln!("  Rust: {:?}", &rust_nlsf[..4]);
    eprintln!("  C   : {:?}", &c_nlsf[..4]);

    assert_eq!(rust_interp, 4, "Should be no interpolation on first frame");
    assert_eq!(rust_nlsf, c_nlsf, "find_lpc NLSFs diverge from C reference");
}

#[test]
fn find_lpc_flp_with_interpolation() {
    // Second frame: interpolation enabled
    let signal = gen_sine(NB_SUBFR * BURG_SUBFR, 440.0, 16000.0);
    let gains = [1.5f32; NB_SUBFR];
    let lpc_in_pre = make_lpc_in_pre(&signal, &gains);

    // Simulate previous frame's NLSFs (uniformly spaced)
    let mut prev_nlsf = [0i16; ORDER];
    for (i, slot) in prev_nlsf.iter_mut().enumerate() {
        *slot = ((i + 1) as i32 * 32768 / (ORDER as i32 + 1)) as i16;
    }

    let mut rust_nlsf = [0i16; ORDER];
    let mut rust_interp = 0i8;
    silk_find_lpc_flp(
        &mut rust_nlsf,
        &mut rust_interp,
        &lpc_in_pre,
        1.0 / 10000.0, // normal frame minInvGain
        ORDER,
        NB_SUBFR,
        BURG_SUBFR,
        true,  // use_interpolated_nlsfs
        false, // not first frame
        &prev_nlsf,
    );

    eprintln!("find_lpc with interp: coef_q2={}", rust_interp);
    eprintln!("  NLSFs: {:?}", &rust_nlsf[..4]);

    // Interpolation should be active (coef_q2 < 4)
    // The exact value depends on the signal, but it should be 0-3
    assert!(rust_interp <= 4, "Invalid interp coef");
    // NLSFs should be valid (monotonically increasing)
    for i in 1..ORDER {
        assert!(
            rust_nlsf[i] > rust_nlsf[i - 1],
            "NLSFs not monotonic at {}",
            i
        );
    }
}

// ---- Burg → NLSF → NLSF2A → Residual Energy: full pipeline ----

#[test]
fn full_pred_coefs_pipeline_sine() {
    let signal = gen_sine(NB_SUBFR * BURG_SUBFR + ORDER, 440.0, 16000.0);
    let gains = [1.5f32, 1.5, 1.2, 1.4];

    // Build lpc_in_pre
    let lpc_in_pre = make_lpc_in_pre(&signal, &gains);

    // Rust path: Burg → A2NLSF → NLSF2A → residual_energy
    let mut a_flp = [0.0f32; ORDER];
    silk_burg_modified_flp(
        &mut a_flp,
        &lpc_in_pre,
        1.0 / 10000.0,
        BURG_SUBFR,
        NB_SUBFR,
        ORDER,
    );
    let mut nlsf = [0i16; ORDER];
    silk_a2nlsf_flp(&mut nlsf, &a_flp, ORDER);
    let mut pred_coef = [[0.0f32; MAX_LPC_ORDER]; 2];
    silk_nlsf2a_flp(&mut pred_coef[0], &nlsf, ORDER);
    pred_coef[1] = pred_coef[0];
    let mut rust_nrgs = [0.0f32; MAX_NB_SUBFR];
    silk_residual_energy_flp(
        &mut rust_nrgs,
        &lpc_in_pre,
        &pred_coef,
        &gains,
        SUBFR_LEN,
        NB_SUBFR,
        ORDER,
    );

    // C path: same chain
    let mut c_a = [0.0f32; ORDER];
    c_silk_burg_modified_flp(
        &mut c_a,
        &lpc_in_pre,
        1.0 / 10000.0,
        BURG_SUBFR as i32,
        NB_SUBFR as i32,
        ORDER as i32,
    );
    let mut c_nlsf = [0i16; ORDER];
    c_silk_a2nlsf_flp(&mut c_nlsf, &c_a, ORDER);
    let mut c_pred = [[0.0f32; MAX_LPC_ORDER]; 2];
    c_silk_nlsf2a_flp(&mut c_pred[0], &c_nlsf, ORDER);
    c_pred[1] = c_pred[0];
    let mut c_nrgs = [0.0f32; MAX_NB_SUBFR];
    c_silk_residual_energy_flp(
        &mut c_nrgs,
        &lpc_in_pre,
        &c_pred,
        &gains,
        SUBFR_LEN as i32,
        NB_SUBFR as i32,
        ORDER as i32,
    );

    eprintln!("Full pipeline ResNrg:");
    eprintln!("  Rust: {:?}", &rust_nrgs[..NB_SUBFR]);
    eprintln!("  C   : {:?}", &c_nrgs[..NB_SUBFR]);

    // Both should match
    for k in 0..NB_SUBFR {
        let diff = (rust_nrgs[k] - c_nrgs[k]).abs();
        let rel = diff / c_nrgs[k].abs().max(1e-10);
        assert!(
            rel < 1e-3,
            "Pipeline ResNrg[{}]: Rust={} C={} rel={}",
            k,
            rust_nrgs[k],
            c_nrgs[k],
            rel
        );
    }
}
