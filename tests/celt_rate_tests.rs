//! Cross-validation tests for CELT rate allocation functions.
//! bits2pulses, pulses2bits, init_caps — all exact-match (integer functions).

use opus_rust::celt::mode::CeltMode;
use opus_rust::celt::rate;
use opus_ffi::*;

// ══════════════════════════════════════════════════════════════════════
// bits2pulses / pulses2bits sweep
// ══════════════════════════════════════════════════════════════════════

#[test]
fn bits2pulses_sweep_all_bands() {
    let m = CeltMode::get_mode();
    // Sweep all 21 bands, LM 0..3, various bit counts
    for lm in 0..=3i32 {
        for band in 0..m.nb_ebands {
            for bits in [0, 8, 16, 32, 64, 128, 256, 512, 1024] {
                let rust = rate::bits2pulses(m, band, lm, bits);
                let c = c_bits2pulses(band, lm as usize, bits);
                assert_eq!(
                    rust, c,
                    "bits2pulses(band={band}, LM={lm}, bits={bits}): Rust={rust} C={c}"
                );
            }
        }
    }
}

#[test]
fn pulses2bits_sweep_all_bands() {
    let m = CeltMode::get_mode();
    for lm in 0..=3i32 {
        for band in 0..m.nb_ebands {
            // Only test valid pulse counts: those returned by bits2pulses
            for bits in [0, 8, 16, 32, 64, 128, 256, 512, 1024] {
                let k = rate::bits2pulses(m, band, lm, bits);
                if k > 0 {
                    let rust = rate::pulses2bits(m, band, lm, k);
                    let c = c_pulses2bits(band, lm as usize, k);
                    assert_eq!(
                        rust, c,
                        "pulses2bits(band={band}, LM={lm}, k={k}): Rust={rust} C={c}"
                    );
                }
            }
        }
    }
}

#[test]
fn bits2pulses_pulses2bits_roundtrip() {
    // bits2pulses then pulses2bits should give a consistent result
    let m = CeltMode::get_mode();
    for lm in 0..=3i32 {
        for band in 0..m.nb_ebands {
            for bits in [32, 64, 128, 256] {
                let rust_k = rate::bits2pulses(m, band, lm, bits);
                let c_k = c_bits2pulses(band, lm as usize, bits);
                assert_eq!(rust_k, c_k, "roundtrip bits2pulses mismatch");

                if rust_k > 0 {
                    let rust_b = rate::pulses2bits(m, band, lm, rust_k);
                    let c_b = c_pulses2bits(band, lm as usize, c_k);
                    assert_eq!(rust_b, c_b, "roundtrip pulses2bits mismatch");
                }
            }
        }
    }
}

// ══════════════════════════════════════════════════════════════════════
// init_caps
// ══════════════════════════════════════════════════════════════════════

#[test]
fn init_caps_all_lm_mono() {
    let m = CeltMode::get_mode();
    for lm in 0..=3i32 {
        let mut rust_cap = vec![0i32; m.nb_ebands];
        let mut c_cap = vec![0i32; m.nb_ebands];

        rate::init_caps(m, &mut rust_cap, lm, 1);
        c_init_caps(&mut c_cap, lm as usize, 1);

        assert_eq!(rust_cap, c_cap, "init_caps(LM={lm}, C=1) mismatch");
    }
}

#[test]
fn init_caps_all_lm_stereo() {
    let m = CeltMode::get_mode();
    for lm in 0..=3i32 {
        let mut rust_cap = vec![0i32; m.nb_ebands];
        let mut c_cap = vec![0i32; m.nb_ebands];

        rate::init_caps(m, &mut rust_cap, lm, 2);
        c_init_caps(&mut c_cap, lm as usize, 2);

        assert_eq!(rust_cap, c_cap, "init_caps(LM={lm}, C=2) mismatch");
    }
}
