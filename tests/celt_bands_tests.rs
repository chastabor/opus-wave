//! Cross-validation tests for CELT band processing functions.

mod common;

use common::{assert_f32_slice_close, gen_noise, gen_sine_vec};
use opus_rust::celt::bands;
use opus_rust::celt::mode::CeltMode;
use opus_ffi::*;

// ── compute_band_energies ──

#[test]
fn compute_band_energies_sine_mono() {
    let m = CeltMode::get_mode();
    let lm = 0usize;
    let mm = 1usize << lm;
    let eff_end = m.nb_ebands;
    let c = 1;
    let freq_size = mm * m.ebands[eff_end] as usize;
    let freq = gen_sine_vec(freq_size, 1000.0, 48000.0, 0.5);

    let mut rust_band_e = vec![0.0f32; eff_end * c];
    let mut c_band_e = vec![0.0f32; eff_end * c];
    bands::compute_band_energies(m, &freq, &mut rust_band_e, eff_end, c, lm as i32);
    c_compute_band_energies(&freq, &mut c_band_e, eff_end, c, lm);
    assert_f32_slice_close(
        &rust_band_e,
        &c_band_e,
        1e-4,
        "compute_band_energies(sine, mono)",
    );
}

#[test]
fn compute_band_energies_noise_mono() {
    let m = CeltMode::get_mode();
    let lm = 0usize;
    let mm = 1usize << lm;
    let eff_end = m.nb_ebands;
    let c = 1;
    let freq_size = mm * m.ebands[eff_end] as usize;
    let freq = gen_noise(freq_size, 42);

    let mut rust_band_e = vec![0.0f32; eff_end * c];
    let mut c_band_e = vec![0.0f32; eff_end * c];
    bands::compute_band_energies(m, &freq, &mut rust_band_e, eff_end, c, lm as i32);
    c_compute_band_energies(&freq, &mut c_band_e, eff_end, c, lm);
    assert_f32_slice_close(
        &rust_band_e,
        &c_band_e,
        1e-4,
        "compute_band_energies(noise, mono)",
    );
}

// ── normalise_bands + denormalise_bands roundtrip ──

#[test]
fn normalise_denormalise_roundtrip() {
    let m = CeltMode::get_mode();
    let lm = 0usize;
    let mm = 1usize << lm;
    let eff_end = m.nb_ebands;
    let c = 1;
    let freq_size = mm * m.ebands[eff_end] as usize;
    let freq = gen_noise(freq_size, 42);

    let mut band_e = vec![0.0f32; eff_end * c];
    bands::compute_band_energies(m, &freq, &mut band_e, eff_end, c, lm as i32);

    let mut rust_norm = vec![0.0f32; freq_size];
    let mut c_norm = vec![0.0f32; freq_size];
    bands::normalise_bands(m, &freq, &mut rust_norm, &band_e, eff_end, c, mm);
    c_normalise_bands(&freq, &mut c_norm, &band_e, eff_end, c, mm);
    assert_f32_slice_close(&rust_norm, &c_norm, 1e-4, "normalise_bands");

    let mut band_log_e = vec![0.0f32; eff_end * c];
    bands::amp2_log2(m, eff_end, eff_end, &band_e, &mut band_log_e, c);

    let full_n = mm * m.short_mdct_size;
    let mut rust_freq = vec![0.0f32; full_n * c];
    let mut c_freq = vec![0.0f32; full_n * c];
    bands::denormalise_bands(
        m,
        &rust_norm,
        &mut rust_freq,
        &band_log_e,
        0,
        eff_end,
        mm,
        1,
        false,
    );
    c_denormalise_bands(&c_norm, &mut c_freq, &band_log_e, 0, eff_end, mm, 1, false);
    assert_f32_slice_close(&rust_freq, &c_freq, 1e-2, "denormalise_bands");
}

// ── anti_collapse ──

#[test]
fn anti_collapse_with_collapsed_bands() {
    let m = CeltMode::get_mode();
    let lm = 0i32;
    let mm = 1usize << lm;
    let c = 1;
    let start = 0;
    let end = m.nb_ebands;
    let size = mm * m.short_mdct_size;

    // Create spectral coefficients — some bands zeroed to trigger anti-collapse
    let mut rust_x = gen_noise(size * c, 42);
    // Zero out bands 5..10 to simulate collapsed bands
    for i in 5..10 {
        let band_start = mm * m.ebands[i] as usize;
        let band_end = mm * m.ebands[i + 1] as usize;
        for x in rust_x.iter_mut().take(band_end).skip(band_start) {
            *x = 0.0;
        }
    }
    let mut c_x = rust_x.clone();

    // Collapse masks: 0 = collapsed, 1 = not collapsed
    let mut rust_masks = vec![1u8; end * c];
    for mask in rust_masks.iter_mut().take(10).skip(5) {
        *mask = 0; // these bands collapsed
    }
    let mut c_masks = rust_masks.clone();

    // Pulses: 0 for collapsed bands
    let mut pulses = vec![8i32; end];
    for pulse in pulses.iter_mut().take(10).skip(5) {
        *pulse = 0;
    }

    // Band energies (current + two previous frames)
    // Anti-collapse always accesses [nb_ebands + i] even for mono, so allocate for 2 channels
    let log_e = gen_noise(end * 2, 77);
    let prev1 = gen_noise(end * 2, 88);
    let prev2 = gen_noise(end * 2, 99);

    let seed = 12345u32;

    bands::anti_collapse(
        m,
        &mut rust_x,
        &rust_masks,
        lm,
        c,
        size,
        start,
        end,
        &log_e,
        &prev1,
        &prev2,
        &pulses,
        seed,
    );
    c_anti_collapse(
        &mut c_x,
        &mut c_masks,
        lm,
        c,
        size,
        start,
        end,
        &log_e,
        &prev1,
        &prev2,
        &pulses,
        seed,
        false,
    );

    assert_f32_slice_close(&rust_x, &c_x, 1e-4, "anti_collapse");
}
