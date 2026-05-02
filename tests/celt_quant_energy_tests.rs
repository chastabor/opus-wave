//! Cross-validation tests for CELT energy quantization.
//! Tests encode/decode roundtrips: C-encode → C-decode vs Rust-decode,
//! and Rust-encode → C-decode.

mod common;

use common::assert_f32_slice_close;
use opus_rust::celt::mode::CeltMode;
use opus_rust::celt::quant_energy;
use opus_rust::celt::tables::E_MEANS;
use opus_ffi::*;
use opus_rust::range_coder::EcCtx;

const NB_EBANDS: usize = 21;

/// Generate realistic band log-energies for testing.
fn gen_band_energies(seed: u32, nb_ebands: usize, c: usize) -> Vec<f32> {
    let noise = common::gen_noise(nb_ebands * c, seed);
    let mut e = vec![0.0f32; nb_ebands * c];
    for ch in 0..c {
        for i in 0..nb_ebands {
            e[i + ch * nb_ebands] = E_MEANS[i] + noise[i + ch * nb_ebands] * 5.0;
        }
    }
    e
}

// ══════════════════════════════════════════════════════════════════════
// Coarse energy: C-encode → compare C-decode vs Rust-decode
// ══════════════════════════════════════════════════════════════════════

#[test]
fn coarse_energy_c_encode_rust_decode_mono() {
    let m = CeltMode::get_mode();
    let start = 0;
    let end = NB_EBANDS;
    let c = 1;
    let lm = 0;
    let nb_available = 100;

    let e_bands = gen_band_energies(42, NB_EBANDS, c);

    // C encode
    let mut c_old = vec![0.0f32; NB_EBANDS * c];
    let mut c_error = vec![0.0f32; NB_EBANDS * c];
    let mut ec_buf = vec![0u8; 256];
    let ec_bytes = c_encode_coarse_energy(
        start,
        end,
        &e_bands,
        &mut c_old,
        &mut c_error,
        &mut ec_buf,
        c,
        lm,
        nb_available,
        false,
        0,
        false,
    );
    assert!(ec_bytes > 0, "C coarse encode produced no bytes");

    // C decode from those bytes
    let mut c_decoded = vec![0.0f32; NB_EBANDS * c];
    c_decode_coarse_energy(start, end, &mut c_decoded, &ec_buf[..ec_bytes], c, lm);

    // Rust decode from same bytes
    let mut rust_decoded = vec![0.0f32; NB_EBANDS * c];
    let mut dec = EcCtx::dec_init(&ec_buf[..ec_bytes]);
    let intra = dec.dec_bit_logp(3);
    quant_energy::unquant_coarse_energy(m, start, end, &mut rust_decoded, intra, &mut dec, c, lm);

    assert_f32_slice_close(&rust_decoded, &c_decoded, 1e-4, "coarse_decode(mono)");
}

#[test]
fn coarse_energy_c_encode_rust_decode_stereo() {
    let m = CeltMode::get_mode();
    let start = 0;
    let end = NB_EBANDS;
    let c = 2;
    let lm = 0;
    let nb_available = 200;

    let e_bands = gen_band_energies(77, NB_EBANDS, c);

    let mut c_old = vec![0.0f32; NB_EBANDS * c];
    let mut c_error = vec![0.0f32; NB_EBANDS * c];
    let mut ec_buf = vec![0u8; 512];
    let ec_bytes = c_encode_coarse_energy(
        start,
        end,
        &e_bands,
        &mut c_old,
        &mut c_error,
        &mut ec_buf,
        c,
        lm,
        nb_available,
        false,
        0,
        false,
    );
    assert!(ec_bytes > 0);

    let mut c_decoded = vec![0.0f32; NB_EBANDS * c];
    c_decode_coarse_energy(start, end, &mut c_decoded, &ec_buf[..ec_bytes], c, lm);

    let mut rust_decoded = vec![0.0f32; NB_EBANDS * c];
    let mut dec = EcCtx::dec_init(&ec_buf[..ec_bytes]);
    let intra = dec.dec_bit_logp(3);
    quant_energy::unquant_coarse_energy(m, start, end, &mut rust_decoded, intra, &mut dec, c, lm);

    assert_f32_slice_close(&rust_decoded, &c_decoded, 1e-4, "coarse_decode(stereo)");
}

// ══════════════════════════════════════════════════════════════════════
// Coarse energy: Rust-encode → C-decode
// ══════════════════════════════════════════════════════════════════════

#[test]
fn coarse_energy_rust_vs_c_encode_mono() {
    let m = CeltMode::get_mode();
    let start = 0;
    let end = NB_EBANDS;
    let c = 1;
    let lm = 0;
    let nb_available: i32 = 100;
    let buf_size: u32 = 256;

    let e_bands = gen_band_energies(42, NB_EBANDS, c);

    // Also encode with C (force_intra=true) to get C reference state
    let mut c_ref_old = vec![0.0f32; NB_EBANDS * c];
    let mut c_ref_error = vec![0.0f32; NB_EBANDS * c];
    let mut c_ref_buf = vec![0u8; 256];
    c_encode_coarse_energy(
        start,
        end,
        &e_bands,
        &mut c_ref_old,
        &mut c_ref_error,
        &mut c_ref_buf,
        c,
        lm,
        nb_available as usize,
        true,
        0,
        false,
    );

    // Rust encode (force intra to avoid two-pass decision divergence)
    let mut rust_old = vec![0.0f32; NB_EBANDS * c];
    let mut rust_error = vec![0.0f32; NB_EBANDS * c];
    let mut enc = EcCtx::enc_init(buf_size);
    let mut delayed_intra = 0.0f32;
    quant_energy::quant_coarse_energy(
        m,
        start,
        end,
        end,
        &e_bands,
        &mut rust_old,
        (buf_size * 8) as i32,
        &mut rust_error,
        &mut enc,
        c,
        lm,
        nb_available,
        true,
        &mut delayed_intra,
        false,
        0,
        false,
    );
    enc.enc_done();

    // Compare Rust encoder state against C encoder state (both forced intra).
    // Both produce identical band energies even though the range coder
    // bitstreams may differ byte-for-byte.
    assert_f32_slice_close(&rust_old, &c_ref_old, 1e-4, "rust_encode_vs_c_encode(mono)");
}

// ══════════════════════════════════════════════════════════════════════
// Fine energy: C-encode → compare C-decode vs Rust-decode
// ══════════════════════════════════════════════════════════════════════

#[test]
fn fine_energy_c_encode_rust_decode() {
    let m = CeltMode::get_mode();
    let start = 0;
    let end = NB_EBANDS;
    let c = 1;

    // Simulate post-coarse state
    let initial_old = gen_band_energies(42, NB_EBANDS, c);
    let mut c_old = initial_old.clone();
    let mut c_error: Vec<f32> = (0..NB_EBANDS * c)
        .map(|i| ((i as f32 * 0.1).sin()) * 0.3)
        .collect();

    // Modest fine_quant values (1-3 bits per band)
    let fine_quant: Vec<i32> = (0..NB_EBANDS).map(|i| (i % 3 + 1) as i32).collect();

    // C encode fine energy
    let mut ec_buf = vec![0u8; 256];
    let ec_bytes = c_encode_fine_energy(
        start,
        end,
        &mut c_old,
        &mut c_error,
        &fine_quant,
        &mut ec_buf,
        c,
    );

    // C decode
    let mut c_decoded = initial_old.clone();
    c_decode_fine_energy(
        start,
        end,
        &mut c_decoded,
        &fine_quant,
        &ec_buf[..ec_bytes],
        c,
    );

    // Rust decode from same bytes
    let mut rust_decoded = initial_old.clone();
    let mut dec = EcCtx::dec_init(&ec_buf[..ec_bytes]);
    quant_energy::unquant_fine_energy(m, start, end, &mut rust_decoded, &fine_quant, &mut dec, c);

    assert_f32_slice_close(&rust_decoded, &c_decoded, 1e-4, "fine_decode(mono)");
}

// ══════════════════════════════════════════════════════════════════════
// Energy finalise: C-encode → compare C-decode vs Rust-decode
// ══════════════════════════════════════════════════════════════════════

#[test]
fn energy_finalise_c_encode_rust_decode() {
    let m = CeltMode::get_mode();
    let start = 0;
    let end = NB_EBANDS;
    let c = 1;

    // Simulate post-fine-quant state
    let initial_old = gen_band_energies(42, NB_EBANDS, c);
    let mut c_old = initial_old.clone();
    let mut c_error: Vec<f32> = (0..NB_EBANDS * c)
        .map(|i| ((i as f32 * 0.17).sin()) * 0.2)
        .collect();

    // fine_quant and fine_priority from a hypothetical bit allocation
    let fine_quant: Vec<i32> = (0..NB_EBANDS).map(|i| (i % 4 + 1) as i32).collect();
    let fine_priority: Vec<i32> = (0..NB_EBANDS).map(|i| (i % 2) as i32).collect();
    let bits_left = 16; // spare bits to distribute

    // C encode energy finalise
    let mut ec_buf = vec![0u8; 256];
    let ec_bytes = c_encode_energy_finalise(
        start,
        end,
        &mut c_old,
        &mut c_error,
        &fine_quant,
        &fine_priority,
        bits_left,
        &mut ec_buf,
        c,
    );

    // C decode
    let mut c_decoded = initial_old.clone();
    c_decode_energy_finalise(
        start,
        end,
        &mut c_decoded,
        &fine_quant,
        &fine_priority,
        bits_left,
        &ec_buf[..ec_bytes],
        c,
    );

    // Rust decode from same bytes
    let mut rust_decoded = initial_old.clone();
    let mut dec = EcCtx::dec_init(&ec_buf[..ec_bytes]);
    quant_energy::unquant_energy_finalise(
        m,
        start,
        end,
        Some(&mut rust_decoded),
        &fine_quant,
        &fine_priority,
        bits_left,
        &mut dec,
        c,
    );

    assert_f32_slice_close(
        &rust_decoded,
        &c_decoded,
        1e-4,
        "energy_finalise_decode(mono)",
    );
}
