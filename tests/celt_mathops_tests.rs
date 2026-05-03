//! Cross-validation tests for CELT math functions.
//! Each test calls both the Rust port and the C reference with the same input
//! and asserts matching output.

mod common;

use common::{assert_f32_close, assert_f32_slice_close, gen_noise, gen_sine_vec};
use opus_wave::celt::mathops;
use opus_ffi::*;

// ══════════════════════════════════════════════════════════════════════
// Integer math (exact match)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn bitexact_cos_sweep() {
    for x in (0..=16384i16).step_by(100) {
        let rust = mathops::bitexact_cos(x);
        let c = c_bitexact_cos(x);
        assert_eq!(rust, c, "bitexact_cos({x}): Rust={rust} C={c}");
    }
}

#[test]
fn bitexact_cos_boundaries() {
    for &x in &[0i16, 1, 100, 4096, 8192, 12288, 16384] {
        let rust = mathops::bitexact_cos(x);
        let c = c_bitexact_cos(x);
        assert_eq!(rust, c, "bitexact_cos({x}): Rust={rust} C={c}");
    }
}

#[test]
fn bitexact_log2tan_matches_c() {
    let pairs = [
        (16384, 16384),
        (32767, 1),
        (1, 32767),
        (10000, 20000),
        (30000, 5000),
        (8192, 8192),
    ];
    for &(isin, icos) in &pairs {
        let rust = mathops::bitexact_log2tan(isin, icos);
        let c = c_bitexact_log2tan(isin, icos);
        assert_eq!(
            rust, c,
            "bitexact_log2tan({isin}, {icos}): Rust={rust} C={c}"
        );
    }
}

#[test]
fn isqrt32_matches_c() {
    let vals: &[u32] = &[
        0,
        1,
        2,
        3,
        4,
        9,
        15,
        16,
        100,
        255,
        256,
        1000,
        65536,
        1_000_000,
        u32::MAX,
    ];
    for &val in vals {
        let rust = mathops::isqrt32(val);
        let c = c_isqrt32(val);
        assert_eq!(rust, c, "isqrt32({val}): Rust={rust} C={c}");
    }
}

#[test]
fn celt_lcg_rand_chain() {
    let mut rust_seed = 42u32;
    let mut c_seed = 42u32;
    for i in 0..20 {
        rust_seed = mathops::celt_lcg_rand(rust_seed);
        c_seed = c_celt_lcg_rand(c_seed);
        assert_eq!(rust_seed, c_seed, "celt_lcg_rand iteration {i}");
    }
}

#[test]
fn frac_mul16_matches_c() {
    let pairs: &[(i32, i32)] = &[
        (0, 0),
        (1, 1),
        (16384, 16384),
        (-16384, 16384),
        (16384, -16384),
        (-626, 100),
        (-626, 1000),
        (8277, 5000),
        (-7651, 10000),
        (32767, 32767),
        (-32768, 32767),
    ];
    for &(a, b) in pairs {
        let rust = mathops::frac_mul16(a, b);
        let c = c_frac_mul16(a, b);
        assert_eq!(rust, c, "frac_mul16({a}, {b}): Rust={rust} C={c}");
    }
}

// ══════════════════════════════════════════════════════════════════════
// Float math (tolerance-based)
// ══════════════════════════════════════════════════════════════════════

#[test]
fn celt_exp2_matches_c() {
    let vals = [0.0f32, 1.0, -1.0, 0.5, -0.5, 2.0, 5.0, 10.0, -10.0, -50.0];
    for &x in &vals {
        let rust = mathops::celt_exp2(x);
        let c = c_celt_exp2(x);
        let tol = if x < -50.0 {
            1e-10
        } else {
            c.abs() * 1e-5 + 1e-10
        };
        assert_f32_close(rust, c, tol, &format!("celt_exp2({x})"));
    }
}

#[test]
fn celt_exp2_sweep() {
    for i in -200..=100 {
        let x = i as f32 * 0.1;
        let rust = mathops::celt_exp2(x);
        let c = c_celt_exp2(x);
        let tol = c.abs() * 1e-5 + 1e-10;
        assert_f32_close(rust, c, tol, &format!("celt_exp2({x})"));
    }
}

#[test]
fn celt_log2_matches_c() {
    let vals = [0.001f32, 0.01, 0.1, 0.5, 1.0, 2.0, 10.0, 100.0, 1000.0];
    for &x in &vals {
        let rust = mathops::celt_log2(x);
        let c = c_celt_log2(x);
        assert_f32_close(rust, c, 1e-6, &format!("celt_log2({x})"));
    }
}

#[test]
fn celt_inner_prod_sine() {
    let a = gen_sine_vec(320, 440.0, 48000.0, 0.5);
    let b = gen_sine_vec(320, 440.0, 48000.0, 0.5);
    let rust = mathops::celt_inner_prod(&a, &b, 320);
    let c = c_celt_inner_prod(&a, &b);
    assert_f32_close(rust, c, 1e-3, "inner_prod(sine, sine)");
}

#[test]
fn celt_inner_prod_noise() {
    let a = gen_noise(256, 42);
    let b = gen_noise(256, 123);
    let rust = mathops::celt_inner_prod(&a, &b, 256);
    let c = c_celt_inner_prod(&a, &b);
    assert_f32_close(rust, c, 1e-3, "inner_prod(noise, noise)");
}

#[test]
fn celt_maxabs_matches_c() {
    let signal = gen_sine_vec(320, 1000.0, 48000.0, 0.8);
    let rust = mathops::celt_maxabs(&signal, signal.len());
    let c = c_celt_maxabs16(&signal);
    assert_f32_close(rust, c, 1e-7, "celt_maxabs(sine)");
}

#[test]
fn celt_maxabs_noise() {
    let signal = gen_noise(256, 77);
    let rust = mathops::celt_maxabs(&signal, signal.len());
    let c = c_celt_maxabs16(&signal);
    assert_f32_close(rust, c, 1e-7, "celt_maxabs(noise)");
}

#[test]
fn celt_rcp_matches_c() {
    let vals = [0.5f32, 1.0, 2.0, 10.0, 0.001, 100.0, 0.123456];
    for &x in &vals {
        let rust = mathops::celt_rcp(x);
        let c = c_celt_rcp(x);
        assert_f32_close(rust, c, 1e-7, &format!("celt_rcp({x})"));
    }
}

#[test]
fn renormalise_vector_matches_c() {
    let mut rust_vec = gen_noise(64, 42);
    let mut c_vec = rust_vec.clone();
    mathops::renormalise_vector(&mut rust_vec, 64, 1.0);
    c_renormalise_vector(&mut c_vec, 1.0);
    assert_f32_slice_close(&rust_vec, &c_vec, 1e-5, "renormalise_vector(gain=1.0)");
}

#[test]
fn renormalise_vector_half_gain() {
    let mut rust_vec = gen_noise(32, 99);
    let mut c_vec = rust_vec.clone();
    mathops::renormalise_vector(&mut rust_vec, 32, 0.5);
    c_renormalise_vector(&mut c_vec, 0.5);
    assert_f32_slice_close(&rust_vec, &c_vec, 1e-5, "renormalise_vector(gain=0.5)");
}
