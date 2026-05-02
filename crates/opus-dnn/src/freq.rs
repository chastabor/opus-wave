use opus_celt::fft::{KissFftCpx, KissFftState, opus_fft};
use std::sync::LazyLock;

use crate::burg::silk_burg_analysis;

pub const LPC_ORDER: usize = 16;
pub const PREEMPHASIS: f32 = 0.85;

pub const FRAME_SIZE_5MS: usize = 2;
pub const OVERLAP_SIZE_5MS: usize = 2;
pub const TRAINING_OFFSET_5MS: usize = 1;
pub const WINDOW_SIZE_5MS: usize = FRAME_SIZE_5MS + OVERLAP_SIZE_5MS;

pub const FRAME_SIZE: usize = 80 * FRAME_SIZE_5MS;
pub const OVERLAP_SIZE: usize = 80 * OVERLAP_SIZE_5MS;
pub const TRAINING_OFFSET: usize = 80 * TRAINING_OFFSET_5MS;
pub const WINDOW_SIZE: usize = FRAME_SIZE + OVERLAP_SIZE;
pub const FREQ_SIZE: usize = WINDOW_SIZE / 2 + 1;

pub const NB_BANDS: usize = 18;

/// Band edge frequencies in 5ms units (200Hz per band at 16kHz).
/// Matches C `eband5ms` from freq.c.
const EBAND5MS: [usize; NB_BANDS] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40,
];

/// Band energy normalization. Matches C `compensation` from freq.c.
const COMPENSATION: [f32; NB_BANDS] = [
    0.8, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.666667, 0.5, 0.5, 0.5, 0.333333, 0.25, 0.25, 0.2,
    0.166667, 0.173913,
];

/// Pre-computed 320-point FFT state (lazy, computed once on first use).
static KFFT: LazyLock<KissFftState> = LazyLock::new(|| KissFftState::new(WINDOW_SIZE));

/// Pre-computed half-window for overlap (sin^2 window).
/// half_window[i] = sin(pi/2 * sin(pi * i / (2*OVERLAP_SIZE))^2)
/// Matches C `half_window` from lpcnet_tables.c.
static HALF_WINDOW: LazyLock<[f32; OVERLAP_SIZE]> = LazyLock::new(|| {
    let mut w = [0.0f32; OVERLAP_SIZE];
    for (i, w_i) in w.iter_mut().enumerate() {
        let x = std::f64::consts::PI * i as f64 / (2.0 * OVERLAP_SIZE as f64);
        let s = x.sin();
        *w_i = (std::f64::consts::FRAC_PI_2 * s * s).sin() as f32;
    }
    w
});

/// Pre-computed DCT basis matrix (NB_BANDS x NB_BANDS).
/// dct_table[j*NB_BANDS + i] = cos(pi/NB_BANDS * (j+0.5) * i)
/// Matches C `dct_table` from lpcnet_tables.c.
static DCT_TABLE: LazyLock<[f32; NB_BANDS * NB_BANDS]> = LazyLock::new(|| {
    let mut table = [0.0f32; NB_BANDS * NB_BANDS];
    for j in 0..NB_BANDS {
        for i in 0..NB_BANDS {
            table[j * NB_BANDS + i] =
                (std::f64::consts::PI / NB_BANDS as f64 * (j as f64 + 0.5) * i as f64).cos() as f32;
        }
    }
    table
});

/// Compute band energies from FFT spectrum using triangular overlap.
/// Matches C `lpcn_compute_band_energy` from freq.c.
pub fn lpcn_compute_band_energy(band_e: &mut [f32; NB_BANDS], x: &[KissFftCpx]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = EBAND5MS[i] * WINDOW_SIZE_5MS + j;
            let tmp = x[idx].r * x[idx].r + x[idx].i * x[idx].i;
            sum[i] += (1.0 - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    *band_e = sum;
}

/// Compute inverse band energies (1/power) for Burg cepstral analysis.
fn compute_band_energy_inverse(band_e: &mut [f32; NB_BANDS], x: &[KissFftCpx]) {
    let mut sum = [0.0f32; NB_BANDS];
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            let idx = EBAND5MS[i] * WINDOW_SIZE_5MS + j;
            let tmp = x[idx].r * x[idx].r + x[idx].i * x[idx].i;
            let inv = 1.0 / (tmp + 1e-9);
            sum[i] += (1.0 - frac) * inv;
            sum[i + 1] += frac * inv;
        }
    }
    sum[0] *= 2.0;
    sum[NB_BANDS - 1] *= 2.0;
    *band_e = sum;
}

/// Apply analysis window (symmetric sin^2 window on overlap regions).
/// Matches C `apply_window` from freq.c.
pub fn apply_window(x: &mut [f32; WINDOW_SIZE]) {
    let hw = &*HALF_WINDOW;
    for i in 0..OVERLAP_SIZE {
        x[i] *= hw[i];
        x[WINDOW_SIZE - 1 - i] *= hw[i];
    }
}

/// DCT: out[i] = sqrt(2/NB_BANDS) * sum_j(in[j] * dct_table[j*NB_BANDS + i]).
/// Matches C `dct` from freq.c.
pub fn dct(out: &mut [f32; NB_BANDS], input: &[f32; NB_BANDS]) {
    let table = &*DCT_TABLE;
    let scale = (2.0f32 / NB_BANDS as f32).sqrt();
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * table[j * NB_BANDS + i];
        }
        out[i] = sum * scale;
    }
}

/// Inverse DCT. Matches C `idct` from freq.c.
fn idct(out: &mut [f32; NB_BANDS], input: &[f32; NB_BANDS]) {
    let table = &*DCT_TABLE;
    let scale = (2.0f32 / NB_BANDS as f32).sqrt();
    for i in 0..NB_BANDS {
        let mut sum = 0.0f32;
        for j in 0..NB_BANDS {
            sum += input[j] * table[i * NB_BANDS + j];
        }
        out[i] = sum * scale;
    }
}

/// Forward FFT transform for WINDOW_SIZE real input.
/// Matches C `forward_transform` from freq.c.
pub fn forward_transform(out: &mut [KissFftCpx], input: &[f32; WINDOW_SIZE]) {
    let mut x = [KissFftCpx::default(); WINDOW_SIZE];
    let mut y = [KissFftCpx::default(); WINDOW_SIZE];
    for i in 0..WINDOW_SIZE {
        x[i].r = input[i];
        x[i].i = 0.0;
    }
    opus_fft(&KFFT, &x, &mut y);
    out[..FREQ_SIZE].copy_from_slice(&y[..FREQ_SIZE]);
}

/// Inverse FFT transform. Matches C `inverse_transform` from freq.c.
fn inverse_transform(out: &mut [f32; WINDOW_SIZE], input: &[KissFftCpx; FREQ_SIZE]) {
    let mut x = [KissFftCpx::default(); WINDOW_SIZE];
    let mut y = [KissFftCpx::default(); WINDOW_SIZE];
    x[..FREQ_SIZE].copy_from_slice(input);
    // Conjugate symmetry for real-valued output
    for i in FREQ_SIZE..WINDOW_SIZE {
        x[i].r = x[WINDOW_SIZE - i].r;
        x[i].i = -x[WINDOW_SIZE - i].i;
    }
    opus_fft(&KFFT, &x, &mut y);
    out[0] = WINDOW_SIZE as f32 * y[0].r;
    for i in 1..WINDOW_SIZE {
        out[i] = WINDOW_SIZE as f32 * y[WINDOW_SIZE - i].r;
    }
}

/// Interpolate band gains to per-bin gains. Matches C `interp_band_gain`.
fn interp_band_gain(g: &mut [f32; FREQ_SIZE], band_e: &[f32; NB_BANDS]) {
    for v in g.iter_mut() {
        *v = 0.0;
    }
    for i in 0..NB_BANDS - 1 {
        let band_size = (EBAND5MS[i + 1] - EBAND5MS[i]) * WINDOW_SIZE_5MS;
        for j in 0..band_size {
            let frac = j as f32 / band_size as f32;
            g[EBAND5MS[i] * WINDOW_SIZE_5MS + j] = (1.0 - frac) * band_e[i] + frac * band_e[i + 1];
        }
    }
}

/// Levinson-Durbin LPC from autocorrelation. Matches C `lpcn_lpc` from freq.c.
fn lpcn_lpc(lpc: &mut [f32; LPC_ORDER], ac: &[f32; LPC_ORDER + 1]) -> f32 {
    let mut rc_arr = [0.0f32; LPC_ORDER];
    for v in lpc.iter_mut() {
        *v = 0.0;
    }
    let mut error = ac[0];
    if ac[0] == 0.0 {
        return error;
    }
    for i in 0..LPC_ORDER {
        let mut rr = 0.0f32;
        for j in 0..i {
            rr += lpc[j] * ac[i - j];
        }
        rr += ac[i + 1];
        let r = -rr / error;
        rc_arr[i] = r;
        lpc[i] = r;
        for j in 0..(i + 1) >> 1 {
            let tmp1 = lpc[j];
            let tmp2 = lpc[i - 1 - j];
            lpc[j] = tmp1 + r * tmp2;
            lpc[i - 1 - j] = tmp2 + r * tmp1;
        }
        error -= r * r * error;
        if error < 0.001 * ac[0] {
            break;
        }
    }
    let _ = rc_arr;
    error
}

/// Compute LPC from band energies. Matches C `lpc_from_bands` from freq.c.
fn lpc_from_bands(lpc: &mut [f32; LPC_ORDER], ex: &[f32; NB_BANDS]) -> f32 {
    let mut xr = [0.0f32; FREQ_SIZE];
    interp_band_gain(&mut xr, ex);
    xr[FREQ_SIZE - 1] = 0.0;
    let mut x_auto_cpx = [KissFftCpx::default(); FREQ_SIZE];
    for i in 0..FREQ_SIZE {
        x_auto_cpx[i].r = xr[i];
    }
    let mut x_auto = [0.0f32; WINDOW_SIZE];
    inverse_transform(&mut x_auto, &x_auto_cpx);
    let mut ac = [0.0f32; LPC_ORDER + 1];
    ac.copy_from_slice(&x_auto[..LPC_ORDER + 1]);
    // -40 dB noise floor
    ac[0] += ac[0] * 1e-4 + 320.0 / 12.0 / 38.0;
    // Lag windowing
    for (i, ac_i) in ac.iter_mut().enumerate().skip(1) {
        *ac_i *= 1.0 - 6e-5 * (i * i) as f32;
    }
    lpcn_lpc(lpc, &ac)
}

/// Compute LPC from cepstrum via inverse DCT + band energy interpolation.
/// Matches C `lpc_from_cepstrum` from freq.c.
pub fn lpc_from_cepstrum(lpc: &mut [f32; LPC_ORDER], cepstrum: &[f32; NB_BANDS]) -> f32 {
    let mut tmp = *cepstrum;
    tmp[0] += 4.0;
    let mut ex = [0.0f32; NB_BANDS];
    idct(&mut ex, &tmp);
    for i in 0..NB_BANDS {
        ex[i] = 10.0f32.powf(ex[i]) * COMPENSATION[i];
    }
    lpc_from_bands(lpc, &ex)
}

/// Apply bandwidth expansion to LPC coefficients.
/// Matches C `lpc_weighting` from freq.c.
pub fn lpc_weighting(lpc: &mut [f32; LPC_ORDER], gamma: f32) {
    let mut gamma_i = gamma;
    for coeff in lpc.iter_mut() {
        *coeff *= gamma_i;
        gamma_i *= gamma;
    }
}

/// Compute Burg cepstrum for a single sub-frame.
/// Matches C `compute_burg_cepstrum` from freq.c.
fn compute_burg_cepstrum(pcm: &[f32], burg_cepstrum: &mut [f32; NB_BANDS], len: usize) {
    let mut burg_in = [0.0f32; FRAME_SIZE];
    for i in 0..len - 1 {
        burg_in[i] = pcm[i + 1] - PREEMPHASIS * pcm[i];
    }

    let mut burg_lpc = [0.0f32; LPC_ORDER];
    let g = silk_burg_analysis(&mut burg_lpc, &burg_in, 1e-3, len - 1, 1, LPC_ORDER);
    let g = g / (len - 2 * (LPC_ORDER - 1)) as f32;

    let mut x = [0.0f32; WINDOW_SIZE];
    x[0] = 1.0;
    for i in 0..LPC_ORDER {
        x[i + 1] = -burg_lpc[i] * (0.995f32).powi(i as i32 + 1);
    }

    let mut lpc_fft = [KissFftCpx::default(); FREQ_SIZE];
    forward_transform(&mut lpc_fft, &x);

    let mut eburg = [0.0f32; NB_BANDS];
    compute_band_energy_inverse(&mut eburg, &lpc_fft);

    let scale = 0.45 * g * (1.0 / (WINDOW_SIZE as f32 * WINDOW_SIZE as f32 * WINDOW_SIZE as f32));
    for e in eburg.iter_mut() {
        *e *= scale;
    }

    let mut ly = [0.0f32; NB_BANDS];
    let mut log_max = -2.0f32;
    let mut follow = -2.0f32;
    for i in 0..NB_BANDS {
        ly[i] = (1e-2 + eburg[i]).log10();
        ly[i] = ly[i].max(log_max - 8.0).max(follow - 2.5);
        log_max = log_max.max(ly[i]);
        follow = (follow - 2.5).max(ly[i]);
    }

    dct(burg_cepstrum, &ly);
    burg_cepstrum[0] -= 4.0;
}

/// Compute Burg cepstral analysis on two half-frames, then average/diff.
/// Matches C `burg_cepstral_analysis` from freq.c.
pub fn burg_cepstral_analysis(ceps: &mut [f32], x: &[f32]) {
    debug_assert!(ceps.len() >= 2 * NB_BANDS);
    let mut ceps0 = [0.0f32; NB_BANDS];
    let mut ceps1 = [0.0f32; NB_BANDS];
    compute_burg_cepstrum(x, &mut ceps0, FRAME_SIZE / 2);
    compute_burg_cepstrum(&x[FRAME_SIZE / 2..], &mut ceps1, FRAME_SIZE / 2);
    for i in 0..NB_BANDS {
        ceps[i] = 0.5 * (ceps0[i] + ceps1[i]);
        ceps[NB_BANDS + i] = ceps0[i] - ceps1[i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dct_energy_preservation() {
        let input = [1.0f32; NB_BANDS];
        let mut output = [0.0f32; NB_BANDS];
        dct(&mut output, &input);
        // DC component should be sqrt(NB_BANDS)*sqrt(2/NB_BANDS) = sqrt(2*NB_BANDS/NB_BANDS) = sqrt(2)... approximately NB_BANDS * scale
        // Actually for constant input, only out[0] should be nonzero
        assert!(output[0].abs() > 1.0);
        for (i, &val) in output.iter().enumerate().skip(1) {
            assert!(val.abs() < 1e-5, "out[{i}] = {val}");
        }
    }

    #[test]
    fn test_dct_idct_consistency() {
        // The C DCT/IDCT are not exactly orthogonal (uniform sqrt(2/N) scale),
        // so roundtrip has a known factor. Verify the transform is at least
        // invertible up to a constant and produces deterministic output.
        let input = [
            0.0f32, 1.0, -1.0, 0.5, -0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0,
        ];
        let mut dct_out = [0.0f32; NB_BANDS];
        let mut roundtrip = [0.0f32; NB_BANDS];
        dct(&mut dct_out, &input);
        idct(&mut roundtrip, &dct_out);
        // Applying twice should give back input scaled by N/2 * (2/N) = 1
        // for AC terms, but 2x for DC. Verify relative structure is preserved.
        // The non-zero elements should have consistent relative magnitudes.
        let ratio = if roundtrip[1].abs() > 1e-10 {
            roundtrip[1] / input[1]
        } else {
            1.0
        };
        for i in 1..5 {
            if input[i].abs() > 1e-10 {
                let r = roundtrip[i] / input[i];
                assert!(
                    (r - ratio).abs() < 0.01,
                    "inconsistent ratio at {i}: {r} vs {ratio}"
                );
            }
        }
    }

    #[test]
    fn test_apply_window_zeros_edges() {
        let mut x = [1.0f32; WINDOW_SIZE];
        apply_window(&mut x);
        // First sample should be windowed to ~0
        assert!(x[0].abs() < 0.01);
        // Last sample should be windowed to ~0
        assert!(x[WINDOW_SIZE - 1].abs() < 0.01);
        // Middle samples should be unchanged (not in overlap region)
        assert!((x[OVERLAP_SIZE] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_forward_transform_dc() {
        let input = [1.0f32; WINDOW_SIZE];
        let mut out = [KissFftCpx::default(); FREQ_SIZE];
        forward_transform(&mut out, &input);
        // opus_fft is normalized (scales by 1/N), so DC of all-ones = 1.0
        assert!((out[0].r - 1.0).abs() < 0.01, "DC = {}", out[0].r);
        assert!(out[0].i.abs() < 1e-5);
    }

    #[test]
    fn test_lpc_weighting() {
        let mut lpc = [1.0f32; LPC_ORDER];
        lpc_weighting(&mut lpc, 0.9);
        assert!((lpc[0] - 0.9).abs() < 1e-6);
        assert!((lpc[1] - 0.81).abs() < 1e-5);
    }
}
