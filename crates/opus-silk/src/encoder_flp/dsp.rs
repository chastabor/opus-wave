// Layer 0: Leaf DSP functions for the float SILK encoder.
// Each function is a faithful port of the corresponding silk/float/*.c file.
// All functions are stateless and operate on f32 slices.

// The Schur and LPC functions need to handle both predict_lpc_order (<=16)
// and shaping_lpc_order (<=24). Use 24 as the max.
const SILK_MAX_ORDER_LPC: usize = crate::nsq::MAX_SHAPE_LPC_ORDER;

// ---- energy_FLP.c ----

/// Compute energy (sum of squares) of a float signal.
/// Returns f64 for accumulation precision (matching C's double return).
pub fn silk_energy_flp(data: &[f32]) -> f64 {
    let mut result: f64 = 0.0;
    let n = data.len();
    let n4 = n & !3;

    // 4x unrolled
    let mut i = 0;
    while i < n4 {
        result += data[i] as f64 * data[i] as f64
            + data[i + 1] as f64 * data[i + 1] as f64
            + data[i + 2] as f64 * data[i + 2] as f64
            + data[i + 3] as f64 * data[i + 3] as f64;
        i += 4;
    }
    while i < n {
        result += data[i] as f64 * data[i] as f64;
        i += 1;
    }
    result
}

// ---- inner_product_FLP.c ----

/// Compute inner product (dot product) of two float signals.
/// Returns f64 for accumulation precision.
pub fn silk_inner_product_flp(data1: &[f32], data2: &[f32]) -> f64 {
    let n = data1.len().min(data2.len());
    let mut result: f64 = 0.0;
    let n4 = n & !3;

    let mut i = 0;
    while i < n4 {
        result += data1[i] as f64 * data2[i] as f64
            + data1[i + 1] as f64 * data2[i + 1] as f64
            + data1[i + 2] as f64 * data2[i + 2] as f64
            + data1[i + 3] as f64 * data2[i + 3] as f64;
        i += 4;
    }
    while i < n {
        result += data1[i] as f64 * data2[i] as f64;
        i += 1;
    }
    result
}

// ---- autocorrelation_FLP.c ----

/// Compute autocorrelation of a float signal.
/// results[k] = inner_product(input, input+k) for k = 0..correlation_count-1.
pub fn silk_autocorrelation_flp(results: &mut [f32], input: &[f32], correlation_count: usize) {
    let n = input.len();
    let count = correlation_count.min(n);
    for i in 0..count {
        results[i] = silk_inner_product_flp(input, &input[i..]) as f32;
    }
}

// ---- schur_FLP.c ----

/// Schur recursion: autocorrelation → reflection coefficients + residual energy.
/// Returns residual energy. Fills refl_coef[0..order].
pub fn silk_schur_flp(refl_coef: &mut [f32], auto_corr: &[f32], order: usize) -> f32 {
    let mut c = [[0.0f64; 2]; SILK_MAX_ORDER_LPC + 1];

    // Copy correlations
    for k in 0..=order {
        c[k][0] = auto_corr[k] as f64;
        c[k][1] = auto_corr[k] as f64;
    }

    for k in 0..order {
        // Reflection coefficient
        let rc_tmp = -c[k + 1][0] / c[0][1].max(1e-9);
        refl_coef[k] = rc_tmp as f32;

        // Update correlations
        for n in 0..(order - k) {
            let ctmp1 = c[n + k + 1][0];
            let ctmp2 = c[n][1];
            c[n + k + 1][0] = ctmp1 + ctmp2 * rc_tmp;
            c[n][1] = ctmp2 + ctmp1 * rc_tmp;
        }
    }

    c[0][1] as f32
}

// ---- k2a_FLP.c ----

/// Convert reflection coefficients to LPC prediction coefficients.
/// a[0..order] is filled with the LPC coefficients.
pub fn silk_k2a_flp(a: &mut [f32], rc: &[f32], order: usize) {
    for k in 0..order {
        let rck = rc[k];
        for n in 0..((k + 1) >> 1) {
            let tmp1 = a[n];
            let tmp2 = a[k - n - 1];
            a[n] = tmp1 + tmp2 * rck;
            a[k - n - 1] = tmp2 + tmp1 * rck;
        }
        a[k] = -rck;
    }
}

// ---- bwexpander_FLP.c ----

/// Bandwidth expansion (chirp) on AR filter coefficients.
pub fn silk_bwexpander_flp(ar: &mut [f32], d: usize, chirp: f32) {
    let mut cfac = chirp;
    for item in ar.iter_mut().take(d.saturating_sub(1)) {
        *item *= cfac;
        cfac *= chirp;
    }
    if d > 0 {
        ar[d - 1] *= cfac;
    }
}

// ---- apply_sine_window_FLP.c ----

/// Apply a sine window to a signal.
/// win_type: 1 = rising (starts from 0), 2 = falling (starts from 1).
/// length must be a multiple of 4.
pub fn silk_apply_sine_window_flp(px_win: &mut [f32], px: &[f32], win_type: i32, length: usize) {
    let freq = std::f32::consts::PI / (length as f32 + 1.0);
    let c = 2.0f32 - freq * freq;

    let (mut s0, mut s1) = if win_type < 2 {
        (0.0f32, freq) // rising: start from 0
    } else {
        (1.0f32, 0.5 * c) // falling: start from 1
    };

    // Recursive sine generation, 4 samples at a time
    let mut k = 0;
    while k + 3 < length && k + 3 < px.len() && k + 3 < px_win.len() {
        px_win[k] = px[k] * 0.5 * (s0 + s1);
        px_win[k + 1] = px[k + 1] * s1;
        s0 = c * s1 - s0;
        px_win[k + 2] = px[k + 2] * 0.5 * (s1 + s0);
        px_win[k + 3] = px[k + 3] * s0;
        s1 = c * s0 - s1;
        k += 4;
    }
}

// ---- scale_copy_vector_FLP.c ----

/// Scale and copy a float vector: data_out[i] = gain * data_in[i].
pub fn silk_scale_copy_vector_flp(data_out: &mut [f32], data_in: &[f32], gain: f32, len: usize) {
    let n = len.min(data_out.len()).min(data_in.len());
    let n4 = n & !3;

    let mut i = 0;
    while i < n4 {
        data_out[i] = gain * data_in[i];
        data_out[i + 1] = gain * data_in[i + 1];
        data_out[i + 2] = gain * data_in[i + 2];
        data_out[i + 3] = gain * data_in[i + 3];
        i += 4;
    }
    while i < n {
        data_out[i] = gain * data_in[i];
        i += 1;
    }
}

// ---- LPC_analysis_filter_FLP.c ----

/// FIR LPC analysis filter: computes prediction residual.
/// r_lpc[i] = s[i] - sum(pred_coef[j] * s[i-j-1]) for i >= order.
/// r_lpc[0..order] is zeroed.
pub fn silk_lpc_analysis_filter_flp(
    r_lpc: &mut [f32],
    pred_coef: &[f32],
    s: &[f32],
    length: usize,
    order: usize,
) {
    // Generic implementation (handles any order)
    for ix in order..length {
        let mut lpc_pred: f32 = 0.0;
        for j in 0..order {
            lpc_pred += s[ix - j - 1] * pred_coef[j];
        }
        r_lpc[ix] = s[ix] - lpc_pred;
    }

    // Zero first Order samples
    for i in 0..order.min(r_lpc.len()) {
        r_lpc[i] = 0.0;
    }
}

// ---- LPC_inv_pred_gain_FLP.c ----

/// Compute inverse prediction gain from LPC coefficients.
/// Returns 0.0 if the filter is unstable.
pub fn silk_lpc_inverse_pred_gain_flp(a: &[f32], order: usize) -> f32 {
    const MAX_PRED_POWER_GAIN: f64 = crate::MAX_PREDICTION_POWER_GAIN as f64;

    let mut atmp = [0.0f32; SILK_MAX_ORDER_LPC];
    atmp[..order].copy_from_slice(&a[..order]);

    let mut inv_gain: f64 = 1.0;
    for k in (1..order).rev() {
        let rc = -(atmp[k] as f64);
        let rc_mult1 = 1.0 - rc * rc;
        inv_gain *= rc_mult1;
        if inv_gain * MAX_PRED_POWER_GAIN < 1.0 {
            return 0.0;
        }
        let rc_mult2 = 1.0 / rc_mult1;
        for n in 0..((k + 1) >> 1) {
            let tmp1 = atmp[n] as f64;
            let tmp2 = atmp[k - n - 1] as f64;
            atmp[n] = ((tmp1 - tmp2 * rc) * rc_mult2) as f32;
            atmp[k - n - 1] = ((tmp2 - tmp1 * rc) * rc_mult2) as f32;
        }
    }
    let rc = -(atmp[0] as f64);
    let rc_mult1 = 1.0 - rc * rc;
    inv_gain *= rc_mult1;
    if inv_gain * MAX_PRED_POWER_GAIN < 1.0 {
        return 0.0;
    }
    inv_gain as f32
}

// ---- warped_autocorrelation_FLP.c ----

/// Warped autocorrelation using allpass filter chain.
/// Port of silk_warped_autocorrelation_FLP.
pub fn silk_warped_autocorrelation_flp(
    corr: &mut [f32],
    input: &[f32],
    warping: f32,
    length: usize,
    order: usize,
) {
    let mut state = [0.0f64; SILK_MAX_ORDER_LPC + 1];
    let mut c = [0.0f64; SILK_MAX_ORDER_LPC + 1];

    for item in input.iter().take(length) {
        let mut tmp1 = *item as f64;
        // Loop over allpass sections (step by 2)
        let mut i = 0;
        while i < order {
            // Output of first allpass section
            let tmp2 = state[i] + warping as f64 * state[i + 1] - warping as f64 * tmp1;
            state[i] = tmp1;
            c[i] += state[0] * tmp1;
            // Output of second allpass section
            tmp1 = state[i + 1] + warping as f64 * state[i + 2] - warping as f64 * tmp2;
            state[i + 1] = tmp2;
            c[i + 1] += state[0] * tmp2;
            i += 2;
        }
        state[order] = tmp1;
        c[order] += state[0] * tmp1;
    }

    for i in 0..=order {
        corr[i] = c[i] as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_basic() {
        let data = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let e = silk_energy_flp(&data);
        assert!((e - 55.0).abs() < 1e-6);
    }

    #[test]
    fn test_inner_product_basic() {
        let a = [1.0f32, 2.0, 3.0];
        let b = [4.0f32, 5.0, 6.0];
        let ip = silk_inner_product_flp(&a, &b);
        assert!((ip - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_schur_identity() {
        // Autocorrelation of a simple signal
        let auto_corr = [10.0f32, 5.0, 2.0, 1.0];
        let mut rc = [0.0f32; 3];
        let nrg = silk_schur_flp(&mut rc, &auto_corr, 3);
        // Residual energy should be positive and less than auto_corr[0]
        assert!(nrg > 0.0);
        assert!(nrg <= auto_corr[0]);
    }

    #[test]
    fn test_k2a_from_rc() {
        // Simple reflection coefficients
        let rc = [0.5f32, -0.3];
        let mut a = [0.0f32; 2];
        silk_k2a_flp(&mut a, &rc, 2);
        // First coefficient should be -rc[0] adjusted by rc[1]
        assert!((a[1] - 0.3).abs() < 1e-6);
    }

    #[test]
    fn test_bwexpander() {
        let mut ar = [1.0f32, 1.0, 1.0];
        silk_bwexpander_flp(&mut ar, 3, 0.5);
        assert!((ar[0] - 0.5).abs() < 1e-6);
        assert!((ar[1] - 0.25).abs() < 1e-6);
        assert!((ar[2] - 0.125).abs() < 1e-6);
    }

    #[test]
    fn test_scale_copy() {
        let input = [1.0f32, 2.0, 3.0, 4.0, 5.0];
        let mut output = [0.0f32; 5];
        silk_scale_copy_vector_flp(&mut output, &input, 2.0, 5);
        assert!((output[0] - 2.0).abs() < 1e-6);
        assert!((output[4] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_lpc_analysis_filter() {
        // Simple test: constant signal with a[0]=1 should give zero residual
        let s = [5.0f32; 20];
        let pred_coef = [1.0f32, 0.0, 0.0, 0.0];
        let mut r = [0.0f32; 20];
        silk_lpc_analysis_filter_flp(&mut r, &pred_coef, &s, 20, 4);
        // After order samples, residual should be s[i] - 1.0*s[i-1] = 0
        for (i, &val) in r.iter().enumerate().skip(4) {
            assert!(val.abs() < 1e-6, "r[{}] = {}", i, val);
        }
    }

    #[test]
    fn test_inv_pred_gain_stable() {
        // A simple stable filter
        let a = [0.5f32, -0.2, 0.0, 0.0];
        let gain = silk_lpc_inverse_pred_gain_flp(&a, 4);
        assert!(gain > 0.0);
        assert!(gain <= 1.0);
    }
}
