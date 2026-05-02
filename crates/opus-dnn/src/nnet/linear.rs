use super::LinearLayer;

/// Dense float matrix-vector multiply: out = weights * x.
/// Weights are column-major with stride `col_stride`.
/// Matches C `sgemv` from vec.h (generic fallback path).
fn sgemv(out: &mut [f32], weights: &[f32], rows: usize, cols: usize, col_stride: usize, x: &[f32]) {
    if rows & 0xf == 0 {
        sgemv16x1(out, weights, rows, cols, col_stride, x);
    } else if rows & 0x7 == 0 {
        sgemv8x1(out, weights, rows, cols, col_stride, x);
    } else {
        for i in 0..rows {
            out[i] = 0.0;
            for j in 0..cols {
                out[i] += weights[j * col_stride + i] * x[j];
            }
        }
    }
}

/// 16-wide unrolled sgemv. Matches C `sgemv16x1`.
fn sgemv16x1(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }
    let mut i = 0;
    while i < rows {
        for j in 0..cols {
            let w = &weights[j * col_stride + i..];
            let xj = x[j];
            let y = &mut out[i..];
            y[0] += w[0] * xj;
            y[1] += w[1] * xj;
            y[2] += w[2] * xj;
            y[3] += w[3] * xj;
            y[4] += w[4] * xj;
            y[5] += w[5] * xj;
            y[6] += w[6] * xj;
            y[7] += w[7] * xj;
            y[8] += w[8] * xj;
            y[9] += w[9] * xj;
            y[10] += w[10] * xj;
            y[11] += w[11] * xj;
            y[12] += w[12] * xj;
            y[13] += w[13] * xj;
            y[14] += w[14] * xj;
            y[15] += w[15] * xj;
        }
        i += 16;
    }
}

/// 8-wide unrolled sgemv. Matches C `sgemv8x1`.
fn sgemv8x1(
    out: &mut [f32],
    weights: &[f32],
    rows: usize,
    cols: usize,
    col_stride: usize,
    x: &[f32],
) {
    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }
    let mut i = 0;
    while i < rows {
        for j in 0..cols {
            let w = &weights[j * col_stride + i..];
            let xj = x[j];
            let y = &mut out[i..];
            y[0] += w[0] * xj;
            y[1] += w[1] * xj;
            y[2] += w[2] * xj;
            y[3] += w[3] * xj;
            y[4] += w[4] * xj;
            y[5] += w[5] * xj;
            y[6] += w[6] * xj;
            y[7] += w[7] * xj;
        }
        i += 8;
    }
}

/// Sparse float matrix-vector multiply with 8x4 blocking.
/// Matches C `sparse_sgemv8x4` from vec.h.
fn sparse_sgemv8x4(out: &mut [f32], w: &[f32], idx: &[i32], rows: usize, x: &[f32]) {
    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }
    let mut wi = 0usize; // weight index
    let mut ii = 0usize; // idx index
    let mut i = 0;
    while i < rows {
        let cols = idx[ii] as usize;
        ii += 1;
        for _j in 0..cols {
            let pos = idx[ii] as usize;
            ii += 1;
            let xj0 = x[pos];
            let xj1 = x[pos + 1];
            let xj2 = x[pos + 2];
            let xj3 = x[pos + 3];
            let y = &mut out[i..];
            y[0] += w[wi] * xj0;
            y[1] += w[wi + 1] * xj0;
            y[2] += w[wi + 2] * xj0;
            y[3] += w[wi + 3] * xj0;
            y[4] += w[wi + 4] * xj0;
            y[5] += w[wi + 5] * xj0;
            y[6] += w[wi + 6] * xj0;
            y[7] += w[wi + 7] * xj0;

            y[0] += w[wi + 8] * xj1;
            y[1] += w[wi + 9] * xj1;
            y[2] += w[wi + 10] * xj1;
            y[3] += w[wi + 11] * xj1;
            y[4] += w[wi + 12] * xj1;
            y[5] += w[wi + 13] * xj1;
            y[6] += w[wi + 14] * xj1;
            y[7] += w[wi + 15] * xj1;

            y[0] += w[wi + 16] * xj2;
            y[1] += w[wi + 17] * xj2;
            y[2] += w[wi + 18] * xj2;
            y[3] += w[wi + 19] * xj2;
            y[4] += w[wi + 20] * xj2;
            y[5] += w[wi + 21] * xj2;
            y[6] += w[wi + 22] * xj2;
            y[7] += w[wi + 23] * xj2;

            y[0] += w[wi + 24] * xj3;
            y[1] += w[wi + 25] * xj3;
            y[2] += w[wi + 26] * xj3;
            y[3] += w[wi + 27] * xj3;
            y[4] += w[wi + 28] * xj3;
            y[5] += w[wi + 29] * xj3;
            y[6] += w[wi + 30] * xj3;
            y[7] += w[wi + 31] * xj3;
            wi += 32;
        }
        i += 8;
    }
}

/// Maximum input vector size for quantized operations. Matches C `MAX_INPUTS`.
const MAX_INPUTS: usize = 2048;

/// Whether to use unsigned quantization (USE_SU_BIAS) for int8 matmul.
/// - x86: uses unsigned (127 + round(127*x)), matching `vec_avx.h` / `dpbusds` semantics.
/// - ARM: uses signed (round(127*x)), matching `vec_neon.h` / `sdot` semantics.
///
/// This is a compile-time constant matching the C `#ifdef USE_SU_BIAS` pattern.
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
const USE_SU_BIAS: bool = true;

#[cfg(any(target_arch = "aarch64", target_arch = "arm"))]
const USE_SU_BIAS: bool = false;

#[cfg(not(any(
    target_arch = "x86",
    target_arch = "x86_64",
    target_arch = "aarch64",
    target_arch = "arm"
)))]
const USE_SU_BIAS: bool = false; // generic fallback uses signed quantization (matching C vec.h)

/// Quantize float input for int8 matmul, architecture-dependent.
/// x86 (USE_SU_BIAS=true): unsigned byte 127 + round(127*x), range [0, 254].
/// ARM (USE_SU_BIAS=false): signed int8 round(127*x), range [-127, 127].
fn quantize_input(xq_unsigned: &mut [u8], xq_signed: &mut [i8], x: &[f32], cols: usize) {
    if USE_SU_BIAS {
        for i in 0..cols {
            xq_unsigned[i] = (127.0 + (0.5 + 127.0 * x[i]).floor()).clamp(0.0, 255.0) as u8;
        }
    } else {
        for i in 0..cols {
            xq_signed[i] = (0.5 + 127.0 * x[i]).floor() as i8;
        }
    }
}

/// Read a quantized input value as i32 (handles both unsigned and signed paths).
#[inline(always)]
fn read_quantized(xq_unsigned: &[u8], xq_signed: &[i8], idx: usize) -> i32 {
    if USE_SU_BIAS {
        xq_unsigned[idx] as i32 // u8 -> i32: zero-extend (0..255)
    } else {
        xq_signed[idx] as i32 // i8 -> i32: sign-extend (-128..127)
    }
}

/// Quantized int8 dense matrix-vector multiply with 8x4 blocking.
/// On x86, uses unsigned input quantization matching `vec_avx.h` (dpbusds semantics).
/// On ARM, uses signed input quantization matching `vec_neon.h` (sdot semantics).
fn cgemv8x4(out: &mut [f32], w: &[i8], scale: &[f32], rows: usize, cols: usize, x: &[f32]) {
    debug_assert!(cols <= MAX_INPUTS);
    let mut xq_u = [0u8; MAX_INPUTS];
    let mut xq_s = [0i8; MAX_INPUTS];
    quantize_input(&mut xq_u, &mut xq_s, x, cols);

    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }

    let mut wi = 0usize;
    let mut i = 0;
    while i < rows {
        let mut j = 0;
        while j < cols {
            let xj0 = read_quantized(&xq_u, &xq_s, j);
            let xj1 = read_quantized(&xq_u, &xq_s, j + 1);
            let xj2 = read_quantized(&xq_u, &xq_s, j + 2);
            let xj3 = read_quantized(&xq_u, &xq_s, j + 3);
            let y = &mut out[i..];
            y[0] += (w[wi] as i32 * xj0
                + w[wi + 1] as i32 * xj1
                + w[wi + 2] as i32 * xj2
                + w[wi + 3] as i32 * xj3) as f32;
            y[1] += (w[wi + 4] as i32 * xj0
                + w[wi + 5] as i32 * xj1
                + w[wi + 6] as i32 * xj2
                + w[wi + 7] as i32 * xj3) as f32;
            y[2] += (w[wi + 8] as i32 * xj0
                + w[wi + 9] as i32 * xj1
                + w[wi + 10] as i32 * xj2
                + w[wi + 11] as i32 * xj3) as f32;
            y[3] += (w[wi + 12] as i32 * xj0
                + w[wi + 13] as i32 * xj1
                + w[wi + 14] as i32 * xj2
                + w[wi + 15] as i32 * xj3) as f32;
            y[4] += (w[wi + 16] as i32 * xj0
                + w[wi + 17] as i32 * xj1
                + w[wi + 18] as i32 * xj2
                + w[wi + 19] as i32 * xj3) as f32;
            y[5] += (w[wi + 20] as i32 * xj0
                + w[wi + 21] as i32 * xj1
                + w[wi + 22] as i32 * xj2
                + w[wi + 23] as i32 * xj3) as f32;
            y[6] += (w[wi + 24] as i32 * xj0
                + w[wi + 25] as i32 * xj1
                + w[wi + 26] as i32 * xj2
                + w[wi + 27] as i32 * xj3) as f32;
            y[7] += (w[wi + 28] as i32 * xj0
                + w[wi + 29] as i32 * xj1
                + w[wi + 30] as i32 * xj2
                + w[wi + 31] as i32 * xj3) as f32;
            wi += 32;
            j += 4;
        }
        i += 8;
    }

    for i in 0..rows {
        out[i] *= scale[i];
    }
}

/// Sparse quantized int8 matrix-vector multiply with 8x4 blocking.
fn sparse_cgemv8x4(
    out: &mut [f32],
    w: &[i8],
    idx: &[i32],
    scale: &[f32],
    rows: usize,
    cols: usize,
    x: &[f32],
) {
    debug_assert!(cols <= MAX_INPUTS);
    let mut xq_u = [0u8; MAX_INPUTS];
    let mut xq_s = [0i8; MAX_INPUTS];
    quantize_input(&mut xq_u, &mut xq_s, x, cols);

    for v in out[..rows].iter_mut() {
        *v = 0.0;
    }

    let mut wi = 0usize;
    let mut ii = 0usize;
    let mut i = 0;
    while i < rows {
        let colblocks = idx[ii] as usize;
        ii += 1;
        for _j in 0..colblocks {
            let pos = idx[ii] as usize;
            ii += 1;
            let xj0 = read_quantized(&xq_u, &xq_s, pos);
            let xj1 = read_quantized(&xq_u, &xq_s, pos + 1);
            let xj2 = read_quantized(&xq_u, &xq_s, pos + 2);
            let xj3 = read_quantized(&xq_u, &xq_s, pos + 3);
            let y = &mut out[i..];
            y[0] += (w[wi] as i32 * xj0
                + w[wi + 1] as i32 * xj1
                + w[wi + 2] as i32 * xj2
                + w[wi + 3] as i32 * xj3) as f32;
            y[1] += (w[wi + 4] as i32 * xj0
                + w[wi + 5] as i32 * xj1
                + w[wi + 6] as i32 * xj2
                + w[wi + 7] as i32 * xj3) as f32;
            y[2] += (w[wi + 8] as i32 * xj0
                + w[wi + 9] as i32 * xj1
                + w[wi + 10] as i32 * xj2
                + w[wi + 11] as i32 * xj3) as f32;
            y[3] += (w[wi + 12] as i32 * xj0
                + w[wi + 13] as i32 * xj1
                + w[wi + 14] as i32 * xj2
                + w[wi + 15] as i32 * xj3) as f32;
            y[4] += (w[wi + 16] as i32 * xj0
                + w[wi + 17] as i32 * xj1
                + w[wi + 18] as i32 * xj2
                + w[wi + 19] as i32 * xj3) as f32;
            y[5] += (w[wi + 20] as i32 * xj0
                + w[wi + 21] as i32 * xj1
                + w[wi + 22] as i32 * xj2
                + w[wi + 23] as i32 * xj3) as f32;
            y[6] += (w[wi + 24] as i32 * xj0
                + w[wi + 25] as i32 * xj1
                + w[wi + 26] as i32 * xj2
                + w[wi + 27] as i32 * xj3) as f32;
            y[7] += (w[wi + 28] as i32 * xj0
                + w[wi + 29] as i32 * xj1
                + w[wi + 30] as i32 * xj2
                + w[wi + 31] as i32 * xj3) as f32;
            wi += 32;
        }
        i += 8;
    }

    for i in 0..rows {
        out[i] *= scale[i];
    }
}

/// Compute linear layer output: out = W*in + bias (+ diag terms for GRU).
/// Matches C `compute_linear_c` from nnet_arch.h.
///
/// Dispatches to the appropriate kernel based on which weight format is present:
/// - `float_weights` + `weights_idx` => sparse float (sparse_sgemv8x4)
/// - `float_weights` alone => dense float (sgemv)
/// - `weights` (int8) + `weights_idx` => sparse quantized (sparse_cgemv8x4)
/// - `weights` (int8) alone => dense quantized (cgemv8x4)
/// - neither => zero output
pub fn compute_linear(layer: &LinearLayer, out: &mut [f32], input: &[f32]) {
    let m = layer.nb_inputs;
    let n = layer.nb_outputs;
    debug_assert!(out.len() >= n);
    debug_assert!(input.len() >= m);

    if let Some(ref float_weights) = layer.float_weights {
        if let Some(ref idx) = layer.weights_idx {
            sparse_sgemv8x4(out, float_weights, idx, n, input);
        } else {
            sgemv(out, float_weights, n, m, n, input);
        }
    } else if let Some(ref weights) = layer.weights {
        let scale = layer.scale.as_ref().expect("int8 weights require scale");
        if let Some(ref idx) = layer.weights_idx {
            sparse_cgemv8x4(out, weights, idx, scale, n, m, input);
        } else {
            cgemv8x4(out, weights, scale, n, m, input);
        }
    } else {
        for v in out[..n].iter_mut() {
            *v = 0.0;
        }
    }

    // Select bias: on USE_SU_BIAS architectures (x86), use subias instead of bias
    // ONLY when the int8 path was actually used (not when float_weights took priority).
    // Mirrors C: the `#ifdef USE_SU_BIAS` reassignment is inside the int8 branch only.
    let used_int8_path = layer.float_weights.is_none() && layer.weights.is_some();
    let effective_bias = if USE_SU_BIAS && used_int8_path {
        layer.subias.as_deref().or(layer.bias.as_deref())
    } else {
        layer.bias.as_deref()
    };
    if let Some(bias) = effective_bias {
        for i in 0..n {
            out[i] += bias[i];
        }
    }

    if let Some(ref diag) = layer.diag {
        debug_assert_eq!(3 * m, n);
        for i in 0..m {
            out[i] += diag[i] * input[i];
            out[i + m] += diag[i + m] * input[i];
            out[i + 2 * m] += diag[i + 2 * m] * input[i];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgemv_simple() {
        // 2x3 matrix (column-major with col_stride=2):
        // [[1, 3, 5],
        //  [2, 4, 6]]
        // x = [1, 1, 1] => out = [9, 12]
        let weights = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        let mut out = [0.0f32; 2];
        sgemv(&mut out, &weights, 2, 3, 2, &x);
        assert!((out[0] - 9.0).abs() < 1e-6);
        assert!((out[1] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_linear_float_dense() {
        // 8x8 identity-like layer
        let mut weights = vec![0.0f32; 64];
        for i in 0..8 {
            weights[i * 8 + i] = 1.0;
        }
        let bias = vec![0.5f32; 8];
        let layer = LinearLayer {
            bias: Some(bias),
            subias: None,
            weights: None,
            float_weights: Some(weights),
            weights_idx: None,
            diag: None,
            scale: None,
            nb_inputs: 8,
            nb_outputs: 8,
        };
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut out = [0.0f32; 8];
        compute_linear(&layer, &mut out, &input);
        for i in 0..8 {
            assert!((out[i] - (input[i] + 0.5)).abs() < 1e-5, "mismatch at {i}");
        }
    }

    #[test]
    fn test_compute_linear_no_weights() {
        let layer = LinearLayer::new(4, 4);
        let input = [1.0, 2.0, 3.0, 4.0];
        let mut out = [99.0f32; 4];
        compute_linear(&layer, &mut out, &input);
        assert_eq!(out, [0.0, 0.0, 0.0, 0.0]);
    }
}
