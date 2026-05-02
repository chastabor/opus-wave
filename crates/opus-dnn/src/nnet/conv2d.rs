use super::activations::compute_activation;
use super::{Activation, Conv2dLayer};

const MAX_CONV2D_INPUTS: usize = 8192;

/// Generic 2D convolution for float weights.
/// Matches C `conv2d_float` from nnet_arch.h.
///
/// Input layout: [ktime x in_channels x in_stride] where in_stride = height + kheight - 1.
/// Kernel layout: [out_channels x in_channels x ktime x kheight].
/// Output layout: [out_channels x hstride] (only first `height` elements per channel used).
fn conv2d_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    ktime: usize,
    kheight: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        for v in out[i * hstride..i * hstride + height].iter_mut() {
            *v = 0.0;
        }
        for m in 0..in_channels {
            for t in 0..ktime {
                for h in 0..kheight {
                    let w = weights
                        [i * in_channels * ktime * kheight + m * ktime * kheight + t * kheight + h];
                    for j in 0..height {
                        out[i * hstride + j] +=
                            w * input[t * in_channels * in_stride + m * in_stride + j + h];
                    }
                }
            }
        }
    }
}

/// Specialized 3x3 convolution matching C `conv2d_3x3_float`.
/// Fully unrolled inner loop for ktime=3, kheight=3.
fn conv2d_3x3_float(
    out: &mut [f32],
    weights: &[f32],
    in_channels: usize,
    out_channels: usize,
    input: &[f32],
    height: usize,
    hstride: usize,
) {
    let kheight = 3;
    let ktime = 3;
    let in_stride = height + kheight - 1;
    for i in 0..out_channels {
        for v in out[i * hstride..i * hstride + height].iter_mut() {
            *v = 0.0;
        }
        for m in 0..in_channels {
            let wbase = i * in_channels * ktime * kheight + m * ktime * kheight;
            let in0 = m * in_stride;
            let in1 = in_channels * in_stride + m * in_stride;
            let in2 = 2 * in_channels * in_stride + m * in_stride;
            for j in 0..height {
                out[i * hstride + j] += weights[wbase] * input[in0 + j]
                    + weights[wbase + 1] * input[in0 + j + 1]
                    + weights[wbase + 2] * input[in0 + j + 2]
                    + weights[wbase + 3] * input[in1 + j]
                    + weights[wbase + 4] * input[in1 + j + 1]
                    + weights[wbase + 5] * input[in1 + j + 2]
                    + weights[wbase + 6] * input[in2 + j]
                    + weights[wbase + 7] * input[in2 + j + 1]
                    + weights[wbase + 8] * input[in2 + j + 2];
            }
        }
    }
}

/// Compute 2D convolution with memory management for temporal causality.
/// Matches C `compute_conv2d_c` from nnet_arch.h.
///
/// `mem` holds `(ktime-1)` previous time steps of input. On each call, the
/// memory is shifted and the new input is appended, then the convolution is
/// computed over the full temporal window.
pub fn compute_conv2d(
    conv: &Conv2dLayer,
    out: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    height: usize,
    hstride: usize,
    activation: Activation,
) {
    let weights = conv
        .float_weights
        .as_ref()
        .expect("conv2d requires float_weights");
    let time_stride = conv.in_channels * (height + conv.kheight - 1);
    debug_assert!(conv.ktime * time_stride <= MAX_CONV2D_INPUTS);

    let buf_size = conv.ktime * time_stride;
    let mut in_buf = [0.0f32; MAX_CONV2D_INPUTS];
    let mem_size = (conv.ktime - 1) * time_stride;
    in_buf[..mem_size].copy_from_slice(&mem[..mem_size]);
    in_buf[mem_size..mem_size + time_stride].copy_from_slice(&input[..time_stride]);

    // Shift memory forward.
    mem[..mem_size].copy_from_slice(&in_buf[time_stride..time_stride + mem_size]);

    // Dispatch to specialized or generic convolution.
    if conv.kheight == 3 && conv.ktime == 3 {
        conv2d_3x3_float(
            out,
            weights,
            conv.in_channels,
            conv.out_channels,
            &in_buf[..buf_size],
            height,
            hstride,
        );
    } else {
        conv2d_float(
            out,
            weights,
            conv.in_channels,
            conv.out_channels,
            conv.ktime,
            conv.kheight,
            &in_buf[..buf_size],
            height,
            hstride,
        );
    }

    // Add bias.
    if let Some(ref bias) = conv.bias {
        for i in 0..conv.out_channels {
            for j in 0..height {
                out[i * hstride + j] += bias[i];
            }
        }
    }

    // Apply activation per channel.
    for i in 0..conv.out_channels {
        let start = i * hstride;
        compute_activation(&mut out[start..start + height], activation);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conv2d_float_identity_kernel() {
        // 1 in_channel, 1 out_channel, ktime=1, kheight=1 => identity
        let weights = [1.0f32];
        let input = [1.0, 2.0, 3.0, 4.0]; // height=4
        let mut out = [0.0f32; 4];
        conv2d_float(&mut out, &weights, 1, 1, 1, 1, &input, 4, 4);
        assert_eq!(out, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_compute_conv2d_with_bias() {
        let conv = Conv2dLayer {
            bias: Some(vec![10.0]),
            float_weights: Some(vec![1.0]),
            in_channels: 1,
            out_channels: 1,
            ktime: 1,
            kheight: 1,
        };
        let input = [5.0f32; 4];
        let mut mem = vec![]; // ktime-1 = 0, no memory needed
        let mut out = [0.0f32; 4];
        compute_conv2d(&conv, &mut out, &mut mem, &input, 4, 4, Activation::Linear);
        assert_eq!(out, [15.0, 15.0, 15.0, 15.0]);
    }
}
