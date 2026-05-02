use super::activations::compute_activation;
use super::linear::compute_linear;
use super::{Activation, LinearLayer};

/// Maximum RNN neuron count across all models.
/// Matches C `MAX_RNN_NEURONS_ALL` (derived from model constants).
const MAX_RNN_NEURONS: usize = 512;

/// Maximum conv1d input size. Matches C `MAX_CONV_INPUTS_ALL`.
const MAX_CONV_INPUTS: usize = 4096;

/// Maximum GLU/gated activation size. Matches C `MAX_INPUTS`.
const MAX_INPUTS: usize = 2048;

/// Dense layer: linear transform + activation.
/// Matches C `compute_generic_dense` from nnet.c.
pub fn compute_generic_dense(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: Activation,
) {
    compute_linear(layer, output, input);
    compute_activation(&mut output[..layer.nb_outputs], activation);
}

/// GRU (Gated Recurrent Unit) update.
/// Matches C `compute_generic_gru` from nnet.c.
///
/// `input_weights` maps input -> [z, r, h] (3*N outputs).
/// `recurrent_weights` maps state -> [z, r, h] (3*N outputs).
/// `state` is updated in-place (N elements).
pub fn compute_generic_gru(
    input_weights: &LinearLayer,
    recurrent_weights: &LinearLayer,
    state: &mut [f32],
    input: &[f32],
) {
    debug_assert_eq!(
        3 * recurrent_weights.nb_inputs,
        recurrent_weights.nb_outputs
    );
    debug_assert_eq!(input_weights.nb_outputs, recurrent_weights.nb_outputs);
    let n = recurrent_weights.nb_inputs;
    debug_assert!(n <= MAX_RNN_NEURONS);

    let mut zrh = [0.0f32; 3 * MAX_RNN_NEURONS];
    let mut recur = [0.0f32; 3 * MAX_RNN_NEURONS];

    compute_linear(input_weights, &mut zrh[..3 * n], input);
    compute_linear(recurrent_weights, &mut recur[..3 * n], state);

    for i in 0..2 * n {
        zrh[i] += recur[i];
    }
    compute_activation(&mut zrh[..2 * n], Activation::Sigmoid);

    // h (candidate): h[i] += recur[2*N+i] * r[i], then tanh
    for i in 0..n {
        zrh[2 * n + i] += recur[2 * n + i] * zrh[n + i];
    }
    compute_activation(&mut zrh[2 * n..3 * n], Activation::Tanh);

    for i in 0..n {
        let z = zrh[i];
        let h = zrh[2 * n + i];
        state[i] = z * state[i] + (1.0 - z) * h;
    }
}

/// Gated Linear Unit: output = input * sigmoid(W * input).
/// Matches C `compute_glu` from nnet.c.
pub fn compute_glu(layer: &LinearLayer, output: &mut [f32], input: &[f32]) {
    debug_assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let n = layer.nb_outputs;
    debug_assert!(n <= MAX_INPUTS);

    let mut act2 = [0.0f32; MAX_INPUTS];
    compute_linear(layer, &mut act2[..n], input);
    compute_activation(&mut act2[..n], Activation::Sigmoid);
    for i in 0..n {
        output[i] = input[i] * act2[i];
    }
}

/// In-place GLU: data = data * sigmoid(W * data).
/// Avoids the aliasing issue when caller wants output == input.
pub fn compute_glu_inplace(layer: &LinearLayer, data: &mut [f32]) {
    debug_assert_eq!(layer.nb_inputs, layer.nb_outputs);
    let n = layer.nb_outputs;
    debug_assert!(n <= MAX_INPUTS);

    let mut act2 = [0.0f32; MAX_INPUTS];
    compute_linear(layer, &mut act2[..n], data);
    compute_activation(&mut act2[..n], Activation::Sigmoid);
    for i in 0..n {
        data[i] *= act2[i];
    }
}

/// Gated activation: output = activation(first_half) * sigmoid(second_half).
/// Matches C `compute_gated_activation` from nnet.c.
pub fn compute_gated_activation(
    layer: &LinearLayer,
    output: &mut [f32],
    input: &[f32],
    activation: Activation,
) {
    let n = layer.nb_outputs;
    debug_assert!(n.is_multiple_of(2));
    debug_assert!(n <= MAX_INPUTS);
    let half = n / 2;

    let mut act = [0.0f32; MAX_INPUTS];
    compute_linear(layer, &mut act[..n], input);
    compute_activation(&mut act[..half], activation);
    compute_activation(&mut act[half..n], Activation::Sigmoid);
    for i in 0..half {
        output[i] = act[i] * act[half + i];
    }
}

/// Causal 1D convolution: prepend memory, apply linear + activation.
/// Matches C `compute_generic_conv1d` from nnet.c.
///
/// `mem` holds the previous `(nb_inputs - input_size)` samples.
pub fn compute_generic_conv1d(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    activation: Activation,
) {
    compute_generic_conv1d_dilation(layer, output, mem, input, input_size, 1, activation);
}

/// Causal 1D convolution with dilation.
/// Matches C `compute_generic_conv1d_dilation` from nnet.c.
pub fn compute_generic_conv1d_dilation(
    layer: &LinearLayer,
    output: &mut [f32],
    mem: &mut [f32],
    input: &[f32],
    input_size: usize,
    dilation: usize,
    activation: Activation,
) {
    let ksize = layer.nb_inputs / input_size;
    debug_assert!(layer.nb_inputs <= MAX_CONV_INPUTS);

    let mut tmp = [0.0f32; MAX_CONV_INPUTS];

    if dilation == 1 {
        let mem_size = layer.nb_inputs - input_size;
        if mem_size > 0 {
            tmp[..mem_size].copy_from_slice(&mem[..mem_size]);
        }
    } else {
        for i in 0..ksize - 1 {
            tmp[i * input_size..(i + 1) * input_size].copy_from_slice(
                &mem[i * input_size * dilation..i * input_size * dilation + input_size],
            );
        }
    }
    tmp[(ksize - 1) * input_size..ksize * input_size].copy_from_slice(&input[..input_size]);

    compute_linear(layer, output, &tmp[..layer.nb_inputs]);
    compute_activation(&mut output[..layer.nb_outputs], activation);

    if dilation == 1 {
        let mem_size = layer.nb_inputs - input_size;
        if mem_size > 0 {
            mem[..mem_size].copy_from_slice(&tmp[input_size..input_size + mem_size]);
        }
    } else {
        let total = input_size * dilation * (ksize - 1);
        mem.copy_within(input_size..total, 0);
        mem[total - input_size..total].copy_from_slice(&input[..input_size]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_layer(n: usize) -> LinearLayer {
        let mut weights = vec![0.0f32; n * n];
        for i in 0..n {
            weights[i * n + i] = 1.0;
        }
        LinearLayer {
            bias: None,
            subias: None,
            weights: None,
            float_weights: Some(weights),
            weights_idx: None,
            diag: None,
            scale: None,
            nb_inputs: n,
            nb_outputs: n,
        }
    }

    #[test]
    fn test_generic_dense_linear() {
        let layer = identity_layer(8);
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0f32; 8];
        compute_generic_dense(&layer, &mut output, &input, Activation::Linear);
        assert_eq!(output, input);
    }

    #[test]
    fn test_generic_dense_relu() {
        let mut layer = identity_layer(8);
        layer.bias = Some(vec![-3.0; 8]);
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0f32; 8];
        compute_generic_dense(&layer, &mut output, &input, Activation::Relu);
        assert_eq!(output, [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_glu_basic() {
        let layer = identity_layer(8);
        let input = [0.0f32; 8];
        let mut output = [0.0f32; 8];
        compute_glu(&layer, &mut output, &input);
        for &v in &output {
            assert!(v.abs() < 1e-5);
        }
    }

    #[test]
    fn test_conv1d_passthrough() {
        let layer = identity_layer(8);
        let input = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut output = [0.0f32; 8];
        let mut mem = vec![];
        compute_generic_conv1d(&layer, &mut output, &mut mem, &input, 8, Activation::Linear);
        assert_eq!(output, input);
    }

    #[test]
    fn test_gru_state_update() {
        // Zero weights => z=sigmoid(0)=0.5, h=tanh(0)=0
        // state = 0.5*state + 0.5*0 = 0.5*state
        let input_w = LinearLayer::new(4, 6);
        let recur_w = LinearLayer::new(2, 6);
        let mut state = [1.0f32, 2.0];
        let input = [0.0f32; 4];
        compute_generic_gru(&input_w, &recur_w, &mut state, &input);
        assert!((state[0] - 0.5).abs() < 0.01);
        assert!((state[1] - 1.0).abs() < 0.01);
    }
}
