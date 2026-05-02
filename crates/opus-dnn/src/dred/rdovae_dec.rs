use crate::nnet::ops::{
    compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu,
};
use crate::nnet::weights::{WeightError, linear_init, weight_input_dim, weight_output_dim};
use crate::nnet::{Activation, LinearLayer, WeightArray};

use super::DRED_NUM_FEATURES;

const MAX_BUFFER_SIZE: usize = 4096;
const MAX_CONV_TMP: usize = 1024;
const MAX_STATE_INIT: usize = 2048;

/// One GRU+GLU+Conv stage in the RDOVAE decoder.
struct DecStage {
    gru_input: LinearLayer,
    gru_recurrent: LinearLayer,
    glu: LinearLayer,
    conv_dense: LinearLayer,
    conv: LinearLayer,
}

/// RDOVAE decoder model. Matches auto-generated C `RDOVAEDec`.
pub struct RdovaeDec {
    pub dec_hidden_init: LinearLayer,
    pub dec_gru_init: LinearLayer,
    pub dec_dense1: LinearLayer,
    stages: [DecStage; 5],
    pub dec_output: LinearLayer,
    /// GRU sizes for state init (sum of all 5 GRU state sizes).
    gru_sizes: [usize; 5],
    /// Latent dimension (from model).
    pub latent_dim: usize,
    /// State dimension (from model).
    pub state_dim: usize,
}

/// RDOVAE decoder state. Matches C `RDOVAEDecState`.
pub struct RdovaeDecState {
    pub initialized: bool,
    pub gru_states: [Vec<f32>; 5],
    pub conv_states: [Vec<f32>; 5],
}

/// Initialize RDOVAE decoder model from weight arrays.
pub fn init_rdovae_dec(arrays: &[WeightArray]) -> Result<RdovaeDec, WeightError> {
    let dim = |name: &str| weight_output_dim(arrays, name);

    let hidden_init_out = dim("dec_hidden_init_bias")?;
    let gru_init_out = dim("dec_gru_init_bias")?;
    let dense1_out = dim("dec_dense1_bias")?;
    let output_out = dim("dec_output_bias")?;

    fn init_stage(
        arrays: &[WeightArray],
        n: usize,
        input_size: usize,
    ) -> Result<(DecStage, usize, usize), WeightError> {
        let dim = |name: &str| weight_output_dim(arrays, name);
        let gru_3n = dim(&format!("dec_gru{n}_input_bias"))?;
        let gru_out = gru_3n / 3;
        let conv_dense_out = dim(&format!("dec_conv_dense{n}_bias"))?;
        let conv_out = dim(&format!("dec_conv{n}_bias"))?;

        let stage = DecStage {
            gru_input: linear_init(
                arrays,
                Some(&format!("dec_gru{n}_input_bias")),
                Some(&format!("dec_gru{n}_input_weights")),
                Some(&format!("dec_gru{n}_input_weights")),
                Some(&format!("dec_gru{n}_input_weights_idx")),
                None,
                Some(&format!("dec_gru{n}_input_scale")),
                input_size,
                gru_3n,
            )?,
            gru_recurrent: linear_init(
                arrays,
                Some(&format!("dec_gru{n}_recurrent_bias")),
                Some(&format!("dec_gru{n}_recurrent_weights")),
                Some(&format!("dec_gru{n}_recurrent_weights")),
                None,
                None,
                Some(&format!("dec_gru{n}_recurrent_scale")),
                gru_out,
                gru_3n,
            )?,
            glu: linear_init(
                arrays,
                Some(&format!("dec_glu{n}_bias")),
                Some(&format!("dec_glu{n}_weights")),
                Some(&format!("dec_glu{n}_weights")),
                None,
                None,
                Some(&format!("dec_glu{n}_scale")),
                gru_out,
                gru_out,
            )?,
            conv_dense: linear_init(
                arrays,
                Some(&format!("dec_conv_dense{n}_bias")),
                Some(&format!("dec_conv_dense{n}_weights")),
                Some(&format!("dec_conv_dense{n}_weights")),
                Some(&format!("dec_conv_dense{n}_weights_idx")),
                None,
                Some(&format!("dec_conv_dense{n}_scale")),
                input_size + gru_out,
                conv_dense_out,
            )?,
            conv: linear_init(
                arrays,
                Some(&format!("dec_conv{n}_bias")),
                Some(&format!("dec_conv{n}_weights")),
                Some(&format!("dec_conv{n}_weights")),
                None,
                None,
                Some(&format!("dec_conv{n}_scale")),
                2 * conv_dense_out,
                conv_out,
            )?,
        };
        Ok((stage, input_size + gru_out + conv_out, gru_out))
    }

    let (s1, acc1, g1) = init_stage(arrays, 1, dense1_out)?;
    let (s2, acc2, g2) = init_stage(arrays, 2, acc1)?;
    let (s3, acc3, g3) = init_stage(arrays, 3, acc2)?;
    let (s4, acc4, g4) = init_stage(arrays, 4, acc3)?;
    let (s5, acc5, g5) = init_stage(arrays, 5, acc4)?;

    let actual_state_dim = weight_input_dim(arrays, "dec_hidden_init_weights", hidden_init_out)?;
    let actual_latent_dim = weight_input_dim(arrays, "dec_dense1_weights", dense1_out)?;

    Ok(RdovaeDec {
        dec_hidden_init: linear_init(
            arrays,
            Some("dec_hidden_init_bias"),
            None,
            Some("dec_hidden_init_weights"),
            None,
            None,
            None,
            actual_state_dim,
            hidden_init_out,
        )?,
        dec_gru_init: linear_init(
            arrays,
            Some("dec_gru_init_bias"),
            Some("dec_gru_init_weights"),
            Some("dec_gru_init_weights"),
            Some("dec_gru_init_weights_idx"),
            None,
            Some("dec_gru_init_scale"),
            hidden_init_out,
            gru_init_out,
        )?,
        dec_dense1: linear_init(
            arrays,
            Some("dec_dense1_bias"),
            None,
            Some("dec_dense1_weights"),
            None,
            None,
            None,
            actual_latent_dim,
            dense1_out,
        )?,
        stages: [s1, s2, s3, s4, s5],
        dec_output: linear_init(
            arrays,
            Some("dec_output_bias"),
            Some("dec_output_weights"),
            Some("dec_output_weights"),
            Some("dec_output_weights_idx"),
            None,
            Some("dec_output_scale"),
            acc5,
            output_out,
        )?,
        gru_sizes: [g1, g2, g3, g4, g5],
        latent_dim: actual_latent_dim,
        state_dim: actual_state_dim,
    })
}

/// Create decoder state from model.
pub fn rdovae_dec_state_init(model: &RdovaeDec) -> RdovaeDecState {
    RdovaeDecState {
        initialized: false,
        gru_states: [
            vec![0.0; model.gru_sizes[0]],
            vec![0.0; model.gru_sizes[1]],
            vec![0.0; model.gru_sizes[2]],
            vec![0.0; model.gru_sizes[3]],
            vec![0.0; model.gru_sizes[4]],
        ],
        conv_states: [
            vec![0.0; model.stages[0].conv.nb_inputs],
            vec![0.0; model.stages[1].conv.nb_inputs],
            vec![0.0; model.stages[2].conv.nb_inputs],
            vec![0.0; model.stages[3].conv.nb_inputs],
            vec![0.0; model.stages[4].conv.nb_inputs],
        ],
    }
}

/// Initialize decoder GRU states from an initial state vector.
/// Matches C `dred_rdovae_dec_init_states`.
pub fn dred_rdovae_dec_init_states(
    h: &mut RdovaeDecState,
    model: &RdovaeDec,
    initial_state: &[f32],
) {
    let hidden_out = model.dec_hidden_init.nb_outputs;
    let gru_init_out = model.dec_gru_init.nb_outputs;

    let mut hidden = [0.0f32; MAX_STATE_INIT];
    compute_generic_dense(
        &model.dec_hidden_init,
        &mut hidden[..hidden_out],
        initial_state,
        Activation::Tanh,
    );

    let mut state_init = [0.0f32; MAX_STATE_INIT];
    compute_generic_dense(
        &model.dec_gru_init,
        &mut state_init[..gru_init_out],
        &hidden[..hidden_out],
        Activation::Tanh,
    );

    let mut counter = 0;
    for i in 0..5 {
        let size = model.gru_sizes[i];
        h.gru_states[i][..size].copy_from_slice(&state_init[counter..counter + size]);
        counter += size;
    }
    h.initialized = false;
}

/// Decode one quadruple-frame from a latent vector.
/// Matches C `dred_rdovae_decode_qframe`.
pub fn dred_rdovae_decode_qframe(
    dec_state: &mut RdovaeDecState,
    model: &RdovaeDec,
    qframe: &mut [f32],
    input: &[f32],
) {
    let mut buffer = [0.0f32; MAX_BUFFER_SIZE];
    let mut conv_tmp = [0.0f32; MAX_CONV_TMP];
    let mut output_index = 0;

    let dense1_out = model.dec_dense1.nb_outputs;
    compute_generic_dense(
        &model.dec_dense1,
        &mut buffer[..dense1_out],
        input,
        Activation::Tanh,
    );
    output_index += dense1_out;

    for (si, stage) in model.stages.iter().enumerate() {
        let gru_out = model.gru_sizes[si];
        let conv_dense_out = stage.conv_dense.nb_outputs;
        let conv_out = stage.conv.nb_outputs;

        compute_generic_gru(
            &stage.gru_input,
            &stage.gru_recurrent,
            &mut dec_state.gru_states[si],
            &buffer[..output_index],
        );
        compute_glu(
            &stage.glu,
            &mut buffer[output_index..output_index + gru_out],
            &dec_state.gru_states[si],
        );
        output_index += gru_out;

        if !dec_state.initialized {
            for v in dec_state.conv_states[si].iter_mut() {
                *v = 0.0;
            }
        }

        compute_generic_dense(
            &stage.conv_dense,
            &mut conv_tmp[..conv_dense_out],
            &buffer[..output_index],
            Activation::Tanh,
        );
        compute_generic_conv1d(
            &stage.conv,
            &mut buffer[output_index..output_index + conv_out],
            &mut dec_state.conv_states[si],
            &conv_tmp[..conv_dense_out],
            conv_dense_out,
            Activation::Tanh,
        );
        output_index += conv_out;
    }
    dec_state.initialized = true;

    let output_size = model.dec_output.nb_outputs;
    compute_generic_dense(
        &model.dec_output,
        &mut qframe[..output_size],
        &buffer[..output_index],
        Activation::Linear,
    );
}

/// Decode all latent frames into features.
/// Matches C `DRED_rdovae_decode_all`.
pub fn dred_rdovae_decode_all(
    dec: &mut RdovaeDecState,
    model: &RdovaeDec,
    features: &mut [f32],
    state: &[f32],
    latents: &[f32],
    nb_latents: usize,
    latent_dim: usize,
) {
    dred_rdovae_dec_init_states(dec, model, state);
    for i in (0..2 * nb_latents).step_by(2) {
        dred_rdovae_decode_qframe(
            dec,
            model,
            &mut features[2 * i * DRED_NUM_FEATURES..],
            &latents[(i / 2) * (latent_dim + 1)..],
        );
    }
}
