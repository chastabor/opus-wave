use crate::nnet::ops::{
    compute_generic_conv1d, compute_generic_conv1d_dilation, compute_generic_dense,
    compute_generic_gru,
};
use crate::nnet::weights::{WeightError, linear_init, weight_output_dim};
use crate::nnet::{Activation, LinearLayer, WeightArray};

/// Maximum buffer size for the concatenated encoder stack output.
const MAX_BUFFER_SIZE: usize = 4096;
const MAX_CONV_TMP: usize = 1024;

/// One GRU+Conv stage in the RDOVAE encoder.
struct EncStage {
    gru_input: LinearLayer,
    gru_recurrent: LinearLayer,
    conv_dense: LinearLayer,
    conv: LinearLayer,
}

/// RDOVAE encoder model. Matches auto-generated C `RDOVAEEnc`.
pub struct RdovaeEnc {
    pub enc_dense1: LinearLayer,
    stages: [EncStage; 5],
    pub enc_zdense: LinearLayer,
    pub gdense1: LinearLayer,
    pub gdense2: LinearLayer,
    /// Latent dimension (output of enc_zdense, may be padded).
    pub latent_dim: usize,
    /// State dimension (output of gdense2, may be padded).
    pub state_dim: usize,
}

/// RDOVAE encoder state (recurrent + conv memory).
/// Matches C `RDOVAEEncState`.
pub struct RdovaeEncState {
    pub initialized: bool,
    pub gru_states: [Vec<f32>; 5],
    pub conv1_state: Vec<f32>,
    pub conv_states: [Vec<f32>; 4],
}

/// Initialize RDOVAE encoder model from weight arrays.
pub fn init_rdovae_enc(arrays: &[WeightArray]) -> Result<RdovaeEnc, WeightError> {
    let dim = |name| weight_output_dim(arrays, name);
    let dense1_out = dim("enc_dense1_bias")?;
    let zdense_out = dim("enc_zdense_bias")?;
    let gdense1_out = dim("gdense1_bias")?;
    let gdense2_out = dim("gdense2_bias")?;

    fn init_stage(
        arrays: &[WeightArray],
        n: usize,
        input_size: usize,
    ) -> Result<(EncStage, usize), WeightError> {
        let dim = |name: &str| weight_output_dim(arrays, name);
        let gru_3n = dim(&format!("enc_gru{n}_input_bias"))?;
        let gru_out = gru_3n / 3;
        let conv_dense_out = dim(&format!("enc_conv_dense{n}_bias"))?;
        let conv_out = dim(&format!("enc_conv{n}_bias"))?;

        let stage = EncStage {
            gru_input: linear_init(
                arrays,
                Some(&format!("enc_gru{n}_input_bias")),
                Some(&format!("enc_gru{n}_input_weights")),
                Some(&format!("enc_gru{n}_input_weights")),
                Some(&format!("enc_gru{n}_input_weights_idx")),
                None,
                Some(&format!("enc_gru{n}_input_scale")),
                input_size,
                gru_3n,
            )?,
            gru_recurrent: linear_init(
                arrays,
                Some(&format!("enc_gru{n}_recurrent_bias")),
                Some(&format!("enc_gru{n}_recurrent_weights")),
                Some(&format!("enc_gru{n}_recurrent_weights")),
                None,
                None,
                Some(&format!("enc_gru{n}_recurrent_scale")),
                gru_out,
                gru_3n,
            )?,
            conv_dense: linear_init(
                arrays,
                Some(&format!("enc_conv_dense{n}_bias")),
                Some(&format!("enc_conv_dense{n}_weights")),
                Some(&format!("enc_conv_dense{n}_weights")),
                Some(&format!("enc_conv_dense{n}_weights_idx")),
                None,
                Some(&format!("enc_conv_dense{n}_scale")),
                input_size + gru_out,
                conv_dense_out,
            )?,
            conv: linear_init(
                arrays,
                Some(&format!("enc_conv{n}_bias")),
                Some(&format!("enc_conv{n}_weights")),
                Some(&format!("enc_conv{n}_weights")),
                None,
                None,
                Some(&format!("enc_conv{n}_scale")),
                2 * conv_dense_out,
                conv_out,
            )?,
        };
        // Accumulated output size: input + gru_out + conv_out
        Ok((stage, input_size + gru_out + conv_out))
    }

    let (s1, acc1) = init_stage(arrays, 1, dense1_out)?;
    let (s2, acc2) = init_stage(arrays, 2, acc1)?;
    let (s3, acc3) = init_stage(arrays, 3, acc2)?;
    let (s4, acc4) = init_stage(arrays, 4, acc3)?;
    let (s5, acc5) = init_stage(arrays, 5, acc4)?;

    Ok(RdovaeEnc {
        enc_dense1: linear_init(
            arrays,
            Some("enc_dense1_bias"),
            None,
            Some("enc_dense1_weights"),
            None,
            None,
            None,
            40,
            dense1_out,
        )?,
        stages: [s1, s2, s3, s4, s5],
        enc_zdense: linear_init(
            arrays,
            Some("enc_zdense_bias"),
            Some("enc_zdense_weights"),
            Some("enc_zdense_weights"),
            None,
            None,
            Some("enc_zdense_scale"),
            acc5,
            zdense_out,
        )?,
        gdense1: linear_init(
            arrays,
            Some("gdense1_bias"),
            Some("gdense1_weights"),
            Some("gdense1_weights"),
            Some("gdense1_weights_idx"),
            None,
            Some("gdense1_scale"),
            acc5,
            gdense1_out,
        )?,
        gdense2: linear_init(
            arrays,
            Some("gdense2_bias"),
            Some("gdense2_weights"),
            Some("gdense2_weights"),
            None,
            None,
            Some("gdense2_scale"),
            gdense1_out,
            gdense2_out,
        )?,
        latent_dim: zdense_out,
        state_dim: gdense2_out,
    })
}

/// Create encoder state from model.
pub fn rdovae_enc_state_init(model: &RdovaeEnc) -> RdovaeEncState {
    let gru_sizes: Vec<usize> = model
        .stages
        .iter()
        .map(|s| s.gru_recurrent.nb_inputs)
        .collect();
    let conv1_in = model.stages[0].conv.nb_inputs;
    let conv_states: Vec<Vec<f32>> = model.stages[1..]
        .iter()
        .map(|s| vec![0.0; 2 * s.conv.nb_inputs])
        .collect();
    RdovaeEncState {
        initialized: false,
        gru_states: [
            vec![0.0; gru_sizes[0]],
            vec![0.0; gru_sizes[1]],
            vec![0.0; gru_sizes[2]],
            vec![0.0; gru_sizes[3]],
            vec![0.0; gru_sizes[4]],
        ],
        conv1_state: vec![0.0; conv1_in],
        conv_states: [
            conv_states.first().cloned().unwrap_or_default(),
            conv_states.get(1).cloned().unwrap_or_default(),
            conv_states.get(2).cloned().unwrap_or_default(),
            conv_states.get(3).cloned().unwrap_or_default(),
        ],
    }
}

/// Encode one double-frame through the RDOVAE encoder.
/// Produces latent codes and initial decoder state.
/// Matches C `dred_rdovae_encode_dframe`.
pub fn dred_rdovae_encode_dframe(
    enc_state: &mut RdovaeEncState,
    model: &RdovaeEnc,
    latents: &mut [f32],
    initial_state: &mut [f32],
    input: &[f32],
) {
    let mut buffer = [0.0f32; MAX_BUFFER_SIZE];
    let mut conv_tmp = [0.0f32; MAX_CONV_TMP];
    let mut output_index = 0;

    // Dense1
    let dense1_out = model.enc_dense1.nb_outputs;
    compute_generic_dense(
        &model.enc_dense1,
        &mut buffer[..dense1_out],
        input,
        Activation::Tanh,
    );
    output_index += dense1_out;

    // 5 GRU+Conv stages
    for (si, stage) in model.stages.iter().enumerate() {
        let gru_out = stage.gru_recurrent.nb_inputs;
        let conv_dense_out = stage.conv_dense.nb_outputs;
        let conv_out = stage.conv.nb_outputs;

        compute_generic_gru(
            &stage.gru_input,
            &stage.gru_recurrent,
            &mut enc_state.gru_states[si],
            &buffer[..output_index],
        );
        buffer[output_index..output_index + gru_out].copy_from_slice(&enc_state.gru_states[si]);
        output_index += gru_out;

        // Conditional init for conv state
        if !enc_state.initialized {
            if si == 0 {
                for v in enc_state.conv1_state.iter_mut() {
                    *v = 0.0;
                }
            } else {
                for v in enc_state.conv_states[si - 1].iter_mut() {
                    *v = 0.0;
                }
            }
        }

        compute_generic_dense(
            &stage.conv_dense,
            &mut conv_tmp[..conv_dense_out],
            &buffer[..output_index],
            Activation::Tanh,
        );

        if si == 0 {
            compute_generic_conv1d(
                &stage.conv,
                &mut buffer[output_index..output_index + conv_out],
                &mut enc_state.conv1_state,
                &conv_tmp[..conv_dense_out],
                conv_dense_out,
                Activation::Tanh,
            );
        } else {
            compute_generic_conv1d_dilation(
                &stage.conv,
                &mut buffer[output_index..output_index + conv_out],
                &mut enc_state.conv_states[si - 1],
                &conv_tmp[..conv_dense_out],
                conv_dense_out,
                2,
                Activation::Tanh,
            );
        }
        output_index += conv_out;
    }
    enc_state.initialized = true;

    // Latent output
    let mut padded_latents = [0.0f32; 256];
    compute_generic_dense(
        &model.enc_zdense,
        &mut padded_latents[..model.latent_dim],
        &buffer[..output_index],
        Activation::Linear,
    );
    latents[..model.latent_dim].copy_from_slice(&padded_latents[..model.latent_dim]);

    // State output
    let gdense1_out = model.gdense1.nb_outputs;
    let mut state_hidden = [0.0f32; 512];
    compute_generic_dense(
        &model.gdense1,
        &mut state_hidden[..gdense1_out],
        &buffer[..output_index],
        Activation::Tanh,
    );
    let mut padded_state = [0.0f32; 256];
    compute_generic_dense(
        &model.gdense2,
        &mut padded_state[..model.state_dim],
        &state_hidden[..gdense1_out],
        Activation::Linear,
    );
    initial_state[..model.state_dim].copy_from_slice(&padded_state[..model.state_dim]);
}
