pub mod bbwenet;
pub mod common;
pub mod config;
pub mod features;
pub mod lace;
pub mod nolace;
pub mod structs;

use crate::nnet::weights::{WeightError, parse_weights};
use config::*;
use structs::OsceModel;

/// Load OSCE models (LACE, NoLACE, BBWENet) from weight data.
/// Matches C `osce_load_models` from osce.c.
pub fn osce_load_models(
    model: &mut OsceModel,
    data: &[u8],
    _len: usize,
) -> Result<(), WeightError> {
    let arrays = parse_weights(data).ok_or(WeightError)?;
    osce_load_models_from_arrays(model, &arrays)
}

/// Load OSCE models from pre-parsed weight arrays.
/// Avoids re-parsing the blob when arrays are already available.
pub fn osce_load_models_from_arrays(
    model: &mut OsceModel,
    arrays: &[crate::nnet::WeightArray],
) -> Result<(), WeightError> {
    if let Ok(lace_model) = lace::init_lace(arrays) {
        model.lace = Some(lace_model);
        model.method = OSCE_METHOD_LACE;
    }

    if let Ok(nolace_model) = nolace::init_nolace(arrays) {
        model.nolace = Some(nolace_model);
        model.method = OSCE_METHOD_NOLACE; // NoLACE preferred when both available
    }

    model.loaded = model.lace.is_some();
    Ok(())
}

/// Reset OSCE state for a given method.
/// Matches C `osce_reset` from osce.c.
pub fn osce_reset(model: &mut OsceModel, method: i32) {
    model.method = method;
}

/// Enhance a decoded SILK frame using OSCE.
/// Matches C `osce_enhance_frame` from osce.c.
///
/// The `features` array provides the 93-dimensional OSCE feature vector
/// extracted from the SILK decoder state. The `periods` array provides
/// pitch lags for each subframe. The `numbits` array provides the
/// bit-count for each 10ms half-frame.
///
/// `xq` is modified in-place with the enhanced output.
pub fn osce_enhance_frame(
    model: &OsceModel,
    lace_state: &mut Option<lace::LaceState>,
    nolace_state: &mut Option<nolace::NoLaceState>,
    xq: &mut [f32],
    features: &[f32],
    numbits: &[f32; 2],
    periods: &[usize; 4],
) {
    if !model.loaded {
        return;
    }

    match model.method {
        OSCE_METHOD_LACE => {
            if let (Some(lace_model), Some(state)) = (&model.lace, lace_state.as_mut()) {
                let mut output = [0.0f32; 4 * lace::LACE_FRAME_SIZE];
                let len = xq.len().min(output.len());
                lace::lace_process_20ms_frame(
                    lace_model,
                    state,
                    &mut output,
                    xq,
                    features,
                    numbits,
                    periods,
                );
                xq[..len].copy_from_slice(&output[..len]);
            }
        }
        OSCE_METHOD_NOLACE => {
            if let (Some(nolace_model), Some(state)) = (&model.nolace, nolace_state.as_mut()) {
                let mut output = [0.0f32; 4 * nolace::NOLACE_FRAME_SIZE];
                let len = xq.len().min(output.len());
                nolace::nolace_process_20ms_frame(
                    nolace_model,
                    state,
                    &mut output,
                    xq,
                    features,
                    numbits,
                    periods,
                );
                xq[..len].copy_from_slice(&output[..len]);
            }
        }
        _ => {}
    }
}
