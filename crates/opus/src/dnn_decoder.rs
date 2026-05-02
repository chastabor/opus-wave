//! DNN decoder integration: DRED extension parsing, PLC, and OSCE.

use crate::decoder::OpusDecoder;
use crate::extensions::OpusExtensionData;

use opus_dnn::osce::config::OSCE_FEATURE_DIM;
use opus_dnn::osce::features::{OsceInput, osce_calculate_features};
use opus_dnn::osce::structs::{OsceFeatureState, OsceModel};
use opus_silk::decoder::SilkPostFilter;
use opus_silk::{ChannelState, DecoderControl};

/// DRED process_stage value indicating latents have been decoded.
const DRED_STAGE_DECODED: i32 = 1;

/// Parse DRED data from a decoded Opus extension (ID 126).
/// Extracts DRED latents from the extension payload via range decoding,
/// runs the RDOVAE decoder to recover FEC features, and feeds them
/// to the PLC system for use during packet loss concealment.
pub fn decoder_process_dred_extension(
    decoder: &mut OpusDecoder,
    extension: &OpusExtensionData,
    dred_frame_offset: i32,
) {
    let dred_ext_id = opus_dnn::dred::DRED_EXTENSION_ID as i32;
    if extension.id != dred_ext_id {
        return;
    }
    let Some(dnn) = decoder.dnn.as_mut() else {
        return;
    };
    if !dnn.loaded {
        return;
    }

    let _nb_latents = opus_dnn::dred::decoder::dred_ec_decode(
        &mut dnn.dred,
        &extension.data,
        extension.data.len(),
        opus_dnn::dred::DRED_NUM_REDUNDANCY_FRAMES,
        dred_frame_offset,
        &dnn.dred_stats,
    );

    if dnn.dred.nb_latents > 0 && dnn.dred.process_stage == DRED_STAGE_DECODED {
        opus_dnn::dred::rdovae_dec::dred_rdovae_decode_all(
            &mut dnn.rdovae_dec_state,
            &dnn.rdovae_dec,
            &mut dnn.dred.fec_features,
            &dnn.dred.state,
            &dnn.dred.latents,
            dnn.dred.nb_latents,
            dnn.dred.latent_dim,
        );

        for i in 0..dnn.dred.nb_latents * 2 {
            let feature_start = i * opus_dnn::dred::DRED_NUM_FEATURES;
            let feature_end = feature_start + opus_dnn::fargan::NB_FEATURES;
            if feature_end <= dnn.dred.fec_features.len() {
                opus_dnn::lpcnet::plc::lpcnet_plc_fec_add(
                    &mut dnn.plc,
                    Some(&dnn.dred.fec_features[feature_start..feature_end]),
                );
            }
        }
    }
}

/// Update PLC state with a successfully decoded (good) packet.
pub fn decoder_plc_update(decoder: &mut OpusDecoder, pcm: &[i16]) {
    let Some(dnn) = decoder.dnn.as_mut() else {
        return;
    };
    if !dnn.loaded {
        return;
    }
    opus_dnn::lpcnet::plc::lpcnet_plc_update(&mut dnn.plc, pcm);
    opus_dnn::lpcnet::plc::lpcnet_plc_fec_clear(&mut dnn.plc);
}

/// Conceal a lost packet using DNN-based PLC.
/// Returns true if DNN PLC was applied, false if not available.
pub fn decoder_plc_conceal(decoder: &mut OpusDecoder, pcm: &mut [i16]) -> bool {
    let Some(dnn) = decoder.dnn.as_mut() else {
        return false;
    };
    if !dnn.loaded {
        return false;
    }
    opus_dnn::lpcnet::plc::lpcnet_plc_conceal(&mut dnn.plc, pcm);
    true
}

// ============ OSCE Post-Filter ============

/// OSCE post-filter that enhances decoded SILK output.
/// Implements `SilkPostFilter` to be called inside the SILK decode loop.
pub(crate) struct OscePostFilter<'a> {
    pub(crate) model: &'a OsceModel,
    pub(crate) feature_state: &'a mut OsceFeatureState,
    pub(crate) lace_state: &'a mut Option<opus_dnn::osce::lace::LaceState>,
    pub(crate) nolace_state: &'a mut Option<opus_dnn::osce::nolace::NoLaceState>,
}

impl SilkPostFilter for OscePostFilter<'_> {
    fn enhance_frame(
        &mut self,
        p_out: &mut [i16],
        channel: &ChannelState,
        control: &DecoderControl,
        num_bits: i32,
    ) {
        // OSCE only operates at 16kHz with 20ms frames (4 subframes)
        if channel.fs_khz != 16 || channel.nb_subfr != 4 {
            return;
        }
        if !self.model.loaded {
            return;
        }

        let input = OsceInput {
            pred_coef_q12: &control.pred_coef_q12,
            pitch_lags: &control.pitch_l,
            ltp_coef_q14: &control.ltp_coef_q14,
            gains_q16: &control.gains_q16,
            lpc_order: channel.lpc_order as usize,
            signal_type: channel.indices.signal_type as i32,
            nb_subfr: channel.nb_subfr as usize,
            num_bits,
        };

        // Calculate OSCE features from SILK decoder state
        let mut features = [0.0f32; 4 * OSCE_FEATURE_DIM];
        let mut numbits = [0.0f32; 2];
        let mut periods = [0usize; 4];
        osce_calculate_features(
            self.feature_state,
            &input,
            p_out,
            &mut features,
            &mut numbits,
            &mut periods,
        );

        // Convert i16 -> f32 for enhancement
        let n = (channel.nb_subfr * channel.subfr_length) as usize;
        let n = n.min(320);
        let mut xq_f32 = [0.0f32; 320];
        for i in 0..n {
            xq_f32[i] = p_out[i] as f32 / 32768.0;
        }

        // Run OSCE enhancement
        opus_dnn::osce::osce_enhance_frame(
            self.model,
            self.lace_state,
            self.nolace_state,
            &mut xq_f32[..n],
            &features,
            &numbits,
            &periods,
        );

        // Convert f32 -> i16 back
        for i in 0..n {
            p_out[i] = (xq_f32[i] * 32768.0).round().clamp(-32768.0, 32767.0) as i16;
        }
    }
}
