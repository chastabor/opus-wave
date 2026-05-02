//! DNN-related types for DRED, OSCE, and deep PLC.
//!
//! These types are available when the `dnn` feature is enabled.
//! They provide CTL-like configuration and the public API surface
//! for DNN features in the Opus encoder and decoder.

use crate::error::OpusError;

/// CTL request IDs for DNN features, matching C opus_defines.h.
pub const OPUS_SET_DRED_DURATION_REQUEST: i32 = 4050;
pub const OPUS_GET_DRED_DURATION_REQUEST: i32 = 4051;
pub const OPUS_SET_DNN_BLOB_REQUEST: i32 = 4052;
pub const OPUS_SET_OSCE_BWE_REQUEST: i32 = 4054;
pub const OPUS_GET_OSCE_BWE_REQUEST: i32 = 4055;

/// DNN decoder state, wrapping the opus-dnn components.
/// Holds the DRED decoder, PLC state, and OSCE model.
///
/// This is stored as an `Option` in `OpusDecoder` — `None` when
/// DNN is not loaded (no weight blob provided).
pub struct DnnDecoderState {
    /// Whether DNN models are loaded and ready.
    pub(crate) loaded: bool,
    /// DRED decoded state (latents, features, offsets).
    pub(crate) dred: opus_dnn::dred::decoder::OpusDred,
    /// DRED quantization statistics (from model weight data).
    pub(crate) dred_stats: opus_dnn::dred::decoder::DredStats,
    /// DRED RDOVAE decoder model.
    pub(crate) rdovae_dec: opus_dnn::dred::rdovae_dec::RdovaeDec,
    /// RDOVAE decoder state (reusable across calls).
    pub(crate) rdovae_dec_state: opus_dnn::dred::rdovae_dec::RdovaeDecState,
    /// LPCNet PLC state (deep packet loss concealment).
    pub(crate) plc: opus_dnn::lpcnet::plc::LpcnetPlcState,
    /// OSCE model (speech enhancement).
    pub(crate) osce: opus_dnn::osce::structs::OsceModel,
    /// Per-channel OSCE feature extraction state.
    pub(crate) osce_feature_state: [opus_dnn::osce::structs::OsceFeatureState; 2],
    /// Per-channel LACE processing state.
    pub(crate) osce_lace_state: [Option<opus_dnn::osce::lace::LaceState>; 2],
    /// Per-channel NoLACE processing state.
    pub(crate) osce_nolace_state: [Option<opus_dnn::osce::nolace::NoLaceState>; 2],
}

impl DnnDecoderState {
    /// Load decoder DNN state from a binary weight blob.
    ///
    /// Parses the blob to initialize all decoder DNN components:
    /// RDOVAE decoder, DRED stats, PLC (FARGAN + PitchDNN), and OSCE.
    pub fn from_blob(data: &[u8]) -> Result<Self, OpusError> {
        let (rdovae_dec, plc, osce) =
            opus_dnn::load::load_decoder_dnn(data).map_err(|_| OpusError::BadArg)?;

        let dred_stats = opus_dnn::dred::stats::init_dred_stats();
        let latent_dim = rdovae_dec.latent_dim;
        let state_dim = rdovae_dec.state_dim;
        let rdovae_dec_state = opus_dnn::dred::rdovae_dec::rdovae_dec_state_init(&rdovae_dec);
        let dred = opus_dnn::dred::decoder::OpusDred::new(latent_dim, state_dim);

        // Initialize per-channel OSCE state from model dimensions
        let osce_lace_state = [
            osce.lace
                .as_ref()
                .map(opus_dnn::osce::lace::lace_state_init),
            osce.lace
                .as_ref()
                .map(opus_dnn::osce::lace::lace_state_init),
        ];
        let osce_nolace_state = [
            osce.nolace
                .as_ref()
                .map(opus_dnn::osce::nolace::nolace_state_init),
            osce.nolace
                .as_ref()
                .map(opus_dnn::osce::nolace::nolace_state_init),
        ];

        Ok(DnnDecoderState {
            loaded: true,
            dred,
            dred_stats,
            rdovae_dec,
            rdovae_dec_state,
            plc,
            osce,
            osce_feature_state: Default::default(),
            osce_lace_state,
            osce_nolace_state,
        })
    }
}

/// DNN encoder state, wrapping the opus-dnn DRED encoder.
///
/// Stored as `Option` in `OpusEncoder` — `None` when DRED is disabled.
pub struct DnnEncoderState {
    /// Whether DNN models are loaded and ready.
    pub(crate) loaded: bool,
    /// DRED encoder state (latent computation + encoding).
    pub(crate) dred_enc: opus_dnn::dred::encoder::DredEnc,
    /// DRED quantization statistics (for encoding latents).
    pub(crate) dred_stats: opus_dnn::dred::decoder::DredStats,
    /// DRED duration in frames (0 = disabled).
    pub(crate) dred_duration: i32,
    /// Per-2.5ms voice activity flags for DRED encoding.
    /// Size: DRED_MAX_FRAMES * 4. Shifted left each frame, new entries appended.
    pub(crate) activity_mem: Vec<u8>,
}

impl DnnEncoderState {
    /// Load encoder DNN state from a binary weight blob.
    ///
    /// Parses the blob to initialize the RDOVAE encoder and PitchDNN.
    /// DRED duration defaults to 0 (disabled); call `set_dred_duration`
    /// on the encoder to enable DRED after loading.
    pub fn from_blob(data: &[u8]) -> Result<Self, OpusError> {
        let (rdovae_enc, lpcnet_enc_state) =
            opus_dnn::load::load_encoder_dnn(data).map_err(|_| OpusError::BadArg)?;

        let dred_enc = opus_dnn::dred::encoder::dred_encoder_init(rdovae_enc, lpcnet_enc_state);
        let dred_stats = opus_dnn::dred::stats::init_dred_stats();

        Ok(DnnEncoderState {
            loaded: true,
            dred_enc,
            dred_stats,
            dred_duration: 0,
            activity_mem: vec![0u8; opus_dnn::dred::DRED_MAX_FRAMES * 4],
        })
    }
}
