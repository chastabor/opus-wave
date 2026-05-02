pub mod coding;
pub mod decoder;
pub mod encoder;
pub mod rdovae_dec;
pub mod rdovae_enc;
pub mod stats;

use crate::lpcnet::enc::NB_TOTAL_FEATURES;

/// DRED configuration constants from dred_config.h.
pub const DRED_EXTENSION_ID: u8 = 126;
pub const DRED_EXPERIMENTAL_VERSION: u8 = 12;
pub const DRED_EXPERIMENTAL_BYTES: usize = 2;
pub const DRED_MIN_BYTES: usize = 8;

pub const DRED_SILK_ENCODER_DELAY: usize = 79 + 12 - 80;
pub const DRED_FRAME_SIZE: usize = 160;
pub const DRED_DFRAME_SIZE: usize = 2 * DRED_FRAME_SIZE;
pub const DRED_MAX_DATA_SIZE: usize = 1000;
pub const DRED_ENC_Q0: i32 = 6;
pub const DRED_ENC_Q1: i32 = 15;

pub const DRED_MAX_LATENTS: usize = 26;
pub const DRED_NUM_REDUNDANCY_FRAMES: usize = 2 * DRED_MAX_LATENTS;
pub const DRED_MAX_FRAMES: usize = 4 * DRED_MAX_LATENTS;

/// Number of features per DRED frame (same as LPCNet total features).
pub const DRED_NUM_FEATURES: usize = NB_TOTAL_FEATURES;
