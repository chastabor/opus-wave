pub mod adacomb;
pub mod adaconv;
pub mod adashape;

// Adaptive convolution constants.
pub const ADACONV_MAX_KERNEL_SIZE: usize = 32;
pub const ADACONV_MAX_INPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_OUTPUT_CHANNELS: usize = 3;
pub const ADACONV_MAX_FRAME_SIZE: usize = 240;
pub const ADACONV_MAX_OVERLAP_SIZE: usize = 120;

// Adaptive comb filter constants.
pub const ADACOMB_MAX_LAG: usize = 300;
pub const ADACOMB_MAX_KERNEL_SIZE: usize = 16;
pub const ADACOMB_MAX_FRAME_SIZE: usize = 80;
pub const ADACOMB_MAX_OVERLAP_SIZE: usize = 40;

// Adaptive shape constants.
pub const ADASHAPE_MAX_INPUT_DIM: usize = 512;
pub const ADASHAPE_MAX_FRAME_SIZE: usize = 240;

/// AdaConv state: history buffer + last kernel/gain for overlap blending.
/// Matches C `AdaConvState` from nndsp.h.
pub struct AdaConvState {
    pub history: [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
    pub last_kernel:
        [f32; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS * ADACONV_MAX_OUTPUT_CHANNELS],
    pub last_gain: f32,
}

impl Default for AdaConvState {
    fn default() -> Self {
        AdaConvState {
            history: [0.0; ADACONV_MAX_KERNEL_SIZE * ADACONV_MAX_INPUT_CHANNELS],
            last_kernel: [0.0; ADACONV_MAX_KERNEL_SIZE
                * ADACONV_MAX_INPUT_CHANNELS
                * ADACONV_MAX_OUTPUT_CHANNELS],
            last_gain: 0.0,
        }
    }
}

/// AdaComb state: history buffer + last kernel/gain/pitch for overlap blending.
/// Matches C `AdaCombState` from nndsp.h.
pub struct AdaCombState {
    pub history: [f32; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
    pub last_kernel: [f32; ADACOMB_MAX_KERNEL_SIZE],
    pub last_global_gain: f32,
    pub last_pitch_lag: usize,
}

impl Default for AdaCombState {
    fn default() -> Self {
        AdaCombState {
            history: [0.0; ADACOMB_MAX_KERNEL_SIZE + ADACOMB_MAX_LAG],
            last_kernel: [0.0; ADACOMB_MAX_KERNEL_SIZE],
            last_global_gain: 0.0,
            last_pitch_lag: 0,
        }
    }
}

/// AdaShape state: conv memory + interpolation state.
/// Matches C `AdaShapeState` from nndsp.h.
pub struct AdaShapeState {
    pub conv_alpha1f_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha1t_state: [f32; ADASHAPE_MAX_INPUT_DIM],
    pub conv_alpha2_state: [f32; ADASHAPE_MAX_FRAME_SIZE],
    pub interpolate_state: [f32; 1],
}

impl Default for AdaShapeState {
    fn default() -> Self {
        AdaShapeState {
            conv_alpha1f_state: [0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha1t_state: [0.0; ADASHAPE_MAX_INPUT_DIM],
            conv_alpha2_state: [0.0; ADASHAPE_MAX_FRAME_SIZE],
            interpolate_state: [0.0; 1],
        }
    }
}

/// Compute overlap-add window (cosine half-window).
/// Matches C `compute_overlap_window` from nndsp.c.
pub fn compute_overlap_window(window: &mut [f32], overlap_size: usize) {
    for (i, w) in window[..overlap_size].iter_mut().enumerate() {
        *w = 0.5 + 0.5 * (std::f32::consts::PI * (i as f32 + 0.5) / overlap_size as f32).cos();
    }
}
