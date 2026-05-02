// Top-level SILK encoder
// Port of silk/enc_API.c, silk/control_codec.c (simplified)

use crate::encode_indices;
use crate::encode_pulses;
use crate::gain_quant;
use crate::lpc_analysis;
use crate::nlsf::*;
use crate::nlsf_encode;
use crate::noise_shape_analysis;
use crate::nsq::{self, MAX_SHAPE_LPC_ORDER, NsqState};
use crate::nsq_del_dec;
use crate::pitch_analysis;
use crate::tables::*;
use crate::vad;
use crate::*;
use opus_range_coder::EcCtx;

/// Fixed-point division with variable Q shift: (a << q_shift) / b
#[allow(dead_code)]
fn silk_div32_varq(a: i32, b: i32, q_shift: i32) -> i32 {
    (((a as i64) << q_shift as u32) / b.max(1) as i64) as i32
}

// ── Bitrate-to-SNR lookup tables (from silk/control_SNR.c) ──
// Values are SNR_dB / 21, stored as u8. Multiply by 21 to get SNR_dB_Q7.
// Index 0 corresponds to 4000 bps (first 10 entries at 400bps spacing omitted).

#[rustfmt::skip]
static SNR_TABLE_NB: &[u8] = &[
                                          0, 15, 39, 52, 61, 68,
     74, 79, 84, 88, 92, 95, 99,102,105,108,111,114,117,119,122,124,
    126,129,131,133,135,137,139,142,143,145,147,149,151,153,155,157,
    158,160,162,163,165,167,168,170,171,173,174,176,177,179,180,182,
    183,185,186,187,189,190,192,193,194,196,197,199,200,201,203,204,
    205,207,208,209,211,212,213,215,216,217,219,220,221,223,224,225,
    227,228,230,231,232,234,235,236,238,239,241,242,243,245,246,248,
    249,250,252,253,255,
];

#[rustfmt::skip]
static SNR_TABLE_MB: &[u8] = &[
                                          0,  0, 28, 43, 52, 59,
     65, 70, 74, 78, 81, 85, 87, 90, 93, 95, 98,100,102,105,107,109,
    111,113,115,116,118,120,122,123,125,127,128,130,131,133,134,136,
    137,138,140,141,143,144,145,147,148,149,151,152,153,154,156,157,
    158,159,160,162,163,164,165,166,167,168,169,171,172,173,174,175,
    176,177,178,179,180,181,182,183,184,185,186,187,188,188,189,190,
    191,192,193,194,195,196,197,198,199,200,201,202,203,203,204,205,
    206,207,208,209,210,211,212,213,214,214,215,216,217,218,219,220,
    221,222,223,224,224,225,226,227,228,229,230,231,232,233,234,235,
    236,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,
    251,252,253,254,255,
];

#[rustfmt::skip]
static SNR_TABLE_WB: &[u8] = &[
                                          0,  0,  0,  8, 29, 41,
     49, 56, 62, 66, 70, 74, 77, 80, 83, 86, 88, 91, 93, 95, 97, 99,
    101,103,105,107,108,110,112,113,115,116,118,119,121,122,123,125,
    126,127,129,130,131,132,134,135,136,137,138,140,141,142,143,144,
    145,146,147,148,149,150,151,152,153,154,156,157,158,159,159,160,
    161,162,163,164,165,166,167,168,169,170,171,171,172,173,174,175,
    176,177,177,178,179,180,181,181,182,183,184,185,185,186,187,188,
    189,189,190,191,192,192,193,194,195,195,196,197,198,198,199,200,
    200,201,202,203,203,204,205,206,206,207,208,209,209,210,211,211,
    212,213,214,214,215,216,216,217,218,219,219,220,221,221,222,223,
    224,224,225,226,226,227,228,229,229,230,231,232,232,233,234,234,
    235,236,237,237,238,239,240,240,241,242,243,243,244,245,246,246,
    247,248,249,249,250,251,252,253,255,
];

/// Map target bitrate to SNR_dB_Q7 (matches C reference `silk_control_SNR`).
pub fn silk_control_snr(fs_khz: i32, nb_subfr: i32, target_rate_bps: i32) -> i32 {
    let mut rate = target_rate_bps;
    if nb_subfr == 2 {
        rate -= 2000 + fs_khz / 16;
    }
    let table: &[u8] = match fs_khz {
        ..=8 => SNR_TABLE_NB,
        9..=12 => SNR_TABLE_MB,
        _ => SNR_TABLE_WB,
    };
    let id = ((rate + 200) / 400 - 10).clamp(0, table.len() as i32 - 1) as usize;
    table[id] as i32 * 21
}

/// Encoder control parameters (from API)
#[derive(Clone)]
pub struct SilkEncControl {
    /// API sampling rate in Hz (8000, 12000, 16000, 24000, 48000)
    pub api_sample_rate: i32,
    /// Maximum internal sample rate in Hz
    pub max_internal_fs_hz: i32,
    /// Payload size in milliseconds (10 or 20)
    pub payload_size_ms: i32,
    /// Target bitrate in bits per second
    pub bitrate_bps: i32,
    /// Maximum number of bits allowed for this frame (0 = no limit).
    /// Matches C reference `maxBits` in `silk_EncControlStruct`.
    pub max_bits: i32,
    /// Encoder complexity (0-10)
    pub complexity: i32,
    /// Whether to use in-band FEC (LBRR)
    pub use_in_band_fec: bool,
    /// Expected packet loss percentage (0-100)
    pub packet_loss_percentage: i32,
    /// Number of internal channels (1 = mono, 2 = stereo)
    pub n_channels_internal: i32,
    /// Last frame before a stereo->mono transition
    pub to_mono: bool,
}

impl Default for SilkEncControl {
    fn default() -> Self {
        Self {
            api_sample_rate: 16000,
            max_internal_fs_hz: 16000,
            payload_size_ms: 20,
            bitrate_bps: 25000,
            max_bits: 0,
            complexity: 2,
            use_in_band_fec: false,
            packet_loss_percentage: 0,
            n_channels_internal: 1,
            to_mono: false,
        }
    }
}

// Maximum buffer sizes for SILK encoder scratch (16kHz, 20ms frame)
const MAX_SILK_FRAME: usize = MAX_FRAME_LENGTH; // 320
const MAX_SILK_SUBFR: usize = MAX_SUB_FRAME_LENGTH; // 80
const MAX_SILK_LTP_MEM: usize = 320; // LTP_MEM_LENGTH_MS * MAX_FS_KHZ
const MAX_SILK_TOTAL: usize = MAX_SILK_LTP_MEM + MAX_SILK_FRAME; // 640
const MAX_SILK_WIN: usize = 240; // shape_win_length max (15 * 16)

/// Pre-allocated scratch buffers for SILK encoder (avoids per-frame heap allocations).
#[derive(Default)]
pub struct SilkScratch {
    pub delayed_input: Vec<i16>, // frame_length for resampler delay
    pub analysis_buf: Vec<i16>,  // MAX_SILK_TOTAL
    pub vad_x: Vec<i16>,         // ~800
    pub x_windowed: Vec<i16>,    // MAX_SILK_WIN
    pub nsq_s_ltp_q15: Vec<i32>, // MAX_SILK_TOTAL
    pub nsq_s_ltp: Vec<i16>,     // MAX_SILK_TOTAL
    pub nsq_x_sc_q10: Vec<i32>,  // MAX_SILK_SUBFR
    pub nsq_xq_tmp: Vec<i16>,    // MAX_SILK_SUBFR
    // LBRR NSQ scratch buffers (separate from main NSQ to avoid conflicts)
    pub lbrr_nsq_s_ltp_q15: Vec<i32>, // MAX_SILK_TOTAL
    pub lbrr_nsq_s_ltp: Vec<i16>,     // MAX_SILK_TOTAL
    pub lbrr_nsq_x_sc_q10: Vec<i32>,  // MAX_SILK_SUBFR
    pub lbrr_nsq_xq_tmp: Vec<i16>,    // MAX_SILK_SUBFR
}

impl SilkScratch {
    fn new() -> Self {
        Self {
            delayed_input: vec![0i16; MAX_FRAME_LENGTH],
            analysis_buf: vec![0i16; MAX_SILK_TOTAL],
            vad_x: vec![0i16; 800],
            x_windowed: vec![0i16; MAX_SILK_WIN],
            nsq_s_ltp_q15: vec![0i32; MAX_SILK_TOTAL],
            nsq_s_ltp: vec![0i16; MAX_SILK_TOTAL],
            nsq_x_sc_q10: vec![0i32; MAX_SILK_SUBFR],
            nsq_xq_tmp: vec![0i16; MAX_SILK_SUBFR],
            lbrr_nsq_s_ltp_q15: vec![0i32; MAX_SILK_TOTAL],
            lbrr_nsq_s_ltp: vec![0i16; MAX_SILK_TOTAL],
            lbrr_nsq_x_sc_q10: vec![0i32; MAX_SILK_SUBFR],
            lbrr_nsq_xq_tmp: vec![0i16; MAX_SILK_SUBFR],
        }
    }
}

/// Per-channel encoder state
struct EncChannelState {
    // Frame configuration
    fs_khz: i32,
    nb_subfr: i32,
    frame_length: i32,
    subfr_length: i32,
    ltp_mem_length: i32,
    lpc_order: i32,

    // NLSF codebook selection
    nlsf_cb_sel: NlsfCbSel,
    pitch_contour_sel: PitchContourSel,
    pitch_lag_low_bits_sel: PitchLagLowBitsSel,

    // Previous frame state
    prev_nlsf_q15: [i16; MAX_LPC_ORDER],
    last_gain_index: i8,
    prev_signal_type: i32,
    ec_prev_signal_type: i32,
    ec_prev_lag_index: i16,
    first_frame_after_reset: bool,

    // Indices
    indices: SideInfoIndices,

    // NSQ state
    nsq_state: NsqState,

    // VAD state
    vad_state: vad::VadState,
    speech_activity_q8: i32,
    snr_db_q7: i32,

    // Noise shape state (smoothing across frames)
    prev_tilt_smth_q16: i32,
    prev_harm_smth_q16: i32,
    shaping_lpc_order: i32,
    warping_q16: i32,

    // Resampler delay buffer (C: silk_resampler_state_struct.delayBuf)
    // Even for same-rate "copy", the C reference introduces a delay of
    // inputDelay = delay_matrix_enc[rateID(fs)][rateID(fs)] samples.
    // For 16kHz: inputDelay = 10.
    resampler_delay_buf: [i16; 16], // Fs_in_kHz max
    resampler_input_delay: usize,

    // Input buffer (for LPC analysis history)
    input_buf: Vec<i16>,
    input_buf_idx: usize,

    // Frame counter
    n_frames_encoded: i32,

    // Pitch analysis state (for lag tracking across frames)
    prev_lag: i32,
    ltp_corr_q15: i32,

    // LBRR state
    lbrr_enabled: bool,
    lbrr_gain_increases: i32,
    lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    lbrr_prev_last_gain_index: i8,
    indices_lbrr: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    pulses_lbrr: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],

    // Previous packet LBRR data (written at start of next packet)
    prev_lbrr_flags: [i32; MAX_FRAMES_PER_PACKET],
    prev_indices_lbrr: [SideInfoIndices; MAX_FRAMES_PER_PACKET],
    prev_pulses_lbrr: [[i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
    prev_lbrr_any: bool,
    prev_n_frames_per_packet: i32,
}

impl EncChannelState {
    fn new() -> Self {
        Self {
            fs_khz: 0,
            nb_subfr: 0,
            frame_length: 0,
            subfr_length: 0,
            ltp_mem_length: 0,
            lpc_order: 0,
            nlsf_cb_sel: NlsfCbSel::NbMb,
            pitch_contour_sel: PitchContourSel::Nb,
            pitch_lag_low_bits_sel: PitchLagLowBitsSel::Uniform4,
            prev_nlsf_q15: [0; MAX_LPC_ORDER],
            last_gain_index: 10,
            prev_signal_type: TYPE_NO_VOICE_ACTIVITY,
            ec_prev_signal_type: 0,
            ec_prev_lag_index: 0,
            first_frame_after_reset: true,
            indices: SideInfoIndices::default(),
            nsq_state: NsqState::new(),
            vad_state: vad::VadState::default(),
            resampler_delay_buf: [0; 16],
            resampler_input_delay: 0, // Set by set_fs based on sample rate
            speech_activity_q8: 128,
            snr_db_q7: 0,
            prev_tilt_smth_q16: 0,
            prev_harm_smth_q16: 0,
            shaping_lpc_order: 16,
            warping_q16: 0,
            input_buf: vec![0i16; MAX_FRAME_LENGTH + 2 * MAX_SUB_FRAME_LENGTH],
            input_buf_idx: 0,
            n_frames_encoded: 0,
            prev_lag: 0,
            ltp_corr_q15: 0,

            // LBRR state
            lbrr_enabled: false,
            lbrr_gain_increases: 0,
            lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            lbrr_prev_last_gain_index: 10,
            indices_lbrr: Default::default(),
            pulses_lbrr: [[0i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],

            // Previous packet LBRR data
            prev_lbrr_flags: [0; MAX_FRAMES_PER_PACKET],
            prev_indices_lbrr: Default::default(),
            prev_pulses_lbrr: [[0i8; MAX_FRAME_LENGTH]; MAX_FRAMES_PER_PACKET],
            prev_lbrr_any: false,
            prev_n_frames_per_packet: 1,
        }
    }

    fn set_fs(&mut self, fs_khz: i32, payload_size_ms: i32) {
        // Reset state on sample rate change
        if self.fs_khz != fs_khz {
            self.first_frame_after_reset = true;
            self.prev_nlsf_q15 = [0; MAX_LPC_ORDER];
            self.last_gain_index = 10;
            // Resampler delay matching C reference: delay_matrix_enc[rateID][rateID]
            // 8kHz→6, 12kHz→7, 16kHz→10
            self.resampler_input_delay = match fs_khz {
                8 => 6,
                12 => 7,
                16 => 10,
                _ => 10,
            } as usize;
            self.resampler_delay_buf = [0; 16];
        }

        self.fs_khz = fs_khz;
        self.nb_subfr = if payload_size_ms == 10 {
            2
        } else {
            MAX_NB_SUBFR as i32
        };
        self.subfr_length = SUB_FRAME_LENGTH_MS as i32 * fs_khz;
        self.frame_length = self.nb_subfr * self.subfr_length;
        self.ltp_mem_length = LTP_MEM_LENGTH_MS as i32 * fs_khz;

        if fs_khz == 8 || fs_khz == 12 {
            self.lpc_order = MIN_LPC_ORDER as i32;
            self.nlsf_cb_sel = NlsfCbSel::NbMb;
        } else {
            self.lpc_order = MAX_LPC_ORDER as i32;
            self.nlsf_cb_sel = NlsfCbSel::Wb;
        }

        if fs_khz == 8 {
            self.pitch_contour_sel = if self.nb_subfr == MAX_NB_SUBFR as i32 {
                PitchContourSel::Nb
            } else {
                PitchContourSel::Nb10ms
            };
        } else {
            self.pitch_contour_sel = if self.nb_subfr == MAX_NB_SUBFR as i32 {
                PitchContourSel::Wb
            } else {
                PitchContourSel::Wb10ms
            };
        }

        self.pitch_lag_low_bits_sel = match fs_khz {
            16 => PitchLagLowBitsSel::Uniform8,
            12 => PitchLagLowBitsSel::Uniform6,
            _ => PitchLagLowBitsSel::Uniform4,
        };

        // Shaping LPC order based on complexity (from silk/control_codec.c)
        // and warping for bilinear transform
        self.shaping_lpc_order = 16; // default, updated in encode() based on complexity
        self.warping_q16 = 0; // default, updated based on complexity >= 4
    }
}

/// Top-level SILK encoder (mono or stereo)
#[allow(dead_code)]
pub struct SilkEncoder {
    state: EncChannelState,
    /// Second channel state for stereo (side channel)
    state_side: EncChannelState,
    /// Stereo encoder state for L/R to M/S conversion
    stereo_state: crate::stereo_encode::StereoEncState,
    initialized: bool,
    scratch: SilkScratch,
    /// Scratch buffers for side channel (stereo only)
    scratch_side: SilkScratch,
    /// Number of internal channels currently configured
    n_channels_internal: i32,
    /// Previous frame's n_channels_internal
    prev_n_channels_internal: i32,
    /// Previous frame's decode_only_middle flag
    pub prev_decode_only_middle: bool,
    /// Speech activity from previous frame (for stereo smoothing)
    prev_speech_activity_q8: i32,
}

impl SilkEncoder {
    /// Create a new SILK encoder
    pub fn new() -> Self {
        Self {
            state: EncChannelState::new(),
            state_side: EncChannelState::new(),
            stereo_state: crate::stereo_encode::StereoEncState::default(),
            initialized: false,
            scratch: SilkScratch::new(),
            scratch_side: SilkScratch::new(),
            n_channels_internal: 1,
            prev_n_channels_internal: 1,
            prev_decode_only_middle: false,
            prev_speech_activity_q8: 128,
        }
    }

    /// Encode one SILK frame.
    ///
    /// Returns the number of bytes written, or a negative error code.
    /// The encoder writes into the provided range coder `enc`.
    pub fn encode(&mut self, control: &SilkEncControl, enc: &mut EcCtx, samples: &[i16]) -> i32 {
        // Take scratch buffers to avoid borrow conflicts with self.state
        let mut scratch = std::mem::take(&mut self.scratch);
        let cs = &mut self.state;

        // Determine internal sampling rate
        let fs_khz = match control.max_internal_fs_hz {
            ..=8000 => 8,
            8001..=12000 => 12,
            _ => 16,
        };

        // Initialize / reconfigure if needed
        if !self.initialized || cs.fs_khz != fs_khz {
            cs.set_fs(fs_khz, control.payload_size_ms);
            self.initialized = true;
        }

        let frame_length = cs.frame_length as usize;
        let subfr_length = cs.subfr_length as usize;
        let nb_subfr = cs.nb_subfr as usize;
        let lpc_order = cs.lpc_order as usize;
        let ltp_mem_length = cs.ltp_mem_length as usize;

        // Number of frames per packet
        let n_frames_per_packet: i32 = 1;

        // Ensure we have enough input
        if samples.len() < frame_length {
            return -1;
        }

        // ====== LBRR setup (from silk/control_codec.c silk_setup_LBRR) ======
        let lbrr_in_previous = cs.lbrr_enabled;
        cs.lbrr_enabled = control.use_in_band_fec && control.packet_loss_percentage > 0;
        if cs.lbrr_enabled {
            // 13107 = SILK_FIX_CONST(0.2, 16) -- maps percentage to gain increase
            cs.lbrr_gain_increases = if !lbrr_in_previous {
                7
            } else {
                (7 - silk_smulwb(control.packet_loss_percentage, 13107)).max(2)
            };
        }

        // At the start of a new packet (n_frames_encoded == 0), reset LBRR flags
        if cs.n_frames_encoded == 0 {
            cs.lbrr_flags = [0; MAX_FRAMES_PER_PACKET];
            cs.lbrr_prev_last_gain_index = cs.last_gain_index;
        }

        // Build analysis buffer: history + current frame
        let analysis_buf_len = ltp_mem_length + frame_length;
        let analysis_buf = &mut scratch.analysis_buf[..analysis_buf_len];
        // Copy history from input_buf
        let history_len = ltp_mem_length.min(cs.input_buf_idx);
        if history_len > 0 {
            let src_start = cs.input_buf_idx - history_len;
            analysis_buf[ltp_mem_length - history_len..ltp_mem_length]
                .copy_from_slice(&cs.input_buf[src_start..cs.input_buf_idx]);
        }
        // Copy current frame
        analysis_buf[ltp_mem_length..ltp_mem_length + frame_length]
            .copy_from_slice(&samples[..frame_length]);

        // Update input buffer with current frame for next call's history
        if cs.input_buf_idx + frame_length > cs.input_buf.len() {
            let shift = cs.input_buf_idx + frame_length - cs.input_buf.len();
            cs.input_buf.copy_within(shift..cs.input_buf_idx, 0);
            cs.input_buf_idx -= shift;
        }
        let start = cs.input_buf_idx;
        let end = start + frame_length;
        if end <= cs.input_buf.len() {
            cs.input_buf[start..end].copy_from_slice(&samples[..frame_length]);
            cs.input_buf_idx = end;
        }

        // Apply the same-rate resampler delay as the C reference (silk_resampler copy mode).
        // Even for 16kHz→16kHz, the C resampler delays by inputDelay=10 samples using
        // a delay buffer. This shifts the signal phase and is essential for matching
        // the C encoder's LPC analysis and gain trajectory.
        let delayed = &mut scratch.delayed_input[..frame_length];
        {
            let delay = cs.resampler_input_delay;
            let fs_khz_u = cs.fs_khz as usize;
            let n_copy = fs_khz_u.saturating_sub(delay);

            // First Fs_kHz samples: delayBuf[0..delay] + input[0..n_copy]
            delayed[..delay].copy_from_slice(&cs.resampler_delay_buf[..delay]);
            delayed[delay..fs_khz_u].copy_from_slice(&samples[..n_copy]);

            // Remaining samples: direct from input[n_copy..]
            if frame_length > fs_khz_u {
                let remaining = frame_length - fs_khz_u;
                delayed[fs_khz_u..frame_length]
                    .copy_from_slice(&samples[n_copy..n_copy + remaining]);
            }

            // Save last inputDelay samples for next frame's delay buffer
            cs.resampler_delay_buf[..delay]
                .copy_from_slice(&samples[frame_length - delay..frame_length]);
        }

        // Use the delayed signal for ALL subsequent processing (matching C encoder's inputBuf+1).
        let samples = &delayed[..frame_length];

        // ====== Analysis ======

        // 1. LPC analysis: compute autocorrelation -> Levinson-Durbin -> LPC coefficients
        let mut corr = [0i32; MAX_LPC_ORDER + 1];
        lpc_analysis::silk_autocorrelation(
            &mut corr,
            &analysis_buf[ltp_mem_length..],
            frame_length,
            lpc_order,
        );

        let mut a_q16 = [0i32; MAX_LPC_ORDER];
        let _pred_gain = lpc_analysis::silk_levinson_durbin(&mut a_q16, &corr, lpc_order);

        // Convert to Q12 for the filter
        let mut a_q12 = [0i16; MAX_LPC_ORDER];
        for i in 0..lpc_order {
            a_q12[i] = silk_rshift_round(a_q16[i], 4) as i16;
        }

        // 2. Convert LPC to NLSF
        let mut nlsf_q15 = [0i16; MAX_LPC_ORDER];
        lpc_analysis::silk_a2nlsf(&mut nlsf_q15, &mut a_q16, lpc_order);

        // 3. Quantize NLSFs
        let mut w_q2 = [0i16; MAX_LPC_ORDER];
        lpc_analysis::silk_nlsf_vq_weights_laroia(&mut w_q2, &nlsf_q15, lpc_order);

        let nlsf_cb = get_nlsf_cb(cs.nlsf_cb_sel);
        cs.indices.nlsf_indices = [0i8; MAX_LPC_ORDER + 1];
        // NLSF mu from C reference: 6 for voiced, 4 for unvoiced (scaled by 2^20)
        // Use previous signal type since pitch analysis hasn't run yet for this frame
        let prev_voiced = cs.prev_signal_type == TYPE_VOICED;
        let nlsf_mu_q20 = if prev_voiced {
            6 << 20 >> 2
        } else {
            4 << 20 >> 2
        };
        // Number of NLSF survivors based on complexity (from silk/control_codec.c)
        let nlsf_survivors = match control.complexity {
            0 => 2,
            1 => 3,
            2 => 2,
            3 => 4,
            4 | 5 => 6,
            6 | 7 => 8,
            _ => 16,
        };
        nlsf_encode::silk_nlsf_encode(
            &mut cs.indices.nlsf_indices,
            &mut nlsf_q15,
            nlsf_cb,
            &w_q2,
            nlsf_mu_q20,
            nlsf_survivors,
            cs.indices.signal_type as i32,
        );

        // 4. Convert quantized NLSFs back to LPC for filtering
        let mut pred_coef_q12 = [0i16; 2 * MAX_LPC_ORDER];
        silk_nlsf2a(
            &mut pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order],
            &nlsf_q15,
            lpc_order,
        );
        // For interpolation: use the same coefficients for both halves (simplified)
        // Copy second half to first half via temp to satisfy borrow checker
        let mut tmp_coefs = [0i16; MAX_LPC_ORDER];
        tmp_coefs[..lpc_order]
            .copy_from_slice(&pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order]);
        pred_coef_q12[..lpc_order].copy_from_slice(&tmp_coefs[..lpc_order]);

        // nlsf_interp_coef_q2 is set by the interpolation search in Step 3

        // 5. Pitch analysis using full 3-stage hierarchical search
        let mut pitch_lags = [0i32; MAX_NB_SUBFR];

        // Derive pitch estimation complexity from encoder complexity (matches C reference)
        let pe_complexity = match control.complexity {
            0 => 0,
            1 => 1,
            2 => 0,
            3 => 1,
            4..=7 => 1,
            _ => 2,
        };

        // Derive search threshold from complexity (matches C reference)
        let search_thres1_q16 = match control.complexity {
            0 | 2 => 52429, // SILK_FIX_CONST(0.8, 16)
            1 | 3 => 49807, // SILK_FIX_CONST(0.76, 16)
            4 | 5 => 48497, // SILK_FIX_CONST(0.74, 16)
            6 | 7 => 47186, // SILK_FIX_CONST(0.72, 16)
            _ => 45875,     // SILK_FIX_CONST(0.7, 16)
        };
        let search_thres2_q13 = 2458; // SILK_FIX_CONST(0.3, 13)

        let ret = pitch_analysis::silk_pitch_analysis_core(
            &analysis_buf[..analysis_buf_len],
            &mut pitch_lags,
            &mut cs.indices.lag_index,
            &mut cs.indices.contour_index,
            &mut cs.ltp_corr_q15,
            cs.prev_lag,
            search_thres1_q16,
            search_thres2_q13,
            cs.fs_khz,
            pe_complexity,
            cs.nb_subfr,
        );

        let voiced = ret == 0;

        // Set signal type
        if voiced {
            cs.indices.signal_type = TYPE_VOICED as i8;
            cs.prev_lag = pitch_lags[nb_subfr - 1]; // Track lag for next frame
        } else {
            cs.indices.signal_type = TYPE_UNVOICED as i8;
            cs.prev_lag = 0;
        }
        cs.indices.quant_offset_type = 0; // Low offset

        // 6. LTP analysis (for voiced frames)
        // Note: lag_index and contour_index are already set by silk_pitch_analysis_core
        if voiced {
            // LTP analysis: find per_index and ltp_index
            pitch_analysis::silk_find_ltp_params(
                &mut cs.indices.ltp_index,
                &mut cs.indices.per_index,
                &pitch_lags,
                analysis_buf,
                &pred_coef_q12[MAX_LPC_ORDER..],
                cs.subfr_length,
                cs.nb_subfr,
                cs.ltp_mem_length,
                cs.lpc_order,
            );

            cs.indices.ltp_scale_index = 0;
        }

        // 7. Build LTP coefficients from indices
        let mut ltp_coef_q14 = [0i16; MAX_NB_SUBFR * LTP_ORDER];
        if voiced {
            let cbk = SILK_LTP_VQ_PTRS_Q7[cs.indices.per_index as usize];
            for k in 0..nb_subfr {
                let entry = &cbk[cs.indices.ltp_index[k] as usize];
                for j in 0..LTP_ORDER {
                    ltp_coef_q14[k * LTP_ORDER + j] = (entry[j] as i16) << 7;
                }
            }
        }

        // Set complexity-dependent shaping parameters
        cs.shaping_lpc_order = match control.complexity {
            0 => 12,
            1 | 3 => 14,
            2 => 12,
            4 | 5 => 16,
            6 | 7 => 20,
            _ => 24,
        };
        cs.warping_q16 = if control.complexity >= 4 {
            ((0.015 * cs.fs_khz as f64) * 65536.0) as i32
        } else {
            0
        };

        // Run VAD for speech activity estimation
        let mut quality_bands_q15 = [0i32; 4];
        let mut tilt_q15 = 0i32;
        vad::silk_vad_get_sa_q8(
            &mut cs.vad_state,
            &mut cs.speech_activity_q8,
            &mut cs.snr_db_q7,
            &mut quality_bands_q15,
            &mut tilt_q15,
            &samples[..frame_length],
            cs.frame_length,
        );

        // Compute SNR from target bitrate (matches C reference silk_control_SNR)
        let snr_for_shaping = silk_control_snr(cs.fs_khz, cs.nb_subfr, control.bitrate_bps);

        // Noise shape analysis: compute spectral shaping filter parameters AND gains.
        // The noise_shape_analysis internally computes gains from the signal's
        // autocorrelation, then adjusts them using the bitrate-derived SNR via
        // gain_mult_q16 and gain_add_q16 (matching C reference process_gains).
        // C reference: noise_shape receives x_frame (which starts at x_buf + ltp_mem_length)
        // and internally accesses x_ptr = x - la_shape (80 samples before x_frame).
        // We pass analysis_buf starting la_shape samples before the frame,
        // so the noise_shape's x_ptr_start can access the lookback.
        let la_shape = 5 * cs.fs_khz as usize; // LA_SHAPE_MS * fs_kHz
        let ns_offset = ltp_mem_length.saturating_sub(la_shape);
        let ns_input = &analysis_buf[ns_offset..ltp_mem_length + frame_length];
        // The noise_shape sees: ns_input[0..la_shape] = lookback, ns_input[la_shape..] = frame
        // But our noise_shape function indexes from 0 at the frame start.
        // We pass with la_shape prefix so the function can window backward.
        let shape_result = noise_shape_analysis::silk_noise_shape_analysis(
            ns_input,
            &pitch_lags,
            voiced,
            &mut cs.prev_tilt_smth_q16,
            &mut cs.prev_harm_smth_q16,
            cs.fs_khz,
            cs.nb_subfr,
            cs.subfr_length,
            cs.frame_length,
            cs.lpc_order,
            cs.shaping_lpc_order,
            cs.warping_q16,
            cs.speech_activity_q8,
            0, // coding_quality_q14: computed internally from SNR
            snr_for_shaping,
        );

        // ====== find_pred_coefs: gain normalize → recompute LPC → residual energy ======
        // Matches C reference find_pred_coefs_FIX.c.
        // Uses noise_shape gains to normalize the input, recomputes LPC on the
        // normalized input, then measures residual energy for process_gains.

        // Use noise_shape gains directly (fixed-point), converted to float.
        // The noise_shape computes per-subframe Schur residual energy and applies
        // gain_mult/gain_add in fixed-point. The float vs fixed-point difference
        // in gain_mult/gain_add is <0.4% — negligible compared to the Schur energy.
        let mut gains_q16 = [0i32; MAX_NB_SUBFR];
        gains_q16[..nb_subfr].copy_from_slice(&shape_result.gains_q16[..nb_subfr]);
        let mut gains_f = [0.0f32; MAX_NB_SUBFR];
        for k in 0..nb_subfr {
            gains_f[k] = gains_q16[k] as f32 / 65536.0;
        }

        // Step 1: Compute invGains (C float: find_pred_coefs_FLP.c)
        let mut inv_gains_f = [0.0f32; MAX_NB_SUBFR];
        for k in 0..nb_subfr {
            inv_gains_f[k] = 1.0 / gains_f[k].max(1e-12);
        }

        // Step 2: Create gain-normalized LPC_in_pre (float, matching C float encoder)
        // C: silk_scale_copy_vector_FLP(x_pre_ptr, x_ptr, invGains[i], len)
        let lpc_pre_len = nb_subfr * (lpc_order + subfr_length);
        let mut lpc_in_pre_f = vec![0.0f32; lpc_pre_len];
        {
            let mut pre_idx = 0usize;
            for (k, &ig_f) in inv_gains_f.iter().enumerate().take(nb_subfr) {
                let body_start = ltp_mem_length + k * subfr_length;
                let src_start = body_start - lpc_order;
                for j in 0..(lpc_order + subfr_length) {
                    let si = src_start + j;
                    if si < analysis_buf.len() && pre_idx < lpc_pre_len {
                        lpc_in_pre_f[pre_idx] = ig_f * (analysis_buf[si] as f32);
                    }
                    pre_idx += 1;
                }
            }
        }

        // Step 3: Recompute LPC on gain-normalized input using Burg's modified algorithm
        // Matches C reference: find_LPC_FIX.c line 57-63.
        // Burg receives subfr_length = predictLPCOrder + actual_subfr_length (includes D history).
        {
            let burg_subfr = lpc_order + subfr_length;

            // Float minInvGain (C: find_pred_coefs_FLP.c lines 97-100)
            let coding_quality_f = shape_result.coding_quality_q14 as f32 / 16384.0;
            let ltp_pred_cod_gain_f = 0.0f32; // TODO: compute for voiced
            let min_inv_gain_f: f32 = if cs.first_frame_after_reset {
                1.0 / 100.0 // MAX_PREDICTION_POWER_GAIN_AFTER_RESET
            } else {
                let base = 2.0f32.powf(ltp_pred_cod_gain_f / 3.0) / 10000.0; // MAX_PREDICTION_POWER_GAIN
                base / (0.25 + 0.75 * coding_quality_f)
            };

            let mut a_flp = [0.0f32; MAX_LPC_ORDER];
            let _burg_res_nrg_f = lpc_analysis::silk_burg_modified_flp(
                &mut a_flp,
                &lpc_in_pre_f,
                min_inv_gain_f,
                burg_subfr,
                nb_subfr,
                lpc_order,
            );

            // Convert float coefficients to Q16 (C: silk_A2NLSF_FLP does this)
            // silk_float2int rounds to nearest integer
            let mut a_q16 = [0i32; MAX_LPC_ORDER];
            for i in 0..lpc_order {
                a_q16[i] = (a_flp[i] as f64 * 65536.0).round() as i32;
            }

            // Default: no interpolation
            cs.indices.nlsf_interp_coef_q2 = 4;

            let mut new_nlsf_q15 = [0i16; MAX_LPC_ORDER];

            // NLSF interpolation search (C: find_LPC_FIX.c lines 65-141)
            let use_interpolated =
                control.complexity >= 5 && !cs.first_frame_after_reset && nb_subfr == MAX_NB_SUBFR;

            // Float residual energy for interpolation comparison
            let mut burg_res_nrg_f = _burg_res_nrg_f;

            if use_interpolated {
                // Float Burg on last 2 subframes (second half)
                let mut a_tmp_flp = [0.0f32; MAX_LPC_ORDER];
                let res_tmp_nrg_f = lpc_analysis::silk_burg_modified_flp(
                    &mut a_tmp_flp,
                    &lpc_in_pre_f[2 * burg_subfr..],
                    min_inv_gain_f,
                    burg_subfr,
                    2,
                    lpc_order,
                );

                // Subtract second-half energy from full-frame energy
                burg_res_nrg_f -= res_tmp_nrg_f;

                // Convert second-half float LPC to Q16 then NLSFs
                let mut a_tmp_q16 = [0i32; MAX_LPC_ORDER];
                for i in 0..lpc_order {
                    a_tmp_q16[i] = (a_tmp_flp[i] as f64 * 65536.0).round() as i32;
                }
                lpc_analysis::silk_a2nlsf(&mut new_nlsf_q15, &mut a_tmp_q16, lpc_order);

                // Search interpolation coefficients k=3,2,1,0 (C: find_LPC_FLP.c)
                // Uses float residual energy comparison for simplicity and C-reference matching.
                let mut best_res_nrg_f = burg_res_nrg_f;
                let mut res_nrg_2nd = f32::MAX;

                for k in (0..=3i32).rev() {
                    // Interpolate NLSFs for first half
                    let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
                    for i in 0..lpc_order {
                        nlsf0_q15[i] = (cs.prev_nlsf_q15[i] as i32
                            + ((k * (new_nlsf_q15[i] as i32 - cs.prev_nlsf_q15[i] as i32)) >> 2))
                            as i16;
                    }

                    // Convert to float LPC for residual energy evaluation
                    let mut a_tmp_q12 = [0i16; MAX_LPC_ORDER];
                    silk_nlsf2a(&mut a_tmp_q12, &nlsf0_q15, lpc_order);
                    let mut a_tmp_flp2 = [0.0f32; MAX_LPC_ORDER];
                    for i in 0..lpc_order {
                        a_tmp_flp2[i] = a_tmp_q12[i] as f32 / 4096.0;
                    }

                    // LPC analysis filter on first 2 subframes (float)
                    let filter_len = 2 * burg_subfr;
                    let mut lpc_res_f = vec![0.0f32; filter_len];
                    for ix in lpc_order..filter_len.min(lpc_in_pre_f.len()) {
                        let mut sum = lpc_in_pre_f[ix];
                        for j in 0..lpc_order {
                            if ix > j {
                                sum -= a_tmp_flp2[j] * lpc_in_pre_f[ix - j - 1];
                            }
                        }
                        lpc_res_f[ix] = sum;
                    }

                    // Measure residual energy of first 2 subframes
                    let mut nrg0: f64 = 0.0;
                    for item in lpc_res_f.iter().take(burg_subfr).skip(lpc_order) {
                        nrg0 += (*item as f64) * (*item as f64);
                    }
                    let mut nrg1: f64 = 0.0;
                    for item in lpc_res_f
                        .iter()
                        .take(filter_len.min(lpc_res_f.len()))
                        .skip(lpc_order + burg_subfr)
                    {
                        nrg1 += (*item as f64) * (*item as f64);
                    }
                    let res_nrg_interp = (nrg0 + nrg1) as f32;

                    // Compare (C: find_LPC_FLP.c lines 76-86)
                    if res_nrg_interp < best_res_nrg_f {
                        best_res_nrg_f = res_nrg_interp;
                        cs.indices.nlsf_interp_coef_q2 = k as i8;
                    } else if res_nrg_interp > res_nrg_2nd {
                        break; // No reason to continue
                    }
                    res_nrg_2nd = res_nrg_interp;
                }
            }

            if cs.indices.nlsf_interp_coef_q2 == 4 {
                // No interpolation — convert full-frame Burg LPC to NLSFs
                lpc_analysis::silk_a2nlsf(&mut new_nlsf_q15, &mut a_q16, lpc_order);
            }
            // else: new_nlsf_q15 already contains second-half NLSFs from above

            let mut w_q2 = [0i16; MAX_LPC_ORDER];
            lpc_analysis::silk_nlsf_vq_weights_laroia(&mut w_q2, &new_nlsf_q15, lpc_order);

            let nlsf_cb = get_nlsf_cb(cs.nlsf_cb_sel);
            cs.indices.nlsf_indices = [0i8; MAX_LPC_ORDER + 1];
            let _prev_voiced = cs.prev_signal_type == TYPE_VOICED;
            // C reference: NLSF_mu_Q20 = 0.003 - 0.001 * speech_activity (Q20)
            // speech_activity_Q8 ≈ 256 (active signal). For 10ms: multiply by 1.5.
            // Range: [2098, 3146] for 20ms packets.
            let speech_activity_q8: i32 = 256; // TODO: compute from VAD
            let mut nlsf_mu_q20 = silk_smlawb(3146, -268435, speech_activity_q8); // 0.003 - 0.001*activity
            if nb_subfr == 2 {
                nlsf_mu_q20 += nlsf_mu_q20 >> 1; // 1.5x for 10ms
            }
            let nlsf_survivors = match control.complexity {
                0 => 2,
                1 => 3,
                2 => 2,
                3 => 4,
                4 | 5 => 6,
                6 | 7 => 8,
                _ => 16,
            };
            nlsf_encode::silk_nlsf_encode(
                &mut cs.indices.nlsf_indices,
                &mut new_nlsf_q15,
                nlsf_cb,
                &w_q2,
                nlsf_mu_q20,
                nlsf_survivors,
                cs.indices.signal_type as i32,
            );

            // Convert quantized NLSFs back to LPC Q12 for second half
            // C: silk_NLSF2A(PredCoef_Q12[1], pNLSF_Q15, ...)
            silk_nlsf2a(
                &mut pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order],
                &new_nlsf_q15,
                lpc_order,
            );

            // First half: interpolate or copy (C: process_NLSFs.c lines 94-106)
            if cs.indices.nlsf_interp_coef_q2 < 4 {
                // Interpolate between prev_NLSFq and current quantized NLSFs
                let k = cs.indices.nlsf_interp_coef_q2 as i32;
                let mut nlsf0_q15 = [0i16; MAX_LPC_ORDER];
                for i in 0..lpc_order {
                    nlsf0_q15[i] = (cs.prev_nlsf_q15[i] as i32
                        + ((k * (new_nlsf_q15[i] as i32 - cs.prev_nlsf_q15[i] as i32)) >> 2))
                        as i16;
                }
                silk_nlsf2a(&mut pred_coef_q12[..lpc_order], &nlsf0_q15, lpc_order);
            } else {
                // No interpolation — copy second half to first half
                let mut tmp_coefs = [0i16; MAX_LPC_ORDER];
                tmp_coefs[..lpc_order]
                    .copy_from_slice(&pred_coef_q12[MAX_LPC_ORDER..MAX_LPC_ORDER + lpc_order]);
                pred_coef_q12[..lpc_order].copy_from_slice(&tmp_coefs[..lpc_order]);
            }

            nlsf_q15[..lpc_order].copy_from_slice(&new_nlsf_q15[..lpc_order]);
        }

        // Step 4: Float residual energy (C: residual_energy_FLP.c)
        // nrgs[k] = gains[k]² * energy(LPC_residual_of_subframe_k)
        // Uses per-half-frame LPC (a[0] for first half, a[1] for second half)
        let mut res_nrg_f = [0.0f32; MAX_NB_SUBFR];
        {
            let shift = lpc_order + subfr_length; // offset per subframe in lpc_in_pre_f
            let half_subfr = MAX_NB_SUBFR / 2;

            for half in 0..(nb_subfr / half_subfr).max(1) {
                // Select LPC coefficients for this half-frame
                let a_q12_offset = if half >= 1 || cs.indices.nlsf_interp_coef_q2 == 4 {
                    MAX_LPC_ORDER
                } else {
                    0
                };
                let a_q12 = &pred_coef_q12[a_q12_offset..a_q12_offset + lpc_order];
                // Convert Q12 LPC to float for filtering
                let mut a_flp_filt = [0.0f32; MAX_LPC_ORDER];
                for i in 0..lpc_order {
                    a_flp_filt[i] = a_q12[i] as f32 / 4096.0;
                }

                // Float LPC analysis filter on 2 subframes
                let start_sf = half * half_subfr;
                let n_sf = half_subfr.min(nb_subfr - start_sf);
                let filter_len = n_sf * shift;
                let pre_start = start_sf * shift;
                let mut lpc_res_f = vec![0.0f32; filter_len];

                // silk_LPC_analysis_filter_FLP
                for ix in lpc_order..filter_len.min(lpc_in_pre_f.len() - pre_start) {
                    let mut sum = lpc_in_pre_f[pre_start + ix];
                    for j in 0..lpc_order {
                        sum -= a_flp_filt[j] * lpc_in_pre_f[pre_start + ix - j - 1];
                    }
                    lpc_res_f[ix] = sum;
                }

                // Measure per-subframe energy, scaled by gains² (C: residual_energy_FLP.c)
                for j in 0..n_sf {
                    let sf_idx = start_sf + j;
                    let res_start = j * shift + lpc_order;
                    let mut energy: f64 = 0.0;
                    for i in 0..subfr_length {
                        if res_start + i < lpc_res_f.len() {
                            let v = lpc_res_f[res_start + i] as f64;
                            energy += v * v;
                        }
                    }
                    // C: nrgs[k] = gains[k] * gains[k] * silk_energy_FLP(...)
                    res_nrg_f[sf_idx] =
                        (gains_f[sf_idx] as f64 * gains_f[sf_idx] as f64 * energy) as f32;
                }
            }
        }

        // Step 5: Float process_gains (C: process_gains_FLP.c)
        // gain = sqrt(gain² + ResNrg * InvMaxSqrVal), all in float
        {
            let snr_db = snr_for_shaping as f32 / 128.0;
            let inv_max_sqr_val: f32 = 2.0f32.powf(0.33 * (21.0 - snr_db)) / (subfr_length as f32);

            for k in 0..nb_subfr {
                let gain = gains_f[k];
                let new_gain = (gain * gain + res_nrg_f[k] * inv_max_sqr_val).sqrt();
                gains_f[k] = new_gain.min(32767.0);
            }

            // Convert float gains to Q16 for quantization
            for k in 0..nb_subfr {
                gains_q16[k] = (gains_f[k] * 65536.0) as i32;
            }
        }

        // Save unquantized gains for the iterative bitrate loop
        let gains_unq_q16 = gains_q16;

        // Determine coding mode
        let cond_coding = if cs.n_frames_encoded > 0 && !cs.first_frame_after_reset {
            CODE_CONDITIONALLY
        } else {
            CODE_INDEPENDENTLY
        };

        // Quantize gains
        gain_quant::silk_gains_quant(
            &mut cs.indices.gains_indices,
            &mut gains_q16,
            &mut cs.last_gain_index,
            cond_coding == CODE_CONDITIONALLY,
            nb_subfr,
        );

        // Use noise shape results
        let ar_q13 = shape_result.ar_q13;
        let harm_shape_gain_q14 = &shape_result.harm_shape_gain_q14[..nb_subfr];
        let tilt_q14 = &shape_result.tilt_q14[..nb_subfr];
        let lf_shp_q14 = &shape_result.lf_shp_q14[..nb_subfr];
        let lambda_q10 = shape_result.lambda_q10;

        // Use noise shape analysis quant offset type
        cs.indices.quant_offset_type = shape_result.quant_offset_type;

        let ltp_scale_q14 = if cond_coding == CODE_INDEPENDENTLY {
            SILK_LTP_SCALES_TABLE_Q14[cs.indices.ltp_scale_index as usize] as i32
        } else {
            SILK_LTP_SCALES_TABLE_Q14[0] as i32
        };

        // Set random seed
        cs.indices.seed = (cs.n_frames_encoded & 3) as i8;

        // ====== LBRR encoding (before main NSQ, so we can clone NSQ state) ======
        let frame_idx = cs.n_frames_encoded as usize;
        if cs.lbrr_enabled {
            // Only encode LBRR for frames with sufficient speech activity
            // Threshold: speech_activity_q8 > 0.3 * 256 = 76
            if cs.speech_activity_q8 > 76 {
                cs.lbrr_flags[frame_idx] = 1;

                // Copy current indices for LBRR
                cs.indices_lbrr[frame_idx] = cs.indices.clone();

                // Boost first subframe gain index by lbrr_gain_increases
                let mut lbrr_gains_indices = cs.indices.gains_indices;
                if cond_coding == CODE_INDEPENDENTLY {
                    // Absolute coding: boost the absolute gain index
                    lbrr_gains_indices[0] = ((lbrr_gains_indices[0] as i32
                        + cs.lbrr_gain_increases)
                        .min(N_LEVELS_QGAIN - 1)) as i8;
                } else {
                    // Delta coding: boost the delta to increase gain
                    lbrr_gains_indices[0] = ((lbrr_gains_indices[0] as i32
                        + cs.lbrr_gain_increases)
                        .min(MAX_DELTA_GAIN_QUANT - MIN_DELTA_GAIN_QUANT))
                        as i8;
                }
                cs.indices_lbrr[frame_idx].gains_indices = lbrr_gains_indices;

                // Dequantize LBRR gains
                let mut lbrr_gains_q16 = [0i32; MAX_NB_SUBFR];
                let mut lbrr_prev_gain_idx = cs.lbrr_prev_last_gain_index;
                gain_quant::silk_gains_dequant(
                    &mut lbrr_gains_q16,
                    &lbrr_gains_indices,
                    &mut lbrr_prev_gain_idx,
                    cond_coding == CODE_CONDITIONALLY,
                    nb_subfr,
                );

                // Clone NSQ state for LBRR (does not affect main encoder state)
                let mut lbrr_nsq_state = cs.nsq_state.clone();
                let mut lbrr_indices = cs.indices_lbrr[frame_idx].clone();

                // Read NSQ config values for LBRR
                let lbrr_signal_type = cs.indices.signal_type as i32;
                let lbrr_quant_offset_type = cs.indices.quant_offset_type as i32;
                let lbrr_nlsf_interp_coef_q2 = cs.indices.nlsf_interp_coef_q2 as i32;

                // Run NSQ with LBRR gains
                let mut lbrr_pulses = [0i8; MAX_FRAME_LENGTH];
                nsq::silk_nsq(
                    &mut lbrr_nsq_state,
                    &mut lbrr_indices,
                    &samples[..frame_length],
                    &mut lbrr_pulses,
                    &pred_coef_q12,
                    &ltp_coef_q14,
                    &ar_q13,
                    harm_shape_gain_q14,
                    tilt_q14,
                    lf_shp_q14,
                    &lbrr_gains_q16,
                    &pitch_lags,
                    lambda_q10,
                    ltp_scale_q14,
                    cs.frame_length,
                    cs.subfr_length,
                    cs.ltp_mem_length,
                    cs.lpc_order,
                    MAX_SHAPE_LPC_ORDER as i32,
                    cs.nb_subfr,
                    lbrr_signal_type,
                    lbrr_quant_offset_type,
                    lbrr_nlsf_interp_coef_q2,
                    &mut scratch.lbrr_nsq_s_ltp_q15,
                    &mut scratch.lbrr_nsq_s_ltp,
                    &mut scratch.lbrr_nsq_x_sc_q10,
                    &mut scratch.lbrr_nsq_xq_tmp,
                );

                // Store LBRR pulses
                cs.pulses_lbrr[frame_idx][..frame_length]
                    .copy_from_slice(&lbrr_pulses[..frame_length]);
            } else {
                cs.lbrr_flags[frame_idx] = 0;
            }
        } else {
            cs.lbrr_flags[frame_idx] = 0;
        }

        // ====== Main NSQ + Bitstream writing with iterative gain adjustment ======
        // Matches C reference encode_frame_FIX.c: iteratively adjust gain multiplier
        // until encoded bits fit within max_bits budget.

        let nsq_signal_type = cs.indices.signal_type as i32;
        let nsq_quant_offset_type = cs.indices.quant_offset_type as i32;
        let nsq_nlsf_interp_coef_q2 = cs.indices.nlsf_interp_coef_q2 as i32;
        let nsq_frame_length = cs.frame_length;
        let nsq_subfr_length = cs.subfr_length;
        let nsq_ltp_mem_length = cs.ltp_mem_length;
        let nsq_lpc_order = cs.lpc_order;
        let nsq_nb_subfr = cs.nb_subfr;

        let n_states_delayed_decision: i32 = match control.complexity {
            0 | 1 => 1,
            2..=5 => 2,
            6 | 7 => 3,
            _ => 4,
        };

        let max_bits = control.max_bits;
        let prev_sig_type = cs.ec_prev_signal_type;
        let prev_lag_idx = cs.ec_prev_lag_index;

        // Save state for iterative loop
        let saved_nsq = cs.nsq_state.clone();
        let saved_indices = cs.indices.clone();
        let saved_enc = enc.save_state();
        let saved_last_gain_idx = cs.last_gain_index;

        let max_iter = if max_bits > 0 { 6 } else { 1 };
        let mut gain_mult_q8: i32 = 256; // 1.0x

        for iter in 0..max_iter {
            // On retry, restore state and re-apply gains with multiplier
            if iter > 0 {
                cs.nsq_state = saved_nsq.clone();
                cs.indices = saved_indices.clone();
                enc.restore_state(&saved_enc);
                cs.last_gain_index = saved_last_gain_idx;

                // Apply gain multiplier to unquantized gains and re-quantize
                for k in 0..nb_subfr {
                    gains_q16[k] = ((gains_unq_q16[k] as i64 * gain_mult_q8 as i64) >> 8) as i32;
                }
                cs.last_gain_index = saved_last_gain_idx;
                gain_quant::silk_gains_quant(
                    &mut cs.indices.gains_indices,
                    &mut gains_q16,
                    &mut cs.last_gain_index,
                    cond_coding == CODE_CONDITIONALLY,
                    nb_subfr,
                );
            }

            // Run NSQ
            let mut pulses = [0i8; MAX_FRAME_LENGTH];
            if n_states_delayed_decision > 1 {
                nsq_del_dec::silk_nsq_del_dec(
                    &mut cs.nsq_state,
                    &mut cs.indices,
                    &samples[..frame_length],
                    &mut pulses,
                    &pred_coef_q12,
                    &ltp_coef_q14,
                    &ar_q13,
                    harm_shape_gain_q14,
                    tilt_q14,
                    lf_shp_q14,
                    &gains_q16,
                    &pitch_lags,
                    lambda_q10,
                    ltp_scale_q14,
                    nsq_frame_length,
                    nsq_subfr_length,
                    nsq_ltp_mem_length,
                    nsq_lpc_order,
                    MAX_SHAPE_LPC_ORDER as i32,
                    nsq_nb_subfr,
                    nsq_signal_type,
                    nsq_quant_offset_type,
                    nsq_nlsf_interp_coef_q2,
                    n_states_delayed_decision,
                    cs.warping_q16,
                    &mut scratch.nsq_s_ltp_q15,
                    &mut scratch.nsq_s_ltp,
                );
            } else {
                nsq::silk_nsq(
                    &mut cs.nsq_state,
                    &mut cs.indices,
                    &samples[..frame_length],
                    &mut pulses,
                    &pred_coef_q12,
                    &ltp_coef_q14,
                    &ar_q13,
                    harm_shape_gain_q14,
                    tilt_q14,
                    lf_shp_q14,
                    &gains_q16,
                    &pitch_lags,
                    lambda_q10,
                    ltp_scale_q14,
                    nsq_frame_length,
                    nsq_subfr_length,
                    nsq_ltp_mem_length,
                    nsq_lpc_order,
                    MAX_SHAPE_LPC_ORDER as i32,
                    nsq_nb_subfr,
                    nsq_signal_type,
                    nsq_quant_offset_type,
                    nsq_nlsf_interp_coef_q2,
                    &mut scratch.nsq_s_ltp_q15,
                    &mut scratch.nsq_s_ltp,
                    &mut scratch.nsq_x_sc_q10,
                    &mut scratch.nsq_xq_tmp,
                );
            }

            // Encode indices + pulses
            encode_indices::silk_encode_indices(
                &cs.indices,
                enc,
                0,
                false,
                cond_coding,
                cs.nb_subfr,
                cs.nlsf_cb_sel,
                cs.pitch_contour_sel,
                cs.pitch_lag_low_bits_sel,
                cs.fs_khz,
                prev_sig_type,
                prev_lag_idx,
            );
            encode_pulses::silk_encode_pulses(
                enc,
                &pulses,
                cs.indices.signal_type as i32,
                cs.indices.quant_offset_type as i32,
                cs.frame_length,
            );

            let n_bits = enc.tell() - saved_enc.tell();

            // Check if we fit within budget
            if max_bits <= 0 || n_bits <= max_bits {
                break; // Fits or no constraint
            }

            // Over budget: increase gain multiplier by 2x per iteration
            // (conservative to avoid NSQ context mismatch from large gain jumps)
            if iter < max_iter - 1 {
                gain_mult_q8 = (gain_mult_q8 * 3 / 2).min(1024); // C ref: 1.5x per iter, max 4x
            } else {
                // Last iteration: damage control — zero pulses
                enc.restore_state(&saved_enc);
                cs.indices = saved_indices.clone();
                let zero_pulses = [0i8; MAX_FRAME_LENGTH];
                encode_indices::silk_encode_indices(
                    &cs.indices,
                    enc,
                    0,
                    false,
                    cond_coding,
                    cs.nb_subfr,
                    cs.nlsf_cb_sel,
                    cs.pitch_contour_sel,
                    cs.pitch_lag_low_bits_sel,
                    cs.fs_khz,
                    prev_sig_type,
                    prev_lag_idx,
                );
                encode_pulses::silk_encode_pulses(
                    enc,
                    &zero_pulses,
                    cs.indices.signal_type as i32,
                    cs.indices.quant_offset_type as i32,
                    cs.frame_length,
                );
                break;
            }
        }

        // Update ec_prev state
        cs.ec_prev_signal_type = cs.indices.signal_type as i32;
        if voiced {
            cs.ec_prev_lag_index = cs.indices.lag_index;
        }

        // Update state for next frame
        cs.prev_nlsf_q15[..lpc_order].copy_from_slice(&nlsf_q15[..lpc_order]);
        cs.prev_signal_type = cs.indices.signal_type as i32;
        cs.first_frame_after_reset = false;
        cs.n_frames_encoded += 1;

        // If this is the last frame in the packet, save LBRR data for next packet
        if cs.n_frames_encoded >= n_frames_per_packet {
            // Move current packet's LBRR data to "previous" storage
            cs.prev_lbrr_flags = cs.lbrr_flags;
            cs.prev_indices_lbrr = cs.indices_lbrr.clone();
            cs.prev_pulses_lbrr = cs.pulses_lbrr;
            cs.prev_lbrr_any = cs.lbrr_flags[..n_frames_per_packet as usize]
                .iter()
                .any(|&f| f != 0);
            cs.prev_n_frames_per_packet = n_frames_per_packet;

            // Reset frame counter for next packet
            cs.n_frames_encoded = 0;
        }

        // Put scratch buffers back
        self.scratch = scratch;

        0 // Success
    }

    /// Write LBRR data from the previous packet into the bitstream.
    ///
    /// This should be called at the start of encoding a new packet, after writing
    /// the LBRR flag bit. Returns true if LBRR data was written.
    ///
    /// The LBRR data written here is from the PREVIOUS packet. The first packet
    /// encoded never has LBRR data (no previous packet to reference).
    pub fn write_lbrr_data(&self, enc: &mut EcCtx) -> bool {
        let cs = &self.state;

        if !cs.prev_lbrr_any {
            return false;
        }

        let n_frames = cs.prev_n_frames_per_packet as usize;

        // If more than one frame per packet, encode per-frame LBRR flags
        if n_frames > 1 {
            let lbrr_flags_icdf = if n_frames == 2 {
                &SILK_LBRR_FLAGS_2_ICDF[..]
            } else {
                &SILK_LBRR_FLAGS_3_ICDF[..]
            };

            // Compute combined LBRR symbol: binary encoding of per-frame flags
            // For the iCDF tables: symbol = combined_flags - 1 (since 0 means no LBRR,
            // which is never encoded here because prev_lbrr_any is true)
            let mut lbrr_symbol = 0usize;
            for i in 0..n_frames {
                lbrr_symbol |= (cs.prev_lbrr_flags[i] as usize) << i;
            }
            enc.enc_icdf(lbrr_symbol - 1, lbrr_flags_icdf, 8);
        }

        // Encode LBRR indices and pulses for each flagged frame
        for i in 0..n_frames {
            if cs.prev_lbrr_flags[i] != 0 {
                let lbrr_cond_coding = if i > 0 {
                    CODE_CONDITIONALLY
                } else {
                    CODE_INDEPENDENTLY
                };

                encode_indices::silk_encode_indices(
                    &cs.prev_indices_lbrr[i],
                    enc,
                    i as i32,
                    true, // encode_lbrr = true
                    lbrr_cond_coding,
                    cs.nb_subfr,
                    cs.nlsf_cb_sel,
                    cs.pitch_contour_sel,
                    cs.pitch_lag_low_bits_sel,
                    cs.fs_khz,
                    // For LBRR ec_prev state: use previous LBRR frame's signal type/lag
                    if i > 0 {
                        cs.prev_indices_lbrr[i - 1].signal_type as i32
                    } else {
                        0
                    },
                    if i > 0 {
                        cs.prev_indices_lbrr[i - 1].lag_index
                    } else {
                        0
                    },
                );

                encode_pulses::silk_encode_pulses(
                    enc,
                    &cs.prev_pulses_lbrr[i],
                    cs.prev_indices_lbrr[i].signal_type as i32,
                    cs.prev_indices_lbrr[i].quant_offset_type as i32,
                    cs.frame_length,
                );
            }
        }

        true
    }

    /// Encode a stereo SILK frame.
    ///
    /// Takes interleaved L/R samples. Converts to mid/side using adaptive
    /// prediction, then encodes mid channel (and optionally side channel).
    /// The stereo prediction info is written BEFORE per-channel frame data.
    ///
    /// Returns 0 on success, negative on error.
    pub fn encode_stereo(
        &mut self,
        control: &SilkEncControl,
        enc: &mut EcCtx,
        samples_left: &[i16],
        samples_right: &[i16],
    ) -> i32 {
        // Determine internal sampling rate
        let fs_khz = match control.max_internal_fs_hz {
            ..=8000 => 8,
            8001..=12000 => 12,
            _ => 16,
        };

        // Initialize / reconfigure if needed
        if !self.initialized || self.state.fs_khz != fs_khz {
            self.state.set_fs(fs_khz, control.payload_size_ms);
            self.state_side.set_fs(fs_khz, control.payload_size_ms);
            self.initialized = true;
        }
        self.n_channels_internal = control.n_channels_internal;

        let frame_length = self.state.frame_length as usize;

        // Ensure we have enough input
        if samples_left.len() < frame_length || samples_right.len() < frame_length {
            return -1;
        }

        // Prepare buffers with 2-sample prefix for stereo L/R to M/S conversion
        let buf_len = frame_length + 2;
        let mut x1 = vec![0i16; buf_len]; // left -> mid
        let mut x2 = vec![0i16; buf_len]; // right -> side

        // Copy current frame samples (offset by 2 for the prefix buffer)
        x1[2..buf_len].copy_from_slice(&samples_left[..frame_length]);
        x2[2..buf_len].copy_from_slice(&samples_right[..frame_length]);

        // Convert L/R to M/S with adaptive stereo prediction
        let mut ix = [[0i8; 3]; 2];
        let mut mid_only_flag = 0i8;
        let mut mid_side_rates_bps = [0i32; 2];

        crate::stereo_encode::silk_stereo_lr_to_ms(
            &mut self.stereo_state,
            &mut x1,
            &mut x2,
            &mut ix,
            &mut mid_only_flag,
            &mut mid_side_rates_bps,
            control.bitrate_bps,
            self.prev_speech_activity_q8,
            control.to_mono,
            fs_khz,
            frame_length,
        );

        // Write stereo prediction info BEFORE per-channel frame data
        crate::stereo_encode::silk_stereo_encode_pred(enc, &ix);

        // Write mid-only flag if we have 2 internal channels
        if self.n_channels_internal == 2 {
            crate::stereo_encode::silk_stereo_encode_mid_only(enc, mid_only_flag);
        }

        // Encode mid channel (x1 contains mid signal, skip the 2-sample prefix)
        // Build mono samples from the mid signal
        let mid_samples: Vec<i16> = x1[1..1 + frame_length].to_vec();

        // Create a mono control with mid bitrate
        let mid_control = SilkEncControl {
            bitrate_bps: mid_side_rates_bps[0],
            ..control.clone()
        };
        let result = self.encode(&mid_control, enc, &mid_samples);
        if result != 0 {
            return result;
        }

        // Update speech activity tracking
        self.prev_speech_activity_q8 = self.state.speech_activity_q8;

        // Encode side channel (if not mid-only)
        if mid_only_flag == 0 && self.n_channels_internal == 2 {
            // x2 contains side residual, skip the 2-sample prefix
            let side_samples: Vec<i16> = x2[1..1 + frame_length].to_vec();

            let side_control = SilkEncControl {
                bitrate_bps: mid_side_rates_bps[1],
                ..control.clone()
            };

            // Swap side channel state into the main slot to reuse encode()
            std::mem::swap(&mut self.state, &mut self.state_side);

            // Ensure side channel is configured
            if self.state.fs_khz != fs_khz {
                self.state.set_fs(fs_khz, control.payload_size_ms);
            }

            // Encode the side channel using the existing mono pipeline
            let result = self.encode(&side_control, enc, &side_samples);

            // Swap back
            std::mem::swap(&mut self.state, &mut self.state_side);

            if result != 0 {
                return result;
            }
        }

        // Track state
        self.prev_decode_only_middle = mid_only_flag != 0;

        0
    }

    /// Encode a complete SILK packet with proper header (VAD + LBRR flags).
    ///
    /// This is the high-level API that handles the packet header format:
    /// 1. Write VAD flag (1 bit) and LBRR flag (1 bit) as placeholders
    /// 2. If LBRR data exists from previous packet, write it
    /// 3. Encode the current frame
    /// 4. Patch initial bits to correct VAD + LBRR flag values
    pub fn encode_packet(
        &mut self,
        control: &SilkEncControl,
        enc: &mut EcCtx,
        samples: &[i16],
    ) -> i32 {
        // Write placeholder VAD flag + LBRR flag (will be patched later)
        enc.enc_bit_logp(false, 1); // VAD flag placeholder
        enc.enc_bit_logp(false, 1); // LBRR flag placeholder

        // Write LBRR data from previous packet (if any)
        let has_lbrr = self.write_lbrr_data(enc);

        // Encode the current frame
        let result = self.encode(control, enc, samples);
        if result != 0 {
            return result;
        }

        // Determine actual VAD flag from the encoded frame
        let actual_vad = self.state.prev_signal_type != TYPE_NO_VOICE_ACTIVITY;

        // Patch the initial VAD + LBRR bits
        // Bit layout in first byte: bit 7 = VAD flag, bit 6 = LBRR flag
        // enc_patch_initial_bits patches the MSBs of the first byte
        let flags_byte = (actual_vad as u32) | ((has_lbrr as u32) << 1);
        enc.enc_patch_initial_bits(flags_byte, 2);

        0
    }

    /// Encode a complete stereo SILK packet with proper header.
    ///
    /// Takes separate left and right channel sample buffers.
    /// Stereo prediction is written before per-channel frame data.
    ///
    /// For stereo, the packet layout is:
    /// 1. VAD flag (mid) + LBRR flag
    /// 2. (optional) VAD flag (side) + LBRR flag (side)
    /// 3. Stereo prediction indices
    /// 4. Mid-only flag
    /// 5. Mid channel frame data (indices + pulses)
    /// 6. Side channel frame data (indices + pulses) -- skipped if mid_only
    pub fn encode_stereo_packet(
        &mut self,
        control: &SilkEncControl,
        enc: &mut EcCtx,
        samples_left: &[i16],
        samples_right: &[i16],
    ) -> i32 {
        // Write placeholder VAD + LBRR flags for mid channel
        enc.enc_bit_logp(false, 1); // VAD flag placeholder (mid)
        enc.enc_bit_logp(false, 1); // LBRR flag placeholder (mid)

        // Encode stereo
        let result = self.encode_stereo(control, enc, samples_left, samples_right);
        if result != 0 {
            return result;
        }

        // Determine actual VAD flag from the mid channel
        let actual_vad = self.state.prev_signal_type != TYPE_NO_VOICE_ACTIVITY;

        // Patch the initial VAD + LBRR bits (no LBRR for stereo in this simplified version)
        let flags_byte = actual_vad as u32; // no LBRR
        enc.enc_patch_initial_bits(flags_byte, 2);

        0
    }
}

impl Default for SilkEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::{SilkDecControl, SilkDecoder};
    use opus_range_coder::EcCtx;

    #[test]
    fn test_silk_encoder_create() {
        let enc = SilkEncoder::new();
        // Encoder is not initialized until encode() is called
        assert_eq!(enc.state.fs_khz, 0);
        assert!(!enc.initialized);
    }

    #[test]
    fn test_silk_encode_silence() {
        let mut enc = SilkEncoder::new();
        let control = SilkEncControl {
            api_sample_rate: 16000,
            max_internal_fs_hz: 16000,
            payload_size_ms: 20,
            bitrate_bps: 20000,
            max_bits: 0,
            complexity: 0,
            use_in_band_fec: false,
            packet_loss_percentage: 0,
            n_channels_internal: 1,
            to_mono: false,
        };
        let samples = vec![0i16; 320]; // 20ms at 16kHz
        let mut range_enc = EcCtx::enc_init(1275);

        // Write VAD flag (0 = no voice activity) and LBRR flag (0)
        range_enc.enc_bit_logp(false, 1); // VAD flag
        range_enc.enc_bit_logp(false, 1); // LBRR flag

        let result = enc.encode(&control, &mut range_enc, &samples);
        assert_eq!(result, 0);

        range_enc.enc_done();
        let nbytes = ((range_enc.tell() + 7) >> 3) as usize;
        assert!(nbytes > 0, "Should produce some bytes");
    }

    #[test]
    fn test_silk_encode_decode_roundtrip() {
        let mut enc = SilkEncoder::new();
        let control = SilkEncControl {
            api_sample_rate: 16000,
            max_internal_fs_hz: 16000,
            payload_size_ms: 20,
            bitrate_bps: 20000,
            max_bits: 0,
            complexity: 0,
            use_in_band_fec: false,
            packet_loss_percentage: 0,
            n_channels_internal: 1,
            to_mono: false,
        };

        // Generate a simple 200Hz tone at 16kHz
        let n = 320;
        let mut samples = vec![0i16; n];
        for (i, sample) in samples.iter_mut().enumerate() {
            *sample =
                (5000.0 * (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 16000.0).sin()) as i16;
        }

        let mut range_enc = EcCtx::enc_init(1275);

        // Write packet header: VAD flag + LBRR flag
        range_enc.enc_bit_logp(true, 1); // VAD flag = 1 (voice activity)
        range_enc.enc_bit_logp(false, 1); // LBRR flag = 0

        let result = enc.encode(&control, &mut range_enc, &samples);
        assert_eq!(result, 0, "Encode should succeed");

        range_enc.enc_done();

        // Get encoded bytes
        let nbytes = (range_enc.tell() + 7) >> 3;
        let buf = range_enc.buf[..nbytes as usize].to_vec();
        assert!(buf.len() > 2, "Should produce a non-trivial bitstream");

        // Decode
        let mut dec = SilkDecoder::new();
        let mut dec_control = SilkDecControl {
            n_channels_api: 1,
            n_channels_internal: 1,
            api_sample_rate: 16000,
            internal_sample_rate: 16000,
            payload_size_ms: 20,
            prev_pitch_lag: 0,
        };

        let mut range_dec = EcCtx::dec_init(&buf);
        let mut decoded = vec![0i16; n];
        let mut n_samples_out = 0i32;

        let ret = dec.decode(
            &mut dec_control,
            0,    // not lost
            true, // new packet
            &mut range_dec,
            &mut decoded,
            &mut n_samples_out,
            &mut crate::decoder::NoPostFilter,
        );

        assert_eq!(ret, 0, "Decode should succeed on encoder output");
        assert_eq!(
            n_samples_out, n as i32,
            "Should decode correct number of samples"
        );

        // Verify decoded signal has energy (not all zeros)
        let energy: i64 = decoded.iter().map(|&x| x as i64 * x as i64).sum();
        assert!(energy > 0, "Decoded signal should have non-zero energy");
    }
}
