use opus_celt::CeltEncoder;
use opus_range_coder::EcCtx;
use opus_silk::{SilkEncoder, encoder::SilkEncControl};

use crate::error::OpusError;
use crate::types::*;

/// Generate the TOC (Table of Contents) byte for an Opus packet.
///
/// This encodes the mode, frame rate, bandwidth, and channel count into a
/// single byte that is the first byte of every Opus packet.
///
/// The `framerate` parameter is `fs / frame_size` (e.g., 48000/960 = 50 for 20ms).
fn gen_toc(mode: i32, framerate: i32, bandwidth: i32, channels: i32) -> u8 {
    // Compute period: number of times we need to double the framerate
    // to reach at least 400 (the 2.5ms base rate).
    let mut period = 0u8;
    let mut fr = framerate;
    while fr < 400 {
        fr <<= 1;
        period += 1;
    }

    let toc;
    if mode == Mode::SilkOnly as i32 {
        // SILK: bits 7-5 = bandwidth - NB, bits 4-3 = period - 2, bit 2 = stereo
        let bw = (bandwidth - Bandwidth::Narrowband as i32) as u8;
        toc = (bw << 5) | ((period - 2) << 3);
    } else if mode == Mode::CeltOnly as i32 {
        // CELT: bit 7 set, bits 6-5 = max(0, bandwidth - MB), bits 4-3 = period, bit 2 = stereo
        let tmp = (bandwidth - Bandwidth::Mediumband as i32) as u8;
        toc = 0x80 | (tmp << 5) | (period << 3);
    } else {
        // Hybrid: bits 7-5 = 011, bit 4 = bandwidth - SWB, bit 3 = period - 2, bit 2 = stereo
        let bw = (bandwidth - Bandwidth::Superwideband as i32) as u8;
        toc = 0x60 | (bw << 4) | ((period - 2) << 3);
    }

    toc | (if channels == 2 { 0x04 } else { 0x00 })
}

/// The main Opus encoder.
pub struct OpusEncoder {
    /// Number of input channels (1 or 2).
    channels: i32,
    /// Input sampling rate.
    fs: i32,
    /// Application type.
    application: i32,
    /// SILK encoder state (old fixed-point, used for stereo).
    silk_enc: SilkEncoder,
    /// Float SILK encoder state (new, used for mono SILK).
    silk_flp: opus_silk::encoder_flp::state::SilkEncoderStateFlp,
    /// CELT encoder state.
    celt_enc: CeltEncoder,
    /// Current encoding mode (MODE_SILK_ONLY / MODE_HYBRID / MODE_CELT_ONLY).
    mode: i32,
    /// Previous frame's mode.
    prev_mode: i32,
    /// Current bandwidth (OPUS_BANDWIDTH_*).
    bandwidth: i32,
    /// Number of channels in the stream (may differ from input channels).
    stream_channels: i32,
    /// Target bitrate in bits per second.
    bitrate_bps: i32,
    /// Whether to use variable bitrate.
    use_vbr: bool,
    /// Encoder complexity (0-10).
    complexity: i32,
    /// Signal type hint (OPUS_SIGNAL_VOICE / OPUS_SIGNAL_MUSIC / OPUS_AUTO).
    signal_type: i32,
    /// Force channel count (-1 = auto, 1 = mono, 2 = stereo).
    force_channels: i32,
    /// Maximum allowed bandwidth.
    max_bandwidth: i32,
    /// User-requested bitrate (before clamping).
    user_bitrate_bps: i32,
    /// Whether to use in-band FEC (LBRR) for SILK frames.
    use_inband_fec: bool,
    /// Expected packet loss percentage (0-100) for FEC rate control.
    packet_loss_perc: i32,
    /// Final range coder state for testing/verification.
    pub range_final: u32,
    /// HP filter biquad state (C: hp_mem[4], 2 per channel).
    hp_mem: [i32; 4],
    /// Variable HP cutoff smoothing state (C: variable_HP_smth2_Q15).
    variable_hp_smth2_q15: i32,
    /// Delay buffer for look-ahead (C: st->delay_buffer, size encoder_buffer*channels).
    /// Stores the last encoder_buffer samples for providing delay_compensation look-ahead.
    delay_buffer: Vec<i16>,
    /// Size of delay buffer in samples per channel (C: st->encoder_buffer = Fs/100 = 10ms).
    encoder_buffer: usize,
    /// Delay compensation in samples (C: st->delay_compensation = Fs/250 = 4ms).
    delay_compensation: usize,
    /// DNN encoder state (DRED). None when DNN is not loaded or DRED is disabled.
    #[cfg(feature = "dnn")]
    pub(crate) dnn: Option<Box<crate::dnn_types::DnnEncoderState>>,
}

impl OpusEncoder {
    /// Create a new Opus encoder.
    pub fn new(
        sample_rate: SampleRate,
        channels: Channels,
        application: Application,
    ) -> Result<Self, OpusError> {
        let fs = i32::from(sample_rate);
        let channels = i32::from(channels);
        let application = i32::from(application);

        let silk_enc = SilkEncoder::new();
        let silk_flp = opus_silk::encoder_flp::state::SilkEncoderStateFlp::new();
        let celt_enc =
            CeltEncoder::new(48000, channels as usize).map_err(|_| OpusError::InternalError)?;

        let default_bitrate = if application == Application::Voip as i32 {
            20000
        } else {
            64000
        };

        Ok(OpusEncoder {
            channels,
            fs,
            application,
            silk_flp,
            silk_enc,
            celt_enc,
            mode: Mode::CeltOnly as i32,
            prev_mode: 0,
            bandwidth: Bandwidth::Fullband as i32,
            stream_channels: channels,
            bitrate_bps: default_bitrate,
            use_vbr: true,
            complexity: 10,
            signal_type: i32::from(Signal::Auto),
            force_channels: i32::from(ForceChannels::Auto),
            max_bandwidth: Bandwidth::Fullband as i32,
            user_bitrate_bps: default_bitrate,
            use_inband_fec: false,
            packet_loss_perc: 0,
            range_final: 0,
            hp_mem: [0; 4],
            // C: silk_LSHIFT(silk_lin2log(VARIABLE_HP_MIN_CUTOFF_HZ), 8)
            // silk_lin2log(60) ≈ 756, << 8 = 193536
            variable_hp_smth2_q15: 755 << 8, // silk_lin2log(60) << 8
            // C: encoder_buffer = Fs/100 (10ms), delay_compensation = Fs/250 (4ms)
            delay_buffer: vec![0i16; (fs / 100) as usize * channels as usize],
            encoder_buffer: (fs / 100) as usize,
            delay_compensation: (fs / 250) as usize,
            #[cfg(feature = "dnn")]
            dnn: None,
        })
    }

    /// Set the target bitrate in bits per second.
    /// Apply HP biquad filter matching C reference hp_cutoff + silk_biquad_alt_stride1.
    /// Uses Direct Form II Transposed with split A coefficients for precision.
    /// State S[0], S[1] are in Q12.
    fn hp_filter_i16(&mut self, samples: &mut [i16], cutoff_hz: i32) {
        use opus_silk::{silk_rshift_round, silk_smlawb, silk_smulwb};

        // Design HP biquad: b = r*[1, -2, 1], a = [1, -2r(1-0.5*Fc^2), r^2]
        // C: Fc_Q19 = silk_DIV32_16(silk_SMULBB(SILK_FIX_CONST(1.5*pi/1000, 19), cutoff_Hz), Fs/1000)
        let fc_q19 = (2474i32 * cutoff_hz) / (self.fs / 1000); // 2474 = 1.5*pi/1000 in Q19
        let r_q28 = (1i32 << 28) - 471i32 * fc_q19; // 471 = 0.92 in Q9

        let b_q28 = [r_q28, r_q28.wrapping_neg() << 1, r_q28];
        let r_q22 = r_q28 >> 6;
        // a[0] = -r * (2 - Fc^2): C uses silk_SMULWW
        let a0_q28 = ((r_q22 as i64 * (((fc_q19 as i64 * fc_q19 as i64) >> 16) - ((2i64) << 22)))
            >> 16) as i32;
        let a1_q28 = ((r_q22 as i64 * r_q22 as i64) >> 16) as i32;

        // Split negated A coefficients (C: biquad_alt.c lines 56-59)
        let a0_neg = -a0_q28;
        let a0_l = a0_neg & 0x00003FFF;
        let a0_u = a0_neg >> 14;
        let a1_neg = -a1_q28;
        let a1_l = a1_neg & 0x00003FFF;
        let a1_u = a1_neg >> 14;

        // Apply biquad (C: silk_biquad_alt_stride1, biquad_alt.c lines 61-76)
        let s = &mut self.hp_mem;
        for sample in samples.iter_mut() {
            let inval = *sample as i32;
            // out32_Q14 = (S[0] + B[0]*inval >> 16) << 2
            let out32_q14 = silk_smlawb(s[0], b_q28[0], inval) << 2;

            // S[0] = S[1] + round(out*A0_L >> 14) + out*A0_U>>16 + B[1]*inval>>16
            s[0] = s[1] + silk_rshift_round(silk_smulwb(out32_q14, a0_l), 14);
            s[0] = silk_smlawb(s[0], out32_q14, a0_u);
            s[0] = silk_smlawb(s[0], b_q28[1], inval);

            // S[1] = round(out*A1_L >> 14) + out*A1_U>>16 + B[2]*inval>>16
            s[1] = silk_rshift_round(silk_smulwb(out32_q14, a1_l), 14);
            s[1] = silk_smlawb(s[1], out32_q14, a1_u);
            s[1] = silk_smlawb(s[1], b_q28[2], inval);

            // Scale back to Q0 and saturate
            let out_val = (out32_q14 + (1 << 14) - 1) >> 14;
            *sample = out_val.clamp(-32768, 32767) as i16;
        }
    }

    pub fn set_bitrate(&mut self, bitrate: Bitrate) {
        match bitrate {
            Bitrate::Auto | Bitrate::Max => {
                self.user_bitrate_bps = i32::from(bitrate);
                self.bitrate_bps = if self.application == Application::Voip as i32 {
                    20000
                } else {
                    64000
                };
            }
            Bitrate::BitsPerSecond(bps) => {
                self.user_bitrate_bps = bps;
                self.bitrate_bps = bps.clamp(500, 512000);
            }
        }
    }

    /// Set the encoder complexity (0-10).
    pub fn set_complexity(&mut self, complexity: i32) {
        self.complexity = complexity.clamp(0, 10);
    }

    /// Set the signal type hint.
    pub fn set_signal(&mut self, signal: Signal) {
        self.signal_type = i32::from(signal);
    }

    /// Set the maximum bandwidth.
    pub fn set_bandwidth(&mut self, bandwidth: Bandwidth) {
        self.max_bandwidth = i32::from(bandwidth);
    }

    /// Enable or disable in-band FEC (LBRR) for SILK frames.
    pub fn set_inband_fec(&mut self, enabled: bool) {
        self.use_inband_fec = enabled;
    }

    /// Set the expected packet loss percentage (0-100) for FEC rate control.
    pub fn set_packet_loss_perc(&mut self, perc: i32) {
        self.packet_loss_perc = perc.clamp(0, 100);
    }

    /// Force encoding to a specific channel count.
    pub fn set_force_channels(&mut self, channels: ForceChannels) {
        self.force_channels = i32::from(channels);
    }

    /// Get the number of channels.
    pub fn channels(&self) -> i32 {
        self.channels
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> i32 {
        self.fs
    }

    /// Load DNN models from a binary weight blob and enable DRED support.
    ///
    /// The blob must contain at minimum the RDOVAE encoder and PitchDNN
    /// weights. This is equivalent to the C `OPUS_SET_DNN_BLOB` CTL.
    ///
    /// After loading, DRED is not yet active — call [`set_dred_duration`]
    /// with a non-zero value to start encoding DRED redundancy.
    ///
    /// Requires the `dnn` feature.
    #[cfg(feature = "dnn")]
    pub fn load_dnn(&mut self, data: &[u8]) -> Result<(), OpusError> {
        let state = crate::dnn_types::DnnEncoderState::from_blob(data)?;
        self.dnn = Some(Box::new(state));
        Ok(())
    }

    /// Set the DRED duration in frames (0 = disabled).
    ///
    /// Controls how many frames of deep redundancy are encoded per packet.
    /// The DNN models must be loaded first via [`load_dnn`].
    ///
    /// Requires the `dnn` feature.
    #[cfg(feature = "dnn")]
    pub fn set_dred_duration(&mut self, frames: i32) {
        if let Some(ref mut dnn) = self.dnn {
            dnn.dred_duration = frames.max(0);
        }
    }

    /// Get the current DRED duration in frames.
    ///
    /// Returns 0 if DNN is not loaded or DRED is disabled.
    ///
    /// Requires the `dnn` feature.
    #[cfg(feature = "dnn")]
    pub fn dred_duration(&self) -> i32 {
        self.dnn.as_ref().map_or(0, |dnn| dnn.dred_duration)
    }

    /// Returns whether DNN models are loaded and ready.
    ///
    /// Requires the `dnn` feature.
    #[cfg(feature = "dnn")]
    pub fn dnn_loaded(&self) -> bool {
        self.dnn.as_ref().is_some_and(|dnn| dnn.loaded)
    }

    /// Decide the encoding mode based on bitrate, application, bandwidth, and frame size.
    fn decide_mode(&self, frame_size: i32) -> (i32, i32) {
        let frame_duration_ms = frame_size * 1000 / self.fs;

        // SILK requires at least 10ms frames
        let silk_ok = frame_duration_ms >= 10;

        // Determine mode from application, bitrate, and signal type
        let mode;
        let bandwidth;

        if self.application == Application::RestrictedLowDelay as i32 || !silk_ok {
            // Low-delay or sub-10ms frames: CELT only
            mode = Mode::CeltOnly as i32;
            bandwidth = self.decide_bandwidth();
        } else if self.application == Application::Voip as i32
            || self.signal_type == Signal::Voice as i32
        {
            if self.bitrate_bps < 20000 {
                mode = Mode::SilkOnly as i32;
                bandwidth = self.decide_bandwidth();
            } else if self.bitrate_bps < 32000 {
                // Potential hybrid zone
                let bw = self.decide_bandwidth();
                if bw >= Bandwidth::Superwideband as i32 {
                    mode = Mode::Hybrid as i32;
                    bandwidth = bw;
                } else {
                    mode = Mode::SilkOnly as i32;
                    bandwidth = bw;
                }
            } else {
                mode = Mode::CeltOnly as i32;
                bandwidth = self.decide_bandwidth();
            }
        } else {
            // AUDIO application
            if self.bitrate_bps < 12000 {
                mode = Mode::SilkOnly as i32;
                bandwidth = self.decide_bandwidth();
            } else if self.bitrate_bps < 24000 {
                let bw = self.decide_bandwidth();
                if bw >= Bandwidth::Superwideband as i32 {
                    mode = Mode::Hybrid as i32;
                    bandwidth = bw;
                } else {
                    mode = Mode::SilkOnly as i32;
                    bandwidth = bw;
                }
            } else {
                mode = Mode::CeltOnly as i32;
                bandwidth = self.decide_bandwidth();
            }
        }

        // Clamp bandwidth to max
        let bandwidth = bandwidth.min(self.max_bandwidth);

        (mode, bandwidth)
    }

    /// Decide bandwidth based on bitrate.
    fn decide_bandwidth(&self) -> i32 {
        // Thresholds approximate the C reference's bandwidth decision.
        // Per-channel bitrate determines the highest bandwidth that can be
        // encoded with acceptable quality.
        let per_ch = self.bitrate_bps / self.channels.max(1);
        if per_ch < 10000 {
            Bandwidth::Narrowband as i32
        } else if per_ch < 14000 {
            Bandwidth::Mediumband as i32
        } else if per_ch < 28000 {
            Bandwidth::Wideband as i32
        } else if per_ch < 40000 {
            Bandwidth::Superwideband as i32
        } else {
            Bandwidth::Fullband as i32
        }
    }

    /// Encode an Opus frame from floating-point PCM.
    ///
    /// `pcm` contains `frame_size * channels` interleaved samples.
    /// `frame_size` must correspond to 2.5, 5, 10, 20, 40, or 60 ms at the
    /// configured sample rate.
    /// Returns the number of bytes written into `data`.
    pub fn encode_float(
        &mut self,
        pcm: &[f32],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, OpusError> {
        if frame_size <= 0 || max_data_bytes <= 0 {
            return Err(OpusError::BadArg);
        }

        let max_data_bytes = max_data_bytes.min(data.len() as i32);
        if max_data_bytes < 1 {
            return Err(OpusError::BufferTooSmall);
        }

        // Validate frame_size: must be 2.5, 5, 10, 20, 40, or 60ms
        let valid_frame_sizes = [
            self.fs / 400,       // 2.5ms
            self.fs / 200,       // 5ms
            self.fs / 100,       // 10ms
            self.fs / 50,        // 20ms
            self.fs / 25,        // 40ms
            self.fs * 60 / 1000, // 60ms
        ];
        if !valid_frame_sizes.contains(&frame_size) {
            return Err(OpusError::BadArg);
        }

        let total_samples = (frame_size * self.channels) as usize;
        if pcm.len() < total_samples {
            return Err(OpusError::BadArg);
        }

        // Determine stream channels
        let stream_channels = if self.force_channels > 0 {
            self.force_channels.min(self.channels)
        } else {
            self.channels
        };
        self.stream_channels = stream_channels;

        // Mode and bandwidth decision
        let (mode, bandwidth) = self.decide_mode(frame_size);
        self.mode = mode;
        self.bandwidth = bandwidth;

        let frame_rate = self.fs / frame_size;

        // Generate TOC byte
        let toc = gen_toc(mode, frame_rate, bandwidth, stream_channels);
        data[0] = toc;

        // Initialize range encoder for the payload (after the TOC byte)
        let payload_max = (max_data_bytes - 1) as u32;
        if payload_max < 2 {
            // Too small for any payload, produce a DTX-like packet
            self.range_final = 0;
            return Ok(1);
        }

        let mut enc = EcCtx::enc_init(payload_max);

        let mut silk_bytes_used = 0i32;

        // === DRED latent computation (before SILK encoding) ===
        #[cfg(feature = "dnn")]
        if let Some(ref mut dnn) = self.dnn
            && dnn.dred_duration > 0
        {
            // DRED processes mono f32 PCM at the input sample rate.
            // Downmix to mono if stereo.
            let mono_pcm: Vec<f32> = if self.channels == 2 {
                (0..frame_size as usize)
                    .map(|i| 0.5 * (pcm[2 * i] + pcm[2 * i + 1]))
                    .collect()
            } else {
                pcm[..frame_size as usize].to_vec()
            };

            opus_dnn::dred::encoder::dred_compute_latents(
                &mut dnn.dred_enc,
                &mono_pcm,
                frame_size as usize,
                self.delay_compensation,
            );

            // Update activity_mem: shift old entries left, insert new VAD decision.
            // Resolution: 2.5ms frames (Fs/400). C: frame_size_400Hz = frame_size * 400 / Fs.
            let frame_size_400hz = (frame_size * 400 / self.fs) as usize;
            let mem_len = dnn.activity_mem.len();
            if frame_size_400hz < mem_len {
                dnn.activity_mem.copy_within(frame_size_400hz.., 0);
                let activity = u8::from(self.silk_flp.speech_activity_q8 > 0);
                for v in &mut dnn.activity_mem[mem_len - frame_size_400hz..] {
                    *v = activity;
                }
            }
        }

        // === SILK encoding ===
        if mode == Mode::SilkOnly as i32 || mode == Mode::Hybrid as i32 {
            // Determine SILK internal rate
            let silk_internal_rate = if mode == Mode::SilkOnly as i32 {
                match bandwidth {
                    x if x == Bandwidth::Narrowband as i32 => 8000,
                    x if x == Bandwidth::Mediumband as i32 => 12000,
                    _ => 16000,
                }
            } else {
                // Hybrid mode: SILK always runs at 16kHz
                16000
            };

            let frame_duration_ms = frame_size * 1000 / self.fs;

            // Convert f32 PCM to i16 at the SILK internal rate
            let silk_samples = (silk_internal_rate * frame_duration_ms / 1000) as usize;
            let input_samples = frame_size as usize;

            // C reference: allocate pcm_buf with delay_compensation prefix + frame
            // pcm_buf[0..total_buffer] = last total_buffer samples from delay_buffer
            // pcm_buf[total_buffer..] = HP-filtered current frame
            let total_buffer = self.delay_compensation;
            let nch = stream_channels as usize;
            let mut pcm_buf = vec![0i16; (total_buffer + silk_samples) * nch];

            // Copy look-ahead from delay_buffer (C: opus_encoder.c:1966-1967)
            let db_offset = self.encoder_buffer.saturating_sub(total_buffer);
            let copy_len = total_buffer * nch;
            if copy_len > 0 && db_offset * nch + copy_len <= self.delay_buffer.len() {
                pcm_buf[..copy_len].copy_from_slice(
                    &self.delay_buffer[db_offset * nch..db_offset * nch + copy_len],
                );
            }

            // Convert and place current frame at pcm_buf[total_buffer..]
            {
                let dst = &mut pcm_buf[total_buffer * nch..];
                if self.fs == silk_internal_rate {
                    let n = silk_samples.min(input_samples);
                    for ch in 0..nch {
                        for i in 0..n {
                            let s = (pcm[i * self.channels as usize + ch] * 32768.0).round() as i32;
                            dst[i * nch + ch] = s.clamp(-32768, 32767) as i16;
                        }
                    }
                } else {
                    let ratio = silk_internal_rate as f64 / self.fs as f64;
                    for ch in 0..nch {
                        for i in 0..silk_samples {
                            let src_pos = i as f64 / ratio;
                            let src_idx = src_pos as usize;
                            let frac = src_pos - src_idx as f64;
                            let idx0 = src_idx.min(input_samples - 1);
                            let idx1 = (src_idx + 1).min(input_samples - 1);
                            let s0 = pcm[idx0 * self.channels as usize + ch] as f64;
                            let s1 = pcm[idx1 * self.channels as usize + ch] as f64;
                            let val = ((s0 * (1.0 - frac) + s1 * frac) * 32768.0).round() as i32;
                            dst[i * nch + ch] = val.clamp(-32768, 32767) as i16;
                        }
                    }
                }
            }

            // Update delay_buffer (C: opus_encoder.c:2304-2312)
            // Store the most recent encoder_buffer samples for next frame's look-ahead.
            {
                let db_samples = self.encoder_buffer;
                let pcm_buf_samples = total_buffer + silk_samples;
                let db_len = db_samples * nch;
                let buf_len = pcm_buf_samples * nch;

                if db_samples > pcm_buf_samples {
                    // Shift old data left, append pcm_buf
                    let keep = (db_samples - pcm_buf_samples) * nch;
                    self.delay_buffer.copy_within(buf_len..buf_len + keep, 0);
                    self.delay_buffer[keep..keep + buf_len].copy_from_slice(&pcm_buf[..buf_len]);
                } else {
                    // pcm_buf has enough — take last db_samples
                    let offset = buf_len - db_len;
                    self.delay_buffer[..db_len].copy_from_slice(&pcm_buf[offset..offset + db_len]);
                }
            }

            // pcm_i16 is the HP-filtered frame (at the total_buffer offset)
            let mut pcm_i16 =
                pcm_buf[total_buffer * nch..(total_buffer + silk_samples) * nch].to_vec();

            let silk_bitrate = if mode == Mode::Hybrid as i32 {
                self.bitrate_bps / 2
            } else {
                self.bitrate_bps
            };

            // Maximum bits for the SILK frame. Allow 2x the per-frame bitrate
            // target for VBR headroom (individual frames may exceed the average),
            // capped by the packet budget. The C reference similarly allows overrun
            // on individual frames and relies on VBR averaging.
            // Use packet budget as max_bits (C reference style). Bitrate control
            // will come from proper gain adjustment via process_gains.
            let silk_max_bits = (max_data_bytes - 1) * 8;

            let control = SilkEncControl {
                api_sample_rate: silk_internal_rate,
                max_internal_fs_hz: silk_internal_rate,
                payload_size_ms: frame_duration_ms,
                bitrate_bps: silk_bitrate,
                max_bits: silk_max_bits,
                complexity: self.complexity.min(10),
                use_in_band_fec: self.use_inband_fec,
                packet_loss_percentage: self.packet_loss_perc,
                n_channels_internal: stream_channels,
                to_mono: false,
            };

            let result = if stream_channels == 2 {
                // Stereo SILK: split interleaved pcm_i16 into left and right
                let mut left = vec![0i16; silk_samples];
                let mut right = vec![0i16; silk_samples];
                for i in 0..silk_samples {
                    left[i] = pcm_i16[i * 2];
                    right[i] = pcm_i16[i * 2 + 1];
                }

                // Write placeholder VAD + LBRR flags for both channels (4 bits)
                // These will be patched after encode_stereo determines mid_only state
                enc.enc_bit_logp(false, 1); // VAD (mid) placeholder
                enc.enc_bit_logp(false, 1); // LBRR (mid) placeholder
                enc.enc_bit_logp(false, 1); // VAD (side) placeholder
                enc.enc_bit_logp(false, 1); // LBRR (side) placeholder

                let ret = self
                    .silk_enc
                    .encode_stereo(&control, &mut enc, &left, &right);

                // Patch VAD + LBRR flags: mid VAD=1, mid LBRR=0
                // Side VAD = 0 when mid_only (so decoder reads mid_only flag), 1 otherwise
                let mid_vad = 1u32;
                let mid_lbrr = 0u32;
                let side_active = if self.silk_enc.prev_decode_only_middle {
                    0u32
                } else {
                    1u32
                };
                let side_lbrr = 0u32;
                // Pack: bit 3=mid_vad, bit 2=mid_lbrr, bit 1=side_vad, bit 0=side_lbrr
                let flags = (mid_vad << 3) | (mid_lbrr << 2) | (side_active << 1) | side_lbrr;
                enc.enc_patch_initial_bits(flags, 4);

                ret
            } else {
                // Mono SILK — use float frame encoder (matching C float path)

                // Initialize/reconfigure on first use or rate change
                let fs_khz = silk_internal_rate / 1000;
                if self.silk_flp.fs_khz != fs_khz {
                    self.silk_flp.set_fs(fs_khz, frame_duration_ms);
                }

                // Run VAD
                opus_silk::vad::silk_vad_get_sa_q8(
                    &mut self.silk_flp.vad_state,
                    &mut self.silk_flp.speech_activity_q8,
                    &mut self.silk_flp.snr_db_q7,
                    &mut self.silk_flp.input_quality_bands_q15,
                    &mut self.silk_flp.input_tilt_q15,
                    &pcm_i16[..silk_samples],
                    self.silk_flp.frame_length,
                );

                // Compute SNR from target bitrate
                let snr_db_q7 = opus_silk::encoder::silk_control_snr(
                    fs_khz,
                    self.silk_flp.nb_subfr,
                    silk_bitrate,
                );

                // Apply HP filter in-place (VOIP mode) — pcm_i16 is already a clone
                if self.application == Application::Voip as i32 {
                    let cutoff_hz = opus_silk::silk_log2lin(self.variable_hp_smth2_q15 >> 8);
                    self.hp_filter_i16(&mut pcm_i16[..silk_samples], cutoff_hz);
                }

                // Write VAD flag and LBRR flag
                enc.enc_bit_logp(true, 1);
                enc.enc_bit_logp(false, 1);

                let nlsf_cb = opus_silk::get_nlsf_cb(self.silk_flp.nlsf_cb_sel);
                let sf = &mut self.silk_flp;

                // Reset frame-within-packet counter (single-frame packets)
                sf.n_frames_encoded = 0;

                // Call float frame encoder
                let bytes = opus_silk::encoder_flp::encode_frame::silk_encode_frame_flp(
                    &mut sf.x_buf,
                    &mut sf.nsq_state,
                    &mut sf.indices,
                    &mut sf.prev_nlsf_q15,
                    &mut sf.prev_signal_type,
                    &mut sf.prev_lag,
                    &mut sf.first_frame_after_reset,
                    &mut sf.last_gain_index,
                    &mut sf.prev_harm_smth,
                    &mut sf.prev_tilt_smth,
                    &mut sf.prev_ltp_corr,
                    &mut sf.sum_log_gain_q7,
                    &mut sf.frame_counter,
                    sf.speech_activity_q8,
                    &sf.input_quality_bands_q15,
                    sf.input_tilt_q15,
                    snr_db_q7,
                    &pcm_i16[..silk_samples],
                    fs_khz,
                    sf.nb_subfr,
                    sf.subfr_length,
                    sf.frame_length,
                    sf.ltp_mem_length,
                    sf.predict_lpc_order,
                    sf.shaping_lpc_order,
                    sf.shape_win_length,
                    sf.la_pitch,
                    sf.pitch_lpc_win_length,
                    sf.pitch_estimation_lpc_order,
                    sf.warping_q16,
                    self.complexity.min(10),
                    nlsf_cb,
                    (max_data_bytes - 1) * 8,
                    sf.packet_loss_perc,
                    sf.n_frames_per_packet,
                    sf.n_frames_encoded as usize,
                    &mut sf.lbrr,
                    &mut enc,
                    &mut sf.scratch_s_ltp_q15,
                    &mut sf.scratch_s_ltp,
                    &mut sf.scratch_x_sc_q10,
                    &mut sf.scratch_xq_tmp,
                    &mut sf.lbrr_scratch_s_ltp_q15,
                    &mut sf.lbrr_scratch_s_ltp,
                    &mut sf.lbrr_scratch_x_sc_q10,
                    &mut sf.lbrr_scratch_xq_tmp,
                );

                sf.n_frames_encoded += 1;
                bytes as i32
            };
            if result < 0 {
                return Err(OpusError::InternalError);
            }

            silk_bytes_used = (enc.tell() + 7) >> 3;
        }

        // === CELT encoding ===
        if mode == Mode::CeltOnly as i32 || mode == Mode::Hybrid as i32 {
            let start_band: usize;
            let end_band: usize;

            if mode == Mode::Hybrid as i32 {
                // In hybrid mode, CELT encodes only the high bands
                start_band = 17;
                end_band = if bandwidth == Bandwidth::Superwideband as i32 {
                    19
                } else {
                    21 // Fullband
                };
            } else {
                // CELT-only
                start_band = 0;
                end_band = if bandwidth == Bandwidth::Narrowband as i32 {
                    13
                } else if bandwidth <= Bandwidth::Wideband as i32 {
                    17
                } else if bandwidth == Bandwidth::Superwideband as i32 {
                    19
                } else {
                    21
                };
            }

            self.celt_enc.start = start_band;
            self.celt_enc.end = end_band;
            self.celt_enc.stream_channels = stream_channels as usize;
            self.celt_enc.signalling = false;
            self.celt_enc.complexity = self.complexity;
            self.celt_enc.vbr = self.use_vbr;

            // Compute CELT bitrate: total minus what SILK used
            let celt_bytes = if mode == Mode::Hybrid as i32 {
                ((max_data_bytes - 1) - silk_bytes_used).max(2) as usize
            } else {
                (max_data_bytes - 1).max(2) as usize
            };

            self.celt_enc.bitrate = self.bitrate_bps;

            // CELT operates at 48kHz internally. Compute the CELT frame size.
            let celt_frame_size = (frame_size as usize * 48000) / self.fs as usize;

            // Prepare PCM for CELT (it expects interleaved f32 at 48kHz)
            let celt_pcm: Vec<f32>;
            if self.fs == 48000 {
                celt_pcm = pcm[..total_samples].to_vec();
            } else {
                // Upsample to 48kHz for CELT
                let ratio = 48000.0 / self.fs as f64;
                let out_samples = celt_frame_size * self.channels as usize;
                let mut upsampled = vec![0.0f32; out_samples];
                for ch in 0..self.channels as usize {
                    for i in 0..celt_frame_size {
                        let src_pos = i as f64 / ratio;
                        let src_idx = src_pos as usize;
                        let frac = src_pos - src_idx as f64;
                        let input_len = frame_size as usize;
                        let idx0 = src_idx.min(input_len - 1);
                        let idx1 = (src_idx + 1).min(input_len - 1);
                        let s0 = pcm[idx0 * self.channels as usize + ch] as f64;
                        let s1 = pcm[idx1 * self.channels as usize + ch] as f64;
                        upsampled[i * self.channels as usize + ch] =
                            (s0 * (1.0 - frac) + s1 * frac) as f32;
                    }
                }
                celt_pcm = upsampled;
            }

            // Allocate a temporary output buffer for CELT compressed data
            let mut celt_compressed = vec![0u8; celt_bytes];

            let celt_result = self.celt_enc.encode_with_ec(
                &celt_pcm,
                celt_frame_size,
                &mut celt_compressed,
                celt_bytes,
                Some(&mut enc),
            );

            match celt_result {
                Ok(_nbytes) => {
                    self.range_final = self.celt_enc.rng;
                }
                Err(_) => {
                    // CELT encoding failed; produce a minimal valid packet
                    self.range_final = 0;
                }
            }
        } else {
            // SILK-only: finalize the range coder
            self.range_final = enc.rng;
        }

        // Finalize the range coder
        enc.enc_done();
        let nbytes = ((enc.tell() + 7) >> 3) as usize;

        if nbytes == 0 {
            // DTX: just the TOC byte
            self.range_final = 0;
            return Ok(1);
        }

        // Copy encoded payload after the TOC byte (enc.buf is zero-padded beyond used bits)
        let out_bytes = nbytes;
        data[1..1 + out_bytes].copy_from_slice(&enc.buf[..out_bytes]);

        // Strip trailing zeros for SILK-only mode (matching C reference behavior)
        let mut ret = (out_bytes + 1) as i32; // +1 for TOC
        if mode == Mode::SilkOnly as i32 {
            while ret > 2 && data[ret as usize - 1] == 0 {
                ret -= 1;
            }
        }

        // === DRED extension encoding ===
        #[cfg(feature = "dnn")]
        if let Some(ref mut dnn) = self.dnn
            && dnn.dred_duration > 0
            && dnn.dred_enc.latents_buffer_fill > 0
        {
            // Compute quantization parameters from bitrate (matching C: opus_encoder.c:707-727)
            let bitrate_offset = if self.use_inband_fec { 20000 } else { 12000 };
            let effective_br = (self.bitrate_bps - bitrate_offset).max(1);
            let q0 = 15.min(4.max(51 - 3 * opus_celt::mathops::ec_ilog(effective_br as u32)));
            let dq = if effective_br > 36000 { 3 } else { 5 };
            let qmax = 15;

            let max_chunks = dnn.dred_duration as usize;
            let dred_space = (max_data_bytes as usize).saturating_sub(ret as usize);
            let max_dred_bytes = dred_space.min(opus_dnn::dred::DRED_MAX_DATA_SIZE);

            if max_dred_bytes >= opus_dnn::dred::DRED_MIN_BYTES {
                let mut dred_buf = vec![0u8; max_dred_bytes];
                let dred_bytes = opus_dnn::dred::encoder::dred_encode_silk_frame(
                    &mut dnn.dred_enc,
                    &mut dred_buf,
                    max_chunks,
                    max_dred_bytes,
                    q0,
                    dq,
                    qmax,
                    &dnn.activity_mem,
                    &dnn.dred_stats,
                );

                if dred_bytes > 0 {
                    let ext = crate::extensions::OpusExtensionData {
                        id: opus_dnn::dred::DRED_EXTENSION_ID as i32,
                        frame: 0,
                        data: dred_buf[..dred_bytes].to_vec(),
                    };
                    let ext_space = (max_data_bytes as usize).saturating_sub(ret as usize);
                    if ext_space > 0
                        && let Ok(ext_bytes) = crate::extensions::opus_packet_extensions_generate(
                            &mut data[ret as usize..],
                            &[ext],
                            1,
                        )
                    {
                        ret += ext_bytes;
                    }
                }
            }
        }

        self.prev_mode = mode;

        Ok(ret)
    }

    /// Encode an Opus frame from 16-bit integer PCM.
    ///
    /// `pcm` contains `frame_size * channels` interleaved samples.
    /// Returns the number of bytes written into `data`.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, OpusError> {
        let total_samples = (frame_size * self.channels) as usize;
        if pcm.len() < total_samples {
            return Err(OpusError::BadArg);
        }

        // Convert i16 to f32
        let float_pcm: Vec<f32> = pcm[..total_samples]
            .iter()
            .map(|&s| s as f32 * (1.0 / 32768.0))
            .collect();

        self.encode_float(&float_pcm, frame_size, data, max_data_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::OpusDecoder;
    use crate::packet::*;

    #[test]
    fn test_encoder_create() {
        // Valid configurations
        let enc = OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio);
        assert!(enc.is_ok());
        let enc = enc.unwrap();
        assert_eq!(enc.sample_rate(), 48000);
        assert_eq!(enc.channels(), 2);

        let enc = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip);
        assert!(enc.is_ok());
        let enc = enc.unwrap();
        assert_eq!(enc.channels(), 1);

        let enc = OpusEncoder::new(
            SampleRate::Hz16000,
            Channels::Mono,
            Application::RestrictedLowDelay,
        );
        assert!(enc.is_ok());

        let enc = OpusEncoder::new(SampleRate::Hz8000, Channels::Mono, Application::Voip);
        assert!(enc.is_ok());
    }

    #[test]
    fn test_encode_silence() {
        let mut enc =
            OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
        enc.set_bitrate(Bitrate::BitsPerSecond(64000));

        let frame_size = 960; // 20ms at 48kHz
        let pcm = vec![0.0f32; frame_size];
        let mut data = vec![0u8; 1275];

        let result = enc.encode_float(&pcm, frame_size as i32, &mut data, 1275);
        assert!(
            result.is_ok(),
            "Encoding silence should succeed: {:?}",
            result.err()
        );

        let nbytes = result.unwrap();
        assert!(nbytes >= 1, "Should produce at least the TOC byte");
        assert!(nbytes <= 1275, "Should not exceed max packet size");

        // Verify the TOC byte is valid by parsing it
        let mode = opus_packet_get_mode(&data);
        assert!(
            mode == Mode::SilkOnly || mode == Mode::Hybrid || mode == Mode::CeltOnly,
            "TOC should encode a valid mode"
        );
    }

    #[test]
    fn test_encode_silence_i16() {
        let mut enc =
            OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
        let frame_size = 960;
        let pcm = vec![0i16; frame_size];
        let mut data = vec![0u8; 1275];

        let result = enc.encode(&pcm, frame_size as i32, &mut data, 1275);
        assert!(
            result.is_ok(),
            "i16 encoding should succeed: {:?}",
            result.err()
        );
        assert!(result.unwrap() >= 1);
    }

    #[test]
    fn test_gen_toc_roundtrip() {
        // Test that gen_toc produces bytes that packet.rs can decode correctly.

        // CELT 20ms fullband mono
        let toc = gen_toc(Mode::CeltOnly as i32, 50, Bandwidth::Fullband as i32, 1);
        assert_eq!(opus_packet_get_mode(&[toc]), Mode::CeltOnly);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), Bandwidth::Fullband);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 960);

        // CELT 20ms fullband stereo
        let toc = gen_toc(Mode::CeltOnly as i32, 50, Bandwidth::Fullband as i32, 2);
        assert_eq!(opus_packet_get_mode(&[toc]), Mode::CeltOnly);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), Bandwidth::Fullband);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 2);

        // SILK 20ms narrowband mono
        let toc = gen_toc(Mode::SilkOnly as i32, 50, Bandwidth::Narrowband as i32, 1);
        assert_eq!(opus_packet_get_mode(&[toc]), Mode::SilkOnly);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), Bandwidth::Narrowband);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 960);

        // SILK 10ms wideband stereo
        let toc = gen_toc(Mode::SilkOnly as i32, 100, Bandwidth::Wideband as i32, 2);
        assert_eq!(opus_packet_get_mode(&[toc]), Mode::SilkOnly);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), Bandwidth::Wideband);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 2);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 480);

        // Hybrid 20ms superwideband mono
        let toc = gen_toc(Mode::Hybrid as i32, 50, Bandwidth::Superwideband as i32, 1);
        assert_eq!(opus_packet_get_mode(&[toc]), Mode::Hybrid);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), Bandwidth::Superwideband);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 960);

        // Hybrid 10ms fullband stereo
        let toc = gen_toc(Mode::Hybrid as i32, 100, Bandwidth::Fullband as i32, 2);
        assert_eq!(opus_packet_get_mode(&[toc]), Mode::Hybrid);
        assert_eq!(opus_packet_get_bandwidth(&[toc]), Bandwidth::Fullband);
        assert_eq!(opus_packet_get_nb_channels(&[toc]), 2);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 480);

        // CELT 2.5ms
        let toc = gen_toc(Mode::CeltOnly as i32, 400, Bandwidth::Fullband as i32, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 120);

        // CELT 5ms
        let toc = gen_toc(Mode::CeltOnly as i32, 200, Bandwidth::Fullband as i32, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 240);

        // CELT 10ms
        let toc = gen_toc(Mode::CeltOnly as i32, 100, Bandwidth::Fullband as i32, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 480);

        // SILK 40ms
        let toc = gen_toc(Mode::SilkOnly as i32, 25, Bandwidth::Narrowband as i32, 1);
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 1920);

        // SILK 60ms
        let toc = gen_toc(
            Mode::SilkOnly as i32,
            400 / 24,
            Bandwidth::Narrowband as i32,
            1,
        );
        assert_eq!(opus_packet_get_samples_per_frame(&[toc], 48000), 2880);
    }

    #[test]
    fn test_opus_encode_decode_roundtrip() {
        // Encode with OpusEncoder, decode with OpusDecoder, verify the result.
        let fs = 48000;
        let frame_size = 960; // 20ms

        let mut encoder =
            OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();
        encoder.set_bitrate(Bitrate::BitsPerSecond(64000));

        let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();

        // Generate a simple 440Hz sine tone
        let mut pcm_in = vec![0.0f32; frame_size];
        for (i, sample) in pcm_in.iter_mut().enumerate() {
            *sample = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
        }

        // Encode
        let mut packet = vec![0u8; 1275];
        let nbytes = encoder
            .encode_float(&pcm_in, frame_size as i32, &mut packet, 1275)
            .expect("Encoding should succeed");
        assert!(
            nbytes >= 2,
            "Should produce a non-trivial packet, got {nbytes} bytes"
        );

        // Verify the packet is parseable
        let parsed = opus_packet_parse(&packet[..nbytes as usize]);
        assert!(
            parsed.is_ok(),
            "Encoded packet should be parseable: {:?}",
            parsed.err()
        );

        // Decode
        let mut pcm_out = vec![0.0f32; frame_size];
        let decoded_samples = decoder
            .decode_float(
                Some(&packet[..nbytes as usize]),
                &mut pcm_out,
                frame_size as i32,
                false,
            )
            .expect("Decoding should succeed");
        assert_eq!(
            decoded_samples, frame_size as i32,
            "Should decode the correct number of samples"
        );

        // Verify the decoded signal has energy (not silent)
        let energy: f64 = pcm_out.iter().map(|&x| x as f64 * x as f64).sum();
        assert!(energy > 0.0, "Decoded signal should have non-zero energy");

        // The codec introduces algorithmic latency, so the first decoded frame may
        // have reduced amplitude. We just verify the output is not all zeros, which
        // confirms the encode/decode roundtrip produced valid audio.
        let max_out = pcm_out.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        assert!(
            max_out > 0.0,
            "Decoded signal should not be completely silent, got max {max_out}"
        );
    }

    #[test]
    fn test_opus_encode_decode_roundtrip_stereo() {
        let fs = 48000;
        let channels = 2;
        let frame_size = 960;

        let mut encoder =
            OpusEncoder::new(SampleRate::Hz48000, Channels::Stereo, Application::Audio).unwrap();
        encoder.set_bitrate(Bitrate::BitsPerSecond(96000));

        let mut decoder = OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo).unwrap();

        // Generate stereo sine
        let mut pcm_in = vec![0.0f32; frame_size * channels as usize];
        for i in 0..frame_size {
            let s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
            pcm_in[i * 2] = s; // left
            pcm_in[i * 2 + 1] = s; // right
        }

        let mut packet = vec![0u8; 1275];
        let nbytes = encoder
            .encode_float(&pcm_in, frame_size as i32, &mut packet, 1275)
            .expect("Stereo encoding should succeed");
        assert!(nbytes >= 2);

        let mut pcm_out = vec![0.0f32; frame_size * channels as usize];
        let decoded = decoder
            .decode_float(
                Some(&packet[..nbytes as usize]),
                &mut pcm_out,
                frame_size as i32,
                false,
            )
            .expect("Stereo decoding should succeed");
        assert_eq!(decoded, frame_size as i32);

        let energy: f64 = pcm_out.iter().map(|&x| x as f64 * x as f64).sum();
        assert!(energy > 0.0, "Stereo decoded signal should have energy");
    }

    #[test]
    fn test_encoder_setters() {
        let mut enc =
            OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Audio).unwrap();

        enc.set_bitrate(Bitrate::BitsPerSecond(32000));
        assert_eq!(enc.bitrate_bps, 32000);

        enc.set_bitrate(Bitrate::Max);
        // Should reset to default
        assert!(enc.bitrate_bps > 0);

        enc.set_complexity(5);
        assert_eq!(enc.complexity, 5);

        enc.set_complexity(100); // Should clamp
        assert_eq!(enc.complexity, 10);

        enc.set_complexity(-5); // Should clamp
        assert_eq!(enc.complexity, 0);

        enc.set_signal(Signal::Voice);
        assert_eq!(enc.signal_type, i32::from(Signal::Voice));

        enc.set_bandwidth(Bandwidth::Wideband);
        assert_eq!(enc.max_bandwidth, Bandwidth::Wideband as i32);
    }
}
