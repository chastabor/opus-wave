use opus_celt::CeltDecoder;
use opus_range_coder::EcCtx;
use opus_silk::decoder::{SilkDecControl, SilkDecoder};

use crate::error::OpusError;
use crate::packet::*;
use crate::types::*;

/// The main Opus decoder.
pub struct OpusDecoder {
    /// Number of output channels (1 or 2).
    channels: i32,
    /// Output sampling rate.
    fs: i32,
    /// SILK decoder state.
    silk_dec: SilkDecoder,
    /// SILK decoder control.
    silk_control: SilkDecControl,
    /// CELT decoder state.
    celt_dec: CeltDecoder,
    /// Decode gain in Q8 dB.
    decode_gain: i32,
    /// Cached linear gain computed from decode_gain.
    decode_gain_float: f32,
    /// Number of channels in the stream.
    stream_channels: i32,
    /// Current bandwidth.
    bandwidth: i32,
    /// Current mode.
    mode: i32,
    /// Previous frame's mode.
    prev_mode: i32,
    /// Current frame size in samples at output Fs.
    frame_size: i32,
    /// Whether previous frame had redundancy.
    prev_redundancy: bool,
    /// Duration of the last decoded packet in samples.
    last_packet_duration: i32,
    /// Soft clipping memory per channel.
    softclip_mem: [f32; 2],
    /// Final range coder state for testing/verification.
    pub range_final: u32,
    /// DNN decoder state (DRED, PLC, OSCE). None when DNN is not loaded.
    #[cfg(feature = "dnn")]
    pub(crate) dnn: Option<Box<crate::dnn_types::DnnDecoderState>>,
}

impl OpusDecoder {
    /// Create a new Opus decoder.
    pub fn new(sample_rate: SampleRate, channels: Channels) -> Result<Self, OpusError> {
        let fs = i32::from(sample_rate);
        let channels = i32::from(channels);

        let silk_dec = SilkDecoder::new();
        let silk_control = SilkDecControl {
            n_channels_api: channels,
            n_channels_internal: channels,
            api_sample_rate: fs,
            internal_sample_rate: 16000,
            payload_size_ms: 20,
            prev_pitch_lag: 0,
        };
        let celt_dec =
            CeltDecoder::new(fs, channels as usize).map_err(|_| OpusError::InternalError)?;

        Ok(OpusDecoder {
            channels,
            fs,
            silk_dec,
            silk_control,
            celt_dec,
            decode_gain: 0,
            decode_gain_float: 1.0,
            stream_channels: channels,
            bandwidth: 0,
            mode: 0,
            prev_mode: 0,
            frame_size: fs / 400,
            prev_redundancy: false,
            last_packet_duration: 0,
            softclip_mem: [0.0; 2],
            range_final: 0,
            #[cfg(feature = "dnn")]
            dnn: None,
        })
    }

    /// Reset the decoder state.
    pub fn reset(&mut self) -> Result<(), OpusError> {
        self.decode_gain = 0;
        self.decode_gain_float = 1.0;
        self.stream_channels = self.channels;
        self.bandwidth = 0;
        self.mode = 0;
        self.prev_mode = 0;
        self.frame_size = self.fs / 400;
        self.prev_redundancy = false;
        self.last_packet_duration = 0;
        self.softclip_mem = [0.0; 2];
        self.range_final = 0;
        self.silk_dec.reset();
        self.celt_dec = CeltDecoder::new(self.fs, self.channels as usize)
            .map_err(|_| OpusError::InternalError)?;
        self.celt_dec.signalling = false;
        Ok(())
    }

    pub fn sample_rate(&self) -> i32 {
        self.fs
    }

    pub fn channels(&self) -> i32 {
        self.channels
    }

    pub fn last_packet_duration(&self) -> i32 {
        self.last_packet_duration
    }

    pub fn set_gain(&mut self, gain_q8: i32) {
        self.decode_gain = gain_q8;
        let db = gain_q8 as f32 / 256.0;
        self.decode_gain_float = 10.0f32.powf(db / 20.0);
    }

    pub fn get_gain(&self) -> i32 {
        self.decode_gain
    }

    /// Load DNN models from a binary weight blob and enable DNN-enhanced decoding.
    ///
    /// The blob must contain the RDOVAE decoder, PLC (FARGAN + PitchDNN),
    /// and optionally OSCE (LACE/NoLACE) weights. This is equivalent to
    /// the C `OPUS_SET_DNN_BLOB` CTL.
    ///
    /// Once loaded, the decoder will automatically:
    /// - Parse DRED extensions (ID 126) from incoming packets
    /// - Use deep PLC (FARGAN) for packet loss concealment
    /// - Apply OSCE speech enhancement when available
    ///
    /// Requires the `dnn` feature.
    #[cfg(feature = "dnn")]
    pub fn load_dnn(&mut self, data: &[u8]) -> Result<(), OpusError> {
        let state = crate::dnn_types::DnnDecoderState::from_blob(data)?;
        self.dnn = Some(Box::new(state));
        Ok(())
    }

    /// Returns whether DNN models are loaded and ready.
    ///
    /// Requires the `dnn` feature.
    #[cfg(feature = "dnn")]
    pub fn dnn_loaded(&self) -> bool {
        self.dnn.as_ref().is_some_and(|dnn| dnn.loaded)
    }

    /// Decode a single Opus frame.
    fn decode_frame(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, OpusError> {
        let f20 = self.fs / 50;
        let f10 = f20 >> 1;
        let f5 = f10 >> 1;
        let f2_5 = f5 >> 1;

        if frame_size < f2_5 {
            return Err(OpusError::BufferTooSmall);
        }
        let frame_size = frame_size.min(self.fs / 25 * 3);

        // Determine mode and audiosize
        let (has_data, audiosize, mode, bandwidth) = if let Some(d) = data {
            if d.len() <= 1 {
                let mode = if self.prev_redundancy {
                    Mode::CeltOnly as i32
                } else {
                    self.prev_mode
                };
                (false, frame_size.min(self.frame_size), mode, 0)
            } else {
                (true, self.frame_size, self.mode, self.bandwidth)
            }
        } else {
            let mode = if self.prev_redundancy {
                Mode::CeltOnly as i32
            } else {
                self.prev_mode
            };
            (false, frame_size, mode, 0)
        };

        // No previous mode and no data => output silence
        if !has_data && mode == 0 {
            let n = audiosize as usize * self.channels as usize;
            for s in pcm[..n].iter_mut() {
                *s = 0.0;
            }
            return Ok(audiosize);
        }

        // DNN-based PLC: if we have DNN loaded and data is missing, try deep PLC
        #[cfg(feature = "dnn")]
        if !has_data && self.dnn.as_ref().is_some_and(|d| d.loaded) {
            let n = (audiosize * self.channels) as usize;
            let mut plc_i16 = vec![0i16; n];
            if crate::dnn_decoder::decoder_plc_conceal(self, &mut plc_i16) {
                for i in 0..n.min(pcm.len()) {
                    pcm[i] = plc_i16[i] as f32 * (1.0 / 32768.0);
                }
                self.prev_mode = mode;
                return Ok(audiosize);
            }
        }

        // For PLC with long frames, break into 20ms chunks
        if !has_data && audiosize > f20 {
            let mut remaining = audiosize;
            let mut offset = 0usize;
            while remaining > 0 {
                let chunk = remaining.min(f20);
                let ret = self.decode_frame(None, &mut pcm[offset..], chunk, false)?;
                offset += ret as usize * self.channels as usize;
                remaining -= ret;
            }
            return Ok(audiosize);
        }

        // Adjust audiosize for PLC
        let audiosize = if !has_data {
            let mut a = audiosize;
            if a > f20 {
                a = f20;
            } else if a > f10 {
                a = f10;
            } else if mode != Mode::SilkOnly as i32 && a > f5 && a < f10 {
                a = f5;
            }
            a
        } else {
            audiosize
        };

        let frame_size = audiosize;

        // Initialize range coder if we have data
        let mut ec = if has_data {
            data.map(EcCtx::dec_init)
        } else {
            None
        };

        // === SILK processing ===
        if mode != Mode::CeltOnly as i32 {
            if self.prev_mode == Mode::CeltOnly as i32 {
                self.silk_dec.reset();
            }

            let silk_internal_rate = if mode == Mode::SilkOnly as i32 {
                match bandwidth {
                    x if x == Bandwidth::Narrowband as i32 => 8000,
                    x if x == Bandwidth::Mediumband as i32 => 12000,
                    _ => 16000,
                }
            } else {
                16000
            };

            self.silk_control.n_channels_internal = self.stream_channels;
            self.silk_control.n_channels_api = self.channels;
            self.silk_control.api_sample_rate = self.fs;
            self.silk_control.internal_sample_rate = silk_internal_rate;
            self.silk_control.payload_size_ms = (1000 * audiosize / self.fs).max(10);

            let lost_flag = if !has_data {
                1
            } else if decode_fec {
                2
            } else {
                0
            };
            let first_frame = true; // simplified

            let silk_samples = frame_size * self.channels;
            let mut silk_out_i16 = vec![0i16; silk_samples as usize];
            let mut silk_frame_size = 0i32;

            // Construct OSCE post-filter if DNN is loaded, else no-op.
            // Uses macro to avoid duplicating decode call for both ec paths.
            macro_rules! silk_decode_with_filter {
                ($ec:expr) => {{
                    #[cfg(feature = "dnn")]
                    {
                        // Channel 0 is used for mono; stereo OSCE uses per-channel state
                        if let Some(ref mut dnn) = self.dnn
                            && dnn.osce.loaded
                        {
                            let mut pf = crate::dnn_decoder::OscePostFilter {
                                model: &dnn.osce,
                                feature_state: &mut dnn.osce_feature_state[0],
                                lace_state: &mut dnn.osce_lace_state[0],
                                nolace_state: &mut dnn.osce_nolace_state[0],
                            };
                            self.silk_dec.decode(
                                &mut self.silk_control,
                                lost_flag,
                                first_frame,
                                $ec,
                                &mut silk_out_i16,
                                &mut silk_frame_size,
                                &mut pf,
                            )
                        } else {
                            self.silk_dec.decode(
                                &mut self.silk_control,
                                lost_flag,
                                first_frame,
                                $ec,
                                &mut silk_out_i16,
                                &mut silk_frame_size,
                                &mut opus_silk::decoder::NoPostFilter,
                            )
                        }
                    }
                    #[cfg(not(feature = "dnn"))]
                    {
                        self.silk_dec.decode(
                            &mut self.silk_control,
                            lost_flag,
                            first_frame,
                            $ec,
                            &mut silk_out_i16,
                            &mut silk_frame_size,
                            &mut opus_silk::decoder::NoPostFilter,
                        )
                    }
                }};
            }

            if let Some(ref mut ec) = ec {
                let silk_ret = silk_decode_with_filter!(ec);
                if silk_ret != 0 && !has_data {
                    silk_out_i16.fill(0);
                }
            } else {
                let dummy_buf = [0u8; 2];
                let mut dummy_ec = EcCtx::dec_init(&dummy_buf);
                let silk_ret = silk_decode_with_filter!(&mut dummy_ec);
                if silk_ret != 0 {
                    silk_out_i16.fill(0);
                }
            }

            // Convert SILK i16 output to f32 and write to pcm
            let n = (frame_size * self.channels) as usize;
            for i in 0..n.min(silk_out_i16.len()).min(pcm.len()) {
                pcm[i] = silk_out_i16[i] as f32 * (1.0 / 32768.0);
            }
        }

        // === Redundancy detection ===
        let mut redundancy = false;
        let mut celt_to_silk = false;
        let mut start_band: i32 = 0;
        let mut data_len = if has_data {
            data.map(|d| d.len() as i32).unwrap_or(0)
        } else {
            0
        };

        if !decode_fec
            && mode != Mode::CeltOnly as i32
            && has_data
            && let Some(ref mut ec) = ec
        {
            let bits_left = data_len * 8 - ec.tell();
            let threshold = 17 + if mode == Mode::Hybrid as i32 { 20 } else { 0 };
            if bits_left >= threshold {
                if mode == Mode::Hybrid as i32 {
                    redundancy = ec.dec_bit_logp(12);
                } else {
                    redundancy = true;
                }
                if redundancy {
                    celt_to_silk = ec.dec_bit_logp(1);
                    let redundancy_bytes = if mode == Mode::Hybrid as i32 {
                        ec.dec_uint(256) as i32 + 2
                    } else {
                        data_len - ((ec.tell() + 7) >> 3)
                    };
                    data_len -= redundancy_bytes;
                    if data_len * 8 < ec.tell() {
                        data_len = 0;
                        redundancy = false;
                    } else {
                        ec.storage -= redundancy_bytes as u32;
                    }
                }
            }
        }

        if mode != Mode::CeltOnly as i32 {
            start_band = 17;
        }

        let endband = if bandwidth == Bandwidth::Narrowband as i32 {
            13
        } else if bandwidth <= Bandwidth::Wideband as i32 {
            17
        } else if bandwidth == Bandwidth::Superwideband as i32 {
            19
        } else {
            21
        };

        // === CELT decoding ===
        self.celt_dec.start = start_band as usize;
        self.celt_dec.end = endband as usize;
        self.celt_dec.stream_channels = self.stream_channels as usize;
        self.celt_dec.signalling = false;

        if mode != Mode::SilkOnly as i32 {
            let celt_frame_size = f20.min(frame_size);
            if mode != self.prev_mode && self.prev_mode > 0 && !self.prev_redundancy {
                // Reset CELT on mode transitions
                if let Ok(new_celt) = CeltDecoder::new(self.fs, self.channels as usize) {
                    self.celt_dec = new_celt;
                    self.celt_dec.start = start_band as usize;
                    self.celt_dec.end = endband as usize;
                    self.celt_dec.stream_channels = self.stream_channels as usize;
                    self.celt_dec.signalling = false;
                }
            }

            let celt_data = if decode_fec || !has_data {
                &[] as &[u8]
            } else {
                data.unwrap_or(&[])
            };

            // For hybrid/combined modes, CELT adds to SILK output
            let _ = self.celt_dec.decode_with_ec(
                celt_data,
                pcm,
                celt_frame_size as usize / self.celt_dec.downsample,
                ec.as_mut(),
            );
            self.range_final = self.celt_dec.rng;
        } else {
            // SILK-only: CELT not used for audio, but need fade-out on transitions
            if mode != Mode::CeltOnly as i32 {
                // pcm already has SILK output
            }
            if let Some(ref ec) = ec {
                self.range_final = ec.rng;
            }
        }

        // === Apply decode gain ===
        if self.decode_gain != 0 {
            let gain = self.decode_gain_float;
            let n = frame_size as usize * self.channels as usize;
            for s in pcm[..n].iter_mut() {
                *s *= gain;
            }
        }

        // Update state
        if data_len <= 1 && !has_data {
            self.range_final = 0;
        }
        self.prev_mode = mode;
        self.prev_redundancy = redundancy && !celt_to_silk;

        Ok(audiosize)
    }

    /// Decode an Opus packet to float PCM.
    pub fn decode_float(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [f32],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, OpusError> {
        if frame_size <= 0 {
            return Err(OpusError::BadArg);
        }
        if (decode_fec || data.is_none()) && frame_size % (self.fs / 400) != 0 {
            return Err(OpusError::BadArg);
        }

        // PLC/DTX
        let is_plc = data.is_none() || data.map(|d| d.is_empty()).unwrap_or(true);

        if is_plc {
            let mut pcm_count = 0i32;
            while pcm_count < frame_size {
                let ret = self.decode_frame(
                    None,
                    &mut pcm[pcm_count as usize * self.channels as usize..],
                    frame_size - pcm_count,
                    false,
                )?;
                pcm_count += ret;
            }
            self.last_packet_duration = pcm_count;
            return Ok(pcm_count);
        }

        // Safety: is_plc returns early above when data is None
        let data = data.expect("data guaranteed Some by is_plc check");

        let packet_mode = i32::from(opus_packet_get_mode(data));
        let packet_bandwidth = i32::from(opus_packet_get_bandwidth(data));
        let packet_frame_size = opus_packet_get_samples_per_frame(data, self.fs);
        let packet_stream_channels = opus_packet_get_nb_channels(data);

        let parsed = opus_packet_parse(data)?;
        let count = parsed.frame_sizes.len() as i32;

        // Parse DRED extensions from packet padding area
        #[cfg(feature = "dnn")]
        if parsed.padding_len > 0 {
            let pad_end = parsed.padding_offset + parsed.padding_len;
            if pad_end <= data.len() {
                let ext_data = &data[parsed.padding_offset..pad_end];
                let mut nb_ext = 0i32;
                let mut extensions = Vec::new();
                if crate::extensions::opus_packet_extensions_parse(
                    ext_data,
                    &mut nb_ext,
                    &mut extensions,
                )
                .is_ok()
                {
                    let frame_offset = 0i32; // DRED offset relative to current packet
                    for ext in &extensions {
                        if ext.id == opus_dnn::dred::DRED_EXTENSION_ID as i32 {
                            crate::dnn_decoder::decoder_process_dred_extension(
                                self,
                                ext,
                                frame_offset,
                            );
                        }
                    }
                }
            }
        }

        // FEC
        if decode_fec {
            if frame_size < packet_frame_size
                || packet_mode == Mode::CeltOnly as i32
                || self.mode == Mode::CeltOnly as i32
            {
                return self.decode_float(None, pcm, frame_size, false);
            }
            if frame_size - packet_frame_size != 0 {
                self.decode_float(None, pcm, frame_size - packet_frame_size, false)?;
            }
            self.mode = packet_mode;
            self.bandwidth = packet_bandwidth;
            self.frame_size = packet_frame_size;
            self.stream_channels = packet_stream_channels;
            let offset = (frame_size - packet_frame_size) as usize * self.channels as usize;
            self.decode_frame(
                Some(
                    &data[parsed.payload_offset
                        ..parsed.payload_offset + parsed.frame_sizes[0] as usize],
                ),
                &mut pcm[offset..],
                packet_frame_size,
                true,
            )?;
            self.last_packet_duration = frame_size;
            return Ok(frame_size);
        }

        if count * packet_frame_size > frame_size {
            return Err(OpusError::BufferTooSmall);
        }

        // Update state
        self.mode = packet_mode;
        self.bandwidth = packet_bandwidth;
        self.frame_size = packet_frame_size;
        self.stream_channels = packet_stream_channels;

        let mut nb_samples = 0i32;
        let mut data_offset = parsed.payload_offset;
        for i in 0..count as usize {
            let frame_len = parsed.frame_sizes[i] as usize;
            let pcm_offset = nb_samples as usize * self.channels as usize;
            let ret = self.decode_frame(
                Some(&data[data_offset..data_offset + frame_len]),
                &mut pcm[pcm_offset..],
                frame_size - nb_samples,
                false,
            )?;
            data_offset += frame_len;
            nb_samples += ret;
        }

        self.last_packet_duration = nb_samples;

        // Update DNN PLC state with successfully decoded audio
        #[cfg(feature = "dnn")]
        if self.dnn.as_ref().is_some_and(|d| d.loaded) {
            let total = nb_samples as usize * self.channels as usize;
            let pcm_i16: Vec<i16> = pcm[..total]
                .iter()
                .map(|&s| (s * 32768.0).round().clamp(-32768.0, 32767.0) as i16)
                .collect();
            crate::dnn_decoder::decoder_plc_update(self, &pcm_i16);
        }

        // Soft clipping
        opus_pcm_soft_clip(pcm, nb_samples, self.channels, &mut self.softclip_mem);

        Ok(nb_samples)
    }

    /// Decode an Opus packet to 16-bit integer PCM.
    pub fn decode(
        &mut self,
        data: Option<&[u8]>,
        pcm: &mut [i16],
        frame_size: i32,
        decode_fec: bool,
    ) -> Result<i32, OpusError> {
        let max_samples = frame_size as usize * self.channels as usize;
        let mut float_pcm = vec![0.0f32; max_samples];
        let ret = self.decode_float(data, &mut float_pcm, frame_size, decode_fec)?;
        let samples = ret as usize * self.channels as usize;
        for i in 0..samples {
            let s = (float_pcm[i] * 32768.0 + 0.5).floor() as i32;
            pcm[i] = s.clamp(-32768, 32767) as i16;
        }
        Ok(ret)
    }
}

/// Soft clipping to prevent output from exceeding [-1, 1].
fn opus_pcm_soft_clip(pcm: &mut [f32], n: i32, channels: i32, mem: &mut [f32; 2]) {
    let n = n as usize;
    let c_count = channels as usize;
    if c_count == 0 || n == 0 {
        return;
    }
    // Clamp to [-2, 2]
    for s in pcm[..n * c_count].iter_mut() {
        *s = s.clamp(-2.0, 2.0);
    }
    for c in 0..c_count {
        let mut a = mem[c];
        for i in 0..n {
            let idx = i * c_count + c;
            if pcm[idx] * a >= 0.0 {
                break;
            }
            pcm[idx] += a * pcm[idx] * pcm[idx];
        }

        let mut curr = 0;
        loop {
            let mut clip_idx = n;
            for i in curr..n {
                let idx = i * c_count + c;
                if pcm[idx] > 1.0 || pcm[idx] < -1.0 {
                    clip_idx = i;
                    break;
                }
            }
            if clip_idx == n {
                a = 0.0;
                break;
            }

            let peak_val = pcm[clip_idx * c_count + c];
            let mut start = clip_idx;
            let mut end = clip_idx;
            let mut maxval = peak_val.abs();

            while start > 0 && peak_val * pcm[(start - 1) * c_count + c] >= 0.0 {
                start -= 1;
            }
            while end < n && peak_val * pcm[end * c_count + c] >= 0.0 {
                if pcm[end * c_count + c].abs() > maxval {
                    maxval = pcm[end * c_count + c].abs();
                }
                end += 1;
            }

            a = (maxval - 1.0) / (maxval * maxval);
            a += a * 2.4e-7;
            if peak_val > 0.0 {
                a = -a;
            }
            for i in start..end {
                let idx = i * c_count + c;
                pcm[idx] += a * pcm[idx] * pcm[idx];
            }

            curr = end;
            if curr >= n {
                break;
            }
        }
        mem[c] = a;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_create() {
        let dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Stereo);
        assert!(dec.is_ok());
        let dec = dec.unwrap();
        assert_eq!(dec.sample_rate(), 48000);
        assert_eq!(dec.channels(), 2);
    }

    #[test]
    fn test_decoder_reset() {
        let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
        dec.set_gain(100);
        dec.reset().unwrap();
        assert_eq!(dec.get_gain(), 0);
    }

    #[test]
    fn test_plc_outputs_audio() {
        let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono).unwrap();
        let mut pcm = vec![0.0f32; 960];
        let ret = dec.decode_float(None, &mut pcm, 960, false);
        assert!(ret.is_ok());
        assert_eq!(ret.unwrap(), 960);
    }

    #[test]
    fn test_soft_clip() {
        let mut pcm = vec![0.0f32; 10];
        pcm[3] = 1.5;
        pcm[4] = 1.8;
        pcm[5] = 1.2;
        let mut mem = [0.0f32; 2];
        opus_pcm_soft_clip(&mut pcm, 10, 1, &mut mem);
        for &s in &pcm {
            assert!(
                (-1.001..=1.001).contains(&s),
                "Sample {s} out of range after soft clip"
            );
        }
    }
}
