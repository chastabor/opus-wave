//! Multistream Opus encoder for encoding packets with more than 2 channels.
//!
//! Port of C opus_multistream_encoder.c and related functions.

use crate::encoder::OpusEncoder;
use crate::error::OpusError;
use crate::multistream::ChannelLayout;
use crate::types::*;

/// Multistream Opus encoder.
///
/// Manages multiple `OpusEncoder` instances to encode audio with more than 2
/// channels. Each stream encodes either one (mono) or two (coupled/stereo)
/// channels. The output is a concatenation of self-delimited Opus packets for
/// all streams except the last, which uses the standard Opus packet format.
pub struct OpusMSEncoder {
    layout: ChannelLayout,
    encoders: Vec<OpusEncoder>,
    #[allow(dead_code)]
    application: Application,
    fs: i32,
    bitrate_bps: i32,
    /// Index of the LFE stream, or -1 if none.
    lfe_stream: i32,
    /// Whether this is a surround encoder.
    surround: bool,
}

impl OpusMSEncoder {
    /// Create a new multistream encoder.
    ///
    /// - `fs`: Sample rate (8000, 12000, 16000, 24000, or 48000 Hz)
    /// - `channels`: Total number of input channels (1-255)
    /// - `streams`: Number of independent streams
    /// - `coupled_streams`: Number of stereo (2-channel) coupled streams
    /// - `mapping`: Channel mapping table (`channels` entries). Each entry maps
    ///   an input channel to a stream channel index. Indices 0..2*coupled_streams
    ///   map to coupled stream pairs, 2*coupled_streams..2*coupled_streams+(streams-coupled_streams)
    ///   map to mono streams. Index 255 means the channel is muted/ignored.
    /// - `application`: The Opus application type.
    ///
    /// The first `coupled_streams` streams encode stereo pairs; the remaining
    /// `streams - coupled_streams` streams encode mono.
    pub fn new(
        sample_rate: SampleRate,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: Application,
    ) -> Result<Self, OpusError> {
        let fs = i32::from(sample_rate);
        if channels == 0 || channels > 255 {
            return Err(OpusError::BadArg);
        }
        if streams == 0 || streams > 255 {
            return Err(OpusError::BadArg);
        }
        if coupled_streams > streams {
            return Err(OpusError::BadArg);
        }
        if streams > 255 - coupled_streams {
            return Err(OpusError::BadArg);
        }
        if mapping.len() < channels {
            return Err(OpusError::BadArg);
        }

        let mut layout = ChannelLayout {
            nb_channels: channels,
            nb_streams: streams,
            nb_coupled_streams: coupled_streams,
            mapping: [0u8; 256],
        };
        layout.mapping[..channels].copy_from_slice(&mapping[..channels]);

        if !layout.validate() {
            return Err(OpusError::BadArg);
        }

        // Create one encoder per stream
        let mut encoders = Vec::with_capacity(streams);
        for s in 0..streams {
            let ch = if s < coupled_streams {
                Channels::Stereo
            } else {
                Channels::Mono
            };
            let enc = OpusEncoder::new(sample_rate, ch, application)?;
            encoders.push(enc);
        }

        let default_bitrate = if application == Application::Voip {
            20000
        } else {
            64000
        };
        let total_bitrate = default_bitrate * streams as i32;

        Ok(OpusMSEncoder {
            layout,
            encoders,
            application,
            fs,
            bitrate_bps: total_bitrate,
            lfe_stream: -1,
            surround: false,
        })
    }

    /// Create a multistream encoder configured for surround sound.
    ///
    /// Like `new()`, but additionally identifies the LFE stream for special
    /// bitrate treatment.
    pub fn new_surround(
        sample_rate: SampleRate,
        channels: usize,
        streams: usize,
        coupled_streams: usize,
        mapping: &[u8],
        application: Application,
        lfe_stream: i32,
    ) -> Result<Self, OpusError> {
        let mut enc = Self::new(
            sample_rate,
            channels,
            streams,
            coupled_streams,
            mapping,
            application,
        )?;
        enc.surround = true;
        if lfe_stream >= 0 && (lfe_stream as usize) < streams {
            enc.lfe_stream = lfe_stream;
        }
        Ok(enc)
    }

    /// Set the total target bitrate for all streams.
    pub fn set_bitrate(&mut self, bitrate: i32) {
        self.bitrate_bps = bitrate.clamp(500, 512000 * self.layout.nb_streams as i32);
    }

    /// Get the total number of input channels.
    pub fn channels(&self) -> usize {
        self.layout.nb_channels
    }

    /// Get the number of streams.
    pub fn streams(&self) -> usize {
        self.layout.nb_streams
    }

    /// Get the number of coupled (stereo) streams.
    pub fn coupled_streams(&self) -> usize {
        self.layout.nb_coupled_streams
    }

    /// Get the sample rate.
    pub fn sample_rate(&self) -> i32 {
        self.fs
    }

    /// Allocate per-stream bitrates from the total bitrate.
    ///
    /// Coupled (stereo) streams get approximately 1.5x the rate of mono streams.
    /// If an LFE stream is present, it gets a fixed low bitrate.
    fn rate_allocation(&self) -> Vec<i32> {
        let nb_streams = self.layout.nb_streams;
        let nb_coupled = self.layout.nb_coupled_streams;
        let nb_uncoupled = nb_streams - nb_coupled;

        let mut rates = vec![0i32; nb_streams];

        if nb_streams == 0 {
            return rates;
        }

        let mut total_available = self.bitrate_bps;

        // If there is an LFE stream, assign it a low fixed rate first.
        let lfe_rate = 3500;
        if self.lfe_stream >= 0 && (self.lfe_stream as usize) < nb_streams {
            let lfe_idx = self.lfe_stream as usize;
            rates[lfe_idx] = lfe_rate;
            total_available -= lfe_rate;
        }

        // The remaining budget is split among non-LFE streams.
        // Coupled streams count as 1.5 "units", uncoupled streams as 1 "unit".
        // Count non-LFE coupled vs uncoupled streams
        let non_lfe_coupled = if self.lfe_stream >= 0 && (self.lfe_stream as usize) < nb_coupled {
            nb_coupled - 1
        } else {
            nb_coupled
        };
        let non_lfe_uncoupled = if self.lfe_stream >= 0
            && (self.lfe_stream as usize) >= nb_coupled
            && (self.lfe_stream as usize) < nb_streams
        {
            nb_uncoupled - 1
        } else {
            nb_uncoupled
        };
        // Total weight: coupled = 1.5 unit each, uncoupled = 1.0 unit each
        // Multiply by 2 to avoid floats: coupled = 3, uncoupled = 2
        let total_weight = non_lfe_coupled * 3 + non_lfe_uncoupled * 2;
        if total_weight == 0 {
            return rates;
        }

        // Distribute remaining bitrate
        let unit_rate = (total_available * 2) / total_weight as i32;

        for (s, rate) in rates.iter_mut().enumerate().take(nb_streams) {
            if self.lfe_stream >= 0 && s == self.lfe_stream as usize {
                continue; // Already assigned
            }
            if s < nb_coupled {
                // Coupled stream gets 1.5 * unit_rate
                *rate = (unit_rate * 3) / 2;
            } else {
                // Mono stream gets 1.0 * unit_rate
                *rate = unit_rate;
            }
        }

        // Ensure minimum bitrate per stream
        for rate in rates.iter_mut() {
            if *rate < 500 {
                *rate = 500;
            }
        }

        rates
    }

    /// Encode a multistream Opus frame from interleaved float PCM.
    ///
    /// - `pcm`: Interleaved input samples (`frame_size * channels`)
    /// - `frame_size`: Number of samples per channel
    /// - `data`: Output buffer for the encoded multistream packet
    /// - `max_data_bytes`: Maximum number of bytes to write
    ///
    /// Returns the total number of bytes written. The output contains
    /// self-delimited packets for each non-final stream, followed by a
    /// standard-format packet for the final stream.
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

        let nb_channels = self.layout.nb_channels;
        let nb_streams = self.layout.nb_streams;
        let nb_coupled = self.layout.nb_coupled_streams;

        let total_samples = frame_size as usize * nb_channels;
        if pcm.len() < total_samples {
            return Err(OpusError::BadArg);
        }

        // Allocate bitrates per stream
        let rates = self.rate_allocation();

        // Maximum bytes per stream (estimate)
        let max_per_stream = max_data_bytes as usize / nb_streams.max(1);
        let max_per_stream = max_per_stream.max(64);

        // Temporary buffers for each stream's encoded output
        let mut stream_packets: Vec<Vec<u8>> = Vec::with_capacity(nb_streams);

        for (s, rate) in rates.iter().enumerate().take(nb_streams) {
            let stream_channels = if s < nb_coupled { 2 } else { 1 };

            // Set the per-stream bitrate
            self.encoders[s].set_bitrate(Bitrate::BitsPerSecond(*rate));

            // Gather the input channels for this stream from the interleaved PCM
            let mut stream_pcm = vec![0.0f32; frame_size as usize * stream_channels];

            if s < nb_coupled {
                // Coupled (stereo) stream: find left and right channels
                let left_chan = find_channel_for_stream_left(&self.layout, s);
                let right_chan = find_channel_for_stream_right(&self.layout, s);

                for i in 0..frame_size as usize {
                    stream_pcm[i * 2] = if left_chan >= 0 {
                        pcm[i * nb_channels + left_chan as usize]
                    } else {
                        0.0
                    };
                    stream_pcm[i * 2 + 1] = if right_chan >= 0 {
                        pcm[i * nb_channels + right_chan as usize]
                    } else {
                        0.0
                    };
                }
            } else {
                // Mono stream: find the mono channel
                let mono_chan = find_channel_for_stream_mono(&self.layout, s);

                for i in 0..frame_size as usize {
                    stream_pcm[i] = if mono_chan >= 0 {
                        pcm[i * nb_channels + mono_chan as usize]
                    } else {
                        0.0
                    };
                }
            }

            // Encode this stream
            let mut buf = vec![0u8; max_per_stream.max(1275)];
            let buf_len = buf.len() as i32;
            let nbytes =
                self.encoders[s].encode_float(&stream_pcm, frame_size, &mut buf, buf_len)?;

            stream_packets.push(buf[..nbytes as usize].to_vec());
        }

        // Assemble the multistream packet:
        // For streams 0..nb_streams-1: write in self-delimited format
        // For stream nb_streams-1 (last): write in normal format
        let mut offset = 0usize;
        let out_len = data.len().min(max_data_bytes as usize);

        for (s, pkt) in stream_packets.iter().enumerate().take(nb_streams) {
            let is_last = s == nb_streams - 1;

            if is_last {
                // Final stream: write the packet directly (standard format)
                if offset + pkt.len() > out_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[offset..offset + pkt.len()].copy_from_slice(pkt);
                offset += pkt.len();
            } else {
                // Non-final stream: write in self-delimited format
                // The self-delimited format inserts the last frame's size
                // after the TOC byte (and any code 2/3 header bytes)
                let sd_data = make_self_delimited(pkt)?;
                if offset + sd_data.len() > out_len {
                    return Err(OpusError::BufferTooSmall);
                }
                data[offset..offset + sd_data.len()].copy_from_slice(&sd_data);
                offset += sd_data.len();
            }
        }

        Ok(offset as i32)
    }

    /// Encode a multistream Opus frame from interleaved i16 PCM.
    pub fn encode(
        &mut self,
        pcm: &[i16],
        frame_size: i32,
        data: &mut [u8],
        max_data_bytes: i32,
    ) -> Result<i32, OpusError> {
        let total_samples = frame_size as usize * self.layout.nb_channels;
        if pcm.len() < total_samples {
            return Err(OpusError::BadArg);
        }

        let float_pcm: Vec<f32> = pcm[..total_samples]
            .iter()
            .map(|&s| s as f32 * (1.0 / 32768.0))
            .collect();

        self.encode_float(&float_pcm, frame_size, data, max_data_bytes)
    }
}

/// Find the first input channel that maps to the left of a coupled stream.
/// Coupled stream `s` has left = stream index `2*s`.
fn find_channel_for_stream_left(layout: &ChannelLayout, stream_id: usize) -> i32 {
    let target = (stream_id * 2) as u8;
    for i in 0..layout.nb_channels {
        if layout.mapping[i] == target {
            return i as i32;
        }
    }
    -1
}

/// Find the first input channel that maps to the right of a coupled stream.
/// Coupled stream `s` has right = stream index `2*s + 1`.
fn find_channel_for_stream_right(layout: &ChannelLayout, stream_id: usize) -> i32 {
    let target = (stream_id * 2 + 1) as u8;
    for i in 0..layout.nb_channels {
        if layout.mapping[i] == target {
            return i as i32;
        }
    }
    -1
}

/// Find the first input channel that maps to a mono (uncoupled) stream.
///
/// Per the decoder's `get_mono_channel`: for stream index `s` (where `s >= nb_coupled`),
/// the mapping value is `s + nb_coupled_streams`. This works because:
/// - Coupled streams 0..nb_coupled use mapping values 0..2*nb_coupled (left/right pairs)
/// - Mono streams nb_coupled..nb_streams use mapping values
///   2*nb_coupled..nb_coupled+nb_streams
/// - This gives a contiguous range where max_channel = nb_streams + nb_coupled
fn find_channel_for_stream_mono(layout: &ChannelLayout, stream_id: usize) -> i32 {
    let target = (stream_id + layout.nb_coupled_streams) as u8;
    for i in 0..layout.nb_channels {
        if layout.mapping[i] == target {
            return i as i32;
        }
    }
    -1
}

/// Convert a standard Opus packet to self-delimited format.
///
/// In self-delimited format, the last frame's size is explicitly encoded
/// after the TOC byte (and after any code 2/3 header). This allows the
/// decoder to know how many bytes the stream consumes without reading
/// to the end of the buffer.
///
/// The format mirrors what `opus_packet_parse_self_delimited` in packet.rs
/// expects:
///
/// For a code 0 (single frame) packet `[TOC, payload...]`:
///   Self-delimited: `[TOC, size_bytes..., payload...]`
///   where size_bytes encode the payload length.
///
/// For code 1 (2 CBR frames) `[TOC, frame1, frame2]`:
///   Self-delimited: `[TOC, size_bytes..., frame1, frame2]`
///   where size_bytes encode one frame's size (both are equal).
///
/// For code 2 (2 VBR frames) `[TOC, sz1_bytes, frame1, frame2]`:
///   Self-delimited: `[TOC, sz1_bytes, size_bytes..., frame1, frame2]`
///   where sz1_bytes is frame 1's size and size_bytes is frame 2's size.
///
/// For code 3 (multiple frames) with CBR:
///   Self-delimited: `[TOC, count_byte, size_bytes..., frames...]`
///
/// For code 3 VBR:
///   Self-delimited: `[TOC, count_byte, sz1..szN-1, size_bytes..., frames...]`
///   where size_bytes encode the last frame's size.
fn make_self_delimited(packet: &[u8]) -> Result<Vec<u8>, OpusError> {
    if packet.is_empty() {
        return Err(OpusError::InvalidPacket);
    }

    let toc = packet[0];
    let code = toc & 0x3;

    let mut result = Vec::with_capacity(packet.len() + 2);

    match code {
        0 => {
            // Single frame: TOC + payload
            // Self-delimited: TOC + size(payload_len) + payload
            let payload_len = packet.len() - 1;
            result.push(toc);
            encode_size(payload_len, &mut result);
            result.extend_from_slice(&packet[1..]);
        }
        1 => {
            // Two CBR frames: TOC + frame1 + frame2, both same size
            let payload_len = packet.len() - 1;
            if payload_len & 1 != 0 {
                return Err(OpusError::InvalidPacket);
            }
            let frame_size = payload_len / 2;
            result.push(toc);
            encode_size(frame_size, &mut result);
            result.extend_from_slice(&packet[1..]);
        }
        2 => {
            // Two VBR frames: TOC + size1 + frame1 + frame2
            // Parse size1 first
            if packet.len() < 2 {
                return Err(OpusError::InvalidPacket);
            }
            let (sz1, sz1_bytes) = parse_size_from_slice(&packet[1..])?;
            let header_end = 1 + sz1_bytes;
            let remaining = packet.len() - header_end;
            if (sz1 as usize) > remaining {
                return Err(OpusError::InvalidPacket);
            }
            let sz2 = remaining - sz1 as usize;

            // Self-delimited: TOC + size1 + size2(self-delim) + frame1 + frame2
            result.push(toc);
            encode_size(sz1 as usize, &mut result);
            encode_size(sz2, &mut result);
            result.extend_from_slice(&packet[header_end..]);
        }
        3 => {
            // Code 3: Multiple frames
            if packet.len() < 2 {
                return Err(OpusError::InvalidPacket);
            }
            let ch = packet[1];
            let count = (ch & 0x3F) as usize;
            let vbr = ch & 0x80 != 0;
            let has_padding = ch & 0x40 != 0;

            let mut pos = 2usize;

            // Parse padding
            let mut padding_bytes = 0usize;
            if has_padding {
                loop {
                    if pos >= packet.len() {
                        return Err(OpusError::InvalidPacket);
                    }
                    let p = packet[pos];
                    pos += 1;
                    let tmp = if p == 255 { 254 } else { p as usize };
                    padding_bytes += tmp;
                    if p != 255 {
                        break;
                    }
                }
            }

            if vbr {
                // VBR: parse sizes for frames 0..count-1
                let mut frame_sizes = Vec::with_capacity(count);
                let mut total_sizes = 0usize;
                for _ in 0..count.saturating_sub(1) {
                    let (sz, bytes) = parse_size_from_slice(&packet[pos..])?;
                    pos += bytes;
                    frame_sizes.push(sz as usize);
                    total_sizes += sz as usize;
                }
                // Last frame size is implicit in normal format
                let data_start = pos;
                let total_data = packet.len() - data_start - padding_bytes;
                if total_sizes > total_data {
                    return Err(OpusError::InvalidPacket);
                }
                let last_frame_size = total_data - total_sizes;
                frame_sizes.push(last_frame_size);

                // Build self-delimited output:
                // TOC + count_byte + sizes_for_first_N-1 + size_for_last(self-delim) + all_frames + padding
                result.push(toc);
                result.push(ch);

                // Re-encode padding if present
                if has_padding {
                    let mut rem = padding_bytes;
                    while rem >= 254 {
                        result.push(255);
                        rem -= 254;
                    }
                    result.push(rem as u8);
                }

                // Sizes for frames 0..count-2
                for frame_size in frame_sizes.iter().take(count.saturating_sub(1)) {
                    encode_size(*frame_size, &mut result);
                }
                // Self-delimited: encode last frame size
                encode_size(last_frame_size, &mut result);

                // All frame data
                result.extend_from_slice(&packet[data_start..data_start + total_data]);

                // Padding data (zeros)
                result.extend(std::iter::repeat_n(0u8, padding_bytes));
            } else {
                // CBR: all frames have the same size
                let data_start = pos;
                let total_data = packet.len() - data_start - padding_bytes;
                if count == 0 {
                    return Err(OpusError::InvalidPacket);
                }
                let frame_size = total_data / count;
                if frame_size * count != total_data {
                    return Err(OpusError::InvalidPacket);
                }

                // Build self-delimited output
                result.push(toc);
                result.push(ch);

                // Re-encode padding if present
                if has_padding {
                    let mut rem = padding_bytes;
                    while rem >= 254 {
                        result.push(255);
                        rem -= 254;
                    }
                    result.push(rem as u8);
                }

                // Self-delimited: encode the frame size
                encode_size(frame_size, &mut result);

                // All frame data
                result.extend_from_slice(&packet[data_start..data_start + total_data]);

                // Padding data (zeros)
                result.extend(std::iter::repeat_n(0u8, padding_bytes));
            }
        }
        _ => unreachable!(),
    }

    Ok(result)
}

/// Encode a frame size in variable-length format (same as in repacketizer).
fn encode_size(size: usize, out: &mut Vec<u8>) {
    if size < 252 {
        out.push(size as u8);
    } else {
        out.push(252 + (size & 0x3) as u8);
        out.push(((size - (252 + (size & 0x3))) >> 2) as u8);
    }
}

/// Parse a variable-length size from a byte slice.
/// Returns (size, bytes_consumed).
fn parse_size_from_slice(data: &[u8]) -> Result<(i16, usize), OpusError> {
    if data.is_empty() {
        return Err(OpusError::InvalidPacket);
    }
    if data[0] < 252 {
        Ok((data[0] as i16, 1))
    } else if data.len() < 2 {
        Err(OpusError::InvalidPacket)
    } else {
        Ok((4 * data[1] as i16 + data[0] as i16, 2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multistream::OpusMSDecoder;

    #[test]
    fn test_ms_encoder_create_mono() {
        // Mono: 1 channel, 1 stream, 0 coupled
        let enc = OpusMSEncoder::new(SampleRate::Hz48000, 1, 1, 0, &[0], Application::Audio);
        assert!(enc.is_ok(), "Mono encoder creation failed: {:?}", enc.err());
        let enc = enc.unwrap();
        assert_eq!(enc.channels(), 1);
        assert_eq!(enc.streams(), 1);
        assert_eq!(enc.coupled_streams(), 0);
    }

    #[test]
    fn test_ms_encoder_create_stereo() {
        // Stereo: 2 channels, 1 coupled stream
        let enc = OpusMSEncoder::new(SampleRate::Hz48000, 2, 1, 1, &[0, 1], Application::Audio);
        assert!(
            enc.is_ok(),
            "Stereo encoder creation failed: {:?}",
            enc.err()
        );
        let enc = enc.unwrap();
        assert_eq!(enc.channels(), 2);
        assert_eq!(enc.streams(), 1);
        assert_eq!(enc.coupled_streams(), 1);
    }

    #[test]
    fn test_ms_encoder_create_51() {
        // 5.1 surround: 6 channels, 4 streams, 2 coupled
        // mapping: FL=0, FR=1, FC=4, RL=2, RR=3, LFE=5
        let mapping = [0, 1, 2, 3, 4, 5];
        let enc = OpusMSEncoder::new(SampleRate::Hz48000, 6, 4, 2, &mapping, Application::Audio);
        assert!(enc.is_ok(), "5.1 encoder creation failed: {:?}", enc.err());
        let enc = enc.unwrap();
        assert_eq!(enc.channels(), 6);
        assert_eq!(enc.streams(), 4);
        assert_eq!(enc.coupled_streams(), 2);
    }

    #[test]
    fn test_ms_encoder_invalid_args() {
        // 0 channels
        assert!(OpusMSEncoder::new(SampleRate::Hz48000, 0, 1, 0, &[], Application::Audio).is_err());
        // 0 streams
        assert!(
            OpusMSEncoder::new(SampleRate::Hz48000, 1, 0, 0, &[0], Application::Audio).is_err()
        );
        // coupled > streams
        assert!(
            OpusMSEncoder::new(SampleRate::Hz48000, 2, 1, 2, &[0, 1], Application::Audio).is_err()
        );
    }

    #[test]
    fn test_ms_encode_stereo() {
        // Encode stereo with a multistream encoder, decode with multistream decoder
        let fs = 48000;
        let frame_size = 960; // 20ms

        let mut enc =
            OpusMSEncoder::new(SampleRate::Hz48000, 2, 1, 1, &[0, 1], Application::Audio).unwrap();
        enc.set_bitrate(64000);

        // Generate a stereo sine tone
        let mut pcm_in = vec![0.0f32; frame_size * 2];
        for i in 0..frame_size {
            let s = 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / fs as f32).sin();
            pcm_in[i * 2] = s;
            pcm_in[i * 2 + 1] = s;
        }

        let mut packet = vec![0u8; 4000];
        let nbytes = enc
            .encode_float(&pcm_in, frame_size as i32, &mut packet, 4000)
            .expect("Multistream stereo encode should succeed");
        assert!(
            nbytes >= 2,
            "Should produce a non-trivial packet, got {} bytes",
            nbytes
        );

        // Decode with OpusMSDecoder
        let mut dec = OpusMSDecoder::new(SampleRate::Hz48000, 2, 1, 1, &[0, 1]).unwrap();
        let mut pcm_out = vec![0.0f32; frame_size * 2];
        let decoded = dec
            .decode_float(
                Some(&packet[..nbytes as usize]),
                &mut pcm_out,
                frame_size as i32,
            )
            .expect("Multistream stereo decode should succeed");
        assert_eq!(decoded, frame_size as i32);

        // Verify decoded signal has energy
        let energy: f64 = pcm_out.iter().map(|&x| x as f64 * x as f64).sum();
        assert!(
            energy > 0.0,
            "Decoded stereo signal should have non-zero energy"
        );
    }

    #[test]
    fn test_ms_encode_mono() {
        let fs = 48000;
        let frame_size = 960;

        let mut enc =
            OpusMSEncoder::new(SampleRate::Hz48000, 1, 1, 0, &[0], Application::Audio).unwrap();
        enc.set_bitrate(32000);

        let mut pcm_in = vec![0.0f32; frame_size];
        for (i, sample) in pcm_in.iter_mut().enumerate() {
            *sample = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / fs as f32).sin();
        }

        let mut packet = vec![0u8; 4000];
        let nbytes = enc
            .encode_float(&pcm_in, frame_size as i32, &mut packet, 4000)
            .expect("Multistream mono encode should succeed");
        assert!(nbytes >= 2);

        // Decode
        let mut dec = OpusMSDecoder::new(SampleRate::Hz48000, 1, 1, 0, &[0]).unwrap();
        let mut pcm_out = vec![0.0f32; frame_size];
        let decoded = dec
            .decode_float(
                Some(&packet[..nbytes as usize]),
                &mut pcm_out,
                frame_size as i32,
            )
            .expect("Multistream mono decode should succeed");
        assert_eq!(decoded, frame_size as i32);

        let energy: f64 = pcm_out.iter().map(|&x| x as f64 * x as f64).sum();
        assert!(
            energy > 0.0,
            "Decoded mono signal should have non-zero energy"
        );
    }

    #[test]
    fn test_ms_encode_decode_surround() {
        // 5.1 surround: 6 channels, 4 streams (2 coupled + 2 mono)
        let fs = 48000;
        let frame_size = 960;
        let channels = 6;

        // Simple identity-like mapping:
        // ch0 -> stream 0 left (0), ch1 -> stream 0 right (1),
        // ch2 -> stream 1 left (2), ch3 -> stream 1 right (3),
        // ch4 -> stream 2 mono (4), ch5 -> stream 3 mono (5)
        let mapping = [0u8, 1, 2, 3, 4, 5];

        let mut enc = OpusMSEncoder::new(
            SampleRate::Hz48000,
            channels,
            4,
            2,
            &mapping,
            Application::Audio,
        )
        .unwrap();
        enc.set_bitrate(256000);

        // Generate 6-channel PCM with different tones per channel
        let mut pcm_in = vec![0.0f32; frame_size * channels];
        let freqs = [440.0, 440.0, 330.0, 330.0, 220.0, 110.0];
        for i in 0..frame_size {
            for ch in 0..channels {
                let s = 0.3 * (2.0 * std::f32::consts::PI * freqs[ch] * i as f32 / fs as f32).sin();
                pcm_in[i * channels + ch] = s;
            }
        }

        let mut packet = vec![0u8; 8000];
        let nbytes = enc
            .encode_float(&pcm_in, frame_size as i32, &mut packet, 8000)
            .expect("5.1 surround encode should succeed");
        assert!(
            nbytes >= 4,
            "Should produce a multi-stream packet, got {} bytes",
            nbytes
        );

        // Decode with matching decoder
        let mut dec = OpusMSDecoder::new(SampleRate::Hz48000, channels, 4, 2, &mapping).unwrap();
        let mut pcm_out = vec![0.0f32; frame_size * channels];
        let decoded = dec
            .decode_float(
                Some(&packet[..nbytes as usize]),
                &mut pcm_out,
                frame_size as i32,
            )
            .expect("5.1 surround decode should succeed");
        assert_eq!(decoded, frame_size as i32);

        // Verify each channel has energy
        for ch in 0..channels {
            let ch_energy: f64 = (0..frame_size)
                .map(|i| {
                    let x = pcm_out[i * channels + ch] as f64;
                    x * x
                })
                .sum();
            assert!(
                ch_energy > 0.0,
                "Channel {} should have non-zero energy in surround decode",
                ch
            );
        }
    }

    #[test]
    fn test_rate_allocation() {
        // Test that rate allocation produces reasonable values
        let enc = OpusMSEncoder::new(
            SampleRate::Hz48000,
            6,
            4,
            2,
            &[0, 1, 2, 3, 4, 5],
            Application::Audio,
        )
        .unwrap();

        let rates = enc.rate_allocation();
        assert_eq!(rates.len(), 4);

        // All rates should be positive
        for (i, &rate) in rates.iter().enumerate() {
            assert!(
                rate > 0,
                "Stream {} should have positive rate, got {}",
                i,
                rate
            );
        }

        // Coupled streams should get more than mono streams
        assert!(
            rates[0] > rates[2],
            "Coupled stream rate ({}) should exceed mono stream rate ({})",
            rates[0],
            rates[2]
        );
    }

    #[test]
    fn test_rate_allocation_with_lfe() {
        let mut enc = OpusMSEncoder::new(
            SampleRate::Hz48000,
            6,
            4,
            2,
            &[0, 1, 2, 3, 4, 5],
            Application::Audio,
        )
        .unwrap();
        enc.lfe_stream = 3;
        enc.set_bitrate(256000);

        let rates = enc.rate_allocation();
        assert_eq!(rates.len(), 4);

        // LFE stream should get a low fixed rate
        assert_eq!(rates[3], 3500, "LFE stream should get fixed low rate");
    }

    #[test]
    fn test_make_self_delimited_code0() {
        // Code 0 single-frame packet: TOC + 10 bytes payload
        let mut pkt = vec![0x00u8]; // TOC: code 0
        pkt.extend_from_slice(&[0xAA; 10]);

        let sd = make_self_delimited(&pkt).unwrap();
        // Should be: TOC + size(10) + 10 bytes
        assert_eq!(sd[0], 0x00); // TOC preserved
        assert_eq!(sd[1], 10); // size byte
        assert_eq!(sd.len(), 1 + 1 + 10); // TOC + size + payload

        // Verify it can be parsed by self-delimited parser
        let parsed = crate::packet::opus_packet_parse_self_delimited(&sd);
        assert!(
            parsed.is_ok(),
            "Self-delimited packet should parse: {:?}",
            parsed.err()
        );
        let parsed = parsed.unwrap();
        assert_eq!(parsed.frame_sizes.len(), 1);
        assert_eq!(parsed.frame_sizes[0], 10);
    }

    #[test]
    fn test_make_self_delimited_code1() {
        // Code 1: two CBR frames, TOC + 20 bytes (10+10)
        let mut pkt = vec![0x01u8]; // TOC: code 1
        pkt.extend_from_slice(&[0xBB; 20]);

        let sd = make_self_delimited(&pkt).unwrap();
        assert_eq!(sd[0], 0x01);
        assert_eq!(sd[1], 10); // each frame is 10 bytes
        assert_eq!(sd.len(), 1 + 1 + 20);

        let parsed = crate::packet::opus_packet_parse_self_delimited(&sd);
        assert!(
            parsed.is_ok(),
            "Self-delimited code 1 should parse: {:?}",
            parsed.err()
        );
        let parsed = parsed.unwrap();
        assert_eq!(parsed.frame_sizes.len(), 2);
        assert_eq!(parsed.frame_sizes[0], 10);
        assert_eq!(parsed.frame_sizes[1], 10);
    }

    #[test]
    fn test_encode_i16_multistream() {
        let frame_size = 960;
        let mut enc =
            OpusMSEncoder::new(SampleRate::Hz48000, 1, 1, 0, &[0], Application::Audio).unwrap();
        enc.set_bitrate(32000);

        let pcm = vec![0i16; frame_size];
        let mut data = vec![0u8; 4000];

        let result = enc.encode(&pcm, frame_size as i32, &mut data, 4000);
        assert!(
            result.is_ok(),
            "i16 multistream encode should succeed: {:?}",
            result.err()
        );
        assert!(result.unwrap() >= 1);
    }
}
