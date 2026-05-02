//! Shared benchmark configurations and signal generation.

use opus_rust::{Application, Bandwidth, Channels, SampleRate};

pub const SAMPLE_RATE: SampleRate = SampleRate::Hz48000;
pub const FRAME_SIZE: i32 = 960; // 20ms at 48kHz
pub const FRAMES_PER_ITER: usize = 100; // 2 seconds of audio

pub struct BenchConfig {
    pub name: &'static str,
    pub channels: Channels,
    pub application: Application,
    pub max_bandwidth: Bandwidth,
    pub bitrate: i32,
    pub complexity: i32,
    pub freq_l: f32,
    pub freq_r: f32,
    pub amp: f32,
}

pub fn bench_configs() -> Vec<BenchConfig> {
    vec![
        BenchConfig {
            name: "celt_mono_64k",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 64000,
            complexity: 10,
            freq_l: 440.0,
            freq_r: 0.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "celt_mono_128k",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 128000,
            complexity: 10,
            freq_l: 1000.0,
            freq_r: 0.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "celt_stereo_128k",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 128000,
            complexity: 10,
            freq_l: 440.0,
            freq_r: 880.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "silk_mono_nb_12k",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Narrowband,
            bitrate: 12000,
            complexity: 10,
            freq_l: 200.0,
            freq_r: 0.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "silk_mono_wb_20k",
            channels: Channels::Mono,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Wideband,
            bitrate: 20000,
            complexity: 10,
            freq_l: 500.0,
            freq_r: 0.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "silk_stereo_wb_32k",
            channels: Channels::Stereo,
            application: Application::Voip,
            max_bandwidth: Bandwidth::Wideband,
            bitrate: 32000,
            complexity: 10,
            freq_l: 400.0,
            freq_r: 600.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "hybrid_stereo_fb_36k",
            channels: Channels::Stereo,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 36000,
            complexity: 10,
            freq_l: 440.0,
            freq_r: 880.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "celt_mono_64k_c0",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 64000,
            complexity: 0,
            freq_l: 440.0,
            freq_r: 0.0,
            amp: 0.5,
        },
        BenchConfig {
            name: "celt_mono_64k_c5",
            channels: Channels::Mono,
            application: Application::Audio,
            max_bandwidth: Bandwidth::Fullband,
            bitrate: 64000,
            complexity: 5,
            freq_l: 440.0,
            freq_r: 0.0,
            amp: 0.5,
        },
    ]
}

pub const MAX_PACKET: usize = 4000;

/// Pre-encode packets using the C encoder for decode benchmarks.
pub fn pre_encode_with_c(cfg: &BenchConfig) -> Vec<Vec<u8>> {
    use opus_ffi::COpusEncoder;
    let sample_rate = i32::from(SAMPLE_RATE);
    let mut enc = COpusEncoder::new(
        sample_rate,
        i32::from(cfg.channels),
        i32::from(cfg.application),
    )
    .unwrap();
    enc.set_max_bandwidth(i32::from(cfg.max_bandwidth)).unwrap();
    enc.set_complexity(cfg.complexity).unwrap();
    enc.set_bitrate(cfg.bitrate).unwrap();

    let input_frames = generate_input_frames(cfg);
    let mut packets = Vec::with_capacity(input_frames.len());
    for frame in &input_frames {
        let mut packet = vec![0u8; MAX_PACKET];
        match enc.encode_float(frame, FRAME_SIZE, &mut packet) {
            Ok(len) => packets.push(packet[..len as usize].to_vec()),
            Err(_) => return Vec::new(),
        }
    }
    packets
}

/// Generate FRAMES_PER_ITER frames of test signal for a given config.
pub fn generate_input_frames(cfg: &BenchConfig) -> Vec<Vec<f32>> {
    let channels = i32::from(cfg.channels);
    let sample_rate = i32::from(SAMPLE_RATE);
    let samples_per_frame = FRAME_SIZE as usize * channels as usize;
    (0..FRAMES_PER_ITER)
        .map(|frame| {
            let mut buf = vec![0.0f32; samples_per_frame];
            let offset = frame * FRAME_SIZE as usize;
            if channels == 2 {
                for i in 0..FRAME_SIZE as usize {
                    let t = (i + offset) as f32 / sample_rate as f32;
                    buf[i * 2] = cfg.amp * (2.0 * std::f32::consts::PI * cfg.freq_l * t).sin();
                    buf[i * 2 + 1] = cfg.amp * (2.0 * std::f32::consts::PI * cfg.freq_r * t).sin();
                }
            } else {
                for (i, sample) in buf.iter_mut().enumerate().take(FRAME_SIZE as usize) {
                    *sample = cfg.amp
                        * (2.0 * std::f32::consts::PI * cfg.freq_l * (i + offset) as f32
                            / sample_rate as f32)
                            .sin();
                }
            }
            buf
        })
        .collect()
}
