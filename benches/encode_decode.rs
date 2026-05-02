//! Criterion benchmarks for the Rust Opus encoder and decoder.

mod common;

use common::*;
use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use opus_rust::Bitrate;
use opus_rust::decoder::OpusDecoder;
use opus_rust::encoder::OpusEncoder;

fn bench_rust_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_encode");
    for cfg in &bench_configs() {
        let input_frames = generate_input_frames(cfg);
        group.throughput(Throughput::Elements(FRAMES_PER_ITER as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || {
                    let mut enc =
                        OpusEncoder::new(SAMPLE_RATE, cfg.channels, cfg.application).unwrap();
                    enc.set_bandwidth(cfg.max_bandwidth);
                    enc.set_complexity(cfg.complexity);
                    enc.set_bitrate(Bitrate::BitsPerSecond(cfg.bitrate));
                    (enc, vec![0u8; MAX_PACKET])
                },
                |(mut enc, mut packet)| {
                    for frame in &input_frames {
                        let _ = enc.encode_float(frame, FRAME_SIZE, &mut packet, MAX_PACKET as i32);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

fn bench_rust_decode(c: &mut Criterion) {
    let mut group = c.benchmark_group("rust_decode");
    for cfg in &bench_configs() {
        let packets = pre_encode_with_c(cfg);
        if packets.is_empty() {
            continue;
        }
        group.throughput(Throughput::Elements(packets.len() as u64));
        group.bench_function(cfg.name, |b| {
            b.iter_batched(
                || {
                    let dec = OpusDecoder::new(SAMPLE_RATE, cfg.channels).unwrap();
                    let pcm = vec![0.0f32; FRAME_SIZE as usize * i32::from(cfg.channels) as usize];
                    (dec, pcm)
                },
                |(mut dec, mut pcm)| {
                    for pkt in &packets {
                        let _ = dec.decode_float(Some(pkt), &mut pcm, FRAME_SIZE, false);
                    }
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

criterion_group!(benches, bench_rust_encode, bench_rust_decode);
criterion_main!(benches);
