//! Criterion benchmarks for CELT internal hot-path functions.
//! Compares Rust vs C reference performance.

use criterion::{BatchSize, Criterion, Throughput, criterion_group, criterion_main};
use opus_wave::celt::fft::{KissFftCpx, KissFftState, opus_fft};
use opus_wave::celt::lpc;
use opus_wave::celt::mdct::{MdctLookup, clt_mdct_backward, clt_mdct_forward};
use opus_wave::celt::pitch;
use opus_wave::celt::tables::WINDOW_120;
use opus_ffi::*;

fn gen_sine(len: usize, freq: f32, fs: f32, amp: f32) -> Vec<f32> {
    (0..len)
        .map(|i| amp * (2.0 * std::f32::consts::PI * freq * i as f32 / fs).sin())
        .collect()
}

fn gen_noise(len: usize, seed: u32) -> Vec<f32> {
    let mut x = seed;
    (0..len)
        .map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            (x as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

// ── FFT benchmarks ──

fn bench_fft(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_fft");

    for &nfft in &[120, 240, 480] {
        let input: Vec<KissFftCpx> = gen_noise(nfft * 2, 42)
            .chunks(2)
            .map(|c| KissFftCpx { r: c[0], i: c[1] })
            .collect();

        group.throughput(Throughput::Elements(nfft as u64));

        group.bench_function(format!("rust_fft_{nfft}"), |b| {
            b.iter_batched(
                || {
                    (
                        KissFftState::new(nfft),
                        vec![KissFftCpx { r: 0.0, i: 0.0 }; nfft],
                    )
                },
                |(st, mut out)| opus_fft(&st, &input, &mut out),
                BatchSize::SmallInput,
            );
        });

        let fin_r: Vec<f32> = input.iter().map(|c| c.r).collect();
        let fin_i: Vec<f32> = input.iter().map(|c| c.i).collect();

        group.bench_function(format!("c_fft_{nfft}"), |b| {
            c_fft_bench_init(nfft);
            b.iter_batched(
                || (vec![0.0f32; nfft], vec![0.0f32; nfft]),
                |(mut fout_r, mut fout_i)| {
                    c_fft_bench_run(&fin_r, &fin_i, &mut fout_r, &mut fout_i)
                },
                BatchSize::SmallInput,
            );
        });
    }
    group.finish();
}

// ── MDCT benchmarks ──

fn bench_mdct(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_mdct");
    let n = 960;
    let overlap = 120;
    let n2 = n / 2;
    let input_len = n2 + overlap;
    let input = gen_sine(input_len, 440.0, 48000.0, 0.5);

    group.throughput(Throughput::Elements(n as u64));

    group.bench_function("rust_mdct_fwd_960", |b| {
        b.iter_batched(
            || (MdctLookup::new(n, 3), vec![0.0f32; n2]),
            |(mut mdct, mut out)| {
                clt_mdct_forward(&mut mdct, &input, &mut out, &WINDOW_120, overlap, 0, 1);
            },
            BatchSize::SmallInput,
        );
    });

    c_mdct_bench_init(n);

    group.bench_function("c_mdct_fwd_960", |b| {
        b.iter_batched(
            || (input.clone(), vec![0.0f32; n2]),
            |(mut inp, mut out)| c_mdct_bench_forward(&mut inp, &mut out, overlap, 0, 1),
            BatchSize::SmallInput,
        );
    });

    let freq_input = gen_noise(n2, 42);

    group.bench_function("rust_mdct_bwd_960", |b| {
        b.iter_batched(
            || (MdctLookup::new(n, 3), vec![0.0f32; n]),
            |(mut mdct, mut out)| {
                clt_mdct_backward(&mut mdct, &freq_input, &mut out, &WINDOW_120, overlap, 0, 1);
            },
            BatchSize::SmallInput,
        );
    });

    group.bench_function("c_mdct_bwd_960", |b| {
        b.iter_batched(
            || (freq_input.clone(), vec![0.0f32; n]),
            |(mut inp, mut out)| c_mdct_bench_backward(&mut inp, &mut out, overlap, 0, 1),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ── Pitch xcorr benchmarks ──

fn bench_pitch_xcorr(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_pitch_xcorr");
    let len = 240;
    let max_pitch = 512;
    let x = gen_noise(len, 42);
    let y = gen_noise(len + max_pitch, 99);

    group.throughput(Throughput::Elements((len * max_pitch) as u64));

    group.bench_function("rust_xcorr", |b| {
        b.iter_batched(
            || vec![0.0f32; max_pitch],
            |mut xcorr| pitch::celt_pitch_xcorr(&x, &y, &mut xcorr, len, max_pitch),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("c_xcorr", |b| {
        b.iter_batched(
            || vec![0.0f32; max_pitch],
            |mut xcorr| c_celt_pitch_xcorr(&x, &y, &mut xcorr, len, max_pitch),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

// ── LPC benchmarks ──

fn bench_lpc(c: &mut Criterion) {
    let mut group = c.benchmark_group("celt_lpc");
    let signal = gen_sine(960, 440.0, 48000.0, 0.5);
    let order = 24;

    // Pre-compute autocorrelation
    let mut ac = vec![0.0f32; order + 1];
    for k in 0..=order {
        for i in k..signal.len() {
            ac[k] += signal[i] * signal[i - k];
        }
    }

    group.throughput(Throughput::Elements(order as u64));

    group.bench_function("rust_lpc_24", |b| {
        b.iter_batched(
            || vec![0.0f32; order],
            |mut lpc_out| lpc::celt_lpc(&mut lpc_out, &ac, order),
            BatchSize::SmallInput,
        );
    });

    group.bench_function("c_lpc_24", |b| {
        b.iter_batched(
            || vec![0.0f32; order],
            |mut lpc_out| c_celt_lpc(&mut lpc_out, &ac, order),
            BatchSize::SmallInput,
        );
    });

    group.finish();
}

criterion_group!(benches, bench_fft, bench_mdct, bench_pitch_xcorr, bench_lpc);
criterion_main!(benches);
