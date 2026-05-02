//! Criterion benchmarks for DNN inference primitives at realistic model dimensions.
//!
//! Uses random weights at the actual dimensions from the downloaded model headers.
//! The compute cost is data-independent (matrix multiply timing does not depend on values).

use criterion::{Criterion, criterion_group, criterion_main};
use opus_dnn::nnet::activations::compute_activation;
use opus_dnn::nnet::linear::compute_linear;
use opus_dnn::nnet::ops::{
    compute_generic_conv1d, compute_generic_dense, compute_generic_gru, compute_glu,
};
use opus_dnn::nnet::{Activation, LinearLayer};

/// Simple LCG for reproducible random test data.
struct Rng(u32);
impl Rng {
    fn new(seed: u32) -> Self {
        Rng(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1103515245).wrapping_add(12345);
        ((self.0 >> 16) as f32 / 32768.0) - 1.0
    }
    fn vec_f32(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32()).collect()
    }
}

fn make_layer(rng: &mut Rng, ni: usize, no: usize) -> LinearLayer {
    LinearLayer {
        bias: Some(rng.vec_f32(no)),
        subias: None,
        weights: None,
        float_weights: Some(rng.vec_f32(ni * no)),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs: ni,
        nb_outputs: no,
    }
}

fn make_gru_layers(rng: &mut Rng, ni: usize, nn: usize) -> (LinearLayer, LinearLayer) {
    let n3 = 3 * nn;
    let input_layer = make_layer(rng, ni, n3);
    let recur_layer = LinearLayer {
        bias: Some(rng.vec_f32(n3)),
        subias: None,
        weights: None,
        float_weights: Some(rng.vec_f32(nn * n3)),
        weights_idx: None,
        diag: Some(rng.vec_f32(n3)),
        scale: None,
        nb_inputs: nn,
        nb_outputs: n3,
    };
    (input_layer, recur_layer)
}

// ============ Activation benchmarks ============

fn bench_activations(c: &mut Criterion) {
    let mut group = c.benchmark_group("activations");
    let input: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();

    for (name, act) in [
        ("sigmoid_512", Activation::Sigmoid),
        ("tanh_512", Activation::Tanh),
        ("relu_512", Activation::Relu),
        ("swish_512", Activation::Swish),
    ] {
        group.bench_function(name, |b| {
            let mut buf = input.clone();
            b.iter(|| {
                buf.copy_from_slice(&input);
                compute_activation(&mut buf, act);
            });
        });
    }
    group.finish();
}

// ============ Linear layer benchmarks at model dimensions ============

fn bench_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("linear");
    let mut rng = Rng::new(42);

    // PitchDNN: dense_if_upsampler_1 (88 → 64)
    let layer_88x64 = make_layer(&mut rng, 88, 64);
    let input_88 = rng.vec_f32(88);
    group.bench_function("dense_88x64_pitchdnn", |b| {
        let mut out = vec![0.0f32; 64];
        b.iter(|| compute_linear(&layer_88x64, &mut out, &input_88));
    });

    // FARGAN: cond_net_fdense2 (128 → 320)
    let layer_128x320 = make_layer(&mut rng, 128, 320);
    let input_128 = rng.vec_f32(128);
    group.bench_function("dense_128x320_fargan_cond", |b| {
        let mut out = vec![0.0f32; 320];
        b.iter(|| compute_linear(&layer_128x320, &mut out, &input_128));
    });

    // LACE/NoLACE: fnet_tconv (128/160 → 512/640)
    let layer_160x640 = make_layer(&mut rng, 160, 640);
    let input_160 = rng.vec_f32(160);
    group.bench_function("dense_160x640_osce_tconv", |b| {
        let mut out = vec![0.0f32; 640];
        b.iter(|| compute_linear(&layer_160x640, &mut out, &input_160));
    });

    group.finish();
}

// ============ GRU benchmarks at model dimensions ============

fn bench_gru(c: &mut Criterion) {
    let mut group = c.benchmark_group("gru");
    let mut rng = Rng::new(123);

    // PitchDNN GRU: 64 inputs → 64 neurons
    let (inp64, rec64) = make_gru_layers(&mut rng, 64, 64);
    let input_64 = rng.vec_f32(64);
    group.bench_function("gru_64x64_pitchdnn", |b| {
        let mut state = vec![0.0f32; 64];
        b.iter(|| compute_generic_gru(&inp64, &rec64, &mut state, &input_64));
    });

    // FARGAN: sig_net_gru1 (192+80 → 160 neurons)
    let ni_fargan = 192 + 80;
    let (inp_fg, rec_fg) = make_gru_layers(&mut rng, ni_fargan, 160);
    let input_fg = rng.vec_f32(ni_fargan);
    group.bench_function("gru_272x160_fargan", |b| {
        let mut state = vec![0.0f32; 160];
        b.iter(|| compute_generic_gru(&inp_fg, &rec_fg, &mut state, &input_fg));
    });

    // NoLACE: fnet_gru (160 → 160 neurons)
    let (inp_nl, rec_nl) = make_gru_layers(&mut rng, 160, 160);
    let input_nl = rng.vec_f32(160);
    group.bench_function("gru_160x160_nolace", |b| {
        let mut state = vec![0.0f32; 160];
        b.iter(|| compute_generic_gru(&inp_nl, &rec_nl, &mut state, &input_nl));
    });

    group.finish();
}

// ============ GLU benchmarks ============

fn bench_glu(c: &mut Criterion) {
    let mut group = c.benchmark_group("glu");
    let mut rng = Rng::new(456);

    // FARGAN: sig_net_fwc0_glu_gate (192 → 192)
    let layer = make_layer(&mut rng, 192, 192);
    let input = rng.vec_f32(192);
    group.bench_function("glu_192_fargan", |b| {
        let mut out = vec![0.0f32; 192];
        b.iter(|| compute_glu(&layer, &mut out, &input));
    });

    group.finish();
}

// ============ Conv1D benchmarks ============

fn bench_conv1d(c: &mut Criterion) {
    let mut group = c.benchmark_group("conv1d");
    let mut rng = Rng::new(789);

    // LACE: fnet_conv2 (384*2 → 128, kernel=2 over 384-dim input frames)
    let input_size = 384;
    let kernel_size = 2;
    let nb_inputs = input_size * kernel_size;
    let nb_outputs = 128;
    let conv_layer = LinearLayer {
        bias: Some(rng.vec_f32(nb_outputs)),
        subias: None,
        weights: None,
        float_weights: Some(rng.vec_f32(nb_inputs * nb_outputs)),
        weights_idx: None,
        diag: None,
        scale: None,
        nb_inputs,
        nb_outputs,
    };
    let input_384 = rng.vec_f32(input_size);
    let mem_size = nb_inputs - input_size; // (kernel-1) * input_size
    let mut mem = vec![0.0f32; mem_size];

    group.bench_function("conv1d_384x128_lace", |b| {
        let mut out = vec![0.0f32; nb_outputs];
        b.iter(|| {
            compute_generic_conv1d(
                &conv_layer,
                &mut out,
                &mut mem,
                &input_384,
                input_size,
                Activation::Tanh,
            );
        });
    });

    group.finish();
}

// ============ Dense + activation (full layer) benchmarks ============

fn bench_dense_with_activation(c: &mut Criterion) {
    let mut group = c.benchmark_group("dense_act");
    let mut rng = Rng::new(321);

    // RDOVAE encoder: enc_dense1 (72 → varies, tanh)
    let layer_72x128 = make_layer(&mut rng, 72, 128);
    let input_72 = rng.vec_f32(72);
    group.bench_function("dense_72x128_tanh_rdovae", |b| {
        let mut out = vec![0.0f32; 128];
        b.iter(|| compute_generic_dense(&layer_72x128, &mut out, &input_72, Activation::Tanh));
    });

    // BBWENet: fnet_conv1 output dim (114 → 128, tanh)
    let layer_114x128 = make_layer(&mut rng, 114, 128);
    let input_114 = rng.vec_f32(114);
    group.bench_function("dense_114x128_tanh_bbwenet", |b| {
        let mut out = vec![0.0f32; 128];
        b.iter(|| compute_generic_dense(&layer_114x128, &mut out, &input_114, Activation::Tanh));
    });

    group.finish();
}

// ============ Model loading + inference with actual weights ============

fn blobs_dir() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("model-data/blobs")
}

fn load_blob(name: &str) -> Option<Vec<opus_dnn::nnet::WeightArray>> {
    let path = blobs_dir().join(name);
    let data = std::fs::read(&path).ok()?;
    opus_dnn::nnet::weights::parse_weights(&data)
}

fn bench_model_weights(c: &mut Criterion) {
    let mut group = c.benchmark_group("model_weights");

    // PitchDNN: load weights + init + inference
    if let Some(arrays) = load_blob("pitchdnn.bin")
        && let Ok(model) = opus_dnn::pitchdnn::init_pitchdnn(&arrays)
    {
        let gru_n = model.gru_1_recurrent.nb_inputs;
        let mut state = opus_dnn::pitchdnn::PitchDnnState {
            model,
            gru_state: vec![0.0f32; gru_n],
            xcorr_mem1: vec![0.0f32; (opus_dnn::pitchdnn::NB_XCORR_FEATURES + 2) * 16],
            xcorr_mem2: vec![0.0f32; (opus_dnn::pitchdnn::NB_XCORR_FEATURES + 2) * 16],
        };
        let mut rng = Rng::new(42);
        let if_features = rng.vec_f32(opus_dnn::pitchdnn::NB_IF_FEATURES);
        let xcorr_features = rng.vec_f32(opus_dnn::pitchdnn::NB_XCORR_FEATURES);
        group.bench_function("pitchdnn_compute", |b| {
            b.iter(|| {
                opus_dnn::pitchdnn::compute_pitchdnn(&mut state, &if_features, &xcorr_features)
            });
        });
    }

    // FARGAN: linear layers with actual int8 weights
    if let Some(arrays) = load_blob("fargan.bin")
        && let Ok(model) = opus_dnn::fargan::init_fargan(&arrays)
    {
        let mut rng = Rng::new(99);
        // Benchmark the conditioning network dense layer
        let cond_in = model.cond_net_fconv1.nb_inputs;
        let cond_out = model.cond_net_fconv1.nb_outputs;
        let input = rng.vec_f32(cond_in);
        group.bench_function("fargan_cond_fconv1_int8", |b| {
            let mut out = vec![0.0f32; cond_out];
            b.iter(|| compute_linear(&model.cond_net_fconv1, &mut out, &input));
        });
    }

    // LACE: load + init
    if let Some(arrays) = load_blob("lace.bin") {
        group.bench_function("lace_init", |b| {
            b.iter(|| opus_dnn::osce::lace::init_lace(&arrays).unwrap());
        });
    }

    // Weight parsing benchmark (measures deserialization throughput)
    let pitchdnn_path = blobs_dir().join("pitchdnn.bin");
    if let Ok(data) = std::fs::read(&pitchdnn_path) {
        group.bench_function("parse_weights_pitchdnn", |b| {
            b.iter(|| opus_dnn::nnet::weights::parse_weights(&data).unwrap());
        });
    }

    let fargan_path = blobs_dir().join("fargan.bin");
    if let Ok(data) = std::fs::read(&fargan_path) {
        group.bench_function("parse_weights_fargan", |b| {
            b.iter(|| opus_dnn::nnet::weights::parse_weights(&data).unwrap());
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_activations,
    bench_linear,
    bench_gru,
    bench_glu,
    bench_conv1d,
    bench_dense_with_activation,
    bench_model_weights,
);
criterion_main!(benches);
