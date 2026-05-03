#![cfg(any(feature = "dnn-dred", feature = "dnn-osce", feature = "dnn-deep-plc"))]

//! Test loading weight blobs from the model-data directory.

use opus_wave::dnn::nnet::weights::parse_weights;

fn load_blob(name: &str) -> Option<Vec<opus_wave::dnn::nnet::WeightArray>> {
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let path = manifest.join("model-data/blobs").join(name);
    let data = std::fs::read(&path).ok()?;
    parse_weights(&data)
}

#[test]
fn test_pitchdnn_init() {
    let Some(arrays) = load_blob("pitchdnn.bin") else {
        return;
    };
    opus_wave::dnn::pitchdnn::init_pitchdnn(&arrays).expect("init_pitchdnn failed");
}

#[test]
fn test_fargan_init() {
    let Some(arrays) = load_blob("fargan.bin") else {
        return;
    };
    opus_wave::dnn::fargan::init_fargan(&arrays).expect("init_fargan failed");
}

#[test]
fn test_plcmodel_init() {
    let Some(arrays) = load_blob("plcmodel.bin") else {
        return;
    };
    opus_wave::dnn::lpcnet::plc::init_plcmodel(&arrays).expect("init_plcmodel failed");
}

#[test]
fn test_rdovae_enc_init() {
    let Some(arrays) = load_blob("rdovae_enc.bin") else {
        return;
    };
    opus_wave::dnn::dred::rdovae_enc::init_rdovae_enc(&arrays).expect("init_rdovae_enc failed");
}

#[test]
fn test_rdovae_dec_init() {
    let Some(arrays) = load_blob("rdovae_dec.bin") else {
        return;
    };
    opus_wave::dnn::dred::rdovae_dec::init_rdovae_dec(&arrays).expect("init_rdovae_dec failed");
}

#[test]
fn test_lace_init() {
    let Some(arrays) = load_blob("lace.bin") else {
        return;
    };
    opus_wave::dnn::osce::lace::init_lace(&arrays).expect("init_lace failed");
}

#[test]
fn test_nolace_init() {
    let Some(arrays) = load_blob("nolace.bin") else {
        return;
    };
    opus_wave::dnn::osce::nolace::init_nolace(&arrays).expect("init_nolace failed");
}

#[test]
fn test_bbwenet_init() {
    let Some(arrays) = load_blob("bbwenet.bin") else {
        return;
    };
    opus_wave::dnn::osce::bbwenet::init_bbwenet(&arrays).expect("init_bbwenet failed");
}
