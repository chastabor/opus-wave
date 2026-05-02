//! Compare Rust SILK functions with C reference byte-by-byte.
//! Pinpoints numerical divergence in the LPC→NLSF→encode pipeline.

use opus_ffi::{c_silk_a2nlsf, c_silk_nlsf_encode_wb, c_silk_nlsf_vq_weights_laroia};
use opus_rust::silk::lpc_analysis::{silk_a2nlsf, silk_nlsf_vq_weights_laroia};
use opus_rust::silk::nlsf_encode::silk_nlsf_encode;
use opus_rust::silk::{NlsfCbSel, get_nlsf_cb};

const ORDER: usize = 16;

#[test]
fn a2nlsf_burg_frame0_2tap() {
    let input = [117193i32, -53446, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut a_r = input;
    let mut nlsf_r = [0i16; ORDER];
    silk_a2nlsf(&mut nlsf_r, &mut a_r, ORDER);
    let mut a_c = input;
    let mut nlsf_c = [0i16; ORDER];
    c_silk_a2nlsf(&mut nlsf_c, &mut a_c, ORDER);
    assert_eq!(nlsf_r, nlsf_c, "A2NLSF diverges (frame0)");
}

#[test]
fn a2nlsf_burg_frame1_2tap() {
    let input = [129016i32, -65452, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    let mut a_r = input;
    let mut nlsf_r = [0i16; ORDER];
    silk_a2nlsf(&mut nlsf_r, &mut a_r, ORDER);
    let mut a_c = input;
    let mut nlsf_c = [0i16; ORDER];
    c_silk_a2nlsf(&mut nlsf_c, &mut a_c, ORDER);
    assert_eq!(nlsf_r, nlsf_c, "A2NLSF diverges (frame1)");
}

#[test]
fn a2nlsf_speech_like() {
    let input = [
        65536i32, -32768, 16384, -8192, 4096, -2048, 1024, -512, 256, -128, 64, -32, 16, -8, 4, -2,
    ];
    let mut a_r = input;
    let mut nlsf_r = [0i16; ORDER];
    silk_a2nlsf(&mut nlsf_r, &mut a_r, ORDER);
    let mut a_c = input;
    let mut nlsf_c = [0i16; ORDER];
    c_silk_a2nlsf(&mut nlsf_c, &mut a_c, ORDER);
    assert_eq!(nlsf_r, nlsf_c, "A2NLSF diverges (speech)");
}

#[test]
fn nlsf_encode_frame0_full_pipeline() {
    let a_q16_input = [117193i32, -53446, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];

    let mut a_tmp = a_q16_input;
    let mut nlsf = [0i16; ORDER];
    silk_a2nlsf(&mut nlsf, &mut a_tmp, ORDER);

    let mut w_r = [0i16; ORDER];
    silk_nlsf_vq_weights_laroia(&mut w_r, &nlsf, ORDER);
    let mut w_c = [0i16; ORDER];
    c_silk_nlsf_vq_weights_laroia(&mut w_c, &nlsf, ORDER);
    assert_eq!(w_r, w_c, "Weights diverge");

    let nlsf_cb = get_nlsf_cb(NlsfCbSel::Wb);

    let mut idx_r = [0i8; ORDER + 1];
    let mut nlsf_r = nlsf;
    let rd_r = silk_nlsf_encode(&mut idx_r, &mut nlsf_r, nlsf_cb, &w_r, 2098, 16, 0);

    let mut idx_c = [0i8; ORDER + 1];
    let mut nlsf_c = nlsf;
    let rd_c = c_silk_nlsf_encode_wb(&mut idx_c, &mut nlsf_c, &w_c, 2098, 16, 0);

    assert_eq!(rd_r, rd_c, "NLSF encode RD diverges");
    assert_eq!(idx_r, idx_c, "NLSF encode indices diverge");
    assert_eq!(nlsf_r, nlsf_c, "Quantized NLSFs diverge");
}
