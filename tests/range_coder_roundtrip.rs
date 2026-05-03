use opus_wave::range_coder::EcCtx;

/// Helper: encode, finalize, and return the full buffer for decoding.
/// After enc_done(), the buffer contains range-coded bytes at the front
/// and raw bits at the end. The decoder needs the full buffer.
fn finalize_and_get_buf(enc: &mut EcCtx) -> Vec<u8> {
    enc.enc_done();
    enc.buf[..enc.storage as usize].to_vec()
}

#[test]
fn test_basic_encode_decode_roundtrip() {
    let mut enc = EcCtx::enc_init(1000);
    enc.encode(3, 4, 10);
    enc.encode(0, 1, 5);
    enc.encode(4, 5, 5);
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);

    let s = dec.decode(10);
    assert_eq!(s, 3);
    dec.dec_update(3, 4, 10);

    let s = dec.decode(5);
    assert_eq!(s, 0);
    dec.dec_update(0, 1, 5);

    let s = dec.decode(5);
    assert_eq!(s, 4);
    dec.dec_update(4, 5, 5);
}

#[test]
fn test_bit_logp_roundtrip() {
    let mut enc = EcCtx::enc_init(1000);
    let bits = [true, false, true, true, false, false, true, false];
    for &b in &bits {
        enc.enc_bit_logp(b, 2);
    }
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);
    for &expected in &bits {
        let decoded = dec.dec_bit_logp(2);
        assert_eq!(decoded, expected);
    }
}

#[test]
fn test_icdf_roundtrip() {
    let icdf: &[u8] = &[128, 64, 0];
    let symbols = [0usize, 1, 2, 0, 2, 1, 0, 0, 1, 2];

    let mut enc = EcCtx::enc_init(1000);
    for &s in &symbols {
        enc.enc_icdf(s, icdf, 8);
    }
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);
    for &expected in &symbols {
        let decoded = dec.dec_icdf(icdf, 8);
        assert_eq!(decoded, expected);
    }
}

#[test]
fn test_uint_roundtrip() {
    let values: &[(u32, u32)] = &[
        (0, 10),
        (9, 10),
        (5, 100),
        (99, 100),
        (0, 1000),
        (999, 1000),
        (12345, 100000),
        (0, 2),
        (1, 2),
    ];

    let mut enc = EcCtx::enc_init(1000);
    for &(val, ft) in values {
        enc.enc_uint(val, ft);
    }
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);
    for &(expected, ft) in values {
        let decoded = dec.dec_uint(ft);
        assert_eq!(decoded, expected, "Failed for value {expected} out of {ft}");
    }
}

#[test]
fn test_bits_roundtrip() {
    let mut enc = EcCtx::enc_init(1000);
    enc.enc_bits(0xAB, 8);
    enc.enc_bits(0x3, 2);
    enc.enc_bits(0x1FFFF, 17);
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);
    assert_eq!(dec.dec_bits(8), 0xAB);
    assert_eq!(dec.dec_bits(2), 0x3);
    assert_eq!(dec.dec_bits(17), 0x1FFFF);
}

#[test]
fn test_mixed_coding_roundtrip() {
    let mut enc = EcCtx::enc_init(1000);
    enc.enc_bit_logp(true, 3);
    enc.enc_uint(42, 100);
    enc.enc_bits(0xFF, 8);
    enc.encode(2, 3, 7);
    enc.enc_bit_logp(false, 1);
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);
    assert!(dec.dec_bit_logp(3));
    assert_eq!(dec.dec_uint(100), 42);
    assert_eq!(dec.dec_bits(8), 0xFF);
    let s = dec.decode(7);
    assert_eq!(s, 2);
    dec.dec_update(2, 3, 7);
    assert!(!dec.dec_bit_logp(1));
}

#[test]
fn test_tell_consistency() {
    let mut enc = EcCtx::enc_init(1000);
    let tell_start = enc.tell();
    enc.enc_bit_logp(false, 1);
    let tell_after = enc.tell();
    assert!(tell_after > tell_start);
    enc.enc_done();
}

#[test]
fn test_encode_bin_roundtrip() {
    let mut enc = EcCtx::enc_init(1000);
    enc.encode_bin(100, 101, 8);
    enc.encode_bin(0, 1, 8);
    enc.encode_bin(255, 256, 8);
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);

    let s = dec.decode_bin(8);
    assert_eq!(s, 100);
    dec.dec_update(100, 101, 256);

    let s = dec.decode_bin(8);
    assert_eq!(s, 0);
    dec.dec_update(0, 1, 256);

    let s = dec.decode_bin(8);
    assert_eq!(s, 255);
    dec.dec_update(255, 256, 256);
}

#[test]
fn test_icdf16_roundtrip() {
    let icdf: &[u16] = &[16384, 8192, 0];
    let symbols = [0usize, 1, 2, 0, 1, 2];

    let mut enc = EcCtx::enc_init(1000);
    for &s in &symbols {
        enc.enc_icdf16(s, icdf, 15);
    }
    let buf = finalize_and_get_buf(&mut enc);
    let mut dec = EcCtx::dec_init(&buf);
    for &expected in &symbols {
        let decoded = dec.dec_icdf16(icdf, 15);
        assert_eq!(decoded, expected);
    }
}
