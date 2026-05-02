/// Complex number for FFT.
#[derive(Clone, Copy, Default)]
pub struct KissFftCpx {
    pub r: f32,
    pub i: f32,
}

/// FFT state containing twiddle factors and bit-reversal table.
pub struct KissFftState {
    pub nfft: usize,
    pub scale: f32,
    pub shift: i32,
    /// Factors stored as [p0, m0, p1, m1, ...] matching C layout.
    /// p_i is the radix, m_i = remaining_n after dividing by p_0..p_i.
    pub factors: Vec<i16>,
    pub bitrev: Vec<usize>,
    pub twiddles: Vec<KissFftCpx>,
}

impl KissFftState {
    /// Create a new FFT state for a given size.
    pub fn new(nfft: usize) -> Self {
        let scale = 1.0 / nfft as f32;
        let twiddles = compute_twiddles(nfft);
        let factors = kf_factor(nfft);
        let bitrev = compute_bitrev_top(nfft, &factors);
        KissFftState {
            nfft,
            scale,
            shift: -1,
            factors,
            bitrev,
            twiddles,
        }
    }
}

fn compute_twiddles(nfft: usize) -> Vec<KissFftCpx> {
    let mut tw = Vec::with_capacity(nfft);
    let pi: f64 = std::f64::consts::PI;
    for i in 0..nfft {
        let phase: f64 = (-2.0 * pi / nfft as f64) * i as f64;
        tw.push(KissFftCpx {
            r: phase.cos() as f32,
            i: phase.sin() as f32,
        });
    }
    tw
}

/// Factor n following the C kf_factor logic exactly.
/// Returns factors as [p0, m0, p1, m1, ...].
/// Matches the C implementation character-by-character.
fn kf_factor(n: usize) -> Vec<i16> {
    let nbak = n;
    let mut n = n;
    let mut p: usize = 4;
    let mut stages: usize = 0;
    // facbuf stores [p0, _, p1, _, ...] initially (m values computed later)
    let mut facbuf = vec![0i16; 64];

    loop {
        while !n.is_multiple_of(p) {
            match p {
                4 => p = 2,
                2 => p = 3,
                _ => p += 2,
            }
            if p > 32000 || (p as u64) * (p as u64) > n as u64 {
                p = n;
            }
        }
        n /= p;
        facbuf[2 * stages] = p as i16;
        if p == 2 && stages > 1 {
            facbuf[2 * stages] = 4;
            facbuf[2] = 2;
        }
        stages += 1;
        if n <= 1 {
            break;
        }
    }
    n = nbak;
    // Reverse the order to get the radix 4 at the end
    for i in 0..(stages / 2) {
        facbuf.swap(2 * i, 2 * (stages - i - 1));
    }
    // Compute m values
    for i in 0..stages {
        n /= facbuf[2 * i] as usize;
        facbuf[2 * i + 1] = n as i16;
    }
    facbuf.truncate(2 * stages);
    facbuf
}

/// Compute the bit-reversal permutation table.
/// Matches the C compute_bitrev_table recursive algorithm exactly.
/// bitrev[i] gives the output index for input position i.
fn compute_bitrev_top(nfft: usize, factors: &[i16]) -> Vec<usize> {
    let mut bitrev = vec![0usize; nfft];
    // f_offset = 0 (start of bitrev array)
    // fstride = 1
    // in_stride = 1
    compute_bitrev_recursive(0, &mut bitrev, 0, 1, 1, factors);
    bitrev
}

/// Recursive bitrev computation matching C exactly.
/// `fout`: the Fout value (output DFT index base)
/// `f`: the bitrev array
/// `f_offset`: current write offset into the bitrev array
/// `fstride`: current stride for writing into bitrev array
/// `in_stride`: always 1 for our usage
/// `factors`: remaining [p, m, ...] pairs
fn compute_bitrev_recursive(
    fout: usize,
    f: &mut [usize],
    f_offset: usize,
    fstride: usize,
    in_stride: usize,
    factors: &[i16],
) {
    let p = factors[0] as usize;
    let m = factors[1] as usize;

    if m == 1 {
        for j in 0..p {
            let idx = f_offset + j * fstride * in_stride;
            if idx < f.len() {
                f[idx] = fout + j;
            }
        }
    } else {
        let mut cur_offset = f_offset;
        for j in 0..p {
            compute_bitrev_recursive(
                fout + j * m,
                f,
                cur_offset,
                fstride * p,
                in_stride,
                &factors[2..],
            );
            cur_offset += fstride * in_stride;
        }
    }
}

/// Perform out-of-place complex FFT with bit-reversal and scaling.
/// For float: fout[bitrev[i]] = fin[i] * scale, then in-place butterflies.
pub fn opus_fft(st: &KissFftState, fin: &[KissFftCpx], fout: &mut [KissFftCpx]) {
    let n = st.nfft;
    assert!(fin.len() >= n);
    assert!(fout.len() >= n);
    // Bit-reverse copy with scaling
    for (i, item) in fin.iter().enumerate().take(n) {
        let rev = st.bitrev[i];
        fout[rev].r = item.r * st.scale;
        fout[rev].i = item.i * st.scale;
    }
    opus_fft_impl(st, fout);
}

/// In-place FFT implementation (no bit-reversal, no scaling for float).
/// Data must already be in bit-reversed order.
/// Processes stages from last to first (matching C opus_fft_impl).
pub fn opus_fft_impl(st: &KissFftState, fout: &mut [KissFftCpx]) {
    let n = st.nfft;
    let num_stages = st.factors.len() / 2;

    // Compute fstride array: fstride[0]=1, fstride[i+1]=fstride[i]*p[i]
    assert!(num_stages <= 8, "FFT requires at most 8 stages");
    let mut fstride_arr = [1usize; 9]; // max 8 stages + 1
    for i in 0..num_stages {
        let p = st.factors[2 * i] as usize;
        fstride_arr[i + 1] = fstride_arr[i] * p;
    }

    // Process stages from last to first
    // m starts at factors[2*(L-1)+1] (should be 1 for the last stage)
    // and grows: m = m2 * p
    let mut m = 1usize; // current sub-transform size
    for i in (0..num_stages).rev() {
        let p = st.factors[2 * i] as usize;
        let fstride_i = fstride_arr[i];
        // m is the sub-DFT size at this point
        // After this stage, sub-DFT size becomes m * p

        match p {
            2 => kf_bfly2(fout, n, m, fstride_i, &st.twiddles),
            3 => kf_bfly3(fout, n, m, fstride_i, &st.twiddles),
            4 => kf_bfly4(fout, n, m, fstride_i, &st.twiddles),
            5 => kf_bfly5(fout, n, m, fstride_i, &st.twiddles),
            _ => {}
        }
        m *= p;
    }
}

/// Radix-2 butterfly.
/// m = sub-DFT size (distance between butterfly pairs)
/// N_groups = n / (2*m)
fn kf_bfly2(fout: &mut [KissFftCpx], n: usize, m: usize, fstride: usize, tw: &[KissFftCpx]) {
    if m == 1 {
        // Degenerate case: all twiddles are 1 (m==1 means first radix-2 stage)
        let n_groups = n / 2;
        for i in 0..n_groups {
            let idx = i * 2;
            let t = fout[idx + 1];
            fout[idx + 1] = KissFftCpx {
                r: fout[idx].r - t.r,
                i: fout[idx].i - t.i,
            };
            fout[idx].r += t.r;
            fout[idx].i += t.i;
        }
    } else {
        // General radix-2: the C code has a special case for m==4 (after radix-4).
        // For float, we can handle it generically.
        let group_size = 2 * m;
        let n_groups = n / group_size;
        for i in 0..n_groups {
            let base = i * group_size;
            let mut tw_idx = 0;
            for j in 0..m {
                let idx0 = base + j;
                let idx1 = base + j + m;
                let t = cmul(fout[idx1], tw[tw_idx]);
                fout[idx1] = KissFftCpx {
                    r: fout[idx0].r - t.r,
                    i: fout[idx0].i - t.i,
                };
                fout[idx0].r += t.r;
                fout[idx0].i += t.i;
                tw_idx += fstride;
                if tw_idx >= n {
                    tw_idx -= n;
                }
            }
        }
    }
}

/// Radix-4 butterfly.
fn kf_bfly4(fout: &mut [KissFftCpx], n: usize, m: usize, fstride: usize, tw: &[KissFftCpx]) {
    let group_size = 4 * m;
    let n_groups = n / group_size;

    if m == 1 {
        // Degenerate case: all twiddles are 1
        for i in 0..n_groups {
            let base = i * 4;
            let a0 = fout[base];
            let a1 = fout[base + 1];
            let a2 = fout[base + 2];
            let a3 = fout[base + 3];

            let scratch0 = KissFftCpx {
                r: a0.r - a2.r,
                i: a0.i - a2.i,
            };
            let sum02 = KissFftCpx {
                r: a0.r + a2.r,
                i: a0.i + a2.i,
            };
            let scratch1_add = KissFftCpx {
                r: a1.r + a3.r,
                i: a1.i + a3.i,
            };
            let scratch1_sub = KissFftCpx {
                r: a1.r - a3.r,
                i: a1.i - a3.i,
            };

            fout[base + 2] = KissFftCpx {
                r: sum02.r - scratch1_add.r,
                i: sum02.i - scratch1_add.i,
            };
            fout[base] = KissFftCpx {
                r: sum02.r + scratch1_add.r,
                i: sum02.i + scratch1_add.i,
            };
            // Forward FFT sign convention
            fout[base + 1] = KissFftCpx {
                r: scratch0.r + scratch1_sub.i,
                i: scratch0.i - scratch1_sub.r,
            };
            fout[base + 3] = KissFftCpx {
                r: scratch0.r - scratch1_sub.i,
                i: scratch0.i + scratch1_sub.r,
            };
        }
    } else {
        for i in 0..n_groups {
            let base = i * group_size;
            let (mut tw1_idx, mut tw2_idx, mut tw3_idx) = (0, 0, 0);
            for j in 0..m {
                let a0 = fout[base + j];
                let a1 = cmul(fout[base + j + m], tw[tw1_idx]);
                let a2 = cmul(fout[base + j + 2 * m], tw[tw2_idx]);
                let a3 = cmul(fout[base + j + 3 * m], tw[tw3_idx]);

                let scratch5 = KissFftCpx {
                    r: a0.r - a2.r,
                    i: a0.i - a2.i,
                };
                let sum02 = KissFftCpx {
                    r: a0.r + a2.r,
                    i: a0.i + a2.i,
                };
                let scratch3 = KissFftCpx {
                    r: a1.r + a3.r,
                    i: a1.i + a3.i,
                };
                let scratch4 = KissFftCpx {
                    r: a1.r - a3.r,
                    i: a1.i - a3.i,
                };

                fout[base + j + 2 * m] = KissFftCpx {
                    r: sum02.r - scratch3.r,
                    i: sum02.i - scratch3.i,
                };
                fout[base + j] = KissFftCpx {
                    r: sum02.r + scratch3.r,
                    i: sum02.i + scratch3.i,
                };
                fout[base + j + m] = KissFftCpx {
                    r: scratch5.r + scratch4.i,
                    i: scratch5.i - scratch4.r,
                };
                fout[base + j + 3 * m] = KissFftCpx {
                    r: scratch5.r - scratch4.i,
                    i: scratch5.i + scratch4.r,
                };
                tw1_idx += fstride;
                if tw1_idx >= n {
                    tw1_idx -= n;
                }
                tw2_idx += 2 * fstride;
                if tw2_idx >= n {
                    tw2_idx -= n;
                }
                tw3_idx += 3 * fstride;
                if tw3_idx >= n {
                    tw3_idx -= n;
                }
            }
        }
    }
}

/// Radix-3 butterfly.
fn kf_bfly3(fout: &mut [KissFftCpx], n: usize, m: usize, fstride: usize, tw: &[KissFftCpx]) {
    let group_size = 3 * m;
    let n_groups = n / group_size;
    let epi3_i = -0.866_025_4_f32; // sin(-2pi/3)

    for i in 0..n_groups {
        let base = i * group_size;
        let (mut tw1_idx, mut tw2_idx) = (0, 0);
        for j in 0..m {
            let a0 = fout[base + j];
            let a1 = cmul(fout[base + j + m], tw[tw1_idx]);
            let a2 = cmul(fout[base + j + 2 * m], tw[tw2_idx]);

            let scratch1_r = a1.r + a2.r;
            let scratch1_i = a1.i + a2.i;

            fout[base + j] = KissFftCpx {
                r: a0.r + scratch1_r,
                i: a0.i + scratch1_i,
            };

            let m_r = a0.r - 0.5 * scratch1_r;
            let m_i = a0.i - 0.5 * scratch1_i;

            let s_r = epi3_i * (a1.i - a2.i);
            let s_i = epi3_i * (a2.r - a1.r);

            fout[base + j + m] = KissFftCpx {
                r: m_r - s_r,
                i: m_i - s_i,
            };
            fout[base + j + 2 * m] = KissFftCpx {
                r: m_r + s_r,
                i: m_i + s_i,
            };

            tw1_idx += fstride;
            if tw1_idx >= n {
                tw1_idx -= n;
            }
            tw2_idx += 2 * fstride;
            if tw2_idx >= n {
                tw2_idx -= n;
            }
        }
    }
}

/// Radix-5 butterfly.
fn kf_bfly5(fout: &mut [KissFftCpx], n: usize, m: usize, fstride: usize, tw: &[KissFftCpx]) {
    let group_size = 5 * m;
    let n_groups = n / group_size;
    let ya_r: f32 = 0.309_017;
    let ya_i: f32 = -0.95105652;
    let yb_r: f32 = -0.809_017;
    let yb_i: f32 = -0.58778525;

    for i in 0..n_groups {
        let base = i * group_size;
        let (mut tw1_idx, mut tw2_idx, mut tw3_idx, mut tw4_idx) = (0, 0, 0, 0);
        for j in 0..m {
            let a0 = fout[base + j];
            let a1 = cmul(fout[base + j + m], tw[tw1_idx]);
            let a2 = cmul(fout[base + j + 2 * m], tw[tw2_idx]);
            let a3 = cmul(fout[base + j + 3 * m], tw[tw3_idx]);
            let a4 = cmul(fout[base + j + 4 * m], tw[tw4_idx]);

            let s12r = a1.r + a4.r;
            let s12i = a1.i + a4.i;
            let d12r = a1.r - a4.r;
            let d12i = a1.i - a4.i;
            let s34r = a2.r + a3.r;
            let s34i = a2.i + a3.i;
            let d34r = a2.r - a3.r;
            let d34i = a2.i - a3.i;

            fout[base + j].r = a0.r + s12r + s34r;
            fout[base + j].i = a0.i + s12i + s34i;

            let t1r = a0.r + s12r * ya_r + s34r * yb_r;
            let t1i = a0.i + s12i * ya_r + s34i * yb_r;
            let t2r = d12i * ya_i + d34i * yb_i;
            let t2i = -(d12r * ya_i + d34r * yb_i);

            fout[base + j + m] = KissFftCpx {
                r: t1r - t2r,
                i: t1i - t2i,
            };
            fout[base + j + 4 * m] = KissFftCpx {
                r: t1r + t2r,
                i: t1i + t2i,
            };

            let t3r = a0.r + s12r * yb_r + s34r * ya_r;
            let t3i = a0.i + s12i * yb_r + s34i * ya_r;
            let t4r = d12i * yb_i - d34i * ya_i;
            let t4i = -(d12r * yb_i - d34r * ya_i);

            fout[base + j + 2 * m] = KissFftCpx {
                r: t3r - t4r,
                i: t3i - t4i,
            };
            fout[base + j + 3 * m] = KissFftCpx {
                r: t3r + t4r,
                i: t3i + t4i,
            };

            tw1_idx += fstride;
            if tw1_idx >= n {
                tw1_idx -= n;
            }
            tw2_idx += 2 * fstride;
            if tw2_idx >= n {
                tw2_idx -= n;
            }
            tw3_idx += 3 * fstride;
            if tw3_idx >= n {
                tw3_idx -= n;
            }
            tw4_idx += 4 * fstride;
            if tw4_idx >= n {
                tw4_idx -= n;
            }
        }
    }
}

#[inline]
fn cmul(a: KissFftCpx, b: KissFftCpx) -> KissFftCpx {
    KissFftCpx {
        r: a.r * b.r - a.i * b.i,
        i: a.r * b.i + a.i * b.r,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fft_delta() {
        let n = 240;
        let st = KissFftState::new(n);

        // Test opus_fft: delta at 0 -> constant 1/n everywhere
        let mut input = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        input[0] = KissFftCpx { r: 1.0, i: 0.0 };
        let mut output = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        opus_fft(&st, &input, &mut output);

        let expected = 1.0 / n as f32;
        let max_err = output
            .iter()
            .map(|c| (c.r - expected).abs().max(c.i.abs()))
            .fold(0.0f32, f32::max);
        println!(
            "opus_fft delta test: max_err={:.6e} (expected constant={:.6e})",
            max_err, expected
        );
        assert!(
            max_err < 1e-6,
            "opus_fft delta test failed: max_err={}",
            max_err
        );
    }

    #[test]
    fn test_fft_impl_delta() {
        let n = 240;
        let st = KissFftState::new(n);

        // opus_fft_impl on bitrev-ordered delta at position 0: should give all 1's
        let mut data = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        data[0] = KissFftCpx { r: 1.0, i: 0.0 };
        opus_fft_impl(&st, &mut data);

        let max_err = data
            .iter()
            .map(|c| (c.r - 1.0).abs().max(c.i.abs()))
            .fold(0.0f32, f32::max);
        println!("opus_fft_impl delta test: max_err={:.6e}", max_err);
        assert!(
            max_err < 1e-6,
            "opus_fft_impl delta test failed: max_err={}",
            max_err
        );
    }

    #[test]
    fn test_fft_480_delta() {
        let n = 480;
        let st = KissFftState::new(n);

        // Delta at 0 -> constant 1/n everywhere
        let mut input = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        input[0] = KissFftCpx { r: 1.0, i: 0.0 };
        let mut output = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        opus_fft(&st, &input, &mut output);

        let expected = 1.0 / n as f32;
        let max_err = output
            .iter()
            .map(|c| (c.r - expected).abs().max(c.i.abs()))
            .fold(0.0f32, f32::max);
        println!(
            "FFT 480 delta: max_err={:.6e}, expected={:.6e}",
            max_err, expected
        );
        assert!(
            max_err < 1e-5,
            "FFT 480 delta test failed: max_err={}",
            max_err
        );
    }

    #[test]
    fn test_fft_480_sinusoid() {
        // Test with a known sinusoid
        let n = 480;
        let st = KissFftState::new(n);
        let pi = std::f64::consts::PI;

        let k = 10; // frequency bin
        let mut input = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        for (i, slot) in input.iter_mut().enumerate() {
            let phase = 2.0 * pi * k as f64 * i as f64 / n as f64;
            slot.r = phase.cos() as f32;
        }
        let mut output = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        opus_fft(&st, &input, &mut output);

        // The peak should be at bin k and n-k
        let scale = 1.0 / n as f32;
        let peak_k = (output[k].r * output[k].r + output[k].i * output[k].i).sqrt();
        let peak_nk =
            (output[n - k].r * output[n - k].r + output[n - k].i * output[n - k].i).sqrt();
        let expected_peak = 0.5 * scale * n as f32; // = 0.5
        println!(
            "FFT 480 sinusoid: peak_k={:.6}, peak_nk={:.6}, expected={:.6}",
            peak_k, peak_nk, expected_peak
        );
        assert!(
            (peak_k - 0.5).abs() < 0.01,
            "Peak at k={} is wrong: {}",
            k,
            peak_k
        );
    }

    #[test]
    fn test_fft_roundtrip() {
        // Test FFT then IFFT roundtrip: opus_ifft(opus_fft(x)) = x
        let n = 240;
        let st = KissFftState::new(n);

        // Create a test signal
        let input: Vec<KissFftCpx> = (0..n)
            .map(|i| KissFftCpx {
                r: (i as f32 * 0.1).sin(),
                i: (i as f32 * 0.2).cos(),
            })
            .collect();

        // Forward FFT (includes 1/n scaling)
        let mut fft_out = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        opus_fft(&st, &input, &mut fft_out);

        // Inverse FFT matching C's opus_ifft_c:
        // 1. bitrev copy (no scaling)
        // 2. conjugate
        // 3. fft_impl
        // 4. conjugate
        let mut ifft_out = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        for (i, &val) in fft_out.iter().enumerate() {
            let rev = st.bitrev[i];
            ifft_out[rev] = val;
        }
        for c in ifft_out.iter_mut() {
            c.i = -c.i;
        }
        opus_fft_impl(&st, &mut ifft_out);
        for c in ifft_out.iter_mut() {
            c.i = -c.i;
        }

        let max_err = input
            .iter()
            .zip(ifft_out.iter())
            .map(|(a, b)| (a.r - b.r).abs().max((a.i - b.i).abs()))
            .fold(0.0f32, f32::max);
        println!("FFT roundtrip test: max_err={:.6e}", max_err);
        assert!(
            max_err < 1e-4,
            "FFT roundtrip test failed: max_err={}",
            max_err
        );
    }

    #[test]
    fn test_factors_240() {
        let factors = kf_factor(240);
        println!("factors for 240: {:?}", factors);
        // 240 = 4*4*3*5, no radix-2
        assert_eq!(factors, vec![5, 48, 3, 16, 4, 4, 4, 1]);
    }

    #[test]
    fn test_factors_120() {
        let factors = kf_factor(120);
        println!("factors for 120: {:?}", factors);
        // 120 = 4*2*3*5, has radix-2
        assert_eq!(factors, vec![5, 24, 3, 8, 2, 4, 4, 1]);
    }

    #[test]
    fn test_fft_roundtrip_60() {
        let n = 60;
        let st = KissFftState::new(n);
        println!("factors for 60: {:?}", st.factors);
        let input: Vec<KissFftCpx> = (0..n)
            .map(|i| KissFftCpx {
                r: (i as f32 * 0.1).sin() * 100.0,
                i: (i as f32 * 0.2).cos() * 100.0,
            })
            .collect();
        let mut fft_out = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        opus_fft(&st, &input, &mut fft_out);
        let mut ifft_out = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        for (i, &val) in fft_out.iter().enumerate() {
            let rev = st.bitrev[i];
            ifft_out[rev] = val;
        }
        for c in ifft_out.iter_mut() {
            c.i = -c.i;
        }
        opus_fft_impl(&st, &mut ifft_out);
        for c in ifft_out.iter_mut() {
            c.i = -c.i;
        }
        let max_err = input
            .iter()
            .zip(ifft_out.iter())
            .map(|(a, b)| (a.r - b.r).abs().max((a.i - b.i).abs()))
            .fold(0.0f32, f32::max);
        println!("FFT roundtrip 60 test: max_err={:.6e}", max_err);
        assert!(
            max_err < 1e-3,
            "FFT roundtrip 60 test failed: max_err={}",
            max_err
        );
    }

    #[test]
    fn test_fft_roundtrip_480() {
        let n = 480;
        let st = KissFftState::new(n);
        let input: Vec<KissFftCpx> = (0..n)
            .map(|i| KissFftCpx {
                r: (i as f32 * 0.1).sin(),
                i: (i as f32 * 0.2).cos(),
            })
            .collect();
        let mut fft_out = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        opus_fft(&st, &input, &mut fft_out);
        let mut ifft_out = vec![KissFftCpx { r: 0.0, i: 0.0 }; n];
        for (i, &val) in fft_out.iter().enumerate() {
            let rev = st.bitrev[i];
            ifft_out[rev] = val;
        }
        for c in ifft_out.iter_mut() {
            c.i = -c.i;
        }
        opus_fft_impl(&st, &mut ifft_out);
        for c in ifft_out.iter_mut() {
            c.i = -c.i;
        }
        let max_err = input
            .iter()
            .zip(ifft_out.iter())
            .map(|(a, b)| (a.r - b.r).abs().max((a.i - b.i).abs()))
            .fold(0.0f32, f32::max);
        println!("FFT roundtrip 480 test: max_err={:.6e}", max_err);
        assert!(
            max_err < 1e-4,
            "FFT roundtrip 480 test failed: max_err={}",
            max_err
        );
    }

    #[test]
    fn test_factors_480() {
        let factors = kf_factor(480);
        println!("factors for 480: {:?}", factors);
        // 480 = 4*4*2*3*5 -> after p==2 swap and reversal: [5,96, 3,32, 4,8, 2,4, 4,1]
        assert_eq!(factors, vec![5, 96, 3, 32, 4, 8, 2, 4, 4, 1]);
    }
}
