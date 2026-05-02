use crate::bands::{amp2_log2, compute_band_energies, normalise_bands, quant_all_bands_enc};
use crate::mathops::ec_ilog;
use crate::mdct::{MdctLookup, clt_mdct_forward};
use crate::mode::CeltMode;
use crate::pitch::{comb_filter, pitch_downsample, pitch_search, remove_doubling};
use crate::quant_energy::{quant_coarse_energy, quant_energy_finalise_enc, quant_fine_energy};
use crate::rate::{clt_compute_allocation_enc, init_caps};
use crate::tables::*;
use opus_range_coder::EcCtx;

// Maximum sizes for scratch buffers (48kHz, stereo, 20ms frame)
const MAX_ENCODE_N: usize = 960; // max frame length = shortMdctSize << maxLM = 120 << 3
const MAX_OVERLAP: usize = 120;
const MAX_CC: usize = 2;
const MAX_INP_SIZE: usize = MAX_CC * (MAX_ENCODE_N + MAX_OVERLAP); // 2160
const MAX_FREQ_SIZE: usize = MAX_CC * MAX_ENCODE_N; // 1920
const MAX_PRE_SIZE: usize = MAX_CC * (MAX_ENCODE_N + COMBFILTER_MAXPERIOD); // 3968
const MAX_PITCH_BUF: usize = (COMBFILTER_MAXPERIOD + MAX_ENCODE_N) / 2; // 992
const MAX_TRANSIENT_TMP: usize = MAX_ENCODE_N + MAX_OVERLAP; // 1080

/// CELT encoder state.
pub struct CeltEncoder {
    pub channels: usize,
    pub stream_channels: usize,
    pub upsample: usize,
    pub start: usize,
    pub end: usize,
    pub signalling: bool,
    pub disable_inv: bool,
    pub force_intra: bool,
    pub clip: bool,
    pub disable_pf: bool,
    pub complexity: i32,
    pub bitrate: i32,
    pub vbr: bool,
    pub constrained_vbr: bool,
    pub loss_rate: i32,
    pub lsb_depth: i32,
    pub lfe: bool,
    // State (gets reset)
    pub rng: u32,
    pub spread_decision: i32,
    pub delayed_intra: f32,
    pub tonal_average: i32,
    pub last_coded_bands: i32,
    pub hf_average: i32,
    pub tapset_decision: i32,
    pub prefilter_period: usize,
    pub prefilter_gain: f32,
    pub prefilter_tapset: usize,
    pub consec_transient: i32,
    pub preemph_mem_e: [f32; 2],
    // VBR state
    pub vbr_reservoir: i32,
    pub vbr_drift: i32,
    pub vbr_offset: i32,
    pub vbr_count: i32,
    pub overlap_max: f32,
    pub stereo_saving: f32,
    pub intensity: i32,
    // Memory buffers (persistent state)
    pub in_mem: Vec<f32>,
    pub prefilter_mem: Vec<f32>,
    pub old_band_e: Vec<f32>,
    pub old_log_e: Vec<f32>,
    pub old_log_e2: Vec<f32>,
    pub energy_error: Vec<f32>,
    // MDCT lookup
    pub mdct: MdctLookup,
    // Scratch buffers (reused across frames, avoid per-frame heap allocation)
    pub scratch: EncoderScratch,
}

/// Pre-allocated scratch buffers for per-frame encoding.
/// Separated from CeltEncoder to allow simultaneous mutable borrows.
#[derive(Default)]
pub struct EncoderScratch {
    pub inp: Vec<f32>,
    pub freq: Vec<f32>,
    pub x_norm: Vec<f32>,
    pub pre: Vec<f32>,
    pub pitch_buf: Vec<f32>,
    pub transient_tmp: Vec<f32>,
}

/// Pre-emphasis filter (float path).
/// Matches C `celt_preemphasis()` for the float case.
///
/// For the common 48 kHz path (`coef[1] == 0`, `upsample == 1`):
///   `out[i] = coef[2] * pcm[i * cc] - mem;  mem = coef[0] * coef[2] * pcm[i * cc]`
/// but since `coef[2] == 1.0` for 48 kHz this simplifies to:
///   `out[i] = pcm[i * cc] - mem;  mem = coef[0] * pcm[i * cc]`
/// If `clip`, clamp the signal value to `[-65536, 65536]` before filtering.
pub fn celt_preemphasis(
    pcm: &[f32],
    out: &mut [f32],
    n: usize,
    cc: usize,
    upsample: usize,
    coef: &[f32; 4],
    mem: &mut f32,
    clip: bool,
) {
    const CELT_SIG_SCALE: f32 = 32768.0;
    let coef0 = coef[0];
    let mut m = *mem;

    // Fast path: common 48 kHz case (coef[1]==0, upsample==1, no clip)
    if coef[1] == 0.0 && upsample == 1 && !clip {
        for i in 0..n {
            let x = pcm[cc * i] * CELT_SIG_SCALE;
            out[i] = x - m;
            m = coef0 * x;
        }
        *mem = m;
        return;
    }

    let nu = n / upsample;
    if upsample != 1 {
        for item in out.iter_mut().take(n) {
            *item = 0.0;
        }
    }
    for i in 0..nu {
        out[i * upsample] = pcm[cc * i] * CELT_SIG_SCALE;
    }

    if clip {
        for i in 0..nu {
            out[i * upsample] =
                out[i * upsample].clamp(-65536.0 * CELT_SIG_SCALE, 65536.0 * CELT_SIG_SCALE);
        }
    }

    // Apply pre-emphasis using coef[0] only (coef[1]==0 for standard modes)
    for item in out.iter_mut().take(n) {
        let x = *item;
        *item = x - m;
        m = coef0 * x;
    }
    *mem = m;
}

/// Compute MDCTs for the encoder.
/// Matches C `compute_mdcts()` for the float case.
///
/// `input` is `CC * (B*N + overlap)` samples, where each channel occupies
/// a contiguous block of `B*N + overlap` samples.
/// `freq` is `CC * B * N` (or `C * B * N` after downmix) frequency-domain output.
fn compute_mdcts(
    mode: &CeltMode,
    short_blocks: i32,
    input: &[f32],
    freq: &mut [f32],
    c: usize,
    cc: usize,
    lm: i32,
    upsample: usize,
    mdct: &mut MdctLookup,
) {
    let overlap = mode.overlap;
    let (b, nb, shift) = if short_blocks != 0 {
        let m = 1usize << lm;
        (m, mode.short_mdct_size, mode.max_lm)
    } else {
        (
            1usize,
            mode.short_mdct_size << lm as usize,
            mode.max_lm - lm as usize,
        )
    };
    let n = b * nb;

    for ch in 0..cc {
        for blk in 0..b {
            // Input: channel ch starts at ch * (n + overlap), block starts at blk * nb
            let in_start = ch * (n + overlap) + blk * nb;
            // Output: interleaved with stride b, starting at blk + ch * n
            clt_mdct_forward(
                mdct,
                &input[in_start..],
                &mut freq[blk + ch * n..],
                mode.window,
                overlap,
                shift,
                b,
            );
        }
    }

    // If stereo input but mono stream, downmix
    if cc == 2 && c == 1 {
        for i in 0..(b * nb) {
            freq[i] = 0.5 * freq[i] + 0.5 * freq[b * nb + i];
        }
    }

    // Handle upsampling: scale and zero-fill
    if upsample != 1 {
        for ch in 0..c {
            let bound = b * nb / upsample;
            let base = ch * b * nb;
            for i in 0..bound {
                freq[base + i] *= upsample as f32;
            }
            for i in bound..(b * nb) {
                freq[base + i] = 0.0;
            }
        }
    }
}

/// Encode TF (time-frequency) resolution decisions.
/// Mirror of the decoder's `tf_decode`, writing bits instead of reading.
fn tf_encode(
    start: usize,
    end: usize,
    is_transient: bool,
    tf_res: &mut [i32],
    lm: i32,
    tf_select: i32,
    enc: &mut EcCtx,
) {
    let budget = enc.storage as i32 * 8;
    let mut tell = enc.tell();
    let mut logp = if is_transient { 2u32 } else { 4u32 };
    let tf_select_rsv = lm > 0 && tell + (logp as i32) < budget;
    let budget = budget - if tf_select_rsv { 1 } else { 0 };
    let mut curr = 0i32;
    let mut tf_changed = 0i32;

    for item in tf_res.iter_mut().take(end).skip(start) {
        if tell + logp as i32 <= budget {
            enc.enc_bit_logp(*item ^ curr != 0, logp);
            tell = enc.tell();
            curr = *item;
            tf_changed |= curr;
        } else {
            *item = curr;
        }
        logp = if is_transient { 4 } else { 5 };
    }

    // Only code tf_select if it would actually make a difference
    let tf_select = if tf_select_rsv
        && TF_SELECT_TABLE[lm as usize][(4 * is_transient as usize) + tf_changed as usize]
            != TF_SELECT_TABLE[lm as usize][(4 * is_transient as usize) + 2 + tf_changed as usize]
    {
        enc.enc_bit_logp(tf_select != 0, 1);
        tf_select
    } else {
        0
    };

    for item in tf_res.iter_mut().take(end).skip(start) {
        *item = TF_SELECT_TABLE[lm as usize]
            [4 * is_transient as usize + 2 * tf_select as usize + *item as usize]
            as i32;
    }
}

/// Transient analysis: detect transients via forward/backward masking.
/// Returns (is_transient, tf_estimate, weak_transient).
fn transient_analysis(
    inp: &[f32],
    len: usize,
    cc: usize,
    allow_weak_transients: bool,
    tmp: &mut [f32],
) -> (bool, f32, bool) {
    static INV_TABLE: [u8; 128] = [
        255, 255, 156, 110, 86, 70, 59, 51, 45, 40, 37, 33, 31, 28, 26, 25, 23, 22, 21, 20, 19, 18,
        17, 16, 16, 15, 15, 14, 13, 13, 12, 12, 12, 12, 11, 11, 11, 10, 10, 10, 9, 9, 9, 9, 9, 9,
        8, 8, 8, 8, 8, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5,
        5, 5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
        4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2,
    ];

    let forward_decay: f32 = if allow_weak_transients {
        0.03125
    } else {
        0.0625
    };
    let len2 = len / 2;
    let mut mask_metric = 0i32;
    // tmp must be at least `len` elements; caller provides scratch buffer.
    // No zeroing needed — the HP filter loop below writes every element.

    for c in 0..cc {
        // High-pass filter: (1 - 2*z^-1 + z^-2) / (1 - z^-1 + 0.5*z^-2)
        let mut mem0: f32 = 0.0;
        let mut mem1: f32 = 0.0;
        for i in 0..len {
            let x = inp[i + c * len];
            let y = mem0 + x;
            let mem00 = mem0;
            mem0 = mem0 - x + 0.5 * mem1;
            mem1 = x - mem00;
            tmp[i] = y;
        }
        // Clear first 12 samples (filter startup)
        for item in tmp.iter_mut().take(12.min(len)) {
            *item = 0.0;
        }

        // Forward masking pass
        let mut mean: f32 = 0.0;
        mem0 = 0.0;
        for i in 0..len2 {
            let x2 = tmp[2 * i] * tmp[2 * i] + tmp[2 * i + 1] * tmp[2 * i + 1];
            mean += x2;
            mem0 = x2 + (1.0 - forward_decay) * mem0;
            tmp[i] = forward_decay * mem0;
        }

        // Backward masking pass
        mem0 = 0.0;
        let mut max_e: f32 = 0.0;
        for i in (0..len2).rev() {
            mem0 = tmp[i] + 0.875 * mem0;
            tmp[i] = 0.125 * mem0;
            max_e = max_e.max(tmp[i]);
        }

        // Mean and norm
        mean = (mean * max_e * 0.5 * len2 as f32).sqrt();
        let norm = if mean > 1e-15 {
            len2 as f32 * 64.0 / (mean * 0.5)
        } else {
            0.0
        };

        // Harmonic mean via inv_table
        let mut unmask = 0i32;
        let mut i = 12;
        while i < len2.saturating_sub(5) {
            let id = (64.0 * norm * (tmp[i] + 1e-15)).clamp(0.0, 127.0) as usize;
            unmask += INV_TABLE[id] as i32;
            i += 4;
        }

        let n_steps = ((len2.saturating_sub(17)) as f32 / 4.0).max(1.0);
        unmask = (64.0 * unmask as f32 * 4.0 / (6.0 * n_steps)) as i32;

        if unmask > mask_metric {
            mask_metric = unmask;
        }
    }

    let mut is_transient = mask_metric > 200;
    let mut weak_transient = false;
    if allow_weak_transients && is_transient && mask_metric < 600 {
        is_transient = false;
        weak_transient = true;
    }

    // Compute tf_estimate
    let tf_max = (27.0 * mask_metric as f32).sqrt() - 42.0;
    let tf_max = tf_max.clamp(0.0, 163.0);
    let tf_estimate = (0.0069 * tf_max - 0.139).max(0.0).sqrt().min(1.0);

    (is_transient, tf_estimate, weak_transient)
}

/// Spreading decision: analyze spectral shape to determine spread mode.
/// Port of bands.c spreading_decision().
fn spreading_decision(
    m: &CeltMode,
    x: &[f32],
    average: &mut i32,
    last_decision: i32,
    hf_average: &mut i32,
    tapset_decision: &mut i32,
    update_hf: bool,
    end: usize,
    c: usize,
    mm: usize,
    spread_weight: Option<&[i32]>,
) -> i32 {
    let n0 = mm * m.short_mdct_size;

    // Early exit for narrow bands
    if mm * (m.ebands[end] as usize - m.ebands[end - 1] as usize) <= 8 {
        return SPREAD_NONE;
    }

    let mut sum = 0i32;
    let mut nb_bands = 0i32;
    let mut hf_sum = 0i32;

    for ch in 0..c {
        for i in 0..end {
            let n = mm * (m.ebands[i + 1] - m.ebands[i]) as usize;
            if n <= 8 {
                continue;
            }
            let x_off = mm * m.ebands[i] as usize + ch * n0;

            let mut tcount = [0i32; 3];
            for j in 0..n {
                let idx = x_off + j;
                if idx < x.len() {
                    let x2n = x[idx] * x[idx] * n as f32;
                    if x2n < 0.25 {
                        tcount[0] += 1;
                    }
                    if x2n < 0.0625 {
                        tcount[1] += 1;
                    }
                    if x2n < 0.015625 {
                        tcount[2] += 1;
                    }
                }
            }

            if i >= m.nb_ebands - 4 {
                hf_sum += 32 * (tcount[1] + tcount[0]) / n as i32;
            }

            let tmp = (if 2 * tcount[2] >= n as i32 { 1 } else { 0 })
                + (if 2 * tcount[1] >= n as i32 { 1 } else { 0 })
                + (if 2 * tcount[0] >= n as i32 { 1 } else { 0 });

            let w = spread_weight.map_or(1, |sw| sw[i]);
            sum += tmp * w;
            nb_bands += w;
        }
    }

    // HF / tapset update
    if update_hf {
        let hf_bands = c as i32 * (4 - m.nb_ebands as i32 + end as i32).max(1);
        if hf_sum > 0 {
            hf_sum /= hf_bands;
        }
        *hf_average = (*hf_average + hf_sum) >> 1;
        hf_sum = *hf_average;
        if *tapset_decision == 2 {
            hf_sum += 4;
        } else if *tapset_decision == 0 {
            hf_sum -= 4;
        }
        *tapset_decision = if hf_sum > 22 {
            2
        } else if hf_sum > 18 {
            1
        } else {
            0
        };
    }

    if nb_bands == 0 {
        return SPREAD_NORMAL;
    }

    // Spreading decision with hysteresis
    sum = (sum << 8) / nb_bands;
    sum = (sum + *average) >> 1;
    *average = sum;

    sum = (3 * sum + (3 - last_decision) * 128 + 64 + 2) >> 2;

    if sum < 80 {
        SPREAD_AGGRESSIVE
    } else if sum < 256 {
        SPREAD_NORMAL
    } else if sum < 384 {
        SPREAD_LIGHT
    } else {
        SPREAD_NONE
    }
}

/// Alloc trim analysis: compute allocation trim from spectral tilt and stereo correlation.
fn alloc_trim_analysis(
    m: &CeltMode,
    x: &[f32],
    band_log_e: &[f32],
    end: usize,
    lm: i32,
    c: usize,
    n0: usize,
    stereo_saving: &mut f32,
    tf_estimate: f32,
    _intensity: i32,
    equiv_rate: i32,
) -> i32 {
    let nb_ebands = m.nb_ebands;

    // Base trim from bitrate
    let mut trim: f32 = if equiv_rate < 64000 {
        4.0
    } else if equiv_rate < 80000 {
        let frac = (equiv_rate - 64000) as f32 / 16000.0;
        4.0 + frac
    } else {
        5.0
    };

    // Stereo correlation
    if c == 2 {
        let mut sum = 0.0f32;
        for i in 0..8.min(end) {
            let band_start = (m.ebands[i] as usize) << lm as usize;
            let band_end = (m.ebands[i + 1] as usize) << lm as usize;
            let band_len = band_end - band_start;
            let mut partial = 0.0f32;
            for j in 0..band_len {
                partial += x[band_start + j] * x[n0 + band_start + j];
            }
            sum += partial;
        }
        sum /= 8.0f32.max(1.0);
        sum = sum.abs().min(1.0);

        let log_xc = (1.001 - sum * sum).ln() / std::f32::consts::LN_2;
        trim += (0.75 * log_xc).max(-4.0);
        *stereo_saving = (*stereo_saving + 0.25).min(-log_xc * 0.5);
    }

    // Spectral tilt
    let mut diff = 0.0f32;
    for ch in 0..c {
        for i in 0..end.saturating_sub(1) {
            diff += band_log_e[i + ch * nb_ebands] * (2 + 2 * i as i32 - end as i32) as f32;
        }
    }
    if end > 1 {
        diff /= (c * (end - 1)) as f32;
    }
    trim -= diff.clamp(-2.0, 2.0);

    // TF estimate adjustment
    trim -= 2.0 * tf_estimate;

    // Quantize
    let trim_index = (trim + 0.5) as i32;
    trim_index.clamp(0, 10)
}

/// Dynamic allocation analysis: compute per-band boost offsets.
fn dynalloc_analysis(
    band_log_e: &[f32],
    old_band_e: &[f32],
    nb_ebands: usize,
    start: usize,
    end: usize,
    c: usize,
    offsets: &mut [i32],
    lsb_depth: i32,
    log_n: &[i16],
    is_transient: bool,
    vbr: bool,
    constrained_vbr: bool,
    ebands: &[i16],
    lm: i32,
    effective_bytes: i32,
    tot_boost: &mut i32,
    lfe: bool,
) -> f32 {
    *tot_boost = 0;
    for v in offsets[..end].iter_mut() {
        *v = 0;
    }

    if effective_bytes < 30 + 5 * lm || lfe {
        return -31.9;
    }

    // Compute noise floor (use stack array since nb_ebands = NB_EBANDS = 21)
    let mut noise_floor = [0.0f32; NB_EBANDS];
    for i in 0..end {
        noise_floor[i] = 0.0625 * log_n[i] as f32 + 0.5 + (9 - lsb_depth) as f32 - E_MEANS[i]
            + 0.0062 * ((i + 5) * (i + 5)) as f32;
    }

    // Compute max depth
    let mut max_depth = -31.9f32;
    for ch in 0..c {
        for i in 0..end {
            max_depth = max_depth.max(band_log_e[ch * nb_ebands + i] - noise_floor[i]);
        }
    }

    // Compute follower (simplified: use max of bandLogE vs noise_floor per band)
    let mut follower = [0.0f32; NB_EBANDS];
    for i in start..end {
        let mut max_e = band_log_e[i];
        if c == 2 {
            max_e = max_e.max(band_log_e[nb_ebands + i]);
        }
        follower[i] = (max_e - noise_floor[i]).max(0.0);
        // Also consider old band energy for stability
        let old_e = if c == 2 {
            old_band_e[i].max(old_band_e[nb_ebands + i])
        } else {
            old_band_e[i]
        };
        let old_diff = (old_e - noise_floor[i]).max(0.0);
        // Use the larger of current and old for transient stability
        if !is_transient {
            follower[i] = follower[i].max(old_diff * 0.5);
        }
    }

    // Forward follower (limit growth to 1.5 dB/band)
    if start < end {
        for i in (start + 1)..end {
            follower[i] = follower[i].min(follower[i - 1] + 1.5);
        }
        // Backward follower (limit growth to 2.0 dB/band)
        for i in (start..end - 1).rev() {
            follower[i] = follower[i].min(follower[i + 1] + 2.0);
        }
    }

    // CBR halving
    if (!vbr || constrained_vbr) && !is_transient {
        for item in follower.iter_mut().take(end).skip(start) {
            *item *= 0.5;
        }
    }

    // Band weighting
    for (i, item) in follower.iter_mut().enumerate().take(end).skip(start) {
        if i < 8 {
            *item *= 2.0;
        }
        if i >= 12 {
            *item *= 0.5;
        }
    }

    // Convert to offsets (number of boost quanta per band)
    let mut boost_total = 0i32;
    for i in start..end {
        follower[i] = follower[i].min(4.0);
        let width = c as i32 * (ebands[i + 1] - ebands[i]) as i32 * (1 << lm);
        // Compute quanta size (matching the encoding loop's quanta calculation)
        let quanta = if width < 6 {
            width
        } else if width > 48 {
            8
        } else {
            6
        };
        // Convert follower (dB above noise floor, 0-4) to boost quanta count.
        const DB_PER_QUANTA: f32 = 2.0;
        const MAX_BOOST_QUANTA: f32 = 2.0;
        let boost = (follower[i] / DB_PER_QUANTA)
            .round()
            .clamp(0.0, MAX_BOOST_QUANTA) as i32;
        let boost_bits = boost * quanta * (1 << BITRES);

        // CBR cap
        if !vbr || (constrained_vbr && !is_transient) {
            let cap = (2 * effective_bytes / 3) << (BITRES + 3);
            if boost_total + boost_bits > cap {
                offsets[i] = (cap - boost_total).max(0) / ((quanta * (1 << BITRES)).max(1));
                *tot_boost = cap;
                break;
            }
        }
        offsets[i] = boost;
        boost_total += boost_bits;
    }
    *tot_boost = boost_total;

    max_depth
}

/// Stereo analysis: decide M/S vs L/R coding.
/// Returns 1 for M/S (dual_stereo), 0 for L/R.
fn stereo_analysis(m: &CeltMode, x: &[f32], lm: i32, n0: usize) -> i32 {
    let mut sum_lr = 1e-15f32;
    let mut sum_ms = 1e-15f32;

    for i in 0..13.min(m.nb_ebands) {
        let band_start = (m.ebands[i] as usize) << lm as usize;
        let band_end = (m.ebands[i + 1] as usize) << lm as usize;
        for j in band_start..band_end {
            if j < x.len() && n0 + j < x.len() {
                let l = x[j];
                let r = x[n0 + j];
                let m_val = l + r;
                let s_val = l - r;
                sum_lr += l.abs() + r.abs();
                sum_ms += m_val.abs() + s_val.abs();
            }
        }
    }

    sum_ms *= std::f32::consts::FRAC_1_SQRT_2;

    let thetas = if lm <= 1 { 5 } else { 13 };
    let band13 = m.ebands[13.min(m.nb_ebands)] as i32;
    let lhs = (((band13 << (lm + 1)) + thetas) as f32) * sum_ms;
    let rhs = ((band13 << (lm + 1)) as f32) * sum_lr;

    if lhs > rhs { 1 } else { 0 }
}

/// VBR rate computation.
fn compute_vbr(
    m: &CeltMode,
    base_target: i32,
    lm: i32,
    _bitrate: i32,
    last_coded_bands: i32,
    c: usize,
    intensity: i32,
    constrained_vbr: bool,
    stereo_saving: f32,
    tot_boost: i32,
    tf_estimate: f32,
    _pitch_change: bool,
    max_depth: f32,
    lfe: bool,
    _has_surround_mask: bool,
) -> i32 {
    let coded_bands = if last_coded_bands != 0 {
        last_coded_bands as usize
    } else {
        m.nb_ebands
    };
    let coded_bins = (m.ebands[coded_bands] as i32) << lm;
    let mut coded_bins_total = coded_bins;
    if c == 2 {
        coded_bins_total += (m.ebands[intensity.min(coded_bands as i32) as usize] as i32) << lm;
    }

    let mut target = base_target;

    // Stereo savings
    if c == 2 {
        let coded_stereo_bands = (intensity as usize).min(coded_bands);
        let coded_stereo_dof =
            ((m.ebands[coded_stereo_bands] as i32) << lm) - coded_stereo_bands as i32;
        if coded_bins_total > 0 && coded_stereo_dof > 0 {
            let max_frac = 0.8 * coded_stereo_dof as f32 / coded_bins_total as f32;
            let saving = stereo_saving.min(1.0);
            let save_amount =
                ((saving - 0.1) * coded_stereo_dof as f32 * (1 << BITRES) as f32) as i32;
            target -= save_amount.min((max_frac * target as f32) as i32);
        }
    }

    // Dynalloc boost
    target += tot_boost - (19 << lm);

    // Transient boost (via tf_estimate)
    let tf_calibration = 0.044f32;
    target += (2.0 * (tf_estimate - tf_calibration) * target as f32) as i32;

    // Floor depth limiting
    if !lfe {
        let bins = (m.ebands[m.nb_ebands.saturating_sub(2)] as i32) << lm;
        let floor_depth = (c as f32 * bins as f32 * (1 << BITRES) as f32 * max_depth) as i32;
        let floor_depth = floor_depth.max(target / 4);
        target = target.min(floor_depth);
    }

    // Constrained VBR dampening
    if constrained_vbr {
        target = base_target + ((target - base_target) as f32 * 0.67) as i32;
    }

    // Cap at 2x base
    target = target.min(2 * base_target);

    target
}

/// Run the prefilter: pitch search + comb filter to remove pitch periodicity.
/// Returns (pf_on, pitch_index, gain, qg).
fn run_prefilter(
    inp: &mut [f32],
    prefilter_mem: &mut [f32],
    in_mem: &mut [f32],
    cc: usize,
    n: usize,
    overlap: usize,
    window: &[f32],
    prefilter_period: usize,
    prefilter_gain: f32,
    prefilter_tapset: usize,
    complexity: i32,
    tf_estimate: f32,
    nb_available_bytes: usize,
    loss_rate: i32,
    pre_scratch: &mut [f32],
    pitch_buf_scratch: &mut [f32],
) -> (bool, usize, f32, i32) {
    let max_period = COMBFILTER_MAXPERIOD;
    let min_period = COMBFILTER_MINPERIOD;

    // Build pre[] buffer: [history | current_frame] per channel
    let pre_len = n + max_period;
    let pre = &mut pre_scratch[..cc * pre_len];
    for ch in 0..cc {
        // Copy history from prefilter memory
        let mem_base = ch * max_period;
        let pre_base = ch * pre_len;
        pre[pre_base..pre_base + max_period]
            .copy_from_slice(&prefilter_mem[mem_base..mem_base + max_period]);
        // Copy current frame from inp (after overlap)
        let in_base = ch * (n + overlap) + overlap;
        pre[pre_base + max_period..pre_base + pre_len].copy_from_slice(&inp[in_base..in_base + n]);
    }

    // Pitch search
    let mut pitch_index;
    let mut gain1: f32;

    if complexity >= 5 {
        let half_len = (max_period + n) >> 1;
        let pitch_buf = &mut pitch_buf_scratch[..half_len];

        // Build references to each channel's pre buffer for pitch_downsample
        // Use stack array (cc is at most MAX_CC = 2)
        let ch0 = &pre[0..2 * half_len];
        let ch1 = if cc == 2 {
            &pre[pre_len..pre_len + 2 * half_len]
        } else {
            &pre[0..0]
        };
        let channel_refs: [&[f32]; 2] = [ch0, ch1];
        pitch_downsample(&channel_refs[..cc], pitch_buf, half_len, cc);

        // Search for pitch
        let mut pi = 0usize;
        pitch_search(
            &pitch_buf[max_period >> 1..],
            pitch_buf,
            n,
            max_period - 3 * min_period,
            &mut pi,
        );
        pitch_index = max_period - pi;

        // Remove doubling
        gain1 = remove_doubling(
            pitch_buf,
            max_period,
            min_period,
            n,
            &mut pitch_index,
            prefilter_period,
            prefilter_gain,
        );

        if pitch_index > max_period - 2 {
            pitch_index = max_period - 2;
        }

        gain1 *= 0.7;
        if loss_rate > 2 {
            gain1 *= 0.5;
        }
        if loss_rate > 4 {
            gain1 *= 0.5;
        }
        if loss_rate > 8 {
            gain1 = 0.0;
        }
    } else {
        gain1 = 0.0;
        pitch_index = COMBFILTER_MINPERIOD;
    }

    // Gain thresholding
    let mut pf_threshold: f32 = 0.2;
    if ((pitch_index as i32 - prefilter_period as i32).unsigned_abs() as usize) * 10 > pitch_index {
        pf_threshold += 0.2;
        if tf_estimate > 0.98 {
            gain1 = 0.0;
        }
    }
    if nb_available_bytes < 25 {
        pf_threshold += 0.1;
    }
    if nb_available_bytes < 35 {
        pf_threshold += 0.1;
    }
    if prefilter_gain > 0.4 {
        pf_threshold -= 0.1;
    }
    if prefilter_gain > 0.55 {
        pf_threshold -= 0.1;
    }
    pf_threshold = pf_threshold.max(0.2);

    let pf_on;
    let qg;
    if gain1 < pf_threshold {
        gain1 = 0.0;
        pf_on = false;
        qg = 0;
    } else {
        if (gain1 - prefilter_gain).abs() < 0.1 {
            gain1 = prefilter_gain;
        }
        let qg_raw = (0.5 + gain1 * 32.0 / 3.0).floor() as i32 - 1;
        qg = qg_raw.clamp(0, 7);
        gain1 = 0.09375 * (qg + 1) as f32;
        pf_on = true;
    };

    // Apply comb filter (with negative gain to remove pitch)
    let old_period = prefilter_period.max(COMBFILTER_MINPERIOD);
    for ch in 0..cc {
        let offset = 120 - overlap; // mode->shortMdctSize - overlap for 48kHz

        // Copy overlap from in_mem into inp
        let in_base = ch * (n + overlap);
        inp[in_base..in_base + overlap].copy_from_slice(&in_mem[ch * overlap..(ch + 1) * overlap]);

        let pre_base = ch * pre_len;
        let inp_start = in_base + overlap;

        // First segment: old period/gain (constant, no transition)
        if offset > 0 {
            comb_filter(
                inp,
                inp_start,
                pre,
                pre_base + max_period,
                old_period,
                old_period,
                offset,
                -prefilter_gain,
                -prefilter_gain,
                prefilter_tapset,
                prefilter_tapset,
                window,
                0, // no overlap transition
            );
        }

        // Second segment: transition from old to new period/gain
        comb_filter(
            inp,
            inp_start + offset,
            pre,
            pre_base + max_period + offset,
            old_period,
            pitch_index,
            n - offset,
            -prefilter_gain,
            -gain1,
            prefilter_tapset,
            if pf_on { prefilter_tapset } else { 0 },
            window,
            overlap,
        );
    }

    // Cancel check (mono only for simplicity)
    let mut cancel = false;
    if cc == 1 {
        let in_base = overlap;
        let mut before_sum = 0.0f32;
        let mut after_sum = 0.0f32;
        let pre_base = max_period;
        for i in 0..n {
            before_sum += pre[pre_base + i].abs();
            after_sum += inp[in_base + i].abs();
        }
        if after_sum > before_sum {
            cancel = true;
        }
    }

    if cancel {
        // Revert: copy unfiltered signal back
        for ch in 0..cc {
            let in_base = ch * (n + overlap) + overlap;
            let pre_base = ch * pre_len + max_period;
            inp[in_base..in_base + n].copy_from_slice(&pre[pre_base..pre_base + n]);

            // Re-apply transition with zero new gain
            let offset = 120 - overlap;
            comb_filter(
                inp,
                in_base + offset,
                pre,
                pre_base + offset,
                old_period,
                pitch_index,
                overlap,
                -prefilter_gain,
                0.0,
                prefilter_tapset,
                prefilter_tapset,
                window,
                overlap,
            );
        }
        return (false, pitch_index, 0.0, 0);
    }

    // Update state: copy overlap tail of inp into in_mem
    for ch in 0..cc {
        let in_base = ch * (n + overlap);
        in_mem[ch * overlap..(ch + 1) * overlap]
            .copy_from_slice(&inp[in_base + n..in_base + n + overlap]);
    }

    // Update prefilter_mem from pre[]
    for ch in 0..cc {
        let mem_base = ch * max_period;
        let pre_base = ch * pre_len;
        if n > max_period {
            prefilter_mem[mem_base..mem_base + max_period]
                .copy_from_slice(&pre[pre_base + n..pre_base + n + max_period]);
        } else {
            prefilter_mem.copy_within(mem_base + n..mem_base + max_period, mem_base);
            prefilter_mem[mem_base + max_period - n..mem_base + max_period]
                .copy_from_slice(&pre[pre_base + max_period..pre_base + max_period + n]);
        }
    }

    (pf_on, pitch_index, gain1, qg)
}

impl CeltEncoder {
    /// Create a new CELT encoder for the given sample rate and channel count.
    pub fn new(sample_rate: i32, channels: usize) -> Result<Self, i32> {
        if !(1..=2).contains(&channels) {
            return Err(-1);
        }
        let mode = CeltMode::get_mode();
        let overlap = mode.overlap;
        let nb_ebands = mode.nb_ebands;

        let upsample = match sample_rate {
            48000 => 1,
            24000 => 2,
            16000 => 3,
            12000 => 4,
            8000 => 6,
            _ => return Err(-1),
        };

        // Allocate memory buffers.
        // in_mem: channels * overlap (holds the overlap portion from previous frame)
        let in_mem = vec![0.0f32; channels * overlap];

        // prefilter_mem: channels * COMBFILTER_MAXPERIOD
        let prefilter_mem = vec![0.0f32; channels * COMBFILTER_MAXPERIOD];

        // Band energy arrays: always 2 * nb_ebands for mono->stereo compatibility
        let old_band_e = vec![0.0f32; 2 * nb_ebands];
        let old_log_e = vec![-28.0f32; 2 * nb_ebands];
        let old_log_e2 = vec![-28.0f32; 2 * nb_ebands];
        let energy_error = vec![0.0f32; 2 * nb_ebands];

        // Create MDCT: N = 2 * shortMdctSize * nbShortMdcts
        let nb_short_mdcts = 1usize << mode.max_lm;
        let mdct_n = 2 * mode.short_mdct_size * nb_short_mdcts;
        let mdct = MdctLookup::new(mdct_n, mode.max_lm);

        let enc = CeltEncoder {
            channels,
            stream_channels: channels,
            upsample,
            start: 0,
            end: mode.eff_ebands,
            signalling: false,
            disable_inv: channels == 1,
            force_intra: false,
            clip: true,
            disable_pf: false,
            complexity: 5,
            bitrate: 510000, // OPUS_BITRATE_MAX equivalent
            vbr: false,
            constrained_vbr: true,
            loss_rate: 0,
            lsb_depth: 24,
            lfe: false,
            rng: 0,
            spread_decision: SPREAD_NORMAL,
            delayed_intra: 0.0,
            tonal_average: 256,
            last_coded_bands: 0,
            hf_average: 0,
            tapset_decision: 0,
            prefilter_period: 0,
            prefilter_gain: 0.0,
            prefilter_tapset: 0,
            consec_transient: 0,
            preemph_mem_e: [0.0; 2],
            vbr_reservoir: 0,
            vbr_drift: 0,
            vbr_offset: 0,
            vbr_count: 0,
            overlap_max: 0.0,
            stereo_saving: 0.0,
            intensity: 0,
            in_mem,
            prefilter_mem,
            old_band_e,
            old_log_e,
            old_log_e2,
            energy_error,
            mdct,
            scratch: EncoderScratch {
                inp: vec![0.0f32; MAX_INP_SIZE],
                freq: vec![0.0f32; MAX_FREQ_SIZE],
                x_norm: vec![0.0f32; MAX_FREQ_SIZE],
                pre: vec![0.0f32; MAX_PRE_SIZE],
                pitch_buf: vec![0.0f32; MAX_PITCH_BUF],
                transient_tmp: vec![0.0f32; MAX_TRANSIENT_TMP],
            },
        };

        Ok(enc)
    }

    /// Encode a CELT frame.
    ///
    /// The main encoding pipeline:
    ///  1. Validate inputs, compute LM from frame_size
    ///  2. Initialize range encoder if ec is None
    ///  3. Pre-emphasis
    ///  4. Silence detection
    ///  5. Skip prefilter (encode pf_on=0)
    ///  6. Transient flag (simplified: always false)
    ///  7. Compute MDCTs
    ///  8. Compute band energies and normalize
    ///  9. Energy error bias
    /// 10. Coarse energy quantization
    /// 11. TF encode (all zeros)
    /// 12. Spread decision and encode
    /// 13. Dynamic allocation encode (simplified)
    /// 14. Alloc trim encode
    /// 15. Bit allocation (clt_compute_allocation_enc)
    /// 16. Fine energy quantization
    /// 17. Band quantization (quant_all_bands_enc)
    /// 18. Anti-collapse
    /// 19. Energy finalization
    /// 20. State update (oldBandE, oldLogE, oldLogE2, energyError)
    /// 21. enc_done() and return compressed size
    pub fn encode_with_ec(
        &mut self,
        pcm: &[f32],
        frame_size: usize,
        compressed: &mut [u8],
        nb_compressed_bytes: usize,
        ec: Option<&mut EcCtx>,
    ) -> Result<usize, i32> {
        let mode = CeltMode::get_mode();
        let nb_ebands = mode.nb_ebands;
        let overlap = mode.overlap;
        let cc = self.channels;
        let c = self.stream_channels;

        if nb_compressed_bytes < 2 {
            return Err(-1);
        }

        let frame_size = frame_size * self.upsample;

        // Find LM from frame_size
        let mut lm = 0i32;
        while lm <= mode.max_lm as i32 {
            if mode.short_mdct_size << lm as usize == frame_size {
                break;
            }
            lm += 1;
        }
        if lm > mode.max_lm as i32 {
            return Err(-1);
        }
        let mm = 1usize << lm;
        let n = mm * mode.short_mdct_size;

        let mut nb_compressed_bytes = nb_compressed_bytes.min(1275);

        let start = self.start;
        let end = self.end;
        let mut eff_end = end;
        if eff_end > mode.eff_ebands {
            eff_end = mode.eff_ebands;
        }

        // -----------------------------------------------------------------
        // 2. Initialize range encoder
        // -----------------------------------------------------------------
        let mut local_enc;
        let nb_filled_bytes = if let Some(ref ext) = ec {
            ((ext.tell() + 4) >> 3) as usize
        } else {
            0
        };

        // For CBR, compute the actual packet size from bitrate
        if !self.vbr && self.bitrate != 510000 {
            let tmp = self.bitrate as i64 * frame_size as i64;
            let new_bytes = 2i32
                .max(nb_compressed_bytes as i32)
                .min(((tmp + 4 * mode.fs as i64) / (8 * mode.fs as i64)) as i32);
            nb_compressed_bytes = new_bytes as usize;
        }

        let mut nb_available_bytes = nb_compressed_bytes.saturating_sub(nb_filled_bytes);

        // Compute VBR rate early (needed for constrained VBR budget limiting).
        let vbr_rate = if self.vbr && self.bitrate != 510000 {
            ((self.bitrate as i64 * frame_size as i64 / mode.fs as i64) as i32) << BITRES
        } else {
            0
        };

        local_enc = EcCtx::enc_init(nb_compressed_bytes as u32);
        // Use either the provided encoder or our local one
        let enc_is_external = ec.is_some();
        let enc = if let Some(ext) = ec {
            ext
        } else {
            &mut local_enc
        };

        // Early constrained VBR budget limiting (matches C reference lines 1941-1958).
        // Prevents encoding more bits than the VBR reservoir allows, which would
        // cause enc_shrink to fail later when the main VBR adjustment runs.
        if vbr_rate > 0 && self.constrained_vbr {
            let tell = enc.tell();
            let vbr_bound = vbr_rate;
            let max_allowed = ((vbr_rate + vbr_bound - self.vbr_reservoir) >> (BITRES + 3))
                .max(if tell == 1 { 2 } else { 0 })
                .min(nb_available_bytes as i32);
            if (max_allowed as usize) < nb_available_bytes {
                nb_compressed_bytes = nb_filled_bytes + max_allowed as usize;
                nb_available_bytes = max_allowed as usize;
                enc.enc_shrink(nb_compressed_bytes as u32);
            }
        }

        let mut total_bits = nb_compressed_bytes as i32 * 8;
        let mut tell = enc.tell();

        // Take scratch buffers out of self to avoid borrow conflicts
        let mut scratch = std::mem::take(&mut self.scratch);

        // -----------------------------------------------------------------
        // 3. Pre-emphasis
        // -----------------------------------------------------------------
        let buf_size = cc * (n + overlap);
        let inp = &mut scratch.inp[..buf_size];

        // Compute sample_max for silence detection
        let sample_max = {
            let mut smax = self.overlap_max;
            let non_overlap_len = cc * (n - overlap) / self.upsample;
            for item in pcm.iter().take(non_overlap_len.min(pcm.len())) {
                smax = smax.max(item.abs());
            }
            let overlap_start = non_overlap_len;
            let overlap_pcm_len = cc * overlap / self.upsample;
            let mut overlap_max = 0.0f32;
            for item in pcm
                .iter()
                .take((overlap_start + overlap_pcm_len).min(pcm.len()))
                .skip(overlap_start)
            {
                overlap_max = overlap_max.max(item.abs());
            }
            self.overlap_max = overlap_max;
            smax.max(overlap_max)
        };

        // Apply pre-emphasis per channel
        for ch in 0..cc {
            let need_clip = self.clip && sample_max > 65536.0;
            let out_start = ch * (n + overlap) + overlap;
            celt_preemphasis(
                &pcm[ch..],
                &mut inp[out_start..out_start + n],
                n,
                cc,
                self.upsample,
                &mode.preemph,
                &mut self.preemph_mem_e[ch],
                need_clip,
            );
            // Copy overlap from prefilter memory (tail of previous frame)
            // prefilter_mem layout: ch * COMBFILTER_MAXPERIOD .. (ch+1)*COMBFILTER_MAXPERIOD
            let mem_start = (ch + 1) * COMBFILTER_MAXPERIOD - overlap;
            let mem_end = (ch + 1) * COMBFILTER_MAXPERIOD;
            let in_base = ch * (n + overlap);
            inp[in_base..in_base + overlap]
                .copy_from_slice(&self.prefilter_mem[mem_start..mem_end]);
        }

        // -----------------------------------------------------------------
        // 4. Silence detection
        // -----------------------------------------------------------------
        let silence = sample_max <= 1.0 / (1i32 << self.lsb_depth) as f32;
        if tell == 1 {
            enc.enc_bit_logp(silence, 15);
        }
        if silence {
            // In VBR mode there is no need to send more than the minimum.
            if self.vbr && self.bitrate != 510000 {
                nb_compressed_bytes = nb_compressed_bytes.min(nb_filled_bytes + 2);
                total_bits = nb_compressed_bytes as i32 * 8;
                nb_available_bytes = nb_compressed_bytes.saturating_sub(nb_filled_bytes);
                enc.enc_shrink(nb_compressed_bytes as u32);
            }
            // Pretend we have filled all remaining bits with zeros
            // (that's what the initialiser did anyway)
            let tell_now = enc.tell();
            enc.nbits_total += total_bits - tell_now;
        }

        // -----------------------------------------------------------------
        // 5. Transient analysis
        // -----------------------------------------------------------------
        let (mut is_transient, tf_estimate, _weak_transient) = if self.complexity >= 1 && !self.lfe
        {
            transient_analysis(inp, n + overlap, cc, false, &mut scratch.transient_tmp)
        } else {
            (false, 0.0f32, false)
        };

        // -----------------------------------------------------------------
        // 5b. Prefilter (pitch search + comb filter)
        // -----------------------------------------------------------------
        let enabled = ((self.lfe && nb_available_bytes > 3) || nb_available_bytes > 12 * c)
            && !silence
            && !self.disable_pf;
        tell = enc.tell();
        let (pf_on, pitch_index, gain1, qg) = if enabled && start == 0 && tell + 16 <= total_bits {
            run_prefilter(
                inp,
                &mut self.prefilter_mem,
                &mut self.in_mem,
                cc,
                n,
                overlap,
                mode.window,
                self.prefilter_period,
                self.prefilter_gain,
                self.prefilter_tapset,
                self.complexity,
                tf_estimate,
                nb_available_bytes,
                self.loss_rate,
                &mut scratch.pre,
                &mut scratch.pitch_buf,
            )
        } else {
            (false, COMBFILTER_MINPERIOD, 0.0, 0)
        };

        // Encode prefilter parameters
        tell = enc.tell();
        if start == 0 && tell + 16 <= total_bits {
            if !pf_on {
                enc.enc_bit_logp(false, 1);
            } else {
                enc.enc_bit_logp(true, 1);
                let pi = pitch_index + 1;
                let octave = (ec_ilog(pi as u32) as usize) - 5;
                enc.enc_uint(octave as u32, 6);
                enc.enc_bits((pi - (16 << octave)) as u32, (4 + octave) as u32);
                enc.enc_bits(qg as u32, 3);
                enc.enc_icdf(self.prefilter_tapset, &TAPSET_ICDF, 2);
            }
        }

        // -----------------------------------------------------------------
        // 6. Transient flag
        // -----------------------------------------------------------------
        let mut short_blocks = 0i32;
        tell = enc.tell();
        if lm > 0 && tell + 3 <= total_bits {
            if is_transient {
                short_blocks = mm as i32;
            }
            enc.enc_bit_logp(is_transient, 3);
        } else {
            is_transient = false;
        }

        // -----------------------------------------------------------------
        // 7. Compute MDCTs
        // -----------------------------------------------------------------
        let freq_size = c * n;
        let freq = &mut scratch.freq[..freq_size.max(1)];

        compute_mdcts(
            mode,
            short_blocks,
            inp,
            freq,
            c,
            cc,
            lm,
            self.upsample,
            &mut self.mdct,
        );

        // -----------------------------------------------------------------
        // 8. Compute band energies and normalize
        // -----------------------------------------------------------------
        let mut band_e = [0.0f32; MAX_CC * NB_EBANDS];
        let mut band_log_e = [0.0f32; MAX_CC * NB_EBANDS];

        compute_band_energies(mode, freq, &mut band_e, eff_end, c, lm);

        if self.lfe {
            for i in 2..end {
                band_e[i] = band_e[i].min(1e-4 * band_e[0]).max(1e-15);
                if c == 2 {
                    band_e[nb_ebands + i] = band_e[nb_ebands + i]
                        .min(1e-4 * band_e[nb_ebands])
                        .max(1e-15);
                }
            }
        }

        amp2_log2(mode, eff_end, end, &band_e, &mut band_log_e, c);

        // Normalize bands (creates normalized MDCTs in x_norm)
        let x_norm = &mut scratch.x_norm[..c * n];
        // Zero the full buffer to avoid stale data for bands beyond eff_end
        for v in x_norm.iter_mut() {
            *v = 0.0;
        }
        normalise_bands(mode, freq, x_norm, &band_e, eff_end, c, mm);

        // -----------------------------------------------------------------
        // 9. Energy error bias
        // -----------------------------------------------------------------
        for ch in 0..c {
            for i in start..end {
                let idx = i + ch * nb_ebands;
                if (band_log_e[idx] - self.old_band_e[idx]).abs() < 2.0 {
                    band_log_e[idx] -= 0.25 * self.energy_error[idx];
                }
            }
        }

        // -----------------------------------------------------------------
        // 10. Coarse energy quantization
        // -----------------------------------------------------------------
        let mut error = [0.0f32; MAX_CC * NB_EBANDS];
        quant_coarse_energy(
            mode,
            start,
            end,
            eff_end,
            &band_log_e,
            &mut self.old_band_e,
            total_bits,
            &mut error,
            enc,
            c,
            lm as usize,
            nb_available_bytes as i32,
            self.force_intra,
            &mut self.delayed_intra,
            self.complexity >= 4,
            self.loss_rate,
            self.lfe,
        );

        // -----------------------------------------------------------------
        // 11. TF encode (all zeros for simplified encoder)
        // -----------------------------------------------------------------
        let mut tf_res = [0i32; NB_EBANDS];
        for item in tf_res.iter_mut().take(nb_ebands) {
            *item = if is_transient { 1 } else { 0 };
        }
        let tf_select = 0i32;
        tf_encode(start, end, is_transient, &mut tf_res, lm, tf_select, enc);

        // -----------------------------------------------------------------
        // 12. Spread decision and encode
        // -----------------------------------------------------------------
        tell = enc.tell();
        if tell + 4 <= total_bits {
            if self.lfe {
                self.spread_decision = SPREAD_NORMAL;
            } else if short_blocks != 0 || self.complexity < 3 || nb_available_bytes < 10 * c {
                if self.complexity == 0 {
                    self.spread_decision = SPREAD_NONE;
                } else {
                    self.spread_decision = SPREAD_NORMAL;
                }
            } else {
                self.spread_decision = spreading_decision(
                    mode,
                    x_norm,
                    &mut self.tonal_average,
                    self.spread_decision,
                    &mut self.hf_average,
                    &mut self.tapset_decision,
                    !self.disable_pf && short_blocks == 0,
                    eff_end,
                    c,
                    mm,
                    None, // spread_weight (simplified)
                );
            }
            enc.enc_icdf(self.spread_decision as usize, &SPREAD_ICDF, 5);
        } else {
            self.spread_decision = SPREAD_NORMAL;
        }

        // -----------------------------------------------------------------
        // 13. Dynamic allocation
        // -----------------------------------------------------------------
        let mut cap = [0i32; NB_EBANDS];
        init_caps(mode, &mut cap, lm, c as i32);

        let mut offsets = [0i32; NB_EBANDS];
        let effective_bytes = nb_available_bytes as i32;
        let mut tot_boost = 0i32;
        let max_depth = dynalloc_analysis(
            &band_log_e,
            &self.old_band_e,
            nb_ebands,
            start,
            end,
            c,
            &mut offsets,
            self.lsb_depth,
            mode.log_n,
            is_transient,
            self.vbr,
            self.constrained_vbr,
            mode.ebands,
            lm,
            effective_bytes,
            &mut tot_boost,
            self.lfe,
        );

        // Encode dynamic allocation
        let mut dynalloc_logp = 6i32;
        let total_bits_shifted = (nb_compressed_bytes as i32 * 8) << BITRES;
        let mut total_boost = 0i32;
        tell = enc.tell_frac() as i32;
        for i in start..end {
            let width = c as i32 * (mode.ebands[i + 1] - mode.ebands[i]) as i32 * mm as i32;
            let quanta = (width << BITRES).min((6i32 << BITRES).max(width));
            let mut dynalloc_loop_logp = dynalloc_logp;
            let mut boost = 0i32;
            let mut j = 0;
            while (tell + (dynalloc_loop_logp << BITRES)) < total_bits_shifted - total_boost
                && boost < cap[i]
            {
                let flag = j < offsets[i];
                enc.enc_bit_logp(flag, dynalloc_loop_logp as u32);
                tell = enc.tell_frac() as i32;
                if !flag {
                    break;
                }
                boost += quanta;
                total_boost += quanta;
                dynalloc_loop_logp = 1;
                j += 1;
            }
            if j > 0 {
                dynalloc_logp = 2i32.max(dynalloc_logp - 1);
            }
            offsets[i] = boost;
        }

        // -----------------------------------------------------------------
        // 14. Alloc trim encode
        // -----------------------------------------------------------------
        let equiv_rate = ((nb_compressed_bytes as i64 * 8 * 50) << (3 - lm)) as i32
            - (40 * c as i32 + 20) * ((400 >> lm) - 50);
        let alloc_trim;
        tell = enc.tell_frac() as i32;
        if tell + (6 << BITRES) <= total_bits_shifted - tot_boost {
            if start > 0 || self.lfe {
                self.stereo_saving = 0.0;
                alloc_trim = 5;
            } else {
                alloc_trim = alloc_trim_analysis(
                    mode,
                    x_norm,
                    &band_log_e,
                    end,
                    lm,
                    c,
                    mm * mode.short_mdct_size,
                    &mut self.stereo_saving,
                    tf_estimate,
                    self.intensity,
                    equiv_rate,
                );
            }
            enc.enc_icdf(alloc_trim as usize, &TRIM_ICDF, 7);
            let _ = enc.tell_frac();
        } else {
            alloc_trim = 5;
        }

        // -----------------------------------------------------------------
        // 14b. VBR rate computation
        // -----------------------------------------------------------------
        // min_allowed: smallest packet that won't break the encoder.
        // Must be at least the physical bytes already written to the range coder.
        let min_allowed = (((enc.tell() + tot_boost + (1 << (BITRES + 3)) - 1) >> (BITRES + 3))
            + 2)
        .max((enc.offs + enc.end_offs) as i32);
        if self.vbr && self.bitrate != 510000 {
            let lm_diff = mode.max_lm as i32 - lm;
            let vbr_rate =
                ((self.bitrate as i64 * frame_size as i64 / mode.fs as i64) as i32) << BITRES;
            let base_target = vbr_rate - ((40 * c as i32 + 20) << BITRES);
            let target = compute_vbr(
                mode,
                base_target,
                lm,
                self.bitrate,
                self.last_coded_bands,
                c,
                self.intensity,
                self.constrained_vbr,
                self.stereo_saving,
                tot_boost,
                tf_estimate,
                false,
                max_depth,
                self.lfe,
                false,
            );
            let target = target + enc.tell();
            let mut nb_avail = (target + (1 << (BITRES + 2))) >> (BITRES + 3);
            nb_avail = nb_avail.max(min_allowed);
            nb_avail = nb_avail.min(nb_compressed_bytes as i32);

            // VBR reservoir update
            let delta = target - vbr_rate;
            if self.constrained_vbr {
                self.vbr_reservoir += target - vbr_rate;
                if self.vbr_reservoir < 0 {
                    let adjust = (-self.vbr_reservoir) / (8 << BITRES);
                    nb_avail += if silence { 0 } else { adjust };
                    self.vbr_reservoir = 0;
                }
            }
            nb_compressed_bytes = nb_avail.min(nb_compressed_bytes as i32) as usize;
            enc.enc_shrink(nb_compressed_bytes as u32);

            // Update VBR drift
            if self.vbr_count < 970 {
                self.vbr_count += 1;
            }
            if self.constrained_vbr {
                let alpha = if self.vbr_count < 970 {
                    1.0 / (self.vbr_count + 20) as f32
                } else {
                    0.001
                };
                self.vbr_drift +=
                    (alpha * ((delta << lm_diff) - self.vbr_offset - self.vbr_drift) as f32) as i32;
                self.vbr_offset = -self.vbr_drift;
            }
        }

        // -----------------------------------------------------------------
        // 15. Bit allocation
        // -----------------------------------------------------------------
        let mut fine_quant = [0i32; NB_EBANDS];
        let mut pulses = [0i32; NB_EBANDS];
        let mut fine_priority = [0i32; NB_EBANDS];
        let mut balance = 0i32;

        let mut bits = ((nb_compressed_bytes as i32 * 8) << BITRES) - enc.tell_frac() as i32 - 1;
        let anti_collapse_rsv = if is_transient && lm >= 2 && bits >= ((lm + 2) << BITRES) {
            1 << BITRES
        } else {
            0
        };
        bits -= anti_collapse_rsv;

        // For stereo, determine intensity and dual_stereo
        let mut dual_stereo = if c == 2 {
            stereo_analysis(mode, x_norm, lm, n)
        } else {
            0
        };

        // Intensity stereo threshold (from C reference)
        let mut enc_intensity = if c == 2 {
            static INTENSITY_THRESHOLDS: [i32; 21] = [
                1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 36, 44, 50, 56, 62, 67, 72, 79, 88, 106, 134,
            ];
            static INTENSITY_HISTERESIS: [i32; 21] = [
                1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 4, 5, 6, 8, 8,
            ];
            let rate_kbps = equiv_rate / 1000;
            let mut new_intensity = start as i32;
            for j in 0..21 {
                let mut thresh = INTENSITY_THRESHOLDS[j];
                if self.intensity >= (start as i32 + j as i32 + 1) {
                    thresh -= INTENSITY_HISTERESIS[j];
                } else {
                    thresh += INTENSITY_HISTERESIS[j];
                }
                if rate_kbps >= thresh {
                    new_intensity = (start as i32 + j as i32 + 1).min(end as i32);
                }
            }
            self.intensity = new_intensity;
            new_intensity
        } else {
            0
        };

        let coded_bands = clt_compute_allocation_enc(
            mode,
            start,
            end,
            &offsets,
            &cap,
            alloc_trim,
            &mut enc_intensity,
            &mut dual_stereo,
            bits,
            &mut balance,
            &mut pulses,
            &mut fine_quant,
            &mut fine_priority,
            c as i32,
            lm,
            enc,
            self.last_coded_bands,
            end as i32 - 1, // signalBandwidth
        );

        // Update last_coded_bands
        if self.last_coded_bands != 0 {
            self.last_coded_bands = self
                .last_coded_bands
                .min(coded_bands as i32 + 1)
                .max((coded_bands as i32).saturating_sub(1));
        } else {
            self.last_coded_bands = coded_bands as i32;
        }
        self.intensity = enc_intensity;

        // -----------------------------------------------------------------
        // 16. Fine energy quantization
        // -----------------------------------------------------------------
        quant_fine_energy(
            mode,
            start,
            end,
            &mut self.old_band_e,
            &mut error,
            &fine_quant,
            enc,
            c,
        );

        // Clear energy error (will be repopulated at the end)
        for v in self.energy_error.iter_mut() {
            *v = 0.0;
        }

        // -----------------------------------------------------------------
        // 17. Band quantization (quant_all_bands_enc)
        // -----------------------------------------------------------------
        let mut collapse_masks = [0u8; MAX_CC * NB_EBANDS];

        // Split x_norm into X and Y for stereo (safe disjoint borrows via split_at_mut)
        let (x_ref, y_ref): (&mut [f32], Option<&mut [f32]>) = if c == 2 {
            let (x_part, y_part) = x_norm.split_at_mut(n);
            (x_part, Some(y_part))
        } else {
            (&mut x_norm[..], None)
        };

        quant_all_bands_enc(
            mode,
            start,
            end,
            x_ref,
            y_ref,
            &mut collapse_masks,
            &band_e,
            &mut pulses,
            short_blocks,
            self.spread_decision,
            dual_stereo,
            enc_intensity,
            &tf_res,
            nb_compressed_bytes as i32 * (8 << BITRES) - anti_collapse_rsv,
            balance,
            enc,
            lm,
            coded_bands,
            &mut self.rng,
            self.disable_inv,
        );

        // -----------------------------------------------------------------
        // 18. Anti-collapse
        // -----------------------------------------------------------------
        let _anti_collapse_on = if anti_collapse_rsv > 0 {
            let on = self.consec_transient < 2;
            enc.enc_bits(if on { 1 } else { 0 }, 1);
            on
        } else {
            false
        };

        // -----------------------------------------------------------------
        // 19. Energy finalization
        // -----------------------------------------------------------------
        quant_energy_finalise_enc(
            mode,
            start,
            end,
            Some(&mut self.old_band_e),
            &mut error,
            &fine_quant,
            &fine_priority,
            nb_compressed_bytes as i32 * 8 - enc.tell(),
            enc,
            c,
        );

        // -----------------------------------------------------------------
        // 20. State update
        // -----------------------------------------------------------------
        // Update energy error (clamped to [-0.5, 0.5])
        for ch in 0..c {
            for i in start..end {
                self.energy_error[i + ch * nb_ebands] = error[i + ch * nb_ebands].clamp(-0.5, 0.5);
            }
        }

        if silence {
            for i in 0..(c * nb_ebands) {
                self.old_band_e[i] = -28.0;
            }
        }

        // If mono stream but stereo channels, copy
        if cc == 2 && c == 1 {
            self.old_band_e.copy_within(0..nb_ebands, nb_ebands);
        }

        if !is_transient {
            self.old_log_e2[..cc * nb_ebands].copy_from_slice(&self.old_log_e[..cc * nb_ebands]);
            self.old_log_e[..cc * nb_ebands].copy_from_slice(&self.old_band_e[..cc * nb_ebands]);
        } else {
            for i in 0..(cc * nb_ebands) {
                self.old_log_e[i] = self.old_log_e[i].min(self.old_band_e[i]);
            }
        }

        // Clear out-of-range bands
        for ch in 0..cc {
            for i in 0..start {
                self.old_band_e[ch * nb_ebands + i] = 0.0;
                self.old_log_e[ch * nb_ebands + i] = -28.0;
                self.old_log_e2[ch * nb_ebands + i] = -28.0;
            }
            for i in end..nb_ebands {
                self.old_band_e[ch * nb_ebands + i] = 0.0;
                self.old_log_e[ch * nb_ebands + i] = -28.0;
                self.old_log_e2[ch * nb_ebands + i] = -28.0;
            }
        }

        if is_transient {
            self.consec_transient += 1;
        } else {
            self.consec_transient = 0;
        }

        self.rng = enc.rng;

        // Update prefilter state for next frame
        self.prefilter_period = pitch_index;
        self.prefilter_gain = gain1;
        if pf_on {
            self.prefilter_tapset = self.tapset_decision as usize;
        }
        // Note: prefilter_mem and in_mem are already updated inside run_prefilter
        // If prefilter wasn't run, update prefilter_mem from the input buffer
        if !enabled || start != 0 {
            for ch in 0..cc {
                let mem_base = ch * COMBFILTER_MAXPERIOD;
                let in_base = ch * (n + overlap);
                if n >= COMBFILTER_MAXPERIOD {
                    let src_start = in_base + overlap + n - COMBFILTER_MAXPERIOD;
                    self.prefilter_mem[mem_base..mem_base + COMBFILTER_MAXPERIOD]
                        .copy_from_slice(&inp[src_start..src_start + COMBFILTER_MAXPERIOD]);
                } else {
                    let keep = COMBFILTER_MAXPERIOD - n;
                    self.prefilter_mem
                        .copy_within(mem_base + n..mem_base + COMBFILTER_MAXPERIOD, mem_base);
                    let src_start = in_base + overlap;
                    self.prefilter_mem[mem_base + keep..mem_base + COMBFILTER_MAXPERIOD]
                        .copy_from_slice(&inp[src_start..src_start + n]);
                }
            }
        }

        // -----------------------------------------------------------------
        // 21. Finalize and return
        // -----------------------------------------------------------------
        // Only call enc_done() for local (non-external) range coders.
        // When using an external encoder, the caller (opus encoder) handles finalization.
        if !enc_is_external {
            enc.enc_done();
        }

        if enc.get_error() {
            self.scratch = scratch;
            return Err(-3); // OPUS_INTERNAL_ERROR
        }

        // Copy the encoded data to the output buffer
        if !enc_is_external {
            let encoded_len = nb_compressed_bytes.min(compressed.len());
            compressed[..encoded_len].copy_from_slice(&enc.buf[..encoded_len]);
        }

        // Put scratch buffers back
        self.scratch = scratch;

        Ok(nb_compressed_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::CeltDecoder;

    #[test]
    fn test_encoder_create() {
        let enc = CeltEncoder::new(48000, 1);
        assert!(enc.is_ok());
        let enc = enc.unwrap();
        assert_eq!(enc.channels, 1);
        assert_eq!(enc.upsample, 1);

        let enc = CeltEncoder::new(48000, 2);
        assert!(enc.is_ok());

        let enc = CeltEncoder::new(48000, 3);
        assert!(enc.is_err());
    }

    #[test]
    fn test_encode_silence() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        let pcm = vec![0.0f32; 960]; // 20ms at 48kHz
        let mut compressed = vec![0u8; 128];
        let result = enc.encode_with_ec(&pcm, 960, &mut compressed, 128, None);
        assert!(result.is_ok());
        let nbytes = result.unwrap();
        assert!(nbytes > 0);
        assert!(nbytes <= 128);
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        // Encode a sine wave with the CELT encoder, then decode it
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        let mut dec = CeltDecoder::new(48000, 1).unwrap();

        // Generate a 440 Hz sine wave
        let n = 960; // 20ms frame
        let mut pcm = vec![0.0f32; n];
        for (i, sample) in pcm.iter_mut().enumerate() {
            *sample = 10000.0 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin();
        }

        // Encode
        let mut compressed = vec![0u8; 128];
        let nbytes = enc
            .encode_with_ec(&pcm, n, &mut compressed, 128, None)
            .unwrap();
        assert!(nbytes > 2, "Encoded packet should be more than 2 bytes");

        // Decode
        let mut decoded = vec![0.0f32; n];
        let result = dec.decode_with_ec(&compressed[..nbytes], &mut decoded, n, None);
        assert!(result.is_ok(), "Decoder should accept encoder output");

        // Verify the decoded signal has some energy (not silence)
        let energy: f32 = decoded.iter().map(|x| x * x).sum();
        assert!(energy > 0.0, "Decoded signal should have non-zero energy");
    }

    #[test]
    fn test_encode_multiple_frames() {
        let mut enc = CeltEncoder::new(48000, 1).unwrap();
        let mut dec = CeltDecoder::new(48000, 1).unwrap();

        for frame in 0..5 {
            let n = 960;
            let mut pcm = vec![0.0f32; n];
            let freq = 440.0 + frame as f32 * 100.0;
            for (i, sample) in pcm.iter_mut().enumerate() {
                *sample = 5000.0
                    * (2.0 * std::f32::consts::PI * freq * (frame * n + i) as f32 / 48000.0).sin();
            }

            let mut compressed = vec![0u8; 128];
            let nbytes = enc
                .encode_with_ec(&pcm, n, &mut compressed, 128, None)
                .unwrap();

            let mut decoded = vec![0.0f32; n];
            let result = dec.decode_with_ec(&compressed[..nbytes], &mut decoded, n, None);
            assert!(
                result.is_ok(),
                "Frame {frame}: decoder should accept encoder output"
            );
        }
    }
}
