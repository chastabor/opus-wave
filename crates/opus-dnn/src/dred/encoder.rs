use opus_range_coder::EcCtx;

use crate::freq::PREEMPHASIS;
use crate::lpcnet::enc::{LpcnetEncState, NB_TOTAL_FEATURES, compute_frame_features, preemphasis};
use crate::nnet::Activation;
use crate::nnet::activations::compute_activation;

use super::coding::compute_quantizer;
use super::decoder::DredStats;
use super::rdovae_enc::*;
use super::*;

/// DRED encoder state. Matches C `DREDEnc` from dred_encoder.h.
pub struct DredEnc {
    pub model: RdovaeEnc,
    pub lpcnet_enc_state: LpcnetEncState,
    pub rdovae_enc: RdovaeEncState,
    pub loaded: bool,

    pub input_buffer: [f32; 2 * DRED_DFRAME_SIZE],
    pub input_buffer_fill: usize,
    pub dred_offset: i32,
    pub latent_offset: usize,
    pub last_extra_dred_offset: usize,
    pub latents_buffer: Vec<f32>,
    pub latents_buffer_fill: usize,
    pub state_buffer: Vec<f32>,
    pub resample_mem: [f32; 9],
}

/// Initialize DRED encoder state.
pub fn dred_encoder_init(model: RdovaeEnc, lpcnet_enc_state: LpcnetEncState) -> DredEnc {
    let rdovae_enc = rdovae_enc_state_init(&model);
    let latent_dim = model.latent_dim;
    let state_dim = model.state_dim;
    DredEnc {
        model,
        lpcnet_enc_state,
        rdovae_enc,
        loaded: true,
        input_buffer: [0.0; 2 * DRED_DFRAME_SIZE],
        input_buffer_fill: 0,
        dred_offset: 0,
        latent_offset: 0,
        last_extra_dred_offset: 0,
        latents_buffer: vec![0.0; DRED_MAX_FRAMES * latent_dim],
        latents_buffer_fill: 0,
        state_buffer: vec![0.0; DRED_MAX_FRAMES * state_dim],
        resample_mem: [0.0; 9],
    }
}

/// Reset encoder state for a new stream.
pub fn dred_encoder_reset(enc: &mut DredEnc) {
    enc.input_buffer = [0.0; 2 * DRED_DFRAME_SIZE];
    enc.input_buffer_fill = 0;
    enc.dred_offset = 0;
    enc.latent_offset = 0;
    enc.last_extra_dred_offset = 0;
    enc.latents_buffer_fill = 0;
    enc.resample_mem = [0.0; 9];
    enc.rdovae_enc = rdovae_enc_state_init(&enc.model);
}

/// Compute DRED latents from PCM input.
/// Matches C `dred_compute_latents` from dred_encoder.c.
pub fn dred_compute_latents(
    enc: &mut DredEnc,
    pcm: &[f32],
    frame_size: usize,
    _extra_delay: usize,
) {
    let latent_dim = enc.model.latent_dim;
    let state_dim = enc.model.state_dim;
    let mut remaining = frame_size;
    let mut pcm_offset = 0;

    while remaining > 0 {
        let copy_len = remaining.min(DRED_DFRAME_SIZE * 2 - enc.input_buffer_fill);
        enc.input_buffer[enc.input_buffer_fill..enc.input_buffer_fill + copy_len]
            .copy_from_slice(&pcm[pcm_offset..pcm_offset + copy_len]);
        enc.input_buffer_fill += copy_len;
        pcm_offset += copy_len;
        remaining -= copy_len;

        if enc.input_buffer_fill >= DRED_DFRAME_SIZE {
            let mut dframe_features = [0.0f32; 2 * NB_TOTAL_FEATURES];

            let mut x = [0.0f32; DRED_FRAME_SIZE];
            x.copy_from_slice(&enc.input_buffer[..DRED_FRAME_SIZE]);
            let x_copy = x;
            preemphasis(
                &mut x,
                &mut enc.lpcnet_enc_state.mem_preemph,
                &x_copy,
                PREEMPHASIS,
                DRED_FRAME_SIZE,
            );
            compute_frame_features(&mut enc.lpcnet_enc_state, &x);
            dframe_features[..NB_TOTAL_FEATURES].copy_from_slice(&enc.lpcnet_enc_state.features);

            x.copy_from_slice(&enc.input_buffer[DRED_FRAME_SIZE..2 * DRED_FRAME_SIZE]);
            let x_copy = x;
            preemphasis(
                &mut x,
                &mut enc.lpcnet_enc_state.mem_preemph,
                &x_copy,
                PREEMPHASIS,
                DRED_FRAME_SIZE,
            );
            compute_frame_features(&mut enc.lpcnet_enc_state, &x);
            dframe_features[NB_TOTAL_FEATURES..].copy_from_slice(&enc.lpcnet_enc_state.features);

            let lat_start = enc.latents_buffer_fill * latent_dim;
            let state_start = enc.latents_buffer_fill * state_dim;
            if lat_start + latent_dim <= enc.latents_buffer.len()
                && state_start + state_dim <= enc.state_buffer.len()
            {
                dred_rdovae_encode_dframe(
                    &mut enc.rdovae_enc,
                    &enc.model,
                    &mut enc.latents_buffer[lat_start..lat_start + latent_dim],
                    &mut enc.state_buffer[state_start..state_start + state_dim],
                    &dframe_features,
                );
                enc.latents_buffer_fill += 1;
            }

            let dframe_samples = DRED_DFRAME_SIZE;
            enc.input_buffer.copy_within(dframe_samples.., 0);
            enc.input_buffer_fill -= dframe_samples;
        }
    }
}

/// Quantize and entropy-encode a latent vector.
/// Matches C `dred_encode_latents` from dred_encoder.c.
fn dred_encode_latents(
    ec: &mut EcCtx,
    x: &[f32],
    scale: &[u8],
    dzone: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) {
    const MAX_DIM: usize = 256;
    debug_assert!(dim <= MAX_DIM, "latent dim {dim} exceeds MAX_DIM {MAX_DIM}");
    let eps = 0.1f32;

    let mut xq = [0.0f32; MAX_DIM];
    let mut delta = [0.0f32; MAX_DIM];
    let mut deadzone = [0.0f32; MAX_DIM];

    for i in 0..dim {
        delta[i] = dzone[i] as f32 * (1.0 / 256.0);
        xq[i] = x[i] * scale[i] as f32 * (1.0 / 256.0);
        deadzone[i] = xq[i] / (delta[i] + eps);
    }
    compute_activation(&mut deadzone[..dim], Activation::Tanh);
    for i in 0..dim {
        xq[i] -= delta[i] * deadzone[i];
    }

    for i in 0..dim {
        let q = if r[i] == 0 || p0[i] == 255 {
            0
        } else {
            (0.5 + xq[i]).floor() as i32
        };
        if r[i] != 0 && p0[i] != 255 {
            ec.laplace_encode_p0(q, (p0[i] as u16) << 7, (r[i] as u16) << 7);
        }
    }
}

/// Check if voice is active at a given offset in the activity memory.
/// Matches C `dred_voice_active` from dred_encoder.c.
/// `activity_mem` is organized as 16 bytes per offset (8*offset + 0..15).
fn dred_voice_active(activity_mem: &[u8], offset: usize) -> bool {
    let base = 8 * offset;
    for i in 0..16 {
        if base + i < activity_mem.len() && activity_mem[base + i] == 1 {
            return true;
        }
    }
    false
}

/// Encode DRED data into a byte buffer as an Opus extension 126 payload.
/// Matches C `dred_encode_silk_frame` from dred_encoder.c.
///
/// `activity_mem` provides per-frame voice activity flags used to skip
/// encoding silent frames (saving bitrate) and trim trailing silence.
///
/// Returns the number of bytes written, or 0 if encoding failed or
/// no voiced DRED data is available.
pub fn dred_encode_silk_frame(
    enc: &mut DredEnc,
    buf: &mut [u8],
    max_chunks: usize,
    max_bytes: usize,
    q0: i32,
    dq: i32,
    qmax: i32,
    activity_mem: &[u8],
    stats: &DredStats,
) -> usize {
    let state_dim = stats.state_dim;
    let latent_dim = stats.latent_dim;

    if max_bytes == 0 {
        return 0;
    }

    // Voice activity gating: skip leading silent frames, matching C lines 295-307.
    let mut latent_offset = enc.latent_offset;
    let mut extra_dred_offset = 0usize;
    let mut delayed_dred = false;

    // Delay new DRED data when just out of silence (C lines 298-302).
    if !activity_mem.is_empty() && activity_mem[0] != 0 && enc.last_extra_dred_offset > 0 {
        latent_offset = enc.last_extra_dred_offset;
        delayed_dred = true;
        enc.last_extra_dred_offset = 0;
    }

    // Skip forward past silent frames (C lines 303-306).
    while latent_offset < enc.latents_buffer_fill && !dred_voice_active(activity_mem, latent_offset)
    {
        latent_offset += 1;
        extra_dred_offset += 1;
    }
    if !delayed_dred {
        enc.last_extra_dred_offset = extra_dred_offset;
    }

    if latent_offset >= enc.latents_buffer_fill {
        return 0;
    }

    // Entropy coding of state and latents (C lines 310-331).
    let mut ec = EcCtx::enc_init(max_bytes as u32);
    ec.enc_uint(q0 as u32, 16);
    ec.enc_uint(dq as u32, 8);

    let total_offset = 16i32 - (enc.dred_offset - extra_dred_offset as i32 * 8);
    debug_assert!(total_offset >= 0);
    let total_offset = total_offset as u32;
    if total_offset > 31 {
        ec.enc_uint(1, 2);
        ec.enc_uint(total_offset >> 5, 256);
        ec.enc_uint(total_offset & 31, 32);
    } else {
        ec.enc_uint(0, 2);
        ec.enc_uint(total_offset, 32);
    }

    debug_assert!(qmax >= q0);
    if q0 < 14 && dq > 0 {
        debug_assert!(qmax > q0);
        let nvals = 15 - (q0 + 1);
        let (fl, fh) = if qmax >= 15 {
            (0u32, nvals as u32)
        } else {
            let s = (nvals + qmax - (q0 + 1)) as u32;
            (s, s + 1)
        };
        ec.encode(fl, fh, (2 * nvals) as u32);
    }

    // Encode initial state
    let state_qoffset = (q0 as usize) * state_dim;
    if state_qoffset + state_dim > stats.state_quant_scales.len() {
        return 0;
    }
    dred_encode_latents(
        &mut ec,
        &enc.state_buffer[latent_offset * state_dim..latent_offset * state_dim + state_dim],
        &stats.state_quant_scales[state_qoffset..state_qoffset + state_dim],
        &stats.state_dead_zone[state_qoffset..state_qoffset + state_dim],
        &stats.state_r[state_qoffset..state_qoffset + state_dim],
        &stats.state_p0[state_qoffset..state_qoffset + state_dim],
        state_dim,
    );

    if ec.tell() > 8 * max_bytes as i32 {
        return 0;
    }

    // Encode latent pairs with voice activity tracking (C lines 346-373).
    let mut ec_bak = ec.save_state();
    let mut dred_encoded = 0usize;
    let mut prev_active = false;

    let max_pairs = (2 * max_chunks).min(enc.latents_buffer_fill.saturating_sub(latent_offset + 1));
    let mut i = 0;
    while i < max_pairs {
        let q_level = compute_quantizer(q0, dq, qmax, (i / 2) as i32);
        let offset = (q_level as usize) * latent_dim;
        if offset + latent_dim > stats.latent_quant_scales.len() {
            break;
        }

        let lat_start = (i + latent_offset) * latent_dim;
        if lat_start + latent_dim > enc.latents_buffer.len() {
            break;
        }

        dred_encode_latents(
            &mut ec,
            &enc.latents_buffer[lat_start..lat_start + latent_dim],
            &stats.latent_quant_scales[offset..offset + latent_dim],
            &stats.latent_dead_zone[offset..offset + latent_dim],
            &stats.latent_r[offset..offset + latent_dim],
            &stats.latent_p0[offset..offset + latent_dim],
            latent_dim,
        );

        if ec.tell() > 8 * max_bytes as i32 {
            if i == 0 {
                return 0;
            }
            break;
        }

        // Only checkpoint when voice is active (or was active in previous pair).
        // This trims trailing silence from the bitstream (C lines 367-372).
        let active = dred_voice_active(activity_mem, i + latent_offset);
        if active || prev_active {
            ec_bak = ec.save_state();
            dred_encoded = i + 2;
        }
        prev_active = active;

        i += 2;
    }

    // Avoid sending empty or near-empty DRED packets (C line 375).
    if dred_encoded == 0 || (dred_encoded <= 2 && extra_dred_offset > 0) {
        return 0;
    }

    ec.restore_state(&ec_bak);
    let ec_buffer_fill = ((ec.tell() + 7) / 8) as u32;
    ec.enc_shrink(ec_buffer_fill);
    ec.enc_done();

    let nbytes = ec_buffer_fill as usize;
    buf[..nbytes].copy_from_slice(&ec.buf[..nbytes]);
    nbytes
}
