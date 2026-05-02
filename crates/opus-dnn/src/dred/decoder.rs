use opus_range_coder::EcCtx;

use super::coding::compute_quantizer;
use super::*;

/// DRED quantization statistics (loaded from dred_rdovae_stats_data).
/// These tables are indexed by `q_level * dim` for per-element decoding/encoding.
pub struct DredStats {
    pub state_quant_scales: Vec<u8>,
    pub state_dead_zone: Vec<u8>,
    pub state_r: Vec<u8>,
    pub state_p0: Vec<u8>,
    pub latent_quant_scales: Vec<u8>,
    pub latent_dead_zone: Vec<u8>,
    pub latent_r: Vec<u8>,
    pub latent_p0: Vec<u8>,
    pub state_dim: usize,
    pub latent_dim: usize,
}

/// DRED decoded state. Matches C `OpusDRED` from dred_decoder.h.
pub struct OpusDred {
    pub fec_features: Vec<f32>,
    pub state: Vec<f32>,
    pub latents: Vec<f32>,
    pub nb_latents: usize,
    pub process_stage: i32,
    pub dred_offset: i32,
    pub latent_dim: usize,
    pub state_dim: usize,
}

impl OpusDred {
    pub fn new(latent_dim: usize, state_dim: usize) -> Self {
        OpusDred {
            fec_features: vec![0.0; 2 * DRED_NUM_REDUNDANCY_FRAMES * DRED_NUM_FEATURES],
            state: vec![0.0; state_dim],
            latents: vec![0.0; (DRED_NUM_REDUNDANCY_FRAMES / 2) * (latent_dim + 1)],
            nb_latents: 0,
            process_stage: 0,
            dred_offset: 0,
            latent_dim,
            state_dim,
        }
    }
}

/// Decode a latent vector from the range coder using quantization stats.
/// Matches C `dred_decode_latents` from dred_decoder.c.
fn dred_decode_latents(
    ec: &mut EcCtx,
    x: &mut [f32],
    scale: &[u8],
    r: &[u8],
    p0: &[u8],
    dim: usize,
) {
    for i in 0..dim {
        let q = if r[i] == 0 || p0[i] == 255 {
            0
        } else {
            ec.laplace_decode_p0((p0[i] as u16) << 7, (r[i] as u16) << 7)
        };
        let s = if scale[i] == 0 { 1 } else { scale[i] as i32 };
        x[i] = q as f32 * 256.0 / s as f32;
    }
}

/// Decode DRED extension payload from entropy-coded bytes.
/// Matches C `dred_ec_decode` from dred_decoder.c.
///
/// `stats` provides the quantization statistics tables.
/// Returns the number of decoded latent pairs.
pub fn dred_ec_decode(
    dec: &mut OpusDred,
    bytes: &[u8],
    num_bytes: usize,
    min_feature_frames: usize,
    dred_frame_offset: i32,
    stats: &DredStats,
) -> usize {
    if num_bytes == 0 || stats.state_dim == 0 || stats.latent_dim == 0 {
        dec.nb_latents = 0;
        dec.process_stage = 0;
        return 0;
    }

    let state_dim = stats.state_dim;
    let latent_dim = stats.latent_dim;
    let mut ec = EcCtx::dec_init(&bytes[..num_bytes]);

    let q0 = ec.dec_uint(16) as i32;
    let dq = ec.dec_uint(8) as i32;
    let extra_offset = if ec.dec_uint(2) != 0 {
        32 * ec.dec_uint(256) as i32
    } else {
        0
    };
    dec.dred_offset = 16 - ec.dec_uint(32) as i32 - extra_offset + dred_frame_offset;

    let mut qmax = 15i32;
    if q0 < 14 && dq > 0 {
        let nvals = 15 - (q0 + 1);
        let ft = (2 * nvals) as u32;
        let s = ec.decode(ft) as i32;
        if s >= nvals {
            qmax = q0 + (s - nvals) + 1;
            ec.dec_update(s as u32, (s + 1) as u32, ft);
        } else {
            ec.dec_update(0, nvals as u32, ft);
        }
    }

    // Decode initial state
    let state_qoffset = (q0 as usize) * state_dim;
    if state_qoffset + state_dim <= stats.state_quant_scales.len() {
        dred_decode_latents(
            &mut ec,
            &mut dec.state[..state_dim],
            &stats.state_quant_scales[state_qoffset..state_qoffset + state_dim],
            &stats.state_r[state_qoffset..state_qoffset + state_dim],
            &stats.state_p0[state_qoffset..state_qoffset + state_dim],
            state_dim,
        );
    }

    // Decode latent pairs (newest to oldest)
    let max_i = DRED_NUM_REDUNDANCY_FRAMES.min(min_feature_frames.div_ceil(2));
    let mut i = 0;
    while i < max_i {
        if (8 * num_bytes as i32) - ec.tell() <= 7 {
            break;
        }
        let q_level = compute_quantizer(q0, dq, qmax, (i / 2) as i32);
        let offset = (q_level as usize) * latent_dim;
        if offset + latent_dim <= stats.latent_quant_scales.len() {
            let lat_start = (i / 2) * (latent_dim + 1);
            dred_decode_latents(
                &mut ec,
                &mut dec.latents[lat_start..lat_start + latent_dim],
                &stats.latent_quant_scales[offset..offset + latent_dim],
                &stats.latent_r[offset..offset + latent_dim],
                &stats.latent_p0[offset..offset + latent_dim],
                latent_dim,
            );
            dec.latents[lat_start + latent_dim] = q_level as f32 * 0.125 - 1.0;
        }
        i += 2;
    }
    dec.process_stage = 1;
    dec.nb_latents = i / 2;
    i / 2
}
