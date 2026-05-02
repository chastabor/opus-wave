# Opus Rust

A pure-Rust implementation of the [Opus audio codec](https://github.com/xiph/opus)
(v1.6), covering SILK (speech), CELT (audio), and hybrid modes. All library
code is safe Rust — no `unsafe` blocks outside the optional FFI test crate.
The API enforces valid configurations through Rust enums.

## Features

| Feature | Description | Status |
| ------- | ----------- | ------ |
| SILK encoder/decoder | Narrowband–wideband speech (8–16 kHz) | Stable |
| CELT encoder/decoder | Fullband audio (48 kHz) | Stable |
| Hybrid mode | SILK + CELT combined | Stable |
| Multistream | Surround / multi-channel via repacketizer | Stable |
| **DRED** | Deep REDundancy — resilient encoding against packet loss | `dnn` feature |
| **Deep PLC** | FARGAN + PitchDNN packet-loss concealment | `dnn` feature |
| **OSCE** | LACE/NoLACE speech enhancement (post-filter) | `dnn` feature |


## Encoder/Decoder API Enums

| Enum | Variants | Replaces |
| ---- | -------- | -------- |
| Application | Voip, Audio, RestrictedLowDelay | OPUS_APPLICATION_* (2048/2049/2051) |
| Bandwidth | Narrowband, Mediumband, Wideband, Superwideband, Fullband | OPUS_BANDWIDTH_* (1101–1105) |
| Mode | SilkOnly, Hybrid, CeltOnly | MODE_* (1000–1002) |
| Signal | Auto, Voice, Music | OPUS_SIGNAL_* (-1000/3001/3002)/ OPUS_AUTO |
| Bitrate | Auto, Max, BitsPerSecond(i32) | OPUS_AUTO / OPUS_BITRATE_MAX / raw i32 |
| SampleRate | Hz8000, Hz12000, Hz16000, Hz24000, Hz48000 | raw i32 fs param |
| Channels | Mono, Stereo | raw i32 channels param |
| ForceChannels | Auto, Mono, Stereo | raw i32 (-1/1/2) |


## Quick Start

```rust
use opus::{OpusEncoder, OpusDecoder, SampleRate, Channels, Application};

// Encode
let mut enc = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)?;
let mut packet = vec![0u8; 4000];
let len = enc.encode_float(&pcm_input, 960, &mut packet, 4000)?;

// Decode
let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono)?;
let mut pcm_output = vec![0.0f32; 960];
let samples = dec.decode_float(Some(&packet[..len as usize]), &mut pcm_output, 960, false)?;
```


## DNN Features (optional)

Enable the `dnn` feature to access DRED, deep PLC, and OSCE:

```toml
[dependencies]
opus = { version = "2", features = ["dnn"] }
```

The `dnn` feature is an umbrella that enables three sub-features which can also
be selected individually for finer-grained builds:

- `dnn-deep-plc` — FARGAN + PitchDNN packet-loss concealment
- `dnn-dred` — Deep REDundancy encode/decode (implies `dnn-deep-plc`)
- `dnn-osce` — LACE/NoLACE post-filter

### Weight Blob

The DNN models require a runtime weight blob (~16 MB). On first build with any
`dnn-*` feature enabled, the root `build.rs` downloads the official weights
from xiph.org, converts them to the binary blob format, and produces a single
combined file at:

```
model-data/blobs/opus_dnn.blob
```

This is the same format as C libopus `OPUS_SET_DNN_BLOB`. The blob is
self-describing — each weight array carries a 64-byte header with its name
— so both encoder and decoder can load the same file and each will extract
only the weights it needs.

### Loading Weights

```rust
use opus::{OpusEncoder, OpusDecoder, SampleRate, Channels, Application};

// Load the combined weight blob (built automatically by build.rs)
let weights = std::fs::read("model-data/blobs/opus_dnn.blob")?;

// --- Encoder: DRED ---
let mut enc = OpusEncoder::new(SampleRate::Hz48000, Channels::Mono, Application::Voip)?;
enc.load_dnn(&weights)?;
enc.set_dred_duration(10); // 10 frames of deep redundancy per packet

// --- Decoder: DRED + deep PLC + OSCE ---
let mut dec = OpusDecoder::new(SampleRate::Hz48000, Channels::Mono)?;
dec.load_dnn(&weights)?;
// DRED extensions are parsed automatically from incoming packets.
// Deep PLC activates on packet loss. OSCE enhances SILK output at 16 kHz.
```

For embedded or no-std environments you can embed the blob at compile time:

```rust
const WEIGHTS: &[u8] = include_bytes!("path/to/opus_dnn.blob");
enc.load_dnn(WEIGHTS)?;
```

### How It Works

| Component | When it runs | What it does |
| --------- | ------------ | ------------ |
| **DRED encoder** | Every frame (when `dred_duration > 0`) | Extracts latent features via RDOVAE, appends as Opus extension 126 |
| **DRED decoder** | On packet arrival (if extension present) | Decodes latents, feeds FEC features to PLC |
| **Deep PLC** | On packet loss | FARGAN neural vocoder synthesizes concealment audio |
| **OSCE** | After SILK decode (16 kHz, 20 ms frames) | LACE or NoLACE post-filter enhances speech quality |

### API Reference

| Method | Available on | Description |
| ------ | ------------ | ----------- |
| `load_dnn(&[u8])` | Encoder, Decoder | Load DNN model weights from binary blob |
| `set_dred_duration(i32)` | Encoder | Set DRED redundancy frames (0 = disabled) |
| `dred_duration()` | Encoder | Query current DRED duration |
| `dnn_loaded()` | Encoder, Decoder | Check if DNN models are loaded |


## Tests and Benchmarks

```bash
# Correctness against C reference
cargo test --test correctness_vs_c

# Full library test suite (requires C submodule: git submodule update --init)
cargo test

# Including DNN tests (downloads ~16 MB of model weights on first build)
cargo test --all-features

# Benchmarks (Rust vs C reference)
cargo bench
cargo bench --features dnn   # include DNN microbenchmarks
```


## Project Structure

Single-crate layout. Codec subsystems live as modules under `src/`; `opus-ffi`
is the only sibling crate and exists solely for cross-validation against C
libopus.

```
src/
  range_coder/   Arithmetic entropy coder shared by SILK and CELT
  silk/          SILK codec — decoder + float encoder
  celt/          CELT codec — decoder + encoder
  dnn/           DRED, FARGAN, PitchDNN, OSCE (gated by dnn-* features)
  encoder.rs decoder.rs multistream.rs repacketizer.rs ...   Public facade
crates/
  opus-ffi/      C libopus FFI bindings (unsafe, dev-dependency for tests)
```


## License

Royalty-free — see [COPYING](/COPYING), based on the original xiph/opus license.

