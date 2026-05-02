# CLAUDE.md — opus-rust

Pure Rust implementation of the Opus audio codec (encoder + decoder), ported from the C reference implementation (libopus/xiph).

## Project structure

Single-crate layout. Codec subsystems live as modules of the root `opus` crate; `opus-ffi` is the only sibling crate and exists solely for cross-validation against C libopus.

```
src/
  range_coder/   Arithmetic entropy coder shared by SILK and CELT
  silk/          SILK codec (narrowband speech) — decoder + float encoder
  celt/          CELT codec (broadband audio) — decoder + encoder
  dnn/           DRED, FARGAN, PitchDNN, OSCE (gated by dnn-* features)
  encoder.rs decoder.rs multistream.rs repacketizer.rs ...   Public facade
tests/           Integration tests (28 .rs files + common/mod.rs)
benches/         Criterion benchmarks
build.rs         Downloads DNN model weights when a dnn-* feature is enabled
model-data/      Build output — gitignored
crates/
  opus-ffi/      C libopus FFI bindings (unsafe, for testing/benchmarking only)
```

Workspace resolver is `3`. Edition is `2024`. The root `Cargo.toml` is both `[workspace]` (containing `crates/opus-ffi`) and `[package] name = "opus"`.

The `opus-ffi` crate vendors the C reference via a git submodule at `crates/opus-ffi/opus-c` (xiph/opus.git) and builds it with cmake.

## Safety policy

The root `opus` crate **forbids unsafe code** via `unsafe-code = "forbid"` in `[lints.rust]`. All folded modules (`range_coder`, `silk`, `celt`, `dnn`) are subject to this lint — do not introduce `unsafe` blocks in library code.

The `opus-ffi` crate is the sole exception — it requires `unsafe` for C FFI bindings. It is a dev-dependency of `opus`, not a runtime dependency.

## Relationship to C reference

This codebase is a faithful port of C libopus. When the Rust implementation diverges from C in output:

1. **The C reference is the ground truth.** Investigate divergences by comparing against C behavior, not by adjusting thresholds to hide them.
2. **Use the FFI layer to diagnose.** The `opus-ffi` crate wraps C libopus and exposes both high-level (encoder/decoder) and low-level (SILK internals, CELT DSP) functions for side-by-side comparison.
3. **Small floating-point divergences are expected** due to operation ordering differences between C and Rust. Threshold-based tests accommodate this, but thresholds should be as tight as possible and documented when loosened.
4. **Watch for FFI shim bugs masquerading as algorithm divergences.** If a C-vs-Rust comparison fails with absurd values (e.g. `1e36`), suspect uninitialized memory or a buffer-history mismatch in the FFI shim before suspecting the Rust port. Example: `c_celt_fir` originally passed `x.as_ptr()` directly to a C function that reads `ord` samples of caller-supplied history — fixed by zero-padding inside the shim.

## Features

| Feature | Effect |
| --- | --- |
| `dnn-deep-plc` | Compile DNN module + Deep PLC (FARGAN/PitchDNN) integration |
| `dnn-dred` | Compile DRED encode/decode; implies `dnn-deep-plc` |
| `dnn-osce` | Compile OSCE (LACE/NoLACE) post-filter |
| `dnn` | Umbrella — enables all three DNN sub-features |

The `src/dnn` module is gated on `any(feature = "dnn-deep-plc", "dnn-dred", "dnn-osce")`. The encoder/decoder DNN integration paths (`src/dnn_decoder.rs`, `src/dnn_silk_bridge.rs`, `src/dnn_types.rs`, plus `#[cfg(feature = "dnn")]` blocks in `encoder.rs`/`decoder.rs`) are gated on the umbrella `dnn` feature. Selecting only one sub-feature compiles the lower-level DNN building blocks but not the high-level integration.

## Testing

Integration tests live in root `tests/`. Tests for SILK and CELT subsystems live alongside the rest because they need the FFI layer (`opus-ffi` is a dev-dependency) for C-vs-Rust comparison.

**Where to put new tests:** all new integration tests go in `tests/`. There is no longer a per-sub-crate test directory.

### Test categories

- **C-vs-Rust correctness** (`correctness_vs_c.rs`): Encode/decode comparison between C libopus and Rust opus at multiple configurations. The primary regression gate.
- **Cross-validation** (`cross_validate.rs`): Decode pre-generated C test vectors and compare PCM output.
- **Component tests** (`celt_*.rs`, `silk_*.rs`, `flp_*.rs`): Exercise individual subsystems (FFT, MDCT, pitch analysis, noise shaping, gain quantization, etc.).
- **FFI layer tests** (`ffi_layer1_tests.rs`, `a2nlsf_comparison.rs`): Leaf-function comparison between C and Rust implementations.
- **DNN tests** (`dnn_*.rs`): Each gated with `#![cfg(any(feature = "dnn-dred", "dnn-osce", "dnn-deep-plc"))]` so they compile to a no-op when no DNN feature is selected.

### Running tests

```bash
# Full test suite (requires C libopus submodule initialized + DNN weights downloaded)
cargo test --all-features

# Library-only (skip DNN tests)
cargo test

# Specific test file
cargo test --test correctness_vs_c -- --nocapture

# Filter by test name
cargo test silk_pitch.*10ms -- --nocapture
```

### Test vectors

Pre-generated test vectors are in `crates/opus-ffi/tests/vectors/`. The C generators are:

- `crates/opus-ffi/tests/gen_test_vectors.c` — CELT, SILK, and hybrid packets with reference PCM
- `crates/opus-ffi/tests/gen_ms_test_vectors.c` — multistream surround test vectors

To regenerate, compile the C generators against the vendored libopus and run them.

### Writing new tests

- Place shared utilities (signal generation, comparison helpers) in `tests/common/mod.rs`.
- Parameterize tests over frame sizes and configurations rather than duplicating test functions — see `correctness_vs_c.rs` `TestConfig` pattern.
- When a C-vs-Rust comparison fails, investigate the root cause rather than loosening thresholds. If a threshold must be loosened, add a comment explaining why.

## Benchmarks

Criterion benchmarks live in `benches/`:

- `encode_decode.rs` — Rust encoder/decoder throughput
- `c_reference.rs` — C libopus throughput (via FFI) for comparison
- `celt_internal.rs` — CELT subsystem microbenchmarks
- `dnn_bench.rs` — DNN layer microbenchmarks (requires `--features dnn`)

```bash
cargo bench
cargo bench --features dnn   # include dnn_bench
```

## Code style

### Formatting

Use default `rustfmt` (no custom config):

```bash
cargo fmt --all
```

### Linting

The root `opus` crate and `opus-ffi` configure clippy via `[lints.clippy]` with `too-many-arguments = "allow"` (the codec port necessarily has large function signatures matching C).

```bash
cargo clippy --workspace --tests --benches --all-features
```

### Conventions

- **Match C structure.** Function and variable names follow the C reference where possible to ease cross-referencing. Do not rename for Rust idioms if it would make the C mapping unclear.
- **Avoid premature abstraction.** The codec code is arithmetic-heavy and maps closely to the C reference. Prefer direct translation over Rust-idiomatic wrappers that obscure the algorithm.
- **Idiomatic loops are fine.** Style lints like `needless_range_loop` should be fixed (use `iter().enumerate()`, `iter_mut().take(n).skip(m)`, etc.) rather than allowed crate-wide. The C-vs-Rust algorithm mapping remains clear with iterator forms; only the indexing form changes.
- **Type-safe enums** (`Application`, `Bandwidth`, `Mode`, `SampleRate`, `Channels`, etc.) replace C magic constants at the public API boundary.

## Build requirements

- Rust edition 2024 (nightly or recent stable)
- C compiler + cmake (for the `opus-ffi` crate / test infrastructure)
- Git submodule: run `git submodule update --init` to fetch the C libopus source
