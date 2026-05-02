# CLAUDE.md — opus-rust

Pure Rust implementation of the Opus audio codec (encoder + decoder), ported from the C reference implementation (libopus/xiph).

## Project structure

```
crates/
  opus/              Main facade — encoder, decoder, multistream, repacketizer
  opus-silk/         SILK codec (narrowband speech) — decoder + float encoder
  opus-celt/         CELT codec (broadband audio) — decoder + encoder
  opus-range-coder/  Arithmetic entropy coder shared by SILK and CELT
  opus-ffi/          C libopus FFI bindings (unsafe, for testing/benchmarking only)
```

Workspace resolver is `3`. All crates use edition `2024`.

The `opus-ffi` crate vendors the C reference via a git submodule at `crates/opus-ffi/opus-c` (xiph/opus.git) and builds it with cmake.

## Safety policy

All library crates (`opus`, `opus-silk`, `opus-celt`, `opus-range-coder`) **forbid unsafe code** via `unsafe-code = "forbid"` in their `[lints.rust]` Cargo.toml sections. Do not introduce `unsafe` blocks in library code.

The `opus-ffi` crate is the sole exception — it requires `unsafe` for C FFI bindings. This crate exists only for cross-validation and benchmarking; it is not a runtime dependency of the library.

## Relationship to C reference

This codebase is a faithful port of C libopus. When the Rust implementation diverges from C in output:

1. **The C reference is the ground truth.** Investigate divergences by comparing against C behavior, not by adjusting thresholds to hide them.
2. **Use the FFI layer to diagnose.** The `opus-ffi` crate wraps C libopus and exposes both high-level (encoder/decoder) and low-level (SILK internals, CELT DSP) functions for side-by-side comparison.
3. **Small floating-point divergences are expected** due to operation ordering differences between C and Rust. Threshold-based tests accommodate this, but thresholds should be as tight as possible and documented when loosened.

## Testing

Integration tests live in `crates/opus/tests/` (23 test files + `common/mod.rs` shared utilities). This includes tests for SILK and CELT subsystems — they are in the `opus` crate because they need access to the FFI layer (`opus-ffi` is a dev-dependency of `opus`) for C-vs-Rust comparison. Individual crates may have their own focused tests (e.g., `opus-range-coder/tests/roundtrip.rs`, `opus-celt/tests/encoder_roundtrip.rs`) for self-contained functionality that doesn't require FFI cross-validation.

**Where to put new tests:** If the test compares against C libopus or exercises cross-crate integration, put it in `crates/opus/tests/`. If it tests a single crate's logic in isolation, put it in that crate's `tests/` directory.

### Test categories

- **C-vs-Rust correctness** (`correctness_vs_c.rs`): Encode/decode comparison between C libopus and Rust opus at multiple configurations. The primary regression gate.
- **Cross-validation** (`cross_validate.rs`): Decode pre-generated C test vectors and compare PCM output.
- **Component tests** (`celt_*.rs`, `silk_*.rs`, `flp_*.rs`): Exercise individual subsystems (FFT, MDCT, pitch analysis, noise shaping, gain quantization, etc.).
- **FFI layer tests** (`ffi_layer1_tests.rs`, `a2nlsf_comparison.rs`): Leaf-function comparison between C and Rust implementations.

### Running tests

```bash
# Full test suite (requires C libopus submodule initialized)
cargo test --package opus

# Specific test file
cargo test --package opus --test correctness_vs_c -- --nocapture

# Filter by test name
cargo test --package opus silk_pitch.*10ms -- --nocapture
```

### Test vectors

Pre-generated test vectors are in `tests/vectors/`. The C generators are:

- `tests/gen_test_vectors.c` — CELT, SILK, and hybrid packets with reference PCM
- `tests/gen_ms_test_vectors.c` — multistream surround test vectors

To regenerate, compile the C generators against the vendored libopus and run them.

### Writing new tests

- Place shared utilities (signal generation, comparison helpers) in `tests/common/mod.rs`.
- Parameterize tests over frame sizes and configurations rather than duplicating test functions — see `correctness_vs_c.rs` `TestConfig` pattern.
- When a C-vs-Rust comparison fails, investigate the root cause rather than loosening thresholds. If a threshold must be loosened, add a comment explaining why.

## Benchmarks

Criterion benchmarks are in `crates/opus/benches/`:

- `encode_decode.rs` — Rust encoder/decoder throughput
- `c_reference.rs` — C libopus throughput (via FFI) for comparison
- `celt_internal.rs` — CELT subsystem microbenchmarks

```bash
cargo bench --package opus
```

## Code style

### Formatting

Use default `rustfmt` (no custom config):

```bash
cargo fmt --all
```

### Linting

All crates configure clippy via `[lints.clippy]` in Cargo.toml with `too-many-arguments = "allow"` (the codec port necessarily has large function signatures matching C).

```bash
cargo clippy --workspace --tests --benches
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
