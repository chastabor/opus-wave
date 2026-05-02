use std::env;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());

    // Check if DNN weight data files are present (placed by opus-dnn build.rs).
    let dnn_data_present = manifest_dir.join("opus-c/dnn/fargan_data.c").exists()
        && manifest_dir.join("opus-c/dnn/pitchdnn_data.c").exists()
        && manifest_dir.join("opus-c/dnn/plc_data.c").exists();

    let dred_flag = if dnn_data_present { "ON" } else { "OFF" };
    let osce_flag = if dnn_data_present { "ON" } else { "OFF" };

    if dnn_data_present {
        eprintln!("opus-ffi: DNN data files found, building with DRED+OSCE enabled");
    } else {
        eprintln!("opus-ffi: DNN data files not found, building without DNN features");
        eprintln!("  (run `cargo build -p opus-dnn` first to download model weights)");
    }

    // Build C libopus from the vendored submodule using cmake.
    let dst = cmake::Config::new("opus-c")
        .define("OPUS_BUILD_PROGRAMS", "OFF")
        .define("OPUS_BUILD_TESTING", "OFF")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("OPUS_DRED", dred_flag)
        .define("OPUS_OSCE", osce_flag)
        // Enable custom modes to expose FFT alloc/free and MDCT init/clear.
        .define("OPUS_CUSTOM_MODES", "ON")
        // Note: keeping float build (default) for compatibility with correctness tests.
        // With FIXED_POINT=ON, all SILK primitives (Burg, A2NLSF, NLSF encode) are
        // verified identical between C and Rust.
        .build();

    let lib_dir = dst.join("lib");
    let include_dir = dst.join("include").join("opus");

    // Tell cargo to link the static library.
    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=static=opus");
    println!("cargo:rustc-link-lib=m");

    // Compile wrapper.c (non-variadic CTL shims) and celt_wrapper.c
    // (CELT internal function shims for cross-validation).
    let mut build = cc::Build::new();
    build
        .file(manifest_dir.join("src/wrapper.c"))
        .file(manifest_dir.join("src/celt_wrapper.c"))
        .include(&include_dir)
        .include(manifest_dir.join("opus-c/include"))
        .include(manifest_dir.join("opus-c/celt"));

    // Add DNN wrapper when DNN is enabled.
    if dnn_data_present {
        build
            .file(manifest_dir.join("src/dnn_wrapper.c"))
            .file(manifest_dir.join("src/dnn_model_wrapper.c"))
            .file(manifest_dir.join("src/dnn_pitchdnn_layers.c"))
            .include(manifest_dir.join("opus-c")) // for celt/x86/x86cpu.h
            .include(manifest_dir.join("opus-c/dnn"))
            .include(manifest_dir.join("opus-c/silk")); // for structs.h
    }

    build.compile("opus_wrapper");
}
