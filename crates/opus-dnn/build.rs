//! Build script for opus-dnn: downloads model weight data from xiph.org.
//!
//! The weight tarball is downloaded once to `model-data/` (gitignored),
//! verified via SHA256, and extracted. The extracted C data files and
//! .pth model files become available for:
//!
//! - The opus-ffi C build (cmake compiles the *_data.c files)
//! - Rust runtime loading via parse_weights + model init functions
//! - FFI comparison tests that need both C and Rust models loaded

use std::collections::HashMap;
use std::env;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

include!("tools/parse_c_weights.rs");

const MODEL_HASH: &str = "a5177ec6fb7d15058e99e57029746100121f68e4890b1467d4094aa336b6013e";
const MODEL_URL: &str = "https://media.xiph.org/opus/models";

fn main() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let model_data_dir = manifest_dir.join("model-data");
    let tarball_name = format!("opus_data-{MODEL_HASH}.tar.gz");
    let tarball_path = model_data_dir.join(&tarball_name);

    // Marker file to avoid re-extracting on every build.
    let extracted_marker = model_data_dir.join(".extracted");

    if extracted_marker.exists() {
        // Already downloaded and extracted — just ensure combined blob exists.
        generate_combined_blob(&model_data_dir.join("blobs"));
        println!("cargo:rerun-if-changed=build.rs");
        println!("cargo:rerun-if-changed=tools/parse_c_weights.rs");
        return;
    }

    fs::create_dir_all(&model_data_dir).expect("failed to create model-data directory");

    // Download if not cached.
    if !tarball_path.exists() {
        let url = format!("{MODEL_URL}/{tarball_name}");
        eprintln!("Downloading DNN model weights from {url}...");

        let status = if which_exists("wget") {
            Command::new("wget")
                .args(["-q", "-O"])
                .arg(&tarball_path)
                .arg(&url)
                .status()
        } else {
            Command::new("curl")
                .args(["-s", "-o"])
                .arg(&tarball_path)
                .arg(&url)
                .status()
        };

        match status {
            Ok(s) if s.success() => eprintln!("Download complete."),
            Ok(s) => {
                let _ = fs::remove_file(&tarball_path);
                panic!("Download failed with exit code: {s}");
            }
            Err(e) => {
                let _ = fs::remove_file(&tarball_path);
                panic!("Failed to run download command: {e}");
            }
        }
    }

    // Verify SHA256.
    if let Some(sha_cmd) = find_sha256_command() {
        eprintln!("Verifying checksum...");
        let output = Command::new(&sha_cmd.0)
            .args(&sha_cmd.1)
            .arg(&tarball_path)
            .output()
            .expect("failed to run sha256 command");
        let stdout = String::from_utf8_lossy(&output.stdout);
        let computed_hash = stdout.split_whitespace().next().unwrap_or("");
        if computed_hash != MODEL_HASH {
            let _ = fs::remove_file(&tarball_path);
            panic!(
                "SHA256 mismatch!\n  expected: {MODEL_HASH}\n  computed: {computed_hash}\n\
                 The tarball may be corrupted. Deleted it — re-run to retry."
            );
        }
        eprintln!("Checksum verified.");
    } else {
        eprintln!("Warning: no sha256sum/shasum found, skipping checksum verification.");
    }

    // Extract tarball into model-data/.
    eprintln!("Extracting model data...");
    let status = Command::new("tar")
        .args(["xzf"])
        .arg(&tarball_path)
        .current_dir(&model_data_dir)
        .status()
        .expect("failed to run tar");
    if !status.success() {
        panic!("tar extraction failed");
    }

    // Also copy the C data files into the opus-ffi C submodule's dnn/ directory
    // so that cmake can find them when building with OPUS_DRED=ON / OPUS_OSCE=ON.
    let c_dnn_dir = manifest_dir.join("../opus-ffi/opus-c/dnn");
    if c_dnn_dir.exists() {
        let extracted_dnn = model_data_dir.join("dnn");
        if extracted_dnn.exists() {
            copy_data_files(&extracted_dnn, &c_dnn_dir);
            eprintln!("Copied C data files to opus-ffi/opus-c/dnn/");
        }
    }

    // Parse C data files and generate binary weight blobs (pure Rust, no C compiler needed).
    let blobs_dir = model_data_dir.join("blobs");
    if !blobs_dir.join("fargan.bin").exists() || !blobs_dir.join("opus_dnn.blob").exists() {
        fs::create_dir_all(&blobs_dir).expect("failed to create blobs dir");
        let dnn_dir = model_data_dir.join("dnn");

        if dnn_dir.exists() {
            let models = [
                ("pitchdnn_data.c", "pitchdnn.bin", "pitchdnn_arrays"),
                ("plc_data.c", "plcmodel.bin", "plcmodel_arrays"),
                ("fargan_data.c", "fargan.bin", "fargan_arrays"),
                ("lace_data.c", "lace.bin", "lacelayers_arrays"),
                ("nolace_data.c", "nolace.bin", "nolacelayers_arrays"),
                (
                    "dred_rdovae_enc_data.c",
                    "rdovae_enc.bin",
                    "rdovaeenc_arrays",
                ),
                (
                    "dred_rdovae_dec_data.c",
                    "rdovae_dec.bin",
                    "rdovaedec_arrays",
                ),
                ("bbwenet_data.c", "bbwenet.bin", "bbwenetlayers_arrays"),
            ];

            eprintln!("Generating binary weight blobs from C data files (pure Rust)...");
            for (c_file, bin_file, table_name) in &models {
                let c_path = dnn_dir.join(c_file);
                let bin_path = blobs_dir.join(bin_file);
                if c_path.exists() {
                    match convert_c_to_blob(
                        c_path.to_str().unwrap(),
                        bin_path.to_str().unwrap(),
                        table_name,
                    ) {
                        Ok(count) => eprintln!("  {} -> {} ({} arrays)", c_file, bin_file, count),
                        Err(e) => eprintln!("  Warning: failed to convert {}: {}", c_file, e),
                    }
                }
            }

            generate_combined_blob(&blobs_dir);
        }
    }

    // Write marker.
    fs::write(&extracted_marker, MODEL_HASH).expect("failed to write marker");
    eprintln!("Model data ready at {}", model_data_dir.display());

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=tools/parse_c_weights.rs");
}

/// Concatenate individual model blobs into a single `opus_dnn.blob`.
///
/// The blob format is self-describing — each weight array record has a 64-byte
/// header with `"wght"` magic and a name field — so concatenation produces a
/// valid multi-model blob. Applications pass this single file to `load_dnn()`.
fn generate_combined_blob(blobs_dir: &Path) {
    let combined_path = blobs_dir.join("opus_dnn.blob");
    if combined_path.exists() {
        return;
    }
    let blob_order = [
        "rdovae_enc.bin",
        "rdovae_dec.bin",
        "pitchdnn.bin",
        "plcmodel.bin",
        "fargan.bin",
        "lace.bin",
        "nolace.bin",
        "bbwenet.bin",
    ];
    let mut combined = Vec::new();
    for name in &blob_order {
        let path = blobs_dir.join(name);
        if let Ok(data) = fs::read(&path) {
            combined.extend_from_slice(&data);
        }
    }
    if !combined.is_empty() {
        fs::write(&combined_path, &combined).expect("failed to write opus_dnn.blob");
        eprintln!(
            "  Combined blob: opus_dnn.blob ({:.1} MB)",
            combined.len() as f64 / 1_048_576.0
        );
    }
}

fn which_exists(cmd: &str) -> bool {
    Command::new("which")
        .arg(cmd)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

fn find_sha256_command() -> Option<(String, Vec<String>)> {
    if which_exists("sha256sum") {
        Some(("sha256sum".into(), vec![]))
    } else if which_exists("shasum") {
        Some(("shasum".into(), vec!["-a".into(), "256".into()]))
    } else {
        None
    }
}

/// Copy *_data.c, *_data.h, and *_constants.h files from extracted tarball to C dnn/ dir.
fn copy_data_files(src_dir: &Path, dst_dir: &Path) {
    if let Ok(entries) = fs::read_dir(src_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if (name_str.ends_with("_data.c")
                || name_str.ends_with("_data.h")
                || name_str.ends_with("_constants.h")
                || name_str.ends_with("_stats_data.c")
                || name_str.ends_with("_stats_data.h"))
                && entry.file_type().map(|t| t.is_file()).unwrap_or(false)
            {
                let dst = dst_dir.join(&name);
                if let Err(e) = fs::copy(entry.path(), &dst) {
                    eprintln!("Warning: failed to copy {}: {e}", name_str);
                }
            }
        }
    }
}
