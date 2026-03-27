pub mod onnx;

use anyhow::Result;
use std::path::PathBuf;

pub trait EmbeddingBackend {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

/// Ensure `ORT_DYLIB_PATH` is set so `ort` with `load-dynamic` can find the
/// ONNX Runtime shared library.  Returns the resolved path on success, or
/// `None` if the library cannot be found anywhere.
pub fn ensure_ort_dylib() -> Option<PathBuf> {
    // Already set by the user — trust it.
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        let path = PathBuf::from(&p);
        if path.exists() {
            return Some(path);
        }
    }

    // Search well-known locations (Homebrew on Apple Silicon / Intel).
    let candidates = [
        "/opt/homebrew/lib/libonnxruntime.dylib",
        "/usr/local/lib/libonnxruntime.dylib",
    ];

    for candidate in &candidates {
        let path = PathBuf::from(candidate);
        if path.exists() {
            // SAFETY: kiok is single-threaded at this point (called before
            // spawning any threads or loading ONNX Runtime).
            unsafe { std::env::set_var("ORT_DYLIB_PATH", candidate) };
            return Some(path);
        }
    }

    None
}
