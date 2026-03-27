pub mod onnx;

use std::path::PathBuf;
use std::sync::OnceLock;

use anyhow::Result;

pub trait EmbeddingBackend {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;
    fn dimensions(&self) -> usize;
}

/// Cached result of the ONNX Runtime dylib search.
static ORT_DYLIB: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Ensure `ORT_DYLIB_PATH` is set so `ort` with `load-dynamic` can find the
/// ONNX Runtime shared library.  Returns the resolved path on success, or
/// `None` if the library cannot be found anywhere.
///
/// The result is cached for the process lifetime.
pub fn ensure_ort_dylib() -> Option<PathBuf> {
    ORT_DYLIB
        .get_or_init(|| {
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
                    // SAFETY: called during OnceLock init, before any threads
                    // or ONNX Runtime loading.
                    unsafe { std::env::set_var("ORT_DYLIB_PATH", candidate) };
                    return Some(path);
                }
            }

            None
        })
        .clone()
}

/// Try to load the ONNX embedding backend from `model_dir`.
///
/// Returns `None` if ONNX Runtime dylib or the model files are unavailable.
pub fn try_load_backend(model_dir: &std::path::Path) -> Option<onnx::OnnxBackend> {
    ensure_ort_dylib()?;

    if !model_dir.join("model.onnx").exists() {
        return None;
    }

    onnx::OnnxBackend::load(model_dir).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_load_backend_returns_none_for_missing_model() {
        let tmp = std::env::temp_dir().join("kiok-test-empty-model-dir");
        let _ = std::fs::create_dir_all(&tmp);

        // No model.onnx in the directory — should return None.
        let result = try_load_backend(&tmp);
        assert!(result.is_none());

        let _ = std::fs::remove_dir(&tmp);
    }

    #[test]
    fn test_ensure_ort_dylib_returns_some_when_installed() {
        // This test validates the runtime environment.
        // If onnxruntime is installed, ensure_ort_dylib should find it.
        // If not installed, it should return None (not panic).
        let result = ensure_ort_dylib();

        if std::path::Path::new("/opt/homebrew/lib/libonnxruntime.dylib").exists() {
            assert!(result.is_some(), "should find Homebrew onnxruntime");
        }
        // Either way, the function must not panic.
    }
}
