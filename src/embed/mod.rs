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

/// Search well-known locations for the ONNX Runtime shared library and set
/// `ORT_DYLIB_PATH` so the `ort` crate can load it.  Returns the resolved
/// path on success, or `None` if the library cannot be found anywhere.
///
/// The result is cached for the process lifetime.
pub fn ensure_ort_dylib() -> Option<PathBuf> {
    ORT_DYLIB
        .get_or_init(|| {
            let candidates = ort_search_paths();

            for candidate in &candidates {
                if candidate.exists() {
                    // SAFETY: called before ONNX Runtime is loaded and before
                    // spawning worker threads (save runs single-threaded,
                    // embed/recall call this at startup).
                    unsafe { std::env::set_var("ORT_DYLIB_PATH", candidate) };
                    return Some(candidate.clone());
                }
            }

            None
        })
        .clone()
}

/// Return platform-specific search paths for the ONNX Runtime shared library.
fn ort_search_paths() -> Vec<PathBuf> {
    let mut paths = Vec::new();

    // Honour ORT_DYLIB_PATH if already set (e.g. passed from parent process).
    if let Ok(p) = std::env::var("ORT_DYLIB_PATH") {
        paths.push(PathBuf::from(p));
        return paths;
    }

    platform_ort_paths(&mut paths);
    paths
}

#[cfg(target_os = "macos")]
fn platform_ort_paths(paths: &mut Vec<PathBuf>) {
    if let Some(prefix) = command_stdout("brew", &["--prefix", "onnxruntime"]) {
        paths.push(PathBuf::from(prefix).join("lib").join("libonnxruntime.dylib"));
    }
}

#[cfg(target_os = "linux")]
fn platform_ort_paths(paths: &mut Vec<PathBuf>) {
    if let Some(libdir) = command_stdout("pkg-config", &["--variable=libdir", "libonnxruntime"]) {
        paths.push(PathBuf::from(libdir).join("libonnxruntime.so"));
    }
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn platform_ort_paths(_paths: &mut Vec<PathBuf>) {}

/// Run a command and return its trimmed stdout, or `None` on failure.
fn command_stdout(program: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
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
    fn test_try_load_backend_returns_none_for_empty_dir() {
        let tmp = std::env::temp_dir().join("kiok-test-empty-model-dir");
        let _ = std::fs::create_dir_all(&tmp);
        // Ensure no model.onnx exists.
        let _ = std::fs::remove_file(tmp.join("model.onnx"));

        let result = try_load_backend(&tmp);
        assert!(result.is_none(), "should return None when model.onnx is absent");

        let _ = std::fs::remove_dir(&tmp);
    }

    #[test]
    fn test_try_load_backend_returns_none_for_nonexistent_dir() {
        let result = try_load_backend(std::path::Path::new("/nonexistent/path"));
        assert!(result.is_none(), "should return None for nonexistent directory");
    }

    #[test]
    fn test_ensure_ort_dylib_is_consistent_and_valid() {
        let first = ensure_ort_dylib();
        let second = ensure_ort_dylib();
        assert_eq!(first, second, "cached result must be stable across calls");

        // If a path was found, verify it actually exists on disk.
        if let Some(ref path) = first {
            assert!(path.exists(), "returned dylib path should exist on disk");
        }
    }
}
