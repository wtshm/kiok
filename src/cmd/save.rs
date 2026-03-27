use anyhow::{Context, Result, bail};
use std::fs;
use std::path::PathBuf;

use crate::chunk;
use crate::db::Database;

use crate::policy;

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

/// Returns the kiok data directory: `~/.kiok/`.
pub fn data_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home.join(".kiok"))
}

/// Returns the path to the kiok SQLite database: `~/.kiok/memory.db`.
pub fn db_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("memory.db"))
}

/// Extracts the last path component from a project path.
///
/// For example, `/Users/kenta/workspace/my-app` → `"my-app"`.
pub fn project_name(project_path: &str) -> String {
    std::path::Path::new(project_path)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(project_path)
        .to_owned()
}

// ---------------------------------------------------------------------------
// Claude Code path encoding
// ---------------------------------------------------------------------------

/// Encode a filesystem path the same way Claude Code does when creating
/// the projects directory: every character that is not ASCII alphanumeric
/// is replaced with `-`.
fn encode_path(path: &str) -> String {
    path.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '-' })
        .collect()
}

// ---------------------------------------------------------------------------
// Main save logic
// ---------------------------------------------------------------------------

/// Run the save pipeline for the given project path.
pub fn run(project_path: &str) -> Result<()> {
    // --- 1. Locate the most recent JSONL file for this project. ---
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let encoded = encode_path(project_path);
    let projects_dir = home.join(".claude").join("projects").join(&encoded);

    let jsonl_path = most_recent_jsonl(&projects_dir)
        .with_context(|| format!("No JSONL sessions found in {}", projects_dir.display()))?;

    // --- 2. Extract session_id from the filename stem. ---
    let session_id = jsonl_path
        .file_stem()
        .and_then(|s| s.to_str())
        .context("Could not extract session_id from filename")?
        .to_owned();

    // --- 3. Open the database (create data dir if needed). ---
    let data = data_dir()?;
    fs::create_dir_all(&data)
        .with_context(|| format!("Could not create data directory {}", data.display()))?;
    let db = Database::open(db_path()?)?;

    // --- 4. Parse + chunk the session. ---
    let content = fs::read_to_string(&jsonl_path)
        .with_context(|| format!("Could not read {}", jsonl_path.display()))?;

    let chunks = chunk::parse_session(&content)
        .with_context(|| format!("Could not parse session {}", jsonl_path.display()))?;

    if chunks.is_empty() {
        eprintln!("save: no Q&A chunks found in session {}", session_id);
        return Ok(());
    }

    // --- 5. Insert session (relies on INSERT OR IGNORE for idempotency). ---
    // Load the memory policy scope for this project so it is stored correctly.
    let scope = policy::load_scope(project_path).unwrap_or_default();
    let scope_str = scope.as_str();
    let project = project_name(project_path);
    let started_at: Option<&str> = None;
    let inserted = db.insert_session(&session_id, &project, scope_str, started_at)?;
    if !inserted {
        eprintln!("Session {} already saved, skipping", session_id);
        return Ok(());
    }

    let mut saved = 0usize;
    let mut chunk_ids: Vec<Option<i64>> = Vec::with_capacity(chunks.len());

    for c in &chunks {
        let inserted = db.insert_chunk(
            &session_id,
            c.uuid.as_deref(),
            &c.question,
            &c.answer,
            c.timestamp.as_deref(),
            None,
        )?;
        if inserted.is_some() {
            saved += 1;
        }
        chunk_ids.push(inserted);
    }

    // --- 7. Spawn background embed process. ---
    // Embedding loads a 1.2GB ONNX model and is too slow to run inline.
    // Spawn a detached process so save returns immediately.
    if let Ok(kiok_bin) = std::env::current_exe() {
        let mut cmd = std::process::Command::new(kiok_bin);
        cmd.arg("embed")
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null());

        // Propagate ORT_DYLIB_PATH so the child can find the ONNX Runtime dylib.
        if let Some(dylib_path) = crate::embed::ensure_ort_dylib() {
            cmd.env("ORT_DYLIB_PATH", dylib_path);
        }

        let _ = cmd.spawn();
    }

    // --- 8. Print summary. ---
    eprintln!(
        "save: session={} project={} chunks={}/{}",
        session_id,
        project,
        saved,
        chunks.len()
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Path helpers for the ONNX model
// ---------------------------------------------------------------------------

/// Returns `~/.kiok/models/ruri-v3-310m/model.onnx`.
pub fn model_onnx_path() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    Ok(home
        .join(".kiok")
        .join("models")
        .join("ruri-v3-310m")
        .join("model.onnx"))
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Return the most recently modified `.jsonl` file under `dir`,
/// or an error if none exist.
fn most_recent_jsonl(dir: &std::path::Path) -> Result<PathBuf> {
    let entries = fs::read_dir(dir)
        .with_context(|| format!("Could not read directory {}", dir.display()))?;

    let mut candidates: Vec<(std::time::SystemTime, PathBuf)> = Vec::new();

    for entry in entries {
        let entry = entry.context("Could not read directory entry")?;
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
            continue;
        }
        let mtime = entry
            .metadata()
            .and_then(|m| m.modified())
            .unwrap_or(std::time::SystemTime::UNIX_EPOCH);
        candidates.push((mtime, path));
    }

    if candidates.is_empty() {
        bail!("No .jsonl files found in {}", dir.display());
    }

    candidates.sort_by(|a, b| b.0.cmp(&a.0));
    Ok(candidates.remove(0).1)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_name_extracts_last_component() {
        assert_eq!(project_name("/Users/kenta/workspace/my-app"), "my-app");
        assert_eq!(project_name("/home/user/project"), "project");
        assert_eq!(project_name("simple"), "simple");
    }

    #[test]
    fn test_encode_path_replaces_non_alphanumeric() {
        assert_eq!(encode_path("/Users/kenta/workspace/my-app"), "-Users-kenta-workspace-my-app");
        assert_eq!(encode_path("abc123"), "abc123");
        assert_eq!(encode_path("/a/b"), "-a-b");
    }

    #[test]
    fn test_data_dir_returns_kiok_under_home() {
        let path = data_dir().expect("data_dir failed");
        assert!(path.to_string_lossy().ends_with(".kiok"));
    }

    #[test]
    fn test_db_path_returns_memory_db() {
        let path = db_path().expect("db_path failed");
        assert!(path.to_string_lossy().ends_with("memory.db"));
    }
}
