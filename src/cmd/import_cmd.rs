use anyhow::{Context, Result};
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::path::{Path, PathBuf};

use crate::chunk;
use crate::db::Database;
use super::save::{data_dir, db_path};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Import statistics returned by `run_quiet`.
pub struct ImportStats {
    pub chunks: usize,
    pub sessions: usize,
    pub skipped: usize,
    pub errors: usize,
}

/// Run the import command with default output.
pub fn run() -> Result<()> {
    let stats = run_quiet()?;
    println!(
        "Imported {} chunks from {} sessions ({} skipped, {} errors)",
        stats.chunks, stats.sessions, stats.skipped, stats.errors
    );
    Ok(())
}

/// Run the import command, returning statistics without printing a summary.
pub fn run_quiet() -> Result<ImportStats> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let projects_dir = home.join(".claude").join("projects");

    if !projects_dir.exists() {
        return Ok(ImportStats { chunks: 0, sessions: 0, skipped: 0, errors: 0 });
    }

    // Open / create the database.
    let data = data_dir()?;
    fs::create_dir_all(&data)
        .with_context(|| format!("Could not create data directory {}", data.display()))?;
    let db = Database::open(db_path()?)?;

    let mut total_chunks = 0usize;
    let mut total_sessions = 0usize;
    let mut skipped = 0usize;
    let mut errors = 0usize;

    // First pass: collect all (project_name, jsonl_path) pairs.
    let project_entries = fs::read_dir(&projects_dir)
        .with_context(|| format!("Could not read {}", projects_dir.display()))?;

    let mut work_items: Vec<(String, PathBuf)> = Vec::new();
    for entry in project_entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("import: error reading directory entry: {}", e);
                errors += 1;
                continue;
            }
        };

        let project_dir = entry.path();
        if !project_dir.is_dir() {
            continue;
        }

        // Decode the human-readable project name from the encoded directory name.
        let encoded_name = project_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_owned();
        let project = decode_project_name(&encoded_name);

        // Find all JSONL files (including subagent subdirs).
        for jsonl_path in find_jsonl_files(&project_dir) {
            work_items.push((project.clone(), jsonl_path));
        }
    }

    // Create a progress bar sized to the total number of sessions.
    let pb = ProgressBar::new(work_items.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("  [{bar:40.cyan/blue}] {pos}/{len} sessions")
            .expect("invalid progress bar template")
            .progress_chars("#>-"),
    );

    // Second pass: process each work item and increment the progress bar.
    for (project, jsonl_path) in work_items {
        let session_id = match jsonl_path
            .file_stem()
            .and_then(|s| s.to_str())
        {
            Some(s) => s.to_owned(),
            None => {
                eprintln!("import: could not extract session_id from {}", jsonl_path.display());
                errors += 1;
                pb.inc(1);
                continue;
            }
        };

        // Skip already-imported sessions.
        match db.session_exists(&session_id) {
            Ok(true) => {
                skipped += 1;
                pb.inc(1);
                continue;
            }
            Ok(false) => {}
            Err(e) => {
                eprintln!("import: DB error for session {}: {}", session_id, e);
                errors += 1;
                pb.inc(1);
                continue;
            }
        }

        // Parse + chunk the session.
        let content = match fs::read_to_string(&jsonl_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("import: could not read {}: {}", jsonl_path.display(), e);
                errors += 1;
                pb.inc(1);
                continue;
            }
        };

        let chunks = match chunk::parse_session(&content) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("import: parse error for {}: {}", jsonl_path.display(), e);
                errors += 1;
                pb.inc(1);
                continue;
            }
        };

        // NOTE: import uses "global" scope by default because the original
        // project path is not available, and the policy file may not have
        // existed when these historical sessions were created.
        if chunks.is_empty() {
            let _ = db.insert_session(&session_id, &project, "global", None);
            total_sessions += 1;
            pb.inc(1);
            continue;
        }

        if let Err(e) = db.insert_session(&session_id, &project, "global", None) {
            eprintln!("import: failed to insert session {}: {}", session_id, e);
            errors += 1;
            pb.inc(1);
            continue;
        }

        let mut inserted_count = 0usize;
        for c in &chunks {
            match db.insert_chunk(
                &session_id,
                c.uuid.as_deref(),
                &c.question,
                &c.answer,
                c.timestamp.as_deref(),
                None,
            ) {
                Ok(Some(_)) => inserted_count += 1,
                Ok(None) => {}
                Err(e) => {
                    eprintln!("import: failed to insert chunk: {}", e);
                    errors += 1;
                }
            }
        }

        total_chunks += inserted_count;
        total_sessions += 1;
        pb.inc(1);
    }

    pb.finish_and_clear();

    Ok(ImportStats {
        chunks: total_chunks,
        sessions: total_sessions,
        skipped,
        errors,
    })
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Decode a Claude Code encoded project directory name back to the project name.
///
/// Claude Code encodes paths by replacing non-alphanumeric characters with `-`.
/// For example, `-Users-kenta-workspace-myapp` → `"myapp"`.
pub fn decode_project_name(encoded: &str) -> String {
    // Take the last segment when split by `-`.
    encoded
        .split('-').rfind(|s| !s.is_empty())
        .unwrap_or(encoded)
        .to_owned()
}

/// Recursively find all `.jsonl` files under `dir`.
pub fn find_jsonl_files(dir: &Path) -> Vec<PathBuf> {
    let mut result = Vec::new();
    _find_jsonl_files(dir, &mut result);
    result
}

fn _find_jsonl_files(dir: &Path, result: &mut Vec<PathBuf>) {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            _find_jsonl_files(&path, result);
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            result.push(path);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_project_name() {
        assert_eq!(decode_project_name("-Users-kenta-workspace-myapp"), "myapp");
        assert_eq!(decode_project_name("-Users-kenta-workspace-my-app"), "app");
        assert_eq!(decode_project_name("simple"), "simple");
        assert_eq!(decode_project_name("-a-b-c"), "c");
    }
}
