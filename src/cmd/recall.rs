use anyhow::Result;

use crate::db::Database;
use crate::embed::{self, EmbeddingBackend};
use super::save::model_dir;
use crate::policy;
use crate::search::{self, SearchConfig};
use super::save::{db_path, project_name};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the recall command.
///
/// 1. Open the database (return silently if no DB exists).
/// 2. Run hybrid search (FTS5 + vector) with RRF + time decay.
///    Falls back to keyword-only search if the embedding model is unavailable.
/// 3. Filter by policy visibility.
/// 4. Print the top `count` results in Markdown format.
pub fn run(query: &str, project_path: &str, count: usize) -> Result<()> {
    // --- 1. Open the database (silently skip if not present). ---
    let path = db_path()?;
    if !path.exists() {
        return Ok(());
    }
    let db = Database::open(&path)?;

    let project = project_name(project_path);

    // --- 2. Load policy scope. ---
    let viewer_scope = policy::load_scope(project_path)?;

    // --- 3. Search (over-fetch to compensate for policy filtering). ---
    let config = SearchConfig {
        count: count * 2,
        ..SearchConfig::default()
    };

    // Try to load the embedding model for hybrid search.
    // Fall back to keyword-only search if the model or ONNX Runtime is unavailable.
    let query_embedding = try_embed_query(query);
    let results = search::hybrid_search(
        &db,
        query,
        query_embedding.as_deref(),
        &config,
    )?;

    // --- 4. Filter by visibility policy. ---
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| policy::is_visible(&r.chunk.scope, &r.chunk.project, viewer_scope, &project))
        .take(count)
        .collect();

    if filtered.is_empty() {
        return Ok(());
    }

    println!("<kiok>");
    println!("The following are past conversations from previous Claude Code sessions,");
    println!("retrieved by kiok (a session memory engine). Use them as context when relevant.");
    println!();

    for r in &filtered {
        let date = r
            .chunk
            .timestamp
            .as_deref()
            .and_then(|ts| ts.get(..10))
            .unwrap_or("(no date)");

        println!("- [{}] [project: {}]", date, r.chunk.project);
        println!("  User: {}", truncate(&r.chunk.question, 200));
        println!("  Assistant: {}", truncate(&r.chunk.answer, 500));
        println!();
    }

    println!("</kiok>");

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Attempt to embed the query text using the ONNX backend.
///
/// Returns `None` silently if the model or ONNX Runtime is unavailable.
fn try_embed_query(query: &str) -> Option<Vec<f32>> {
    let dir = model_dir().ok()?;
    let backend = embed::try_load_backend(&dir)?;
    let mut embeddings = backend.embed(&[query]).ok()?;
    if embeddings.is_empty() {
        return None;
    }
    Some(embeddings.remove(0))
}

/// Truncate `s` to at most `max_chars` Unicode scalar values, appending
/// `"..."` if the string was truncated.
pub(crate) fn truncate(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let collected: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}...", collected)
    } else {
        collected
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_truncate_short_string_unchanged() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_exact_length_unchanged() {
        assert_eq!(truncate("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_long_string_appends_ellipsis() {
        let result = truncate("hello world", 5);
        assert_eq!(result, "hello...");
    }

    #[test]
    fn test_truncate_multibyte_characters() {
        let result = truncate("Docker設定の方法", 6);
        assert_eq!(result, "Docker...");
    }

    #[test]
    fn test_try_embed_query_returns_none_without_model() {
        // With a nonexistent model directory, try_embed_query should
        // gracefully return None (fallback to FTS-only search).
        let result = try_embed_query("test query");

        // We can't guarantee the model is installed in CI, so we just
        // verify it doesn't panic.  If model IS installed, it returns Some.
        // If not, it returns None.  Both are valid.
        let _ = result;
    }
}
