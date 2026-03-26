use anyhow::Result;

use crate::db::Database;
use crate::search::fts;
use super::save::db_path;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the search command: open the database, execute an FTS query, and
/// display the results to stdout.
pub fn run(query: &str, project: Option<&str>, count: usize) -> Result<()> {
    let path = db_path()?;
    let db = Database::open(&path)?;

    let results = fts::search(&db, query, count)?;

    if results.is_empty() {
        println!("No results found for \"{}\".", query);
        return Ok(());
    }

    for r in &results {
        let chunk = &r.chunk;

        // Header line: [scope] project | timestamp
        let timestamp = chunk.timestamp.as_deref().unwrap_or("(no timestamp)");
        let project_display = match project {
            Some(p) if p != chunk.project => {
                // Caller filtered by project but this result belongs to a
                // different one — skip it (simple client-side filter).
                continue;
            }
            _ => &chunk.project,
        };
        println!(
            "[{}] {} | {}",
            chunk.scope, project_display, timestamp
        );

        // Q/A content (truncated).
        println!("Q: {}", truncate(&chunk.question, 120));
        println!("A: {}", truncate(&chunk.answer, 240));
        println!();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
        // "Docker設定" is 8 Unicode scalar values.
        let result = truncate("Docker設定の方法", 6);
        assert_eq!(result, "Docker...");
    }
}
