use anyhow::Result;

use crate::db::Database;
use crate::policy;
use crate::search::{self, SearchConfig};
use super::save::{db_path, project_name};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the recall command.
///
/// 1. Open the database (return silently if no DB exists).
/// 2. Determine the project name from `project_path`.
/// 3. Build a query from the 3 most-recent Q&A chunks (questions only,
///    concatenated and truncated to 512 chars).
/// 4. Load the memory policy for `project_path`.
/// 5. Run `keyword_search` with `count * 2` candidates.
/// 6. Filter by policy visibility.
/// 7. Print the top `count` results in Markdown format.
pub fn run(project_path: &str, count: usize) -> Result<()> {
    // --- 1. Open the database (silently skip if not present). ---
    let path = db_path()?;
    if !path.exists() {
        return Ok(());
    }
    let db = Database::open(&path)?;

    // --- 2. Resolve project name. ---
    let project = project_name(project_path);

    // --- 3. Build query from the 3 most-recent chunks. ---
    let recent = db.recent_chunks(&project, 3)?;
    if recent.is_empty() {
        return Ok(());
    }

    let raw_query: String = recent
        .iter()
        .map(|c| c.question.as_str())
        .collect::<Vec<_>>()
        .join(" ");
    let query = truncate(&raw_query, 512);

    // --- 4. Load policy scope. ---
    let viewer_scope = policy::load_scope(project_path)?;

    // --- 5. Run keyword search with count*2 candidates. ---
    let config = SearchConfig {
        count: count * 2,
        ..SearchConfig::default()
    };
    let results = search::keyword_search(&db, &query, &config)?;

    // --- 6. Filter by visibility policy. ---
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| policy::is_visible(&r.scope, &r.project, viewer_scope, &project))
        .take(count)
        .collect();

    if filtered.is_empty() {
        return Ok(());
    }

    // --- 7. Print in Markdown format. ---
    println!("## Related memories");
    println!();

    for r in &filtered {
        // Extract date portion from ISO 8601 timestamp, if present.
        let date = r
            .timestamp
            .as_deref()
            .and_then(|ts| ts.get(..10))
            .unwrap_or("(no date)");

        println!("### {} | project: {}", date, r.project);
        println!("Q: {}", truncate(&r.question, 200));
        println!("A: {}", truncate(&r.answer, 500));
        println!();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Truncate `s` to at most `max_chars` Unicode scalar values, appending
/// `"..."` if truncated.
fn truncate(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let collected: String = chars.by_ref().take(max_chars).collect();
    if chars.next().is_some() {
        format!("{}...", collected)
    } else {
        collected
    }
}
