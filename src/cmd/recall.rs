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
/// 2. Build queries from the 3 most-recent Q&A chunks.
/// 3. Load the memory policy for `project_path`.
/// 4. Search and deduplicate results.
/// 5. Filter by policy visibility.
/// 6. Print the top `count` results in Markdown format.
pub fn run(project_path: &str, count: usize) -> Result<()> {
    // --- 1. Open the database (silently skip if not present). ---
    let path = db_path()?;
    if !path.exists() {
        return Ok(());
    }
    let db = Database::open(&path)?;

    // --- 2. Resolve project name and build queries. ---
    let project = project_name(project_path);

    let recent = db.recent_chunks(&project, 3)?;
    if recent.is_empty() {
        return Ok(());
    }
    let queries: Vec<String> = recent
        .iter()
        .map(|c| c.question.chars().take(80).collect())
        .collect();

    // --- 3. Load policy scope. ---
    let viewer_scope = policy::load_scope(project_path)?;

    // --- 4. Search and deduplicate. ---
    let config = SearchConfig {
        count: count * 2,
        ..SearchConfig::default()
    };

    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();
    for q in &queries {
        if q.trim().is_empty() {
            continue;
        }
        if let Ok(hits) = search::keyword_search(&db, q, &config) {
            for hit in hits {
                if seen.insert(hit.chunk_id) {
                    results.push(hit);
                }
            }
        }
    }

    // --- 5. Filter by visibility policy. ---
    let filtered: Vec<_> = results
        .into_iter()
        .filter(|r| policy::is_visible(&r.scope, &r.project, viewer_scope, &project))
        .take(count)
        .collect();

    if filtered.is_empty() {
        return Ok(());
    }

    // --- 6. Print in Markdown format. ---
    // Use a distinct header so Claude doesn't confuse this with its built-in memory system.
    println!("<kiok-recall>");
    println!("The following are past conversations from previous Claude Code sessions,");
    println!("retrieved by kiok (a session memory engine). Use them as context when relevant.");
    println!();

    for r in &filtered {
        let date = r
            .timestamp
            .as_deref()
            .and_then(|ts| ts.get(..10))
            .unwrap_or("(no date)");

        println!("- [{}] [project: {}]", date, r.project);
        println!("  User: {}", truncate(&r.question, 200));
        println!("  Assistant: {}", truncate(&r.answer, 500));
        println!();
    }

    println!("</kiok-recall>");

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

use crate::cmd::search_cmd::truncate;
