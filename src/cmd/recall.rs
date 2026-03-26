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

    // --- 3. Build queries from the 3 most-recent chunks. ---
    let recent = db.recent_chunks(&project, 3)?;
    if recent.is_empty() {
        return Ok(());
    }

    // --- 4. Load policy scope. ---
    let viewer_scope = policy::load_scope(project_path)?;

    // --- 5. Search each recent question separately and merge results. ---
    // FTS5 trigram wraps queries as phrases, so a long concatenated query
    // would require an exact phrase match and return nothing.
    // Instead, search each question individually and deduplicate by chunk_id.
    let config = SearchConfig {
        count: count * 2,
        ..SearchConfig::default()
    };

    let mut seen = std::collections::HashSet::new();
    let mut results = Vec::new();
    for chunk in &recent {
        // Use a short substring of each question (trigram works best with short queries)
        let q: String = chunk.question.chars().take(80).collect();
        if q.trim().is_empty() {
            continue;
        }
        if let Ok(hits) = search::keyword_search(&db, &q, &config) {
            for hit in hits {
                if seen.insert(hit.chunk_id) {
                    results.push(hit);
                }
            }
        }
    }

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
