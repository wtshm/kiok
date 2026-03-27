use anyhow::Result;

use crate::db::Database;
use super::SearchResult;

/// Run a full-text keyword search against the database and return results
/// with 1-based rank positions.
pub fn search(db: &Database, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let rows = db.fts_search(query, limit)?;
    let results = rows
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| SearchResult {
            rank: i + 1,
            chunk,
        })
        .collect();
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;

    #[test]
    fn test_fts_search_returns_ranked_results() {
        let db = Database::open_in_memory().expect("open_in_memory failed");
        db.insert_session("s1", "my-project", "global", None)
            .expect("insert_session failed");

        // Two Docker-related chunks and one Rust chunk.
        db.insert_chunk(
            "s1",
            Some("uuid-d1"),
            "How do I set up Docker?",
            "Use docker-compose to define your services.",
            None,
            None,
        )
        .expect("insert chunk 1 failed");

        db.insert_chunk(
            "s1",
            Some("uuid-d2"),
            "Docker networking explained",
            "Docker uses bridge networks by default.",
            None,
            None,
        )
        .expect("insert chunk 2 failed");

        db.insert_chunk(
            "s1",
            Some("uuid-r1"),
            "How do I learn Rust?",
            "Read The Rust Programming Language book.",
            None,
            None,
        )
        .expect("insert chunk 3 failed");

        let results = search(&db, "Docker", 10).expect("search failed");

        assert_eq!(results.len(), 2, "expected 2 Docker results");
        assert_eq!(results[0].rank, 1, "first result must have rank 1");
        assert_eq!(results[1].rank, 2, "second result must have rank 2");

        // Both results should mention Docker.
        for r in &results {
            let text = format!("{} {}", r.chunk.question, r.chunk.answer);
            assert!(
                text.contains("Docker") || text.contains("docker"),
                "result should contain 'Docker' or 'docker'"
            );
        }
    }
}
