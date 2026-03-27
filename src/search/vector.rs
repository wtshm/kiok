use anyhow::Result;

use crate::db::Database;
use super::SearchResult;

/// Run a KNN vector search against the database and return results
/// with 1-based rank positions (nearest first).
pub fn search(db: &Database, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
    let rows = db.vec_search(query_embedding, limit)?;
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

    /// Insert two chunks with 768-dim embeddings (all-1.0 and all-0.0),
    /// query with all-0.9, and expect the all-1.0 chunk to be nearest.
    #[test]
    fn test_vec_search_returns_nearest() {
        let db = Database::open_in_memory().expect("open_in_memory failed");

        db.insert_session("s1", "proj", "global", None)
            .expect("insert_session failed");

        let id1 = db
            .insert_chunk("s1", None, "All ones question", "All ones answer", None, None)
            .expect("insert_chunk 1 failed")
            .expect("expected chunk id 1");

        let id2 = db
            .insert_chunk("s1", None, "All zeros question", "All zeros answer", None, None)
            .expect("insert_chunk 2 failed")
            .expect("expected chunk id 2");

        // Embedding of all 1.0 values (un-normalized for test clarity).
        let emb_ones: Vec<f32> = vec![1.0f32; 768];
        // Embedding of all 0.0 values.
        let emb_zeros: Vec<f32> = vec![0.0f32; 768];

        db.insert_embedding(id1, &emb_ones)
            .expect("insert_embedding 1 failed");
        db.insert_embedding(id2, &emb_zeros)
            .expect("insert_embedding 2 failed");

        // Query with all-0.9 — should be closest to the all-1.0 chunk.
        let query: Vec<f32> = vec![0.9f32; 768];
        let results = search(&db, &query, 2).expect("vec_search failed");

        assert_eq!(results.len(), 2, "expected 2 results");
        assert_eq!(results[0].rank, 1, "first result rank must be 1");
        assert_eq!(results[1].rank, 2, "second result rank must be 2");
        assert_eq!(
            results[0].chunk.question,
            "All ones question",
            "nearest chunk should be the all-1.0 chunk"
        );
    }
}
