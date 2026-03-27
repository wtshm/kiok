pub mod fts;
pub mod rrf;
pub mod vector;

use anyhow::Result;
use chrono::Utc;

use crate::db::{ChunkRow, Database};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A ranked search result from either FTS or vector search.
pub struct SearchResult {
    pub chunk: ChunkRow,
    /// 1-based rank position in the result list.
    pub rank: usize,
}

/// Configuration for hybrid search.
pub struct SearchConfig {
    /// Maximum number of results to return.
    pub count: usize,
    /// Multiplier used to over-fetch candidates before fusing and truncating.
    pub candidate_multiplier: usize,
    /// Half-life (in days) for the time-decay factor.
    pub half_life_days: f64,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            count: 5,
            candidate_multiplier: 4,
            half_life_days: 30.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run FTS5 and (optionally) vector search, fuse results with RRF + time
/// decay, and return up to `config.count` results.
///
/// If `query_embedding` is `None`, only FTS results are used.
pub fn hybrid_search(
    db: &Database,
    query_text: &str,
    query_embedding: Option<&[f32]>,
    config: &SearchConfig,
) -> Result<Vec<rrf::ScoredResult>> {
    let candidates = config.count * config.candidate_multiplier;

    let fts_results = fts::search(db, query_text, candidates)?;

    let vec_results = if let Some(emb) = query_embedding {
        vector::search(db, emb, candidates)?
    } else {
        Vec::new()
    };

    let now = Utc::now();
    let mut fused = rrf::fuse(&fts_results, &vec_results, config.half_life_days, now);

    // Truncate to the requested count.
    fused.truncate(config.count);

    Ok(fused)
}

/// Run keyword-only hybrid search (no embedding).
///
/// Convenience wrapper around [`hybrid_search`] that passes `None` for the
/// embedding.
pub fn keyword_search(
    db: &Database,
    query_text: &str,
    config: &SearchConfig,
) -> Result<Vec<rrf::ScoredResult>> {
    hybrid_search(db, query_text, None, config)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn seed_db() -> Database {
        let db = Database::open_in_memory().expect("open_in_memory failed");
        db.insert_session("s1", "proj", "global", None).unwrap();

        let id1 = db
            .insert_chunk(
                "s1", Some("u1"),
                "How to set up Docker?",
                "Use docker-compose.",
                Some("2024-06-01T00:00:00Z"), None,
            )
            .unwrap()
            .unwrap();

        let id2 = db
            .insert_chunk(
                "s1", Some("u2"),
                "Rust ownership explained",
                "Rust uses move semantics.",
                Some("2024-06-01T00:00:00Z"), None,
            )
            .unwrap()
            .unwrap();

        // Store embeddings: Docker chunk gets [1.0; 768], Rust chunk gets [0.0; 768].
        db.insert_embedding(id1, &vec![1.0f32; 768]).unwrap();
        db.insert_embedding(id2, &vec![0.0f32; 768]).unwrap();

        db
    }

    #[test]
    fn test_keyword_search_returns_fts_results() {
        let db = seed_db();
        let config = SearchConfig { count: 10, ..SearchConfig::default() };
        let results = keyword_search(&db, "Docker", &config).unwrap();

        assert_eq!(results.len(), 1);
        assert!(results[0].chunk.question.contains("Docker"));
    }

    #[test]
    fn test_hybrid_search_fuses_fts_and_vector() {
        let db = seed_db();
        let config = SearchConfig { count: 10, ..SearchConfig::default() };

        // Query embedding close to [1.0; 768] (the Docker chunk).
        let query_emb = vec![0.9f32; 768];
        let results = hybrid_search(&db, "Docker", Some(&query_emb), &config).unwrap();

        // Docker chunk should appear: it matches both FTS ("Docker") and vector (nearest).
        assert!(!results.is_empty());
        assert!(results[0].chunk.question.contains("Docker"));

        // Docker chunk gets RRF contributions from both FTS and vector,
        // so it should score higher than Rust chunk (vector-only match).
        if results.len() > 1 {
            assert!(
                results[0].score >= results[1].score,
                "Docker chunk should have highest score from dual RRF contributions"
            );
        }
    }

    #[test]
    fn test_hybrid_search_without_embedding_equals_keyword() {
        let db = seed_db();
        let config = SearchConfig { count: 10, ..SearchConfig::default() };

        let hybrid = hybrid_search(&db, "Docker", None, &config).unwrap();
        let keyword = keyword_search(&db, "Docker", &config).unwrap();

        assert_eq!(hybrid.len(), keyword.len());
    }

    #[test]
    fn test_hybrid_search_respects_count_limit() {
        let db = seed_db();
        let config = SearchConfig { count: 1, ..SearchConfig::default() };

        let query_emb = vec![0.9f32; 768];
        let results = hybrid_search(&db, "Docker", Some(&query_emb), &config).unwrap();

        assert!(results.len() <= 1, "should respect count limit");
    }
}
