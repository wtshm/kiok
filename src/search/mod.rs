pub mod fts;
pub mod rrf;
pub mod vector;

use anyhow::Result;
use chrono::Utc;

use crate::db::Database;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

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

    // Run FTS search.
    let fts_results = fts::search(db, query_text, candidates)?;

    // Run vector search if an embedding is provided.  Convert vector::SearchResult
    // into fts::SearchResult so that rrf::fuse() receives a homogeneous slice type.
    let vec_results_raw = if let Some(emb) = query_embedding {
        vector::search(db, emb, candidates)?
    } else {
        Vec::new()
    };
    let vec_results: Vec<fts::SearchResult> = vec_results_raw
        .into_iter()
        .map(|sr| fts::SearchResult {
            chunk: sr.chunk,
            rank: sr.rank,
        })
        .collect();

    // Fuse with RRF + time decay.
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
