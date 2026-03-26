use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::search::fts::SearchResult;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const RRF_K: f64 = 60.0;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// A search result after RRF fusion and optional time-decay scoring.
#[derive(Debug, Clone)]
pub struct ScoredResult {
    pub chunk_id: i64,
    pub session_id: String,
    pub project: String,
    pub scope: String,
    pub question: String,
    pub answer: String,
    pub timestamp: Option<String>,
    pub score: f64,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Fuse FTS and vector search results using Reciprocal Rank Fusion (RRF),
/// then apply an exponential time-decay penalty.
///
/// `half_life_days` is the number of days after which a score is halved.
/// If a result has no timestamp, no decay is applied (full score).
pub fn fuse(
    fts_results: &[SearchResult],
    vec_results: &[SearchResult],
    half_life_days: f64,
    now: DateTime<Utc>,
) -> Vec<ScoredResult> {
    // Map from chunk_id → accumulated RRF score + metadata.
    let mut map: HashMap<i64, ScoredResult> = HashMap::new();

    // Helper closure: accumulate RRF score from one ranked list.
    let mut accumulate = |results: &[SearchResult]| {
        for sr in results {
            let rrf_contrib = 1.0 / (RRF_K + sr.rank as f64);
            let entry = map.entry(sr.chunk.chunk_id).or_insert_with(|| ScoredResult {
                chunk_id: sr.chunk.chunk_id,
                session_id: sr.chunk.session_id.clone(),
                project: sr.chunk.project.clone(),
                scope: sr.chunk.scope.clone(),
                question: sr.chunk.question.clone(),
                answer: sr.chunk.answer.clone(),
                timestamp: sr.chunk.timestamp.clone(),
                score: 0.0,
            });
            entry.score += rrf_contrib;
        }
    };

    accumulate(fts_results);
    accumulate(vec_results);

    // Apply time decay to each accumulated score.
    let lambda = std::f64::consts::LN_2 / half_life_days;

    let mut scored: Vec<ScoredResult> = map
        .into_values()
        .map(|mut r| {
            if let Some(ts) = &r.timestamp {
                if let Ok(parsed) = ts.parse::<DateTime<Utc>>() {
                    let age_days = (now - parsed).num_seconds() as f64 / 86_400.0;
                    let age_days = age_days.max(0.0);
                    r.score *= (-lambda * age_days).exp();
                }
                // If timestamp is present but unparseable, leave score as-is.
            }
            // If timestamp is None, no decay — leave score as-is.
            r
        })
        .collect();

    // Sort descending by score; break ties by chunk_id for determinism.
    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk_id.cmp(&b.chunk_id))
    });

    scored
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ChunkRow;
    use chrono::TimeZone;

    fn make_chunk(chunk_id: i64, timestamp: Option<&str>) -> ChunkRow {
        ChunkRow {
            chunk_id,
            session_id: format!("sess-{}", chunk_id),
            project: "test-project".to_owned(),
            scope: "global".to_owned(),
            question: format!("Question {}", chunk_id),
            answer: format!("Answer {}", chunk_id),
            timestamp: timestamp.map(str::to_owned),
            rank: 0.0,
        }
    }

    fn make_sr(chunk_id: i64, rank: usize, timestamp: Option<&str>) -> SearchResult {
        SearchResult {
            chunk: make_chunk(chunk_id, timestamp),
            rank,
        }
    }

    /// A chunk appearing in both FTS and vector results accumulates two RRF
    /// contributions and should therefore have the highest score.
    #[test]
    fn test_rrf_fusion_combines_scores() {
        let now = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();

        // chunk 1 appears in both lists at rank 1.
        let fts = vec![make_sr(1, 1, None), make_sr(2, 2, None)];
        let vec = vec![make_sr(1, 1, None), make_sr(3, 2, None)];

        let results = fuse(&fts, &vec, 30.0, now);

        // chunk 1 must be first.
        assert_eq!(results[0].chunk_id, 1, "chunk appearing in both lists must rank highest");

        // chunk 1 should have approximately 2 * (1 / (60+1)).
        let expected = 2.0 / (RRF_K + 1.0);
        assert!(
            (results[0].score - expected).abs() < 1e-10,
            "chunk 1 score mismatch: got {}, expected {}",
            results[0].score,
            expected
        );
    }

    /// A 30-day-old chunk with rank 1 should score lower than a 1-day-old
    /// chunk with rank 2, given a 30-day half-life.
    #[test]
    fn test_time_decay_reduces_old_scores() {
        let now = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();

        let ts_old = "2024-05-02T00:00:00Z"; // ~30 days ago
        let ts_new = "2024-05-31T00:00:00Z"; // ~1 day ago

        let fts = vec![make_sr(1, 1, Some(ts_old)), make_sr(2, 2, Some(ts_new))];
        let vec: Vec<SearchResult> = vec![];

        let results = fuse(&fts, &vec, 30.0, now);

        let r1 = results.iter().find(|r| r.chunk_id == 1).expect("chunk 1 missing");
        let r2 = results.iter().find(|r| r.chunk_id == 2).expect("chunk 2 missing");

        assert!(
            r2.score > r1.score,
            "1-day-old rank-2 chunk ({}) should outscore 30-day-old rank-1 chunk ({})",
            r2.score,
            r1.score
        );
    }

    /// A chunk with no timestamp must receive its full RRF score (no decay).
    #[test]
    fn test_no_timestamp_no_decay() {
        let now = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();

        let fts = vec![make_sr(1, 1, None)];
        let vec: Vec<SearchResult> = vec![];

        let results = fuse(&fts, &vec, 30.0, now);

        assert_eq!(results.len(), 1);
        let expected = 1.0 / (RRF_K + 1.0);
        assert!(
            (results[0].score - expected).abs() < 1e-10,
            "no-timestamp chunk must receive full score {}, got {}",
            expected,
            results[0].score
        );
    }
}
