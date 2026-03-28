use chrono::{DateTime, Utc};
use std::collections::HashMap;

use crate::db::ChunkRow;
use super::SearchResult;

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
    pub chunk: ChunkRow,
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
    let mut map: HashMap<i64, ScoredResult> = HashMap::new();

    let mut accumulate = |results: &[SearchResult]| {
        for sr in results {
            let rrf_contrib = 1.0 / (RRF_K + sr.rank as f64);
            let entry = map
                .entry(sr.chunk.chunk_id)
                .or_insert_with(|| ScoredResult {
                    chunk: sr.chunk.clone(),
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
            if let Some(ts) = &r.chunk.timestamp
                && let Ok(parsed) = ts.parse::<DateTime<Utc>>()
            {
                let age_days = (now - parsed).num_seconds() as f64 / 86_400.0;
                r.score *= (-lambda * age_days.max(0.0)).exp();
            }
            r
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.chunk.chunk_id.cmp(&b.chunk.chunk_id))
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

    fn make_sr(chunk_id: i64, rank: usize, timestamp: Option<&str>) -> SearchResult {
        SearchResult {
            chunk: ChunkRow {
                chunk_id,
                session_id: format!("sess-{}", chunk_id),
                project: "test-project".to_owned(),
                scope: "global".to_owned(),
                question: format!("Question {}", chunk_id),
                answer: format!("Answer {}", chunk_id),
                timestamp: timestamp.map(str::to_owned),
            },
            rank,
        }
    }

    #[test]
    fn test_rrf_fusion_combines_scores() {
        let now = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();

        // chunk 1: appears in both lists at rank 1
        // chunk 2: FTS only at rank 2
        // chunk 3: vector only at rank 2
        let fts = vec![make_sr(1, 1, None), make_sr(2, 2, None)];
        let vec = vec![make_sr(1, 1, None), make_sr(3, 2, None)];

        let results = fuse(&fts, &vec, 30.0, now);

        assert_eq!(results.len(), 3, "should have 3 unique chunks");
        assert_eq!(results[0].chunk.chunk_id, 1, "chunk appearing in both lists must rank highest");

        // chunk 1: 2 contributions from rank 1
        let expected_dual = 2.0 / (RRF_K + 1.0);
        assert!(
            (results[0].score - expected_dual).abs() < 1e-10,
            "chunk 1 score mismatch: got {}, expected {}",
            results[0].score,
            expected_dual
        );

        // chunks 2 and 3: 1 contribution each from rank 2
        let expected_single = 1.0 / (RRF_K + 2.0);
        let r2 = results.iter().find(|r| r.chunk.chunk_id == 2).expect("chunk 2 missing");
        let r3 = results.iter().find(|r| r.chunk.chunk_id == 3).expect("chunk 3 missing");
        assert!(
            (r2.score - expected_single).abs() < 1e-10,
            "chunk 2 score mismatch: got {}, expected {}",
            r2.score, expected_single
        );
        assert!(
            (r3.score - expected_single).abs() < 1e-10,
            "chunk 3 score mismatch: got {}, expected {}",
            r3.score, expected_single
        );
    }

    #[test]
    fn test_time_decay_reduces_old_scores() {
        let now = Utc.with_ymd_and_hms(2024, 6, 1, 0, 0, 0).unwrap();

        let ts_old = "2024-05-02T00:00:00Z";
        let ts_new = "2024-05-31T00:00:00Z";

        let fts = vec![make_sr(1, 1, Some(ts_old)), make_sr(2, 2, Some(ts_new))];
        let vec: Vec<SearchResult> = vec![];

        let results = fuse(&fts, &vec, 30.0, now);

        let r1 = results.iter().find(|r| r.chunk.chunk_id == 1).expect("chunk 1 missing");
        let r2 = results.iter().find(|r| r.chunk.chunk_id == 2).expect("chunk 2 missing");

        assert!(
            r2.score > r1.score,
            "1-day-old rank-2 chunk ({}) should outscore 30-day-old rank-1 chunk ({})",
            r2.score,
            r1.score
        );
    }

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
