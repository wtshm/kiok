use kiok::chunk;
use kiok::db::Database;
use kiok::policy::{self, Scope};
use kiok::search::{self, SearchConfig};

// ---------------------------------------------------------------------------
// Test 1: save + FTS search
// ---------------------------------------------------------------------------

/// Parse the sample fixture JSONL, insert the session and chunks into a
/// temporary in-memory database, then run a keyword search for "Docker" and
/// assert that at least one result is returned.
#[test]
fn test_save_and_fts_search() {
    let db = Database::open_in_memory().expect("open_in_memory failed");

    // Parse the sample fixture.
    let content = include_str!("fixtures/sample_session.jsonl");
    let chunks = chunk::parse_session(content).expect("parse_session failed");
    assert!(!chunks.is_empty(), "fixture should contain at least one chunk");

    // Insert session.
    let session_id = "test-session-1";
    db.insert_session(session_id, "my-project", "global", None)
        .expect("insert_session failed");

    // Insert chunks.
    for c in &chunks {
        db.insert_chunk(
            session_id,
            c.uuid.as_deref(),
            &c.question,
            &c.answer,
            c.timestamp.as_deref(),
            None,
        )
        .expect("insert_chunk failed");
    }

    // Search for "Docker" — must find at least one result.
    let config = SearchConfig {
        count: 10,
        ..SearchConfig::default()
    };
    let results = search::keyword_search(&db, "Docker", &config).expect("keyword_search failed");

    assert!(
        !results.is_empty(),
        "expected at least one result for 'Docker' but got none"
    );

    // The top result should mention Docker somewhere.
    let top = &results[0];
    let combined = format!("{} {}", top.chunk.question, top.chunk.answer).to_lowercase();
    assert!(
        combined.contains("docker"),
        "top result should contain 'docker', got: question='{}' answer='{}'",
        top.chunk.question,
        top.chunk.answer
    );
}

// ---------------------------------------------------------------------------
// Test 2: policy filtering
// ---------------------------------------------------------------------------

/// Insert chunks from two projects:
///   - "global-project" with scope "global"
///   - "isolated-project" with scope "isolated"
///
/// Searching for "Docker" from a "project" scope viewer for "global-project"
/// should return a result from global-project (same project, always visible).
/// Searching from a viewer for "isolated-project" should NOT see
/// global-project's chunks.
#[test]
fn test_policy_filtering() {
    let db = Database::open_in_memory().expect("open_in_memory failed");

    // Insert global-project session + chunk.
    db.insert_session("sess-global", "global-project", "global", None)
        .expect("insert global session failed");
    db.insert_chunk(
        "sess-global",
        Some("uuid-g1"),
        "How do I set up Docker?",
        "Use docker-compose to define your services.",
        None,
        None,
    )
    .expect("insert global chunk failed");

    // Insert isolated-project session + chunk.
    db.insert_session("sess-isolated", "isolated-project", "isolated", None)
        .expect("insert isolated session failed");
    db.insert_chunk(
        "sess-isolated",
        Some("uuid-i1"),
        "Docker on isolated project",
        "docker run hello-world",
        None,
        None,
    )
    .expect("insert isolated chunk failed");

    let config = SearchConfig {
        count: 20,
        ..SearchConfig::default()
    };
    let results = search::keyword_search(&db, "Docker", &config).expect("keyword_search failed");

    // Both chunks should match the raw search.
    assert_eq!(
        results.len(),
        2,
        "expected 2 raw results before policy filtering"
    );

    // --- Filter from a "project"-scoped viewer for "global-project". ---
    let viewer_project = "global-project";
    let viewer_scope = Scope::Project;

    let visible_to_global_viewer: Vec<_> = results
        .iter()
        .filter(|r| policy::is_visible(&r.chunk.scope, &r.chunk.project, viewer_scope, viewer_project))
        .collect();

    // The viewer is global-project with Project scope. It should see:
    //   - Its own project's chunks (global-project, scope="global") → visible
    //   - isolated-project chunk (scope="isolated") → NOT visible (cross-project, non-global scope)
    assert_eq!(
        visible_to_global_viewer.len(),
        1,
        "global-project viewer with Project scope should see exactly 1 result"
    );
    assert_eq!(
        visible_to_global_viewer[0].chunk.project, "global-project",
        "the visible result should belong to global-project"
    );

    // --- Filter from an "isolated"-scoped viewer for "isolated-project". ---
    let isolated_viewer_project = "isolated-project";
    let isolated_viewer_scope = Scope::Isolated;

    let visible_to_isolated_viewer: Vec<_> = results
        .iter()
        .filter(|r| {
            policy::is_visible(
                &r.chunk.scope,
                &r.chunk.project,
                isolated_viewer_scope,
                isolated_viewer_project,
            )
        })
        .collect();

    // Isolated viewers can only see their own project's chunks.
    assert_eq!(
        visible_to_isolated_viewer.len(),
        1,
        "isolated-project viewer should see exactly 1 result (own project only)"
    );
    assert_eq!(
        visible_to_isolated_viewer[0].chunk.project, "isolated-project",
        "the visible result for the isolated viewer should be from isolated-project"
    );
}
