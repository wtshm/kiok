use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{Query, State},
    http::header,
    response::{Html, IntoResponse},
    routing::get,
};
use serde::{Deserialize, Serialize};

use crate::db::Database;
use super::save::db_path;

type AppState = Arc<Mutex<Database>>;

const INDEX_HTML: &str = include_str!("../viewer/index.html");
const STYLE_CSS: &str = include_str!("../viewer/style.css");
const APP_JS: &str = include_str!("../viewer/app.js");

pub fn run(port: u16) -> Result<()> {
    let path = db_path()?;
    if !path.exists() {
        anyhow::bail!("No database found. Run `kiok save` or `kiok import` first.");
    }

    let db = Database::open(path)?;
    let state: AppState = Arc::new(Mutex::new(db));

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async move {
        let app = Router::new()
            .route("/", get(page_index))
            .route("/style.css", get(page_css))
            .route("/app.js", get(page_js))
            .route("/api/stats", get(api_stats))
            .route("/api/sessions", get(api_sessions))
            .route("/api/chunks", get(api_chunks))
            .route("/api/search", get(api_search))
            .with_state(state);

        let addr = format!("127.0.0.1:{}", port);
        let listener = tokio::net::TcpListener::bind(&addr).await
            .context(format!("Failed to bind to {}", addr))?;

        let url = format!("http://{}", addr);
        eprintln!("Listening on {}", url);
        eprintln!("Press Ctrl+C to stop");
        let _ = open::that(&url);

        axum::serve(listener, app).await.context("Server error")
    })
}

#[derive(Serialize)]
struct StatsResponse { sessions: i64, chunks: i64 }

#[derive(Serialize)]
struct SessionRow {
    session_id: String, project: String, scope: String,
    started_at: Option<String>, imported_at: String, chunk_count: i64,
}

#[derive(Serialize)]
struct ChunkResponse {
    chunk_id: i64, session_id: String, project: String,
    question: String, answer: String, timestamp: Option<String>,
}

impl From<crate::db::ChunkRow> for ChunkResponse {
    fn from(r: crate::db::ChunkRow) -> Self {
        Self {
            chunk_id: r.chunk_id, session_id: r.session_id, project: r.project,
            question: r.question, answer: r.answer, timestamp: r.timestamp,
        }
    }
}

#[derive(Deserialize)]
struct SearchParams {
    q: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

#[derive(Deserialize)]
struct PaginationParams {
    #[serde(default = "default_limit")]
    limit: usize,
    #[serde(default)]
    offset: usize,
}

fn default_limit() -> usize { 100 }

async fn api_stats(State(state): State<AppState>) -> axum::Json<StatsResponse> {
    let db = state.lock().unwrap();
    let (sessions, chunks) = db.stats().unwrap_or((0, 0));
    axum::Json(StatsResponse { sessions, chunks })
}

async fn api_sessions(
    State(state): State<AppState>,
    Query(params): Query<PaginationParams>,
) -> axum::Json<Vec<SessionRow>> {
    let db = state.lock().unwrap();
    let rows = query_sessions(&db, &params).unwrap_or_default();
    axum::Json(rows)
}

async fn api_chunks(
    State(state): State<AppState>,
    Query(params): Query<PaginationParams>,
) -> axum::Json<Vec<ChunkResponse>> {
    let db = state.lock().unwrap();
    let rows = query_chunks(&db, &params).unwrap_or_default();
    axum::Json(rows)
}

async fn api_search(
    State(state): State<AppState>,
    Query(params): Query<SearchParams>,
) -> axum::Json<Vec<ChunkResponse>> {
    let db = state.lock().unwrap();
    let rows = search_chunks(&db, &params).unwrap_or_default();
    axum::Json(rows)
}

fn query_sessions(db: &Database, params: &PaginationParams) -> Result<Vec<SessionRow>> {
    let rows = db.list_sessions(params.limit, params.offset)?;
    Ok(rows.into_iter().map(|s| SessionRow {
        session_id: s.session_id, project: s.project, scope: s.scope,
        started_at: s.started_at, imported_at: s.imported_at, chunk_count: s.chunk_count,
    }).collect())
}

fn query_chunks(db: &Database, params: &PaginationParams) -> Result<Vec<ChunkResponse>> {
    Ok(db.list_chunks(params.limit, params.offset)?.into_iter().map(ChunkResponse::from).collect())
}

fn search_chunks(db: &Database, params: &SearchParams) -> Result<Vec<ChunkResponse>> {
    Ok(db.fts_search(&params.q, params.limit)?.into_iter().map(ChunkResponse::from).collect())
}

async fn page_index() -> Html<&'static str> {
    Html(INDEX_HTML)
}

async fn page_css() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "text/css")], STYLE_CSS)
}

async fn page_js() -> impl IntoResponse {
    ([(header::CONTENT_TYPE, "application/javascript")], APP_JS)
}
