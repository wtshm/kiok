use std::sync::{Arc, Mutex};
use anyhow::{Context, Result};
use axum::{
    Router,
    extract::{Query, State},
    response::Html,
    routing::get,
};
use serde::{Deserialize, Serialize};

use crate::db::Database;
use super::save::db_path;

type AppState = Arc<Mutex<Database>>;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

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
            .route("/", get(index_page))
            .route("/api/stats", get(api_stats))
            .route("/api/sessions", get(api_sessions))
            .route("/api/chunks", get(api_chunks))
            .route("/api/search", get(api_search))
            .with_state(state);

        let addr = format!("127.0.0.1:{}", port);
        let listener = tokio::net::TcpListener::bind(&addr).await
            .context(format!("Failed to bind to {}", addr))?;

        eprintln!("kiok view: http://{}", addr);
        let _ = open::that(format!("http://{}", addr));

        axum::serve(listener, app).await.context("Server error")
    })
}

// ---------------------------------------------------------------------------
// API types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct StatsResponse {
    sessions: i64,
    chunks: i64,
}

#[derive(Serialize)]
struct SessionRow {
    session_id: String,
    project: String,
    scope: String,
    started_at: Option<String>,
    imported_at: String,
    chunk_count: i64,
}

#[derive(Serialize)]
struct ChunkResponse {
    chunk_id: i64,
    session_id: String,
    project: String,
    question: String,
    answer: String,
    timestamp: Option<String>,
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
    #[serde(default)]
    project: Option<String>,
}

fn default_limit() -> usize { 50 }

// ---------------------------------------------------------------------------
// API handlers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// DB queries
// ---------------------------------------------------------------------------

fn query_sessions(db: &Database, params: &PaginationParams) -> Result<Vec<SessionRow>> {
    let rows = db.list_sessions(params.limit, params.offset)?;
    Ok(rows
        .into_iter()
        .map(|s| SessionRow {
            session_id: s.session_id,
            project: s.project,
            scope: s.scope,
            started_at: s.started_at,
            imported_at: s.imported_at,
            chunk_count: s.chunk_count,
        })
        .collect())
}

fn query_chunks(db: &Database, params: &PaginationParams) -> Result<Vec<ChunkResponse>> {
    let rows = db.list_chunks(params.limit, params.offset)?;
    Ok(rows
        .into_iter()
        .map(|r| ChunkResponse {
            chunk_id: r.chunk_id,
            session_id: r.session_id,
            project: r.project,
            question: r.question,
            answer: r.answer,
            timestamp: r.timestamp,
        })
        .collect())
}

fn search_chunks(db: &Database, params: &SearchParams) -> Result<Vec<ChunkResponse>> {
    let fts_results = db.fts_search(&params.q, params.limit)?;
    Ok(fts_results
        .into_iter()
        .map(|r| ChunkResponse {
            chunk_id: r.chunk_id,
            session_id: r.session_id,
            project: r.project,
            question: r.question,
            answer: r.answer,
            timestamp: r.timestamp,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// HTML page
// ---------------------------------------------------------------------------

async fn index_page() -> Html<&'static str> {
    Html(INDEX_HTML)
}

const INDEX_HTML: &str = r##"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>kiok</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace;
         background: #0d1117; color: #c9d1d9; padding: 20px; max-width: 1200px; margin: 0 auto; }
  h1 { color: #58a6ff; margin-bottom: 4px; font-size: 1.4em; }
  .subtitle { color: #8b949e; margin-bottom: 20px; font-size: 0.85em; }
  .stats { display: flex; gap: 20px; margin-bottom: 20px; }
  .stat { background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 12px 20px; }
  .stat-value { font-size: 1.8em; font-weight: bold; color: #58a6ff; }
  .stat-label { color: #8b949e; font-size: 0.85em; }
  .search-box { width: 100%; padding: 10px 14px; background: #0d1117; border: 1px solid #30363d;
                 border-radius: 6px; color: #c9d1d9; font-size: 1em; margin-bottom: 20px; }
  .search-box:focus { outline: none; border-color: #58a6ff; }
  .tabs { display: flex; gap: 2px; margin-bottom: 16px; }
  .tab { padding: 8px 16px; background: #161b22; border: 1px solid #30363d; border-radius: 6px 6px 0 0;
         cursor: pointer; color: #8b949e; font-size: 0.9em; }
  .tab.active { background: #0d1117; color: #58a6ff; border-bottom-color: #0d1117; }
  table { width: 100%; border-collapse: collapse; }
  th { text-align: left; padding: 8px 12px; border-bottom: 1px solid #30363d; color: #8b949e;
       font-size: 0.8em; text-transform: uppercase; }
  td { padding: 8px 12px; border-bottom: 1px solid #21262d; font-size: 0.9em; vertical-align: top; }
  tr:hover { background: #161b22; }
  .chunk-q { color: #58a6ff; }
  .chunk-a { color: #8b949e; max-width: 600px; overflow: hidden; text-overflow: ellipsis;
             white-space: nowrap; }
  .project-badge { background: #1f6feb22; color: #58a6ff; padding: 2px 8px; border-radius: 12px;
                   font-size: 0.8em; }
  .scope-badge { padding: 2px 8px; border-radius: 12px; font-size: 0.75em; }
  .scope-global { background: #23863622; color: #3fb950; }
  .scope-project { background: #9e6a0322; color: #d29922; }
  .scope-isolated { background: #f8514922; color: #f85149; }
  .ts { color: #484f58; font-size: 0.8em; white-space: nowrap; }
  .empty { text-align: center; padding: 40px; color: #484f58; }
  .session-id { font-family: monospace; font-size: 0.8em; color: #484f58; }
</style>
</head>
<body>
<h1>kiok</h1>
<p class="subtitle">Memory Engine for Claude Code</p>

<div class="stats" id="stats"></div>

<input class="search-box" id="search" type="text" placeholder="Search memories... (trigram FTS5)">

<div class="tabs">
  <div class="tab active" data-tab="chunks">Chunks</div>
  <div class="tab" data-tab="sessions">Sessions</div>
</div>

<div id="content"></div>

<script>
const $ = s => document.querySelector(s);
let currentTab = 'chunks';

async function loadStats() {
  const r = await fetch('/api/stats');
  const d = await r.json();
  $('#stats').innerHTML = `
    <div class="stat"><div class="stat-value">${d.sessions}</div><div class="stat-label">Sessions</div></div>
    <div class="stat"><div class="stat-value">${d.chunks}</div><div class="stat-label">Chunks</div></div>
  `;
}

function esc(s) { if(!s) return ''; const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function trunc(s, n) { if(!s) return ''; return s.length > n ? s.slice(0, n) + '...' : s; }
function scopeBadge(s) { return `<span class="scope-badge scope-${s}">${s}</span>`; }

async function loadSessions() {
  const r = await fetch('/api/sessions?limit=100');
  const rows = await r.json();
  if (!rows.length) { $('#content').innerHTML = '<div class="empty">No sessions</div>'; return; }
  let html = '<table><tr><th>Project</th><th>Scope</th><th>Chunks</th><th>Session</th><th>Imported</th></tr>';
  for (const s of rows) {
    html += `<tr>
      <td><span class="project-badge">${esc(s.project)}</span></td>
      <td>${scopeBadge(s.scope)}</td>
      <td>${s.chunk_count}</td>
      <td class="session-id">${esc(s.session_id.slice(0,8))}</td>
      <td class="ts">${esc(s.imported_at)}</td>
    </tr>`;
  }
  html += '</table>';
  $('#content').innerHTML = html;
}

async function loadChunks() {
  const r = await fetch('/api/chunks?limit=100');
  const rows = await r.json();
  if (!rows.length) { $('#content').innerHTML = '<div class="empty">No chunks</div>'; return; }
  renderChunks(rows);
}

async function doSearch(q) {
  if (!q.trim()) { loadChunks(); return; }
  const r = await fetch('/api/search?q=' + encodeURIComponent(q) + '&limit=50');
  const rows = await r.json();
  if (!rows.length) { $('#content').innerHTML = '<div class="empty">No results for "'+esc(q)+'"</div>'; return; }
  renderChunks(rows);
}

function renderChunks(rows) {
  let html = '<table><tr><th>Project</th><th>Q</th><th>A</th><th>Time</th></tr>';
  for (const c of rows) {
    html += `<tr>
      <td><span class="project-badge">${esc(c.project)}</span></td>
      <td class="chunk-q">${esc(trunc(c.question, 80))}</td>
      <td class="chunk-a" title="${esc(c.answer)}">${esc(trunc(c.answer, 120))}</td>
      <td class="ts">${esc(c.timestamp ? c.timestamp.slice(0,10) : '')}</td>
    </tr>`;
  }
  html += '</table>';
  $('#content').innerHTML = html;
}

// Events
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  currentTab = t.dataset.tab;
  if (currentTab === 'sessions') loadSessions();
  else loadChunks();
}));

let searchTimeout;
$('#search').addEventListener('input', e => {
  clearTimeout(searchTimeout);
  searchTimeout = setTimeout(() => doSearch(e.target.value), 300);
});

// Init
loadStats();
loadChunks();
</script>
</body>
</html>
"##;
