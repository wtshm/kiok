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
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
:root {
  --bg: #08080a; --surface: #111114; --surface2: #18181c;
  --border: #1e1e24; --border-hover: #2a2a32;
  --text: #d4d4d8; --text2: #71717a; --text3: #3f3f46;
  --accent: #3b82f6; --accent-dim: #3b82f618; --accent-glow: #3b82f610;
  --green: #4ade80; --yellow: #fbbf24; --red: #f87171;
  --mono: 'JetBrains Mono', monospace;
  --sans: 'DM Sans', sans-serif;
}
*, *::before, *::after { margin: 0; padding: 0; box-sizing: border-box; }
html { font-size: 15px; }
body { background: var(--bg); color: var(--text); font-family: var(--sans);
       min-height: 100vh; overflow-x: hidden; }

/* --- Layout --- */
.shell { display: grid; grid-template-columns: 260px 1fr; min-height: 100vh; }

/* --- Sidebar --- */
.sidebar { background: var(--surface); border-right: 1px solid var(--border);
           padding: 32px 24px; display: flex; flex-direction: column; gap: 32px;
           position: sticky; top: 0; height: 100vh; overflow-y: auto; }
.logo { display: flex; align-items: baseline; gap: 12px; }
.logo-mark { font-family: var(--mono); font-size: 1.4rem; font-weight: 500;
             color: var(--accent); line-height: 1; letter-spacing: -0.03em; }
.logo-ver { font-family: var(--mono); font-size: 0.6rem; font-weight: 300;
            color: var(--text3); letter-spacing: 0.05em; }
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.stat-card { background: var(--bg); border: 1px solid var(--border); border-radius: 8px;
             padding: 16px; }
.stat-val { font-family: var(--mono); font-size: 1.6rem; font-weight: 500;
            color: var(--text); line-height: 1; }
.stat-lbl { font-size: 0.7rem; color: var(--text3); text-transform: uppercase;
            letter-spacing: 0.1em; margin-top: 6px; }
.nav { display: flex; flex-direction: column; gap: 4px; }
.nav-item { padding: 10px 14px; border-radius: 6px; cursor: pointer; font-size: 0.85rem;
            color: var(--text2); display: flex; align-items: center; gap: 10px;
            font-weight: 400; letter-spacing: 0.01em; }
.nav-item:hover { background: var(--accent-glow); color: var(--text); }
.nav-item.active { background: var(--accent-dim); color: var(--accent); font-weight: 500; }
.nav-icon { font-size: 1rem; width: 20px; text-align: center; opacity: 0.7; }
.sidebar-footer { margin-top: auto; font-family: var(--mono); font-size: 0.65rem;
                  color: var(--text3); letter-spacing: 0.05em; }

/* --- Main --- */
.main { padding: 32px 40px; }
.search-wrap { position: relative; margin-bottom: 28px; }
.search-input { width: 100%; padding: 14px 18px 14px 44px; background: var(--surface);
                border: 1px solid var(--border); border-radius: 10px; color: var(--text);
                font-family: var(--mono); font-size: 0.85rem; font-weight: 300;
                transition: border-color 0.15s; letter-spacing: 0.02em; }
.search-input:focus { outline: none; border-color: var(--accent); }
.search-input::placeholder { color: var(--text3); }
.search-icon { position: absolute; left: 16px; top: 50%; transform: translateY(-50%);
               color: var(--text3); font-size: 0.9rem; pointer-events: none; }
.search-count { position: absolute; right: 16px; top: 50%; transform: translateY(-50%);
                font-family: var(--mono); font-size: 0.7rem; color: var(--text3); }

/* --- Cards --- */
.cards { display: flex; flex-direction: column; gap: 8px; }
.card { background: var(--surface); border: 1px solid var(--border); border-radius: 10px;
        padding: 20px 24px; cursor: pointer; contain: content; }
.card:hover { border-color: var(--border-hover); }
.card.expanded { background: var(--surface2); }
.card-head { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.card-project { font-family: var(--mono); font-size: 0.7rem; font-weight: 500;
                color: var(--accent); background: var(--accent-dim); padding: 3px 10px;
                border-radius: 20px; letter-spacing: 0.03em; }
.card-time { font-family: var(--mono); font-size: 0.7rem; color: var(--text3);
             margin-left: auto; letter-spacing: 0.02em; }
.card-q { font-size: 0.9rem; color: var(--text); line-height: 1.5; font-weight: 400; }
.card-a { font-size: 0.82rem; color: var(--text2); line-height: 1.6; margin-top: 12px;
          padding-top: 12px; border-top: 1px solid var(--border);
          white-space: pre-wrap; word-break: break-word;
          display: none; }
.card.expanded .card-a { display: block; }
.card-preview { font-size: 0.8rem; color: var(--text3); margin-top: 8px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.card.expanded .card-preview { display: none; }

/* --- Sessions table --- */
.sessions-grid { display: flex; flex-direction: column; gap: 6px; }
.session-row { display: grid; grid-template-columns: 140px 80px 60px 1fr 100px;
               gap: 16px; align-items: center; padding: 14px 20px;
               background: var(--surface); border: 1px solid var(--border);
               border-radius: 8px; font-size: 0.82rem; contain: content; }
.session-row:hover { border-color: var(--border-hover); }
.session-row .project { font-family: var(--mono); color: var(--accent); font-weight: 500; font-size: 0.78rem; }
.session-row .scope { font-family: var(--mono); font-size: 0.7rem; padding: 2px 8px;
                      border-radius: 4px; text-align: center; }
.scope-global { background: #4ade8012; color: var(--green); }
.scope-project { background: #fbbf2412; color: var(--yellow); }
.scope-isolated { background: #f8717112; color: var(--red); }
.session-row .chunks-count { font-family: var(--mono); color: var(--text2); text-align: center; }
.session-row .sid { font-family: var(--mono); color: var(--text3); font-size: 0.72rem; }
.session-row .time { font-family: var(--mono); color: var(--text3); font-size: 0.72rem; text-align: right; }

/* --- Header row --- */
.grid-header { display: grid; grid-template-columns: 140px 80px 60px 1fr 100px;
               gap: 16px; padding: 0 20px 10px; font-family: var(--mono);
               font-size: 0.65rem; color: var(--text3); text-transform: uppercase;
               letter-spacing: 0.12em; }

/* --- Empty state --- */
.empty { text-align: center; padding: 60px 20px; color: var(--text3);
         font-family: var(--mono); font-size: 0.85rem; }
.empty-icon { font-size: 2rem; color: var(--text3); margin-bottom: 12px; opacity: 0.3; }

/* --- Reduced motion --- */
@media (prefers-reduced-motion: reduce) { * { animation: none !important; } }
</style>
</head>
<body>
<div class="shell">
  <aside class="sidebar">
    <div class="logo">
      <span class="logo-mark">kiok</span>
      <span class="logo-ver">v0.1</span>
    </div>
    <div class="stats-grid" id="stats"></div>
    <nav class="nav">
      <div class="nav-item active" data-tab="chunks">
        <span class="nav-icon">&#9638;</span> Memories
      </div>
      <div class="nav-item" data-tab="sessions">
        <span class="nav-icon">&#9776;</span> Sessions
      </div>
    </nav>
    <div class="sidebar-footer">~/.kiok/memory.db</div>
  </aside>
  <main class="main">
    <div class="search-wrap">
      <span class="search-icon">&#8981;</span>
      <input class="search-input" id="search" type="text" placeholder="Search memories...">
      <span class="search-count" id="result-count"></span>
    </div>
    <div id="content"></div>
  </main>
</div>

<script>
const $ = s => document.querySelector(s);
let currentTab = 'chunks';

async function loadStats() {
  const r = await fetch('/api/stats');
  const d = await r.json();
  $('#stats').innerHTML = `
    <div class="stat-card"><div class="stat-val">${d.sessions}</div><div class="stat-lbl">sessions</div></div>
    <div class="stat-card"><div class="stat-val">${d.chunks}</div><div class="stat-lbl">memories</div></div>
  `;
}

function esc(s) { if(!s)return ''; const d=document.createElement('div'); d.textContent=s; return d.innerHTML; }
function trunc(s, n) { if(!s)return ''; return s.length>n ? s.slice(0,n)+'...' : s; }
function relTime(ts) {
  if(!ts) return '';
  const d = new Date(ts), now = new Date(), diff = (now-d)/1000;
  if(diff<60) return 'just now';
  if(diff<3600) return Math.floor(diff/60)+'m ago';
  if(diff<86400) return Math.floor(diff/3600)+'h ago';
  if(diff<604800) return Math.floor(diff/86400)+'d ago';
  return ts.slice(0,10);
}

async function loadSessions() {
  const r = await fetch('/api/sessions?limit=100');
  const rows = await r.json();
  if(!rows.length){ $('#content').innerHTML='<div class="empty"><div class="empty-icon">—</div>No sessions yet</div>'; return; }
  let html = '<div class="grid-header"><span>Project</span><span>Scope</span><span>Chunks</span><span>Session ID</span><span style="text-align:right">Imported</span></div>';
  html += '<div class="sessions-grid">';
  rows.forEach((s,i) => {
    html += `<div class="session-row">
      <span class="project">${esc(s.project)}</span>
      <span class="scope scope-${s.scope}">${s.scope}</span>
      <span class="chunks-count">${s.chunk_count}</span>
      <span class="sid">${esc(s.session_id.slice(0,12))}...</span>
      <span class="time">${relTime(s.imported_at)}</span>
    </div>`;
  });
  html += '</div>';
  $('#content').innerHTML = html;
  $('#result-count').textContent = '';
}

async function loadChunks() {
  const r = await fetch('/api/chunks?limit=100');
  const rows = await r.json();
  if(!rows.length){ $('#content').innerHTML='<div class="empty"><div class="empty-icon">—</div>No memories yet</div>'; return; }
  renderChunks(rows);
  $('#result-count').textContent = '';
}

async function doSearch(q) {
  if(!q.trim()){ loadChunks(); return; }
  const r = await fetch('/api/search?q='+encodeURIComponent(q)+'&limit=50');
  const rows = await r.json();
  if(!rows.length){ $('#content').innerHTML='<div class="empty"><div class="empty-icon">—</div>No matches for "'+esc(q)+'"</div>';
    $('#result-count').textContent='0'; return; }
  renderChunks(rows);
  $('#result-count').textContent = rows.length + ' found';
}

function renderChunks(rows) {
  let html = '<div class="cards">';
  rows.forEach((c,i) => {
    html += `<div class="card" onclick="this.classList.toggle('expanded')">
      <div class="card-head">
        <span class="card-project">${esc(c.project)}</span>
        <span class="card-time">${relTime(c.timestamp)}</span>
      </div>
      <div class="card-q">${esc(trunc(c.question, 200))}</div>
      <div class="card-preview">${esc(trunc(c.answer, 150))}</div>
      <div class="card-a">${esc(c.answer)}</div>
    </div>`;
  });
  html += '</div>';
  $('#content').innerHTML = html;
}

// Nav
document.querySelectorAll('.nav-item').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.nav-item').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  currentTab = t.dataset.tab;
  $('#search').value = '';
  if(currentTab==='sessions') loadSessions(); else loadChunks();
}));

// Search
let st;
$('#search').addEventListener('input', e => { clearTimeout(st); st=setTimeout(()=>doSearch(e.target.value),300); });

// Init
loadStats();
loadChunks();
</script>
</body>
</html>
"##;
