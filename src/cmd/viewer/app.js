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

function esc(s) {
  if (!s) return '';
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

function trunc(s, n) {
  if (!s) return '';
  return s.length > n ? s.slice(0, n) + '...' : s;
}

function relTime(ts) {
  if (!ts) return '';
  const d = new Date(ts), now = new Date(), diff = (now - d) / 1000;
  if (diff < 60) return 'just now';
  if (diff < 3600) return Math.floor(diff / 60) + 'm ago';
  if (diff < 86400) return Math.floor(diff / 3600) + 'h ago';
  if (diff < 604800) return Math.floor(diff / 86400) + 'd ago';
  return ts.slice(0, 10);
}

async function loadSessions() {
  const r = await fetch('/api/sessions?limit=100');
  const rows = await r.json();
  if (!rows.length) {
    $('#content').innerHTML = '<div class="empty"><div class="empty-icon">\u2014</div>No sessions yet</div>';
    return;
  }
  let html = '<div class="grid-header"><span>Project</span><span>Scope</span><span>Chunks</span><span>Session ID</span><span style="text-align:right">Imported</span></div>';
  html += '<div class="sessions-grid">';
  for (const s of rows) {
    html += `<div class="session-row">
      <span class="project">${esc(s.project)}</span>
      <span class="scope scope-${s.scope}">${s.scope}</span>
      <span class="chunks-count">${s.chunk_count}</span>
      <span class="sid">${esc(s.session_id.slice(0, 12))}...</span>
      <span class="time">${relTime(s.imported_at)}</span>
    </div>`;
  }
  html += '</div>';
  $('#content').innerHTML = html;
  $('#result-count').textContent = '';
}

async function loadChunks() {
  const r = await fetch('/api/chunks?limit=100');
  const rows = await r.json();
  if (!rows.length) {
    $('#content').innerHTML = '<div class="empty"><div class="empty-icon">\u2014</div>No memories yet</div>';
    return;
  }
  renderChunks(rows);
  $('#result-count').textContent = '';
}

async function doSearch(q) {
  if (!q.trim()) { loadChunks(); return; }
  const r = await fetch('/api/search?q=' + encodeURIComponent(q) + '&limit=50');
  const rows = await r.json();
  if (!rows.length) {
    $('#content').innerHTML = '<div class="empty"><div class="empty-icon">\u2014</div>No matches for "' + esc(q) + '"</div>';
    $('#result-count').textContent = '0';
    return;
  }
  renderChunks(rows);
  $('#result-count').textContent = rows.length + ' found';
}

function renderChunks(rows) {
  let html = '<div class="cards">';
  for (const c of rows) {
    html += `<div class="card" onclick="this.classList.toggle('expanded')">
      <div class="card-head">
        <span class="card-project">${esc(c.project)}</span>
        <span class="card-time">${relTime(c.timestamp)}</span>
      </div>
      <div class="card-q">${esc(trunc(c.question, 200))}</div>
      <div class="card-preview">${esc(trunc(c.answer, 150))}</div>
      <div class="card-a">${esc(c.answer)}</div>
    </div>`;
  }
  html += '</div>';
  $('#content').innerHTML = html;
}

// Nav
document.querySelectorAll('.nav-item').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.nav-item').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  currentTab = t.dataset.tab;
  $('#search').value = '';
  if (currentTab === 'sessions') loadSessions(); else loadChunks();
}));

// Search
let st;
$('#search').addEventListener('input', e => {
  clearTimeout(st);
  st = setTimeout(() => doSearch(e.target.value), 300);
});

// Init
loadStats();
loadChunks();
