const $ = s => document.querySelector(s);
const PAGE_SIZE = 100;
let currentTab = 'chunks';
let chunksOffset = 0;
let sessionsOffset = 0;

async function loadStats() {
  const r = await fetch('/api/stats');
  const d = await r.json();
  $('#stats').innerHTML = `
    <span class="stat-item"><span class="stat-val">${d.sessions}</span> sessions</span>
    <span class="stat-item"><span class="stat-val">${d.chunks}</span> chunks</span>
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

function renderPager(offset, count, onPrev, onNext) {
  const page = Math.floor(offset / PAGE_SIZE) + 1;
  const hasPrev = offset > 0;
  const hasNext = count === PAGE_SIZE;
  let html = '<div class="pager">';
  if (hasPrev) html += `<button class="pager-btn" id="pg-prev">\u2190 Prev</button>`;
  html += `<span class="pager-info">Page ${page}</span>`;
  if (hasNext) html += `<button class="pager-btn" id="pg-next">Next \u2192</button>`;
  html += '</div>';
  return { html, bind() {
    const prev = document.getElementById('pg-prev');
    const next = document.getElementById('pg-next');
    if (prev) prev.addEventListener('click', onPrev);
    if (next) next.addEventListener('click', onNext);
  }};
}

async function loadSessions(offset) {
  if (offset === undefined) offset = sessionsOffset;
  sessionsOffset = offset;
  const r = await fetch(`/api/sessions?limit=${PAGE_SIZE}&offset=${offset}`);
  const rows = await r.json();
  if (!rows.length && offset === 0) {
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
  const pg = renderPager(offset, rows.length,
    () => loadSessions(Math.max(0, offset - PAGE_SIZE)),
    () => loadSessions(offset + PAGE_SIZE));
  html += pg.html;
  $('#content').innerHTML = html;
  pg.bind();
  $('#result-count').textContent = '';
}

async function loadChunks(offset) {
  if (offset === undefined) offset = chunksOffset;
  chunksOffset = offset;
  const r = await fetch(`/api/chunks?limit=${PAGE_SIZE}&offset=${offset}`);
  const rows = await r.json();
  if (!rows.length && offset === 0) {
    $('#content').innerHTML = '<div class="empty"><div class="empty-icon">\u2014</div>No chunks yet</div>';
    return;
  }
  renderChunks(rows, offset);
  $('#result-count').textContent = '';
}

async function doSearch(q) {
  if (!q.trim()) { chunksOffset = 0; loadChunks(0); return; }
  const r = await fetch('/api/search?q=' + encodeURIComponent(q) + '&limit=' + PAGE_SIZE);
  const rows = await r.json();
  if (!rows.length) {
    $('#content').innerHTML = '<div class="empty"><div class="empty-icon">\u2014</div>No matches for "' + esc(q) + '"</div>';
    $('#result-count').textContent = '0';
    return;
  }
  renderChunks(rows, null);
  $('#result-count').textContent = rows.length + ' found';
}

function renderChunks(rows, offset) {
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
  if (offset !== null) {
    const pg = renderPager(offset, rows.length,
      () => { scrollTo(0,0); loadChunks(Math.max(0, offset - PAGE_SIZE)); },
      () => { scrollTo(0,0); loadChunks(offset + PAGE_SIZE); });
    html += pg.html;
    $('#content').innerHTML = html;
    pg.bind();
  } else {
    $('#content').innerHTML = html;
  }
}

// Nav
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  currentTab = t.dataset.tab;
  $('#search').value = '';
  if (currentTab === 'sessions') { sessionsOffset = 0; loadSessions(0); }
  else { chunksOffset = 0; loadChunks(0); }
}));

// Search
let st;
$('#search').addEventListener('input', e => {
  clearTimeout(st);
  st = setTimeout(() => doSearch(e.target.value), 300);
});

// Init
loadStats();
loadChunks(0);
