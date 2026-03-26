# kiok — Memory Engine for Claude Code

A Rust-based memory engine that archives Claude Code session conversations into a searchable SQLite database with hybrid search (FTS5 + vector). Inspired by sui-memory's design philosophy: no external services, no LLM token consumption, zero manual operation.

## Problem

Claude Code loses conversation context between sessions. CLAUDE.md conveys static project info, but not past conversation context — design decisions, rejected approaches, debugging history. Starting every session from scratch is costly when using Claude Code as a thinking partner.

## Goals

- **Fast** — SessionStart recall completes within ~500ms (model load + search)
- **Automatic** — Save on SessionEnd, recall on SessionStart via hooks. Zero manual steps
- **Self-contained** — Single binary, SQLite single file, local ONNX model. No external API
- **Japanese-first** — Ruri v3 embeddings + trigram tokenizer for FTS5
- **Privacy-preserving** — All data stays local. No conversation logs sent to external services

## Non-goals

- General-purpose AI agent memory (Claude Code only)
- LLM-based summarization or compression
- Real-time streaming or daemon architecture
- MCP Server integration (hooks only)

## Architecture

```
~/.claude/projects/<project>/<session>.jsonl
        |
        v  [SessionEnd Hook]
+----------------------------------------------+
|  kiok save                                   |
|  +----------+  +----------+  +-------------+ |
|  |  Parse   |->|  Chunk   |->|  Embed      | |
|  |  JSONL   |  |  (Q&A)   |  |  (Ruri v3)  | |
|  +----------+  +----------+  +------+------+ |
|                                      v        |
|                               +-------------+ |
|                               |   SQLite    | |
|                               |  FTS5 +     | |
|                               |  sqlite-vec | |
|                               +-------------+ |
+----------------------------------------------+

        |  [SessionStart Hook]
        v
+----------------------------------------------+
|  kiok recall                                 |
|  +-----------+  +-----------+  +----------+  |
|  |Query Embed|->| Hybrid    |->| Time     |  |
|  |(Ruri v3)  |  | Search    |  | Decay    |  |
|  +-----------+  | FTS5+vec  |  | + RRF    |  |
|                 +-----------+  +----------+  |
+----------------------------------------------+
        |
        v  stdout -> injected into Claude Code context
```

## CLI Commands

| Command | Purpose | Timing |
|---------|---------|--------|
| `kiok save` | Save session conversation | SessionEnd Hook |
| `kiok recall` | Search and output related memories | SessionStart Hook |
| `kiok search <query>` | Manual search | Any time |
| `kiok import` | Bulk import existing JSONL files | Initial setup |
| `kiok policy` | Manage policies | Configuration |
| `kiok stats` | Database statistics | Any time |
| `kiok setup` | Download model, configure hooks | Initial setup |

## Data Pipeline (Save)

### 1. JSONL Parsing

Extract user/assistant messages from session JSONL files.

### 2. Noise Filtering

Remove noise to keep only meaningful conversation content.

| Category | Removed |
|----------|---------|
| Tool results | All `tool_result` content |
| System tags | `<system-reminder>`, `<local-command-caveat>`, `<local-command-stdout>`, `<command-name>`, `<command-message>`, `<command-args>` |
| Read-only tools | Read, Glob, Grep, LSP, ToolSearch, browser snapshot/navigation, TaskGet/TaskOutput/TaskList |
| Meta messages | eval-loop iterations, Stop hook feedback, empty/whitespace-only content |

Preserved: user text input, assistant responses, code-modifying tool calls (Edit, Write, Bash).

### 3. Q&A Chunk Splitting

- User message -> question
- Following assistant message -> answer
- One chunk = one Q&A pair
- Answers exceeding ~2000 tokens are split into multiple chunks

### 4. Policy Check

- Look up policy file from working directory
- If `exclude`: stop here, save nothing
- Otherwise: tag chunk with scope (global / project / isolated)

### 5. Embedding (Ruri v3, ONNX, 310M)

- Concatenate question + answer text per chunk
- Batch embed for efficiency
- Output: 1024-dimensional vector per chunk

### 6. SQLite Insert

- Insert into chunks table + FTS5 index (trigram) + sqlite-vec embeddings
- UUID-based deduplication (safe to re-import)

## Search Engine (Recall / Search)

### Recall Flow (SessionStart)

1. **Context collection** — working directory (project identification). Query is built from the most recent session's last 3 Q&A chunks: extract question texts, concatenate, and truncate to 512 tokens. If no prior session exists for this project, use the project directory name as the query.
2. **Query embedding** — Ruri v3 ONNX (~300-500ms load + ~30-80ms inference)
3. **Parallel search**
   - FTS5 keyword search (trigram tokenizer) — top K×4 candidates
   - sqlite-vec vector nearest neighbor search (cosine similarity) — top K×4 candidates
4. **RRF score fusion** — `score = sum(1 / (k + rank_i))`, k=60
5. **Time decay** — `decayed_score = score * e^(-lambda * age_days)`, lambda = ln(2)/30 (30-day half-life)
6. **Policy filtering**
   - global: search all memories
   - project: same project + other projects' global memories
   - isolated: same project only
7. **Output** — top K results (default 5) as Markdown to stdout

### Recall Output Format

```markdown
## Related memories

### 2026-03-25 | project: my-app
Q: How to set up multi-stage builds in Docker Compose
A: Specify build.target in docker-compose.yml...

### 2026-03-20 | project: my-app
Q: Build cache not working in CI
A: Use BuildKit inline cache with GitHub Actions...
```

### Difference Between recall and search

| | `recall` | `search` |
|---|---------|----------|
| Query | Auto-generated from recent session context | User-specified |
| Output | stdout (for hook injection) | Terminal display (human-readable) |
| Default count | 5 | 10 |

## SQLite Schema

```sql
CREATE TABLE sessions (
    session_id  TEXT PRIMARY KEY,
    project     TEXT NOT NULL,
    scope       TEXT NOT NULL DEFAULT 'global',
    started_at  TEXT,
    imported_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE chunks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id),
    uuid        TEXT,
    question    TEXT NOT NULL,
    answer      TEXT NOT NULL,
    timestamp   TEXT,
    token_count INTEGER
);
CREATE UNIQUE INDEX idx_chunks_uuid ON chunks(uuid) WHERE uuid IS NOT NULL;

CREATE VIRTUAL TABLE chunks_fts USING fts5(
    question,
    answer,
    content_rowid='id',
    content='chunks',
    tokenize='trigram'
);

CREATE VIRTUAL TABLE chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
);
```

## Policy Control

### Policy File

Located at `<project-root>/.claude/memory-policy.json`:

```json
{
  "scope": "project"
}
```

Three possible values: `global`, `project`, `isolated`.

### Scope Behavior

| Scope | Expose own memories to other projects | Receive memories from other projects |
|-------|--------------------------------------|--------------------------------------|
| global | Yes | Yes |
| project | No | Yes (global only) |
| isolated | No | No |

Default (no policy file): `global`.

## Embedding Model

- **Model**: Ruri v3-310M (ONNX format)
- **Parameters**: 310M
- **Output dimensions**: 1024
- **Load time**: ~300-500ms (CPU, Apple Silicon)
- **Inference time**: ~30-80ms per query
- **Language**: Japanese-optimized, multilingual capable

### Pluggable Backend

```rust
trait EmbeddingBackend {
    fn embed(&self, texts: &[&str]) -> Vec<Vec<f32>>;
    fn dimensions(&self) -> usize;
}
```

Default: `OnnxBackend` (Ruri v3). Future extensions: Ollama, OpenAI-compatible API.

## File Layout

```
~/.kiok/
  +-- memory.db              # SQLite (FTS5 + sqlite-vec)
  +-- models/
  |   +-- ruri-v3-310m.onnx  # Embedding model (~600MB, auto-downloaded)
  +-- config.json            # Global settings (optional)
```

### config.json (optional)

```json
{
  "default_scope": "global",
  "recall_count": 5,
  "search_count": 10,
  "time_decay_half_life_days": 30,
  "embedding": {
    "backend": "onnx",
    "model_path": "~/.kiok/models/ruri-v3-310m.onnx"
  }
}
```

## Claude Code Hook Configuration

Generated by `kiok setup` in `~/.claude/settings.json`:

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "kiok recall --project $PWD"
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "kiok save --project $PWD &"
          }
        ]
      }
    ],
    "PreCompact": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "kiok save --project $PWD"
          }
        ]
      }
    ]
  }
}
```

| Hook | Timing | Mode | Reason |
|------|--------|------|--------|
| SessionStart | Session begins | Synchronous | Output to stdout for context injection |
| SessionEnd | Session ends | Background (`&`) | Don't block user |
| PreCompact | Before `/compact` | Synchronous | Capture data before compaction discards it |

## Setup Flow

```
$ kiok setup
1. Creating ~/.kiok/ ...
2. Downloading ruri-v3-310m.onnx (~600MB) ...
3. Initializing SQLite database ...
4. Adding hooks to ~/.claude/settings.json ...
Done. Run `kiok import` to import existing sessions.
```

## Rust Crate Structure

```
kiok/
  +-- Cargo.toml
  +-- src/
      +-- main.rs              # CLI entry point (clap)
      +-- cmd/
      |   +-- mod.rs
      |   +-- save.rs
      |   +-- recall.rs
      |   +-- search.rs
      |   +-- import.rs
      |   +-- policy.rs
      |   +-- stats.rs
      |   +-- setup.rs
      +-- db.rs                # SQLite operations (FTS5, sqlite-vec)
      +-- chunk.rs             # JSONL parsing + Q&A chunk splitting
      +-- filter.rs            # Noise filtering
      +-- embed/
      |   +-- mod.rs           # EmbeddingBackend trait
      |   +-- onnx.rs          # ONNX Runtime implementation
      +-- search/
      |   +-- mod.rs           # Hybrid search orchestration
      |   +-- fts.rs           # FTS5 keyword search
      |   +-- vector.rs        # sqlite-vec vector search
      |   +-- rrf.rs           # RRF score fusion + time decay
      +-- policy.rs            # Policy evaluation
```

### Key Dependencies

| Crate | Purpose |
|-------|---------|
| `clap` | CLI argument parsing |
| `rusqlite` | SQLite (FTS5-enabled build) |
| `sqlite-vec` | Vector search extension |
| `ort` | ONNX Runtime (model inference) |
| `serde` / `serde_json` | JSON processing |
| `chrono` | Timestamp handling |
| `uuid` | Deduplication |
| `tokio` | Async (model download) |
| `indicatif` | Progress bars (import) |

## Comparison with Existing Tools

| Aspect | sui-memory | kiok |
|--------|-----------|------|
| Language | Python (1,759 lines) | Rust |
| Embedding | Ruri v3 (sentence-transformers) | Ruri v3 (ONNX Runtime) |
| FTS | FTS5 trigram | FTS5 trigram |
| Hybrid search | RRF | RRF |
| Time decay | 30-day half-life | 30-day half-life (configurable) |
| Storage | SQLite + sqlite-vec | SQLite + sqlite-vec |
| LLM usage | None | None |
| Policy | 3 actions x 3 scopes + overrides | 3 scopes (minimal) |
| Distribution | `uv sync` | Single binary |
| Performance | Python overhead + PyTorch load | Native binary + ONNX Runtime |
