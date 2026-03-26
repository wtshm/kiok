# kiok Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Rust CLI memory engine for Claude Code that saves session conversations and recalls relevant memories via hybrid search (FTS5 trigram + sqlite-vec vector search with Ruri v3 embeddings).

**Architecture:** Single binary CLI (`kiok`) with SQLite storage. SessionEnd hook saves conversations as Q&A chunks with embeddings. SessionStart hook recalls related memories via hybrid search (FTS5 + vector) fused with RRF and time decay. Policy system controls cross-project memory visibility.

**Tech Stack:** Rust, rusqlite (FTS5 bundled), sqlite-vec, ort (ONNX Runtime), tokenizers (HuggingFace), clap, serde, chrono

---

## File Structure

```
kiok/
  Cargo.toml
  src/
    main.rs                # CLI entry point (clap)
    db.rs                  # SQLite connection, schema, CRUD
    filter.rs              # Noise filtering rules
    chunk.rs               # JSONL parsing + Q&A chunk splitting
    policy.rs              # Policy file loading + scope evaluation
    embed/
      mod.rs               # EmbeddingBackend trait definition
      onnx.rs              # OnnxBackend: Ruri v3 via ort + tokenizers
    search/
      mod.rs               # HybridSearch: orchestrate FTS + vec + RRF
      fts.rs               # FTS5 trigram keyword search
      vector.rs            # sqlite-vec nearest neighbor search
      rrf.rs               # RRF score fusion + time decay
    cmd/
      mod.rs               # Command enum + dispatch
      save.rs              # `kiok save` — full save pipeline
      recall.rs            # `kiok recall` — auto-query + hybrid search
      search.rs            # `kiok search` — manual query
      import_cmd.rs        # `kiok import` — bulk import
      stats.rs             # `kiok stats` — DB statistics
      setup.rs             # `kiok setup` — model download + hook config
  tests/
    fixtures/
      sample_session.jsonl # Test fixture: minimal Claude Code session
    integration.rs         # End-to-end tests
```

---

### Task 1: Project Scaffold + CLI Skeleton

**Files:**
- Create: `Cargo.toml`
- Create: `src/main.rs`
- Create: `src/cmd/mod.rs`

- [ ] **Step 1: Create Cargo.toml**

```toml
[package]
name = "kiok"
version = "0.1.0"
edition = "2024"
description = "Memory engine for Claude Code — hybrid search over session history"
license = "MIT"

[dependencies]
clap = { version = "4", features = ["derive"] }
anyhow = "1"
serde = { version = "1", features = ["derive"] }
serde_json = "1"
chrono = { version = "0.4", features = ["serde"] }
rusqlite = { version = "0.34", features = ["bundled", "blob"] }
uuid = { version = "1", features = ["v4"] }
```

Note: Start with minimal dependencies. Add `ort`, `tokenizers`, `sqlite-vec`, `tokio`, `indicatif` in later tasks when needed.

- [ ] **Step 2: Create src/main.rs**

```rust
use anyhow::Result;
use clap::{Parser, Subcommand};

mod cmd;

#[derive(Parser)]
#[command(name = "kiok", about = "Memory engine for Claude Code")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Save session conversation to memory
    Save {
        /// Project directory path
        #[arg(long)]
        project: String,
    },
    /// Recall related memories for current session
    Recall {
        /// Project directory path
        #[arg(long)]
        project: String,
        /// Number of results to return
        #[arg(long, default_value = "5")]
        count: usize,
    },
    /// Search memories manually
    Search {
        /// Search query
        query: String,
        /// Filter by project
        #[arg(long)]
        project: Option<String>,
        /// Number of results to return
        #[arg(long, default_value = "10")]
        count: usize,
    },
    /// Import existing Claude Code sessions
    Import,
    /// Show database statistics
    Stats,
    /// Set up kiok (download model, configure hooks)
    Setup,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    cmd::run(cli.command)
}
```

- [ ] **Step 3: Create src/cmd/mod.rs**

```rust
use anyhow::Result;

pub fn run(command: crate::Commands) -> Result<()> {
    match command {
        crate::Commands::Save { project } => {
            eprintln!("save: project={}", project);
            Ok(())
        }
        crate::Commands::Recall { project, count } => {
            eprintln!("recall: project={}, count={}", project, count);
            Ok(())
        }
        crate::Commands::Search { query, project, count } => {
            eprintln!("search: query={}, project={:?}, count={}", query, project, count);
            Ok(())
        }
        crate::Commands::Import => {
            eprintln!("import");
            Ok(())
        }
        crate::Commands::Stats => {
            eprintln!("stats");
            Ok(())
        }
        crate::Commands::Setup => {
            eprintln!("setup");
            Ok(())
        }
    }
}
```

- [ ] **Step 4: Verify build**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 5: Verify CLI**

Run: `cargo run -- --help`
Expected: Shows help with all subcommands (save, recall, search, import, stats, setup)

Run: `cargo run -- save --project /tmp/test`
Expected: Prints `save: project=/tmp/test`

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/main.rs src/cmd/mod.rs
git commit -m "feat: scaffold project with CLI skeleton"
```

---

### Task 2: SQLite Database Layer

**Files:**
- Create: `src/db.rs`
- Modify: `src/main.rs` (add `mod db`)

- [ ] **Step 1: Write tests for db module**

Add to `src/db.rs`:

```rust
use anyhow::Result;
use rusqlite::Connection;

/// Database handle wrapping a SQLite connection.
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open (or create) the database at `path` and initialize schema.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    /// Open an in-memory database (for tests).
    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        todo!()
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_in_memory_creates_tables() {
        let db = Database::open_in_memory().unwrap();
        // sessions table exists
        let count: i64 = db.conn().query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='sessions'",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1);

        // chunks table exists
        let count: i64 = db.conn().query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks'",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1);

        // chunks_fts virtual table exists
        let count: i64 = db.conn().query_row(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='chunks_fts'",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_insert_and_query_session() {
        let db = Database::open_in_memory().unwrap();
        db.insert_session("sess-1", "my-project", "global", Some("2026-03-26T00:00:00Z")).unwrap();

        let project: String = db.conn().query_row(
            "SELECT project FROM sessions WHERE session_id = ?1",
            ["sess-1"],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(project, "my-project");
    }

    #[test]
    fn test_insert_and_query_chunk() {
        let db = Database::open_in_memory().unwrap();
        db.insert_session("sess-1", "my-project", "global", None).unwrap();
        db.insert_chunk(
            "sess-1",
            Some("uuid-1"),
            "How do I configure Docker?",
            "You can use docker-compose.yml to define services...",
            Some("2026-03-26T00:00:00Z"),
            42,
        ).unwrap();

        let question: String = db.conn().query_row(
            "SELECT question FROM chunks WHERE uuid = ?1",
            ["uuid-1"],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(question, "How do I configure Docker?");
    }

    #[test]
    fn test_duplicate_uuid_skipped() {
        let db = Database::open_in_memory().unwrap();
        db.insert_session("sess-1", "my-project", "global", None).unwrap();
        db.insert_chunk("sess-1", Some("uuid-1"), "q1", "a1", None, 10).unwrap();
        // Inserting same UUID again should not error
        let result = db.insert_chunk("sess-1", Some("uuid-1"), "q2", "a2", None, 10);
        assert!(result.is_ok());

        // Should still have only one chunk
        let count: i64 = db.conn().query_row(
            "SELECT count(*) FROM chunks",
            [],
            |row| row.get(0),
        ).unwrap();
        assert_eq!(count, 1);
    }

    #[test]
    fn test_fts5_search() {
        let db = Database::open_in_memory().unwrap();
        db.insert_session("sess-1", "my-project", "global", None).unwrap();
        db.insert_chunk("sess-1", Some("u1"), "Docker設定", "docker-compose.ymlを使います", None, 10).unwrap();
        db.insert_chunk("sess-1", Some("u2"), "Rust入門", "cargo buildでビルドします", None, 10).unwrap();

        let results = db.fts_search("Docker", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].question.contains("Docker"));
    }
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cargo test db::tests`
Expected: FAIL — `init_schema` has `todo!()`, missing methods

- [ ] **Step 3: Implement Database methods**

Replace the `todo!()` in `init_schema` and add insert/search methods:

```rust
use anyhow::Result;
use rusqlite::{params, Connection};

pub struct Database {
    conn: Connection,
}

/// A row returned by search operations.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub chunk_id: i64,
    pub session_id: String,
    pub project: String,
    pub scope: String,
    pub question: String,
    pub answer: String,
    pub timestamp: Option<String>,
    pub rank: f64,
}

impl Database {
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;")?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    pub fn open_in_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        let db = Self { conn };
        db.init_schema()?;
        Ok(db)
    }

    fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                session_id  TEXT PRIMARY KEY,
                project     TEXT NOT NULL,
                scope       TEXT NOT NULL DEFAULT 'global',
                started_at  TEXT,
                imported_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL REFERENCES sessions(session_id),
                uuid        TEXT,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                timestamp   TEXT,
                token_count INTEGER
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_uuid
                ON chunks(uuid) WHERE uuid IS NOT NULL;

            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                question,
                answer,
                content_rowid='id',
                content='chunks',
                tokenize='trigram'
            );

            -- Triggers to keep FTS index in sync
            CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, question, answer)
                VALUES (new.id, new.question, new.answer);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, question, answer)
                VALUES ('delete', old.id, old.question, old.answer);
            END;

            CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, question, answer)
                VALUES ('delete', old.id, old.question, old.answer);
                INSERT INTO chunks_fts(rowid, question, answer)
                VALUES (new.id, new.question, new.answer);
            END;"
        )?;
        Ok(())
    }

    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    /// Insert a session record. Returns Ok(false) if session already exists.
    pub fn insert_session(
        &self,
        session_id: &str,
        project: &str,
        scope: &str,
        started_at: Option<&str>,
    ) -> Result<bool> {
        let changed = self.conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, project, scope, started_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![session_id, project, scope, started_at],
        )?;
        Ok(changed > 0)
    }

    /// Insert a chunk. Skips if uuid already exists. Returns the chunk id or None if skipped.
    pub fn insert_chunk(
        &self,
        session_id: &str,
        uuid: Option<&str>,
        question: &str,
        answer: &str,
        timestamp: Option<&str>,
        token_count: i64,
    ) -> Result<Option<i64>> {
        if let Some(u) = uuid {
            let exists: bool = self.conn.query_row(
                "SELECT EXISTS(SELECT 1 FROM chunks WHERE uuid = ?1)",
                [u],
                |row| row.get(0),
            )?;
            if exists {
                return Ok(None);
            }
        }
        self.conn.execute(
            "INSERT INTO chunks (session_id, uuid, question, answer, timestamp, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![session_id, uuid, question, answer, timestamp, token_count],
        )?;
        Ok(Some(self.conn.last_insert_rowid()))
    }

    /// Full-text search via FTS5 trigram index.
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<ChunkRow>> {
        let escaped = query.replace('"', "\"\"");
        let fts_query = format!("\"{}\"", escaped);

        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.session_id, s.project, s.scope,
                    c.question, c.answer, c.timestamp, f.rank
             FROM chunks_fts f
             JOIN chunks c ON c.id = f.rowid
             JOIN sessions s ON s.session_id = c.session_id
             WHERE chunks_fts MATCH ?1
             ORDER BY f.rank
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![fts_query, limit as i64], |row| {
            Ok(ChunkRow {
                chunk_id: row.get(0)?,
                session_id: row.get(1)?,
                project: row.get(2)?,
                scope: row.get(3)?,
                question: row.get(4)?,
                answer: row.get(5)?,
                timestamp: row.get(6)?,
                rank: row.get(7)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Get the most recent chunks for a project (for recall query building).
    pub fn recent_chunks(&self, project: &str, limit: usize) -> Result<Vec<ChunkRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.session_id, s.project, s.scope,
                    c.question, c.answer, c.timestamp, 0.0 as rank
             FROM chunks c
             JOIN sessions s ON s.session_id = c.session_id
             WHERE s.project = ?1
             ORDER BY c.timestamp DESC
             LIMIT ?2"
        )?;

        let rows = stmt.query_map(params![project, limit as i64], |row| {
            Ok(ChunkRow {
                chunk_id: row.get(0)?,
                session_id: row.get(1)?,
                project: row.get(2)?,
                scope: row.get(3)?,
                question: row.get(4)?,
                answer: row.get(5)?,
                timestamp: row.get(6)?,
                rank: row.get(7)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Count sessions and chunks.
    pub fn stats(&self) -> Result<(i64, i64)> {
        let sessions: i64 = self.conn.query_row(
            "SELECT count(*) FROM sessions", [], |row| row.get(0),
        )?;
        let chunks: i64 = self.conn.query_row(
            "SELECT count(*) FROM chunks", [], |row| row.get(0),
        )?;
        Ok((sessions, chunks))
    }

    /// Check if a session already exists.
    pub fn session_exists(&self, session_id: &str) -> Result<bool> {
        let exists: bool = self.conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id = ?1)",
            [session_id],
            |row| row.get(0),
        )?;
        Ok(exists)
    }
}
```

- [ ] **Step 4: Add `mod db` to main.rs**

Add `mod db;` at the top of `src/main.rs`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `cargo test db::tests`
Expected: All 5 tests pass

- [ ] **Step 6: Commit**

```bash
git add src/db.rs src/main.rs
git commit -m "feat: add SQLite database layer with FTS5 trigram search"
```

---

### Task 3: Noise Filtering

**Files:**
- Create: `src/filter.rs`
- Modify: `src/main.rs` (add `mod filter`)

- [ ] **Step 1: Write tests for filter module**

```rust
use std::collections::HashSet;

/// Tool names whose results should be stripped from content.
const READ_ONLY_TOOLS: &[&str] = &[
    "Read", "Glob", "Grep", "LSP", "ToolSearch",
    "TaskGet", "TaskOutput", "TaskList",
];

/// XML-style system tags to remove from content.
const SYSTEM_TAGS: &[&str] = &[
    "system-reminder",
    "local-command-caveat",
    "local-command-stdout",
    "command-name",
    "command-message",
    "command-args",
    "task-notification",
];

/// Check if a JSONL record type should be kept.
pub fn is_relevant_record_type(record_type: &str) -> bool {
    matches!(record_type, "user" | "assistant")
}

/// Check if a tool use should be preserved (code-modifying tools only).
pub fn is_preserved_tool(tool_name: &str) -> bool {
    !READ_ONLY_TOOLS.contains(&tool_name)
}

/// Remove system XML tags from text content.
pub fn strip_system_tags(content: &str) -> String {
    let mut result = content.to_string();
    for tag in SYSTEM_TAGS {
        // Remove <tag>...</tag> blocks (non-greedy)
        let open = format!("<{}", tag);
        while let Some(start) = result.find(&open) {
            let close_tag = format!("</{}>", tag);
            if let Some(end) = result[start..].find(&close_tag) {
                let end_pos = start + end + close_tag.len();
                result.replace_range(start..end_pos, "");
            } else {
                break;
            }
        }
    }
    result.trim().to_string()
}

/// Check if content is empty or whitespace-only after filtering.
pub fn is_empty_content(content: &str) -> bool {
    content.trim().is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relevant_record_types() {
        assert!(is_relevant_record_type("user"));
        assert!(is_relevant_record_type("assistant"));
        assert!(!is_relevant_record_type("progress"));
        assert!(!is_relevant_record_type("queue-operation"));
        assert!(!is_relevant_record_type("system"));
    }

    #[test]
    fn test_preserved_tools() {
        assert!(is_preserved_tool("Edit"));
        assert!(is_preserved_tool("Write"));
        assert!(is_preserved_tool("Bash"));
        assert!(!is_preserved_tool("Read"));
        assert!(!is_preserved_tool("Glob"));
        assert!(!is_preserved_tool("Grep"));
        assert!(!is_preserved_tool("ToolSearch"));
    }

    #[test]
    fn test_strip_system_tags() {
        let input = "Hello <system-reminder>hidden stuff</system-reminder> World";
        assert_eq!(strip_system_tags(input), "Hello  World");
    }

    #[test]
    fn test_strip_multiple_tags() {
        let input = "<command-name>test</command-name>Real content<command-args>args</command-args>";
        assert_eq!(strip_system_tags(input), "Real content");
    }

    #[test]
    fn test_strip_nested_content() {
        let input = "Before <local-command-stdout>line1\nline2\nline3</local-command-stdout> After";
        assert_eq!(strip_system_tags(input), "Before  After");
    }

    #[test]
    fn test_empty_content() {
        assert!(is_empty_content(""));
        assert!(is_empty_content("   "));
        assert!(is_empty_content("\n\t "));
        assert!(!is_empty_content("hello"));
    }
}
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `cargo test filter::tests`
Expected: All 6 tests pass

- [ ] **Step 3: Add `mod filter` to main.rs**

- [ ] **Step 4: Commit**

```bash
git add src/filter.rs src/main.rs
git commit -m "feat: add noise filtering for JSONL content"
```

---

### Task 4: JSONL Parsing + Q&A Chunk Splitting

**Files:**
- Create: `src/chunk.rs`
- Create: `tests/fixtures/sample_session.jsonl`
- Modify: `src/main.rs` (add `mod chunk`)

- [ ] **Step 1: Create test fixture**

Create `tests/fixtures/sample_session.jsonl` with minimal Claude Code session records (one user, one assistant, one progress, one queue-operation):

```jsonl
{"type":"queue-operation","operation":"enqueue","timestamp":"2026-03-26T00:00:00Z","sessionId":"test-session-1"}
{"type":"progress","data":{"type":"hook_progress","hookEvent":"SessionStart"},"timestamp":"2026-03-26T00:00:01Z","sessionId":"test-session-1","uuid":"p1","cwd":"/tmp/my-project"}
{"type":"user","message":{"role":"user","content":"Dockerの設定方法を教えて"},"timestamp":"2026-03-26T00:00:02Z","sessionId":"test-session-1","uuid":"u1","parentUuid":"p1","cwd":"/tmp/my-project"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"docker-compose.ymlを使ってサービスを定義します。"}]},"timestamp":"2026-03-26T00:00:03Z","sessionId":"test-session-1","uuid":"a1","cwd":"/tmp/my-project"}
{"type":"user","message":{"role":"user","content":"ビルドキャッシュが効かない"},"timestamp":"2026-03-26T00:00:04Z","sessionId":"test-session-1","uuid":"u2","cwd":"/tmp/my-project"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"thinking","thinking":"Let me think about this..."},{"type":"text","text":"BuildKitのインラインキャッシュを使ってください。"},{"type":"tool_use","id":"t1","name":"Read","input":{"path":"/tmp/test"}}]},"timestamp":"2026-03-26T00:00:05Z","sessionId":"test-session-1","uuid":"a2","cwd":"/tmp/my-project"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"tool_result","tool_use_id":"t1","content":"file content here"}]},"timestamp":"2026-03-26T00:00:06Z","sessionId":"test-session-1","uuid":"a3","cwd":"/tmp/my-project"}
{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"ファイルの内容を確認しました。<system-reminder>This is hidden</system-reminder>設定を変更してください。"}]},"timestamp":"2026-03-26T00:00:07Z","sessionId":"test-session-1","uuid":"a4","cwd":"/tmp/my-project"}
```

- [ ] **Step 2: Write chunk module with tests**

```rust
use anyhow::Result;
use serde_json::Value;
use crate::filter;

/// A parsed Q&A chunk from a session.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub session_id: String,
    pub uuid: String,
    pub question: String,
    pub answer: String,
    pub timestamp: Option<String>,
}

/// A raw parsed message from JSONL.
#[derive(Debug)]
struct Message {
    role: String, // "user" or "assistant"
    content: String,
    uuid: String,
    timestamp: Option<String>,
    session_id: String,
}

/// Parse a JSONL file and return Q&A chunks.
pub fn parse_session(jsonl_content: &str) -> Result<Vec<Chunk>> {
    let messages = extract_messages(jsonl_content)?;
    Ok(pair_into_chunks(messages))
}

/// Extract user/assistant messages from JSONL lines.
fn extract_messages(jsonl_content: &str) -> Result<Vec<Message>> {
    let mut messages = Vec::new();

    for line in jsonl_content.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let record: Value = serde_json::from_str(line)?;

        let record_type = record.get("type").and_then(|v| v.as_str()).unwrap_or("");
        if !filter::is_relevant_record_type(record_type) {
            continue;
        }

        let session_id = record.get("sessionId")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let uuid = record.get("uuid")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let timestamp = record.get("timestamp")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let message = &record["message"];
        let role = message.get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let content = extract_content(message)?;
        let content = filter::strip_system_tags(&content);

        if filter::is_empty_content(&content) {
            continue;
        }

        messages.push(Message {
            role,
            content,
            uuid,
            timestamp,
            session_id,
        });
    }

    Ok(messages)
}

/// Extract text content from a message's content field.
/// Handles both string content (user) and array content (assistant).
fn extract_content(message: &Value) -> Result<String> {
    let content = &message["content"];

    // User messages: content is a string
    if let Some(s) = content.as_str() {
        return Ok(s.to_string());
    }

    // Assistant messages: content is an array of blocks
    if let Some(blocks) = content.as_array() {
        let mut parts = Vec::new();
        for block in blocks {
            let block_type = block.get("type").and_then(|v| v.as_str()).unwrap_or("");
            match block_type {
                "text" => {
                    if let Some(text) = block.get("text").and_then(|v| v.as_str()) {
                        parts.push(text.to_string());
                    }
                }
                "tool_use" => {
                    let tool_name = block.get("name").and_then(|v| v.as_str()).unwrap_or("");
                    if filter::is_preserved_tool(tool_name) {
                        let input = &block["input"];
                        parts.push(format!("[tool_use: {}] {}", tool_name, input));
                    }
                    // Skip read-only tools entirely
                }
                // Skip "thinking", "tool_result", and other block types
                _ => {}
            }
        }
        return Ok(parts.join("\n"));
    }

    Ok(String::new())
}

/// Pair consecutive user/assistant messages into Q&A chunks.
fn pair_into_chunks(messages: Vec<Message>) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut i = 0;

    while i < messages.len() {
        if messages[i].role == "user" {
            let question = messages[i].content.clone();
            let uuid = messages[i].uuid.clone();
            let timestamp = messages[i].timestamp.clone();
            let session_id = messages[i].session_id.clone();

            // Collect all subsequent assistant messages as the answer
            let mut answer_parts = Vec::new();
            let mut j = i + 1;
            while j < messages.len() && messages[j].role == "assistant" {
                answer_parts.push(messages[j].content.clone());
                j += 1;
            }

            if !answer_parts.is_empty() {
                let answer = answer_parts.join("\n");
                chunks.push(Chunk {
                    session_id,
                    uuid,
                    question,
                    answer,
                    timestamp,
                });
            }
            i = j;
        } else {
            i += 1;
        }
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_session_from_fixture() {
        let content = include_str!("../tests/fixtures/sample_session.jsonl");
        let chunks = parse_session(content).unwrap();

        assert_eq!(chunks.len(), 2, "Should produce 2 Q&A chunks");

        assert!(chunks[0].question.contains("Docker"));
        assert!(chunks[0].answer.contains("docker-compose"));

        assert!(chunks[1].question.contains("ビルドキャッシュ"));
        assert!(chunks[1].answer.contains("BuildKit"));
    }

    #[test]
    fn test_system_tags_stripped() {
        let content = include_str!("../tests/fixtures/sample_session.jsonl");
        let chunks = parse_session(content).unwrap();

        // The second Q&A's answer should not contain system-reminder
        for chunk in &chunks {
            assert!(!chunk.answer.contains("system-reminder"));
            assert!(!chunk.answer.contains("This is hidden"));
        }
    }

    #[test]
    fn test_tool_result_excluded() {
        let content = include_str!("../tests/fixtures/sample_session.jsonl");
        let chunks = parse_session(content).unwrap();

        for chunk in &chunks {
            assert!(!chunk.answer.contains("file content here"));
        }
    }

    #[test]
    fn test_read_tool_excluded() {
        let content = include_str!("../tests/fixtures/sample_session.jsonl");
        let chunks = parse_session(content).unwrap();

        for chunk in &chunks {
            assert!(!chunk.answer.contains("[tool_use: Read]"));
        }
    }

    #[test]
    fn test_pair_chunks_handles_multiple_assistant_messages() {
        let messages = vec![
            Message { role: "user".into(), content: "Q1".into(), uuid: "u1".into(), timestamp: None, session_id: "s1".into() },
            Message { role: "assistant".into(), content: "A1 part1".into(), uuid: "a1".into(), timestamp: None, session_id: "s1".into() },
            Message { role: "assistant".into(), content: "A1 part2".into(), uuid: "a2".into(), timestamp: None, session_id: "s1".into() },
            Message { role: "user".into(), content: "Q2".into(), uuid: "u2".into(), timestamp: None, session_id: "s1".into() },
            Message { role: "assistant".into(), content: "A2".into(), uuid: "a3".into(), timestamp: None, session_id: "s1".into() },
        ];

        let chunks = pair_into_chunks(messages);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].answer, "A1 part1\nA1 part2");
        assert_eq!(chunks[1].answer, "A2");
    }
}
```

- [ ] **Step 3: Add `mod chunk` to main.rs**

- [ ] **Step 4: Run tests**

Run: `cargo test chunk::tests`
Expected: All 5 tests pass

- [ ] **Step 5: Commit**

```bash
git add src/chunk.rs src/main.rs tests/fixtures/sample_session.jsonl
git commit -m "feat: add JSONL parser with Q&A chunk splitting and noise filtering"
```

---

### Task 5: Save Command (without embedding)

**Files:**
- Create: `src/cmd/save.rs`
- Modify: `src/cmd/mod.rs`

- [ ] **Step 1: Implement save command**

Create `src/cmd/save.rs`:

```rust
use std::fs;
use std::path::{Path, PathBuf};
use anyhow::{Context, Result};
use crate::db::Database;
use crate::chunk;

/// Resolve the kiok data directory (~/.kiok/).
pub fn data_dir() -> Result<PathBuf> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    Ok(home.join(".kiok"))
}

/// Resolve the database path.
pub fn db_path() -> Result<PathBuf> {
    Ok(data_dir()?.join("memory.db"))
}

/// Find the most recent JSONL session file for a project.
fn find_latest_session(project_path: &str) -> Result<Option<PathBuf>> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    let encoded = project_path.replace(|c: char| !c.is_alphanumeric(), "-");
    let sessions_dir = home.join(".claude").join("projects").join(&encoded);

    if !sessions_dir.exists() {
        return Ok(None);
    }

    let mut files: Vec<_> = fs::read_dir(&sessions_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "jsonl"))
        .collect();

    files.sort_by_key(|e| std::cmp::Reverse(e.metadata().ok().and_then(|m| m.modified().ok())));

    Ok(files.first().map(|e| e.path()))
}

/// Normalize a project path to a short project name.
/// e.g., "/Users/kenta/workspace/my-app" -> "my-app"
fn project_name(project_path: &str) -> String {
    Path::new(project_path)
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| project_path.to_string())
}

/// Run the save command.
pub fn run(project_path: &str) -> Result<()> {
    let session_path = find_latest_session(project_path)?;
    let session_path = match session_path {
        Some(p) => p,
        None => {
            eprintln!("No session found for project: {}", project_path);
            return Ok(());
        }
    };

    let session_id = session_path
        .file_stem()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_default();

    let db_path = db_path()?;
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let db = Database::open(&db_path)?;

    if db.session_exists(&session_id)? {
        eprintln!("Session {} already saved, skipping", session_id);
        return Ok(());
    }

    let content = fs::read_to_string(&session_path)
        .with_context(|| format!("Failed to read {}", session_path.display()))?;

    let chunks = chunk::parse_session(&content)?;
    if chunks.is_empty() {
        eprintln!("No chunks extracted from session {}", session_id);
        return Ok(());
    }

    let project = project_name(project_path);
    let started_at = chunks.first().and_then(|c| c.timestamp.as_deref());

    // TODO: Policy check will be added later
    let scope = "global";

    db.insert_session(&session_id, &project, scope, started_at)?;

    let mut saved = 0;
    for chunk in &chunks {
        if db.insert_chunk(
            &session_id,
            Some(&chunk.uuid),
            &chunk.question,
            &chunk.answer,
            chunk.timestamp.as_deref(),
            chunk.question.len() as i64 + chunk.answer.len() as i64, // approximate token count
        )?.is_some() {
            saved += 1;
        }
    }

    eprintln!("Saved {} chunks from session {}", saved, session_id);
    Ok(())
}
```

- [ ] **Step 2: Add `dirs` dependency to Cargo.toml**

Add to `[dependencies]`:
```toml
dirs = "6"
```

- [ ] **Step 3: Update cmd/mod.rs to dispatch save command**

```rust
use anyhow::Result;

pub mod save;

pub fn run(command: crate::Commands) -> Result<()> {
    match command {
        crate::Commands::Save { project } => save::run(&project),
        crate::Commands::Recall { project, count } => {
            eprintln!("recall: project={}, count={}", project, count);
            Ok(())
        }
        crate::Commands::Search { query, project, count } => {
            eprintln!("search: query={}, project={:?}, count={}", query, project, count);
            Ok(())
        }
        crate::Commands::Import => {
            eprintln!("import: not yet implemented");
            Ok(())
        }
        crate::Commands::Stats => {
            eprintln!("stats: not yet implemented");
            Ok(())
        }
        crate::Commands::Setup => {
            eprintln!("setup: not yet implemented");
            Ok(())
        }
    }
}
```

- [ ] **Step 4: Build and verify**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/cmd/mod.rs src/cmd/save.rs
git commit -m "feat: implement save command (parse, chunk, store without embedding)"
```

---

### Task 6: FTS5 Search + Search Command

**Files:**
- Create: `src/search/mod.rs`
- Create: `src/search/fts.rs`
- Create: `src/cmd/search.rs`
- Modify: `src/main.rs` (add `mod search`)
- Modify: `src/cmd/mod.rs` (add search dispatch)

- [ ] **Step 1: Create search/fts.rs**

```rust
use anyhow::Result;
use crate::db::{ChunkRow, Database};

/// Result from a single search backend, with a normalized rank.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub chunk: ChunkRow,
    /// Rank position (1-based). Lower is better.
    pub rank: usize,
}

/// Run FTS5 keyword search.
pub fn search(db: &Database, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
    let rows = db.fts_search(query, limit)?;
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| SearchResult {
            chunk,
            rank: i + 1,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fts_search_returns_ranked_results() {
        let db = Database::open_in_memory().unwrap();
        db.insert_session("s1", "proj", "global", None).unwrap();
        db.insert_chunk("s1", Some("u1"), "Dockerの設定", "compose.ymlを使う", None, 10).unwrap();
        db.insert_chunk("s1", Some("u2"), "Rustの入門", "cargo buildを使う", None, 10).unwrap();
        db.insert_chunk("s1", Some("u3"), "Dockerのネットワーク", "bridgeモードで接続", None, 10).unwrap();

        let results = search(&db, "Docker", 10).unwrap();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].rank, 1);
        assert_eq!(results[1].rank, 2);
    }
}
```

- [ ] **Step 2: Create search/mod.rs**

```rust
pub mod fts;
```

- [ ] **Step 3: Create cmd/search.rs**

```rust
use anyhow::Result;
use crate::cmd::save;
use crate::db::Database;
use crate::search::fts;

pub fn run(query: &str, project: Option<&str>, count: usize) -> Result<()> {
    let db_path = save::db_path()?;
    if !db_path.exists() {
        eprintln!("No database found. Run `kiok save` or `kiok import` first.");
        return Ok(());
    }

    let db = Database::open(&db_path)?;
    let results = fts::search(&db, query, count)?;

    if results.is_empty() {
        eprintln!("No results found for: {}", query);
        return Ok(());
    }

    for result in &results {
        let c = &result.chunk;
        let ts = c.timestamp.as_deref().unwrap_or("unknown");
        let project_display = &c.project;

        // Filter by project if specified
        if let Some(p) = project {
            if c.project != p {
                continue;
            }
        }

        println!("[{}] {} | {}", c.scope, project_display, ts);
        println!("Q: {}", truncate(&c.question, 200));
        println!("A: {}", truncate(&c.answer, 300));
        println!("---");
    }

    Ok(())
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}
```

- [ ] **Step 4: Update cmd/mod.rs and main.rs**

Add `mod search;` to `src/main.rs`.
Add `pub mod search;` to `src/cmd/mod.rs` (rename to avoid collision with top-level search).
Actually, name the cmd module `search_cmd` to avoid name collision:

Rename file to `src/cmd/search_cmd.rs` and update `cmd/mod.rs`:

```rust
pub mod save;
pub mod search_cmd;
```

Update the dispatch in `cmd/mod.rs`:

```rust
crate::Commands::Search { query, project, count } => {
    search_cmd::run(&query, project.as_deref(), count)
}
```

- [ ] **Step 5: Run tests**

Run: `cargo test`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/search/ src/cmd/search_cmd.rs src/cmd/mod.rs src/main.rs
git commit -m "feat: add FTS5 keyword search and search command"
```

---

### Task 7: Embedding Backend Trait + ONNX Implementation

**Files:**
- Create: `src/embed/mod.rs`
- Create: `src/embed/onnx.rs`
- Modify: `Cargo.toml` (add ort, tokenizers)
- Modify: `src/main.rs` (add `mod embed`)

- [ ] **Step 1: Add dependencies to Cargo.toml**

```toml
ort = { version = "2.0.0-rc.12", features = ["load-dynamic"] }
tokenizers = { version = "0.21", default-features = false, features = ["onig"] }
```

Note: `load-dynamic` avoids bundling the ONNX Runtime shared library. The user must have it installed or we download it at setup time.

- [ ] **Step 2: Create embed/mod.rs with trait definition**

```rust
pub mod onnx;

use anyhow::Result;

/// Trait for embedding backends.
pub trait EmbeddingBackend {
    /// Embed a batch of texts into vectors.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Return the dimensionality of the output vectors.
    fn dimensions(&self) -> usize;
}
```

- [ ] **Step 3: Create embed/onnx.rs**

```rust
use anyhow::{Context, Result};
use ort::session::Session;
use tokenizers::Tokenizer;
use std::path::Path;

use super::EmbeddingBackend;

pub struct OnnxBackend {
    session: Session,
    tokenizer: Tokenizer,
    dimensions: usize,
}

impl OnnxBackend {
    /// Load an ONNX model and its tokenizer.
    /// `model_dir` should contain `model.onnx` and `tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()?
            .with_intra_threads(4)?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model from {}", model_path.display()))?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Determine dimensions from the model output shape
        let output_info = &session.outputs[0];
        let dimensions = match &output_info.output_type {
            ort::value::ValueType::Tensor { dimensions: dims, .. } => {
                dims.last().copied().unwrap_or(1024) as usize
            }
            _ => 1024,
        };

        Ok(Self {
            session,
            tokenizer,
            dimensions,
        })
    }
}

impl EmbeddingBackend for OnnxBackend {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let encodings = self.tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        let batch_size = encodings.len();
        let max_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap_or(0);

        // Build padded input tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            for (j, (&id, &m)) in ids.iter().zip(mask.iter()).enumerate() {
                input_ids[i * max_len + j] = id as i64;
                attention_mask[i * max_len + j] = m as i64;
            }
        }

        let input_ids_array = ndarray::Array2::from_shape_vec(
            (batch_size, max_len),
            input_ids,
        )?;
        let attention_mask_array = ndarray::Array2::from_shape_vec(
            (batch_size, max_len),
            attention_mask,
        )?;

        let outputs = self.session.run(
            ort::inputs![
                "input_ids" => input_ids_array,
                "attention_mask" => attention_mask_array,
            ]?
        )?;

        // Extract embeddings: output shape is [batch_size, seq_len, hidden_dim]
        // Apply mean pooling over the sequence dimension
        let output_tensor = outputs[0].try_extract_tensor::<f32>()?;
        let output_view = output_tensor.view();
        let shape = output_view.shape();

        let mut embeddings = Vec::with_capacity(batch_size);

        if shape.len() == 3 {
            // [batch, seq_len, hidden_dim] — mean pooling
            let hidden_dim = shape[2];
            for i in 0..batch_size {
                let seq_len = encodings[i].get_attention_mask().iter()
                    .filter(|&&m| m == 1).count();
                let mut embedding = vec![0.0f32; hidden_dim];
                for j in 0..seq_len {
                    for k in 0..hidden_dim {
                        embedding[k] += output_view[[i, j, k]];
                    }
                }
                // Divide by seq_len for mean
                for v in &mut embedding {
                    *v /= seq_len as f32;
                }
                // L2 normalize
                let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for v in &mut embedding {
                        *v /= norm;
                    }
                }
                embeddings.push(embedding);
            }
        } else if shape.len() == 2 {
            // [batch, hidden_dim] — already pooled
            let hidden_dim = shape[1];
            for i in 0..batch_size {
                let mut embedding = vec![0.0f32; hidden_dim];
                for k in 0..hidden_dim {
                    embedding[k] = output_view[[i, k]];
                }
                embeddings.push(embedding);
            }
        }

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }
}
```

- [ ] **Step 4: Add ndarray dependency**

```toml
ndarray = "0.16"
```

- [ ] **Step 5: Add `mod embed` to main.rs**

- [ ] **Step 6: Verify build**

Run: `cargo build`
Expected: Compiles successfully (tests for OnnxBackend require an actual model, so no unit test yet — integration test in Task 14)

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml src/embed/ src/main.rs
git commit -m "feat: add embedding backend trait with ONNX Runtime implementation"
```

---

### Task 8: sqlite-vec Vector Search

**Files:**
- Modify: `Cargo.toml` (add sqlite-vec, zerocopy)
- Modify: `src/db.rs` (add vec0 table, embedding insert/search)
- Create: `src/search/vector.rs`
- Modify: `src/search/mod.rs`

- [ ] **Step 1: Add dependencies**

```toml
sqlite-vec = "0.1.6"
zerocopy = { version = "0.8", features = ["derive"] }
```

- [ ] **Step 2: Update db.rs — initialize sqlite-vec extension**

Add at the top of `db.rs`:

```rust
use sqlite_vec::sqlite3_vec_init;
use rusqlite::ffi::sqlite3_auto_extension;
```

Add before `Connection::open` in both `open` and `open_in_memory`:

```rust
// Register sqlite-vec extension (must be called before opening connection)
unsafe {
    sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
}
```

Add to `init_schema`:

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
    chunk_id INTEGER PRIMARY KEY,
    embedding FLOAT[1024]
);
```

- [ ] **Step 3: Add embedding insert method to Database**

```rust
/// Insert an embedding vector for a chunk.
pub fn insert_embedding(&self, chunk_id: i64, embedding: &[f32]) -> Result<()> {
    let blob = zerocopy::AsBytes::as_bytes(embedding);
    self.conn.execute(
        "INSERT INTO chunks_vec (chunk_id, embedding) VALUES (?1, ?2)",
        params![chunk_id, blob],
    )?;
    Ok(())
}

/// Vector nearest-neighbor search.
pub fn vec_search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<ChunkRow>> {
    let blob = zerocopy::AsBytes::as_bytes(query_embedding);
    let mut stmt = self.conn.prepare(
        "SELECT c.id, c.session_id, s.project, s.scope,
                c.question, c.answer, c.timestamp, v.distance
         FROM chunks_vec v
         JOIN chunks c ON c.id = v.chunk_id
         JOIN sessions s ON s.session_id = c.session_id
         WHERE v.embedding MATCH ?1
         AND k = ?2
         ORDER BY v.distance"
    )?;

    let rows = stmt.query_map(params![blob, limit as i64], |row| {
        Ok(ChunkRow {
            chunk_id: row.get(0)?,
            session_id: row.get(1)?,
            project: row.get(2)?,
            scope: row.get(3)?,
            question: row.get(4)?,
            answer: row.get(5)?,
            timestamp: row.get(6)?,
            rank: row.get(7)?,
        })
    })?;

    let mut results = Vec::new();
    for row in rows {
        results.push(row?);
    }
    Ok(results)
}
```

- [ ] **Step 4: Create search/vector.rs**

```rust
use anyhow::Result;
use crate::db::Database;
use crate::search::fts::SearchResult;

/// Run vector nearest-neighbor search.
pub fn search(db: &Database, query_embedding: &[f32], limit: usize) -> Result<Vec<SearchResult>> {
    let rows = db.vec_search(query_embedding, limit)?;
    Ok(rows
        .into_iter()
        .enumerate()
        .map(|(i, chunk)| SearchResult {
            chunk,
            rank: i + 1,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_search_returns_nearest() {
        let db = Database::open_in_memory().unwrap();
        db.insert_session("s1", "proj", "global", None).unwrap();

        let id1 = db.insert_chunk("s1", Some("u1"), "Docker設定", "composeを使う", None, 10).unwrap().unwrap();
        let id2 = db.insert_chunk("s1", Some("u2"), "Rust入門", "cargoを使う", None, 10).unwrap().unwrap();

        // Insert fake embeddings (4-dimensional for test simplicity)
        // Note: actual schema uses 1024-dim, but for unit tests we test the API
        // This test will only work if vec0 table dimension matches. We'll test with real dims in integration.
        let emb1 = vec![1.0f32; 1024];
        let emb2 = vec![0.0f32; 1024];
        db.insert_embedding(id1, &emb1).unwrap();
        db.insert_embedding(id2, &emb2).unwrap();

        // Query vector close to emb1
        let query = vec![0.9f32; 1024];
        let results = search(&db, &query, 2).unwrap();
        assert_eq!(results.len(), 2);
        // First result should be closer to the query (emb1)
        assert_eq!(results[0].chunk.chunk_id, id1);
    }
}
```

- [ ] **Step 5: Update search/mod.rs**

```rust
pub mod fts;
pub mod vector;
```

- [ ] **Step 6: Run tests**

Run: `cargo test`
Expected: All tests pass (including vector search test)

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml src/db.rs src/search/vector.rs src/search/mod.rs
git commit -m "feat: add sqlite-vec vector search with embedding storage"
```

---

### Task 9: RRF Score Fusion + Time Decay

**Files:**
- Create: `src/search/rrf.rs`
- Modify: `src/search/mod.rs`

- [ ] **Step 1: Implement RRF + time decay with tests**

```rust
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use crate::search::fts::SearchResult;

/// RRF constant (standard value from the literature).
const RRF_K: f64 = 60.0;

/// A scored result after RRF fusion and time decay.
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

/// Fuse two ranked lists using Reciprocal Rank Fusion.
pub fn fuse(
    fts_results: &[SearchResult],
    vec_results: &[SearchResult],
    half_life_days: f64,
    now: DateTime<Utc>,
) -> Vec<ScoredResult> {
    let mut scores: HashMap<i64, ScoredResult> = HashMap::new();

    // Add FTS scores
    for r in fts_results {
        let rrf_score = 1.0 / (RRF_K + r.rank as f64);
        let entry = scores.entry(r.chunk.chunk_id).or_insert_with(|| ScoredResult {
            chunk_id: r.chunk.chunk_id,
            session_id: r.chunk.session_id.clone(),
            project: r.chunk.project.clone(),
            scope: r.chunk.scope.clone(),
            question: r.chunk.question.clone(),
            answer: r.chunk.answer.clone(),
            timestamp: r.chunk.timestamp.clone(),
            score: 0.0,
        });
        entry.score += rrf_score;
    }

    // Add vector scores
    for r in vec_results {
        let rrf_score = 1.0 / (RRF_K + r.rank as f64);
        let entry = scores.entry(r.chunk.chunk_id).or_insert_with(|| ScoredResult {
            chunk_id: r.chunk.chunk_id,
            session_id: r.chunk.session_id.clone(),
            project: r.chunk.project.clone(),
            scope: r.chunk.scope.clone(),
            question: r.chunk.question.clone(),
            answer: r.chunk.answer.clone(),
            timestamp: r.chunk.timestamp.clone(),
            score: 0.0,
        });
        entry.score += rrf_score;
    }

    // Apply time decay
    let lambda = (2.0_f64).ln() / half_life_days;
    for result in scores.values_mut() {
        if let Some(ts) = &result.timestamp {
            if let Ok(parsed) = ts.parse::<DateTime<Utc>>() {
                let age_days = (now - parsed).num_seconds() as f64 / 86400.0;
                let decay = (-lambda * age_days).exp();
                result.score *= decay;
            }
        }
    }

    // Sort by score descending
    let mut results: Vec<_> = scores.into_values().collect();
    results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::ChunkRow;

    fn make_result(chunk_id: i64, rank: usize, timestamp: &str, project: &str) -> SearchResult {
        SearchResult {
            chunk: ChunkRow {
                chunk_id,
                session_id: "s1".into(),
                project: project.into(),
                scope: "global".into(),
                question: format!("q{}", chunk_id),
                answer: format!("a{}", chunk_id),
                timestamp: Some(timestamp.into()),
                rank: 0.0,
            },
            rank,
        }
    }

    #[test]
    fn test_rrf_fusion_combines_scores() {
        let now = "2026-03-26T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let ts = "2026-03-26T00:00:00Z"; // same as now, no decay

        let fts = vec![
            make_result(1, 1, ts, "proj"),
            make_result(2, 2, ts, "proj"),
        ];
        let vec = vec![
            make_result(2, 1, ts, "proj"), // chunk 2 appears in both
            make_result(3, 2, ts, "proj"),
        ];

        let results = fuse(&fts, &vec, 30.0, now);

        // Chunk 2 should have highest score (appears in both lists)
        assert_eq!(results[0].chunk_id, 2);

        // All 3 chunks should be present
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_time_decay_reduces_old_scores() {
        let now = "2026-03-26T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let recent = "2026-03-25T00:00:00Z"; // 1 day ago
        let old = "2026-02-24T00:00:00Z"; // 30 days ago (half-life)

        let fts = vec![
            make_result(1, 1, old, "proj"),    // old, rank 1
            make_result(2, 2, recent, "proj"), // recent, rank 2
        ];
        let vec = vec![];

        let results = fuse(&fts, &vec, 30.0, now);

        // Despite chunk 1 having better rank, chunk 2 should score higher
        // because chunk 1 is 30 days old (score halved)
        assert_eq!(results[0].chunk_id, 2);
    }

    #[test]
    fn test_no_timestamp_no_decay() {
        let now = "2026-03-26T00:00:00Z".parse::<DateTime<Utc>>().unwrap();
        let fts = vec![SearchResult {
            chunk: ChunkRow {
                chunk_id: 1,
                session_id: "s1".into(),
                project: "proj".into(),
                scope: "global".into(),
                question: "q".into(),
                answer: "a".into(),
                timestamp: None,
                rank: 0.0,
            },
            rank: 1,
        }];
        let vec = vec![];

        let results = fuse(&fts, &vec, 30.0, now);
        assert_eq!(results.len(), 1);
        // Score should be 1/(60+1) with no decay applied
        let expected = 1.0 / 61.0;
        assert!((results[0].score - expected).abs() < 1e-10);
    }
}
```

- [ ] **Step 2: Update search/mod.rs**

```rust
pub mod fts;
pub mod vector;
pub mod rrf;
```

- [ ] **Step 3: Run tests**

Run: `cargo test search::rrf::tests`
Expected: All 3 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/search/rrf.rs src/search/mod.rs
git commit -m "feat: add RRF score fusion with time decay"
```

---

### Task 10: Hybrid Search Orchestration

**Files:**
- Modify: `src/search/mod.rs` (add `hybrid_search` function)

- [ ] **Step 1: Implement hybrid search**

Add to `src/search/mod.rs`:

```rust
pub mod fts;
pub mod vector;
pub mod rrf;

use anyhow::Result;
use chrono::Utc;
use crate::db::Database;
use rrf::ScoredResult;

/// Configuration for hybrid search.
pub struct SearchConfig {
    pub count: usize,
    pub candidate_multiplier: usize,
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

/// Run hybrid search: FTS5 keyword + vector nearest-neighbor, fused with RRF.
/// `query_text` is used for FTS5.
/// `query_embedding` is used for vector search (None to skip vector search).
pub fn hybrid_search(
    db: &Database,
    query_text: &str,
    query_embedding: Option<&[f32]>,
    config: &SearchConfig,
) -> Result<Vec<ScoredResult>> {
    let candidate_limit = config.count * config.candidate_multiplier;

    // FTS5 keyword search
    let fts_results = fts::search(db, query_text, candidate_limit)?;

    // Vector search (if embedding is provided)
    let vec_results = if let Some(emb) = query_embedding {
        vector::search(db, emb, candidate_limit)?
    } else {
        Vec::new()
    };

    let now = Utc::now();
    let mut results = rrf::fuse(&fts_results, &vec_results, config.half_life_days, now);

    results.truncate(config.count);
    Ok(results)
}

/// Run FTS-only search (for when embedding model is not available).
pub fn keyword_search(
    db: &Database,
    query_text: &str,
    config: &SearchConfig,
) -> Result<Vec<ScoredResult>> {
    hybrid_search(db, query_text, None, config)
}
```

- [ ] **Step 2: Run all tests**

Run: `cargo test`
Expected: All tests pass

- [ ] **Step 3: Commit**

```bash
git add src/search/mod.rs
git commit -m "feat: add hybrid search orchestration (FTS5 + vector + RRF)"
```

---

### Task 11: Policy System

**Files:**
- Create: `src/policy.rs`
- Modify: `src/main.rs` (add `mod policy`)

- [ ] **Step 1: Implement policy with tests**

```rust
use anyhow::Result;
use serde::Deserialize;
use std::path::Path;

/// Memory scope for a project.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    Global,
    Project,
    Isolated,
}

impl Scope {
    pub fn as_str(&self) -> &'static str {
        match self {
            Scope::Global => "global",
            Scope::Project => "project",
            Scope::Isolated => "isolated",
        }
    }
}

impl Default for Scope {
    fn default() -> Self {
        Scope::Global
    }
}

#[derive(Debug, Deserialize)]
struct PolicyFile {
    scope: Scope,
}

/// Load the policy for a project directory.
/// Looks for `.claude/memory-policy.json` in the project root.
/// Returns `Scope::Global` if no policy file exists.
pub fn load_scope(project_path: &str) -> Result<Scope> {
    let policy_path = Path::new(project_path)
        .join(".claude")
        .join("memory-policy.json");

    if !policy_path.exists() {
        return Ok(Scope::default());
    }

    let content = std::fs::read_to_string(&policy_path)?;
    let policy: PolicyFile = serde_json::from_str(&content)?;
    Ok(policy.scope)
}

/// Check if a chunk with `chunk_scope` should be visible
/// from a project whose scope is `viewer_scope` and project is `viewer_project`.
pub fn is_visible(
    chunk_scope: &str,
    chunk_project: &str,
    viewer_scope: Scope,
    viewer_project: &str,
) -> bool {
    // Same project is always visible
    if chunk_project == viewer_project {
        return true;
    }

    // Cross-project visibility depends on both scopes
    match viewer_scope {
        Scope::Isolated => false, // isolated sees only own project
        Scope::Project | Scope::Global => {
            // Can see other projects' chunks only if they are global
            chunk_scope == "global"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_scope_is_global() {
        assert_eq!(Scope::default(), Scope::Global);
    }

    #[test]
    fn test_same_project_always_visible() {
        assert!(is_visible("isolated", "proj-a", Scope::Isolated, "proj-a"));
        assert!(is_visible("project", "proj-a", Scope::Global, "proj-a"));
    }

    #[test]
    fn test_isolated_blocks_cross_project() {
        assert!(!is_visible("global", "proj-b", Scope::Isolated, "proj-a"));
    }

    #[test]
    fn test_project_scope_sees_global() {
        assert!(is_visible("global", "proj-b", Scope::Project, "proj-a"));
        assert!(!is_visible("project", "proj-b", Scope::Project, "proj-a"));
        assert!(!is_visible("isolated", "proj-b", Scope::Project, "proj-a"));
    }

    #[test]
    fn test_global_scope_sees_global() {
        assert!(is_visible("global", "proj-b", Scope::Global, "proj-a"));
        assert!(!is_visible("project", "proj-b", Scope::Global, "proj-a"));
    }

    #[test]
    fn test_parse_policy_json() {
        let json = r#"{"scope": "isolated"}"#;
        let policy: PolicyFile = serde_json::from_str(json).unwrap();
        assert_eq!(policy.scope, Scope::Isolated);
    }
}
```

- [ ] **Step 2: Add `mod policy` to main.rs**

- [ ] **Step 3: Run tests**

Run: `cargo test policy::tests`
Expected: All 6 tests pass

- [ ] **Step 4: Commit**

```bash
git add src/policy.rs src/main.rs
git commit -m "feat: add policy system with global/project/isolated scopes"
```

---

### Task 12: Recall Command

**Files:**
- Create: `src/cmd/recall.rs`
- Modify: `src/cmd/mod.rs`

- [ ] **Step 1: Implement recall command**

```rust
use anyhow::Result;
use crate::cmd::save;
use crate::db::Database;
use crate::policy;
use crate::search::{self, SearchConfig, rrf::ScoredResult};

pub fn run(project_path: &str, count: usize) -> Result<()> {
    let db_path = save::db_path()?;
    if !db_path.exists() {
        // No database yet — nothing to recall
        return Ok(());
    }

    let db = Database::open(&db_path)?;
    let project = save::project_name(project_path);

    // Build query from recent chunks
    let query = build_query(&db, &project)?;
    if query.is_empty() {
        return Ok(());
    }

    let scope = policy::load_scope(project_path)?;

    let config = SearchConfig {
        count: count * 2, // fetch extra, then filter by policy
        ..Default::default()
    };

    // For now, FTS-only search. Vector search added when embedding is integrated.
    let results = search::keyword_search(&db, &query, &config)?;

    // Filter by policy
    let visible: Vec<_> = results
        .into_iter()
        .filter(|r| policy::is_visible(&r.scope, &r.project, scope, &project))
        .take(count)
        .collect();

    if visible.is_empty() {
        return Ok(());
    }

    // Output as Markdown to stdout (for Claude Code context injection)
    print_recall_output(&visible);
    Ok(())
}

/// Build a search query from the most recent Q&A chunks for this project.
fn build_query(db: &Database, project: &str) -> Result<String> {
    let recent = db.recent_chunks(project, 3)?;
    if recent.is_empty() {
        return Ok(String::new());
    }

    let query: String = recent
        .iter()
        .map(|c| c.question.as_str())
        .collect::<Vec<_>>()
        .join(" ");

    // Truncate to ~512 chars (approximate token limit)
    let truncated: String = query.chars().take(512).collect();
    Ok(truncated)
}

fn print_recall_output(results: &[ScoredResult]) {
    println!("## Related memories\n");
    for r in results {
        let ts = r.timestamp.as_deref().unwrap_or("unknown");
        let date = &ts[..10.min(ts.len())]; // extract YYYY-MM-DD
        println!("### {} | project: {}", date, r.project);
        println!("Q: {}", truncate(&r.question, 200));
        println!("A: {}", truncate(&r.answer, 500));
        println!();
    }
}

fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        s.to_string()
    } else {
        let truncated: String = s.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}
```

- [ ] **Step 2: Make `project_name` public in save.rs**

Change `fn project_name` to `pub fn project_name` in `src/cmd/save.rs`.

- [ ] **Step 3: Update cmd/mod.rs**

```rust
pub mod save;
pub mod search_cmd;
pub mod recall;

pub fn run(command: crate::Commands) -> Result<()> {
    match command {
        crate::Commands::Save { project } => save::run(&project),
        crate::Commands::Recall { project, count } => recall::run(&project, count),
        crate::Commands::Search { query, project, count } => {
            search_cmd::run(&query, project.as_deref(), count)
        }
        // ... rest unchanged
    }
}
```

- [ ] **Step 4: Build and verify**

Run: `cargo build`
Expected: Compiles successfully

- [ ] **Step 5: Commit**

```bash
git add src/cmd/recall.rs src/cmd/save.rs src/cmd/mod.rs
git commit -m "feat: add recall command with auto-query and policy filtering"
```

---

### Task 13: Import Command

**Files:**
- Create: `src/cmd/import_cmd.rs`
- Modify: `src/cmd/mod.rs`

- [ ] **Step 1: Implement import command**

```rust
use std::fs;
use std::path::PathBuf;
use anyhow::{Context, Result};
use crate::cmd::save;
use crate::db::Database;
use crate::chunk;

pub fn run() -> Result<()> {
    let home = dirs::home_dir().context("Cannot determine home directory")?;
    let projects_dir = home.join(".claude").join("projects");

    if !projects_dir.exists() {
        eprintln!("No Claude Code projects found at {}", projects_dir.display());
        return Ok(());
    }

    let db_path = save::db_path()?;
    if let Some(parent) = db_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let db = Database::open(&db_path)?;

    let mut total_sessions = 0;
    let mut total_chunks = 0;
    let mut skipped = 0;
    let mut errors = 0;

    // Iterate through project directories
    let project_dirs: Vec<_> = fs::read_dir(&projects_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| e.file_type().ok().is_some_and(|t| t.is_dir()))
        .collect();

    for project_entry in &project_dirs {
        let project_name = decode_project_name(&project_entry.file_name().to_string_lossy());

        // Find all JSONL files in this project directory (including subdirs for subagents)
        let jsonl_files = find_jsonl_files(&project_entry.path());

        for jsonl_path in jsonl_files {
            let session_id = jsonl_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_string())
                .unwrap_or_default();

            if db.session_exists(&session_id)? {
                skipped += 1;
                continue;
            }

            match import_session(&db, &jsonl_path, &session_id, &project_name) {
                Ok(chunk_count) => {
                    total_sessions += 1;
                    total_chunks += chunk_count;
                }
                Err(e) => {
                    eprintln!("Error importing {}: {}", jsonl_path.display(), e);
                    errors += 1;
                }
            }
        }
    }

    eprintln!(
        "Imported {} chunks from {} sessions ({} skipped, {} errors)",
        total_chunks, total_sessions, skipped, errors
    );
    Ok(())
}

fn import_session(
    db: &Database,
    path: &PathBuf,
    session_id: &str,
    project: &str,
) -> Result<usize> {
    let content = fs::read_to_string(path)?;
    let chunks = chunk::parse_session(&content)?;

    if chunks.is_empty() {
        return Ok(0);
    }

    let started_at = chunks.first().and_then(|c| c.timestamp.as_deref());
    db.insert_session(session_id, project, "global", started_at)?;

    let mut count = 0;
    for c in &chunks {
        if db.insert_chunk(
            session_id,
            Some(&c.uuid),
            &c.question,
            &c.answer,
            c.timestamp.as_deref(),
            (c.question.len() + c.answer.len()) as i64,
        )?.is_some() {
            count += 1;
        }
    }

    Ok(count)
}

/// Decode a Claude Code project directory name back to a project name.
/// e.g., "-Users-kenta-workspace-my-app" -> "my-app"
fn decode_project_name(encoded: &str) -> String {
    // Claude Code encodes "/Users/kenta/workspace/my-app" as "-Users-kenta-workspace-my-app".
    // We reverse the encoding by treating the encoded string as a path with '-' separators
    // and taking the last component. This is imperfect for project names containing '-',
    // but matches the save command's project_name() behavior (last path component).
    let trimmed = encoded.trim_start_matches('-');
    trimmed
        .rsplit('-')
        .next()
        .unwrap_or(trimmed)
        .to_string()
}

/// Recursively find all .jsonl files under a directory.
fn find_jsonl_files(dir: &std::path::Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.filter_map(|e| e.ok()) {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_jsonl_files(&path));
            } else if path.extension().is_some_and(|ext| ext == "jsonl") {
                files.push(path);
            }
        }
    }
    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_project_name() {
        assert_eq!(decode_project_name("-Users-kenta-workspace-myapp"), "myapp");
        assert_eq!(decode_project_name("-Users-kenta-workspace"), "workspace");
        assert_eq!(decode_project_name("-Users-kenta"), "kenta");
    }
}
```

- [ ] **Step 2: Update cmd/mod.rs**

Add `pub mod import_cmd;` and update the Import dispatch:

```rust
crate::Commands::Import => import_cmd::run(),
```

- [ ] **Step 3: Run tests and build**

Run: `cargo test`
Expected: All tests pass

- [ ] **Step 4: Commit**

```bash
git add src/cmd/import_cmd.rs src/cmd/mod.rs
git commit -m "feat: add import command for bulk session ingestion"
```

---

### Task 14: Stats Command

**Files:**
- Create: `src/cmd/stats.rs`
- Modify: `src/cmd/mod.rs`

- [ ] **Step 1: Implement stats command**

```rust
use anyhow::Result;
use crate::cmd::save;
use crate::db::Database;

pub fn run() -> Result<()> {
    let db_path = save::db_path()?;
    if !db_path.exists() {
        eprintln!("No database found at {}", db_path.display());
        return Ok(());
    }

    let db = Database::open(&db_path)?;
    let (sessions, chunks) = db.stats()?;

    println!("Database: {}", db_path.display());
    println!("Sessions: {}", sessions);
    println!("Chunks:   {}", chunks);

    Ok(())
}
```

- [ ] **Step 2: Update cmd/mod.rs**

Add `pub mod stats;` and update dispatch:

```rust
crate::Commands::Stats => stats::run(),
```

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add src/cmd/stats.rs src/cmd/mod.rs
git commit -m "feat: add stats command"
```

---

### Task 15: Setup Command

**Files:**
- Create: `src/cmd/setup.rs`
- Modify: `Cargo.toml` (add reqwest, tokio)
- Modify: `src/cmd/mod.rs`

- [ ] **Step 1: Add dependencies**

```toml
tokio = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["stream"] }
futures-util = "0.3"
```

- [ ] **Step 2: Implement setup command**

```rust
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use anyhow::{Context, Result};
use crate::cmd::save;

/// HuggingFace model repo for Ruri v3 ONNX
const MODEL_REPO: &str = "keitokei1994/ruri-v3-310m-onnx";
const MODEL_FILES: &[&str] = &["model.onnx", "tokenizer.json"];

pub fn run() -> Result<()> {
    let data_dir = save::data_dir()?;

    // Step 1: Create directory structure
    eprintln!("1. Creating {} ...", data_dir.display());
    let models_dir = data_dir.join("models").join("ruri-v3-310m");
    fs::create_dir_all(&models_dir)?;

    // Step 2: Download model files
    eprintln!("2. Downloading Ruri v3 ONNX model ...");
    let rt = tokio::runtime::Runtime::new()?;
    for file_name in MODEL_FILES {
        let dest = models_dir.join(file_name);
        if dest.exists() {
            eprintln!("   {} already exists, skipping", file_name);
            continue;
        }
        eprintln!("   Downloading {} ...", file_name);
        rt.block_on(download_hf_file(MODEL_REPO, file_name, &dest))?;
    }

    // Step 3: Initialize database
    eprintln!("3. Initializing SQLite database ...");
    let db_path = save::db_path()?;
    crate::db::Database::open(&db_path)?;
    eprintln!("   Database at {}", db_path.display());

    // Step 4: Show hook configuration
    eprintln!("4. Add the following to ~/.claude/settings.json:");
    eprintln!();
    print_hook_config();
    eprintln!();
    eprintln!("Done. Run `kiok import` to import existing sessions.");

    Ok(())
}

async fn download_hf_file(repo: &str, filename: &str, dest: &PathBuf) -> Result<()> {
    let url = format!(
        "https://huggingface.co/{}/resolve/main/{}",
        repo, filename
    );

    let response = reqwest::get(&url).await
        .with_context(|| format!("Failed to download {}", url))?;

    if !response.status().is_success() {
        anyhow::bail!("HTTP {} for {}", response.status(), url);
    }

    let bytes = response.bytes().await?;
    let mut file = fs::File::create(dest)?;
    file.write_all(&bytes)?;
    eprintln!("   Saved {} ({:.1} MB)", filename, bytes.len() as f64 / 1_000_000.0);

    Ok(())
}

fn print_hook_config() {
    println!(r#"{{
  "hooks": {{
    "SessionStart": [
      {{
        "hooks": [
          {{
            "type": "command",
            "command": "kiok recall --project $PWD"
          }}
        ]
      }}
    ],
    "SessionEnd": [
      {{
        "hooks": [
          {{
            "type": "command",
            "command": "kiok save --project $PWD &"
          }}
        ]
      }}
    ],
    "PreCompact": [
      {{
        "hooks": [
          {{
            "type": "command",
            "command": "kiok save --project $PWD"
          }}
        ]
      }}
    ]
  }}
}}"#);
}
```

- [ ] **Step 3: Update cmd/mod.rs**

Add `pub mod setup;` and update dispatch:

```rust
crate::Commands::Setup => setup::run(),
```

- [ ] **Step 4: Change main.rs to not require tokio runtime globally**

No change needed — setup uses `tokio::runtime::Runtime::new()` locally, so main stays synchronous.

- [ ] **Step 5: Build**

Run: `cargo build`
Expected: Compiles

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml src/cmd/setup.rs src/cmd/mod.rs
git commit -m "feat: add setup command with model download and hook config"
```

---

### Task 16: Integrate Embedding into Save Pipeline

**Files:**
- Modify: `src/cmd/save.rs` (add embedding step)
- Modify: `src/cmd/import_cmd.rs` (add embedding step)
- Modify: `src/cmd/recall.rs` (use hybrid search)

This task connects the embedding backend to the save and recall pipelines.

- [ ] **Step 1: Update save.rs to embed chunks**

Add to `save::run` after inserting chunks:

```rust
use crate::embed::{EmbeddingBackend, onnx::OnnxBackend};

// Load embedding model if available
let model_dir = data_dir()?.join("models").join("ruri-v3-310m");
if model_dir.join("model.onnx").exists() {
    let backend = OnnxBackend::load(&model_dir)?;
    let texts: Vec<String> = chunks.iter()
        .map(|c| format!("{} {}", c.question, c.answer))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
    let embeddings = backend.embed(&text_refs)?;

    // chunk_ids were collected during insert; pair with embeddings
    for (chunk_id, embedding) in chunk_ids.iter().zip(embeddings.iter()) {
        if let Some(id) = chunk_id {
            db.insert_embedding(*id, embedding)?;
        }
    }
    eprintln!("Embedded {} chunks", embeddings.len());
} else {
    eprintln!("Model not found, skipping embedding. Run `kiok setup` to download.");
}
```

Note: collect `chunk_ids: Vec<Option<i64>>` during the chunk insert loop.

- [ ] **Step 2: Update recall.rs to use hybrid search when model is available**

```rust
use crate::embed::{EmbeddingBackend, onnx::OnnxBackend};

// Try to load embedding model for hybrid search
let model_dir = save::data_dir()?.join("models").join("ruri-v3-310m");
let results = if model_dir.join("model.onnx").exists() {
    let backend = OnnxBackend::load(&model_dir)?;
    let query_embedding = backend.embed(&[query.as_str()])?;
    search::hybrid_search(&db, &query, Some(&query_embedding[0]), &config)?
} else {
    search::keyword_search(&db, &query, &config)?
};
```

- [ ] **Step 3: Build**

Run: `cargo build`
Expected: Compiles

- [ ] **Step 4: Commit**

```bash
git add src/cmd/save.rs src/cmd/recall.rs src/cmd/import_cmd.rs
git commit -m "feat: integrate embedding into save and recall pipelines"
```

---

### Task 17: End-to-End Integration Test

**Files:**
- Create: `tests/integration.rs`

- [ ] **Step 1: Write integration test**

```rust
use std::fs;
use std::path::PathBuf;

fn fixture_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
        .join("sample_session.jsonl")
}

#[test]
fn test_save_and_fts_search() {
    // Setup temp database
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("test.db");

    let db = kiok::db::Database::open(&db_path).unwrap();

    // Parse fixture
    let content = fs::read_to_string(fixture_path()).unwrap();
    let chunks = kiok::chunk::parse_session(&content).unwrap();
    assert!(!chunks.is_empty());

    // Insert
    db.insert_session("test-session-1", "test-project", "global", None).unwrap();
    for chunk in &chunks {
        db.insert_chunk(
            "test-session-1",
            Some(&chunk.uuid),
            &chunk.question,
            &chunk.answer,
            chunk.timestamp.as_deref(),
            (chunk.question.len() + chunk.answer.len()) as i64,
        ).unwrap();
    }

    // FTS search
    let config = kiok::search::SearchConfig::default();
    let results = kiok::search::keyword_search(&db, "Docker", &config).unwrap();
    assert!(!results.is_empty());
    assert!(results[0].question.contains("Docker"));
}

#[test]
fn test_policy_filtering() {
    let tmp_dir = tempfile::tempdir().unwrap();
    let db_path = tmp_dir.path().join("test.db");
    let db = kiok::db::Database::open(&db_path).unwrap();

    // Insert chunks from two projects with different scopes
    db.insert_session("s1", "public-project", "global", None).unwrap();
    db.insert_session("s2", "secret-project", "isolated", None).unwrap();

    db.insert_chunk("s1", Some("u1"), "Docker setup", "Use compose", None, 10).unwrap();
    db.insert_chunk("s2", Some("u2"), "Docker secret", "NDA content", None, 10).unwrap();

    // From a project scope viewer, should see global but not isolated
    let config = kiok::search::SearchConfig { count: 10, ..Default::default() };
    let results = kiok::search::keyword_search(&db, "Docker", &config).unwrap();

    let visible: Vec<_> = results
        .iter()
        .filter(|r| kiok::policy::is_visible(&r.scope, &r.project, kiok::policy::Scope::Project, "other-project"))
        .collect();

    assert_eq!(visible.len(), 1);
    assert_eq!(visible[0].project, "public-project");
}
```

- [ ] **Step 2: Add tempfile dev-dependency**

Add to Cargo.toml:
```toml
[dev-dependencies]
tempfile = "3"
```

- [ ] **Step 3: Make modules public in lib.rs**

Create `src/lib.rs`:
```rust
pub mod db;
pub mod chunk;
pub mod filter;
pub mod policy;
pub mod embed;
pub mod search;
```

Update `src/main.rs` to use the library:
```rust
use kiok::*;
```

- [ ] **Step 4: Run integration tests**

Run: `cargo test --test integration`
Expected: All integration tests pass

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml src/lib.rs src/main.rs tests/integration.rs
git commit -m "test: add end-to-end integration tests for save, search, and policy"
```

---

## Execution Checklist

| Task | Description | Est. |
|------|-------------|------|
| 1 | Project scaffold + CLI skeleton | 5 min |
| 2 | SQLite database layer | 10 min |
| 3 | Noise filtering | 5 min |
| 4 | JSONL parser + Q&A chunker | 10 min |
| 5 | Save command (without embedding) | 10 min |
| 6 | FTS5 search + search command | 10 min |
| 7 | Embedding backend + ONNX | 15 min |
| 8 | sqlite-vec vector search | 10 min |
| 9 | RRF score fusion + time decay | 10 min |
| 10 | Hybrid search orchestration | 5 min |
| 11 | Policy system | 5 min |
| 12 | Recall command | 10 min |
| 13 | Import command | 10 min |
| 14 | Stats command | 5 min |
| 15 | Setup command | 10 min |
| 16 | Integrate embedding into pipelines | 10 min |
| 17 | Integration tests | 10 min |
