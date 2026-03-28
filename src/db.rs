use anyhow::{Context, Result};
use bytemuck::cast_slice;
use rusqlite::{Connection, ffi::sqlite3_auto_extension, params};
use sqlite_vec::sqlite3_vec_init;
use std::path::Path;

/// A single chunk row combining chunk and session data.
#[derive(Debug, Clone)]
pub struct ChunkRow {
    pub chunk_id: i64,
    pub session_id: String,
    pub project: String,
    pub scope: String,
    pub question: String,
    pub answer: String,
    pub timestamp: Option<String>,
}

impl ChunkRow {
    /// Concatenate question and answer for embedding input.
    pub fn embed_text(&self) -> String {
        format!("{} {}", self.question, self.answer)
    }

    /// Construct from a rusqlite Row with the standard 7-column layout:
    /// (id, session_id, project, scope, question, answer, timestamp).
    fn from_row(row: &rusqlite::Row) -> rusqlite::Result<Self> {
        Ok(Self {
            chunk_id: row.get(0)?,
            session_id: row.get(1)?,
            project: row.get(2)?,
            scope: row.get(3)?,
            question: row.get(4)?,
            answer: row.get(5)?,
            timestamp: row.get(6)?,
        })
    }
}

/// A session row with chunk count for listing.
#[derive(Debug, Clone)]
pub struct SessionInfo {
    pub session_id: String,
    pub project: String,
    pub scope: String,
    pub started_at: Option<String>,
    pub imported_at: String,
    pub chunk_count: i64,
}

/// Wrapper around a SQLite connection providing all kiok database operations.
pub struct Database {
    conn: Connection,
}

impl Database {
    /// Register the sqlite-vec extension so it is loaded for every new connection.
    ///
    /// This must be called before opening a connection.  It is safe to call
    /// multiple times because SQLite deduplicates auto-extension entries.
    fn register_vec_extension() {
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute::<
                *const (),
                unsafe extern "C" fn(
                    *mut rusqlite::ffi::sqlite3,
                    *mut *mut i8,
                    *const rusqlite::ffi::sqlite3_api_routines,
                ) -> i32,
            >(sqlite3_vec_init as *const ())));
        }
    }

    /// Open (or create) the database at the given filesystem path.
    /// Initializes schema, enables WAL mode, and sets a busy timeout.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::register_vec_extension();
        let conn = Connection::open(path).context("Failed to open SQLite database")?;
        let db = Self { conn };
        db.configure()?;
        db.init_schema()?;
        Ok(db)
    }

    /// Open an in-memory database — used in tests.
    pub fn open_in_memory() -> Result<Self> {
        Self::register_vec_extension();
        let conn = Connection::open_in_memory().context("Failed to open in-memory SQLite database")?;
        let db = Self { conn };
        db.configure()?;
        db.init_schema()?;
        Ok(db)
    }

    /// Apply connection-level pragmas (WAL mode, busy timeout).
    fn configure(&self) -> Result<()> {
        self.conn.execute_batch(
            "PRAGMA journal_mode = WAL;
             PRAGMA busy_timeout = 5000;
             PRAGMA foreign_keys = ON;",
        )
        .context("Failed to configure SQLite pragmas")?;
        Ok(())
    }

    /// Create all tables, indexes, and FTS5 sync triggers if they do not exist.
    pub fn init_schema(&self) -> Result<()> {
        self.conn.execute_batch("
            CREATE TABLE IF NOT EXISTS sessions (
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

            -- Insert trigger: keep FTS in sync when a chunk is added.
            CREATE TRIGGER IF NOT EXISTS chunks_ai
            AFTER INSERT ON chunks BEGIN
                INSERT INTO chunks_fts(rowid, question, answer)
                VALUES (new.id, new.question, new.answer);
            END;

            -- Delete trigger: remove FTS entry when a chunk is deleted.
            CREATE TRIGGER IF NOT EXISTS chunks_ad
            AFTER DELETE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, question, answer)
                VALUES ('delete', old.id, old.question, old.answer);
            END;

            -- Update trigger: replace FTS entry when a chunk is updated.
            CREATE TRIGGER IF NOT EXISTS chunks_au
            AFTER UPDATE ON chunks BEGIN
                INSERT INTO chunks_fts(chunks_fts, rowid, question, answer)
                VALUES ('delete', old.id, old.question, old.answer);
                INSERT INTO chunks_fts(rowid, question, answer)
                VALUES (new.id, new.question, new.answer);
            END;
        ").context("Failed to initialize schema")?;

        // vec0 virtual table for approximate nearest-neighbour vector search.
        // chunk_id is a foreign key to chunks.id (enforced by application code).
        //
        // Migration: if an older schema with the wrong dimensions exists,
        // drop and recreate.  Safe because we can always re-embed.
        self.migrate_chunks_vec()?;

        self.conn.execute_batch("
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_vec USING vec0(
                chunk_id INTEGER PRIMARY KEY,
                embedding FLOAT[768]
            );
        ").context("Failed to create chunks_vec virtual table")?;

        Ok(())
    }

    /// Drop `chunks_vec` if its dimension no longer matches the expected 768.
    ///
    /// Detection: the sqlite_master CREATE statement contains `FLOAT[N]`.
    /// This is a one-shot migration; the table will be recreated immediately
    /// afterward by the caller.
    fn migrate_chunks_vec(&self) -> Result<()> {
        let sql: Option<String> = self
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
                [],
                |row| row.get(0),
            )
            .ok();

        if let Some(create_sql) = sql
            && !create_sql.contains("FLOAT[768]")
        {
            eprintln!("kiok: migrating chunks_vec to FLOAT[768]");
            self.conn
                .execute_batch("DROP TABLE IF EXISTS chunks_vec;")
                .context("Failed to drop old chunks_vec")?;
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Session operations
    // -----------------------------------------------------------------------

    /// Insert a session record.  Uses INSERT OR IGNORE for idempotency.
    /// Returns `true` if a new row was inserted, `false` if it already existed.
    pub fn insert_session(
        &self,
        session_id: &str,
        project: &str,
        scope: &str,
        started_at: Option<&str>,
    ) -> Result<bool> {
        let rows = self.conn.execute(
            "INSERT OR IGNORE INTO sessions (session_id, project, scope, started_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![session_id, project, scope, started_at],
        )
        .context("Failed to insert session")?;
        Ok(rows > 0)
    }

    /// Check whether a session with the given ID exists.
    pub fn session_exists(&self, session_id: &str) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sessions WHERE session_id = ?1",
            params![session_id],
            |row| row.get(0),
        )
        .context("Failed to query session existence")?;
        Ok(count > 0)
    }

    // -----------------------------------------------------------------------
    // Chunk operations
    // -----------------------------------------------------------------------

    /// Insert a chunk into the database.
    ///
    /// If `uuid` is `Some`, duplicate UUIDs are silently skipped (returns `None`).
    /// Returns `Some(chunk_id)` on success.
    pub fn insert_chunk(
        &self,
        session_id: &str,
        uuid: Option<&str>,
        question: &str,
        answer: &str,
        timestamp: Option<&str>,
        token_count: Option<i64>,
    ) -> Result<Option<i64>> {
        // If a UUID is provided, check for an existing row first.
        if let Some(uid) = uuid {
            let existing: Option<i64> = self.conn.query_row(
                "SELECT id FROM chunks WHERE uuid = ?1",
                params![uid],
                |row| row.get(0),
            )
            .ok();
            if existing.is_some() {
                return Ok(None);
            }
        }

        self.conn.execute(
            "INSERT INTO chunks (session_id, uuid, question, answer, timestamp, token_count)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![session_id, uuid, question, answer, timestamp, token_count],
        )
        .context("Failed to insert chunk")?;

        Ok(Some(self.conn.last_insert_rowid()))
    }

    /// Return chunks that have no corresponding embedding yet, up to `limit`.
    pub fn chunks_without_embeddings(&self, limit: usize) -> Result<Vec<ChunkRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.session_id, s.project, s.scope,
                    c.question, c.answer, c.timestamp
             FROM chunks c
             JOIN sessions s ON s.session_id = c.session_id
             LEFT JOIN chunks_vec v ON v.chunk_id = c.id
             WHERE v.chunk_id IS NULL
             ORDER BY c.id DESC
             LIMIT ?1",
        )?;

        let rows = stmt
            .query_map(params![limit as i64], ChunkRow::from_row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // -----------------------------------------------------------------------
    // Search / query operations
    // -----------------------------------------------------------------------

    /// Full-text search using FTS5 trigram index.
    ///
    /// Each whitespace-separated token is wrapped in double-quotes (trigram
    /// substring match) and combined with AND so that all terms must appear
    /// somewhere in the document, but not necessarily as a contiguous phrase.
    pub fn fts_search(&self, query: &str, limit: usize) -> Result<Vec<ChunkRow>> {
        let terms: Vec<&str> = query.split_whitespace().collect();
        let fts_query = if terms.len() <= 1 {
            // Single term (or empty): wrap as a quoted substring.
            let escaped = query.replace('"', "\"\"");
            format!("\"{}\"", escaped)
        } else {
            // Multiple terms: AND them together as individual substrings.
            terms
                .iter()
                .map(|t| format!("\"{}\"", t.replace('"', "\"\"")))
                .collect::<Vec<_>>()
                .join(" AND ")
        };

        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.session_id, s.project, s.scope,
                    c.question, c.answer, c.timestamp
             FROM chunks_fts
             JOIN chunks  c ON chunks_fts.rowid = c.id
             JOIN sessions s ON c.session_id = s.session_id
             WHERE chunks_fts MATCH ?1
             ORDER BY chunks_fts.rank
             LIMIT ?2",
        )
        .context("Failed to prepare FTS search statement")?;

        let rows = stmt
            .query_map(params![fts_query, limit as i64], ChunkRow::from_row)
            .context("Failed to execute FTS search")?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to collect FTS search results")?;

        Ok(rows)
    }

    // -----------------------------------------------------------------------
    // Vector search operations
    // -----------------------------------------------------------------------

    /// Store a 768-dim embedding for the given chunk.
    ///
    /// The embedding is passed as a BLOB (raw little-endian f32 bytes).
    pub fn insert_embedding(&self, chunk_id: i64, embedding: &[f32]) -> Result<()> {
        let blob: &[u8] = cast_slice(embedding);
        self.conn
            .execute(
                "INSERT OR REPLACE INTO chunks_vec (chunk_id, embedding) VALUES (?1, ?2)",
                params![chunk_id, blob],
            )
            .context("Failed to insert embedding into chunks_vec")?;
        Ok(())
    }

    /// Execute `BEGIN` / `COMMIT` around a closure for batch writes.
    pub fn in_transaction<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce() -> Result<T>,
    {
        self.conn.execute_batch("BEGIN")?;
        match f() {
            Ok(val) => {
                self.conn.execute_batch("COMMIT")?;
                Ok(val)
            }
            Err(e) => {
                let _ = self.conn.execute_batch("ROLLBACK");
                Err(e)
            }
        }
    }

    /// Return the `limit` closest chunks to `query_embedding` using vec0 KNN search.
    ///
    /// Results are ordered by ascending distance (nearest first).
    pub fn vec_search(&self, query_embedding: &[f32], limit: usize) -> Result<Vec<ChunkRow>> {
        let blob: &[u8] = cast_slice(query_embedding);

        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.session_id, s.project, s.scope,
                    c.question, c.answer, c.timestamp
             FROM chunks_vec cv
             JOIN chunks   c ON cv.chunk_id = c.id
             JOIN sessions s ON c.session_id = s.session_id
             WHERE cv.embedding MATCH ?1 AND cv.k = ?2
             ORDER BY cv.distance",
        )
        .context("Failed to prepare vec_search statement")?;

        let rows = stmt
            .query_map(params![blob, limit as i64], ChunkRow::from_row)
            .context("Failed to execute vec_search")?
            .collect::<Result<Vec<_>, _>>()
            .context("Failed to collect vec_search results")?;

        Ok(rows)
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Query sessions with chunk counts, ordered by import time descending.
    pub fn list_sessions(&self, limit: usize, offset: usize) -> Result<Vec<SessionInfo>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.session_id, s.project, s.scope, s.started_at, s.imported_at,
                    (SELECT count(*) FROM chunks c WHERE c.session_id = s.session_id)
             FROM sessions s
             ORDER BY s.imported_at DESC
             LIMIT ?1 OFFSET ?2"
        )?;

        let rows = stmt.query_map(params![limit as i64, offset as i64], |row| {
            Ok(SessionInfo {
                session_id: row.get(0)?,
                project: row.get(1)?,
                scope: row.get(2)?,
                started_at: row.get(3)?,
                imported_at: row.get(4)?,
                chunk_count: row.get(5)?,
            })
        })?;

        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Query chunks ordered by timestamp descending.
    pub fn list_chunks(&self, limit: usize, offset: usize) -> Result<Vec<ChunkRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.session_id, s.project, s.scope,
                    c.question, c.answer, c.timestamp
             FROM chunks c
             JOIN sessions s ON s.session_id = c.session_id
             ORDER BY c.timestamp DESC
             LIMIT ?1 OFFSET ?2"
        )?;

        let rows = stmt
            .query_map(params![limit as i64, offset as i64], ChunkRow::from_row)?
            .collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Return the total number of sessions and chunks stored in the database.
    pub fn stats(&self) -> Result<(i64, i64)> {
        let sessions: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM sessions",
            [],
            |row| row.get(0),
        )
        .context("Failed to count sessions")?;

        let chunks: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM chunks",
            [],
            |row| row.get(0),
        )
        .context("Failed to count chunks")?;

        Ok((sessions, chunks))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: open an in-memory database ready for testing.
    fn make_db() -> Database {
        Database::open_in_memory().expect("open_in_memory failed")
    }

    #[test]
    fn test_open_in_memory_creates_tables() {
        let db = make_db();

        // Each table must appear in sqlite_master.
        for table in &["sessions", "chunks", "chunks_fts", "chunks_vec"] {
            let count: i64 = db
                .conn
                .query_row(
                    "SELECT COUNT(*) FROM sqlite_master WHERE name = ?1",
                    params![table],
                    |row| row.get(0),
                )
                .expect("query failed");
            assert_eq!(count, 1, "table '{}' was not created", table);
        }
    }

    #[test]
    fn test_insert_and_query_session() {
        let db = make_db();
        let inserted = db
            .insert_session("sess-1", "/home/user/proj", "global", Some("2024-01-01T00:00:00"))
            .expect("insert_session failed");
        assert!(inserted, "expected new row");

        // Second insert of same session_id should be a no-op.
        let dup = db
            .insert_session("sess-1", "/home/user/proj", "global", None)
            .expect("insert_session (dup) failed");
        assert!(!dup, "expected no new row for duplicate session_id");

        assert!(db.session_exists("sess-1").expect("session_exists failed"));
        assert!(!db.session_exists("sess-999").expect("session_exists failed"));
    }

    #[test]
    fn test_insert_and_query_chunk() {
        let db = make_db();
        db.insert_session("sess-2", "/proj", "global", None)
            .expect("insert_session failed");

        let id = db
            .insert_chunk(
                "sess-2",
                Some("uuid-abc"),
                "How does Rust work?",
                "Rust uses ownership for memory safety.",
                Some("2024-01-02T10:00:00"),
                Some(42),
            )
            .expect("insert_chunk failed");
        assert!(id.is_some(), "expected a chunk id");

        let (sessions, chunks) = db.stats().expect("stats failed");
        assert_eq!(sessions, 1);
        assert_eq!(chunks, 1);
    }

    #[test]
    fn test_duplicate_uuid_skipped() {
        let db = make_db();
        db.insert_session("sess-3", "/proj", "global", None)
            .expect("insert_session failed");

        // Insert with a known UUID.
        let first = db
            .insert_chunk("sess-3", Some("uuid-dup"), "Q1", "A1", None, None)
            .expect("first insert failed");
        assert!(first.is_some());

        // Insert the same UUID again — must be skipped.
        let second = db
            .insert_chunk("sess-3", Some("uuid-dup"), "Q2", "A2", None, None)
            .expect("second insert failed");
        assert!(second.is_none(), "duplicate UUID must return None");

        let (_, chunks) = db.stats().expect("stats failed");
        assert_eq!(chunks, 1, "only one row should exist");
    }

    #[test]
    fn test_fts5_search() {
        let db = make_db();
        db.insert_session("sess-4", "/proj", "global", None)
            .expect("insert_session failed");

        db.insert_chunk(
            "sess-4",
            Some("uuid-docker"),
            "Docker設定の方法",
            "docker-compose.ymlを作成する",
            None,
            None,
        )
        .expect("insert docker chunk failed");

        db.insert_chunk(
            "sess-4",
            Some("uuid-rust"),
            "Rust入門",
            "cargoでプロジェクトを作成する",
            None,
            None,
        )
        .expect("insert rust chunk failed");

        let results = db.fts_search("Docker", 10).expect("fts_search failed");
        assert_eq!(results.len(), 1, "expected exactly 1 result for 'Docker'");
        assert!(
            results[0].question.contains("Docker"),
            "result question should contain 'Docker'"
        );
    }

    #[test]
    fn test_embed_text_concatenates_question_and_answer() {
        let chunk = ChunkRow {
            chunk_id: 1,
            session_id: "s1".to_owned(),
            project: "proj".to_owned(),
            scope: "global".to_owned(),
            question: "What is Rust?".to_owned(),
            answer: "A systems language.".to_owned(),
            timestamp: None,
        };
        assert_eq!(chunk.embed_text(), "What is Rust? A systems language.");
    }

    #[test]
    fn test_in_transaction_commits_on_success() {
        let db = make_db();
        db.insert_session("s1", "proj", "global", None).unwrap();

        db.in_transaction(|| {
            db.insert_chunk("s1", Some("u1"), "Q1", "A1", None, None)?;
            db.insert_chunk("s1", Some("u2"), "Q2", "A2", None, None)?;
            Ok(())
        })
        .unwrap();

        let (_, chunks) = db.stats().unwrap();
        assert_eq!(chunks, 2, "both chunks should be committed");
    }

    #[test]
    fn test_in_transaction_rolls_back_on_error() {
        let db = make_db();
        db.insert_session("s1", "proj", "global", None).unwrap();

        let result: Result<()> = db.in_transaction(|| {
            db.insert_chunk("s1", Some("u1"), "Q1", "A1", None, None)?;
            anyhow::bail!("simulated error");
        });

        assert!(result.is_err());
        let (_, chunks) = db.stats().unwrap();
        assert_eq!(chunks, 0, "chunk should be rolled back");
    }

    #[test]
    fn test_insert_embedding_and_chunks_without_embeddings() {
        let db = make_db();
        db.insert_session("s1", "proj", "global", None).unwrap();

        let id1 = db
            .insert_chunk("s1", None, "Q1", "A1", None, None)
            .unwrap()
            .unwrap();
        let id2 = db
            .insert_chunk("s1", None, "Q2", "A2", None, None)
            .unwrap()
            .unwrap();

        // Both should be pending initially.
        let pending = db.chunks_without_embeddings(10).unwrap();
        assert_eq!(pending.len(), 2);

        // Embed one chunk.
        let emb = vec![0.1f32; 768];
        db.insert_embedding(id1, &emb).unwrap();

        // Only one should remain pending.
        let pending = db.chunks_without_embeddings(10).unwrap();
        assert_eq!(pending.len(), 1);
        assert_eq!(pending[0].chunk_id, id2);
    }

    #[test]
    fn test_migrate_chunks_vec_recreates_on_dimension_change() {
        // Create a DB file with old 1024-dim chunks_vec, then reopen via
        // Database::open to verify migration drops and recreates with 768.
        let tmp = std::env::temp_dir().join("kiok-test-migrate.db");
        let _ = std::fs::remove_file(&tmp);

        // Create old-schema DB on disk.
        {
            Database::register_vec_extension();
            let conn = Connection::open(&tmp).unwrap();
            conn.execute_batch("
                PRAGMA journal_mode=WAL;
                CREATE TABLE sessions (session_id TEXT PRIMARY KEY, project TEXT NOT NULL, scope TEXT NOT NULL DEFAULT 'global', started_at TEXT, imported_at TEXT NOT NULL DEFAULT (datetime('now')));
                CREATE TABLE chunks (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, uuid TEXT, question TEXT NOT NULL, answer TEXT NOT NULL, timestamp TEXT, token_count INTEGER);
                CREATE VIRTUAL TABLE chunks_fts USING fts5(question, answer, content_rowid='id', content='chunks', tokenize='trigram');
                CREATE VIRTUAL TABLE chunks_vec USING vec0(chunk_id INTEGER PRIMARY KEY, embedding FLOAT[1024]);
            ").unwrap();

            let sql: String = conn.query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
                [], |row| row.get(0),
            ).unwrap();
            assert!(sql.contains("1024"), "should start as 1024-dim");
        }

        // Reopen with Database::open — triggers migration.
        let db = Database::open(&tmp).expect("reopen failed");
        let sql: String = db.conn.query_row(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='chunks_vec'",
            [], |row| row.get(0),
        ).unwrap();
        assert!(sql.contains("768"), "should be migrated to 768-dim, got: {}", sql);

        let _ = std::fs::remove_file(&tmp);
    }
}
