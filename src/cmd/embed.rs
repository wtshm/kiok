use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

use anyhow::Result;

use crate::db::Database;
use crate::embed::{self, EmbeddingBackend};
use super::save::{data_dir, db_path, model_dir};

/// Maximum number of chunks to embed in a single ONNX inference call.
/// Kept small to avoid OOM on machines with limited RAM (~1.2GB model + batch).
const BATCH_SIZE: usize = 8;

/// Acquire an exclusive, non-blocking file lock on `~/.kiok/embed.lock`.
///
/// Returns the open `File` (holding the flock) on success, or `None` if
/// another embed process already holds the lock.  The lock is released
/// automatically when the returned `File` is dropped.
fn acquire_lock() -> Option<std::fs::File> {
    let lock_path = data_dir().ok()?.join("embed.lock");
    let file = OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(lock_path)
        .ok()?;

    let ret = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if ret == 0 {
        Some(file)
    } else {
        None
    }
}

/// Embed all chunks that don't yet have embeddings.
///
/// This command is designed to run in the background after `kiok save`.
/// It loads the ONNX model once and processes all un-embedded chunks in
/// small batches to keep peak memory manageable.
pub fn run() -> Result<()> {
    let _lock = match acquire_lock() {
        Some(f) => f,
        None => return Ok(()),
    };

    let dir = model_dir()?;
    let backend = match embed::try_load_backend(&dir) {
        Some(b) => b,
        None => return Ok(()),
    };

    let path = db_path()?;
    if !path.exists() {
        return Ok(());
    }
    let db = Database::open(path)?;

    // Clean up embeddings whose chunks have been deleted.
    let orphaned = db.delete_orphaned_embeddings()?;
    if orphaned > 0 {
        eprintln!("embed: cleaned up {} orphaned embeddings", orphaned);
    }

    let mut total = 0usize;
    loop {
        let pending = db.chunks_without_embeddings(BATCH_SIZE)?;
        if pending.is_empty() {
            break;
        }

        let texts: Vec<String> = pending
            .iter()
            .map(|c| c.embed_text())
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let embeddings = backend.embed(&text_refs)?;

        // Wrap each batch insert in a transaction to reduce fsync overhead.
        db.in_transaction(|| {
            for (chunk, embedding) in pending.iter().zip(embeddings.iter()) {
                db.insert_embedding(chunk.chunk_id, embedding)?;
                total += 1;
            }
            Ok(())
        })?;
    }

    if total > 0 {
        eprintln!("embed: processed {} chunks", total);
    }
    Ok(())
}
