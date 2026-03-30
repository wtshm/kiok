use std::fs::OpenOptions;
use std::os::unix::io::AsRawFd;

use anyhow::Result;
use indicatif::{ProgressBar, ProgressStyle};

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

/// Result of an embed run.
pub enum EmbedOutcome {
    /// Another embed process holds the lock.
    Locked,
    /// Model or database not available.
    Unavailable,
    /// Completed successfully; `usize` is the number of chunks embedded.
    Done(usize),
}

/// Embed all chunks that don't yet have embeddings.
///
/// This command is designed to run in the background after `kiok save`.
/// It loads the ONNX model once and processes all un-embedded chunks in
/// small batches to keep peak memory manageable.
pub fn run() -> Result<EmbedOutcome> {
    let _lock = match acquire_lock() {
        Some(f) => f,
        None => return Ok(EmbedOutcome::Locked),
    };

    let dir = model_dir()?;
    let backend = match embed::try_load_backend(&dir) {
        Some(b) => b,
        None => return Ok(EmbedOutcome::Unavailable),
    };

    let path = db_path()?;
    if !path.exists() {
        return Ok(EmbedOutcome::Unavailable);
    }
    let db = Database::open(path)?;

    // Clean up embeddings whose chunks have been deleted.
    let orphaned = db.delete_orphaned_embeddings()?;
    if orphaned > 0 {
        eprintln!("embed: cleaned up {} orphaned embeddings", orphaned);
    }

    let pending_count = db.count_pending_embeddings()?;
    if pending_count == 0 {
        return Ok(EmbedOutcome::Done(0));
    }

    let pb = ProgressBar::new(pending_count as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("      [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
            .expect("invalid progress bar template")
            .progress_chars("#>-"),
    );

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
                pb.inc(1);
            }
            Ok(())
        })?;
    }

    pb.finish_and_clear();

    if total > 0 {
        eprintln!("embed: processed {} chunks", total);
    }
    Ok(EmbedOutcome::Done(total))
}
