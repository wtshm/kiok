use anyhow::Result;

use crate::db::Database;
use crate::embed::{self, EmbeddingBackend};
use super::save::{db_path, model_dir};

/// Maximum number of chunks to embed in a single ONNX inference call.
/// Kept small to avoid OOM on machines with limited RAM (~1.2GB model + batch).
const BATCH_SIZE: usize = 8;

/// Embed all chunks that don't yet have embeddings.
///
/// This command is designed to run in the background after `kiok save`.
/// It loads the ONNX model once and processes all un-embedded chunks in
/// small batches to keep peak memory manageable.
pub fn run() -> Result<()> {
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
