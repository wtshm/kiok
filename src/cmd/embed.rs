use anyhow::{Context, Result, bail};

use crate::db::Database;
use crate::embed::{self, EmbeddingBackend};
use crate::embed::onnx::OnnxBackend;
use super::save::{db_path, model_onnx_path};

/// Maximum number of chunks to embed in a single ONNX inference call.
/// Kept small to avoid OOM on machines with limited RAM (~1.2GB model + batch).
const BATCH_SIZE: usize = 8;

/// Embed all chunks that don't yet have embeddings.
///
/// This command is designed to run in the background after `kiok save`.
/// It loads the ONNX model once and processes all un-embedded chunks in
/// small batches to keep peak memory manageable.
pub fn run() -> Result<()> {
    let model_path = model_onnx_path()?;
    if !model_path.exists() {
        return Ok(()); // No model, nothing to do
    }

    // Fail fast if ONNX Runtime shared library is not available.
    if embed::ensure_ort_dylib().is_none() {
        bail!(
            "ONNX Runtime shared library not found. \
             Install it (brew install onnxruntime) or set ORT_DYLIB_PATH."
        );
    }

    let path = db_path()?;
    if !path.exists() {
        return Ok(());
    }
    let db = Database::open(path)?;

    let backend = OnnxBackend::load(
        model_path.parent().context("Invalid model path")?,
    )?;

    let mut total = 0usize;
    loop {
        let pending = db.chunks_without_embeddings(BATCH_SIZE)?;
        if pending.is_empty() {
            break;
        }

        let texts: Vec<String> = pending
            .iter()
            .map(|c| format!("{} {}", c.question, c.answer))
            .collect();
        let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

        let embeddings = backend.embed(&text_refs)?;

        for (chunk, embedding) in pending.iter().zip(embeddings.iter()) {
            match db.insert_embedding(chunk.chunk_id, embedding) {
                Ok(_) => total += 1,
                Err(e) => {
                    eprintln!(
                        "embed: failed to insert chunk {} (dim={}): {}",
                        chunk.chunk_id,
                        embedding.len(),
                        e
                    );
                    return Err(e);
                }
            }
        }
    }

    if total > 0 {
        eprintln!("embed: processed {} chunks", total);
    }
    Ok(())
}
