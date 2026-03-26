use anyhow::{Context, Result};

use crate::db::Database;
use crate::embed::EmbeddingBackend;
use crate::embed::onnx::OnnxBackend;
use super::save::{db_path, model_onnx_path};

/// Embed all chunks that don't yet have embeddings.
///
/// This command is designed to run in the background after `kiok save`.
/// It loads the ONNX model once and processes all un-embedded chunks in batch.
pub fn run() -> Result<()> {
    let model_path = model_onnx_path()?;
    if !model_path.exists() {
        return Ok(()); // No model, nothing to do
    }

    let path = db_path()?;
    if !path.exists() {
        return Ok(());
    }
    let db = Database::open(path)?;

    // Find chunks without embeddings
    let pending = db.chunks_without_embeddings(100)?;
    if pending.is_empty() {
        return Ok(());
    }

    let backend = OnnxBackend::load(
        model_path.parent().context("Invalid model path")?,
    )?;

    let texts: Vec<String> = pending
        .iter()
        .map(|c| format!("{} {}", c.question, c.answer))
        .collect();
    let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

    let embeddings = backend.embed(&text_refs)?;

    let mut count = 0usize;
    for (chunk, embedding) in pending.iter().zip(embeddings.iter()) {
        if db.insert_embedding(chunk.chunk_id, embedding).is_ok() {
            count += 1;
        }
    }

    eprintln!("embed: processed {} chunks", count);
    Ok(())
}
