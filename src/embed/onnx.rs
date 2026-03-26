use std::cell::RefCell;
use std::path::Path;

use anyhow::{Context, Result, anyhow};
use ort::session::Session;
use ort::value::{Tensor, ValueType};
use tokenizers::{PaddingParams, Tokenizer};

use super::EmbeddingBackend;

/// ONNX Runtime-based embedding backend.
///
/// Loads a `model.onnx` and `tokenizer.json` from a directory, tokenizes text
/// batches, runs inference, and returns L2-normalized mean-pooled embeddings.
pub struct OnnxBackend {
    /// Session wrapped in RefCell so we can call the `&mut self` run() method
    /// through the `&self` EmbeddingBackend trait bound.
    session: RefCell<Session>,
    tokenizer: Tokenizer,
    dims: usize,
}

impl OnnxBackend {
    /// Load the ONNX model and tokenizer from `model_dir`.
    ///
    /// Expects `model_dir/model.onnx` and `model_dir/tokenizer.json`.
    pub fn load(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir.join("model.onnx");
        let tokenizer_path = model_dir.join("tokenizer.json");

        let session = Session::builder()
            .context("Failed to create ONNX session builder")?
            .commit_from_file(&model_path)
            .with_context(|| format!("Failed to load ONNX model from {:?}", model_path))?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow!("Failed to load tokenizer from {:?}: {}", tokenizer_path, e))?;

        // Enable batch padding so all sequences in a batch have the same length.
        tokenizer.with_padding(Some(PaddingParams::default()));

        // Infer output dimensions from the model's output metadata, defaulting to 1024.
        let dims = session
            .outputs()
            .first()
            .and_then(|o| {
                if let ValueType::Tensor { shape, .. } = o.dtype() {
                    shape.last().copied()
                } else {
                    None
                }
            })
            .and_then(|d| if d > 0 { Some(d as usize) } else { None })
            .unwrap_or(1024);

        Ok(Self {
            session: RefCell::new(session),
            tokenizer,
            dims,
        })
    }
}

impl EmbeddingBackend for OnnxBackend {
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        let batch_size = texts.len();
        if batch_size == 0 {
            return Ok(vec![]);
        }

        // Tokenize the entire batch at once.
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow!("Tokenization failed: {}", e))?;

        let seq_len = encodings
            .first()
            .map(|e| e.get_ids().len())
            .unwrap_or(0);

        // Build padded input_ids and attention_mask as flat Vec<i64> with shape
        // [batch_size, seq_len], using the (shape, Box<[T]>) tuple form accepted by ort.
        let mut input_ids_data: Vec<i64> = Vec::with_capacity(batch_size * seq_len);
        let mut attention_mask_data: Vec<i64> = Vec::with_capacity(batch_size * seq_len);

        for enc in &encodings {
            for (&id, &mask) in enc.get_ids().iter().zip(enc.get_attention_mask().iter()) {
                input_ids_data.push(id as i64);
                attention_mask_data.push(mask as i64);
            }
        }

        let shape = [batch_size, seq_len];

        // Run inference using named inputs.
        // Keep the session borrow alive long enough for the outputs to be extracted.
        let mut session = self.session.borrow_mut();
        let outputs = session
            .run(ort::inputs! {
                "input_ids"      => Tensor::<i64>::from_array((shape, input_ids_data.into_boxed_slice()))?,
                "attention_mask" => Tensor::<i64>::from_array((shape, attention_mask_data.into_boxed_slice()))?
            })
            .context("ONNX inference failed")?;

        // Extract the first output tensor as raw (shape, slice).
        // Expected shape is [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim].
        let (out_shape, out_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("Failed to extract f32 output tensor")?;

        let ndim = out_shape.len();
        if ndim < 2 {
            return Err(anyhow!("Unexpected output tensor rank: {}", ndim));
        }

        let hidden_dim = out_shape[ndim - 1] as usize;

        // Mean pooling: average over the sequence dimension for 3D output
        // [batch, seq_len, hidden_dim], or just reshape for 2D [batch, hidden_dim].
        let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(batch_size);

        if ndim == 3 {
            // 3D: [batch, seq_len, hidden_dim]
            for (i, encoding) in encodings.iter().enumerate().take(batch_size) {
                let mut sum = vec![0.0f32; hidden_dim];
                let mut count = 0usize;

                for j in 0..seq_len {
                    let mask_val = encoding.get_attention_mask().get(j).copied().unwrap_or(0);

                    if mask_val != 0 {
                        let base = (i * seq_len + j) * hidden_dim;
                        for k in 0..hidden_dim {
                            sum[k] += out_data[base + k];
                        }
                        count += 1;
                    }
                }

                let divisor = if count == 0 { 1 } else { count } as f32;
                let mut embedding: Vec<f32> = sum.into_iter().map(|v| v / divisor).collect();
                l2_normalize(&mut embedding);
                embeddings.push(embedding);
            }
        } else {
            // 2D: [batch, hidden_dim] — model already provides a pooled output.
            for i in 0..batch_size {
                let base = i * hidden_dim;
                let mut embedding: Vec<f32> = out_data[base..base + hidden_dim].to_vec();
                l2_normalize(&mut embedding);
                embeddings.push(embedding);
            }
        }

        Ok(embeddings)
    }

    fn dimensions(&self) -> usize {
        self.dims
    }
}

/// Normalize a vector in-place to unit L2 norm.
fn l2_normalize(v: &mut [f32]) {
    let norm = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if norm > 1e-9 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}
