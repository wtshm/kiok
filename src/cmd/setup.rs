use anyhow::{Context, Result};
use std::fs;
use std::io::Write;

use crate::db::Database;
use super::save::{data_dir, db_path, model_dir};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HF_BASE_URL: &str =
    "https://huggingface.co/keitokei1994/ruri-v3-310m-onnx/resolve/main";

const MODEL_FILES: &[&str] = &["model.onnx", "tokenizer.json"];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the setup command:
/// 1. Create the model directory.
/// 2. Download model files from HuggingFace.
/// 3. Initialize the database.
/// 4. Print hook configuration JSON.
pub fn run() -> Result<()> {
    // --- 1. Resolve paths ---
    let model_dir = model_dir()?;
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("Could not create model directory {}", model_dir.display()))?;

    eprintln!("setup: model directory: {}", model_dir.display());

    // --- 2. Download model files ---
    let rt = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    rt.block_on(download_model_files(&model_dir))?;

    // --- 3. Initialize database ---
    let data = data_dir()?;
    fs::create_dir_all(&data)
        .with_context(|| format!("Could not create data directory {}", data.display()))?;
    let the_db_path = db_path()?;
    Database::open(&the_db_path).context("Failed to initialize database")?;
    eprintln!("setup: database initialized at {}", the_db_path.display());

    // --- 4. Print hook configuration ---
    print_hook_config()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// model_dir() is re-exported from save.rs

/// Download all required model files from HuggingFace into `model_dir`.
async fn download_model_files(model_dir: &std::path::Path) -> Result<()> {
    let client = reqwest::Client::new();

    for filename in MODEL_FILES {
        let dest = model_dir.join(filename);

        if dest.exists() {
            eprintln!("setup: {} already exists, skipping download", filename);
            continue;
        }

        let url = format!("{}/{}", HF_BASE_URL, filename);
        eprintln!("setup: downloading {} ...", url);

        let response = client
            .get(&url)
            .send()
            .await
            .with_context(|| format!("Failed to fetch {}", url))?;

        if !response.status().is_success() {
            return Err(anyhow::anyhow!(
                "HTTP {} when downloading {}",
                response.status(),
                url
            ));
        }

        let bytes = response
            .bytes()
            .await
            .with_context(|| format!("Failed to read response body for {}", filename))?;

        let mut file = fs::File::create(&dest)
            .with_context(|| format!("Could not create file {}", dest.display()))?;
        file.write_all(&bytes)
            .with_context(|| format!("Could not write to {}", dest.display()))?;

        eprintln!("setup: saved {} ({} bytes)", dest.display(), bytes.len());
    }

    Ok(())
}

/// Print the hook configuration JSON that the user should add to
/// ~/.claude/settings.json.
fn print_hook_config() -> Result<()> {
    let hook_config = serde_json::json!({
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
    });

    println!("\nAdd the following to ~/.claude/settings.json:");
    println!("{}", serde_json::to_string_pretty(&hook_config)?);

    Ok(())
}
