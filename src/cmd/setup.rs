use anyhow::{Context, Result};
use std::fs;
use std::io::{self, BufRead, Write};

use crate::db::Database;
use crate::embed::ensure_ort_dylib;
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

/// Run the interactive setup wizard: download model, init DB, configure hooks,
/// and optionally import existing sessions and run embedding.
pub fn run() -> Result<()> {
    // --- 1. Resolve paths ---
    let model_dir = model_dir()?;
    fs::create_dir_all(&model_dir)
        .with_context(|| format!("Could not create model directory {}", model_dir.display()))?;

    eprintln!("setup: model directory: {}", model_dir.display());

    // --- 2. Ensure ONNX Runtime is available ---
    if let Some(path) = ensure_ort_dylib() {
        eprintln!("setup: ONNX Runtime found at {}", path.display());
    } else {
        anyhow::bail!(
            "ONNX Runtime not found.\n\n\
             Install it before running setup:\n  \
             https://github.com/microsoft/onnxruntime"
        );
    }

    // --- 3. Download model files ---
    let rt = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    rt.block_on(download_model_files(&model_dir))?;

    // --- 4. Initialize database ---
    let data = data_dir()?;
    fs::create_dir_all(&data)
        .with_context(|| format!("Could not create data directory {}", data.display()))?;
    let the_db_path = db_path()?;
    Database::open(&the_db_path).context("Failed to initialize database")?;
    eprintln!("setup: database initialized at {}", the_db_path.display());

    // --- 5. Configure hooks in ~/.claude/settings.json ---
    if confirm("\nAdd kiok hooks to ~/.claude/settings.json? [y/N] ")? {
        install_hooks()?;
    } else {
        print_hook_config()?;
    }

    // --- 6. Optionally import existing sessions ---
    let imported_chunks = if confirm("\nImport existing Claude Code sessions? [y/N] ")? {
        let db = Database::open(&the_db_path)?;
        let before = db.stats()?.chunks;
        drop(db);
        super::import_cmd::run()?;
        let db = Database::open(&the_db_path)?;
        let after = db.stats()?.chunks;
        after - before
    } else {
        0
    };

    // --- 7. Optionally embed imported chunks ---
    if imported_chunks > 0 {
        eprintln!(
            "\nNote: Embedding {} chunks with Ruri v3 may take several minutes\n\
             depending on your machine (estimated ~0.8s per chunk on CPU).",
            imported_chunks
        );
        if confirm("Run embedding now? [y/N] ")? {
            match super::embed::run()? {
                super::embed::EmbedOutcome::Locked => {
                    eprintln!("setup: another embed process is running. Run `kiok embed` later.");
                }
                super::embed::EmbedOutcome::Unavailable => {
                    eprintln!("setup: embedding model not available. Run `kiok setup` first.");
                }
                super::embed::EmbedOutcome::Done(n) => {
                    eprintln!("setup: embedded {} chunks.", n);
                }
            }
        } else {
            eprintln!("Skipped. You can run `kiok embed` later to enable vector search.");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Prompt the user with a yes/no question. Returns true for 'y' or 'Y'.
fn confirm(prompt: &str) -> Result<bool> {
    eprint!("{}", prompt);
    io::stderr().flush()?;
    let mut line = String::new();
    io::stdin().lock().read_line(&mut line)?;
    Ok(line.trim().eq_ignore_ascii_case("y"))
}

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

/// The kiok hook entries to install.  Each tuple is (event, command).
const KIOK_HOOKS: &[(&str, &str)] = &[
    ("SessionEnd", "kiok save --project $PWD &"),
    ("PreCompact", "kiok save --project $PWD"),
];

/// Marker substring used to detect whether a kiok hook is already present.
const KIOK_MARKER: &str = "kiok ";

/// Install kiok hooks into ~/.claude/settings.json, preserving existing hooks.
fn install_hooks() -> Result<()> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let settings_path = home.join(".claude").join("settings.json");

    // Read existing settings or start from an empty object.
    let mut settings: serde_json::Value = match fs::read_to_string(&settings_path) {
        Ok(content) => serde_json::from_str(&content)
            .with_context(|| format!("Could not parse {}", settings_path.display()))?,
        Err(e) if e.kind() == io::ErrorKind::NotFound => serde_json::json!({}),
        Err(e) => return Err(anyhow::Error::from(e)
            .context(format!("Could not read {}", settings_path.display()))),
    };

    let hooks = settings
        .as_object_mut()
        .context("settings.json root is not an object")?
        .entry("hooks")
        .or_insert_with(|| serde_json::json!({}));

    let hooks_obj = hooks
        .as_object_mut()
        .context("hooks field is not an object")?;

    let mut added = Vec::new();
    let mut skipped = Vec::new();

    for &(event, command) in KIOK_HOOKS {
        let arr = hooks_obj
            .entry(event)
            .or_insert_with(|| serde_json::json!([]));

        let arr = arr.as_array_mut()
            .with_context(|| format!("hooks.{} is not an array", event))?;

        // Check if a kiok hook already exists in this event.
        let already_exists = arr.iter().any(|group| {
            group.get("hooks")
                .and_then(|h| h.as_array())
                .is_some_and(|hooks| {
                    hooks.iter().any(|h| {
                        h.get("command")
                            .and_then(|c| c.as_str())
                            .is_some_and(|c| c.contains(KIOK_MARKER))
                    })
                })
        });

        if already_exists {
            skipped.push(event);
        } else {
            arr.push(serde_json::json!({
                "hooks": [{
                    "type": "command",
                    "command": command
                }]
            }));
            added.push(event);
        }
    }

    // Write back.
    let formatted = serde_json::to_string_pretty(&settings)
        .context("Could not serialize settings")?;
    fs::write(&settings_path, formatted.as_bytes())
        .with_context(|| format!("Could not write {}", settings_path.display()))?;

    if !added.is_empty() {
        eprintln!("setup: added hooks: {}", added.join(", "));
    }
    if !skipped.is_empty() {
        eprintln!("setup: already configured: {}", skipped.join(", "));
    }

    Ok(())
}

/// Print the hook configuration JSON that the user should add to
/// ~/.claude/settings.json manually.
fn print_hook_config() -> Result<()> {
    let mut hooks = serde_json::Map::new();
    for &(event, command) in KIOK_HOOKS {
        hooks.insert(event.to_owned(), serde_json::json!([{
            "hooks": [{ "type": "command", "command": command }]
        }]));
    }

    let config = serde_json::json!({ "hooks": hooks });
    println!("\nAdd the following to ~/.claude/settings.json:");
    println!("{}", serde_json::to_string_pretty(&config)?);

    Ok(())
}
