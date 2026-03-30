use anyhow::{Context, Result};
use futures_util::StreamExt;
use indicatif::{ProgressBar, ProgressStyle};
use std::fs;
use std::io::{self, BufRead, Write};

use crate::db::Database;
use crate::embed::ensure_ort_dylib;
use super::save::{data_dir, db_path, model_dir};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const HF_BASE_URL: &str =
    "https://huggingface.co/sirasagi62/ruri-v3-310m-ONNX/resolve/main";

/// (remote_path, local_filename) pairs for model download.
const MODEL_FILES: &[(&str, &str)] = &[
    ("onnx/model.onnx", "model.onnx"),
    ("tokenizer.json", "tokenizer.json"),
];

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

    // --- [1/5] Check ONNX Runtime ---
    eprintln!("[1/5] Checking ONNX Runtime...");
    if let Some(path) = ensure_ort_dylib() {
        eprintln!("      found at {}", path.display());
    } else {
        let install_hint = if cfg!(target_os = "macos") {
            "  brew install onnxruntime"
        } else if cfg!(target_os = "linux") {
            "  sudo apt install libonnxruntime-dev   # Debian/Ubuntu\n  \
             # or download from https://github.com/microsoft/onnxruntime/releases"
        } else {
            "  https://github.com/microsoft/onnxruntime/releases"
        };
        anyhow::bail!(
            "ONNX Runtime not found.\n\n\
             Install it and run setup again:\n{}\n",
            install_hint
        );
    }

    // --- [2/5] Download model files ---
    eprintln!("[2/5] Downloading model files...");
    let rt = tokio::runtime::Runtime::new().context("Failed to create Tokio runtime")?;
    rt.block_on(download_model_files(&model_dir))?;

    // --- [3/5] Initialize database ---
    eprintln!("[3/5] Initializing database...");
    let data = data_dir()?;
    fs::create_dir_all(&data)
        .with_context(|| format!("Could not create data directory {}", data.display()))?;
    let the_db_path = db_path()?;
    Database::open(&the_db_path).context("Failed to initialize database")?;
    eprintln!("      initialized at {}", the_db_path.display());

    // --- [4/5] Configure hooks in ~/.claude/settings.json ---
    eprintln!("[4/5] Configuring hooks...");
    if confirm("      Add kiok hooks to ~/.claude/settings.json? [y/N] ")? {
        install_hooks()?;
    } else {
        print_hook_config()?;
    }

    // --- [5/5] Import & embedding ---
    eprintln!("[5/5] Import & embedding...");
    let imported_chunks = if confirm("      Import existing Claude Code sessions? [y/N] ")? {
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

    if imported_chunks > 0 {
        eprintln!(
            "      Note: Embedding {} chunks with Ruri v3 may take several minutes depending on your machine.",
            imported_chunks
        );
        if confirm("      Run embedding now? [y/N] ")? {
            match super::embed::run()? {
                super::embed::EmbedOutcome::Locked => {
                    eprintln!("      another embed process is running. Run `kiok embed` later.");
                }
                super::embed::EmbedOutcome::Unavailable => {
                    eprintln!("      embedding model not available. Try `kiok embed` after verifying the model files.");
                }
                super::embed::EmbedOutcome::Done(n) => {
                    eprintln!("      embedded {} chunks.", n);
                }
            }
        } else {
            eprintln!("      Skipped. You can run `kiok embed` later to enable vector search.");
        }
    }

    // --- Done ---
    eprintln!();
    eprintln!("Setup complete!");
    eprintln!("  Database : {}", the_db_path.display());
    if imported_chunks > 0 {
        eprintln!("  Imported : {} chunks", imported_chunks);
    }
    eprintln!();
    eprintln!("kiok is ready. Memories will be saved automatically via hooks.");

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

    for &(remote_path, local_name) in MODEL_FILES {
        let dest = model_dir.join(local_name);

        if dest.exists() {
            eprintln!("      {} already exists, skipping download", local_name);
            continue;
        }

        let url = format!("{}/{}", HF_BASE_URL, remote_path);
        eprintln!("      downloading {} ...", local_name);

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

        let total_size = response.content_length().unwrap_or(0);

        let pb = ProgressBar::new(total_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")
                .expect("invalid progress bar template")
                .progress_chars("#>-"),
        );

        let part_path = model_dir.join(format!("{}.part", local_name));
        let mut file = fs::File::create(&part_path)
            .with_context(|| format!("Could not create temp file {}", part_path.display()))?;

        let result: Result<()> = async {
            let mut stream = response.bytes_stream();
            while let Some(chunk) = stream.next().await {
                let chunk = chunk
                    .with_context(|| format!("Error reading stream for {}", local_name))?;
                file.write_all(&chunk)
                    .with_context(|| format!("Could not write to {}", part_path.display()))?;
                pb.inc(chunk.len() as u64);
            }
            drop(file);
            fs::rename(&part_path, &dest)
                .with_context(|| format!("Could not rename {} to {}", part_path.display(), dest.display()))?;
            Ok(())
        }
        .await;

        pb.finish_and_clear();

        if result.is_err() {
            let _ = fs::remove_file(&part_path);
        }
        result?;

        eprintln!("      saved {}", dest.display());
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
        eprintln!("      added hooks: {}", added.join(", "));
    }
    if !skipped.is_empty() {
        eprintln!("      already configured: {}", skipped.join(", "));
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
