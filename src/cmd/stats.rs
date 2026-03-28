use anyhow::Result;

use crate::db::Database;
use super::save::db_path;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run the stats command: print session and chunk counts for the database.
pub fn run() -> Result<()> {
    let path = db_path()?;

    if !path.exists() {
        println!("Database: {} (not found)", path.display());
        println!("Sessions: 0");
        println!("Chunks:   0");
        return Ok(());
    }

    let db = Database::open(&path)?;
    let s = db.stats()?;

    // Display the path relative to home if possible, otherwise absolute.
    let display_path = if let Ok(home) = std::env::var("HOME") {
        let home_path = std::path::Path::new(&home);
        if let Ok(rel) = path.strip_prefix(home_path) {
            format!("~/{}", rel.display())
        } else {
            path.display().to_string()
        }
    } else {
        path.display().to_string()
    };

    println!("Database:   {}", display_path);
    println!("Sessions:   {}", s.sessions);
    println!("Chunks:     {}", s.chunks);
    println!("Embeddings: {} / {}", s.embeddings, s.chunks);

    Ok(())
}
