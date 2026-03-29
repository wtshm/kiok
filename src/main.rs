use anyhow::Result;
use clap::{Parser, Subcommand};

// Re-use the public modules from the library crate.
use kiok::chunk;
use kiok::db;
use kiok::embed;
use kiok::policy;
use kiok::search;

mod cmd;

#[derive(Parser)]
#[command(name = "kiok", about = "Memory engine for Claude Code")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Save session conversation to memory
    Save {
        /// Project directory path
        #[arg(long)]
        project: String,
    },
    /// Search and recall related memories
    Recall {
        /// Search query
        #[arg(required_unless_present = "session", conflicts_with = "session")]
        query: Option<String>,
        /// Project directory path
        #[arg(long, required_unless_present = "session", conflicts_with = "session")]
        project: Option<String>,
        /// Look up chunks by session ID (full or prefix)
        #[arg(long)]
        session: Option<String>,
        /// Number of results to return
        #[arg(long, default_value = "5")]
        count: usize,
    },
    /// Embed chunks that don't have embeddings yet
    Embed,
    /// Browse memory database in the browser
    View {
        /// Port to listen on
        #[arg(long, default_value = "8718")]
        port: u16,
    },
    /// Import existing Claude Code sessions
    Import,
    /// Show database statistics
    Stats,
    /// Set up kiok (download model, configure hooks)
    Setup,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    cmd::run(cli.command)
}
