use anyhow::Result;
use clap::{Parser, Subcommand};

mod cmd;
mod db;

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
    /// Recall related memories for current session
    Recall {
        /// Project directory path
        #[arg(long)]
        project: String,
        /// Number of results to return
        #[arg(long, default_value = "5")]
        count: usize,
    },
    /// Search memories manually
    Search {
        /// Search query
        query: String,
        /// Filter by project
        #[arg(long)]
        project: Option<String>,
        /// Number of results to return
        #[arg(long, default_value = "10")]
        count: usize,
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
