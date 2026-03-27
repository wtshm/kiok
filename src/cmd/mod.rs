use anyhow::Result;

pub mod embed;
pub mod import_cmd;
pub mod view;
pub mod recall;
pub mod save;
pub mod search_cmd;
pub mod setup;
pub mod stats;

pub fn run(command: crate::Commands) -> Result<()> {
    match command {
        crate::Commands::Save { project } => {
            save::run(&project)
        }
        crate::Commands::Recall { project, count } => {
            recall::run(&project, count)
        }
        crate::Commands::Search { query, project, count } => {
            search_cmd::run(&query, project.as_deref(), count)
        }
        crate::Commands::Embed => {
            embed::run()
        }
        crate::Commands::Import => {
            import_cmd::run()
        }
        crate::Commands::Stats => {
            stats::run()
        }
        crate::Commands::Setup => {
            setup::run()
        }
        crate::Commands::View { port } => {
            view::run(port)
        }
    }
}
