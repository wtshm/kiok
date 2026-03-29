use anyhow::Result;

pub mod embed;
pub mod import_cmd;
pub mod view;
pub mod recall;
pub mod save;

pub mod setup;
pub mod stats;

pub fn run(command: crate::Commands) -> Result<()> {
    match command {
        crate::Commands::Save { project } => {
            save::run(&project)
        }
        crate::Commands::Recall { query, project, session, count } => {
            if let Some(sid) = session {
                recall::run_session(&sid, count)
            } else {
                // clap guarantees these are present when --session is absent.
                recall::run(&query.unwrap(), &project.unwrap(), count)
            }
        }
        crate::Commands::Embed => {
            embed::run()?;
            Ok(())
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
