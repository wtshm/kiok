use anyhow::Result;

pub mod recall;
pub mod save;
pub mod search_cmd;

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
        crate::Commands::Import => {
            eprintln!("import");
            Ok(())
        }
        crate::Commands::Stats => {
            eprintln!("stats");
            Ok(())
        }
        crate::Commands::Setup => {
            eprintln!("setup");
            Ok(())
        }
    }
}
