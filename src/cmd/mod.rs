use anyhow::Result;

pub fn run(command: crate::Commands) -> Result<()> {
    match command {
        crate::Commands::Save { project } => {
            eprintln!("save: project={}", project);
            Ok(())
        }
        crate::Commands::Recall { project, count } => {
            eprintln!("recall: project={}, count={}", project, count);
            Ok(())
        }
        crate::Commands::Search { query, project, count } => {
            eprintln!("search: query={}, project={:?}, count={}", query, project, count);
            Ok(())
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
