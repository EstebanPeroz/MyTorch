mod cli;

use std::path::Path;
use std::process::ExitCode;

use cli::{Args, HELP};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(84)
        }
    }
}

fn run() -> Result<(), String> {
    let Some(args) = Args::try_parse()? else {
        print!("{HELP}");
        return Ok(());
    };
    for request in args.requests() {
        generate(request.config_file(), request.count())?;
    }
    Ok(())
}

fn generate(_config_file: &Path, _count: usize) -> Result<(), String> {
    todo!()
}
