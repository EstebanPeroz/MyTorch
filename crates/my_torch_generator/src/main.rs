mod cli;

use std::path::Path;
use std::process::ExitCode;

use cli::Args;

fn main() -> ExitCode {
    cli_core::run(|args: Args| {
        for request in args.requests() {
            generate(request.config_file(), request.count())?;
        }
        Ok(())
    })
}

fn generate(_config_file: &Path, _count: usize) -> Result<(), String> {
    Err("error: network generation is not implemented yet".into())
}
