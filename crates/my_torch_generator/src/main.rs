mod cli;

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use cli::Args;
use my_torch::ModelConfig;

fn main() -> ExitCode {
    cli_core::run(|args: Args| {
        let configs = args
            .requests()
            .iter()
            .map(|request| {
                ModelConfig::load(request.config_file())
                    .map(|config| (request, config))
                    .map_err(|error| format!("error: {error}"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (request, config) in configs {
            generate(&config, request.config_file(), request.count())?;
        }
        Ok(())
    })
}

fn generate(config: &ModelConfig, config_file: &Path, count: usize) -> Result<(), String> {
    let name = config_file
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy();
    for index in 0..count {
        let path = PathBuf::from(format!("{name}_{}.nn", index + 1));
        config
            .generate(index as u64)
            .save(&path)
            .map_err(|error| format!("error: {error}"))?;
    }
    Ok(())
}
