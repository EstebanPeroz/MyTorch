mod cli;
mod evaluate;
mod search;
mod space;

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;
use std::process::ExitCode;

use cli::Args;
use evaluate::Data;
use my_torch::ModelConfig;
use search::{Budget, Trial};
use serde_json::Value;
use space::SearchSpace;

fn main() -> ExitCode {
    cli_core::run(|args: Args| {
        tune(&args).map_err(|error| format!("error: {}", error.trim_start_matches("error: ")))
    })
}

fn read(path: &Path) -> Result<String, String> {
    fs::read_to_string(path).map_err(|error| format!("cannot read {}: {error}", path.display()))
}

fn tune(args: &Args) -> Result<(), String> {
    let text = read(args.base_config())?;
    ModelConfig::from_json(&text)
        .map_err(|error| format!("{}: {error}", args.base_config().display()))?;
    let base: Value = serde_json::from_str(&text).expect("the configuration was just parsed");
    let space = SearchSpace::from_json(&read(args.search_space())?)
        .map_err(|error| format!("{}: {error}", args.search_space().display()))?;
    let seed = args.seed().unwrap_or_else(|| {
        base["training"]["seed"]
            .as_u64()
            .expect("the configuration was just parsed")
    });
    let data = Data::read(args.chess_file(), args.samples(), args.validation(), seed)?;
    println!(
        "searching with {} chessboards to train on and {} to validate",
        data.training_len(),
        data.validation_len()
    );
    let mut log = BufWriter::new(
        File::create(args.log())
            .map_err(|error| format!("cannot write {}: {error}", args.log().display()))?,
    );
    let budget = Budget {
        trials: args.trials(),
        rounds: args.rounds(),
        epochs: args.epochs(),
        seed,
    };
    let found = search::search(
        &space,
        budget,
        |params, epochs| evaluate::evaluate(&data, &base, params, epochs),
        |trial| {
            println!("{}", describe(trial));
            writeln!(log, "{}", trial.to_json())
                .and_then(|()| log.flush())
                .map_err(|error| format!("cannot write {}: {error}", args.log().display()))
        },
    )?;
    let best = space::SearchSpace::apply(&base, &found.best.params)?;
    let mut text = serde_json::to_string_pretty(&best).expect("a JSON value can be written");
    text.push('\n');
    fs::write(args.output(), text)
        .map_err(|error| format!("cannot write {}: {error}", args.output().display()))?;
    println!("{} trials run", found.trials.len());
    println!("best: {}", describe(&found.best));
    println!(
        "wrote {} and {}",
        args.output().display(),
        args.log().display()
    );
    Ok(())
}

fn describe(trial: &Trial) -> String {
    let head = format!(
        "round {} trial {} ({} epochs)",
        trial.round + 1,
        trial.id + 1,
        trial.epochs
    );
    let params = serde_json::to_string(&trial.params).expect("a JSON value can be written");
    match &trial.result {
        Ok(score) => format!(
            "{head}: accuracy {:.2}%, loss {:.4}, {params}",
            score.accuracy * 100.0,
            score.loss
        ),
        Err(error) => format!("{head}: failed ({error}), {params}"),
    }
}
