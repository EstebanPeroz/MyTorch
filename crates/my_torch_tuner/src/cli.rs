use std::path::{Path, PathBuf};

use clap::{Arg, ArgMatches, Command};
use cli_core::Cli;

const DEFAULT_TRIALS: usize = 12;
const DEFAULT_ROUNDS: usize = 3;
const DEFAULT_EPOCHS: usize = 5;
const DEFAULT_VALIDATION: f64 = 0.2;

#[derive(Debug)]
pub struct Args {
    base_config: PathBuf,
    search_space: PathBuf,
    chess_file: PathBuf,
    trials: usize,
    rounds: usize,
    epochs: usize,
    samples: Option<usize>,
    validation: f64,
    seed: Option<u64>,
    log: PathBuf,
    output: PathBuf,
}

impl Args {
    pub fn base_config(&self) -> &Path {
        &self.base_config
    }

    pub fn search_space(&self) -> &Path {
        &self.search_space
    }

    pub fn chess_file(&self) -> &Path {
        &self.chess_file
    }

    pub fn trials(&self) -> usize {
        self.trials
    }

    pub fn rounds(&self) -> usize {
        self.rounds
    }

    pub fn epochs(&self) -> usize {
        self.epochs
    }

    pub fn samples(&self) -> Option<usize> {
        self.samples
    }

    pub fn validation(&self) -> f64 {
        self.validation
    }

    /// The seed asked for, if any. Otherwise the search uses the seed of the base configuration.
    pub fn seed(&self) -> Option<u64> {
        self.seed
    }

    pub fn log(&self) -> &Path {
        &self.log
    }

    pub fn output(&self) -> &Path {
        &self.output
    }
}

fn option(name: &'static str, value_name: &'static str) -> Arg {
    Arg::new(name).long(name).value_name(value_name)
}

fn count(matches: &ArgMatches, name: &str, default: usize) -> Result<usize, String> {
    match matches.get_one::<String>(name) {
        None => Ok(default),
        Some(text) => text
            .parse::<usize>()
            .ok()
            .filter(|&count| count > 0)
            .ok_or_else(|| format!("error: invalid value for --{name}: '{text}'")),
    }
}

impl Cli for Args {
    const HELP: &'static str = include_str!("help.txt");

    fn command() -> Command {
        Command::new("my_torch_tuner")
            .arg(option("trials", "N"))
            .arg(option("rounds", "N"))
            .arg(option("epochs", "N"))
            .arg(option("samples", "N"))
            .arg(option("validation", "F"))
            .arg(option("seed", "N"))
            .arg(option("log", "FILE").value_parser(clap::value_parser!(PathBuf)))
            .arg(option("output", "FILE").value_parser(clap::value_parser!(PathBuf)))
            .arg(Arg::new("BASE_CONFIG").required(true))
            .arg(Arg::new("SEARCH_SPACE").required(true))
            .arg(Arg::new("CHESSFILE").required(true))
    }

    fn from_matches(matches: &ArgMatches) -> Result<Self, String> {
        let path = |name: &str| PathBuf::from(matches.get_one::<String>(name).unwrap());
        let samples = matches
            .contains_id("samples")
            .then(|| count(matches, "samples", 0))
            .transpose()?;
        let validation = match matches.get_one::<String>("validation") {
            None => DEFAULT_VALIDATION,
            Some(text) => text
                .parse::<f64>()
                .ok()
                .filter(|share| *share > 0.0 && *share < 1.0)
                .ok_or_else(|| format!("error: invalid value for --validation: '{text}'"))?,
        };
        let seed = matches
            .get_one::<String>("seed")
            .map(|text| {
                text.parse::<u64>()
                    .map_err(|_| format!("error: invalid value for --seed: '{text}'"))
            })
            .transpose()?;
        Ok(Args {
            base_config: path("BASE_CONFIG"),
            search_space: path("SEARCH_SPACE"),
            chess_file: path("CHESSFILE"),
            trials: count(matches, "trials", DEFAULT_TRIALS)?,
            rounds: count(matches, "rounds", DEFAULT_ROUNDS)?,
            epochs: count(matches, "epochs", DEFAULT_EPOCHS)?,
            samples,
            validation,
            seed,
            log: matches
                .get_one::<PathBuf>("log")
                .cloned()
                .unwrap_or_else(|| PathBuf::from("tuner_log.jsonl")),
            output: matches
                .get_one::<PathBuf>("output")
                .cloned()
                .unwrap_or_else(|| PathBuf::from("best.conf")),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Args>, String> {
        Args::try_parse_from(std::iter::once("my_torch_tuner").chain(args.iter().copied()))
    }

    #[test]
    fn has_defaults() {
        let args = parse(&["base.conf", "space.json", "boards.txt"])
            .unwrap()
            .unwrap();
        assert_eq!(args.base_config(), Path::new("base.conf"));
        assert_eq!(args.search_space(), Path::new("space.json"));
        assert_eq!(args.chess_file(), Path::new("boards.txt"));
        assert_eq!((args.trials(), args.rounds(), args.epochs()), (12, 3, 5));
        assert_eq!((args.samples(), args.seed()), (None, None));
        assert_eq!(args.validation(), 0.2);
        assert_eq!(args.log(), Path::new("tuner_log.jsonl"));
        assert_eq!(args.output(), Path::new("best.conf"));
    }

    #[test]
    fn reads_every_option_in_any_order() {
        let args = parse(&[
            "--trials",
            "20",
            "base.conf",
            "--rounds",
            "2",
            "space.json",
            "--epochs",
            "4",
            "--samples",
            "500",
            "--validation",
            "0.3",
            "--seed",
            "9",
            "--log",
            "a.jsonl",
            "--output",
            "b.conf",
            "boards.txt",
        ])
        .unwrap()
        .unwrap();
        assert_eq!((args.trials(), args.rounds(), args.epochs()), (20, 2, 4));
        assert_eq!((args.samples(), args.seed()), (Some(500), Some(9)));
        assert_eq!(args.validation(), 0.3);
        assert_eq!(args.log(), Path::new("a.jsonl"));
        assert_eq!(args.output(), Path::new("b.conf"));
    }
}
