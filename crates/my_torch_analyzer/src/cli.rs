use std::path::{Path, PathBuf};

use clap::{Arg, ArgAction, ArgGroup, ArgMatches, Command};
use cli_core::Cli;

const HELP: &str = include_str!("help.txt");

pub enum Mode {
    Predict,
    Train { save_file: PathBuf },
}

pub struct Args {
    mode: Mode,
    load_file: PathBuf,
    chess_file: PathBuf,
}

impl Args {
    pub fn mode(&self) -> &Mode {
        &self.mode
    }

    pub fn load_file(&self) -> &Path {
        &self.load_file
    }

    pub fn chess_file(&self) -> &Path {
        &self.chess_file
    }
}

impl Cli for Args {
    const HELP: &'static str = HELP;

    fn command() -> Command {
        Command::new("my_torch_analyzer")
            .arg(
                Arg::new("predict")
                    .long("predict")
                    .action(ArgAction::SetTrue),
            )
            .arg(Arg::new("train").long("train").action(ArgAction::SetTrue))
            .arg(
                Arg::new("save")
                    .long("save")
                    .value_name("SAVEFILE")
                    .value_parser(clap::value_parser!(PathBuf))
                    .conflicts_with("predict"),
            )
            .group(
                ArgGroup::new("mode")
                    .args(["predict", "train"])
                    .required(true),
            )
            .arg(
                Arg::new("LOADFILE")
                    .required(true)
                    .value_parser(clap::value_parser!(PathBuf)),
            )
            .arg(
                Arg::new("CHESSFILE")
                    .required(true)
                    .value_parser(clap::value_parser!(PathBuf)),
            )
    }

    fn from_matches(matches: &ArgMatches) -> Result<Self, String> {
        let load_file = matches.get_one::<PathBuf>("LOADFILE").unwrap().clone();
        let chess_file = matches.get_one::<PathBuf>("CHESSFILE").unwrap().clone();
        let mode = if matches.get_flag("train") {
            Mode::Train {
                save_file: matches
                    .get_one::<PathBuf>("save")
                    .cloned()
                    .unwrap_or_else(|| load_file.clone()),
            }
        } else {
            Mode::Predict
        };
        Ok(Args {
            mode,
            load_file,
            chess_file,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Args>, String> {
        Args::try_parse_from(std::iter::once("my_torch_analyzer").chain(args.iter().copied()))
    }

    fn train_save_file(args: &Args) -> Option<&Path> {
        match args.mode() {
            Mode::Train { save_file } => Some(save_file),
            Mode::Predict => None,
        }
    }

    #[test]
    fn parses_predict_mode() {
        let args = parse(&["--predict", "net.nn", "boards.txt"])
            .unwrap()
            .unwrap();
        assert!(matches!(args.mode(), Mode::Predict));
        assert_eq!(args.load_file(), Path::new("net.nn"));
        assert_eq!(args.chess_file(), Path::new("boards.txt"));
    }

    #[test]
    fn train_mode_saves_into_load_file_by_default() {
        let args = parse(&["--train", "net.nn", "boards.txt"])
            .unwrap()
            .unwrap();
        assert_eq!(train_save_file(&args), Some(Path::new("net.nn")));

        let args = parse(&["--train", "--save", "out.nn", "net.nn", "boards.txt"])
            .unwrap()
            .unwrap();
        assert_eq!(train_save_file(&args), Some(Path::new("out.nn")));
    }

    #[test]
    fn rejects_invalid_arguments() {
        let invalid: [&[&str]; 6] = [
            &["net.nn", "boards.txt"],
            &["--predict", "--train", "net.nn", "boards.txt"],
            &["--predict", "--save", "out.nn", "net.nn", "boards.txt"],
            &["--predict", "net.nn"],
            &["--predict", "net.nn", "boards.txt", "extra"],
            &["--unknown", "net.nn", "boards.txt"],
        ];
        for args in invalid {
            assert!(parse(args).is_err(), "{args:?}");
        }
    }
}
