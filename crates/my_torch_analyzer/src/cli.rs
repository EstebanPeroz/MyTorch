use std::path::PathBuf;
use std::process;

use clap::{Arg, ArgAction, ArgGroup, ArgMatches, Command};

const HELP: &str = "USAGE
        ./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
        --train        Launch the neural network in training mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation and the expected output separated by
space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise,
it will be saved in the original LOADFILE.

        --predict      Launch the neural network in prediction mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation, and optionally an expected output.

        --save         Save neural network into SAVEFILE. Only works in train mode.

        LOADFILE       File containing an artificial neural network
        CHESSFILE      File containing chessboards
";

pub enum Mode {
    Predict,
    Train { save_file: Option<PathBuf> },
}

pub struct Args {
    pub mode: Mode,
    pub load_file: PathBuf,
    pub chess_file: PathBuf,
}

impl Args {
    pub fn parse() -> Self {
        let matches = Self::command().try_get_matches().unwrap_or_else(|error| {
            let code = if error.use_stderr() { 84 } else { 0 };
            let _ = error.print();
            process::exit(code);
        });
        Self::from_matches(&matches)
    }

    fn command() -> Command {
        Command::new("my_torch_analyzer")
            .override_help(HELP)
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

    fn from_matches(matches: &ArgMatches) -> Self {
        let mode = if matches.get_flag("train") {
            Mode::Train {
                save_file: matches.get_one::<PathBuf>("save").cloned(),
            }
        } else {
            Mode::Predict
        };
        Args {
            mode,
            load_file: matches.get_one::<PathBuf>("LOADFILE").unwrap().clone(),
            chess_file: matches.get_one::<PathBuf>("CHESSFILE").unwrap().clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    fn parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::command()
            .try_get_matches_from(std::iter::once("my_torch_analyzer").chain(args.iter().copied()))
            .map(|matches| Args::from_matches(&matches))
    }

    #[test]
    fn parses_predict_mode() {
        let args = parse(&["--predict", "net.nn", "boards.txt"]).unwrap();
        assert!(matches!(args.mode, Mode::Predict));
        assert_eq!(args.load_file, PathBuf::from("net.nn"));
        assert_eq!(args.chess_file, PathBuf::from("boards.txt"));
    }

    #[test]
    fn parses_train_mode_with_and_without_save() {
        let args = parse(&["--train", "net.nn", "boards.txt"]).unwrap();
        assert!(matches!(args.mode, Mode::Train { save_file: None }));

        let args = parse(&["--train", "--save", "out.nn", "net.nn", "boards.txt"]).unwrap();
        assert!(
            matches!(args.mode, Mode::Train { save_file: Some(path) } if path == Path::new("out.nn"))
        );
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
