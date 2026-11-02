use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap::error::ErrorKind;
use clap::{Arg, ArgAction, ArgGroup, ArgMatches, Command};

pub const HELP: &str = "USAGE
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
    Train { save_file: PathBuf },
}

pub struct Args {
    mode: Mode,
    load_file: PathBuf,
    chess_file: PathBuf,
}

impl Args {
    pub fn try_parse() -> Result<Option<Self>, String> {
        Self::try_parse_from(std::env::args_os())
    }

    pub fn mode(&self) -> &Mode {
        &self.mode
    }

    pub fn load_file(&self) -> &Path {
        &self.load_file
    }

    pub fn chess_file(&self) -> &Path {
        &self.chess_file
    }

    fn try_parse_from<I, T>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        match Self::command().try_get_matches_from(args) {
            Ok(matches) => Ok(Some(Self::from_matches(&matches))),
            Err(error) if error.kind() == ErrorKind::DisplayHelp => Ok(None),
            Err(error) => Err(error.render().to_string().trim_end().to_string()),
        }
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
        Args {
            mode,
            load_file,
            chess_file,
        }
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
    fn help_is_not_an_error() {
        assert!(matches!(parse(&["--help"]), Ok(None)));
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
