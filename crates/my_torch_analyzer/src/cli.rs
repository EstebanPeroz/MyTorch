use std::path::{Path, PathBuf};

use clap::{Arg, ArgAction, ArgGroup, ArgMatches, Command};
use cli_core::Cli;

pub enum Mode {
    Predict,
    Train {
        save_file: PathBuf,
        dashboard: bool,
        mirror: bool,
    },
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
    const HELP: &'static str = include_str!("help.txt");

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
            // Not in the help text, which is fixed by the subject: an extra, optional way to follow a training.
            .arg(
                Arg::new("dashboard")
                    .long("dashboard")
                    .action(ArgAction::SetTrue)
                    .conflicts_with("predict"),
            )
            // Also an extra: trains on every chessboard and on its mirror (the other side's point of view).
            .arg(
                Arg::new("mirror")
                    .long("mirror")
                    .action(ArgAction::SetTrue)
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
                dashboard: matches.get_flag("dashboard"),
                mirror: matches.get_flag("mirror"),
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
            Mode::Train { save_file, .. } => Some(save_file),
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
    fn mirroring_is_optional_and_only_for_training() {
        let mirror = |args: &Args| matches!(args.mode(), Mode::Train { mirror: true, .. });
        let plain = parse(&["--train", "net.nn", "boards.txt"])
            .unwrap()
            .unwrap();
        assert!(!mirror(&plain));
        let mirrored = parse(&["--mirror", "--train", "net.nn", "boards.txt"])
            .unwrap()
            .unwrap();
        assert!(mirror(&mirrored));
        assert!(parse(&["--predict", "--mirror", "net.nn", "boards.txt"]).is_err());
        assert!(parse(&["--mirror", "net.nn", "boards.txt"]).is_err());
    }

    #[test]
    fn the_dashboard_is_optional_and_only_for_training() {
        let dashboard = |args: &Args| {
            matches!(
                args.mode(),
                Mode::Train {
                    dashboard: true,
                    ..
                }
            )
        };
        let plain = parse(&["--train", "net.nn", "boards.txt"])
            .unwrap()
            .unwrap();
        assert!(!dashboard(&plain));
        let watched = parse(&["--dashboard", "--train", "net.nn", "boards.txt"])
            .unwrap()
            .unwrap();
        assert!(dashboard(&watched));
        assert!(parse(&["--predict", "--dashboard", "net.nn", "boards.txt"]).is_err());
        assert!(parse(&["--dashboard", "net.nn", "boards.txt"]).is_err());
    }
}
