mod cli;
mod encode;
mod label;
mod modes;

use std::process::ExitCode;

use cli::{Args, HELP, Mode};

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
    match args.mode() {
        Mode::Predict => modes::predict(args.load_file(), args.chess_file()),
        Mode::Train { save_file } => modes::train(args.load_file(), args.chess_file(), save_file),
    }
}
