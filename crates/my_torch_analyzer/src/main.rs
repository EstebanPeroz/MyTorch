mod cli;
mod modes;

use std::process::ExitCode;

use cli::{Args, Mode};

fn main() -> ExitCode {
    cli_core::run(|args: Args| match args.mode() {
        Mode::Predict => modes::predict(args.load_file(), args.chess_file()),
        Mode::Train { save_file } => modes::train(args.load_file(), args.chess_file(), save_file),
    })
}
