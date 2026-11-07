mod cli;
mod modes;

use std::process::ExitCode;

use cli::{Args, Mode};
use modes::Options;

fn main() -> ExitCode {
    cli_core::run(|args: Args| match args.mode() {
        Mode::Predict { chess_file } => modes::predict(args.load_file(), chess_file),
        Mode::Dump => modes::dump(args.load_file()),
        Mode::Train {
            chess_file,
            save_file,
            dashboard,
            mirror,
        } => modes::train(
            args.load_file(),
            chess_file,
            save_file,
            Options {
                dashboard: *dashboard,
                mirror: *mirror,
            },
        ),
    })
}
