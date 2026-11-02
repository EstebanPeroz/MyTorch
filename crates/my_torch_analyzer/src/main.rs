mod cli;
mod encode;
mod label;
mod modes;

use cli::{Args, Mode};

fn main() {
    let args = Args::parse();
    match args.mode {
        Mode::Predict => modes::predict(&args.load_file, &args.chess_file),
        Mode::Train { save_file } => {
            let save_file = save_file.unwrap_or_else(|| args.load_file.clone());
            modes::train(&args.load_file, &args.chess_file, &save_file)
        }
    }
}
