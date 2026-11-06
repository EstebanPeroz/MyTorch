use chess_fen_parser::*;
use clap::{Parser, Subcommand};
use std::{fs, path::PathBuf, process::exit};

#[derive(Parser)]
#[command(
    name = "chess_fen_parser",
    about = "Parser for Forsyth-Edwards Notation."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    Parse { file: PathBuf },
    Credits,
}

fn main() {
    match Cli::parse().command {
        Commands::Credits => {
            println!("chess_fen_parser v0.1.0");
            println!("Author: Dmytro Shvets");
        }
        Commands::Parse { file } => {
            let content = match fs::read_to_string(&file) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("Failed to read '{}': {}", file.display(), e);
                    exit(1);
                }
            };
            let input = content.trim();
            if input.is_empty() {
                eprintln!("File '{}' contains no valid FEN string", file.display());
                exit(1);
            }
            match parse_fen(input) {
                Ok(fen) => print_parsing_results(&fen),
                Err(e) => {
                    eprintln!("Error parsing FEN: {}", e);
                    exit(2);
                }
            }
        }
    }
}
