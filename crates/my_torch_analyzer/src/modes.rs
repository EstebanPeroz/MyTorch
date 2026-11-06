use std::path::Path;

use my_torch_analyzer::chess_file::{ChessFile, Labels};

pub fn predict(_load_file: &Path, chess_file: &Path) -> Result<(), String> {
    ChessFile::read(chess_file, Labels::Optional)?;
    Err("error: --predict is not implemented yet".into())
}

pub fn train(_load_file: &Path, chess_file: &Path, _save_file: &Path) -> Result<(), String> {
    ChessFile::read(chess_file, Labels::Required)?;
    Err("error: --train is not implemented yet".into())
}
