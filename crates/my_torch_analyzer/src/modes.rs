use std::path::Path;

pub fn predict(_load_file: &Path, _chess_file: &Path) -> Result<(), String> {
    Err("error: --predict is not implemented yet".into())
}

pub fn train(_load_file: &Path, _chess_file: &Path, _save_file: &Path) -> Result<(), String> {
    Err("error: --train is not implemented yet".into())
}
