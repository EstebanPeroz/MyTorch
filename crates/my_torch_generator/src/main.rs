mod cli;

use std::path::Path;

use cli::Args;

fn main() {
    for request in Args::parse().requests {
        generate(&request.config_file, request.count);
    }
}

fn generate(_config_file: &Path, _count: usize) {
    todo!()
}
