# MyTorch

A neural network library written from scratch in Rust, and `my_torch_analyzer`, a program that uses it to
predict the state of a chess game (check, checkmate, stalemate or nothing) from a chessboard in
[FEN](https://en.wikipedia.org/wiki/Forsyth%E2%80%93Edwards_Notation) notation.

> **Status:** command-line parsing, labels and CHESSFILE reading are done. Training, prediction and network
> generation are not implemented yet: those commands currently print `error: … is not implemented yet` and
> exit with 84.

## Requirements

- Rust 1.87 or newer (edition 2024)
- A C compiler (`cc`): two dependencies of `chess_fen_parser` compile C code during the build

Dependencies are vendored in `vendor/`, so the build works offline.

## Build

```sh
make
```

| Target | Effect |
|---|---|
| `make` | builds `my_torch_analyzer` and `my_torch_generator` in release mode and copies them to the repo root |
| `make clean` | removes the build directory (`target/`) |
| `make fclean` | `clean`, then removes both binaries |
| `make re` | `fclean`, then `make` |

## Usage

Both programs exit with **0** on success and **84** on any error, with the error message on stderr.
`--help` prints the usage on stdout and exits with 0.

### my_torch_analyzer

```
USAGE
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
```

```sh
./my_torch_analyzer --train --save my_torch_network.nn blank.nn train.txt
./my_torch_analyzer --predict my_torch_network.nn chessboards.txt
```

Each line of a CHESSFILE is a FEN followed, optionally, by its expected output:

```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
```

The possible outputs are `Nothing`, `Check White`, `Check Black`, `Checkmate White`, `Checkmate Black` and
`Stalemate`. The color is the attacking side. See [docs/labels.md](docs/labels.md).

Labels are required with `--train` and optional with `--predict`. Blank lines are skipped, and any error
points at the line, e.g. `error: boards.txt:12: unknown label 'Chek'`. See
[docs/chess-file.md](docs/chess-file.md).

The training data goes in `dataset/` (ignored by git). See
[docs/dataset.md](docs/dataset.md).

### my_torch_generator

```
USAGE
        ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
        config_file_i    Configuration file containing description of a neural network we want to
generate.
        nb_i             Number of neural networks to generate based on the configuration file.
```

```sh
./my_torch_generator basic_network.conf 3
```

## Repository layout

```
crates/
├── my_torch/             neural network library (chess-agnostic)
├── cli_core/             shared command-line handling: parsing, --help, exit code 84
├── my_torch_analyzer/    the chess analyzer (library for chess logic + thin binary)
├── my_torch_generator/   generates blank networks from configuration files
└── cli_test/             dev-only helpers for end-to-end tests
docs/                     technical documentation
vendor/                   vendored dependencies
```

The technical documentation starts at [docs/README.md](docs/README.md).

## Development

```sh
uv tool install pre-commit
pre-commit install
cargo test --workspace    # unit and end-to-end tests, also run by CI
```

On each commit the hooks run `cargo clippy --fix`, `cargo fmt` and basic file checks, and check that commit
messages follow [Conventional Commits](https://www.conventionalcommits.org) (e.g. `feat(lib): add dropout`).
Warnings are errors (`.cargo/config.toml`).

**Adding a dependency:** declare its version once in the root `Cargo.toml`, use it from the crate that needs it,
then vendor it:

```toml
# Cargo.toml (root)
[workspace.dependencies]
some_crate = "1.2.3"

# crates/<crate>/Cargo.toml
[dependencies]
some_crate.workspace = true
```

```sh
cargo vendor          # updates Cargo.lock and vendor/
git add Cargo.toml Cargo.lock crates/<crate>/Cargo.toml vendor
git commit --author "Yohann Boniface <edhyjox@gmail.com>" -m "build(deps): ..."
```

A hook blocks the commit if `vendor/` is out of date. Never edit files in `vendor/`: Cargo checks their
checksums and the build fails. See [docs/dependencies.md](docs/dependencies.md).
