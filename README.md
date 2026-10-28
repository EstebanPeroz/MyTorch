# MyTorch

A neural network library written from scratch in Rust, and `my_torch_analyzer`, a program that uses it to
predict the state of a chess game.

## Build

```sh
make
```

## Usage

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE

DESCRIPTION
--train     Launch the neural network in training mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation and the expected output separated by
space. If specified, the newly trained neural network will be saved in SAVEFILE. Otherwise,
it will be saved in the original LOADFILE.

--predict   Launch the neural network in prediction mode. Each chessboard in FILE must
contain inputs to send to the neural network in FEN notation, and optionally an expected output.

--save      Save neural network into SAVEFILE. Only works in train mode.

LOADFILE    File containing a neural network
CHESSFILE   File containing chessboards
```

```sh
./my_torch_analyzer --predict my_torch_network.nn chessboards.txt
```

## Development

```sh
uv tool install pre-commit
pre-commit install
```

On each commit the hooks run `cargo clippy --fix`, `cargo fmt` and basic file checks, and check that commit
messages follow [Conventional Commits](https://www.conventionalcommits.org) (e.g. `feat(lib): add dropout`).
Warnings are errors (`.cargo/config.toml`).

**Adding a dependency:** Add the dependency by hand in `cargo.toml` and then vendor the dependency:
```sh
cargo vendor          # updates Cargo.lock and vendor/
git add Cargo.toml Cargo.lock vendor
```

A hook blocks the commit if `vendor/` is out of date.
