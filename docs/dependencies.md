# Dependencies

The subject forbids neural network libraries (PyTorch, TensorFlow…). Everything related to the network is
written from scratch in `my_torch`, which has no dependencies. The programs use two external crates.

| Crate | Version | Used by | Purpose |
|---|---|---|---|
| [clap](https://docs.rs/clap) | 4.6.7 | `cli_core`, both programs | command-line parsing |
| [chess_fen_parser](https://crates.io/crates/chess_fen_parser) | 0.1.1 | `my_torch_analyzer` | reading FEN strings |

## clap

The arguments are described with clap's builder API (`Command`, `Arg`). See [cli.md](cli.md).

## chess_fen_parser

Reading a FEN only needs a few dozen lines, but a crate was preferred over our own parser. Four FEN-only crates
were compared on malformed input:

| Input | `fen` 0.1.0 | `fen-rs` 0.1.5 | `fenx` 0.1.1 | `chess_fen_parser` 0.1.1 |
|---|---|---|---|---|
| 9 pieces on a rank | wrong board or panic | not checked | — | rejected |
| Rank shorter than 8 squares | wrong board | not checked | — | rejected |
| Non-ASCII en passant square (`é`) | panic | ignored | — | rejected |
| Missing fields | rejected | panic | — | rejected |
| License | none | MIT | MIT / Apache-2.0 | MIT |
| Usable | yes | no: a piece's color and type are private | contains move logic | yes |

`chess_fen_parser` is the only one that rejects every malformed input without panicking. It replaced `fen`,
which was used first.

**What to know when using it:**

- `parse_fen(&str) -> Result<FenRecord, FenError>`.
- The six FEN fields must be separated by exactly one space.
- `FenRecord::board` is a `Vec<Vec<Option<Piece>>>` with **rank 8 first**, as in the FEN text.
- The en passant square is an `Option<String>` (`"e3"`), not an index.
- Its types only derive `Debug` (no `Clone`, no `PartialEq`).
- Grammar errors (`FenError::ParseError`) span several lines; they must be reduced to one line before being
  shown to the user.

**Cost:** it brings 24 crates into `vendor/`, which grew to about 36 MB in total, and two of them (`psm`,
`stacker`) compile C code in their build scripts, which is why the build needs a C compiler.

## Vendoring

Every dependency is copied into `vendor/` and `.cargo/config.toml` tells Cargo to use only that folder. The
build therefore needs no network access, and `make` uses `--frozen`, so it fails instead of silently changing
`Cargo.lock`.

- Add a dependency with the procedure in the [main README](../README.md#development).
- **Never edit a file in `vendor/`.** Cargo checks every file against `.cargo-checksum.json` and refuses to
  build if one changed. A project-wide search-and-replace in an IDE can do this by accident.
- The pre-commit hook `cargo vendor is up to date` blocks commits where `vendor/` does not match `Cargo.lock`.
- **Anchor `.gitignore` patterns** (`/target`, `/debug/`). An unanchored `target` or `debug` also ignores
  folders inside `vendor/` (such as `vendor/cc/src/target/`), so a fresh clone misses those files and fails the
  checksum check.
