# Dependencies

No neural network library (PyTorch, TensorFlow…) is used. Everything related to the network (layers,
activations, losses, backpropagation, training) is written from scratch in `my_torch`. It uses `ndarray` for
the matrix arithmetic, `rand` for random numbers and `serde_json` to save networks. The programs use two other
external crates.

| Crate | Version | Used by | Purpose |
|---|---|---|---|
| [clap](https://docs.rs/clap) | 4.6.7 | `cli_core`, both programs | command-line parsing |
| [chess_fen_parser](https://crates.io/crates/chess_fen_parser) | 0.1.1 | `my_torch_analyzer` | reading FEN strings |
| [ndarray](https://docs.rs/ndarray) | 0.17.2 | `my_torch` (`matrix.rs` only) | matrix arithmetic |
| [rand](https://docs.rs/rand) | 0.10.3 | `my_torch` | seeded random numbers |
| [rand_distr](https://docs.rs/rand_distr) | 0.6.0 | `my_torch` | the normal distribution |
| [serde](https://serde.rs) | 1.0.229 | `my_torch` (network and configuration files only) | turning Rust structs into a file format and back |
| [serde_json](https://docs.rs/serde_json) | 1.0.154 | `my_torch` (network and configuration files only) | the JSON format of network and configuration files |

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

## ndarray

`ndarray` is a general-purpose array library, not a neural network library: it knows nothing about layers,
activations, losses or training. `my_torch` only uses it for matrix arithmetic, inside `matrix.rs`; every other
file works with our own `Matrix` type.

It replaced our own matrix products, which were too slow for 723,019 chessboards. Four libraries were compared
on the three products a training step needs, for a batch of 64 chessboards, 768 inputs and 256 neurons, in
`f32` and release mode, on a 16-core machine:

| Product | Our loops | `ndarray` 0.17.2 | `nalgebra` 0.35.0 | `faer` 0.24.4 (1 thread) | `faer` (default, 16 threads) | `matrixmultiply` 0.3.11 |
|---|---|---|---|---|---|---|
| `X · W` (forward) | 2.1 ms | 0.56 ms | 0.64 ms | 0.39 ms | 3.2 ms | 0.61 ms |
| `Xᵀ · dZ` (weight gradient) | 2.2 ms | 0.49 ms | 1.9 ms | 0.44 ms | 6.3 ms | — |
| `dZ · Wᵀ` (input gradient) | 8.6 ms | 0.57 ms | 1.4 ms | 0.44 ms | 3.8 ms | — |

| | `ndarray` | `nalgebra` | `faer` | `matrixmultiply` |
|---|---|---|---|---|
| Crates added | 6 | 18 | 42 to 79 | 2 |
| Size in `vendor/` | 4 MB | 42 MB | 56 to 164 MB | 0.5 MB |
| License | MIT / Apache-2.0 | Apache-2.0 | MIT | MIT / Apache-2.0 |
| Drawback | — | stores matrices column by column, the opposite of our convention; slow transposed products | very heavy; spreads small products over every core by default, which makes them slower than our loops | only the product, through raw pointers: our code would need `unsafe` |

`ndarray` is almost as fast as `faer` on one thread, for a fraction of its weight, and has a safe API.

**What to know when using it:**

- `Matrix` wraps an `Array2<f32>`. Every result goes through `Matrix::wrap`, which keeps the values stored row
  after row, so `data()` and `row()` can return plain slices.
- `x.t()` is a transposed *view*: it reads the same memory in the other order, without copying. This is how
  `transpose_matmul` and `matmul_transpose` avoid building transposed matrices.
- `dot` calls `matrixmultiply`, on a single thread.
- Its panics on mismatched shapes are not used: `Matrix` checks shapes first and panics with its own message.

**Cost:** 6 crates and 4 MB in `vendor/`, with no C code.

## rand

Training needs random numbers twice: to pick the starting weights, and to shuffle the samples before each
epoch. `rand` provides the generator and the shuffle; `rand_distr` adds the normal distribution (the bell curve)
that starting weights are drawn from.

- Every generator is a `StdRng` created from an explicit seed, so the same seed, data and settings train the
  same network. It lives in a private field of `my_torch::Rng` (`Rng::new(seed)`): `rand` never appears in
  the library's API, and only `rng.rs` creates a generator. `rand` may change `StdRng`'s algorithm in a future
  version, but the vendored version only changes when we update it.
- `rand` is used without its default features, only `std` and `std_rng`: the generator is never seeded from the
  operating system, since every seed is explicit.

**Cost:** 8 folders and 3 MB in `vendor/` (about 43 MB in total now), with no C code. `getrandom` and `r-efi`
are among them because `Cargo.lock` lists them, but they are never compiled.

## serde and serde_json

Network files (`.nn`) are JSON, read and written with `serde_json`; `serde` derives the code that turns Rust
structs into JSON and back. See [my_torch.md](my_torch.md#network-files) for the format.

| | JSON (`serde_json`) | Our own text format | Binary (`bincode`) |
|---|---|---|---|
| Readable in an editor | yes | yes | no |
| Parser to write and test | none | ours | none |
| Weights read back exactly | yes | only if we print enough digits | yes |
| Size for 100,000 weights | about 1.2 MB | about 1.2 MB | 400 KB |
| Errors on a broken file | the field and its line and column | ours to write | little detail |

JSON was chosen for being readable and for its precise error messages, at the cost of files about 3 times
larger than binary ones, which still load in milliseconds.

**What to know when using it:**

- Only `model.rs`, `model_config.rs` and `training_file.rs` use `serde`: private structs (`ModelFile`,
  `LayerFile`, `TrainingFile`…) describe the files, and are converted to and from `Model` and `ModelConfig`.
  No public type derives `Serialize` or `Deserialize`, so `serde` stays out of the API.
- `serde_json` writes each `f32` with the shortest text that reads back as the same `f32`, so weights survive a
  save and load exactly. It writes NaN and infinity as `null`, which is why `save` refuses them.
- `#[serde(deny_unknown_fields)]` turns a misspelled field into an error. It does not work together with
  `#[serde(flatten)]`, which the optimizer settings use: an unknown optimizer setting is refused by
  `optimizer_from_settings` instead.
- Its error messages fit on one line and give the line and column of the problem.

**Cost:** 6 crates and about 3 MB in `vendor/` (about 46 MB in total now), with no C code. `serde_derive`
reuses `syn`, `quote` and `proc-macro2`, which were already there.

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
