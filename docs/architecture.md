# Architecture

## Crates

The repository is a Cargo workspace. Every crate lives in `crates/`.

```
                 ┌──────────────┐
                 │   my_torch   │   neural network library
                 └──────▲───────┘
            ┌───────────┴───────────┐
┌───────────┴──────────┐ ┌──────────┴───────────┐
│  my_torch_analyzer   │ │  my_torch_generator  │   programs
└───────────┬──────────┘ └──────────┬───────────┘
            └───────────┬───────────┘
                 ┌──────▼───────┐
                 │   cli_core   │   shared command-line handling
                 └──────────────┘
```

| Crate | Kind | Role | External dependencies |
|---|---|---|---|
| `my_torch` | library | The neural network: creating, training, saving and loading networks. Knows nothing about chess. | `ndarray` (matrix arithmetic), `rand`, `rand_distr` (random numbers) |
| `cli_core` | library | The `Cli` trait and `run`, shared by both programs: argument parsing, `--help`, error printing, exit code 84. | `clap` |
| `my_torch_analyzer` | library + binary | The chess analyzer. The library part holds the chess logic (`color`, `label`, `board`, `sample`, `chess_file`); the binary part (`main`, `cli`, `modes`) only wires it together. | `clap`, `chess_fen_parser` |
| `my_torch_generator` | binary | Generates blank networks from configuration files. | `clap` |
| `cli_test` | library, dev-only | Runs a program in end-to-end tests and checks its exit code (against `cli_core::ERROR_EXIT_CODE`), stdout and stderr. Never compiled into the programs. See [testing.md](testing.md). | none |

Dependency arrows only go one way: programs use libraries, and `my_torch` never depends on chess code. This
keeps the neural network module generic: it could learn any other problem.

## Design rules

- **The library stays chess-agnostic.** Board encoding, labels and FEN parsing belong to the analyzer.
- **No chess rules in the analyzer.** The analyzer must not detect check or checkmate itself: predictions have to
  come from the network. Rule-based labeling only belongs in the dataset tooling.
- **One file per type**, like a `.hpp` + `.cpp` pair: a type, its methods, its trait implementations and its
  unit tests live in the same file. A file is only split into a folder when it gets genuinely big.
- **Encapsulation:** fields are private and read through getters. Constructors validate their input, so an
  invalid value cannot exist (for example, a `NetworkRequest` with a count of 0).
- **No duplication:** shared behavior goes into a trait or a shared crate (for example `cli_core::Cli`).
- **No empty placeholder files:** a module is created when it gets real code.

### Why the analyzer has a library part

`.cargo/config.toml` turns warnings into errors. In a binary, code that nothing calls yet triggers a
`dead_code` warning, which breaks the build. Public items of a library are never reported as unused, so code
written ahead of its first caller (such as `Label`) lives in the analyzer's library part. It also makes that
code easier to unit-test.

## Error handling

No code calls `process::exit`. Every error travels back to `main` as a `Result<_, String>`, and only `cli_core::run`
prints it and chooses the exit code:

```
cli_core::run
  ├── parse arguments ──── error ───────────────┐
  ├── --help ── print help on stdout → exit 0   │
  └── program's closure ── Err(message) ────────┤
                │                               ▼
                Ok → exit 0          message on stderr → exit 84
```

Error messages start with `error:`, like clap's own messages. A `panic` is only acceptable for a programming
bug that valid or invalid input can never trigger (for example `Label::index` on a label missing from
`Label::ALL`).

## Status

| Part | State |
|---|---|
| Workspace, Makefile, vendoring, pre-commit hooks | done |
| Command line of both programs | done, see [cli.md](cli.md) |
| End-to-end tests in CI | done, see [testing.md](testing.md) |
| Labels | done, see [labels.md](labels.md) |
| CHESSFILE reader | done, see [chess-file.md](chess-file.md) |
| Dataset | not in git, see [dataset.md](dataset.md) |
| Neural network (`my_torch`) | `Matrix`, starting weights and activations done, see [my_torch.md](my_torch.md); layers and training not started |
| `--train`, `--predict` | read the CHESSFILE, then return a "not implemented" error (exit 84) |
| Network generation | not implemented: returns an error and exits with 84 |
