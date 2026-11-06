# Testing

```sh
cargo test --workspace
```

CI runs the same command on every push and pull request to `main` (`.github/workflows/checks.yml`, step
*Unit and end-to-end tests*), after the style hooks and `make`.

## Two kinds of tests

| Kind | Where | What it checks |
|---|---|---|
| Unit tests | `#[cfg(test)] mod tests` at the bottom of the file they test | one type or function in isolation: parsed values, conversions, validation |
| End-to-end tests | `crates/<program>/tests/` | the real binary, as a user runs it: exit code, stdout and stderr together |

Each rule is tested in detail in one place only:

- **Command-line arguments** are tested end to end (exit code 84 and the exact message). The values a valid
  command line produces (paths, default save file, network counts) are unit-tested, because they are not
  visible from outside the program.
- **CHESSFILE rules** (every malformed FEN, whitespace, blank lines, byte order mark, invalid UTF-8…) are
  unit-tested in `board.rs`, `sample.rs` and `chess_file.rs`. The end-to-end tests only check one case per
  kind of error, to prove that each mode reads the file with the right rules (labels required with
  `--train`, optional with `--predict`) and that its errors exit with 84.

Unit tests share their FEN strings through the test-only module `crates/my_torch_analyzer/src/test_fens.rs`.

## End-to-end tests

The dev-only crate `cli_test` runs a binary and checks its result. It is a `dev-dependency`, so it is never
compiled into the programs.

```rust
use cli_test::Program;

const HELP: &str = include_str!("../src/help.txt");

fn analyzer() -> Program {
    Program::new(env!("CARGO_BIN_EXE_my_torch_analyzer"))
}

#[test]
fn invalid_arguments_exit_with_84() {
    analyzer().assert_errors(&[
        (&["--predict", "net.nn"], "the following required arguments were not provided"),
    ]);
}
```

| Helper | Checks |
|---|---|
| `program.assert_help(help)` | `--help` and `-h` print exactly `help` on stdout, nothing on stderr, exit code 0 |
| `program.assert_errors(cases)` | for each `(args, message)`: exit code 84, nothing on stdout, stderr starts with `error: {message}` |
| `program.run(args)` | runs once and returns an `Outcome`, for checks the helpers above do not cover |
| `outcome.assert_success(stdout)` / `outcome.assert_error(message)` | the same checks on a single run |

When a check fails, the message shows the command, the exit code, stdout and stderr.

Input files for these tests live in `crates/<program>/tests/fixtures/` (for example a valid CHESSFILE, one
with an unknown label, an empty one). Cargo runs the tests from the crate's folder, so the tests refer to
them as `tests/fixtures/…`.

The help text lives in `src/help.txt` and is read with `include_str!` by both the program and its test, so it
exists only once.

The tests check the error **message**, not only the exit code: valid commands currently also exit with 84
(`… is not implemented yet`), so checking 84 alone would not tell a parsing error from a valid command.
