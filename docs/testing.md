# Testing

```sh
cargo test --workspace
```

CI runs the same command on every push to `main` and on every pull request, whatever branch it targets
(`.github/workflows/checks.yml`, step *Unit and end-to-end tests*), after the style hooks and `make`.

## Three kinds of tests

| Kind | Where | What it checks |
|---|---|---|
| Unit tests | `#[cfg(test)] mod tests` at the bottom of the file they test | one type or function in isolation: parsed values, conversions, validation |
| End-to-end tests | `crates/<program>/tests/` | the real binary, as a user runs it: exit code, stdout and stderr together |
| Library tests | `crates/my_torch/tests/` | the library through its public API only, as another crate uses it |

Each rule is tested in detail in one place only:

- **Command-line arguments** are tested end to end (exit code 84 and the exact message). The values a valid
  command line produces (paths, default save file, network counts) are unit-tested, because they are not
  visible from outside the program.
- **CHESSFILE rules** (every malformed FEN, whitespace, blank lines, byte order mark, invalid UTF-8…) are
  unit-tested in `board.rs`, `sample.rs` and `chess_file.rs`. The end-to-end tests only check one case per
  kind of error, to prove that each mode reads the file with the right rules (labels required with
  `--train`, optional with `--predict`) and that its errors exit with 84.
- **Neural network math** (`my_torch`) is unit-tested in the file of each type: every formula against values
  computed by hand, and every derivative and gradient against measured slopes. To measure a slope, nudge a value
  by `±ε`, compute the result twice, and divide the difference by `2ε`; a wrong formula in `Activation`, in
  `Dense::backward` or in a loss makes the two disagree.
  Each family shares its checks with its members: `assert_gradient_matches_finite_differences` in
  `losses/loss.rs`, `assert_slope_matches_finite_differences` in `activations/activation.rs` and `assert_spread`
  in `initializations/initialization.rs`, so a new member is tested with one line.
  `network.rs` runs the same check on whole networks: every weight and bias of two 3-layer networks, one per
  loss, so a mistake in how backpropagation chains the layers fails too.
- **Learning** is tested end to end in `crates/my_torch/tests/xor.rs`: a `2 → 8 → output` network must learn
  XOR, once with one sigmoid output and mean squared error, once as two classes with softmax cross-entropy.
  Each test checks the final loss and every prediction. XOR cannot be learned by a single layer, so passing
  proves that the layers, backpropagation, the optimizer and the training loop work together. With 4 hidden
  neurons, one seed in 20 got stuck in a poor solution; with 8, all 20 seeds learn it, so the test does not
  depend on a lucky seed. A third test saves the trained network to a file, loads it back, and checks that it
  gives the same predictions and can keep training.
- **Early stopping** is unit-tested in `early_stopping.rs` (the random split, and when to stop on given
  sequences of losses) and in `trainer.rs`: on random-noise data, which can only be learned by heart, the
  validation loss rises, training stops `patience` epochs after the best one, and the network that comes back
  gives exactly the best validation loss.
- **Configurations** are unit-tested in `model_config.rs`: the seed of each generated network, the training
  section written into its network file, and one broken configuration per kind of error.
- **Network files** are unit-tested in `model.rs`: the exact JSON written for a small network, a save and load
  that keeps every weight to the bit (including `f32::MAX` and the smallest numbers), the refusal to save NaN,
  one broken file per kind of error, each derived from the valid file by changing one field, and a save that
  fails halfway (a folder blocks its temporary file) without touching the previous file.

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
