# Bonus

The subject asks for the bonus files to live in a `bonus` directory. In this project the bonus features are
**integrated into the main code** instead of being copied here: each one is a small optional extra of the same
binaries (an option, a label or a config field), and they all share the parsing, encoding and training code.
Duplicating that code in `bonus/` would create two versions to keep in sync, and moving it out would make the
mandatory build depend on a second tree. So this directory is an index: it says what the bonuses are and where
to find them. There is no separate bonus Makefile, `make` at the root builds everything.

| Bonus | What it does | Where it lives | Documentation |
|---|---|---|---|
| Stalemate label | Predicts `Stalemate` as a sixth output, on top of check and checkmate | `crates/my_torch_analyzer/src/label.rs`, `encoding.rs` | [docs/labels.md](../docs/labels.md) |
| Training dashboard | `--dashboard` serves a live web page: loss and accuracy curves, confusion matrix, F1, model shape | `crates/my_torch_analyzer/src/dashboard.rs`, `dashboard.html` | [docs/dashboard.md](../docs/dashboard.md) |
| Mirror augmentation | `--mirror` also trains on each chessboard seen from the other side, kept out of the validation set | `crates/my_torch_analyzer/src/board.rs`, `modes.rs` | [docs/mirror.md](../docs/mirror.md) |
| Early stopping and validation split | Stops when the validation loss stops improving and restores the best epoch | `crates/my_torch/src/early_stopping.rs`, `trainer.rs` | [docs/training-metrics.md](../docs/training-metrics.md) |
| Dropout | Optional per-layer dropout set in the configuration file | `crates/my_torch/src/dense.rs`, `model_config.rs`, `configs/chess.conf` | [docs/my_torch.md](../docs/my_torch.md) |
| Training metrics | Per-epoch history, per-label precision, recall and F1, majority baseline | `crates/my_torch/src/history.rs`, `epoch_report.rs`, `crates/my_torch_analyzer/src/confusion.rs` | [docs/training-metrics.md](../docs/training-metrics.md) |
| `make train` | Generates a blank network and trains it with `--mirror` and `--dashboard` | `Makefile` | [README.md](../README.md) |

None of these options appear in the fixed help text, and without them the programs behave as the subject
describes.
