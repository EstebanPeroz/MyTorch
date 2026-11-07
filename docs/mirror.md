# Mirrored chessboards

```sh
./my_torch_analyzer --train --mirror LOADFILE CHESSFILE
```

`--mirror` adds to the training the **mirror** of every chessboard. It is optional, only works with `--train`,
and is not in the help text (like [`--dashboard`](dashboard.md)). Without it, nothing changes.

## What is a mirror

Chess does not care which color is which. Take a position and:

1. flip the ranks (rank 1 ↔ rank 8), because pawns move in opposite directions;
2. give every piece the other color;
3. give the move to the other side.

The result is a legal position, and its label is the original label with the attacker's color swapped:

| Chessboard | Mirror |
|---|---|
| `rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black` | `rnbqkbnr/ppppp2p/5p2/6pQ/4P3/8/PPPP1PPP/RNB1KBNR b KQkq - 1 3 Checkmate White` |

`Nothing` and `Stalemate` stay the same. Only the pieces and the side to move are mirrored: the network does not
read the castling rights, the en passant square or the counters (see [labels.md](labels.md)).

## Why

- The dataset has more white than black attackers (`Checkmate White` 13%, `Checkmate Black` 11%). The network
  can learn "pieces near this corner mean mate" for one color only. Mirroring makes both colors equally
  represented.
- It doubles the data for free, and teaches the network the symmetry instead of making it learn each color
  separately. That reduces overfitting. It does not replace the other levers (a smaller network, early
  stopping, more varied positions).

## Duplicates

Duplicates are removed first (as always), then the mirrors are added. A mirror is skipped when the same
chessboard is already in the file, so there are still no duplicates. The `training on N chessboards` line counts
the file; a second line, `training on M more chessboards, the mirror of each one`, counts the mirrors.

## Validation

With early stopping, the trainer sets part of the data aside to measure the network on chessboards it has not
learned. A chessboard and its mirror are nearly the same input for the network, so if one is trained on and the
other measured, the validation loss is flattered and early stopping stops too late.

To prevent it, the analyzer gives `my_torch` a group number per row (`Dataset::with_groups`): a chessboard and
its mirror share a group, and the validation split keeps each group on one side. The validation set can then be a
little larger than its share (it takes whole groups). Without `--mirror`, the split is the same as before.

## Limits

- The accuracy printed at the end of the training counts the mirrors too, and it is measured on the chessboards
  the network was trained on, as without `--mirror`. Compare runs on the same separate file with
  `--predict`.
- It only adds positions that are symmetric to existing ones. It does not add stalemates, which the dataset does
  not have.
