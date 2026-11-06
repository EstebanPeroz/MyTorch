# Dataset

The training data is not in git (`/dataset/` is in `.gitignore`): it is large and never changes. To use it,
put it in `dataset/` at the repository root.

## Layout

The positions are sorted by expected output, then by number of pieces on the board:

| File | Lines | Pieces (min / average / max) |
|---|---|---|
| `dataset/nothing/10_pieces.txt` | 100,000 | 2 / 6.5 / 9 |
| `dataset/nothing/20_pieces.txt` | 100,000 | 10 / 14.5 / 19 |
| `dataset/nothing/many_pieces.txt` | 100,000 | 20 / 23.6 / 32 |
| `dataset/check/10_pieces.txt` | 75,198 | 3 / 6.9 / 9 |
| `dataset/check/20_pieces.txt` | 100,000 | 10 / 14.7 / 19 |
| `dataset/check/many_pieces.txt` | 74,102 | 20 / 22.2 / 32 |
| `dataset/checkmate/10_pieces.txt` | 45,087 | 3 / 6.3 / 9 |
| `dataset/checkmate/20_pieces.txt` | 100,000 | 10 / 14.7 / 19 |
| `dataset/checkmate/many_pieces.txt` | 28,632 | 20 / 22.4 / 32 |
| **Total** | **723,019** | |

About 42 MB in total. The split by piece count allows measuring accuracy separately on simple and crowded
positions.

## Format

Every line is already a CHESSFILE line, a FEN followed by its label (see [chess-file.md](chess-file.md)):

```
8/8/R2k4/4r1p1/8/5K2/5P2/8 b - - 7 59 Check White
r4rk1/2p2p1p/3p2p1/ppn2n2/8/2b2NB1/1qPKQPPP/3R1B1R w - - 2 22 Checkmate Black
2k5/p1r4p/2PK2p1/pB3bP1/8/1R6/7P/8 b - - 3 54 Nothing
```

Our reader accepts all 723,019 lines with labels required, as `--train` does. Reading takes about half a
second per 100,000 lines. To check it again:

```sh
for file in dataset/*/*.txt; do ./my_torch_analyzer --train x.nn "$file"; done
```

Each file is valid when the command stops at `error: --train is not implemented yet` instead of pointing at a
line.

## Labels

| Label | Lines | Share |
|---|---|---|
| `Nothing` | 300,000 | 41% |
| `Check White` | 135,587 | 19% |
| `Check Black` | 113,713 | 16% |
| `Checkmate White` | 94,295 | 13% |
| `Checkmate Black` | 79,424 | 11% |
| `Stalemate` | 0 | 0% |

- **The color is always the attacking side**, as described in [labels.md](labels.md): on all 423,019 check and
  checkmate lines, the label's color is the side that is *not* to move.
- Every label matches the folder its file is in.
- **There is no stalemate example.**

## Quality

| Check | Result |
|---|---|
| Lines our reader rejects | none |
| Positions with contradictory labels (same board and side to move, two labels) | none |
| Duplicates (same board and side to move) | 7,464 extra copies of 3,180 positions (about 1%), always inside the same file |

## Open decisions

These are to settle together with the training design:

- **Duplicates must be removed before splitting** the data into training, validation and test sets. Otherwise
  copies of the same position end up on both sides of the split: the network is then evaluated on positions it
  was trained on, which inflates the measured accuracy and hides overfitting.
- **How to split:** the split should keep the proportion of each label and of each piece-count file, and use a
  fixed random seed so that a network can be trained again the same way.
- **Stalemate:** without examples, the network cannot learn it. Either keep the class and add stalemate
  positions later, or drop it and use 5 outputs instead of 6.
- **Class imbalance** (41% / 35% / 24%) is mild, but benchmarks should report accuracy per label, not only
  overall.
