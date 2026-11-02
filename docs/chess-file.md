# CHESSFILE

A CHESSFILE is the text file of chessboards given to `my_torch_analyzer`, in both modes. The code is in
`crates/my_torch_analyzer/src/`:

| File | Type | Role |
|---|---|---|
| `chess_file.rs` | `ChessFile` | the whole file: reading, blank lines, line numbers, required labels |
| `sample.rs` | `Sample` | one line: a `Board` and an optional `Label` |
| `board.rs` | `Board`, `Piece`, `PieceKind` | one position, built from a FEN with `chess_fen_parser` |
| `color.rs` | `Color` | White / Black, shared by `Board` and `Label` |

## Format

One chessboard per line: a FEN (6 fields), then optionally its expected output.

```
rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3 Checkmate Black
rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1 Nothing
rnbqkbnr/pppppppp/8/8/3P4/8/PPP1PPPP/RNBQKBNR b KQkq d3 0 1
```

| Rule | Behavior |
|---|---|
| Fields are separated by whitespace | any number of spaces or tabs; Windows line endings (`\r\n`) are accepted |
| The first 6 words are the FEN | they are rejoined with single spaces and parsed by `chess_fen_parser` |
| The remaining words are the label | `Check White` is two words; see [labels.md](labels.md) for the accepted labels |
| Blank lines | skipped, but still counted for line numbers in error messages |
| `--train` | every line needs a label |
| `--predict` | labels are optional |
| A file without any chessboard | error, in both modes |

## Errors

Every error is one line on stderr, points at the file and line, and exits with 84:

| Cause | Message |
|---|---|
| File missing or unreadable | `error: cannot read 'boards.txt': No such file or directory (os error 2)` |
| No chessboard in the file | `error: boards.txt: no chessboards` |
| Fewer than 6 words | `error: boards.txt:7: expected a FEN with 6 fields, found 4` |
| Wrong number of squares in a rank | `error: boards.txt:3: invalid FEN 'rnbqkbnrr/…': invalid rank length: 9` |
| Any other malformed FEN | `error: boards.txt:3: invalid FEN '8/8/8/8/8/8/8/8 x - - 0 1'` |
| Move counter too large | `error: boards.txt:3: invalid FEN '…': invalid number: number too large to fit in target type` |
| Unknown label | `error: boards.txt:12: unknown label 'Chek'` |
| Label missing with `--train` | `error: boards.txt:5: missing label, required for training` |

The first error stops the reading. For grammar errors, `chess_fen_parser` produces a multi-line message with
a column marker; it is replaced by the FEN text, which is enough to spot the mistake and keeps the message on
one line.

## The board

`Board` stores only what the network uses:

- `squares()`: 64 squares from `a1` (index 0) to `h8` (index 63), so square `file + 8 × (rank − 1)`, each
  `None` or a `Piece` with a `kind()` and a `color()`;
- `side_to_move()`.

Castling rights, the en passant square and the move counters are validated by the parser, then dropped.

**Why not keep `chess_fen_parser`'s `FenRecord`:** one `FenRecord` takes about 480 bytes spread over 10
memory allocations, against about 130 bytes and none for a `Board`. For a training file of a million
chessboards, that is roughly 500 MB against 130 MB. Its types can also not be copied or compared. Only
`board.rs` uses `chess_fen_parser`, so changing the parser again would only touch that file.

## Reading in the modes

`--train` and `--predict` read the CHESSFILE before anything else (`Labels::Required` and `Labels::Optional`),
so a bad file is reported even though the network part is not implemented yet.

`ChessFile::read` only reads the file from disk; `ChessFile::parse` does the rest on the text. The unit tests
use `parse` directly, so they need no files. The end-to-end tests use the files in
`crates/my_torch_analyzer/tests/fixtures/`.
