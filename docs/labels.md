# Labels

A **label** is the expected answer for a chessboard: the state of the game the network has to predict. The
network is trained with supervised learning, so every training example is a chessboard together with its label.

The code is in `crates/my_torch_analyzer/src/label.rs`.

## Classes

| Index | Label | Meaning |
|---|---|---|
| 0 | `Nothing` | nothing in particular |
| 1 | `Check White` | White is giving check |
| 2 | `Check Black` | Black is giving check |
| 3 | `Checkmate White` | White has checkmated Black |
| 4 | `Checkmate Black` | Black has checkmated White |
| 5 | `Stalemate` | the side to move has no legal move and is not in check (bonus) |

The index is the label's position in `Label::ALL`. It is the network's output neuron for that label, so the
order must never change once networks have been trained.

## Which color is written

The color is the **attacking side**, not the side to move. The subject's examples show it:

| Board | Side to move | Subject's output |
|---|---|---|
| `rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3` (Fool's mate) | White | `Checkmate Black` |
| `rnbqkbnr/pppp2pp/8/4pp1Q/3P4/4P3/PPP2PPP/RNB1KBNR b KQkq - 1 3` (White queen on h5) | Black | `Check White` |

## Text format

Labels are read and written exactly as in the table: one capitalized word for the state, then a space and the
color when there is one. Parsing is strict: `Check` without a color, `check White`, `Checkmate Red`,
`Nothing else` and ` Nothing` are all rejected with `unknown label '…'`.

## API

```rust
pub enum Color { White, Black }
pub enum Label { Nothing, Check(Color), Checkmate(Color), Stalemate }
```

| Item | Role |
|---|---|
| `Label::ALL` | every label, in class order |
| `Label::COUNT` | number of classes (6), the size of the network's output layer |
| `label.index()` | label → class index |
| `Label::from_index(i)` | class index → label, `None` if `i` is out of range |
| `label.to_string()` / `{label}` | label → text (`Display`) |
| `"Check White".parse::<Label>()` | text → label (`FromStr`), `Err(String)` if unknown |

During training, a label from the CHESSFILE becomes a class index, which becomes the expected output vector.
During prediction, the index of the network's strongest output becomes a label, which is printed.

## Invariant

`Label::ALL` must list every variant of `Label`. Adding a variant without adding it to `ALL` makes `index()`
panic. `ALL` sits right under the enum in the same file to make this hard to miss, and the tests check that
every label survives the text and index round trips.
