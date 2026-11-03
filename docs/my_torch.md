# The `my_torch` library

`my_torch` is the neural network, written from scratch with no dependencies. It knows nothing about chess: the
analyzer turns chessboards into numbers and labels into target vectors, and `my_torch` only sees matrices.

What exists today: the `Matrix` type. Layers, activations, losses and training come next.

## Numbers are `f32`

Every number is a `Scalar`, an alias for `f32`. It is twice as compact as `f64` and faster, which matters for
723,019 chessboards. The cost is precision (about 7 significant digits instead of 16). Switching to `f64` is a
one-line change in `matrix.rs`.

## One sample per row

A batch of `n` samples with `features` values each is an `n × features` matrix: row `i` is sample `i`. For
XOR, `n = 4` and `features = 2`. For a chessboard, `features` is the size of the board encoding.

```
        x1  x2
      [  0   0 ]   ← sample 1
X =   [  0   1 ]   ← sample 2        4 × 2   (n = 4, features = 2)
      [  1   0 ]   ← sample 3
      [  1   1 ]   ← sample 4
```

A layer with `in` inputs and `out` neurons will hold a weight matrix `W` of `in × out` (column `j` holds the
weights of neuron `j`). Then `X · W` is `n × out`: still one row per sample, now one column per neuron. The
network's final output will therefore have one row per chessboard, in the order of the CHESSFILE, which is
what `--predict` must print.

## `Matrix`

Values are stored row after row in one `Vec<Scalar>` (row-major), so a sample's values sit next to each other
in memory and copying a few rows into a mini-batch is cheap.

| Method | Result |
|---|---|
| `Matrix::zeros(rows, cols)` | a matrix of zeros |
| `Matrix::from_vec(rows, cols, data)` | a matrix from its values, row after row |
| `Matrix::from_rows(&[[1.0, 2.0], [3.0, 4.0]])` | a matrix from its rows |
| `rows()`, `cols()`, `data()`, `row(i)`, `get(i, j)`, `set(i, j, value)` | reading and changing values |
| `a.matmul(&b)` | `A · B` |
| `a.transpose_matmul(&b)` | `Aᵀ · B`, without building `Aᵀ` |
| `a.matmul_transpose(&b)` | `A · Bᵀ`, without building `Bᵀ` |
| `a.add_row(&b)` | `B` (one row) added to every row of `A`: how biases are added |
| `a.sum_rows()` | one row: the sum of all rows of `A` |
| `a.map(f)`, `a.zip_map(&b, f)` | `f` applied value by value |
| `a.select_rows(&[2, 0])` | a new matrix with those rows, in that order: how mini-batches are built |
| `a.argmax_rows()` | for each row, the column of its largest value (the first one on ties): the predicted class |

The two transposed products are what backpropagation needs (`Xᵀ · dZ` for the weight gradients, `dZ · Wᵀ` for
the gradient passed to the previous layer).

## Errors

`from_vec` and `from_rows` refuse a matrix with no rows or no columns, rows of different lengths, and a number
of values that does not match the shape. They return `Err(String)` without the `error:` prefix, like `Label`:
the program that calls the library adds context and the prefix.

The operations **panic** on mismatched shapes (for example `matmul` of a `2 × 3` by a `2 × 3`) and on indices
outside the matrix. The types built on `Matrix` will check their sizes when they are created, so such a panic
would be a bug in `my_torch`, not a user error.
