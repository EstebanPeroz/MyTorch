# The `my_torch` library

`my_torch` is the neural network, written from scratch. It uses `ndarray` for matrix arithmetic and `rand` for
random numbers (see [dependencies.md](dependencies.md)). It knows nothing about chess: the analyzer turns
chessboards into numbers and labels into target vectors, and `my_torch` only sees matrices.

What exists today: the `Matrix` type, the starting weights (`Initialization`), the activation functions
(`Activation`) and the dense layer (`Dense`). Losses, the network and training come next.

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

A layer with `in` inputs and `out` neurons holds a weight matrix `W` of `in × out` (column `j` holds the
weights of neuron `j`). Then `X · W` is `n × out`: still one row per sample, now one column per neuron. The
network's final output will therefore have one row per chessboard, in the order of the CHESSFILE, which is
what `--predict` must print.

## `Matrix`

`Matrix` wraps an `ndarray::Array2<f32>`, and only `matrix.rs` uses `ndarray`: the rest of the library sees our
own type. Values are stored row after row (row-major), so a sample's values sit next to each other in memory and
copying a few rows into a mini-batch is cheap.

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

## Starting weights

A layer's weights cannot all start at the same value: every neuron of the layer would compute the same thing,
receive the same gradient, and stay identical forever. So they start random, drawn from a normal distribution
(the bell curve) centered on 0.

How wide that bell curve is matters. Each neuron adds up `in` weighted inputs, so with weights that are too
large the sums grow from layer to layer and the activations saturate; too small, and they shrink towards 0. Both
stall learning. `Initialization` picks a standard deviation from the layer's size that keeps the values at the
same scale through the layers:

| Method | Standard deviation | For |
|---|---|---|
| `Initialization::Xavier` | `√(2 / (in + out))` | `Sigmoid`, `Tanh` |
| `Initialization::He` | `√(2 / in)` | `ReLU` |

`init.weights(in, out, &mut rng)` returns an `in × out` matrix of such weights. The generator is a
`rand::rngs::StdRng` created with `StdRng::seed_from_u64(seed)`: the same seed always gives the same weights.
It panics on a layer with no inputs or no outputs.

## Activations

A layer only multiplies and adds (`X · W + b`), so two layers in a row are still one multiplication and one
addition: they would collapse into a single layer, which can only separate classes with a straight line. An
activation function, applied to every value after the layer, bends the result so that stacked layers can learn
curved boundaries (XOR, for example, cannot be split by a straight line).

| `Activation` | Value | Slope (derivative) | Used for |
|---|---|---|---|
| `Identity` | `x` | `1` | the output layer, when the loss shapes the output (softmax for classes) |
| `Sigmoid` | `1 / (1 + e⁻ˣ)`, between 0 and 1 | `s × (1 − s)`, at most 0.25 | an output read as a probability |
| `Tanh` | `tanh(x)`, between −1 and 1 | `1 − tanh²(x)`, at most 1 | hidden layers of small networks |
| `Relu` | `max(0, x)` | `1` if `x > 0`, else `0` | hidden layers |
| `LeakyRelu` | `x` if `x > 0`, else `0.01 x` | `1` if `x > 0`, else `0.01` | hidden layers, without dead neurons |

`activation.apply(&z)` returns `f(z)` and `activation.derivative(&z)` returns `f'(z)`, value by value.
Backpropagation multiplies the gradient by that slope at every layer, which is why the choice matters:

- `Sigmoid` and `Tanh` flatten out for large `|x|` (their slope gets close to 0), and the sigmoid's slope is at
  most 0.25, so gradients shrink layer after layer.
- `Relu` lets gradients through unchanged where `x > 0`, but a neuron whose `x` is always negative gets a slope
  of 0 and stops learning. `LeakyRelu` keeps a small slope there.
- At exactly `x = 0`, `Relu` and `LeakyRelu` have no slope; they use the one of the negative side.

Pair each activation with its initialization (see [Starting weights](#starting-weights)): `He` for `Relu` and
`LeakyRelu`, `Xavier` for `Sigmoid` and `Tanh`.

## Dense layer

`Dense` is a fully connected layer: every neuron reads every input. It holds its weights `W` (`in × out`), its
biases `b` (one row of `out` values, one per neuron) and its activation `f`. Neuron `j` is column `j` of `W`
and value `j` of `b`: storing the layer as one matrix lets a single product compute every neuron for every
sample of the batch.

The name comes from Keras, whose `Dense` layer also holds its activation. PyTorch calls the same layer
`nn.Linear`, but without the activation, which is a separate layer there (`nn.ReLU`…).

| Method | Result |
|---|---|
| `Dense::new(weights, biases, activation)` | a layer from existing values; refuses biases that are not `1 × out` |
| `Dense::random(in, out, activation, initialization, &mut rng)` | a new layer: random weights, biases at 0 |
| `input_size()`, `output_size()`, `weights()`, `biases()`, `activation()` | reading the layer |
| `forward(&x)` | the layer's output, for predictions |
| `forward_cached(&x)` | the same output, plus a `DenseCache` that `backward` needs |
| `backward(&cache, &da)` | the gradients of the layer (`DenseGradients`) and the gradient for the previous layer |
| `update(&gradients, rule)` | every weight and bias replaced by `rule(parameter, gradient)` |

Biases start at 0: the random weights already make the neurons different, so the biases do not need to.

### Forward

With an input `X` of `n × in` (one sample per row):

| Step | Formula | Shape |
|---|---|---|
| weighted sum | `Z = X · W + b` (`b` is added to every row) | `n × out` |
| output | `A = f(Z)`, value by value | `n × out` |

### Backward

Backpropagation goes through the layers in reverse. Each layer receives `dA`, how the loss changes with each of
its outputs, and computes:

| Gradient | Formula | Shape |
|---|---|---|
| weighted sum | `dZ = dA ⊙ f'(Z)` (`⊙`: value by value) | `n × out` |
| weights | `dW = Xᵀ · dZ` | `in × out` |
| biases | `db` = sum of the rows of `dZ` | `1 × out` |
| input | `dX = dZ · Wᵀ`, which becomes `dA` for the previous layer | `n × in` |

`Xᵀ · dZ` and the row sum add up the contributions of the `n` samples, so one matrix product handles the whole
batch. `backward` needs `X` and `Z` from the forward pass: `forward_cached` keeps them in a `DenseCache`.
`forward` skips that copy, which predictions do not need.

A layer never changes while it computes: the weights only change in `update`, after the whole backward pass.
`update` takes the rule as a function so that the optimizer decides how to step (gradient descent will use
`parameter − learning_rate × gradient`).

The tests check `backward` against measured slopes: nudge each weight, bias and input by `±0.01`, compute the
change in the output, and compare with what `backward` returned.

## Errors

`from_vec` and `from_rows` refuse a matrix with no rows or no columns, rows of different lengths, and a number
of values that does not match the shape. `Dense::new` refuses biases that are not one row of one value per
neuron. They return `Err(String)` without the `error:` prefix, like `Label`: the program that calls the library
adds context and the prefix.

The operations **panic** on mismatched shapes (for example `matmul` of a `2 × 3` by a `2 × 3`) and on indices
outside the matrix. `Dense` panics on an input with the wrong number of columns and on a gradient of the wrong
shape in `backward`. The types built on these check their sizes when they are created, so such a panic would be
a bug in `my_torch`, not a user error.
