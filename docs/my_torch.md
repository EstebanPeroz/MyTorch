# The `my_torch` library

`my_torch` is the neural network, written from scratch. It uses `ndarray` for matrix arithmetic, `rand` for
random numbers and `serde_json` for network files (see [dependencies.md](dependencies.md)). It knows nothing about chess: the analyzer turns
chessboards into numbers and labels into target vectors, and `my_torch` only sees matrices.

What exists today: the `Matrix` type, the random number generator (`Rng`), the starting weights
(`Initialization`), the activation functions
(`Activation`), the dense layer (`Dense`), the losses (`Loss`), the network (`Network`) and its plan
(`Architecture`), the dataset
(`Dataset`), the optimizer (`Optimizer`), the training loop (`Trainer`) and the model (`Model`): a network with
its training settings, saved to and loaded from a file.

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

## Random numbers

Training draws random numbers twice: for the starting weights, and to shuffle the samples before each epoch.
Both come from an `Rng`, created from a seed with `Rng::new(42)`, and passed as `&mut rng` to `Dense::random`,
`Initialization::weights` and `Trainer::fit`. The same seed always gives the same numbers, so the same seed,
data and settings train exactly the same network.

`Rng` holds a `rand` generator in a private field. `rand` never appears in the library's API, so programs that
use `my_torch` do not depend on it. Only code inside `my_torch` can draw numbers from an `Rng`.

## Starting weights

A layer's weights cannot all start at the same value: every neuron of the layer would compute the same thing,
receive the same gradient, and stay identical forever. So they start random, drawn from a normal distribution
(the bell curve) centered on 0.

How wide that bell curve is matters. Each neuron adds up `in` weighted inputs, so with weights that are too
large the sums grow from layer to layer and the activations saturate; too small, and they shrink towards 0. Both
stall learning. Each initialization picks a standard deviation from the layer's size that keeps the values at
the same scale through the layers:

| Initialization | Standard deviation | For | PyTorch |
|---|---|---|---|
| `XavierNormal` | `√(2 / (in + out))` | `Sigmoid`, `Tanh` | `nn.init.xavier_normal_` |
| `HeNormal` | `√(2 / in)` | `Relu`, `LeakyRelu` | `nn.init.kaiming_normal_` |

`Initialization` is a trait: `name()` gives its name in files (see [Names](#names)), and
`weights(in, out, &mut rng)` returns an `in × out` matrix of
starting weights, for example `HeNormal.weights(768, 128, &mut rng)`, where `rng` is an `Rng` (see
[Random numbers](#random-numbers)): the same seed always gives the same weights.
It panics on a layer with no inputs or no outputs.

Each initialization is a struct in its own file under `initializations/`. Both current ones draw from a bell
curve, so they share `normal_weights` from `initializations/initialization.rs` and only give their standard
deviation. Adding one, such as a uniform version (`xavier_uniform_`), means adding one file (a struct, its
`impl Initialization` and its tests) and one line in the list of `initialization_from_name`. The names end in `Normal`, as in Keras, because each also exists in a
uniform version.

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

Pair each activation with its initialization (see [Starting weights](#starting-weights)): `HeNormal` for `Relu`
and `LeakyRelu`, `XavierNormal` for `Sigmoid` and `Tanh`.

### How activations are organized

Each activation's math (its value and its slope for one number) is a struct in its own file under
`activations/`, implementing the internal `Function` trait. A layer does not store that struct: it stores an
`Activation`, a small enum with one variant per activation. The enum can be copied, compared and saved by
name in a network file, which a trait object (`Box<dyn …>`) could not do as easily. Its one `match` only
routes each variant to its struct.

Adding an activation means adding one file (its struct, its `impl Function` with its name, and its tests, which
reuse the shared measured-slope check from `activations/activation.rs`), one variant in the enum and in
`Activation::ALL`, and one line in the `match`. The compiler points to the `match` if that line is missing.

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
| `Dense::random(in, out, activation, &initialization, &mut rng)` | a new layer: random weights, biases at 0 |
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

## Losses

A loss turns the network's outputs into one number that says how wrong they are: 0 when they match the
targets, larger as they drift away. Training lowers it. Its gradient, how the loss changes with each output, is
the first `dA` of backpropagation: the last layer's `backward` starts from it.

| Loss | Value | Gradient (`dA` for the last layer) | For |
|---|---|---|---|
| `MeanSquaredError` | mean of `(output − target)²` over every value | `2 (output − target) / (n × outputs)` | outputs that are numbers to match (XOR) |
| `SoftmaxCrossEntropy` | mean over samples of `−Σ target × log softmax(output)` | `(softmax(output) × Σ target − target) / n` | picking one class out of several |

`Loss` is a trait: `value(&outputs, &targets)` returns the number, and
`gradient(&outputs, &targets)` returns the gradient, a matrix shaped like the outputs. Each loss is a struct
that implements it, in its own file under `losses/`: `MeanSquaredError.value(&outputs, &targets)`. Both
methods panic if the targets are not shaped like the outputs; each loss starts with the shared
`check_shapes`.

Adding a loss means adding one file (a struct with fields for its settings, such as Huber's `delta`, its
`impl Loss` with its name, and its tests, which reuse the shared measured-slope check from `losses/loss.rs`) and
one line in the list of `loss_from_name`. Activations stay an enum while losses are a trait because every layer
stores an activation: an enum is a small value that can be copied and compared, while a loss lives once, in the
trainer.

### Softmax cross-entropy

For classes, the target of a sample is one-hot: a `1` in the column of its class, `0` elsewhere (`[0, 1, 0]`
for the second of three classes).

- **Softmax** turns a row of raw scores into probabilities that add up to 1: `softmax(x)ᵢ = eˣⁱ / Σⱼ eˣʲ`. The
  largest score gets the largest probability.
- **Cross-entropy** is then `−log(probability of the right class)`: close to 0 when the network gives the right
  class a probability near 1, and large when it gives it almost none.
- **The gradient** of both together is simply `(softmax − target) / n` for one-hot targets: push the right
  class's probability up, and every other one down.

The loss applies softmax itself, so the network's last layer must be `Identity`: it outputs raw scores. A
`Sigmoid` or `Relu` last layer would squash or cut the scores before softmax sees them. The largest raw score
is also the predicted class, so predictions do not need softmax at all. The `Loss` trait has a third method
for this, `needs_raw_scores()`: `false` by default, `true` for `SoftmaxCrossEntropy`. `Trainer::fit` refuses
to train when the loss needs raw scores and the last layer is not `Identity`.

Softmax subtracts each row's largest score before `exp`. It gives the same probabilities, but `e¹⁰⁰⁰` would
overflow `f32` while `e⁰` does not: huge scores (±1000) give finite values.

PyTorch names the same two losses `nn.MSELoss` and `nn.CrossEntropyLoss` (which also applies softmax itself).

## Network

`Network` holds layers in order, like PyTorch's `nn.Sequential`: the output of each layer is the input of the
next one.

| Method | Result |
|---|---|
| `Network::new(layers)` | a network from layers built with `Dense::random` or `Dense::new`; refuses an empty list and layers whose sizes do not chain |
| `input_size()`, `output_size()`, `layers()`, `output_layer()` | reading the network; `output_layer()` is the last layer |
| `predict(&x)` | the output of the last layer, one row per sample |
| `evaluate(&x, &targets, &loss)` | the loss on these samples, without computing gradients (Keras's `evaluate`) |
| `gradients(&x, &targets, &loss)` | the loss on this batch, and the gradients (`Gradients`) of every layer |
| `update(&gradients, rule)` | every layer's weights and biases replaced by `rule(parameter, gradient)` |

```rust
let mut rng = Rng::new(42);
let network = Network::new(vec![
    Dense::random(2, 4, Activation::Tanh, &XavierNormal, &mut rng),
    Dense::random(4, 1, Activation::Sigmoid, &XavierNormal, &mut rng),
])?;
```

Each layer is built with its own initialization, so it can match its activation. An
[`Architecture`](#architecture) builds the same network without repeating the sizes.

`gradients` runs backpropagation:

1. **Forward:** each layer's `forward_cached`, from the first to the last, keeping every `DenseCache`.
2. **Loss:** its value on the last layer's output, and its gradient, the first `dA`.
3. **Backward:** each layer's `backward`, from the last to the first. The `dX` a layer returns is the `dA` of
   the layer before it.

It returns the gradients without changing the network: the optimizer then decides how to apply them with
`update`.

`predict` and `gradients` return an error when the inputs do not have `input_size()` columns, when the targets
do not have `output_size()` columns, or when there are not as many targets as samples. These sizes come from
the user's files (a network file and a CHESSFILE), so a mismatch is a user error, not a bug.

### Gradient check

The network's tests check every gradient that backpropagation computes against a direct measurement: nudge one
weight or bias by `±0.01`, compute the loss twice, and take the slope `(loss(w + 0.01) − loss(w − 0.01)) / 0.02`.
They do this for every weight and bias of two `3 → 4 → 3 → 2` networks on a batch of 5 samples: a classifier
(`Tanh`, `Sigmoid`, `Identity`) with softmax cross-entropy, and one with sigmoid outputs and mean squared error.

## Architecture

An `Architecture` is the plan of a network, without its weights: the number of inputs, then for each layer its
number of neurons, its activation and its initialization. `build` turns the plan into a `Network` with random
starting weights.

```rust
let architecture = Architecture::new(2, vec![
    LayerSpec::new(8, Activation::Tanh, XavierNormal)?,
    LayerSpec::new(1, Activation::Sigmoid, XavierNormal)?,
])?;
let first = architecture.build(&mut Rng::new(1));
let second = architecture.build(&mut Rng::new(2));
```

| Method | Result |
|---|---|
| `LayerSpec::new(neurons, activation, initialization)` | one layer of the plan; refuses 0 neurons |
| `Architecture::new(inputs, layers)` | the plan; refuses 0 inputs and an empty list |
| `inputs()`, `layers()`, and `neurons()`, `activation()`, `initialization()` on each `LayerSpec` | reading the plan |
| `build(&mut rng)` | a new `Network`: each layer drawn with `Dense::random`, biases at 0 |

Each size is written once: a layer's inputs are the neurons of the layer before it, so the sizes always chain
and `build` cannot fail. The plan can be built many times: the two networks above have the same shape and
different weights, and the same seed always builds the same network. This is what the generator needs to
create several networks from one configuration file.

PyTorch's `nn.Sequential(nn.Linear(2, 8), nn.Tanh(), …)` describes a network and draws its weights at once.
Like Keras's `Sequential`, an `Architecture` only asks for each layer's number of neurons; it also keeps the
plan separate from the weights.

## Dataset

`Dataset` pairs inputs with their targets: row `i` of `inputs()` is sample `i`, and row `i` of `targets()` is
the answer the network should give for it. For XOR, the inputs are the four `[x1, x2]` rows and the targets are
`[0]`, `[1]`, `[1]`, `[0]`; for chess, the inputs will be encoded boards and the targets one-hot labels.

| Method | Result |
|---|---|
| `Dataset::new(inputs, targets)` | a dataset; refuses a different number of input and target rows |
| `inputs()`, `targets()`, `sample_count()` | reading the dataset |
| `select(&[2, 0])` | a smaller dataset with those samples, in that order: how mini-batches are built |

`select` picks the same rows from both matrices, so a sample never gets separated from its target.

## Optimizers

An optimizer decides how to change the weights once backpropagation has computed their gradients. `Optimizer`
is a trait: `step(&mut network, &gradients)` applies one update, and `name()` and `settings()` describe the
optimizer for files (see [Names](#names)). `step` takes `&mut self` because some optimizers remember things
between steps (Adam keeps a running average of past gradients). Each optimizer is a struct in its own file under
`optimizers/`, and has one line in the list of `optimizer_from_settings`.

| Optimizer | Step | PyTorch |
|---|---|---|
| `Sgd` | every weight and bias: `parameter − learning_rate × gradient` | `torch.optim.SGD` |

`Sgd::new(learning_rate)` refuses a learning rate that is not a positive number. The learning rate is the size
of each step: too small and training crawls, too large and the loss jumps around or blows up instead of going
down.

## Training

`Trainer` holds the training settings: a loss, an optimizer, a batch size and a number of epochs.
`Trainer::new(loss, optimizer, batch_size, epochs)` refuses a batch size or an epoch count of 0.
`trainer.fit(&mut network, &dataset, &mut rng)` trains the network in place and runs mini-batch gradient
descent:

```
for each epoch:
    shuffle the sample order
    for each group of batch_size samples:
        gradients: forward, loss, backpropagation
        one optimizer step on every weight and bias
    record the average loss of the epoch
```

- **An epoch** is one pass over every sample. With 723,019 samples and batches of 64, it makes about 11,300
  updates.
- **Mini-batches** are a compromise. A step on the whole dataset is precise but leaves one update per epoch. A
  step per sample is noisy and does not benefit from matrix products. A batch of a few dozen samples gives
  many fairly reliable steps, and its noise helps leave poor solutions. The last batch of an epoch is smaller
  when the batch size does not divide the number of samples.
- **Shuffling** before each epoch changes which samples go together. Without it, sorted data (all `Nothing`,
  then all `Check`…) would pull the weights one way, then the other.
- **Reproducibility:** every random choice (starting weights, shuffling) comes from an `Rng` created with an
  explicit seed. The same seed, data and settings train exactly the same network.

`fit` returns a `History`, like Keras: `history.training()` holds the average loss of each epoch, to follow
training or plot learning curves, and `history.epochs()` the number of epochs run. Each batch's loss is
measured just before its step, so the value of an epoch averages a network that keeps changing: it is usually
slightly higher than the loss of the network at the end of the epoch, as in Keras. Each batch counts in
proportion to its number of samples, so a smaller last batch weighs less than a full one. It returns an error
when the data does not fit the network (see [Network](#network)), or when the loss needs raw scores and the
last layer is not `Identity`.

The loss and the optimizer are stored as `Box<dyn Loss>` and `Box<dyn Optimizer>`: the trainer works with any
of them, and `fit` takes `&mut self` because an optimizer may update its own state at each step.

## Model

A `Model` is a network together with everything needed to train it: a `Trainer` (loss, optimizer, batch size,
epochs) and the seed used to shuffle the samples. It is what a network file holds, so that training a network
again needs nothing but its file. Keras keeps the loss and the optimizer with the network the same way.

| Method | Result |
|---|---|
| `Model::new(network, trainer, seed)` | a model; refuses a loss that needs raw scores on a last layer that is not `Identity` |
| `network()`, `trainer()`, `seed()` | reading the model; `trainer().loss()` and `trainer().optimizer()` give the loss and the optimizer |
| `train(&dataset)` | `trainer.fit` on the model's network, shuffled with `Rng::new(seed)`; returns its `History` |
| `save(path)`, `Model::load(path)` | writing and reading a network file (see [Network files](#network-files)) |
| `to_json()`, `Model::from_json(&text)` | the same, as a string |

```rust
let network = architecture.build(&mut Rng::new(1));
let trainer = Trainer::new(SoftmaxCrossEntropy, Sgd::new(0.01)?, 64, 10)?;
let mut model = Model::new(network, trainer, 42)?;
let history = model.train(&dataset)?;
let outputs = model.network().predict(&inputs)?;
```

Every call to `train` shuffles with the same seed, so training the same model on the same data twice gives the
same network.

### Network files

`save` writes a model as JSON, on one line followed by a newline. Spread out, a network with one `2 → 1` layer
looks like this:

```json
{
  "version": 1,
  "layers": [
    { "activation": "identity", "weights": [[0.5], [-1.0]], "biases": [0.25] }
  ],
  "training": {
    "loss": "mean_squared_error",
    "optimizer": { "name": "sgd", "learning_rate": 0.1 },
    "batch_size": 4,
    "epochs": 10,
    "seed": 7
  }
}
```

| Field | Content |
|---|---|
| `version` | the version of this format, 1. A file of another version is refused, so a future change of format cannot be misread. |
| `layers` | one object per layer, in order: its activation by name (see [Names](#names)), its weights one row per input (`in × out`), and its biases (one per neuron) |
| `training` | the loss by name, the optimizer by name with its settings, the batch size, the number of epochs and the seed of the shuffle |

Only what cannot be recomputed is stored. The sizes are not written: a layer's inputs and neurons are the shape
of its weights. The initialization is not written either: it only chose the starting weights, which the file
already holds.

- **Every weight comes back exactly.** `serde_json` writes each `f32` with the fewest digits that read back
  as the same `f32` (`0.1`, not `0.100000001`), so a saved and reloaded network gives the same predictions to
  the bit.
- **A network whose training diverged cannot be saved.** JSON has no NaN or infinity, so `save` refuses a
  weight or bias that is not a finite number instead of writing a file that could not be loaded. Loading
  refuses them too: `serde_json` reads a number too large for an `f32`, such as `1e39`, as infinity.
- **A failed save never loses the previous file.** `save` writes to `path.tmp` first, then renames it over
  `path`, which replaces the file in one step. If writing fails (full disk, no permission), the temporary file
  is removed and the previous file is untouched. This matters because training saves into the file it loaded
  when no other file is given.
- **Loading checks everything** by going through the same constructors as code: `Activation::from_name`,
  `Matrix::from_rows`, `Dense::new`, `Network::new`, the name lookups, `Trainer::new` and `Model::new`. A
  file can therefore not create a network that code could not, and each problem gives the usual message,
  prefixed with the layer it is about: `layer 2: a layer with 64 outputs needs 1 × 64 biases, got a 1 × 63
  matrix`. Fields that are missing, unknown or of the wrong type are reported by `serde_json` with their place
  in the file: ``invalid network file: missing field `epochs` at line 1 column 912``.
- **Size:** about 12 characters per weight or bias, so about 1.2 MB per 100,000. A `768 → 128 → 64 → 7`
  network (107,143 weights and biases) makes a 1.3 MB file, written in 6 ms and read in 8 ms (release build).

## Names

Files describe activations, initializations, losses and optimizers by name. Each one gives its own name, and
each family has one function that turns a name back into a value:

| Family | Names | From a name |
|---|---|---|
| activations | `identity`, `sigmoid`, `tanh`, `relu`, `leaky_relu` | `Activation::from_name("tanh")` |
| initializations | `xavier_normal`, `he_normal` | `initialization_from_name("he_normal")` |
| losses | `mean_squared_error`, `softmax_cross_entropy` | `loss_from_name("softmax_cross_entropy")` |
| optimizers | `sgd`, with the setting `learning_rate` | `optimizer_from_settings("sgd", &settings)` |

An optimizer also has settings: `settings()` returns them as a map from setting name to number
(`{"learning_rate": 0.01}`), and `optimizer_from_settings` takes them back. It refuses a missing setting and a
setting the optimizer does not have, so a typo such as `learnig_rate` is reported instead of ignored. Every
setting of the usual optimizers (learning rate, momentum, Adam's betas) is a number.

An unknown name gives an error that lists the known ones:
`unknown activation "softmax", expected one of: identity, sigmoid, tanh, relu, leaky_relu`.

The lookups return a `Box<dyn Initialization>`, `Box<dyn Loss>` or `Box<dyn Optimizer>`. A box implements its
trait too, so it can be passed wherever a value is expected: `Trainer::new(loss_from_name("…")?, …)`.

## Example: XOR

XOR outputs 1 when exactly one of its two inputs is 1. No straight line separates its two classes, so a single
layer cannot learn it, but two layers can. `crates/my_torch/tests/xor.rs` trains this network:

```rust
use my_torch::*;

let mut rng = Rng::new(42);
let architecture = Architecture::new(2, vec![
    LayerSpec::new(8, Activation::Tanh, XavierNormal)?,
    LayerSpec::new(1, Activation::Sigmoid, XavierNormal)?,
])?;
let mut network = architecture.build(&mut rng);
let xor = Dataset::new(
    Matrix::from_rows(&[[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])?,
    Matrix::from_rows(&[[0.0], [1.0], [1.0], [0.0]])?,
)?;
let mut trainer = Trainer::new(MeanSquaredError, Sgd::new(0.5)?, 2, 2000)?;
let history = trainer.fit(&mut network, &xor, &mut rng)?;
let outputs = network.predict(xor.inputs())?;
```

After 2,000 epochs the last loss is under 0.05 and every output is on the right side of 0.5. The same test
also learns XOR as two classes: an `Identity` output layer of 2 neurons, one-hot targets and
`SoftmaxCrossEntropy`, where `outputs.argmax_rows()` gives the predicted class of each row.

## Errors

`from_vec` and `from_rows` refuse a matrix with no rows or no columns, rows of different lengths, and a number
of values that does not match the shape. `Dense::new` refuses biases that are not one row of one value per
neuron. `Dataset::new` refuses inputs and targets with different numbers of rows, and `Sgd::new` a learning rate
that is not a positive number. `Trainer::new` refuses a batch size or an epoch count of 0, and `Trainer::fit` a
loss that needs raw scores on a last layer that is not `Identity`. `Network::new` refuses an empty list and
layers whose sizes do not chain, and `predict` and `gradients` refuse inputs and targets of the wrong size.
`Architecture::new` refuses 0 inputs and an empty list of layers, and `LayerSpec::new` a layer of 0 neurons.
`Model::new` refuses a loss that needs raw scores on a last layer that is not `Identity`, and `Model::load` a
file that cannot be read or is not a valid network file (see [Network files](#network-files)); `save` refuses a
file that cannot be written and weights that are not finite numbers. The name lookups refuse unknown names, and
`optimizer_from_settings` a missing or unknown setting. They return `Err(String)` without the `error:` prefix,
like `Label`: the program that calls the library adds context and the prefix.

The operations **panic** on mismatched shapes (for example `matmul` of a `2 × 3` by a `2 × 3`) and on indices
outside the matrix. `Dense` panics on an input with the wrong number of columns and on a gradient of the wrong
shape in `backward`, and `Network::update` on gradients from a network with another number of layers. The types
built on these check their sizes when they are created, so such a panic would be a bug in `my_torch`, not a
user error.
