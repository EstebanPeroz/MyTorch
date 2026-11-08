# Design choices

The subject asks that every design choice, hyperparameters included, can be justified. This document lists each
choice of `configs/chess.conf` and of the analyzer, says **why** it was made, what the **alternatives** were, and
how solid the justification is. Each choice carries one of these tags:

| Tag | Meaning |
|---|---|
| **Theory** | follows from a known property of the method or of the data, checked in the code or the data |
| **Commit** | an experiment is recorded in the git history (the numbers are those of the commit message) |
| **Ablation** | measured by the quick ablation of the [last section](#quick-ablation-on-slim), run for this document |
| **Convention** | a standard default, chosen without being tested here |
| **Untested** | no evidence either way in the repository; stated as such |

Nothing below was tuned by an automatic search. Where the repository holds no experiment, this document says so
instead of inventing one.

## The configuration being justified

```json
{
  "inputs": 769,
  "layers": [
    { "neurons": 512, "activation": "relu", "initialization": "he_normal", "dropout": 0.2 },
    { "neurons": 256, "activation": "relu", "initialization": "he_normal", "dropout": 0.2 },
    { "neurons": 64, "activation": "relu", "initialization": "he_normal" },
    { "neurons": 6, "activation": "identity", "initialization": "xavier_normal" }
  ],
  "training": {
    "loss": "softmax_cross_entropy",
    "optimizer": { "name": "sgd", "learning_rate": 0.05 },
    "batch_size": 128,
    "epochs": 100,
    "seed": 42,
    "early_stopping": { "validation": 0.1, "patience": 5 }
  }
}
```

Trained with `make train`, that is `--mirror` on `dataset/full.txt`.

## The problem

**Task.** From a FEN, predict one of `Nothing`, `Check`/`Checkmate` (each with the attacking color) or
`Stalemate`. This is a 6-class classification problem, so the network ends with one score per class and the class
with the highest score is the answer (`classify` in `crates/my_torch_analyzer/src/encoding.rs`).

**Why a plain fully connected network.** The subject asks for a network written from scratch. Dense layers need
only matrix products, which `Matrix` provides, and they are enough to learn this task. Convolutions would suit a
board better but would be a much larger piece of code to write and verify from scratch. **Convention**: no
convolutional network was tried, so there is no measurement that a dense network is the better choice.

## Input encoding: 769 inputs

**What.** `encode` in `encoding.rs`: 12 planes of 64 squares (6 piece kinds times 2 colors) as 0/1 values, plus one
input that is 1 when White is to move. 12 x 64 + 1 = 769.

**Why.**
- One input per (piece, color, square) is a *one-hot* encoding: the network is never told that a queen is "worth
  more" than a pawn through a made-up number, which a single input holding a piece code (1 to 12) would do. It
  has to learn what each piece means. **Theory.**
- Check, checkmate and stalemate depend on where the pieces are and on who moves, so these two things are
  exactly what is encoded. The side to move is needed: the label's color is the attacker, which is the side
  *not* to move ([labels.md](labels.md)). The analyzer also forbids a prediction naming the side to move as the
  attacker (`classify`), which is a rule of chess and not a learned fact.
- The inputs are 0 or 1, so no normalization is needed. **Theory.**
- The input is sparse (at most 32 pieces out of 768 values); commit `fd65ea6` exploits that in the products.

**Not encoded:** castling rights, en passant square, halfmove and fullmove counters. They cannot change whether
a king is attacked or whether a side has a legal move, except in rare en passant or castling cases. This is a
reasoned simplification, **Untested**: no run compares the encoding with and without them.

**Alternatives.** A 8x8 board of piece codes (rejected above); attack maps or legal-move counts computed by a
chess engine as extra inputs (would make the task nearly trivial, but then the "learning" is done by hand-written
chess logic, which defeats the point of the project). Not tried.

## Output size 6 and the label set

**What.** `Label::ALL` has 6 labels, index = output neuron (`label.rs`). The target is a one-hot vector.

**Why.**
- The subject lists the possible answers; with a color for check and checkmate that is
  `Nothing`, 2 x `Check`, 2 x `Checkmate`, `Stalemate` = 6. **Theory.**
- A single 6-way classification (rather than e.g. two outputs "state" and "color") lets one softmax decide, and
  keeps the answer space exactly the set of valid answers.
- **Stalemate has no example in the dataset** ([dataset.md](dataset.md)): 0 of 723,019 lines. The class is kept
  because the subject counts it as a bonus and the output format must be able to express it, but the
  network can never learn it from this data, so it is never predicted in practice. Dropping it for 5 outputs
  was the other option listed in `dataset.md`; it was not chosen, and **no measurement** compares the two. Adding
  stalemate positions to the dataset is the only way to make the class real.

## Layer sizes: 512 / 256 / 64 / 6

**What.** Three hidden layers narrowing by about a factor 2 to 4, then the 6 outputs.

**History (Commit).** The sizes were changed twice, so they are an empirical choice:

| Commit | Hidden sizes | Reason in the commit message |
|---|---|---|
| `021a8f6` | first config | first trained network, 87.6 % accuracy on the *training* data (50 epochs, early stopping) |
| `f0a18fc` | doubled to 1024 / 512 / 128 | none given |
| `5d061fa` | back to 512 / 256 / 64 | "Narrower layers ... since dropout slows the convergence" (new config: dropout, lr 0.05, batch 128, up to 100 epochs) |

**Why this shape.**
- A funnel (wide near the 769 inputs, narrow near the 6 outputs) compresses the board into a few features before
  the decision. **Convention.**
- 512/256/64 is a size that trains in a reasonable time on the whole dataset, while [training-metrics.md](training-metrics.md) hypothesizes that
  "769 inputs into 128 then 64 neurons is small for chess" (an untested hypothesis, not a result). Why 512/256/64 and not 1024/512/128 is stated by the commit only as "narrower, because
  of dropout"; there is **no recorded comparison of the two on the full dataset**, so the final sizes are
  **Untested** as an optimum.
- The number of parameters is about 769x512 + 512x256 + 256x64 + 64x6, around 0.54 million, for 723,019 (and
  about 1.4 million with mirrors) examples. Parameters are fewer than examples, which is a prudent order of
  magnitude against overfitting. **Theory** (rule of thumb only).

**Alternatives.** Fewer, wider or deeper layers: see the ablation, where a single 64-neuron layer did better on
the small subset. That result does not transfer to the full dataset (see its limits).

## Activations: ReLU hidden, identity output

**ReLU (hidden).**
- Its gradient is 1 for positive inputs, so it does not saturate like `sigmoid` or `tanh`: gradients survive
  through several layers. It is also the cheapest to compute (a comparison). **Theory.**
- It is the activation for which **He initialization** is derived (next section). **Theory.**
- Alternatives that exist in the library: `sigmoid`, `tanh`, `leaky_relu`, `identity`. In the ablation they
  gave results within or near the noise of ReLU (tanh slightly higher, leaky ReLU equal). ReLU was not
  chosen *because* it won that comparison; the repository holds no earlier comparison.

**Identity (output).**
- The last layer outputs raw scores (*logits*). The softmax is applied inside the loss, not in the layer: this
  is numerically stabler than a separate softmax followed by a logarithm, and gives the simple gradient
  `softmax(z) - target`. The library even states it in an error ("name raw scores, not softmax", commit
  `64a9057`). **Theory.**
- Prediction needs only the largest score, and softmax does not change which one is the largest, so the
  prediction does not need it. **Theory.**

## Initialization: He normal / Xavier normal

**What.** Hidden ReLU layers: He normal, weights ~ N(0, 2 / inputs) (`he_normal.rs`). Output layer: Xavier normal.

**Why.**
- With ReLU about half of the activations are zeroed, so the variance of the weights must be doubled
  (2 / fan-in) to keep the signal's variance from shrinking from layer to layer. Xavier assumes a roughly
  linear (or tanh) activation, which fits the identity output layer. **Theory** (He et al. 2015, Glorot and
  Bengio 2010).
- Any scale chosen at random could make the signal vanish or explode in a 4-layer network; the 769-input first
  layer would be the most exposed.

**Alternatives.** Xavier everywhere, He on the output. In the ablation both gave results that differ from the
base by about one standard deviation or less. The effect is small at this depth; the rule above is kept because
it is the textbook match between initialization and activation, **not because the ablation shows an advantage**.

## Loss: softmax cross-entropy

**Why.**
- The outputs are mutually exclusive classes; softmax turns scores into a probability distribution and
  cross-entropy is the negative log-likelihood of the right class. It is the standard loss for this and its
  gradient with respect to the scores is just `softmax(z) - target`. **Theory.**
- Mean squared error is available in the library. It treats the target as a regression value, gives weak
  gradients when the network is confidently wrong and is not the maximum-likelihood loss for classes.
  **Ablation:** about 4 to 5 points lower accuracy than cross-entropy on the subset.

## Optimizer: plain SGD, learning rate 0.05

**What.** Mini-batch stochastic gradient descent without momentum (`sgd.rs`), learning rate 0.05.

**Why SGD.** It is the simplest optimizer; it keeps the
from-scratch code small and every update easy to verify against a hand computation. Momentum or Adam usually
converge faster (this is listed as a lever in [training-metrics.md](training-metrics.md), not as a result). They
were **not implemented and not tried**, so SGD is a **Convention / Untested** choice, not a measured winner.

**Why 0.05.** This value is empirical (**Commit** and **Ablation**):
- `31aa6b9` moved to batches of 512 with learning rate 0.1: on 100k boards (with mirrors), 15 epochs, validation
  loss 0.728 against 0.898 for batches of 64 with 0.01.
- `5d061fa` then lowered it to 0.05 together with batch 128 and dropout. The commit gives no number for this
  step. The reason it gives is that dropout slows convergence, which says to allow more epochs, not why 0.05
  rather than 0.1.
- Ablation (learning rate 0.5 and 0.01 against 0.05): 0.5 was 1.8 points lower on average (and very unstable
  from seed to seed, SD 2.2 points); 0.01 was about equal but needed several times more time to stop. So 0.05 is
  in a flat region between a rate too small and a rate that starts to diverge, which is the reasoning for
  choosing it, but the ablation cannot show that 0.05 beats 0.02 or 0.1.

## Batch size: 128

**Why.**
- Larger batches make better use of the cores (the gradient of a batch is computed on parallel shards, commit
  `85ee40e`, and the products are split across threads, `95c140a`), and the `31aa6b9` experiment shows 512
  being faster than 64 for the same quality. Smaller batches give noisier gradients, which help generalization
  and give more updates per epoch. 128 is a compromise. **Commit** (for the throughput side) and **Convention**
  for 128 itself.
- The Ablation found batch 512 clearly worse (4.7 points lower) than 128 at the same learning rate and number of
  epochs: with 8,000 examples a batch of 512 gives only ~14 updates per epoch. This agrees with the commit
  that moved from 512 to 128, but it is on a small dataset, where batch size matters more than on 1.4 million
  examples. Batch 32 was equal to 128 within noise (+0.3 points).

## Dropout: 0.2 on the two widest layers

**What.** Inverted dropout (`dense.rs`: dropped outputs are 0, the others are scaled by `1 / (1 - rate)`, so
nothing changes at prediction time). Rate 0.2 on the 512 and 256 layers; none on the 64 layer or the output.

**Why.**
- The widest layers have the most parameters and so the most capacity to memorize the training boards; dropout
  makes each neuron useful on its own, an inexpensive regularizer. **Theory.**
- Not on the last hidden layer or the output: dropping part of the few remaining features or one of 6 class
  scores adds a lot of noise to the decision itself. **Convention.**
- 0.2 is a light rate (0.5 is the classic value for large layers). Chosen so that the added noise does not
  stall the convergence; the commit says that even with it, "dropout slows the convergence". **Convention**, no
  other rate was compared in the repository.
- **Ablation:** removing dropout lowered the mean accuracy from 51.8 % to 50.5 % and, notably, increased the
  spread between seeds by a factor of 2.7. The mean difference (1.3 points) is about one standard deviation, so
  it is suggestive and not conclusive; the variance effect is clearer.

## Early stopping: validation 0.1, patience 5

**What.** 10 % of the training data is set aside; after each epoch the validation loss is measured; training
stops after 5 epochs without improvement, and the network of the best epoch is restored (`trainer.rs`,
`early_stopping.rs`).

**Why.**
- The training loss always goes down, so it cannot tell when the network begins to memorize. The loss on data the
  network does not train on can. Restoring the best network means a few extra epochs cost nothing. **Theory.**
- With 723,019 positions, 10 % (about 72,000) is plenty for a stable estimate of the loss; with a smaller share
  it would get noisier, with a larger one training data would be wasted. **Convention.**
- Patience 5: with SGD and dropout the validation loss is noisy from epoch to epoch, so stopping at the first
  non-improvement would stop on noise; 5 is long enough to ride out such noise and short enough not to run all
  100 epochs when it has plateaued. **Convention**, **Untested** against 3 or 10.
- `epochs: 100` is a ceiling, not a target. It was raised in `5d061fa` because dropout slows convergence.
- Duplicate boards are removed before the split and a board and its mirror are kept on the same side
  ([mirror.md](mirror.md)); both prevent the validation loss from being flattered by copies of training boards.

## Seed: 42

It seeds the weights, the validation split, the shuffling of the samples and the dropout masks. The same seed
gives the same network (test `the_same_seed_trains_the_same_network`), so a result can be reproduced and a change
can be compared to the previous one. The value 42 is arbitrary and **was not selected**. The ablation shows that
the seed alone moves the accuracy by about 1 point (SD 1.0 on the base), so differences smaller than that between
two configs are not meaningful.

## Mirror augmentation (`--mirror`)

**Why.** Swapping colors, flipping ranks and giving the move to the other side gives a legal position whose label
only swaps the attacker's color ([mirror.md](mirror.md)). Chess is symmetric in this way, so it is a free
doubling of the data and it balances the colors (the data has 13 % `Checkmate White` and 11 % `Checkmate
Black`). **Theory** (the symmetry), checked by the tests of `label.rs` and `encoding.rs`.

**Ablation:** +3.9 points of test accuracy on average (55.7 % against 51.8 %, larger than the SDs), the clearest
single improvement of the ablation. It is the reason `make train` always passes `--mirror`. Note that the flag
is off by default, so a direct `--train` run does not use it.

## Dataset composition

From [dataset.md](dataset.md): 723,019 labeled positions in 9 files, split by label (`Nothing`, `Check`,
`Checkmate`) and by size (up to 9 pieces, 10 to 19, 20 and more). It was supplied for the project, not generated
here, so its composition is a given, not a choice, except for these decisions:

- **Duplicates are removed** (about 1 % of the lines) before any split. **Theory** (leakage between training and
  validation).
- **No class weighting or oversampling.** The imbalance is mild (41 % / 35 % / 24 % by family); the choice is
  **Untested**, and per-label scores are shown by the dashboard to check it.
- **No stalemate**, discussed above.

## Quick ablation on slim

**Setup (reproducible).** `dataset/slim.txt` (10,000 positions: 4,149 `Nothing`, 3,448 `Check`, 2,403
`Checkmate`, 0 `Stalemate`) was shuffled with Python's `random.Random(1)` and split into 8,000 training and 2,000
test positions. Each variant was generated from `configs/chess.conf` with one change, trained with
`my_torch_analyzer --train` (no `--mirror` except for the mirror row), and scored with `--predict` on the 2,000
test positions: the fraction of lines whose printed label equals the expected one. Five seeds (1 to 5) per row;
mean and standard deviation (SD) of the test accuracy. Early stopping is on, as in the config.

| Variant (change from `chess.conf`) | Test accuracy, mean | SD |
|---|---|---|
| **base** (`chess.conf`) | 51.8 % | 1.0 |
| `--mirror` | 55.7 % | 1.2 |
| no dropout | 50.5 % | 2.7 |
| learning rate 0.5 | 50.0 % | 2.2 |
| learning rate 0.01 | 51.6 % | 0.6 |
| batch size 512 | 46.6 % | 0.6 |
| batch size 32 | 52.1 % | 1.2 |
| one hidden layer of 64 | 54.1 % | 0.8 |
| all hidden layers twice as wide (1024/512/128) | 51.1 % | 2.8 |
| Xavier in hidden layers | 53.0 % | 1.3 |
| He on the output layer | 51.9 % | 0.9 |
| leaky ReLU hidden | 52.0 % | 0.9 |
| tanh hidden | 53.8 % | 0.5 |
| mean squared error loss | 48.0 % | 1.1 |

**Reading it.**
- Differences under about 2 points are inside the noise (two seeds of the same config differ by that much).
  Clear effects: `--mirror` (+3.9), batch 512 (-5.2), MSE (-3.8), and probably a learning rate that is too high.
- The single 64-neuron layer (+2.3) and tanh (+2.0) came out *above* the base. This is **not** evidence that the
  config is wrong: on 8,000 examples a bigger network overfits earlier, and the config is meant for the
  full dataset (about 1.4 million examples with mirrors), where more capacity may be useful (not measured here). It does mean the layer sizes
  and ReLU are **not demonstrated by this ablation**; their support is the reasoning above and the commits.
- All the accuracies are low (~52 %) in absolute terms. The model only sees 8,000 positions and the subset is
  too small for the task; the figures only compare variants against each other. They are **not** the accuracy of
  the shipped network, which is trained on the full dataset.

**Limits.** Small subset, 2,000 test positions (a standard error of about 1.1 points on its own), one split,
5 seeds, no tuning of the other hyperparameters to each variant, one change at a time (interactions such as
dropout with width were not explored). Layer sizes, learning rate and patience were not searched on the full
dataset in this document; a proper hyperparameter search, if done, is documented separately.

## What is not justified

- The 512/256/64 sizes as an optimum (only a history of two changes, no comparison on the full dataset).
- The exact learning rate 0.05, the dropout rate 0.2 and the patience 5 (reasoned values, not scanned).
- ReLU over tanh (the ablation does not favor ReLU on the subset).
- SGD over momentum or Adam (never implemented or tried).
- Removing the stalemate class or adding stalemate data (no stalemate data exists).
- Ignoring castling, en passant and the move counters.
