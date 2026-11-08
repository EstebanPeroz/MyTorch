# Hyperparameter optimization

`my_torch_tuner` searches the hyperparameters of the network (learning rate, batch size, dropout, activation,
widths of the hidden layers) instead of picking them by hand. It is a third program, next to the analyzer and the
generator, and its output is a configuration file that `my_torch_generator` accepts as it is.

```sh
make
./my_torch_tuner configs/chess.conf configs/search_space.json dataset/slim.txt
./my_torch_generator best.conf 1        # then train best_1.nn with my_torch_analyzer --train
```

## The method

The metaheuristic is a **random search with successive halving**:

1. **Draw** `--trials` random sets of hyperparameters from the search space.
2. **Train** a network for each, from scratch, for a short budget (`--epochs`), and **score** it on a
   validation split that was held out of the data.
3. **Keep the better half** and train it again from scratch with **twice the epochs**.
4. Repeat for `--rounds` rounds, then return the best trial of the last round.

The budget goes where it pays: a configuration that is poor after 4 epochs is dropped, and only the promising ones
get the 8 and 16 epochs that tell them apart. With `--rounds 1` the search is a plain random search. The
trials of a round are independent, and a trial that fails (the training diverged to NaN, for example) ranks below
every trial that worked.

Why not a grid search? The search space has 5 dimensions, two of them continuous. A grid wastes most of its
trials on values of the unimportant dimensions, while random search tries a new value of every dimension in
each trial (Bergstra and Bengio, 2012). The log-uniform draw gives each decade of the learning rate the same
share of trials.

### What a trial is

| Step | Detail |
|---|---|
| Data | `CHESSFILE` is read like `--train` reads it (labels required), the duplicate chessboards are removed, then at most `--samples` of them are picked at random (default: all) and `--validation` of those (default 0.2) are held out. The split depends on the seed only, so every trial sees the same one. |
| Configuration | `BASE_CONFIG` with the drawn values applied, `epochs` set to the budget of the round and `early_stopping` removed (the search has its own validation split, and a fixed budget keeps the trials comparable). |
| Training | `ModelConfig` builds the network, the same code as the generator and the analyzer. The seed of `BASE_CONFIG` is kept, so every trial starts from the same random numbers and only the hyperparameters differ. |
| Score | accuracy on the held-out chessboards, using the same prediction code as `--predict` (a prediction never names the side to move as the attacker). The validation loss breaks ties. |

The mirrored chessboards of `--mirror` are not used by the search.

## Search space file

A JSON object that names the hyperparameters to search. A hyperparameter that is not in it keeps the value of
`BASE_CONFIG`. Each one is given either a list of `choices`, or, for the two numeric ones, a range:

| Hyperparameter | Allowed forms | Changes in the configuration |
|---|---|---|
| `learning_rate` | `choices`, `uniform`, `log_uniform` (positive) | `training.optimizer.learning_rate` |
| `batch_size` | `choices` (whole numbers >= 1) | `training.batch_size` |
| `dropout` | `choices`, `uniform` (in [0, 1)) | `dropout` of every hidden layer |
| `activation` | `choices` (names of the activations) | `activation` of every hidden layer, and its `initialization` (`he_normal` for `relu` and `leaky_relu`, `xavier_normal` otherwise) |
| `hidden` | `choices` (each one a list of widths) | the hidden layers: one per width, copied from the first hidden layer of `BASE_CONFIG`. The last layer (the outputs) is never changed. |

```json
{
  "learning_rate": { "log_uniform": [0.005, 0.2] },
  "batch_size": { "choices": [32, 64, 128, 256] },
  "dropout": { "uniform": [0.0, 0.4] },
  "activation": { "choices": ["relu", "leaky_relu", "tanh"] },
  "hidden": { "choices": [[512, 256, 64], [256, 128, 64], [256, 64], [128, 64], [512, 128]] }
}
```

This is `configs/search_space.json`. An unknown hyperparameter, an empty or invalid range, an unknown activation
or a value of the wrong kind is refused before anything is trained.

## Command line

```
./my_torch_tuner [OPTIONS] BASE_CONFIG SEARCH_SPACE CHESSFILE
```

| Option | Default | Meaning |
|---|---|---|
| `--trials N` | 12 | networks tried in the first round |
| `--rounds N` | 3 | rounds of halving (1 = random search) |
| `--epochs N` | 5 | epochs of the first round, doubled every round |
| `--samples N` | all | use at most N chessboards |
| `--validation F` | 0.2 | share of the chessboards held out, in ]0, 1[ |
| `--seed N` | seed of `BASE_CONFIG` | seed of the draws and of the split |
| `--log FILE` | `tuner_log.jsonl` | the trial log |
| `--output FILE` | `best.conf` | the best configuration |

The total training cost is `trials x epochs x (1 + 1 + 1 + ...)` over the rounds: with `--trials 16 --rounds 3
--epochs 4`, 16 x 4 + 8 x 8 + 4 x 16 = 192 epochs, the same as 12 full trainings of 16 epochs while looking at 16
configurations. Errors follow the other programs: `error: ...` on stderr and exit code 84 (see
[cli.md](cli.md)).

## Outputs

**The log** has one JSON object per trial, written as soon as the trial is scored, so a search that is stopped
still leaves its results:

```json
{"epochs":4,"error":null,"params":{"activation":"relu","batch_size":32,"dropout":0.01425,"hidden":[512,128],"learning_rate":0.047},"round":1,"seconds":1.35,"trial":12,"validation_accuracy":0.4985,"validation_loss":0.978}
```

`round` and `trial` count from 1 and `trial` is the number of the trial in the first round, so the lines of one
trial can be followed across rounds. A failed trial has `error` set and no scores.

**The best configuration** is `BASE_CONFIG` with the winning hyperparameters. Its `epochs` and `early_stopping` are the
ones of `BASE_CONFIG`: the short budget is only for comparing trials, the real training is long. Keys are written
in alphabetical order.

## Results on `slim.txt`

`configs/chess.conf` as the base, `configs/search_space.json` as the space, the 10 000 chessboards of
`dataset/slim.txt` (7 992 to train on and 1 998 to validate after removing duplicates):

```sh
./my_torch_tuner --trials 16 --rounds 3 --epochs 4 configs/chess.conf configs/search_space.json dataset/slim.txt
```

It ran 28 trials (192 epochs) in 44 seconds on 16 cores.

| Round | Trials | Epochs | Best accuracy | Median accuracy |
|---|---|---|---|---|
| 1 | 16 | 4 | 51.5 % | 43.7 % |
| 2 | 8 | 8 | 54.8 % | 50.1 % |
| 3 | 4 | 16 | 56.2 % | 55.2 % |

The winner, trial 12 (56.2 % accuracy and a validation loss of 1.20 after 16 epochs):

| Hyperparameter | `chess.conf` | Found |
|---|---|---|
| learning rate | 0.05 | 0.047 |
| batch size | 128 | 32 |
| dropout | 0.2 on two layers | 0.014 on both hidden layers |
| activation | relu | relu |
| hidden layers | 512, 256, 64 | 512, 128 |

What the search shows:

- **Most of the configurations are no better than the majority class.** `Nothing` is 41.5 % of `slim.txt`, and the
  stuck trials score 43 %, what a network gets by always answering the majority class. 9 of the 16 first-round trials stayed under 44 % after 4 epochs: the network has not left
  the starting plateau. None of the trials with a batch size of 256 got over 44.7 %. Halving drops the stuck
  trials after the first round, which is where it saves the most time.
- **Small batches win in a short budget.** The three best first-round trials all use a batch size of 32 (with
  learning rates between 0.045 and 0.12): they make 4 to 8 times more updates per epoch than a batch size of 128
  or 256.
- **Short runs are noisy.** The validation split has 1 998 chessboards, so one point of accuracy is 20
  chessboards, and the ranking changes between rounds (trial 11 leads rounds 1 and 2, trial 12 wins round 3
  with a worse loss, 1.20 against 0.96). The accuracy decides, then the loss. Rerun with another `--seed`, or a
  larger `--samples`, before trusting a small gap.
- **The result is for a short budget.** A configuration that is best after 16 epochs is not necessarily best
  after 100 with early stopping. Use the best configuration as a starting point, then train it fully with
  `make train CONFIG=best.conf`.

The log of this run is not kept in git; it is regenerated by the command above (the seed is fixed, so the draws
are the same, though the times differ).

## Code

| File (`crates/my_torch_tuner/src/`) | Role |
|---|---|
| `space.rs` | reads the search space, draws a set of hyperparameters, applies it to a configuration |
| `search.rs` | the halving loop. It does not know about chess: it takes a function that scores a set of hyperparameters |
| `evaluate.rs` | the chessboards (split) and the function that trains and scores one trial |
| `cli.rs`, `main.rs` | arguments, log and output files |

The search and the space are unit-tested without any training (a fake scoring function with a known best
learning rate), `evaluate.rs` on a 40-chessboard fixture, and the program end to end in `tests/cli.rs` (log
shape, halving, a best configuration the generator accepts, same seed same result, error messages).
