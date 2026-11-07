# Training dashboard

```
./my_torch_analyzer --train --dashboard LOADFILE CHESSFILE
```

`--dashboard` follows a training live in a web page. It is optional and only works with `--train`. It is not
in `help.txt` (the help text is fixed by the subject), and without it the program behaves exactly as before.

On start, the program prints one line on **stderr** and trains as usual:

```
dashboard: http://127.0.0.1:41873
```

Open that address in a browser. The port is free and chosen by the system, so it is different each time.
Nothing is opened automatically, which keeps tests and SSH sessions quiet (over SSH, forward the port). stdout
is the same as without the option.

## What the page shows

| Part | Contents |
|---|---|
| Tiles | The current epoch out of the most epochs allowed, the last losses and accuracies, and the best epoch (the lowest validation loss) |
| Loss chart | Training loss (solid) and validation loss (dashed), per epoch |
| Accuracy chart | Training and validation accuracy, per epoch |
| Macro F1, majority baseline | The unweighted mean of the per-label F1, and the accuracy of always predicting the most frequent label, which the network has to beat |
| Model shape | The layer sizes from the input to the output, with each layer's activation and number of parameters (weights and biases), drawn from the network the training starts with |
| Confusion matrix | Expected labels in rows, predicted labels in columns, shaded by the share of each row (so the diagonal is the recall of each label), with the samples, precision, recall and F1 of each label below |
| Table | The same values, in a `Show the values as a table` section, newest epoch first |
| Hover | A tooltip with the values of the epoch under the pointer, on both charts |

The matrix, the F1 and the baseline are measured after the last epoch, on the validation samples, or on the
training samples when there is no validation split (the title says which). The F1 skips labels without any sample.
Looking at the matrix instead of the accuracy alone shows which labels get mixed up, such as a checkmate predicted
as a check. It is the confusion matrix and the per-class metrics that the training metrics document (PR #11,
`docs/training-metrics.md`) lists as a next step.

The validation curves only exist when the network file enables early stopping (as `configs/chess.conf` does).
Accuracy is measured the same way as the final `accuracy:` line: the label chosen by `encoding::classify`, which
only allows the colors that can give check, compared with the expected label.

## How training ends

The page keeps everything it has received. When the training ends, it says so, then stays on screen even after the
program exits:

| Status line | Meaning |
|---|---|
| `● Training, epoch 12 of at most 50` | running (early stopping can end it sooner) |
| `✓ Training done` | the training finished; the curves stay visible |
| `✕ Training failed: …` | the training returned an error |
| `■ Connection lost: the program has stopped …` | the program is gone without saying why (for example `Ctrl+C`) |

After the training, the program stays up for 1.5 seconds so that an open page, which asks for news twice a second,
can see the end. Then it saves the network and exits.

## How it works

```
 my_torch                       my_torch_analyzer
 ┌──────────────────┐ epoch    ┌────────────────────┐   GET /metrics   ┌─────────┐
 │ Trainer::        │ report   │ modes::train       │ ◄─────────────── │ browser │
 │  fit_observed    │ ───────► │  measures accuracy │   (twice a sec)  └─────────┘
 └──────────────────┘          │ dashboard::Dashboard│
                               └────────────────────┘
```

- `Trainer::fit_observed` calls a closure after each epoch with an `EpochReport` (see
  [my_torch.md](my_torch.md#watching-a-training)). The library does not know about accuracy or chess: the report
  carries the network and the row indices of the training and validation samples, and the analyzer measures the
  accuracy on its own data. It costs one pass over the data per epoch, only with `--dashboard`.
- `dashboard::Dashboard` is a small HTTP server written with `std::net`, on 127.0.0.1 only (never reachable from
  another machine). It adds no dependency. `GET /` returns the page (`src/dashboard.html`, embedded in the
  binary, with no external script or font, so it works offline) and `GET /metrics` returns the epochs so far as
  JSON. Each connection runs in its own thread and answers once.
- The page draws the curves itself with SVG and follows the system's light or dark theme.
- Each epoch, `modes::train` fills a `Confusion` (`src/confusion.rs`, counts of expected against predicted
  labels) with `encoding::confusion` for the training and the validation samples. The accuracies of the charts
  come from it, so one pass gives both. Only the latest matrix is sent; the page computes precision, recall and F1.
- A value that is not a number (a diverged loss, for example) is sent as `null` and shown as a gap.

## Limits

- Only one page layout and no controls: no zoom, no pause, no export.
- The server answers one request per connection and does not check the `Host` header. It only listens on the
  loopback interface and serves training numbers, nothing else.
