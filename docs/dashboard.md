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
| Table | The same values, in a `Show the values as a table` section, newest epoch first |
| Hover | A tooltip with the values of the epoch under the pointer, on both charts |

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
- A value that is not a number (a diverged loss, for example) is sent as `null` and shown as a gap.

## Limits

- Only one page layout and no controls: no zoom, no pause, no export.
- The server answers one request per connection and does not check the `Host` header. It only listens on the
  loopback interface and serves training numbers, nothing else.
