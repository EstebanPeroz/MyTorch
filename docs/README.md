# Technical documentation

This folder explains how MyTorch is built and why. The user-facing overview (build, usage, development setup)
is in the [main README](../README.md).

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | Workspace crates, how they depend on each other, design rules, error handling, current status |
| [cli.md](cli.md) | Command-line behavior of both programs: arguments, validation rules, exit codes, help |
| [dashboard.md](dashboard.md) | The optional `--dashboard` page that follows a training live: what it shows, how it works, its limits |
| [labels.md](labels.md) | The outputs the network predicts: classes, colors, text format, class indices |
| [chess-file.md](chess-file.md) | The CHESSFILE format, how it is read, its error messages, the `Board` type |
| [my_torch.md](my_torch.md) | The neural network library: conventions (`f32`, one sample per row) and the `Matrix` type |
| [dataset.md](dataset.md) | The training data: layout, statistics, quality, open decisions |
| [training-metrics.md](training-metrics.md) | Which metrics to follow while training, how to detect overfitting, levers to raise the accuracy |
| [dependencies.md](dependencies.md) | External crates, why they were chosen, and how vendoring works |
| [testing.md](testing.md) | Unit and end-to-end tests, the `cli_test` helpers, CI |

Documents about the neural network itself (layers, training, file format) and the benchmarks will be added
once that code exists.
