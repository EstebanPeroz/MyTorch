# Technical documentation

This folder explains how MyTorch is built and why. The user-facing overview (build, usage, development setup)
is in the [main README](../README.md).

| Document | Contents |
|---|---|
| [architecture.md](architecture.md) | Workspace crates, how they depend on each other, design rules, error handling, current status |
| [cli.md](cli.md) | Command-line behavior of both programs: arguments, validation rules, exit codes, help |
| [labels.md](labels.md) | The outputs the network predicts: classes, colors, text format, class indices |
| [dependencies.md](dependencies.md) | External crates, why they were chosen, and how vendoring works |

Documents about the neural network itself (layers, training, file format), the dataset and the benchmarks will
be added once that code exists.
