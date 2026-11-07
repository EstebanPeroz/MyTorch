# Command line

Both programs read their arguments with [clap](https://docs.rs/clap) through the shared `cli_core` crate.

## Shared behavior

| Situation | stdout | stderr | Exit code |
|---|---|---|---|
| `--help` or `-h` | the usage text | — | 0 |
| Invalid arguments | — | `error: …` (errors detected by clap add a short usage line) | 84 |
| The program's work fails | — | `error: …` | 84 |
| Success | the program's output | — | 0 |

`--help` prints a fixed text exactly (`override_help`), not clap's generated help. Each program keeps
that text in `src/help.txt`, read with `include_str!` by the program and by its end-to-end test.

### How it works

Each program describes its arguments by implementing the `cli_core::Cli` trait:

```rust
pub trait Cli: Sized {
    const HELP: &'static str;
    fn command() -> Command;
    fn from_matches(matches: &ArgMatches) -> Result<Self, String>;
    fn try_parse_from<I, T>(args: I) -> Result<Option<Self>, String> { … }
}
```

| Item | Written by | Role |
|---|---|---|
| `HELP` | each program | the program's help text |
| `command()` | each program | the list of accepted arguments |
| `from_matches()` | each program | turns clap's parsed values into the program's `Args`, with extra validation |
| `try_parse_from()` | `cli_core` (default method) | parses; returns `Ok(Some(args))`, `Ok(None)` for `--help`, or `Err(message)` |

Each `main` then hands its work to `cli_core::run`:

```rust
fn main() -> ExitCode {
    cli_core::run(|args: Args| match args.mode() {
        Mode::Predict => modes::predict(args.load_file(), args.chess_file()),
        Mode::Train { save_file } => modes::train(args.load_file(), args.chess_file(), save_file),
    })
}
```

A new program only has to implement `Cli` and call `cli_core::run`; parsing, help and exit codes come for free.

## my_torch_analyzer

```
./my_torch_analyzer [--predict | --train [--save SAVEFILE]] LOADFILE CHESSFILE
```

| Rule | Example | Result |
|---|---|---|
| Exactly one mode is required | `./my_torch_analyzer a.nn boards.txt` | error |
| The two modes are exclusive | `--predict --train a.nn boards.txt` | error |
| A flag cannot be repeated | `--predict --predict a.nn boards.txt` | error |
| `--save` only works with `--train` | `--predict --save out.nn a.nn boards.txt` | error |
| Without `--save`, training saves into LOADFILE | `--train a.nn boards.txt` | saves into `a.nn` |
| Flags can come in any order | `--save out.nn --train a.nn boards.txt` | accepted |
| Both files are required, nothing else | `--predict a.nn` or `--predict a.nn boards.txt extra` | error |

### `--dashboard` (not in the help text)

`--dashboard` is an extra option that follows a training live in a web page. It is optional, only works with
`--train`, and is deliberately **absent from `help.txt`**: the help text is fixed by the subject, and the program
must behave as the subject describes without the option. See [dashboard.md](dashboard.md).

| Rule | Example | Result |
|---|---|---|
| Only with `--train` | `--predict --dashboard a.nn boards.txt`, `--dashboard a.nn boards.txt` | error |
| Without it, nothing changes | `--train a.nn boards.txt` | same output and same saved network |

### `--mirror` (not in the help text)

`--mirror` is a second extra: it adds to the training the mirror of every chessboard, see [mirror.md](mirror.md).
Like `--dashboard`, it is optional, only works with `--train` and is absent from `help.txt`.

| Rule | Example | Result |
|---|---|---|
| Only with `--train` | `--predict --mirror a.nn boards.txt`, `--mirror a.nn boards.txt` | error |
| Without it, nothing changes | `--train a.nn boards.txt` | same output and same saved network |

### `--dump` (not in the help text)

`--dump` prints what a network file holds instead of using it. Network files are binary, so this is how to read
one: the size of the file, each layer (shape, activation, and the range of its weights) and the training
settings. It is a third mode, so it excludes `--predict` and `--train`, and takes no CHESSFILE. Like the other
extras, it is absent from `help.txt`.

```sh
./my_torch_analyzer --dump my_torch_network.nn
```

| Rule | Example | Result |
|---|---|---|
| Takes only a network file | `--dump a.nn` | accepted |
| No CHESSFILE, no other mode or option | `--dump a.nn boards.txt`, `--dump --predict a.nn`, `--dump --save b.nn a.nn` | error |
| An unreadable file is an error | `--dump missing.nn` | `error: cannot read missing.nn: …`, exit code 84 |

The network is not checked against the chess encoding, so any network file can be dumped.

The parser resolves the save file itself, so the program receives `Mode::Train { save_file, .. }` with a path
that is always set.

**A mode is required.** The usage line shows the mode in brackets (`[--predict | --train …]`), which reads as
optional, but neither mode is a sensible default. Running without a mode is therefore an error.

## my_torch_generator

```
./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]
```

| Rule | Example | Result |
|---|---|---|
| At least one pair | `./my_torch_generator` | error |
| Arguments come in pairs | `basic.conf 3 deep.conf` | error |
| A count is a whole number ≥ 1 | `basic.conf 0`, `basic.conf -1`, `basic.conf three` | error |
| A count must fit in memory (`usize`) | `basic.conf 99999999999999999999` | error |
| Several pairs | `basic.conf 3 deep.conf 1` | accepted |

Each pair becomes a `NetworkRequest`, which checks its own count when it is created: a request with an invalid
count cannot exist.

### Generating

For each pair, the generator reads the configuration file (see [my_torch.md](my_torch.md#configurations)) and
writes `nb` networks named after it, in the current directory:

```sh
./my_torch_generator configs/basic_network.conf 3 deep.conf 1
# writes basic_network_1.nn, basic_network_2.nn, basic_network_3.nn and deep_network_1.nn
```

| Rule | Why |
|---|---|
| Network `i` of a configuration uses the seed `seed + i − 1` | the networks differ, and running the same command again writes exactly the same files |
| Every configuration file is read before any network is written | a broken file in the last pair leaves no half-done output |
| An existing file with the same name is replaced | as when a program writes its output again |
| Two configuration files with the same name (`a/basic.conf`, `b/basic.conf`) write the same files | the second pair replaces the networks of the first |
| Network files at the root of the repository are ignored by git, except `my_torch_network*.nn` | networks generated to try things out cannot be committed by mistake, while the trained network, whose name must start with `my_torch_network`, can |

Errors name the file, then the problem, and exit with 84:

```
error: cannot read nope.conf: No such file or directory (os error 2)
error: basic_network.conf: layer 1: unknown activation "softmax", expected one of: identity, sigmoid, tanh, relu, leaky_relu
```
