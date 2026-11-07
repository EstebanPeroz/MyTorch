use std::fs;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use cli_test::Program;
use my_torch::{Model, ModelConfig};
use my_torch_analyzer::color::Color;
use my_torch_analyzer::label::Label;

const HELP: &str = include_str!("../src/help.txt");
const LABELED: &str = "tests/fixtures/labeled.txt";
const UNLABELED: &str = "tests/fixtures/unlabeled.txt";

fn analyzer() -> Program {
    Program::new(env!("CARGO_BIN_EXE_my_torch_analyzer"))
}

#[test]
fn help_prints_the_usage() {
    analyzer().assert_help(HELP);
    analyzer()
        .run(&["--help", "--predict"])
        .assert_success(HELP);
}

#[test]
fn invalid_arguments_exit_with_84() {
    analyzer().assert_errors(&[
        (&[], "the following required arguments were not provided"),
        (
            &["net.nn", "boards.txt"],
            "the following required arguments were not provided",
        ),
        (
            &["--predict", "net.nn"],
            "the following required arguments were not provided",
        ),
        (
            &["--predict", "--train", "net.nn", "boards.txt"],
            "the argument '--predict' cannot be used with '--train'",
        ),
        (
            &["--predict", "--predict", "net.nn", "boards.txt"],
            "the argument '--predict' cannot be used multiple times",
        ),
        (
            &["--predict", "--save", "out.nn", "net.nn", "boards.txt"],
            "the argument '--predict' cannot be used with '--save",
        ),
        (
            &["--predict", "net.nn", "boards.txt", "extra"],
            "unexpected argument 'extra' found",
        ),
        (
            &["--unknown", "net.nn", "boards.txt"],
            "unexpected argument '--unknown' found",
        ),
        (&["--train", "--save"], "a value is required for '--save"),
        (
            &["--predict", "--dashboard", "net.nn", "boards.txt"],
            "the argument '--predict' cannot be used with '--dashboard'",
        ),
        (
            &["--dashboard", "net.nn", "boards.txt"],
            "the following required arguments were not provided",
        ),
    ]);
}

const TINY_CONFIG: &str = r#"{
  "inputs": 769,
  "layers": [
    { "neurons": 8, "activation": "relu", "initialization": "he_normal" },
    { "neurons": 6, "activation": "identity", "initialization": "xavier_normal" }
  ],
  "training": {
    "loss": "softmax_cross_entropy",
    "optimizer": { "name": "sgd", "learning_rate": 0.05 },
    "batch_size": 2,
    "epochs": 3,
    "seed": 1
  }
}"#;

fn temporary(name: &str) -> PathBuf {
    let directory = std::env::temp_dir().join(format!("my_torch_analyzer_{}", std::process::id()));
    fs::create_dir_all(&directory).unwrap();
    directory.join(name)
}

fn tiny_network(name: &str) -> PathBuf {
    let path = temporary(name);
    ModelConfig::from_json(TINY_CONFIG)
        .unwrap()
        .generate(0)
        .save(&path)
        .unwrap();
    path
}

fn path(path: &Path) -> &str {
    path.to_str().unwrap()
}

#[test]
fn predict_prints_one_label_per_chessboard_in_order() {
    let network = tiny_network("predict.nn");
    for chess_file in [LABELED, UNLABELED] {
        let outcome = analyzer().run(&["--predict", path(&network), chess_file]);
        let expected = fs::read_to_string(chess_file)
            .unwrap()
            .lines()
            .filter(|line| !line.trim().is_empty())
            .count();
        let stdout = outcome.stdout();
        assert_eq!(outcome.code(), Some(0), "{outcome}");
        assert_eq!(stdout.lines().count(), expected, "{outcome}");
        for line in stdout.lines() {
            assert!(line.parse::<Label>().is_ok(), "{line:?}");
        }
    }
}

#[test]
fn predict_never_gives_the_color_of_the_side_to_move() {
    let network = tiny_network("colors.nn");
    let outcome = analyzer().run(&["--predict", path(&network), UNLABELED]);
    let labels: Vec<Label> = outcome
        .stdout()
        .lines()
        .map(|l| l.parse().unwrap())
        .collect();
    for (label, line) in labels
        .iter()
        .zip(fs::read_to_string(UNLABELED).unwrap().lines())
    {
        let white_to_move = line.split_whitespace().nth(1) == Some("w");
        let forbidden = if white_to_move {
            Color::White
        } else {
            Color::Black
        };
        assert!(!matches!(label, Label::Check(c) | Label::Checkmate(c) if *c == forbidden));
    }
}

#[test]
fn train_saves_into_the_load_file_by_default() {
    let network = tiny_network("train.nn");
    let before = fs::read_to_string(&network).unwrap();
    let outcome = analyzer().run(&["--train", path(&network), LABELED]);
    assert_eq!(outcome.code(), Some(0), "{outcome}");
    assert!(outcome.stdout().contains("epoch 3: loss"), "{outcome}");
    assert!(outcome.stdout().contains("accuracy: "), "{outcome}");
    assert_ne!(fs::read_to_string(&network).unwrap(), before);
    assert!(Model::load(&network).is_ok());
}

#[test]
fn train_with_save_leaves_the_load_file_untouched() {
    let network = tiny_network("source.nn");
    let saved = temporary("saved.nn");
    let before = fs::read_to_string(&network).unwrap();
    let outcome = analyzer().run(&["--train", "--save", path(&saved), path(&network), LABELED]);
    assert_eq!(outcome.code(), Some(0), "{outcome}");
    assert_eq!(fs::read_to_string(&network).unwrap(), before);
    assert_ne!(fs::read_to_string(&saved).unwrap(), before);
}

#[test]
fn a_trained_network_predicts_what_it_was_taught() {
    let network = tiny_network("learn.nn");
    let config = TINY_CONFIG.replace("\"epochs\": 3", "\"epochs\": 400");
    ModelConfig::from_json(&config)
        .unwrap()
        .generate(0)
        .save(&network)
        .unwrap();
    let trained = analyzer().run(&["--train", path(&network), LABELED]);
    assert_eq!(trained.code(), Some(0), "{trained}");
    let outcome = analyzer().run(&["--predict", path(&network), LABELED]);
    let expected: Vec<String> = fs::read_to_string(LABELED)
        .unwrap()
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            line.split_whitespace()
                .skip(6)
                .collect::<Vec<_>>()
                .join(" ")
        })
        .collect();
    assert_eq!(outcome.stdout().lines().collect::<Vec<_>>(), expected);
}

#[test]
fn unusable_networks_exit_with_84() {
    let wrong_shape = temporary("wrong_shape.nn");
    ModelConfig::from_json(&TINY_CONFIG.replace("769", "10"))
        .unwrap()
        .generate(0)
        .save(&wrong_shape)
        .unwrap();
    let garbage = temporary("garbage.nn");
    fs::write(&garbage, "not a network").unwrap();
    for mode in ["--predict", "--train"] {
        analyzer()
            .run(&[mode, "tests/fixtures/missing.nn", LABELED])
            .assert_error("cannot read tests/fixtures/missing.nn");
        analyzer()
            .run(&[mode, path(&garbage), LABELED])
            .assert_error(&format!("{}: invalid network file", garbage.display()));
        analyzer()
            .run(&[mode, path(&wrong_shape), LABELED])
            .assert_error(&format!(
                "{}: the network has 10 inputs and 6 outputs",
                wrong_shape.display()
            ));
    }
}

#[test]
fn invalid_chess_files_exit_with_84() {
    analyzer().assert_errors(&[
        (
            &["--predict", "net.nn", "tests/fixtures/missing.txt"],
            "cannot read 'tests/fixtures/missing.txt'",
        ),
        (
            &["--predict", "net.nn", "tests/fixtures/empty.txt"],
            "tests/fixtures/empty.txt: no chessboards",
        ),
        (
            &["--predict", "net.nn", "tests/fixtures/bad_label.txt"],
            "tests/fixtures/bad_label.txt:2: unknown label 'Chek White'",
        ),
        (
            &["--predict", "net.nn", "tests/fixtures/bad_fen.txt"],
            "tests/fixtures/bad_fen.txt:3: invalid FEN 'rnbqkbnrr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1': invalid rank length: 9",
        ),
        (
            &["--train", "net.nn", UNLABELED],
            "tests/fixtures/unlabeled.txt:1: missing label, required for training",
        ),
    ]);
}

fn fetch(url: &str, target: &str) -> Option<String> {
    let mut stream = TcpStream::connect(url.strip_prefix("http://")?).ok()?;
    write!(stream, "GET {target} HTTP/1.1\r\nHost: x\r\n\r\n").ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    Some(response.split_once("\r\n\r\n")?.1.to_string())
}

#[test]
fn the_dashboard_shows_the_training_live_and_changes_nothing_else() {
    let plain = tiny_network("plain.nn");
    let watched = tiny_network("watched.nn");
    let outcome = analyzer().run(&["--train", path(&plain), LABELED]);
    assert_eq!(outcome.code(), Some(0), "{outcome}");

    let mut child = Command::new(env!("CARGO_BIN_EXE_my_torch_analyzer"))
        .args(["--train", "--dashboard", path(&watched), LABELED])
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stderr = BufReader::new(child.stderr.take().unwrap());
    let mut line = String::new();
    stderr.read_line(&mut line).unwrap();
    let url = line
        .trim()
        .strip_prefix("dashboard: ")
        .expect(&line)
        .to_string();

    let deadline = Instant::now() + Duration::from_secs(10);
    let metrics = loop {
        let body = fetch(&url, "/metrics").expect("the dashboard stays up after the training");
        let metrics: serde_json::Value = serde_json::from_str(&body).unwrap();
        if metrics["status"] == "done" {
            break metrics;
        }
        assert!(Instant::now() < deadline, "the training never finished");
        std::thread::sleep(Duration::from_millis(20));
    };
    assert_eq!(metrics["epochs"], 3);
    let points = metrics["points"].as_array().unwrap();
    assert_eq!(points.len(), 3);
    assert_eq!(points[2]["epoch"], 3);
    assert!(points[2]["training_loss"].is_number());
    let accuracy = points[2]["training_accuracy"].as_f64().unwrap();
    assert!((0.0..=1.0).contains(&accuracy), "{accuracy}");
    assert!(fetch(&url, "/").unwrap().contains("MyTorch training"));
    // Without early stopping there is no validation set, so the matrix covers the training samples.
    let confusion = &metrics["confusion"];
    assert_eq!(confusion["on"], "training");
    assert_eq!(confusion["labels"].as_array().unwrap().len(), 6);
    let counted: u64 = confusion["counts"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap())
        .map(|count| count.as_u64().unwrap())
        .sum();

    let mut stdout = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    assert_eq!(child.wait().unwrap().code(), Some(0));
    assert_eq!(stdout, outcome.stdout());
    let trained_on: u64 = stdout
        .strip_prefix("training on ")
        .and_then(|rest| rest.split(' ').next())
        .and_then(|count| count.parse().ok())
        .expect(&stdout);
    assert_eq!(counted, trained_on);
    assert_eq!(
        fs::read_to_string(&watched).unwrap(),
        fs::read_to_string(&plain).unwrap()
    );
}
