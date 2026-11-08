use std::fs;
use std::path::{Path, PathBuf};

use cli_test::Program;
use my_torch::ModelConfig;
use serde_json::Value;

const HELP: &str = include_str!("../src/help.txt");

struct Folder(PathBuf);

impl Folder {
    fn new(name: &str) -> Folder {
        let path =
            std::env::temp_dir().join(format!("my_torch_tuner_{name}_{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Folder(path)
    }

    fn path(&self, name: &str) -> String {
        self.0.join(name).to_string_lossy().into_owned()
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn tuner() -> Program {
    Program::new(env!("CARGO_BIN_EXE_my_torch_tuner"))
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn help_prints_the_usage() {
    tuner().assert_help(HELP);
}

#[test]
fn invalid_arguments_exit_with_84() {
    tuner().assert_errors(&[
        (&[], "the following required arguments were not provided"),
        (
            &["a.conf", "space.json"],
            "the following required arguments were not provided",
        ),
        (
            &["--trials", "0", "a.conf", "space.json", "boards.txt"],
            "invalid value for --trials: '0'",
        ),
        (
            &["--rounds", "two", "a.conf", "space.json", "boards.txt"],
            "invalid value for --rounds: 'two'",
        ),
        (
            &["--validation", "1", "a.conf", "space.json", "boards.txt"],
            "invalid value for --validation: '1'",
        ),
        (
            &["--seed", "-1", "a.conf", "space.json", "boards.txt"],
            "unexpected argument",
        ),
        (
            &["--unknown", "a.conf", "space.json", "boards.txt"],
            "unexpected argument '--unknown' found",
        ),
    ]);
}

#[test]
fn reports_files_that_cannot_be_used() {
    let folder = Folder::new("errors");
    let (base, space, boards) = (
        fixture("base.conf"),
        fixture("space.json"),
        fixture("boards.txt"),
    );
    let missing = folder.path("missing");
    let broken = folder.path("broken.json");
    fs::write(&broken, r#"{"momentum": {"choices": [1]}}"#).unwrap();
    tuner().assert_errors(&[
        (
            &[&missing, &space, &boards],
            &format!("cannot read {missing}: "),
        ),
        (
            &[&base, &missing, &boards],
            &format!("cannot read {missing}: "),
        ),
        (
            &[&base, &broken, &boards],
            &format!("{broken}: unknown hyperparameter \"momentum\""),
        ),
        (
            &[&base, &space, &missing],
            &format!("cannot read '{missing}': "),
        ),
        (
            &["--samples", "1", &base, &space, &boards],
            "1 chessboards are not enough",
        ),
    ]);
}

#[test]
fn searches_and_writes_the_log_and_the_best_configuration() {
    let folder = Folder::new("search");
    let (log, output) = (folder.path("log.jsonl"), folder.path("best.conf"));
    let run = tuner().run(&[
        "--trials",
        "4",
        "--rounds",
        "2",
        "--epochs",
        "1",
        "--log",
        &log,
        "--output",
        &output,
        &fixture("base.conf"),
        &fixture("space.json"),
        &fixture("boards.txt"),
    ]);
    assert_eq!(run.code(), Some(0), "{}", run.stdout());
    assert!(run.stdout().contains("6 trials run"), "{}", run.stdout());

    let lines: Vec<Value> = fs::read_to_string(&log)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    let shape: Vec<_> = lines
        .iter()
        .map(|line| {
            (
                line["round"].as_u64().unwrap(),
                line["epochs"].as_u64().unwrap(),
            )
        })
        .collect();
    assert_eq!(shape, [(1, 1), (1, 1), (1, 1), (1, 1), (2, 2), (2, 2)]);
    for line in &lines {
        assert!(line["validation_accuracy"].is_f64(), "{line}");
        assert!(line["params"]["learning_rate"].is_f64(), "{line}");
    }

    // The best configuration is a usable file: the generator accepts it and it keeps the base training length.
    ModelConfig::load(&output).unwrap();
    let best: Value = serde_json::from_str(&fs::read_to_string(&output).unwrap()).unwrap();
    assert_eq!(best["training"]["epochs"], 10);
    assert_eq!(best["inputs"], 769);
    let last_round_best = lines[4..]
        .iter()
        .max_by(|a, b| {
            let key = |line: &Value| {
                (
                    line["validation_accuracy"].as_f64().unwrap(),
                    -line["validation_loss"].as_f64().unwrap(),
                )
            };
            key(a).partial_cmp(&key(b)).unwrap()
        })
        .unwrap();
    assert_eq!(
        best["training"]["optimizer"]["learning_rate"],
        last_round_best["params"]["learning_rate"]
    );
}

#[test]
fn the_same_seed_gives_the_same_best_configuration() {
    let folder = Folder::new("seed");
    let results: Vec<String> = ["a.conf", "b.conf"]
        .iter()
        .map(|name| {
            let output = folder.path(name);
            tuner()
                .run(&[
                    "--trials",
                    "3",
                    "--rounds",
                    "1",
                    "--epochs",
                    "1",
                    "--seed",
                    "5",
                    "--log",
                    &folder.path("log.jsonl"),
                    "--output",
                    &output,
                    &fixture("base.conf"),
                    &fixture("space.json"),
                    &fixture("boards.txt"),
                ])
                .code()
                .eq(&Some(0))
                .then_some(())
                .expect("the search succeeds");
            fs::read_to_string(Path::new(&output)).unwrap()
        })
        .collect();
    assert_eq!(results[0], results[1]);
}
