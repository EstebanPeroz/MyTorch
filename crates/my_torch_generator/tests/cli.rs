use std::fs;
use std::path::{Path, PathBuf};

use cli_test::Program;
use my_torch::Model;

const HELP: &str = include_str!("../src/help.txt");

struct Folder(PathBuf);

impl Folder {
    fn new(name: &str) -> Folder {
        let path =
            std::env::temp_dir().join(format!("my_torch_generator_{name}_{}", std::process::id()));
        fs::create_dir(&path).unwrap();
        Folder(path)
    }

    fn path(&self) -> &Path {
        &self.0
    }

    fn files(&self) -> Vec<String> {
        let mut files: Vec<String> = fs::read_dir(&self.0)
            .unwrap()
            .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        files.sort();
        files
    }
}

impl Drop for Folder {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn generator() -> Program {
    Program::new(env!("CARGO_BIN_EXE_my_torch_generator"))
}

fn generator_in(folder: &Folder) -> Program {
    generator().in_directory(folder.path())
}

fn fixture(name: &str) -> String {
    format!("{}/tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"))
}

#[test]
fn help_prints_the_usage() {
    generator().assert_help(HELP);
}

#[test]
fn invalid_arguments_exit_with_84() {
    generator().assert_errors(&[
        (&[], "the following required arguments were not provided"),
        (&["basic.conf"], "2 values required"),
        (
            &["basic.conf", "3", "deep.conf"],
            "each configuration file must be followed by a number of networks",
        ),
        (&["basic.conf", "0"], "invalid number of networks: '0'"),
        (
            &["basic.conf", "three"],
            "invalid number of networks: 'three'",
        ),
        (
            &["basic.conf", "99999999999999999999"],
            "invalid number of networks: '99999999999999999999'",
        ),
        (&["basic.conf", "-1"], "unexpected argument '-1' found"),
        (
            &["basic.conf", "2", "deep.conf", "0"],
            "invalid number of networks: '0'",
        ),
    ]);
}

#[test]
fn generates_the_requested_networks_named_after_the_configuration() {
    let folder = Folder::new("basic");
    generator_in(&folder)
        .run(&[&fixture("basic_network.conf"), "3"])
        .assert_success("");
    assert_eq!(
        folder.files(),
        [
            "basic_network_1.nn",
            "basic_network_2.nn",
            "basic_network_3.nn"
        ]
    );
    let models: Vec<Model> = folder
        .files()
        .iter()
        .map(|file| Model::load(folder.path().join(file)).unwrap())
        .collect();
    let seeds: Vec<u64> = models.iter().map(Model::seed).collect();
    assert_eq!(seeds, [42, 43, 44]);
    for model in &models {
        let network = model.network();
        assert_eq!((network.input_size(), network.output_size()), (6, 3));
        assert_eq!(model.trainer().loss().name(), "softmax_cross_entropy");
        assert!(model.trainer().early_stopping().is_some());
    }
    assert_ne!(models[0].network(), models[1].network());
}

#[test]
fn handles_several_configurations_at_once() {
    let folder = Folder::new("several");
    generator_in(&folder)
        .run(&[
            &fixture("basic_network.conf"),
            "2",
            &fixture("deep_network.conf"),
            "1",
        ])
        .assert_success("");
    assert_eq!(
        folder.files(),
        [
            "basic_network_1.nn",
            "basic_network_2.nn",
            "deep_network_1.nn"
        ]
    );
    let deep = Model::load(folder.path().join("deep_network_1.nn")).unwrap();
    assert_eq!(deep.network().layers().len(), 3);
}

#[test]
fn generating_again_gives_the_same_files() {
    let first = Folder::new("first");
    let second = Folder::new("second");
    for folder in [&first, &second] {
        generator_in(folder)
            .run(&[&fixture("basic_network.conf"), "2"])
            .assert_success("");
    }
    for file in ["basic_network_1.nn", "basic_network_2.nn"] {
        assert_eq!(
            fs::read(first.path().join(file)).unwrap(),
            fs::read(second.path().join(file)).unwrap()
        );
    }
}

#[test]
fn reports_configurations_that_cannot_be_used() {
    let folder = Folder::new("errors");
    let missing = fixture("missing.conf");
    let unknown = fixture("unknown_activation.conf");
    generator_in(&folder).assert_errors(&[
        (&[&missing, "1"], &format!("cannot read {missing}: ")),
        (
            &[&unknown, "1"],
            &format!("{unknown}: layer 1: unknown activation \"softmax\""),
        ),
    ]);
    assert!(folder.files().is_empty());
}

#[test]
fn reports_networks_that_cannot_be_saved() {
    let folder = Folder::new("blocked");
    fs::create_dir(folder.path().join("basic_network_1.nn")).unwrap();
    generator_in(&folder)
        .run(&[&fixture("basic_network.conf"), "2"])
        .assert_error("cannot write basic_network_1.nn: ");
    assert_eq!(folder.files(), ["basic_network_1.nn"]);
}

#[test]
fn writes_nothing_when_a_later_configuration_is_invalid() {
    let folder = Folder::new("partial");
    let unknown = fixture("unknown_activation.conf");
    generator_in(&folder)
        .run(&[&fixture("basic_network.conf"), "2", &unknown, "1"])
        .assert_error(&format!("{unknown}: layer 1"));
    assert!(folder.files().is_empty());
}
