use cli_test::Program;

const HELP: &str = include_str!("../src/help.txt");

fn analyzer() -> Program {
    Program::new(env!("CARGO_BIN_EXE_my_torch_analyzer"))
}

#[test]
fn help_prints_the_subject_usage() {
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
    ]);
}

#[test]
fn valid_arguments_reach_the_selected_mode() {
    analyzer().assert_errors(&[
        (
            &["--predict", "net.nn", "boards.txt"],
            "--predict is not implemented yet",
        ),
        (
            &["--train", "net.nn", "boards.txt"],
            "--train is not implemented yet",
        ),
        (
            &["--save", "out.nn", "--train", "net.nn", "boards.txt"],
            "--train is not implemented yet",
        ),
    ]);
}
