use cli_test::Program;

const HELP: &str = include_str!("../src/help.txt");

fn generator() -> Program {
    Program::new(env!("CARGO_BIN_EXE_my_torch_generator"))
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
fn valid_arguments_reach_generation() {
    generator().assert_errors(&[
        (
            &["basic.conf", "2"],
            "network generation is not implemented yet",
        ),
        (
            &["basic.conf", "2", "deep.conf", "1"],
            "network generation is not implemented yet",
        ),
    ]);
}
