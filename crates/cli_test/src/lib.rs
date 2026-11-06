use std::fmt;
use std::path::PathBuf;
use std::process::Command;

const ERROR_EXIT_CODE: i32 = 84;

pub struct Program {
    path: PathBuf,
}

pub struct Outcome {
    command: String,
    code: Option<i32>,
    stdout: String,
    stderr: String,
}

impl Program {
    pub fn new(path: &str) -> Self {
        Program {
            path: PathBuf::from(path),
        }
    }

    pub fn run(&self, args: &[&str]) -> Outcome {
        let output = Command::new(&self.path)
            .args(args)
            .output()
            .unwrap_or_else(|error| panic!("cannot run {}: {error}", self.path.display()));
        Outcome {
            command: format!("{} {}", self.path.display(), args.join(" ")),
            code: output.status.code(),
            stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
            stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        }
    }

    #[track_caller]
    pub fn assert_help(&self, help: &str) {
        for flag in ["--help", "-h"] {
            self.run(&[flag]).assert_success(help);
        }
    }

    #[track_caller]
    pub fn assert_errors(&self, cases: &[(&[&str], &str)]) {
        for (args, message) in cases {
            self.run(args).assert_error(message);
        }
    }
}

impl Outcome {
    #[track_caller]
    pub fn assert_success(&self, expected_stdout: &str) {
        assert_eq!(self.code, Some(0), "wrong exit code: {self}");
        assert_eq!(self.stdout, expected_stdout, "wrong stdout: {self}");
        assert!(self.stderr.is_empty(), "unexpected stderr: {self}");
    }

    #[track_caller]
    pub fn assert_error(&self, message: &str) {
        assert_eq!(self.code, Some(ERROR_EXIT_CODE), "wrong exit code: {self}");
        assert!(self.stdout.is_empty(), "unexpected stdout: {self}");
        assert!(
            self.stderr.starts_with(&format!("error: {message}")),
            "stderr should start with \"error: {message}\": {self}"
        );
    }
}

impl fmt::Display for Outcome {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(formatter, "\n$ {}", self.command)?;
        writeln!(formatter, "exit code: {:?}", self.code)?;
        writeln!(formatter, "stdout:\n{}", self.stdout)?;
        write!(formatter, "stderr:\n{}", self.stderr)
    }
}
