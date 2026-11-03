use std::ffi::OsString;
use std::process::ExitCode;

use clap::error::ErrorKind;
use clap::{ArgMatches, Command};

pub const ERROR_EXIT_CODE: u8 = 84;

pub trait Cli: Sized {
    const HELP: &'static str;

    fn command() -> Command;

    fn from_matches(matches: &ArgMatches) -> Result<Self, String>;

    fn try_parse_from<I, T>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        match Self::command()
            .override_help(Self::HELP)
            .try_get_matches_from(args)
        {
            Ok(matches) => Self::from_matches(&matches).map(Some),
            Err(error) if error.kind() == ErrorKind::DisplayHelp => Ok(None),
            Err(error) => Err(error.render().to_string().trim_end().to_string()),
        }
    }
}

pub fn run<A: Cli>(app: impl FnOnce(A) -> Result<(), String>) -> ExitCode {
    let result = A::try_parse_from(std::env::args_os()).and_then(|args| match args {
        Some(args) => app(args),
        None => {
            print!("{}", A::HELP);
            Ok(())
        }
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("{message}");
            ExitCode::from(ERROR_EXIT_CODE)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Arg;

    struct Name(String);

    impl Cli for Name {
        const HELP: &'static str = "USAGE\n        ./name NAME\n";

        fn command() -> Command {
            Command::new("name").arg(Arg::new("NAME").required(true))
        }

        fn from_matches(matches: &ArgMatches) -> Result<Self, String> {
            let name = matches.get_one::<String>("NAME").unwrap();
            if name.is_empty() {
                return Err("error: empty name".into());
            }
            Ok(Name(name.clone()))
        }
    }

    #[test]
    fn parses_valid_arguments() {
        let name = Name::try_parse_from(["name", "esteban"]).unwrap().unwrap();
        assert_eq!(name.0, "esteban");
    }

    #[test]
    fn help_is_not_an_error() {
        assert!(matches!(Name::try_parse_from(["name", "--help"]), Ok(None)));
        assert!(matches!(Name::try_parse_from(["name", "-h"]), Ok(None)));
    }

    #[test]
    fn clap_and_validation_errors_are_strings() {
        let missing = Name::try_parse_from(["name"]).err().unwrap();
        assert!(missing.starts_with("error:"), "{missing}");
        let invalid = Name::try_parse_from(["name", ""]).err().unwrap();
        assert_eq!(invalid, "error: empty name");
    }
}
