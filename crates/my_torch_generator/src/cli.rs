use std::ffi::OsString;
use std::path::{Path, PathBuf};

use clap::error::ErrorKind;
use clap::{Arg, ArgMatches, Command};

pub const HELP: &str = "USAGE
        ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
        config_file_i    Configuration file containing description of a neural network we want to
generate.
        nb_i             Number of neural networks to generate based on the configuration file.
";

pub struct NetworkRequest {
    config_file: PathBuf,
    count: usize,
}

impl NetworkRequest {
    pub fn config_file(&self) -> &Path {
        &self.config_file
    }

    pub fn count(&self) -> usize {
        self.count
    }

    fn from_pair(config_file: &str, count: &str) -> Result<Self, String> {
        let count = count
            .parse::<usize>()
            .ok()
            .filter(|&count| count > 0)
            .ok_or_else(|| format!("error: invalid number of networks: '{count}'"))?;
        Ok(NetworkRequest {
            config_file: PathBuf::from(config_file),
            count,
        })
    }
}

pub struct Args {
    requests: Vec<NetworkRequest>,
}

impl Args {
    pub fn try_parse() -> Result<Option<Self>, String> {
        Self::try_parse_from(std::env::args_os())
    }

    pub fn requests(&self) -> &[NetworkRequest] {
        &self.requests
    }

    fn try_parse_from<I, T>(args: I) -> Result<Option<Self>, String>
    where
        I: IntoIterator<Item = T>,
        T: Into<OsString> + Clone,
    {
        match Self::command().try_get_matches_from(args) {
            Ok(matches) => Self::from_matches(&matches).map(Some),
            Err(error) if error.kind() == ErrorKind::DisplayHelp => Ok(None),
            Err(error) => Err(error.render().to_string().trim_end().to_string()),
        }
    }

    fn command() -> Command {
        Command::new("my_torch_generator").override_help(HELP).arg(
            Arg::new("PAIRS")
                .value_name("config_file nb")
                .num_args(2..)
                .required(true),
        )
    }

    fn from_matches(matches: &ArgMatches) -> Result<Self, String> {
        let values: Vec<&String> = matches.get_many::<String>("PAIRS").unwrap().collect();
        if !values.len().is_multiple_of(2) {
            return Err(
                "error: each configuration file must be followed by a number of networks".into(),
            );
        }
        let requests = values
            .chunks(2)
            .map(|pair| NetworkRequest::from_pair(pair[0], pair[1]))
            .collect::<Result<_, _>>()?;
        Ok(Args { requests })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Option<Args>, String> {
        Args::try_parse_from(std::iter::once("my_torch_generator").chain(args.iter().copied()))
    }

    #[test]
    fn parses_several_pairs() {
        let args = parse(&["basic.conf", "3", "deep.conf", "1"])
            .unwrap()
            .unwrap();
        let requests: Vec<_> = args
            .requests()
            .iter()
            .map(|request| (request.config_file().to_str().unwrap(), request.count()))
            .collect();
        assert_eq!(requests, [("basic.conf", 3), ("deep.conf", 1)]);
    }

    #[test]
    fn help_is_not_an_error() {
        assert!(matches!(parse(&["--help"]), Ok(None)));
    }

    #[test]
    fn rejects_invalid_arguments() {
        let invalid: [&[&str]; 5] = [
            &[],
            &["basic.conf"],
            &["basic.conf", "3", "deep.conf"],
            &["basic.conf", "zero"],
            &["basic.conf", "0"],
        ];
        for args in invalid {
            assert!(parse(args).is_err(), "{args:?}");
        }
    }
}
