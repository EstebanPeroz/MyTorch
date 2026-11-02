use std::path::PathBuf;
use std::process;

use clap::error::ErrorKind;
use clap::{Arg, ArgMatches, Command};

const HELP: &str = "USAGE
        ./my_torch_generator config_file_1 nb_1 [config_file_2 nb_2...]

DESCRIPTION
        config_file_i    Configuration file containing description of a neural network we want to
generate.
        nb_i             Number of neural networks to generate based on the configuration file.
";

pub struct NetworkRequest {
    pub config_file: PathBuf,
    pub count: usize,
}

pub struct Args {
    pub requests: Vec<NetworkRequest>,
}

impl Args {
    pub fn parse() -> Self {
        let mut command = Self::command();
        let result = command
            .try_get_matches_from_mut(std::env::args_os())
            .and_then(|matches| {
                Self::from_matches(&matches)
                    .map_err(|message| command.error(ErrorKind::ValueValidation, message))
            });
        result.unwrap_or_else(|error| {
            let code = if error.use_stderr() { 84 } else { 0 };
            let _ = error.print();
            process::exit(code);
        })
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
            return Err("each configuration file must be followed by a number of networks".into());
        }
        let requests = values
            .chunks(2)
            .map(|pair| {
                let count = pair[1]
                    .parse::<usize>()
                    .ok()
                    .filter(|&count| count > 0)
                    .ok_or_else(|| format!("invalid number of networks: '{}'", pair[1]))?;
                Ok(NetworkRequest {
                    config_file: PathBuf::from(pair[0]),
                    count,
                })
            })
            .collect::<Result<_, String>>()?;
        Ok(Args { requests })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Option<Args> {
        let matches = Args::command()
            .try_get_matches_from(std::iter::once("my_torch_generator").chain(args.iter().copied()))
            .ok()?;
        Args::from_matches(&matches).ok()
    }

    #[test]
    fn parses_several_pairs() {
        let args = parse(&["basic.conf", "3", "deep.conf", "1"]).unwrap();
        let requests: Vec<_> = args
            .requests
            .iter()
            .map(|request| (request.config_file.to_str().unwrap(), request.count))
            .collect();
        assert_eq!(requests, [("basic.conf", 3), ("deep.conf", 1)]);
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
            assert!(parse(args).is_none(), "{args:?}");
        }
    }
}
